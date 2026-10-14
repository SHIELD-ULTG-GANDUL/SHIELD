"""Test SHIELD Stage 2, Part B — baseline models. Tanpa DB/network (cursor
di-mock di mana perlu). Empat kategori yang diminta brief Part B:
  - no-leakage: fitur tidak boleh berasal dari suhu masa lalu sensor target.
  - split: train/validasi harus terurut waktu, tidak overlap.
  - activation switches the worker's model.
  - fallback works with no active model.

Run: python3 test_ml_part_b.py
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "ml"))
sys.path.insert(0, str(_ROOT / "server"))

import dataset as ds  # noqa: E402
import models as mo  # noqa: E402
import model_registry as mr  # noqa: E402
import train_baseline as tb  # noqa: E402
import lntai_worker as worker  # noqa: E402
from lntai_calculator import LNTAICalculator  # noqa: E402

ARTIFACTS_DIR = _ROOT / "ml" / "artifacts"


# ════════════════════════════════════════════════════════════════════════
# No-leakage
# ════════════════════════════════════════════════════════════════════════

class TestNoLeakage(unittest.TestCase):
    def test_allowed_features_contain_no_temperature_derived_name(self):
        tb.assert_no_target_leakage(tuple(mo.FEATURE_NAMES))  # tidak boleh raise

    def test_feature_named_like_temperature_is_rejected(self):
        with self.assertRaises(AssertionError):
            tb.assert_no_target_leakage(("temp_past_1h",))

    def test_feature_named_target_is_rejected(self):
        with self.assertRaises(AssertionError):
            tb.assert_no_target_leakage(("target_lag",))

    def test_unknown_feature_name_is_rejected_even_if_not_obviously_temperature(self):
        """Fitur baru APA PUN yang tidak ada di ALLOWED_FEATURE_NAMES harus
        ditinjau eksplisit dulu -- daftar putih, bukan daftar hitam saja."""
        with self.assertRaises(AssertionError):
            tb.assert_no_target_leakage(("some_new_feature_nobody_reviewed",))

    def test_dataset_row_schema_never_includes_raw_T_as_a_feature_name(self):
        """Target ('T') sendiri TIDAK boleh ada di ALLOWED_FEATURE_NAMES --
        kalau ini pernah gagal, berarti seseorang menambahkan target sebagai
        fiturnya sendiri (kebocoran paling jelas)."""
        self.assertNotIn("T", tb.ALLOWED_FEATURE_NAMES)


# ════════════════════════════════════════════════════════════════════════
# Split waktu-berurutan
# ════════════════════════════════════════════════════════════════════════

class TestTimeOrderedSplit(unittest.TestCase):
    def _rows(self, n: int) -> list[dict]:
        return [{"ts_epoch": float(i * 300), "T": 50.0 + i * 0.01} for i in range(n)]

    def test_split_is_time_ordered_and_non_overlapping(self):
        rows = self._rows(100)
        train, val = tb.time_ordered_split(rows, train_fraction=0.8)
        tb.assert_valid_time_ordered_split(train, val)  # tidak boleh raise
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)

    def test_train_is_strictly_earlier_than_validation(self):
        rows = self._rows(50)
        train, val = tb.time_ordered_split(rows, train_fraction=0.6)
        self.assertLessEqual(train[-1]["ts_epoch"], val[0]["ts_epoch"])

    def test_shuffled_rows_are_detected_as_invalid_split(self):
        """Simulasi kesalahan (shuffle sebelum split) -- assert HARUS gagal
        kalau urutan waktu rusak, supaya regresi semacam ini ketahuan."""
        rows = self._rows(20)
        shuffled = rows[10:] + rows[:10]  # bukan urutan waktu naik murni di titik potong
        train, val = shuffled[:10], shuffled[10:]
        with self.assertRaises(AssertionError):
            tb.assert_valid_time_ordered_split(train, val)

    def test_overlapping_train_and_validation_detected(self):
        rows = self._rows(20)
        train = rows[:15]
        val = rows[10:]  # overlap sengaja (index 10-14 ada di keduanya)
        with self.assertRaises(AssertionError):
            tb.assert_valid_time_ordered_split(train, val)


# ════════════════════════════════════════════════════════════════════════
# Activation switches the worker's model / Fallback works with no active model
# ════════════════════════════════════════════════════════════════════════

class TestActivationAndFallback(unittest.TestCase):
    def setUp(self):
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.artifact_a = ARTIFACTS_DIR / "test_part_b_2_R_physics_rc_A.json"
        self.artifact_b = ARTIFACTS_DIR / "test_part_b_2_R_physics_rc_B.json"

        model_a = mo.PhysicsRCModel(i_ref=1000.0)
        model_a.tau_minutes, model_a.a, model_a.b = 20.0, 10.0, 5.0  # prediksi RENDAH
        with open(self.artifact_a, "w", encoding="utf-8") as f:
            json.dump(model_a.to_artifact(), f)

        model_b = mo.PhysicsRCModel(i_ref=1000.0)
        model_b.tau_minutes, model_b.a, model_b.b = 20.0, 80.0, 40.0  # prediksi TINGGI (beda jauh)
        with open(self.artifact_b, "w", encoding="utf-8") as f:
            json.dump(model_b.to_artifact(), f)

        mr.invalidate_cache()

    def tearDown(self):
        self.artifact_a.unlink(missing_ok=True)
        self.artifact_b.unlink(missing_ok=True)
        mr.invalidate_cache()

    class _FakeCursor:
        def __init__(self, active_row):
            self.active_row = active_row

        def execute(self, sql, params=None):
            pass

        def fetchone(self):
            return self.active_row

    def test_fallback_returns_none_with_no_active_model(self):
        cur = self._FakeCursor(active_row=None)
        pred, meta = mr.predict_delta_t_step(
            cur, "2", "R", {"current_amps": 600.0, "ts_epoch": 0.0}, ambient_c_v3=30.0,
        )
        self.assertIsNone(pred)
        self.assertIsNone(meta)

    def test_activation_switches_which_model_the_worker_uses(self):
        """Dua 'model aktif' berbeda (A vs B, parameter sangat berbeda)
        menghasilkan delta_t_pred yang BERBEDA -- mensimulasikan efek
        POST /api/ml/models/{id}/activate mengganti baris is_active=true
        di ml_models (di sini disimulasikan lewat cursor mengembalikan
        baris model aktif yang berbeda)."""
        cur_a = self._FakeCursor(active_row={
            "id": 1, "trafo": "2", "phase": "R", "model_type": "physics_rc", "version": "A",
            "artifact_path": str(self.artifact_a.relative_to(_ROOT)), "sigma": 0.3, "mu": 0.0,
        })
        pred_a, meta_a = mr.predict_delta_t_step(
            cur_a, "2", "R", {"current_amps": 600.0, "ts_epoch": 0.0}, ambient_c_v3=30.0,
        )
        self.assertEqual(meta_a["version"], "A")

        # Simulasikan aktivasi berganti ke model B (invalidate_cache = apa
        # yang terjadi begitu cache TTL habis di proses worker nyata).
        mr.invalidate_cache("2", "R")
        cur_b = self._FakeCursor(active_row={
            "id": 2, "trafo": "2", "phase": "R", "model_type": "physics_rc", "version": "B",
            "artifact_path": str(self.artifact_b.relative_to(_ROOT)), "sigma": 0.3, "mu": 0.0,
        })
        pred_b, meta_b = mr.predict_delta_t_step(
            cur_b, "2", "R", {"current_amps": 600.0, "ts_epoch": 0.0}, ambient_c_v3=30.0,
        )
        self.assertEqual(meta_b["version"], "B")
        self.assertNotAlmostEqual(pred_a, pred_b, places=2)

    def test_lntai_v3_uses_override_when_model_active_and_legacy_when_not(self):
        """End-to-end lebih tinggi: lntai_v3_step() memakai override SAAT
        ada, fisika legacy SAAT tidak -- dicek lewat baseline_id di hasil."""
        from lntai_calculator import lntai_v3_step, PHASES
        import config

        kwargs = dict(
            a=config.LNTAI_V3_A, b=config.LNTAI_V3_B, sigma=config.LNTAI_V3_SIGMA, mu=config.LNTAI_V3_MU,
            i_ref=config.LNTAI_V3_I_REF, i_rated_v2=1780.0, ambient_c=config.LNTAI_V3_AMBIENT_C,
            rolling_median_minutes=config.LNTAI_V3_ROLLING_MEDIAN_MINUTES, thresholds=config.LNTAI_V3_THRESHOLDS,
            baseline_id=config.LNTAI_V3_BASELINE_ID, calibrated=config.LNTAI_V3_CALIBRATED,
            model_version=config.LNTAI_V3_MODEL_VERSION,
        )
        per_phase = {p: {"T": 60.0, "i_eff_norm": 0.3} for p in PHASES}
        v2_result = {"level": "NORMAL", "per_phase": per_phase, "timestamp": "t0", "error": None}

        _, result_fallback = lntai_v3_step({}, v2_result, 0.0, **kwargs)
        self.assertEqual(result_fallback["baseline_id"], config.LNTAI_V3_BASELINE_ID)

        overrides = {"R": (12.3, "ml_model:7", True)}
        _, result_override = lntai_v3_step({}, v2_result, 0.0, baseline_overrides=overrides, **kwargs)
        self.assertEqual(result_override["per_phase"]["R"]["baseline_id"], "ml_model:7")
        self.assertEqual(result_override["per_phase"]["R"]["delta_T_pred"], 12.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
