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


class TestValidateProvisionalArgs(unittest.TestCase):
    """Go-live: --provisional dan --provisional-reason WAJIB berpasangan."""

    def test_neither_flag_is_valid(self):
        import train_baseline as tb
        self.assertIsNone(tb.validate_provisional_args(False, None))

    def test_both_flags_is_valid(self):
        import train_baseline as tb
        self.assertIsNone(tb.validate_provisional_args(True, "Suggested by ml/suggest_healthy_period.py"))

    def test_provisional_without_reason_is_invalid(self):
        import train_baseline as tb
        error = tb.validate_provisional_args(True, None)
        self.assertIsNotNone(error)
        self.assertIn("--provisional-reason", error)

    def test_reason_without_provisional_flag_is_invalid(self):
        import train_baseline as tb
        error = tb.validate_provisional_args(False, "some reason")
        self.assertIsNotNone(error)
        self.assertIn("--provisional", error)


# ════════════════════════════════════════════════════════════════════════
# M12 (autopilot 2026-11) — time_series_folds, naive_median, CV, --tune
# ════════════════════════════════════════════════════════════════════════

def _full_rows(n: int, step_seconds: float = 300.0, noise_sd: float = 0.3, seed: int = 1) -> list[dict]:
    """Baris lengkap (skema sama dengan ml/dataset.py) -- T berpola dari
    arus (physics_quadratic-like) supaya model non-trivial punya sesuatu
    untuk benar-benar dipelajari, bukan cuma angka acak."""
    import math
    import random
    random.seed(seed)
    i_ref = 1000.0
    rows = []
    for i in range(n):
        ts = i * step_seconds
        current = 400.0 + 200.0 * math.sin(i / 37.0)
        ambient = 28.0
        delta_t = 15.0 * (current / i_ref) ** 2
        T = ambient + delta_t + random.gauss(0, noise_sd)
        hour_angle = 2 * math.pi * ((ts / 3600.0) % 24) / 24.0
        rows.append({
            "ts_epoch": ts, "T": T, "current_amps": current, "ambient_c": ambient,
            "i2_eff_default_tau": (current / i_ref) ** 2, "load_roc_1h": 0.0,
            "hour_sin": math.sin(hour_angle), "hour_cos": math.cos(hour_angle),
            "weekday": int(ts // 86400) % 7, "quality_flags": [],
        })
    return rows


class TestTimeSeriesFolds(unittest.TestCase):
    def test_expanding_window_train_grows_each_fold(self):
        rows = _full_rows(400)
        folds = tb.time_series_folds(rows, n_folds=4, gap_minutes=0.0)
        self.assertEqual(len(folds), 4)
        train_sizes = [len(train) for train, _ in folds]
        self.assertEqual(train_sizes, sorted(train_sizes))  # train makin besar tiap fold (expanding, bukan sliding)
        for i in range(len(train_sizes) - 1):
            self.assertLess(train_sizes[i], train_sizes[i + 1])

    def test_no_overlap_within_each_fold(self):
        rows = _full_rows(400)
        folds = tb.time_series_folds(rows, n_folds=4, gap_minutes=10.0)
        tb.assert_folds_no_overlap(folds)  # tidak boleh raise

    def test_gap_is_respected(self):
        rows = _full_rows(400, step_seconds=60.0)  # 1 menit/baris
        gap_minutes = 30.0
        folds = tb.time_series_folds(rows, n_folds=3, gap_minutes=gap_minutes)
        for train, val in folds:
            self.assertGreaterEqual(val[0]["ts_epoch"] - train[-1]["ts_epoch"], gap_minutes * 60.0)

    def test_too_few_rows_returns_no_folds(self):
        rows = _full_rows(3)
        self.assertEqual(tb.time_series_folds(rows, n_folds=4), [])

    def test_fold_with_all_validation_consumed_by_gap_is_dropped_not_empty(self):
        # gap lebih besar dari seluruh rentang data -- semua kandidat val
        # harus kena potong, fold itu TIDAK BOLEH muncul dengan val_rows=[].
        rows = _full_rows(50, step_seconds=60.0)
        folds = tb.time_series_folds(rows, n_folds=4, gap_minutes=10000.0)
        for _, val in folds:
            self.assertGreater(len(val), 0)


class TestNaiveMedianModel(unittest.TestCase):
    def test_predicts_constant_delta_t_regardless_of_load(self):
        rows = [
            {"T": 40.0, "ambient_c": 25.0, "current_amps": 100.0},
            {"T": 50.0, "ambient_c": 25.0, "current_amps": 900.0},
            {"T": 45.0, "ambient_c": 25.0, "current_amps": 500.0},
        ]
        m = tb.NaiveMedianModel()
        info = m.fit(rows)
        self.assertEqual(info["median_delta_t"], 20.0)  # median([15,25,20]) = 20

        preds = m.predict([{"ambient_c": 30.0, "current_amps": 1.0}, {"ambient_c": 10.0, "current_amps": 999.0}])
        self.assertAlmostEqual(preds[0], 50.0)  # 30 + 20, SAMA terlepas dari current_amps=1
        self.assertAlmostEqual(preds[1], 30.0)  # 10 + 20, SAMA terlepas dari current_amps=999

    def test_artifact_roundtrip(self):
        rows = [{"T": 40.0, "ambient_c": 25.0}, {"T": 44.0, "ambient_c": 25.0}]
        m = tb.NaiveMedianModel()
        m.fit(rows)
        restored = tb.NaiveMedianModel.from_artifact(m.to_artifact())
        self.assertEqual(restored.median_delta_t, m.median_delta_t)

    def test_predict_before_fit_raises(self):
        m = tb.NaiveMedianModel()
        with self.assertRaises(RuntimeError):
            m.predict([{"ambient_c": 25.0}])


class TestCvMetrics(unittest.TestCase):
    def test_naive_median_cv_returns_mean_std_median(self):
        rows = _full_rows(600)
        cv = tb.cv_metrics("naive_median", i_ref=1000.0, rows=rows, n_folds=4, gap_minutes=0.0)
        self.assertIsNotNone(cv)
        self.assertGreater(cv["n_folds_ok"], 0)
        self.assertGreaterEqual(cv["sigma_mean"], 0.0)
        self.assertGreaterEqual(cv["mae_mean"], 0.0)
        # median harus berada di antara/dekat sigma masing-masing fold -- cek konsistensi longgar
        self.assertGreaterEqual(cv["sigma_median"], 0.0)

    def test_physics_quadratic_beats_naive_median_on_synthetic_data(self):
        # Data sintetis dibuat dari rumus quadratic MURNI (lihat _full_rows)
        # -- physics_quadratic HARUS jelas lebih baik dari median konstan.
        rows = _full_rows(600)
        cv_naive = tb.cv_metrics("naive_median", i_ref=1000.0, rows=rows, n_folds=4, gap_minutes=0.0)
        cv_physics = tb.cv_metrics("physics_quadratic", i_ref=1000.0, rows=rows, n_folds=4, gap_minutes=0.0)
        self.assertIsNotNone(cv_naive)
        self.assertIsNotNone(cv_physics)
        self.assertLess(cv_physics["sigma_mean"], cv_naive["sigma_mean"])

    def test_insufficient_data_returns_none_not_empty_dict(self):
        rows = _full_rows(3)
        cv = tb.cv_metrics("naive_median", i_ref=1000.0, rows=rows, n_folds=4, gap_minutes=0.0)
        self.assertIsNone(cv)


class TestTrainAllModelsIncludesCvAndNaiveMedian(unittest.TestCase):
    def test_naive_median_present_and_marked_skip_save(self):
        rows = _full_rows(500)
        train, val = tb.time_ordered_split(rows, train_fraction=0.8)
        results = tb.train_all_models(train, val, trafo="2", phase="R", cv_n_folds=3, cv_gap_minutes=0.0)
        by_type = {r["model_type"]: r for r in results}
        self.assertIn("naive_median", by_type)
        self.assertTrue(by_type["naive_median"]["skip_save"])
        self.assertFalse(by_type["physics_quadratic"]["skip_save"])

    def test_ok_results_carry_cv_metrics_key(self):
        rows = _full_rows(500)
        train, val = tb.time_ordered_split(rows, train_fraction=0.8)
        results = tb.train_all_models(train, val, trafo="2", phase="R", cv_n_folds=3, cv_gap_minutes=0.0)
        for r in results:
            if r["status"] == "ok":
                self.assertIn("cv", r["metrics"])


class TestGbmTuning(unittest.TestCase):
    def test_tune_selects_params_from_grid(self):
        rows = _full_rows(500)
        result = tb.tune_gbm_hyperparameters(rows, i_ref=1000.0, n_folds=3, gap_minutes=0.0)
        self.assertIsNotNone(result["best_params"])
        self.assertIn(result["best_params"]["max_depth"], tb.GBM_TUNE_GRID["max_depth"])
        self.assertIn(result["best_params"]["learning_rate"], tb.GBM_TUNE_GRID["learning_rate"])
        self.assertIn(result["best_params"]["min_samples_leaf"], tb.GBM_TUNE_GRID["min_samples_leaf"])
        n_expected = (
            len(tb.GBM_TUNE_GRID["max_depth"])
            * len(tb.GBM_TUNE_GRID["learning_rate"])
            * len(tb.GBM_TUNE_GRID["min_samples_leaf"])
        )
        self.assertEqual(len(result["all_trials"]), n_expected)

    def test_tune_does_not_mutate_global_ml_config(self):
        before = json.dumps(ds.ML_CONFIG["gbm"], sort_keys=True)
        tb.tune_gbm_hyperparameters(_full_rows(400), i_ref=1000.0, n_folds=3, gap_minutes=0.0)
        after = json.dumps(ds.ML_CONFIG["gbm"], sort_keys=True)
        self.assertEqual(before, after)

    def test_gbm_fit_accepts_hyperparameter_overrides(self):
        rows = _full_rows(300)
        m = mo.MLGBMModel()
        info = m.fit(rows, max_depth=6, learning_rate=0.1, min_samples_leaf=40)
        self.assertEqual(info["max_depth"], 6)
        self.assertEqual(info["learning_rate"], 0.1)
        self.assertEqual(info["min_samples_leaf"], 40)

    def test_gbm_fit_without_overrides_uses_config_defaults(self):
        rows = _full_rows(300)
        m = mo.MLGBMModel()
        info = m.fit(rows)
        self.assertEqual(info["max_depth"], ds.ML_CONFIG["gbm"]["max_depth"])
        self.assertEqual(info["learning_rate"], ds.ML_CONFIG["gbm"]["learning_rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
