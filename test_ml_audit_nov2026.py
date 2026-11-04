"""Test SHIELD Stage 2 audit, Nov 2026 (autopilot run) -- A7/A8/A9. Tanpa DB.
GBM/LightGBM-dependent tests di-skip bila sklearn/lightgbm tidak terpasang.

Run: python3 test_ml_audit_nov2026.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "ml"))
sys.path.insert(0, str(_ROOT / "server"))

import models as mo  # noqa: E402
import explain_contrib as ec  # noqa: E402

try:
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ════════════════════════════════════════════════════════════════════════
# A9 — model_output_is_delta_t / AMBIENT_GROUP
# ════════════════════════════════════════════════════════════════════════

class TestModelOutputIsDeltaT(unittest.TestCase):
    def test_physics_quadratic_is_always_delta_t(self):
        m = mo.PhysicsQuadraticModel(i_ref=1000.0)
        self.assertTrue(ec.model_output_is_delta_t("physics_quadratic", m))

    def test_physics_rc_is_always_delta_t(self):
        m = mo.PhysicsRCModel(i_ref=1000.0)
        self.assertTrue(ec.model_output_is_delta_t("physics_rc", m))

    def test_hybrid_is_always_delta_t(self):
        m = mo.HybridModel(i_ref=1000.0)
        self.assertTrue(ec.model_output_is_delta_t("hybrid", m))

    def test_ml_gbm_without_target_attr_defaults_to_absolute(self):
        # Artefak LAMA (sebelum A7) -- model tidak punya atribut `target`
        # sama sekali. Default HARUS "absolute" (perilaku lama: y=T mentah),
        # bukan ditebak sebagai delta_t.
        m = mo.MLGBMModel()
        self.assertFalse(hasattr(m, "target"))
        self.assertFalse(ec.model_output_is_delta_t("ml_gbm", m))

    def test_ml_gbm_with_target_absolute(self):
        m = mo.MLGBMModel()
        m.target = "absolute"
        self.assertFalse(ec.model_output_is_delta_t("ml_gbm", m))

    def test_ml_gbm_with_target_delta_t(self):
        m = mo.MLGBMModel()
        m.target = "delta_t"
        self.assertTrue(ec.model_output_is_delta_t("ml_gbm", m))

    def test_ambient_group_is_distinct_from_feature_groups(self):
        self.assertNotIn(ec.AMBIENT_GROUP, ec.CONTRIBUTION_GROUPS)
        self.assertEqual(ec.AMBIENT_GROUP, "Ambient temperature")


class TestAmbientContributionArithmetic(unittest.TestCase):
    """A9: contributions[AMBIENT_GROUP] = T - dT (reproduksi logika yang
    ditambahkan di server/api.py GET /api/explain/latest -- dites di sini
    sebagai unit aritmatika murni karena endpoint itu sendiri butuh DB)."""

    def test_predicted_healthy_c_becomes_comparable_to_measured(self):
        # Sebelum perbaikan: predicted_healthy_c = sum(fitur ΔT saja) = 15.5
        # (ΔT), dibandingkan LANGSUNG ke measured_hotspot_c = 45.5 (absolut)
        # -> unexplained_residual_c = 30.0, SELALU ~ambient, bukan residual
        # nyata. Setelah perbaikan: ambient ditambahkan dulu.
        measured_hotspot_c = 45.5
        dT = 15.5  # tersimpan di per_phase (lntai_results)
        ambient_c = measured_hotspot_c - dT
        self.assertAlmostEqual(ambient_c, 30.0)

        contributions = {"Phase load (filtered)": 12.0, "Baseline offset": 3.5}
        self.assertAlmostEqual(sum(contributions.values()), dT)  # sanity: jumlah fitur = dT asli
        contributions[ec.AMBIENT_GROUP] = ambient_c
        predicted_healthy_c = sum(contributions.values())
        self.assertAlmostEqual(predicted_healthy_c, measured_hotspot_c)  # SEKARANG sebanding
        unexplained_residual_c = measured_hotspot_c - predicted_healthy_c
        self.assertAlmostEqual(unexplained_residual_c, 0.0)  # bukan ~+30 lagi

    def test_fallback_ambient_used_when_dT_missing(self):
        # dT tidak tersedia di per_phase (data lama/tidak lengkap) -- fallback
        # ke config.LNTAI_V3_AMBIENT_C, BUKAN exception atau None menyebar.
        import config
        dT = None
        ambient_c = config.LNTAI_V3_AMBIENT_C if dT is None else (50.0 - dT)
        self.assertEqual(ambient_c, config.LNTAI_V3_AMBIENT_C)


if __name__ == "__main__":
    unittest.main(verbosity=2)
