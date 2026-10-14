"""Test SHIELD Stage 2, Part G — ml/explain_contrib.py (per-feature
contributions) + server/explain.py (CUSUM), the two pieces behind
GET /api/explain/latest. Tanpa DB/network; sklearn/lightgbm TIDAK
tersedia di sandbox ini (sama seperti Part B), jadi GBM contributions
diuji lewat objek regressor PALSU yang meniru API .predict() -- cukup
untuk memverifikasi properti ALJABAR (teleskopik) yang tidak bergantung
pada library ML yang sesungguhnya.

Run: python3 test_ml_part_g.py
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

import explain_contrib as ex  # noqa: E402
import models as mo  # noqa: E402


class _FakeRegressor:
    """Fungsi nonlinear sembarang atas 5 fitur, murni untuk menguji
    properti teleskopik waterfall -- tidak perlu backend ML nyata."""

    def predict(self, X):
        x = X[0]
        return [10.0 + 2 * x[0] + 0.5 * x[0] ** 2 - 3 * x[1] + 5 * x[2] * x[3] + 1.5 * x[4] ** 2]


class _FakeGBMModel:
    def __init__(self, backend_name="sklearn.HistGradientBoostingRegressor"):
        self.backend_name = backend_name
        self._regressor = _FakeRegressor()


FEATURE_ROW = {"i2_eff_default_tau": 3.0, "load_roc_1h": -1.2, "hour_sin": 0.7, "hour_cos": 0.3, "weekday": 4.0}


class TestPhysicsContributions(unittest.TestCase):
    def test_sums_exactly_to_prediction(self):
        contributions = ex.physics_contributions(a=25.0, b=15.0, i_eff_norm=0.6)
        total = sum(contributions.values())
        expected = 25.0 * 0.6 ** 2 + 15.0
        self.assertAlmostEqual(total, expected, places=9)

    def test_only_two_categories_nonzero(self):
        contributions = ex.physics_contributions(a=25.0, b=15.0, i_eff_norm=0.6)
        self.assertEqual(contributions["Evening peak hour"], 0.0)
        self.assertEqual(contributions["Load ramp in last hour"], 0.0)


class TestPhysicsQuadraticExplainModel(unittest.TestCase):
    def test_sums_to_prediction_and_method_is_exact(self):
        pq = mo.PhysicsQuadraticModel(i_ref=1000.0)
        pq.a, pq.b = 25.0, 15.0
        contributions, method = ex.explain_model("physics_quadratic", pq, {"current_amps": 600.0})
        total = sum(contributions.values())
        self.assertAlmostEqual(total, 25.0 * 0.6 ** 2 + 15.0, places=9)
        self.assertEqual(method, "exact_physics")


class TestPhysicsRCExplainModel(unittest.TestCase):
    def test_sums_to_prediction_using_override(self):
        rc = mo.PhysicsRCModel(i_ref=1000.0)
        rc.tau_minutes, rc.a, rc.b = 20.0, 25.0, 15.0
        contributions, method = ex.explain_model("physics_rc", rc, {}, i_eff_norm_override=0.55)
        total = sum(contributions.values())
        self.assertAlmostEqual(total, 25.0 * 0.55 ** 2 + 15.0, places=9)
        self.assertEqual(method, "exact_physics")


class TestGBMContributionsWaterfall(unittest.TestCase):
    def test_sums_exactly_to_prediction_for_arbitrary_nonlinear_function(self):
        gbm = _FakeGBMModel()
        contributions, method = ex.gbm_contributions(gbm, FEATURE_ROW)
        total = sum(contributions.values())
        actual_vec = [FEATURE_ROW[n] for n in ex.FEATURE_NAMES]
        expected = gbm._regressor.predict([actual_vec])[0]
        self.assertAlmostEqual(total, expected, places=9)
        self.assertEqual(method, "approximation_waterfall")

    def test_method_is_treeshap_when_lightgbm_backend(self):
        class _FakeLightGBMRegressor:
            def predict(self, X, pred_contrib=False):
                if pred_contrib:
                    # shape (1, n_features+1): per-feature contrib + bias, sums to raw prediction
                    return [[1.0, 2.0, 0.5, -0.3, 0.8, 10.0]]
                return [15.0]

        gbm = _FakeGBMModel(backend_name="lightgbm")
        gbm._regressor = _FakeLightGBMRegressor()
        contributions, method = ex.gbm_contributions(gbm, FEATURE_ROW)
        self.assertEqual(method, "exact_treeshap")
        total = sum(contributions.values())
        self.assertAlmostEqual(total, 1.0 + 2.0 + 0.5 - 0.3 + 0.8 + 10.0, places=9)

    def test_weekday_folds_into_baseline_offset_category(self):
        gbm = _FakeGBMModel()
        contributions, _ = ex.gbm_contributions(gbm, FEATURE_ROW)
        # "weekday" has no category of its own -- it must fold into "Baseline offset".
        self.assertIn("Baseline offset", contributions)


class TestHybridExplainModel(unittest.TestCase):
    def test_sums_to_prediction_combining_physics_and_gbm(self):
        rc = mo.PhysicsRCModel(i_ref=1000.0)
        rc.tau_minutes, rc.a, rc.b = 20.0, 25.0, 15.0
        hybrid = mo.HybridModel.__new__(mo.HybridModel)
        hybrid.physics = rc
        hybrid.residual_model = _FakeGBMModel()

        i_eff_norm = 0.4
        contributions, method = ex.explain_model("hybrid", hybrid, FEATURE_ROW, i_eff_norm_override=i_eff_norm)
        total = sum(contributions.values())

        physics_part = rc.a * i_eff_norm ** 2 + rc.b
        actual_vec = [FEATURE_ROW[n] for n in ex.FEATURE_NAMES]
        gbm_part = hybrid.residual_model._regressor.predict([actual_vec])[0]
        self.assertAlmostEqual(total, physics_part + gbm_part, places=9)
        self.assertEqual(method, "approximation_waterfall")  # method = metode bagian GBM


class TestUnknownModelType(unittest.TestCase):
    def test_raises_value_error(self):
        with self.assertRaises(ValueError):
            ex.explain_model("not_a_real_model_type", None, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
