"""Test SHIELD Stage 2, Part I — ml/evaluate_fault_injection.py (synthetic
fault injection validation). Tanpa DB/network.

Run: python3 test_ml_part_i.py
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "ml"))
sys.path.insert(0, str(_ROOT / "server"))

import evaluate_fault_injection as efi  # noqa: E402
import models as mo  # noqa: E402


def _make_clean_healthy_rows(n_days: int, a: float, b: float, i_ref: float, seed: int) -> list[dict]:
    random.seed(seed)
    n = n_days * 288
    rows = []
    for i in range(n):
        ts = i * 300.0
        current = 800.0 + 50 * ((i % 288) / 288.0)
        i_norm = current / i_ref
        dT = a * i_norm ** 2 + b
        t_hot = 30.0 + dT + random.gauss(0, 0.3)
        rows.append({
            "ts_epoch": ts, "T": t_hot, "current_amps": current, "ambient_c": 30.0,
            "i2_eff_default_tau": current ** 2, "load_roc_1h": 0.0,
            "hour_sin": 0.0, "hour_cos": 0.0, "weekday": 0,
        })
    return rows


class TestInjectStepFault(unittest.TestCase):
    def test_zero_before_fault_start(self):
        currents = [500.0] * 10
        extra = efi.inject_step_fault(currents, i_peak=1000.0, magnitude_c=10.0, fault_start_idx=5)
        self.assertTrue(all(v == 0.0 for v in extra[:5]))

    def test_full_magnitude_scaled_by_current_squared_after_start(self):
        currents = [500.0] * 10
        extra = efi.inject_step_fault(currents, i_peak=1000.0, magnitude_c=10.0, fault_start_idx=5)
        expected = 10.0 * (500.0 / 1000.0) ** 2
        for v in extra[5:]:
            self.assertAlmostEqual(v, expected, places=9)

    def test_zero_i_peak_returns_all_zero(self):
        extra = efi.inject_step_fault([0.0] * 5, i_peak=0.0, magnitude_c=10.0, fault_start_idx=0)
        self.assertTrue(all(v == 0.0 for v in extra))


class TestInjectRampFault(unittest.TestCase):
    def test_linear_ramp_to_full_magnitude(self):
        currents = [1000.0] * 20  # at i_peak, so (I/I_peak)^2 = 1
        extra = efi.inject_ramp_fault(currents, i_peak=1000.0, magnitude_c=10.0, fault_start_idx=0, ramp_duration_samples=10)
        self.assertAlmostEqual(extra[0], 0.0, places=9)
        self.assertAlmostEqual(extra[5], 5.0, places=9)  # halfway through the ramp
        self.assertAlmostEqual(extra[10], 10.0, places=9)  # ramp complete
        self.assertAlmostEqual(extra[19], 10.0, places=9)  # stays at full magnitude


class TestDetectZScoreMedian(unittest.TestCase):
    def test_no_detection_on_clean_residuals(self):
        residuals = [0.1] * 200
        idx = efi.detect_zscore_median_first(residuals, window_samples=6)
        self.assertIsNone(idx)

    def test_detects_large_sustained_shift(self):
        residuals = [0.1] * 50 + [20.0] * 50
        idx = efi.detect_zscore_median_first(residuals, window_samples=6)
        self.assertIsNotNone(idx)
        self.assertGreaterEqual(idx, 50)

    def test_count_alarms_debounces_single_sustained_episode(self):
        residuals = [0.1] * 20 + [20.0] * 50 + [0.1] * 20
        count = efi.count_zscore_median_alarms(residuals, window_samples=6)
        self.assertEqual(count, 1)

    def test_count_alarms_counts_two_separate_episodes(self):
        residuals = [0.1] * 20 + [20.0] * 20 + [0.1] * 20 + [20.0] * 20
        count = efi.count_zscore_median_alarms(residuals, window_samples=6)
        self.assertEqual(count, 2)


class TestDetectCusum(unittest.TestCase):
    def test_no_detection_on_clean_noise(self):
        random.seed(3)
        residuals = [random.gauss(0, 0.5) for _ in range(200)]
        timestamps = [float(i) for i in range(200)]
        idx = efi.detect_cusum_first(residuals, timestamps)
        self.assertIsNone(idx)

    def test_detects_consistent_shift(self):
        residuals = [0.0] * 50 + [10.0] * 50
        timestamps = [float(i) for i in range(100)]
        idx = efi.detect_cusum_first(residuals, timestamps)
        self.assertIsNotNone(idx)
        self.assertGreaterEqual(idx, 50)


class TestEvaluateDetectorGrid(unittest.TestCase):
    def setUp(self):
        self.model = mo.PhysicsQuadraticModel(i_ref=1000.0)
        self.model.a, self.model.b = 25.0, 15.0
        self.rows = _make_clean_healthy_rows(n_days=30, a=25.0, b=15.0, i_ref=1000.0, seed=7)

    def test_large_step_fault_is_detected(self):
        """Persyaratan inti task brief: a large step fault is detected."""
        results = efi.evaluate_detector_grid(self.model, self.rows)
        large_step = next(r for r in results["fault_results"] if r["fault_type"] == "step" and r["magnitude_c"] == 20)
        self.assertTrue(large_step["z_score_median"]["detected"])
        self.assertTrue(large_step["cusum"]["detected"])

    def test_false_alarm_rate_is_computed(self):
        results = efi.evaluate_detector_grid(self.model, self.rows)
        self.assertIsInstance(results["false_alarms_per_week"]["z_score_median"], float)
        self.assertIsInstance(results["false_alarms_per_week"]["cusum"], float)
        self.assertGreaterEqual(results["false_alarms_per_week"]["z_score_median"], 0.0)
        self.assertGreaterEqual(results["false_alarms_per_week"]["cusum"], 0.0)

    def test_results_are_reproducible(self):
        """Tidak ada RNG di mana pun dalam injeksi/deteksi -- hasil harus
        identik persis di dua pemanggilan berturut-turut."""
        results_a = efi.evaluate_detector_grid(self.model, self.rows)
        results_b = efi.evaluate_detector_grid(self.model, self.rows)
        self.assertEqual(results_a, results_b)

    def test_min_detectable_magnitude_reported_per_detector_and_fault_type(self):
        results = efi.evaluate_detector_grid(self.model, self.rows)
        min_mags = results["min_detectable_magnitude_c"]
        self.assertIn("z_score_median", min_mags)
        self.assertIn("cusum", min_mags)
        self.assertIn("step", min_mags["z_score_median"])
        self.assertIn("ramp", min_mags["z_score_median"])

    def test_cusum_is_at_least_as_sensitive_as_zscore_median(self):
        """CUSUM dirancang mendeteksi pergeseran kecil-konsisten lebih
        cepat daripada median bergulir 30 menit -- magnitude minimum yang
        terdeteksi CUSUM seharusnya <= z-score+median untuk fault yang sama."""
        results = efi.evaluate_detector_grid(self.model, self.rows)
        for fault_type in ("step", "ramp"):
            z_min = results["min_detectable_magnitude_c"]["z_score_median"][fault_type]
            c_min = results["min_detectable_magnitude_c"]["cusum"][fault_type]
            if z_min is not None and c_min is not None:
                self.assertLessEqual(c_min, z_min)

    def test_too_few_rows_raises(self):
        with self.assertRaises(ValueError):
            efi.evaluate_detector_grid(self.model, self.rows[:5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
