"""Test SHIELD Stage 2, Part J2 — server/lntai_vs_aging.py (klasifikasi
kuadran, murni, tanpa DB).

Run: python3 test_lntai_vs_aging.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "server"))

import lntai_vs_aging as lva  # noqa: E402


class TestClassifyQuadrant(unittest.TestCase):
    def test_low_aging_low_lntai_is_healthy(self):
        self.assertEqual(lva.classify_quadrant(0.5, 0.5), lva.QUADRANT_HEALTHY)

    def test_high_aging_low_lntai_is_high_stress_normal(self):
        self.assertEqual(lva.classify_quadrant(2.0, 0.5), lva.QUADRANT_HIGH_STRESS_NORMAL)

    def test_low_aging_high_lntai_is_anomalous_low_stress(self):
        self.assertEqual(lva.classify_quadrant(0.5, 2.0), lva.QUADRANT_ANOMALOUS_LOW_STRESS)

    def test_high_aging_high_lntai_is_anomalous_high_stress(self):
        self.assertEqual(lva.classify_quadrant(2.0, 2.0), lva.QUADRANT_ANOMALOUS_HIGH_STRESS)

    def test_missing_aging_factor_is_unknown(self):
        self.assertEqual(lva.classify_quadrant(None, 2.0), lva.QUADRANT_UNKNOWN)

    def test_missing_lntai_is_unknown(self):
        self.assertEqual(lva.classify_quadrant(2.0, None), lva.QUADRANT_UNKNOWN)

    def test_both_missing_is_unknown(self):
        self.assertEqual(lva.classify_quadrant(None, None), lva.QUADRANT_UNKNOWN)

    def test_aging_factor_exactly_at_split_is_high_side(self):
        """Batas DIMASUKKAN ke sisi 'tinggi' (>=) -- lihat docstring
        classify_quadrant(). Nilai persis di split -> high_aging=True."""
        split = lva.AGING_FACTOR_SPLIT
        self.assertEqual(lva.classify_quadrant(split, 0.0), lva.QUADRANT_HIGH_STRESS_NORMAL)

    def test_aging_factor_just_below_split_is_low_side(self):
        split = lva.AGING_FACTOR_SPLIT
        self.assertEqual(lva.classify_quadrant(split - 0.001, 0.0), lva.QUADRANT_HEALTHY)

    def test_lntai_exactly_at_split_is_high_side(self):
        split = lva.LNTAI_SPLIT
        self.assertEqual(lva.classify_quadrant(0.0, split), lva.QUADRANT_ANOMALOUS_LOW_STRESS)

    def test_lntai_just_below_split_is_low_side(self):
        split = lva.LNTAI_SPLIT
        self.assertEqual(lva.classify_quadrant(0.0, split - 0.001), lva.QUADRANT_HEALTHY)

    def test_both_exactly_at_split_is_high_stress_anomalous(self):
        self.assertEqual(
            lva.classify_quadrant(lva.AGING_FACTOR_SPLIT, lva.LNTAI_SPLIT),
            lva.QUADRANT_ANOMALOUS_HIGH_STRESS,
        )

    def test_custom_splits_override_defaults(self):
        self.assertEqual(
            lva.classify_quadrant(5.0, 5.0, aging_split=10.0, lntai_split=10.0),
            lva.QUADRANT_HEALTHY,
        )


class TestBuildScatterPoint(unittest.TestCase):
    def test_returns_expected_shape(self):
        point = lva.build_scatter_point("2", "R", lntai=2.5, aging_factor=0.8)
        self.assertEqual(point["trafo"], "2")
        self.assertEqual(point["phase"], "R")
        self.assertEqual(point["lntai"], 2.5)
        self.assertEqual(point["aging_factor"], 0.8)
        self.assertEqual(point["quadrant"], lva.QUADRANT_ANOMALOUS_LOW_STRESS)

    def test_missing_data_point_is_unknown_not_dropped(self):
        point = lva.build_scatter_point("3", "S", lntai=None, aging_factor=None)
        self.assertEqual(point["quadrant"], lva.QUADRANT_UNKNOWN)
        self.assertIsNone(point["lntai"])
        self.assertIsNone(point["aging_factor"])


class TestLntaiSplitReusesV3Threshold(unittest.TestCase):
    def test_lntai_split_equals_v3_caution_threshold(self):
        """LNTAI_SPLIT TIDAK boleh jadi angka independen -- harus sama
        persis dengan ambang 'caution' LNTAI v3 (lntai_v3_config.json),
        supaya 'tinggi' di kuadran berarti PERSIS 'Caution' di panel v3."""
        import config
        self.assertEqual(lva.LNTAI_SPLIT, config.LNTAI_V3_THRESHOLDS["caution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
