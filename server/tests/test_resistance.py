"""Test server/resistance.py — formula & aggregasi murni (Stage 2, Part D,
beta). Tanpa DB/network.

Jalankan dari folder server/:
    pytest tests/test_resistance.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import resistance as rz  # noqa: E402

I_RATED = 1780.0


class TestIsLowLoad:
    def test_none_current_is_low_load(self):
        assert rz.is_low_load(None, I_RATED) is True

    def test_below_threshold_is_low_load(self):
        assert rz.is_low_load(500.0, I_RATED) is True  # 500 < 0.6*1780=1068

    def test_at_or_above_threshold_is_not_low_load(self):
        assert rz.is_low_load(1200.0, I_RATED) is False


class TestIsUnsteady:
    def test_none_treated_as_unsteady(self):
        assert rz.is_unsteady(None) is True

    def test_small_rate_is_steady(self):
        assert rz.is_unsteady(10.0) is False

    def test_large_rate_is_unsteady(self):
        assert rz.is_unsteady(100.0) is True  # > default max_abs_di_dt_a_per_min=50


class TestEstimateRRatio:
    def test_recovers_known_resistance_increase(self):
        """Sintetis: R=1.2*R_ref -> residual_c = 0.2*A_ref*i_eff_norm^2
        harus menghasilkan ratio persis 1.2."""
        a_ref, i_eff_norm, known_ratio = 25.0, 0.6, 1.2
        residual_c = (known_ratio - 1.0) * a_ref * i_eff_norm ** 2
        assert abs(rz.estimate_r_ratio(residual_c, a_ref, i_eff_norm) - known_ratio) < 1e-9

    def test_zero_residual_gives_ratio_one(self):
        assert rz.estimate_r_ratio(0.0, 25.0, 0.5) == 1.0

    def test_zero_a_ref_denominator_returns_none(self):
        assert rz.estimate_r_ratio(5.0, 0.0, 0.5) is None

    def test_zero_i_eff_norm_denominator_returns_none(self):
        assert rz.estimate_r_ratio(5.0, 25.0, 0.0) is None


class TestWeeklyStats:
    def test_empty_list(self):
        stats = rz.weekly_stats([])
        assert stats["median"] is None

    def test_median_and_iqr(self):
        stats = rz.weekly_stats([100.0, 105.0, 110.0, 95.0, 102.0])
        assert stats["median"] == 102.0
        assert stats["iqr"] == stats["q3"] - stats["q1"]

    def test_single_value(self):
        stats = rz.weekly_stats([100.0])
        assert stats["median"] == 100.0
        assert stats["iqr"] == 0.0


class TestPhaseRelativeRatio:
    def test_phase_above_median_of_three(self):
        result = rz.phase_relative_ratio(110.0, [100.0, 105.0, 110.0])
        assert abs(result - (110.0 / 105.0 * 100.0)) < 1e-9

    def test_none_when_phase_median_missing(self):
        assert rz.phase_relative_ratio(None, [100.0, 105.0, 110.0]) is None

    def test_none_when_fewer_than_three_phases_available(self):
        assert rz.phase_relative_ratio(110.0, [100.0, 105.0]) is None


class TestGuidanceLabel:
    def test_good(self):
        assert rz.guidance_label(100.0) == "good"

    def test_fair(self):
        assert rz.guidance_label(110.0) == "fair"

    def test_poor(self):
        assert rz.guidance_label(130.0) == "poor"

    def test_none_input(self):
        assert rz.guidance_label(None) is None

    def test_boundary_at_good_max(self):
        assert rz.guidance_label(105.0) == "fair"  # < 105 is good, == 105 is fair

    def test_boundary_at_fair_max(self):
        assert rz.guidance_label(120.0) == "fair"  # <= 120 is fair, > 120 is poor
        assert rz.guidance_label(120.01) == "poor"
