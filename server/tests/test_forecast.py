"""Test server/forecast.py — formula konsistensi, kuantil LNTAI monoton,
level-crossing, dan fallback method (Stage 2, Part F). Tanpa DB/network.

Jalankan dari folder server/:
    pytest tests/test_forecast.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import forecast as fc  # noqa: E402
import config  # noqa: E402


class TestComputeHealthyDeltaT:
    def test_quadratic_formula(self):
        dt = fc.compute_healthy_delta_t(a=25.0, b=15.0, i_ref=1000.0, i_forecast_amps=600.0)
        assert abs(dt - (25.0 * 0.6 ** 2 + 15.0)) < 1e-9

    def test_zero_i_ref_returns_b_only(self):
        assert fc.compute_healthy_delta_t(a=25.0, b=15.0, i_ref=0.0, i_forecast_amps=600.0) == 15.0


class TestComputeExcess:
    def test_r_relative_path_consistency(self):
        excess, method = fc.compute_excess(
            r_over_r_ref=1.15, a_ref=25.0, i_ref=1000.0, i_forecast_amps=600.0, recent_residual_median=99.0,
        )
        expected = (1.15 - 1.0) * 25.0 * (600.0 / 1000.0) ** 2
        assert abs(excess - expected) < 1e-9
        assert method == "r_relative"

    def test_fallback_method_is_labelled_when_r_relative_unavailable(self):
        excess, method = fc.compute_excess(
            r_over_r_ref=None, a_ref=None, i_ref=None, i_forecast_amps=600.0, recent_residual_median=3.5,
        )
        assert excess == 3.5
        assert method == "recent_residual_median"

    def test_unavailable_when_neither_source_exists(self):
        excess, method = fc.compute_excess(None, None, None, 600.0, None)
        assert excess is None
        assert method == "unavailable"

    def test_missing_i_ref_falls_back_even_if_ratio_present(self):
        """r_over_r_ref ada tapi i_ref None -- tidak cukup untuk jalur
        r_relative (perlu PASANGAN a_ref+i_ref), harus fallback."""
        excess, method = fc.compute_excess(1.15, 25.0, None, 600.0, 3.5)
        assert excess == 3.5
        assert method == "recent_residual_median"


class TestComputeLntaiPred:
    def test_consistency_with_v3_mu_sigma(self):
        excess = 5.0
        result = fc.compute_lntai_pred(excess)
        expected = (excess - config.LNTAI_V3_MU) / config.LNTAI_V3_SIGMA
        assert abs(result - expected) < 1e-9

    def test_none_excess_returns_none(self):
        assert fc.compute_lntai_pred(None) is None

    def test_zero_sigma_returns_none(self):
        assert fc.compute_lntai_pred(5.0, mu=0.0, sigma=0.0) is None


class TestComputeTHotPred:
    def test_formula(self):
        result = fc.compute_t_hot_pred(healthy_delta_t=30.0, excess=2.0, ambient_c=30.0)
        assert result == 62.0

    def test_none_excess_returns_none(self):
        assert fc.compute_t_hot_pred(30.0, None) is None


class TestComputeLntaiQuantilesMonotonic:
    def test_monotonic_with_positive_resistance_increase(self):
        p10, p50, p90, method = fc.compute_lntai_quantiles(
            500.0, 700.0, 900.0, r_over_r_ref=1.15, a_ref=25.0, a_ref_i_ref=1000.0,
            recent_residual_median=None,
        )
        assert p10 <= p50 <= p90
        assert method == "r_relative"

    def test_monotonic_even_when_excess_decreases_with_load(self):
        """R/R_ref < 1 -- excess DECREASES as load increases, which would
        reverse the raw ordering; the result MUST still be monotonic
        because it's built via sort(), not a direct quantile mapping."""
        p10, p50, p90, _ = fc.compute_lntai_quantiles(
            500.0, 700.0, 900.0, r_over_r_ref=0.85, a_ref=25.0, a_ref_i_ref=1000.0,
            recent_residual_median=None,
        )
        assert p10 <= p50 <= p90

    def test_missing_quantile_input_returns_all_none(self):
        p10, p50, p90, _ = fc.compute_lntai_quantiles(
            None, 700.0, 900.0, r_over_r_ref=1.15, a_ref=25.0, a_ref_i_ref=1000.0,
            recent_residual_median=None,
        )
        assert (p10, p50, p90) == (None, None, None)

    def test_fallback_method_labelled_in_quantiles_too(self):
        p10, p50, p90, method = fc.compute_lntai_quantiles(
            500.0, 700.0, 900.0, r_over_r_ref=None, a_ref=None, a_ref_i_ref=None,
            recent_residual_median=2.0,
        )
        assert method == "recent_residual_median"
        assert p10 == p50 == p90  # fallback tidak bergantung beban -> degenerat, tetap monoton (==)


class TestFindLevelCrossing:
    def _points(self, scores):
        return [{"target_ts": f"t{i}", "lntai_pred_p50": s} for i, s in enumerate(scores)]

    def test_exact_first_crossing_detected(self):
        points = self._points([0.5, 1.0, 1.8, 2.0, 3.5, 3.2])
        crossing = fc.find_level_crossing(points)
        assert crossing is not None
        assert crossing["point"]["target_ts"] == "t2"
        assert crossing["from_level"] == "Normal"
        assert crossing["to_level"] == "Caution"

    def test_no_crossing_when_flat(self):
        points = self._points([0.5] * 5)
        assert fc.find_level_crossing(points) is None

    def test_no_crossing_when_decreasing(self):
        points = self._points([3.5, 1.0])
        assert fc.find_level_crossing(points) is None

    def test_crossing_exactly_at_threshold_boundary(self):
        points = self._points([1.0, 1.5])  # 1.5 == caution threshold exactly
        crossing = fc.find_level_crossing(points)
        assert crossing is not None
        assert crossing["to_level"] == "Caution"

    def test_none_scores_are_skipped_not_treated_as_crossing(self):
        points = self._points([0.5, None, 1.8])
        crossing = fc.find_level_crossing(points)
        assert crossing is not None
        assert crossing["point"]["target_ts"] == "t2"


class TestSuggestedAction:
    def test_returns_fixed_config_action(self):
        action = fc.suggested_action("Caution")
        assert action is not None
        assert action == fc.FORECAST_CONFIG["suggested_actions"]["Caution"]

    def test_unknown_level_returns_none(self):
        assert fc.suggested_action("NotALevel") is None


class TestCompareAgingRate:
    def test_ratio_computation(self):
        result = fc.compare_aging_rate(last_7d_total_eq_days=10.5, last_30d_daily_avg_eq_days=1.0)
        assert abs(result["last_7d_daily_avg_eq_days"] - 1.5) < 1e-9
        assert abs(result["ratio"] - 1.5) < 1e-9

    def test_none_when_inputs_missing(self):
        assert fc.compare_aging_rate(None, 1.0) is None
        assert fc.compare_aging_rate(10.5, None) is None

    def test_none_when_denominator_zero(self):
        assert fc.compare_aging_rate(10.5, 0.0) is None


class TestBuildWarning:
    def test_assembles_all_required_fields(self):
        crossing = {
            "from_level": "Normal", "to_level": "Caution",
            "point": {
                "target_ts": "2026-10-20T14:00:00+00:00", "load_p50_a": 1500.0, "t_hot_pred_c": 78.5,
                "lntai_pred_p10": 1.2, "lntai_pred_p50": 1.8, "lntai_pred_p90": 2.5, "aging_f_pred": 1.3,
            },
        }
        warning = fc.build_warning(crossing, aging_comparison=None)
        assert warning["triggered"] is True
        assert warning["to_level"] == "Caution"
        assert warning["expected_peak_load_a"] == 1500.0
        assert warning["predicted_hotspot_c"] == 78.5
        assert warning["lntai_pred_interval"] == [1.2, 1.8, 2.5]
        assert warning["suggested_action"] is not None
