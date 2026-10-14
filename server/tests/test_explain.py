"""Test server/explain.py — CUSUM (Stage 2, Part G). Tanpa DB/network.

Jalankan dari folder server/:
    pytest tests/test_explain.py -v
"""

from __future__ import annotations

import os
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import explain as xp  # noqa: E402


class TestCusumStep:
    def test_residual_below_slack_does_not_accumulate(self):
        state = {}
        new_state, triggered, start = xp.cusum_step(state, residual=0.5, ts=0)
        assert new_state["s"] == 0.0
        assert triggered is False
        assert start is None

    def test_residual_above_slack_accumulates(self):
        state = {}
        new_state, triggered, start = xp.cusum_step(state, residual=3.0, ts=0)
        assert new_state["s"] > 0.0


class TestComputeCusumOverHistory:
    def test_triggers_on_synthetic_consistent_ramp(self):
        random.seed(1)
        history = [(i, 0.1 + 0.05 * i + random.gauss(0, 0.1)) for i in range(200)]
        triggered, start = xp.compute_cusum_over_history(history)
        assert triggered is True
        assert start is not None

    def test_does_not_trigger_on_clean_noise(self):
        random.seed(2)
        history = [(i, random.gauss(0, 0.5)) for i in range(200)]
        triggered, start = xp.compute_cusum_over_history(history)
        assert triggered is False
        assert start is None

    def test_resets_after_sufficient_drop(self):
        history = [(i, 5.0) for i in range(20)] + [(i + 20, -10.0) for i in range(20)]
        triggered, start = xp.compute_cusum_over_history(history)
        assert triggered is False

    def test_empty_history_does_not_trigger(self):
        triggered, start = xp.compute_cusum_over_history([])
        assert triggered is False
        assert start is None

    def test_start_date_is_reported_when_triggered(self):
        history = [(i, 5.0) for i in range(10)]
        triggered, start = xp.compute_cusum_over_history(history)
        assert triggered is True
        assert start == 0  # first sample where S departed from 0


class TestBuildExplanation:
    def test_unexplained_residual_is_measured_minus_predicted(self):
        contribs = {"Phase load (filtered)": 20.0, "Evening peak hour": 1.0,
                    "Load ramp in last hour": -0.5, "Baseline offset": 15.0}
        result = xp.build_explanation(
            measured_hotspot_c=80.0, predicted_healthy_c=75.5, contributions_c=contribs,
            contribution_method="exact_physics", cusum_triggered=False, cusum_start_date=None,
        )
        assert abs(result["unexplained_residual_c"] - 4.5) < 1e-9
        assert result["cusum_message"] is None

    def test_cusum_message_formatted_when_triggered(self):
        result = xp.build_explanation(
            80.0, 75.5, {}, "exact_physics", cusum_triggered=True, cusum_start_date=date(2026, 9, 1),
        )
        assert result["cusum_message"] == "Consistent rise since 2026-09-01"

    def test_no_message_when_triggered_but_no_start_date(self):
        result = xp.build_explanation(80.0, 75.5, {}, "exact_physics", cusum_triggered=True, cusum_start_date=None)
        assert result["cusum_message"] is None
