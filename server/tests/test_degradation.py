"""Test server/degradation.py — proyeksi Bayesian degradasi (Stage 2,
Part H, beta). Tanpa DB/network.

numpy TIDAK tersedia di sandbox pengembangan ini (sama seperti scikit-
learn/lightgbm di Part B) -- kasus yang BUTUH numpy (fit_bayesian_trend,
project_crossing, dan build_projection jalur 'ok') ditandai
@pytest.mark.skipif(not degradation._HAS_NUMPY, ...) supaya suite ini
tetap bisa dijalankan (sebagian hijau) di sini, dan akan mengeksekusi
PENUH begitu numpy terpasang (lihat requirements.txt) di server
sesungguhnya. Formula-nya sendiri sudah diverifikasi manual lewat
reimplementasi pure-Python sekali-pakai sebelum modul ini ditulis (lihat
laporan Part H) -- hasilnya: recover crossing date dalam toleransi,
interval melebar dengan lebih sedikit titik.

Jalankan dari folder server/:
    pytest tests/test_degradation.py -v
"""

from __future__ import annotations

import os
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import pytest  # noqa: E402

import degradation as dg  # noqa: E402

_needs_numpy = pytest.mark.skipif(
    not dg._HAS_NUMPY, reason="numpy tidak terpasang di sandbox ini -- lihat requirements.txt"
)


class TestIsSufficientData:
    def test_too_few_weeks_insufficient(self):
        ok, reason = dg.is_sufficient_data(n_valid_weeks=3, n_resistance_tests=2)
        assert ok is False
        assert "valid weekly" in reason

    def test_zero_resistance_tests_insufficient(self):
        ok, reason = dg.is_sufficient_data(n_valid_weeks=20, n_resistance_tests=0)
        assert ok is False
        assert "resistance_test" in reason

    def test_both_conditions_met_is_sufficient(self):
        ok, reason = dg.is_sufficient_data(n_valid_weeks=20, n_resistance_tests=1)
        assert ok is True
        assert reason is None

    def test_exactly_at_minimum_is_sufficient(self):
        min_weeks = dg.DEGRADATION_CONFIG["min_valid_weeks"]
        ok, _ = dg.is_sufficient_data(n_valid_weeks=min_weeks, n_resistance_tests=1)
        assert ok is True


class TestCrossingThreshold:
    def test_reuses_part_d_guidance_threshold(self):
        import resistance as rz

        assert dg.CROSSING_THRESHOLD_PCT == rz.RESISTANCE_CONFIG["guidance"]["fair_max_pct"]


class _FakeCursor:
    def __init__(self, weekly_rows, resistance_rows):
        self._weekly_rows = weekly_rows
        self._resistance_rows = resistance_rows
        self._last = []

    def execute(self, sql, params=None):
        if "FROM r_relative_weekly" in sql:
            self._last = self._weekly_rows
        elif "FROM maintenance_events" in sql:
            self._last = self._resistance_rows

    def fetchall(self):
        return self._last


class TestBuildProjectionInsufficientData:
    def test_no_weekly_points_insufficient(self):
        cur = _FakeCursor(weekly_rows=[], resistance_rows=[])
        result = dg.build_projection(cur, "2", "R")
        assert result["status"] == "insufficient_data"
        assert result["n_valid_weeks"] == 0

    def test_enough_weeks_but_no_resistance_test_insufficient(self):
        weekly_rows = [{"week_start": date(2026, 1, 1) + timedelta(weeks=i), "r_ratio_median_pct": 100.0} for i in range(20)]
        cur = _FakeCursor(weekly_rows=weekly_rows, resistance_rows=[])
        result = dg.build_projection(cur, "2", "R")
        assert result["status"] == "insufficient_data"
        assert "resistance_test" in result["reason"]


@_needs_numpy
class TestFitBayesianTrendAndProjection:
    def test_recovers_known_crossing_within_tolerance(self):
        random.seed(42)
        true_b0, true_b1 = 100.0, 0.5  # starts at 100%, rises 0.5%/week
        n_weeks = 40
        x = list(range(n_weeks))
        y = [true_b0 + true_b1 * xi + random.gauss(0, 1.0) for xi in x]
        weights = [1.0] * n_weeks

        fit = dg.fit_bayesian_trend(x, y, weights)
        true_crossing_x = (120.0 - true_b0) / true_b1

        import numpy as np
        rng = np.random.default_rng(1)
        projection = dg.project_crossing(fit, threshold_pct=120.0, horizon_x=260, mc_samples=5000, rng=rng)

        assert projection["status"] == "ok"
        assert abs(projection["x_p50"] - true_crossing_x) < 15
        assert projection["x_p10"] <= projection["x_p50"] <= projection["x_p90"]

    def test_interval_widens_with_fewer_points(self):
        random.seed(42)
        true_b0, true_b1 = 100.0, 0.5
        x_full = list(range(40))
        y_full = [true_b0 + true_b1 * xi + random.gauss(0, 1.0) for xi in x_full]

        import numpy as np

        fit_full = dg.fit_bayesian_trend(x_full, y_full, [1.0] * 40)
        rng1 = np.random.default_rng(2)
        proj_full = dg.project_crossing(fit_full, 120.0, 260, 5000, rng1)

        fit_few = dg.fit_bayesian_trend(x_full[:10], y_full[:10], [1.0] * 10)
        rng2 = np.random.default_rng(3)
        proj_few = dg.project_crossing(fit_few, 120.0, 260, 5000, rng2)

        interval_full = proj_full["x_p90"] - proj_full["x_p10"]
        interval_few = proj_few["x_p90"] - proj_few["x_p10"]
        assert interval_few > interval_full

    def test_no_crossing_when_slope_is_flat(self):
        random.seed(5)
        x = list(range(30))
        y = [100.0 + random.gauss(0, 0.5) for _ in x]  # no trend

        import numpy as np

        fit = dg.fit_bayesian_trend(x, y, [1.0] * 30)
        rng = np.random.default_rng(4)
        projection = dg.project_crossing(fit, 120.0, 260, 5000, rng)
        assert projection["status"] == "no_crossing_projected"


@_needs_numpy
class TestBuildProjectionWithResistanceTestCalibration:
    def test_calibrated_flag_true_when_resistance_tests_present(self):
        import numpy as np

        weekly_rows = [
            {"week_start": date(2026, 1, 1) + timedelta(weeks=i), "r_ratio_median_pct": 100.0 + 0.5 * i}
            for i in range(20)
        ]
        resistance_rows = [
            {"event_ts": datetime(2026, 1, 1, tzinfo=timezone.utc), "resistance_uohm": 100.0},
            {"event_ts": datetime(2026, 3, 1, tzinfo=timezone.utc), "resistance_uohm": 105.0},
        ]
        cur = _FakeCursor(weekly_rows=weekly_rows, resistance_rows=resistance_rows)
        result = dg.build_projection(cur, "2", "R", rng=np.random.default_rng(7))
        assert result["status"] == "ok"
        assert result["calibrated_with_resistance_test"] is True
        assert result["n_resistance_tests"] == 2
