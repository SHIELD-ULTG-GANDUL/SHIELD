"""Test server/forecast_job.py — pengambilan data pendukung (residual
median, laju penuaan) dan orkestrasi forecast_series end-to-end (Stage 2,
Part F). Tanpa DB nyata (cursor di-mock).

Jalankan dari folder server/:
    pytest tests/test_forecast_job.py -v
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import forecast_job as fj  # noqa: E402
import forecast_models as fm  # noqa: E402
import models as mo  # noqa: E402

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"
NOW = datetime(2026, 10, 14, 10, 0, 0, tzinfo=timezone.utc)


class _FakeRowsCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class TestFetchRecentResidualMedian:
    def test_returns_median_of_residuals(self):
        cur = _FakeRowsCursor([{"residual": 1.0}, {"residual": 3.0}, {"residual": 2.0}])
        assert fj.fetch_recent_residual_median(cur, "2", "R", NOW) == 2.0

    def test_empty_returns_none(self):
        cur = _FakeRowsCursor([])
        assert fj.fetch_recent_residual_median(cur, "2", "R", NOW) is None


class TestFetchAgingRates:
    def test_computes_7d_increment_and_30d_daily_average(self):
        rows = [
            {"date": date(2026, 9, 15), "eq_days_cum": 10.0},
            {"date": date(2026, 10, 7), "eq_days_cum": 32.0},
            {"date": date(2026, 10, 14), "eq_days_cum": 43.5},
        ]
        cur = _FakeRowsCursor(rows)
        last_7d, avg_30d = fj.fetch_aging_rates(cur, "2", "R", NOW)
        span_days = (date(2026, 10, 14) - date(2026, 9, 15)).days
        assert abs(avg_30d - (43.5 - 10.0) / span_days) < 1e-9
        assert abs(last_7d - (43.5 - 32.0)) < 1e-9

    def test_insufficient_rows_returns_none_none(self):
        cur = _FakeRowsCursor([{"date": date(2026, 10, 14), "eq_days_cum": 1.0}])
        assert fj.fetch_aging_rates(cur, "2", "R", NOW) == (None, None)

    def test_empty_returns_none_none(self):
        cur = _FakeRowsCursor([])
        assert fj.fetch_aging_rates(cur, "2", "R", NOW) == (None, None)


class TestComputeForecastSeries:
    def setup_method(self):
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        sn = fm.SeasonalNaiveForecastModel()
        sn.p10_offset, sn.p50_offset, sn.p90_offset = -20.0, 0.0, 20.0
        self.forecast_artifact_path = "ml/artifacts/test_fj_seasonal_pytest.json"
        with open(_ARTIFACTS_DIR / "test_fj_seasonal_pytest.json", "w", encoding="utf-8") as f:
            json.dump(sn.to_artifact(), f)

        pq = mo.PhysicsQuadraticModel(i_ref=1000.0)
        pq.a, pq.b = 25.0, 15.0
        self.baseline_artifact_path = "ml/artifacts/test_fj_baseline_pytest.json"
        with open(_ARTIFACTS_DIR / "test_fj_baseline_pytest.json", "w", encoding="utf-8") as f:
            json.dump(pq.to_artifact(), f)

    def teardown_method(self):
        (_ARTIFACTS_DIR / "test_fj_seasonal_pytest.json").unlink(missing_ok=True)
        (_ARTIFACTS_DIR / "test_fj_baseline_pytest.json").unlink(missing_ok=True)

    def _make_cursor(self, forecast_artifact_path, baseline_artifact_path, has_r_relative=False):
        hourly_history = []
        for i in range(24 * 8):
            ts = NOW - timedelta(hours=24 * 8) + timedelta(hours=i)
            hourly_history.append({"hour_start": ts, "mean_current": 1000.0 + 100.0 * (ts.hour / 24.0)})

        class _Cursor:
            def __init__(self):
                self._last = None

            def execute(self, sql, params=None):
                if "FROM ml_models" in sql and "'seasonal_naive'" in sql:
                    self._last = ("row", {"id": 1, "model_type": "seasonal_naive", "artifact_path": forecast_artifact_path})
                elif "FROM ml_models" in sql and "physics_quadratic" in sql:
                    self._last = ("row", {"id": 2, "model_type": "physics_quadratic", "artifact_path": baseline_artifact_path})
                elif "FROM r_relative_weekly" in sql:
                    if has_r_relative:
                        self._last = ("row", {"r_ratio_median_pct": 115.0, "a_ref": 25.0, "i_ref": 1000.0, "status": "ok"})
                    else:
                        self._last = ("row", None)
                elif "FROM lntai_results" in sql:
                    self._last = ("rows", [{"residual": 2.0}, {"residual": 2.5}, {"residual": 1.5}])
                elif "date_trunc" in sql:
                    self._last = ("rows", hourly_history)

            def fetchone(self):
                return self._last[1] if self._last and self._last[0] == "row" else None

            def fetchall(self):
                return self._last[1] if self._last and self._last[0] == "rows" else []

        return _Cursor()

    def test_returns_none_when_no_active_forecast_model(self):
        class _NoModelCursor:
            def execute(self, sql, params=None):
                pass

            def fetchone(self):
                return None

            def fetchall(self):
                return []

        assert fj.compute_forecast_series(_NoModelCursor(), "2", "R", NOW) is None

    def test_returns_168_hourly_points_with_fallback_method(self):
        cur = self._make_cursor(self.forecast_artifact_path, self.baseline_artifact_path, has_r_relative=False)
        points = fj.compute_forecast_series(cur, "2", "R", NOW)
        assert len(points) == 168
        assert points[0]["method"] == "seasonal_naive"
        assert points[0]["excess_method"] == "recent_residual_median"
        assert points[0]["load_p10_a"] <= points[0]["load_p50_a"] <= points[0]["load_p90_a"]
        assert points[0]["lntai_pred_p10"] <= points[0]["lntai_pred_p50"] <= points[0]["lntai_pred_p90"]

    def test_uses_r_relative_method_when_available(self):
        cur = self._make_cursor(self.forecast_artifact_path, self.baseline_artifact_path, has_r_relative=True)
        points = fj.compute_forecast_series(cur, "2", "R", NOW)
        assert points[0]["excess_method"] == "r_relative"
