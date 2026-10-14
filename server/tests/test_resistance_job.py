"""Test server/resistance_job.py — orkestrasi job mingguan (Stage 2, Part
D, beta). Tanpa DB nyata (cursor di-mock), artefak model ditulis ke file
sementara di ml/artifacts/ lalu dihapus di teardown.

Jalankan dari folder server/:
    pytest tests/test_resistance_job.py -v
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ml"))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import resistance_job as rj  # noqa: E402
import dataset as ds  # noqa: E402
import models as mo  # noqa: E402

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts"
WEEK_START = date(2026, 9, 7)  # a Monday
WEEK_START_DT = datetime(2026, 9, 7, tzinfo=timezone.utc)


class _FakeCursor:
    def __init__(self, temp_rows=None, load_rows=None, ref_row=None, physics_row=None):
        self._temp_rows = temp_rows or []
        self._load_rows = load_rows or []
        self._ref_row = ref_row
        self._physics_row = physics_row
        self._last = None

    def execute(self, sql, params=None):
        if "SELECT id, model_type, artifact_path" in sql:
            self._last = ("row", self._ref_row)
        elif "SELECT id, artifact_path" in sql:
            self._last = ("row", self._physics_row)
        elif "FROM trafo_temperature" in sql:
            self._last = ("rows", self._temp_rows)
        elif "FROM trafo_load" in sql:
            self._last = ("rows", self._load_rows)

    def fetchone(self):
        return self._last[1] if self._last and self._last[0] == "row" else None

    def fetchall(self):
        return self._last[1] if self._last and self._last[0] == "rows" else []


def _write_physics_rc_artifact(filename: str, a: float, b: float, tau: float, i_ref: float = 1000.0) -> str:
    model = mo.PhysicsRCModel(i_ref=i_ref)
    model.tau_minutes, model.a, model.b = tau, a, b
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_artifact(), f)
    return f"ml/artifacts/{filename}"


class TestComputeWeekStatusPaths:
    def test_no_reference(self):
        cur = _FakeCursor(ref_row=None)
        result = rj.compute_week(cur, "2", "R", WEEK_START)
        assert result["status"] == "no_reference"

    def test_needs_recalibration(self):
        cur = _FakeCursor(ref_row={
            "id": 5, "model_type": "physics_rc", "artifact_path": "x",
            "recalibration_note": "Needs re-calibration after maintenance on 2026-09-01.",
        })
        result = rj.compute_week(cur, "2", "R", WEEK_START)
        assert result["status"] == "needs_recalibration"
        assert result["reference_model_id"] == 5

    def test_no_physics_rc_available_for_ml_gbm_reference(self):
        cur = _FakeCursor(
            ref_row={"id": 7, "model_type": "ml_gbm", "artifact_path": "x", "recalibration_note": None},
            physics_row=None,
        )
        result = rj.compute_week(cur, "2", "R", WEEK_START)
        assert result["status"] == "no_physics_rc_available"


class TestComputeWeekLowLoadExclusion:
    def setup_method(self):
        self.artifact_rel = _write_physics_rc_artifact("test_rj_lowload.json", 25.0, 15.0, 20.0)

    def teardown_method(self):
        (_ARTIFACTS_DIR / "test_rj_lowload.json").unlink(missing_ok=True)

    def test_all_samples_below_threshold_yields_insufficient_data(self):
        temp_rows, load_rows = [], []
        for i in range(50):
            ts = WEEK_START_DT + timedelta(minutes=5 * i)
            temp_rows.append({"timestamp": ts, "r_atas": 55.0, "r_bawah": 54.0,
                               "s_atas": 50.0, "s_bawah": 49.0, "t_atas": 50.0, "t_bawah": 49.0})
            load_rows.append({"timestamp": ts, "ia": 200.0, "ib": 500.0, "ic": 500.0})  # << 0.6*1780=1068A

        cur = _FakeCursor(
            temp_rows=temp_rows, load_rows=load_rows,
            ref_row={"id": 1, "model_type": "physics_rc", "artifact_path": self.artifact_rel, "recalibration_note": None},
        )
        result = rj.compute_week(cur, "2", "R", WEEK_START)
        assert result["status"] == "insufficient_data"
        assert result["n_samples_valid"] == 0
        assert result["n_samples_total"] == 50


class TestComputeWeekInsufficientData:
    def setup_method(self):
        self.artifact_rel = _write_physics_rc_artifact("test_rj_fewsamples.json", 25.0, 15.0, 20.0)

    def teardown_method(self):
        (_ARTIFACTS_DIR / "test_rj_fewsamples.json").unlink(missing_ok=True)

    def test_fewer_than_minimum_valid_samples(self):
        temp_rows, load_rows = [], []
        for i in range(15):  # below min_valid_samples_per_week (20)
            ts = WEEK_START_DT + timedelta(minutes=5 * i)
            temp_rows.append({"timestamp": ts, "r_atas": 55.0, "r_bawah": 54.0,
                               "s_atas": 50.0, "s_bawah": 49.0, "t_atas": 50.0, "t_bawah": 49.0})
            load_rows.append({"timestamp": ts, "ia": 1200.0, "ib": 500.0, "ic": 500.0})  # above threshold

        cur = _FakeCursor(
            temp_rows=temp_rows, load_rows=load_rows,
            ref_row={"id": 1, "model_type": "physics_rc", "artifact_path": self.artifact_rel, "recalibration_note": None},
        )
        result = rj.compute_week(cur, "2", "R", WEEK_START)
        assert result["status"] == "insufficient_data"
        assert 0 < result["n_samples_valid"] < 20


class TestComputeWeekFormulaRecovery:
    def setup_method(self):
        self.i_ref = 1000.0
        self.true_a, self.true_b, self.tau = 25.0, 15.0, 20.0
        self.artifact_rel = _write_physics_rc_artifact(
            "test_rj_recovery.json", self.true_a, self.true_b, self.tau, i_ref=self.i_ref,
        )

    def teardown_method(self):
        (_ARTIFACTS_DIR / "test_rj_recovery.json").unlink(missing_ok=True)

    def test_recovers_known_15_percent_resistance_increase(self):
        """Synthetic data applies R=1.15*R_ref directly to the residual --
        the job must recover ~115% in r_ratio_median_pct purely from the
        generated temperature/current series, without being told 1.15."""
        random.seed(11)
        known_ratio = 1.15
        n = 2016  # 1 semana de amostras de 5 min
        ts_list = [i * 300.0 for i in range(n)]
        currents = [1200.0 + 50 * ((i % 12) / 12.0) for i in range(n)]
        i2_eff_true = ds.compute_i2_eff_series(ts_list, currents, self.tau, 60.0)

        temp_rows, load_rows = [], []
        for i in range(n):
            ts = WEEK_START_DT + timedelta(seconds=ts_list[i])
            i_eff_norm = (i2_eff_true[i] ** 0.5) / self.i_ref
            delta_t_ref = self.true_a * i_eff_norm ** 2 + self.true_b
            residual_applied = (known_ratio - 1.0) * self.true_a * i_eff_norm ** 2
            t_hot = delta_t_ref + residual_applied + random.gauss(0, 0.05)
            temp_rows.append({"timestamp": ts, "r_atas": t_hot, "r_bawah": t_hot - 1.0,
                               "s_atas": 50.0, "s_bawah": 49.0, "t_atas": 50.0, "t_bawah": 49.0})
            load_rows.append({"timestamp": ts, "ia": currents[i], "ib": 500.0, "ic": 500.0})

        cur = _FakeCursor(
            temp_rows=temp_rows, load_rows=load_rows,
            ref_row={"id": 1, "model_type": "physics_rc", "artifact_path": self.artifact_rel, "recalibration_note": None},
        )
        result = rj.compute_week(cur, "2", "R", WEEK_START)
        assert result["status"] == "ok"
        assert abs(result["r_ratio_median_pct"] - known_ratio * 100.0) < 1.0


class TestProcessTrafoPhaseRelative:
    def test_phase_relative_ratio_computed_from_all_three_phases(self):
        week = WEEK_START
        fixed = {
            "R": {"trafo": "2", "phase": "R", "week_start": week, "r_ratio_median_pct": 110.0,
                  "r_ratio_q1_pct": 105.0, "r_ratio_q3_pct": 115.0, "r_ratio_iqr_pct": 10.0,
                  "n_samples_valid": 50, "n_samples_total": 50, "reference_model_id": 1,
                  "a_ref": 25.0, "status": "ok", "reason": None},
            "S": {"trafo": "2", "phase": "S", "week_start": week, "r_ratio_median_pct": 100.0,
                  "r_ratio_q1_pct": 95.0, "r_ratio_q3_pct": 105.0, "r_ratio_iqr_pct": 10.0,
                  "n_samples_valid": 50, "n_samples_total": 50, "reference_model_id": 1,
                  "a_ref": 25.0, "status": "ok", "reason": None},
            "T": {"trafo": "2", "phase": "T", "week_start": week, "r_ratio_median_pct": 95.0,
                  "r_ratio_q1_pct": 90.0, "r_ratio_q3_pct": 100.0, "r_ratio_iqr_pct": 10.0,
                  "n_samples_valid": 50, "n_samples_total": 50, "reference_model_id": 1,
                  "a_ref": 25.0, "status": "ok", "reason": None},
        }

        class _InsertTrackingCursor:
            def __init__(self):
                self.inserted = []

            def execute(self, sql, params=None):
                if "SELECT week_start FROM r_relative_weekly" in sql:
                    self._last = ("row", None)
                elif "INSERT INTO r_relative_weekly" in sql:
                    self.inserted.append(params)
                    self._last = None

            def fetchone(self):
                return self._last[1] if self._last[0] == "row" else None

            def fetchall(self):
                return []

        cur = _InsertTrackingCursor()
        with patch.object(rj, "compute_week", side_effect=lambda c, trafo, phase, w: dict(fixed[phase])):
            n = rj.process_trafo(cur, "2", week)

        assert n == 3
        expected = {"R": 110.0, "S": 100.0, "T": 95.0}  # median of [110,100,95] = 100
        for params in cur.inserted:
            phase, median_pct, phase_relative_pct = params[1], params[3], params[7]
            assert abs(phase_relative_pct - expected[phase]) < 1e-9
