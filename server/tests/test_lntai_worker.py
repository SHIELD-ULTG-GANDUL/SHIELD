"""Test server/lntai_worker.py — pairing, data_quality gating, checkpoint.
Tanpa akses jaringan, tanpa DB (cursor di-mock). lntai_calculator.py sendiri
sudah diuji di test_lntai.py (root proyek) -- di sini hanya logika
pairing/pemilihan-sampel/checkpoint, BUKAN formula LNTAI.

Jalankan dari folder server/:
    pytest tests/test_lntai_worker.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import lntai_worker as w  # noqa: E402
import data_quality as dq  # noqa: E402
from lntai_calculator import LNTAICalculator  # noqa: E402


TS1 = datetime(2026, 10, 14, 10, 0, 0, tzinfo=timezone.utc)
TS2 = datetime(2026, 10, 14, 10, 5, 0, tzinfo=timezone.utc)

TEMP_ROW_OK = {"quality_flags": 0}
LOAD_ROW_OK = {"ia": 500, "ib": 505, "ic": 495, "quality_flags": 0}


class TestCombinedQualityFlags:
    def test_both_clean_rows_yield_zero(self):
        assert w.combined_quality_flags(TEMP_ROW_OK, LOAD_ROW_OK, True) == 0

    def test_not_paired_sets_pairing_gap_bit(self):
        qf = w.combined_quality_flags(TEMP_ROW_OK, None, False)
        assert qf == dq.flags_to_bitmask({"PAIRING_GAP"})

    def test_bits_from_both_rows_are_ored_together(self):
        temp_row = {"quality_flags": dq.flags_to_bitmask({"ZERO_READING"})}
        load_row = {"quality_flags": dq.flags_to_bitmask({"OUT_OF_RANGE"})}
        qf = w.combined_quality_flags(temp_row, load_row, True)
        assert qf == dq.flags_to_bitmask({"ZERO_READING", "OUT_OF_RANGE"})


class TestShouldCompute:
    def test_positive_clean_and_paired(self):
        assert w.should_compute(0, True) is True

    def test_negative_not_paired_blocks_even_with_zero_flags(self):
        assert w.should_compute(0, False) is False

    def test_negative_missing_blocks(self):
        qf = dq.flags_to_bitmask({"MISSING"})
        assert w.should_compute(qf, True) is False

    def test_positive_zero_reading_does_not_block(self):
        """T2/T5: ZERO_READING adalah bacaan sah, tidak menghentikan LNTAI."""
        qf = dq.flags_to_bitmask({"ZERO_READING"})
        assert w.should_compute(qf, True) is True


class TestBuildSample:
    def test_with_load_row_fills_currents(self):
        temp_row = {"r_atas": 55, "r_bawah": 54, "s_atas": 56, "s_bawah": 53, "t_atas": 57, "t_bawah": 52}
        sample = w.build_sample(temp_row, LOAD_ROW_OK, 29.0, "ambient_logs", TS1)
        assert sample["I_R"] == 500
        assert sample["I_S"] == 505
        assert sample["I_T"] == 495
        assert sample["t_ambient"] == 29.0
        assert sample["ambient_source"] == "ambient_logs"
        assert sample["timestamp"] == TS1.timestamp()

    def test_without_load_row_currents_are_none(self):
        temp_row = {"r_atas": 55, "r_bawah": 54, "s_atas": 56, "s_bawah": 53, "t_atas": 57, "t_bawah": 52}
        sample = w.build_sample(temp_row, None, None, "default", TS1)
        assert sample["I_R"] is None
        assert sample["I_S"] is None
        assert sample["I_T"] is None


class _FakeCursor:
    """Simulasi minimal worker_state + trafo_temperature + trafo_load +
    ambient_logs + lntai_results lewat pencocokan substring query."""

    def __init__(self, temp_rows: list[dict], load_row: dict | None):
        self.worker_state: dict[tuple[str, str], datetime] = {}
        self.inserted: list[tuple] = []
        self._temp_rows = temp_rows
        self._load_row = load_row
        self._last = None

    def execute(self, sql, params=None):
        if "SELECT last_processed_ts FROM worker_state" in sql:
            key = (params[0], params[1])
            val = self.worker_state.get(key)
            self._last = ("row", {"last_processed_ts": val} if val is not None else None)
        elif "INSERT INTO worker_state" in sql:
            key = (params[0], params[1])
            self.worker_state[key] = params[2]
            self._last = None
        elif "FROM trafo_temperature" in sql:
            since = params[1]
            rows = [r for r in self._temp_rows if since is None or r["timestamp"] > since]
            self._last = ("rows", rows)
        elif "FROM trafo_load" in sql:
            self._last = ("row", self._load_row)
        elif "FROM ambient_logs" in sql:
            self._last = ("row", None)
        elif "INSERT INTO lntai_results" in sql:
            self.inserted.append(params)
            self._last = None

    def fetchone(self):
        return self._last[1] if self._last and self._last[0] == "row" else None

    def fetchall(self):
        return self._last[1] if self._last and self._last[0] == "rows" else []


class TestProcessTrafoBatch:
    def _temp_rows(self):
        return [
            {"timestamp": TS1, "r_atas": 55, "r_bawah": 54, "s_atas": 56, "s_bawah": 53,
             "t_atas": 57, "t_bawah": 52, "quality_flags": 0},
            {"timestamp": TS2, "r_atas": 56, "r_bawah": 55, "s_atas": 57, "s_bawah": 54,
             "t_atas": 58, "t_bawah": 53, "quality_flags": 0},
        ]

    def test_processes_all_new_rows_and_advances_checkpoint(self):
        cur = _FakeCursor(self._temp_rows(), LOAD_ROW_OK)
        calc = LNTAICalculator()
        n = w.process_trafo_batch(cur, calc, "2")
        assert n == 2
        assert len(cur.inserted) == 2
        assert cur.worker_state[("lntai_worker", "2")] == TS2

    def test_respects_checkpoint_on_second_run(self):
        cur = _FakeCursor(self._temp_rows(), LOAD_ROW_OK)
        calc = LNTAICalculator()
        w.process_trafo_batch(cur, calc, "2")
        n_second = w.process_trafo_batch(cur, calc, "2")
        assert n_second == 0

    def test_no_load_row_still_inserts_data_invalid_row(self):
        cur = _FakeCursor(self._temp_rows(), None)
        calc = LNTAICalculator()
        n = w.process_trafo_batch(cur, calc, "2")
        assert n == 2
        assert len(cur.inserted) == 2
        # level is at index 4 in the INSERT params tuple (see insert_result())
        assert all(row[4] == "DATA_INVALID" for row in cur.inserted)

    def test_valid_pairing_produces_non_invalid_level(self):
        cur = _FakeCursor(self._temp_rows(), LOAD_ROW_OK)
        calc = LNTAICalculator()
        w.process_trafo_batch(cur, calc, "2")
        assert all(row[4] in ("NORMAL", "WARNING", "CRITICAL") for row in cur.inserted)

    def test_empty_history_processes_nothing(self):
        cur = _FakeCursor([], LOAD_ROW_OK)
        calc = LNTAICalculator()
        n = w.process_trafo_batch(cur, calc, "2")
        assert n == 0
        assert cur.inserted == []
