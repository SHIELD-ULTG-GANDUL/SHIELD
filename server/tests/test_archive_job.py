"""Test server/archive_job.py — pembacaan maintenance_windows pending dan
penyalinan baris ke tabel *_archive. Tanpa akses jaringan, tanpa DB
(cursor di-mock).

Jalankan dari folder server/:
    pytest tests/test_archive_job.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import archive_job as aj  # noqa: E402

NOW = datetime(2026, 10, 14, 10, 0, 0, tzinfo=timezone.utc)


class _FakeCursor:
    def __init__(self, pending_windows, temp_copied_ids, load_copied_ids):
        self._pending_windows = pending_windows
        self._temp_copied_ids = temp_copied_ids
        self._load_copied_ids = load_copied_ids
        self.updates: list[int] = []
        self._last_rows: list = []

    def execute(self, sql, params=None):
        if "SELECT id, trafo_id" in sql:
            self._last_rows = self._pending_windows
        elif "INSERT INTO trafo_temperature_archive" in sql:
            self._last_rows = [{"id": i} for i in self._temp_copied_ids]
        elif "INSERT INTO trafo_load_archive" in sql:
            self._last_rows = [{"id": i} for i in self._load_copied_ids]
        elif "UPDATE maintenance_windows" in sql:
            self.updates.append(params[0])
            self._last_rows = []

    def fetchall(self):
        return self._last_rows


class TestFetchPendingWindows:
    def test_returns_all_pending_rows(self):
        windows = [{"id": 1, "trafo_id": "2", "start_time": NOW, "end_time": NOW, "reason": "x"}]
        cur = _FakeCursor(windows, [], [])
        result = aj.fetch_pending_windows(cur)
        assert result == windows

    def test_empty_when_none_pending(self):
        cur = _FakeCursor([], [], [])
        assert aj.fetch_pending_windows(cur) == []


class TestArchiveWindow:
    def test_single_trafo_window_copies_and_marks_archived(self):
        window = {"id": 1, "trafo_id": "2", "start_time": NOW, "end_time": NOW, "reason": "kalibrasi"}
        cur = _FakeCursor([], [100, 101], [200])
        temp_n, load_n = aj.archive_window(cur, window)
        assert temp_n == 2
        assert load_n == 1
        assert cur.updates == [1]

    def test_all_trafo_window_trafo_id_none(self):
        window = {"id": 2, "trafo_id": None, "start_time": NOW, "end_time": NOW, "reason": "gangguan semua trafo"}
        cur = _FakeCursor([], [100], [])
        temp_n, load_n = aj.archive_window(cur, window)
        assert temp_n == 1
        assert load_n == 0
        assert cur.updates == [2]

    def test_zero_rows_copied_still_marks_archived(self):
        """Window yang tidak beririsan dengan data apa pun (mis. periode
        sebelum ada data masuk) harus tetap ditandai archived -- supaya
        tidak terus-menerus dicoba ulang setiap run."""
        window = {"id": 3, "trafo_id": "4", "start_time": NOW, "end_time": NOW, "reason": "no data in range"}
        cur = _FakeCursor([], [], [])
        temp_n, load_n = aj.archive_window(cur, window)
        assert temp_n == 0
        assert load_n == 0
        assert cur.updates == [3]
