"""Test server/aging_job.py — orkestrasi job harian (checkpoint per hari,
reset kumulatif). Tanpa DB nyata (cursor di-mock), tanpa network.

Jalankan dari folder server/:
    pytest tests/test_aging_job.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import aging  # noqa: E402
import aging_job as aj  # noqa: E402

RESET_DATE = datetime(2026, 9, 1, tzinfo=timezone.utc)
NEW_RESET_DATE = datetime(2026, 9, 10, tzinfo=timezone.utc)


class _FakeCursor:
    def __init__(self):
        self.rows: dict[tuple[str, str, date], dict] = {}
        self._last = None

    def execute(self, sql, params=None):
        if "SELECT date, eq_days_cum" in sql:
            trafo, phase = params
            matching = [r for (t, p, _), r in self.rows.items() if t == trafo and p == phase]
            matching.sort(key=lambda r: r["date"], reverse=True)
            self._last = ("row", matching[0] if matching else None)
        elif "FROM trafo_temperature" in sql:
            self._last = ("rows", [])
        elif "INSERT INTO aging_daily" in sql:
            (trafo, phase, day, f_mean, f_peak, t_mean, t_peak, eq_cum, cal_days,
             cfg_hash, reset_date, reset_source, n_samples, n_gaps) = params
            self.rows[(trafo, phase, day)] = {
                "date": day, "eq_days_cum": eq_cum, "calendar_days": cal_days,
                "reset_date": reset_date, "config_hash": cfg_hash,
            }
            self._last = None

    def fetchone(self):
        return self._last[1] if self._last and self._last[0] == "row" else None

    def fetchall(self):
        return self._last[1] if self._last and self._last[0] == "rows" else []


class _FakeResetDateCursor:
    def __init__(self, event_row):
        self._event_row = event_row

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._event_row


class TestGetResetDate:
    """Stage 2 Part E, tugas #5: fallback_reset_date diganti tabel
    maintenance_events -- 'reset propagation to aging'."""

    def test_uses_maintenance_event_when_present(self):
        event_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
        cur = _FakeResetDateCursor({"event_ts": event_ts})
        reset_date, reset_source = aj.get_reset_date(cur, "2", "R")
        assert reset_date == event_ts
        assert reset_source == "maintenance_event"

    def test_falls_back_to_config_when_no_event(self):
        cur = _FakeResetDateCursor(None)
        reset_date, reset_source = aj.get_reset_date(cur, "2", "R")
        assert reset_source == "fallback_config"
        assert reset_date == datetime.fromisoformat(aging.AGING_CONFIG["fallback_reset_date"])


class TestConfigHash:
    def test_hash_is_stable_for_same_config(self):
        h1 = aj.config_hash({"ea_ev": 0.5, "t_ref_c": 55.0})
        h2 = aj.config_hash({"ea_ev": 0.5, "t_ref_c": 55.0})
        assert h1 == h2

    def test_hash_changes_when_config_changes(self):
        h1 = aj.config_hash({"ea_ev": 0.5, "t_ref_c": 55.0})
        h2 = aj.config_hash({"ea_ev": 0.6, "t_ref_c": 55.0})
        assert h1 != h2

    def test_underscore_prefixed_keys_ignored_in_hash(self):
        """Komentar/catatan (key berawalan _) tidak boleh ikut mengubah
        hash -- hanya nilai fungsional yang relevan."""
        h1 = aj.config_hash({"ea_ev": 0.5, "t_ref_c": 55.0, "_note": "a"})
        h2 = aj.config_hash({"ea_ev": 0.5, "t_ref_c": 55.0, "_note": "b"})
        assert h1 == h2


class TestProcessTrafoPhase:
    def setup_method(self):
        # aging.AGING_CONFIG adalah objek dict TUNGGAL dibagi seluruh proses
        # (default arg fungsi di aging.py terikat ke objek yang SAMA, bukan
        # dibaca ulang dari modul) -- simpan ISI PENUHnya dulu (termasuk
        # fallback_reset_date dkk. yang tidak relevan untuk test ini tapi
        # dipakai test LAIN) supaya bisa dikembalikan utuh di teardown,
        # bukan ditimpa dengan dict minimal yang menghapus key lain.
        self._original_config = dict(aging.AGING_CONFIG)
        aging.AGING_CONFIG["ea_ev"] = 0.5
        aging.AGING_CONFIG["t_ref_c"] = 55.0
        aging.AGING_CONFIG["max_gap_minutes_for_integration"] = 60

    def teardown_method(self):
        aging.AGING_CONFIG.clear()
        aging.AGING_CONFIG.update(self._original_config)

    def test_writes_one_row_per_day_from_reset_to_yesterday(self):
        cur = _FakeCursor()
        cfg_hash = aj.config_hash()
        n = aj.process_trafo_phase(cur, "2", "R", RESET_DATE, "fallback_config", cfg_hash, yesterday=date(2026, 9, 3))
        assert n == 3
        assert set(d for (_, _, d) in cur.rows) == {date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)}

    def test_calendar_days_accumulates_correctly(self):
        cur = _FakeCursor()
        cfg_hash = aj.config_hash()
        aj.process_trafo_phase(cur, "2", "R", RESET_DATE, "fallback_config", cfg_hash, yesterday=date(2026, 9, 3))
        assert cur.rows[("2", "R", date(2026, 9, 3))]["calendar_days"] == 3.0

    def test_continues_from_last_row_on_subsequent_run_same_reset(self):
        cur = _FakeCursor()
        cfg_hash = aj.config_hash()
        aj.process_trafo_phase(cur, "2", "R", RESET_DATE, "fallback_config", cfg_hash, yesterday=date(2026, 9, 3))
        n2 = aj.process_trafo_phase(cur, "2", "R", RESET_DATE, "fallback_config", cfg_hash, yesterday=date(2026, 9, 5))
        assert n2 == 2  # hanya hari 4 dan 5, tidak mengulang 1-3
        assert cur.rows[("2", "R", date(2026, 9, 5))]["calendar_days"] == 5.0

    def test_cumulative_sum_resets_on_new_reset_date(self):
        """Inti persyaratan test Part C: 'cumulative sum resets correctly'."""
        cur = _FakeCursor()
        cfg_hash = aj.config_hash()
        aj.process_trafo_phase(cur, "2", "R", RESET_DATE, "fallback_config", cfg_hash, yesterday=date(2026, 9, 5))
        assert cur.rows[("2", "R", date(2026, 9, 5))]["calendar_days"] == 5.0

        # Event reset BARU (mis. Part E nanti, atau fallback_reset_date diganti manual)
        n3 = aj.process_trafo_phase(cur, "2", "R", NEW_RESET_DATE, "maintenance_event", cfg_hash, yesterday=date(2026, 9, 12))
        assert n3 == 3  # hari 10, 11, 12
        first_row_after_reset = cur.rows[("2", "R", date(2026, 9, 10))]
        assert first_row_after_reset["calendar_days"] == 1.0, "harus restart dari 1, bukan lanjut dari 6"
        assert first_row_after_reset["eq_days_cum"] == 0.0

    def test_no_rows_written_when_not_configured(self):
        aging.AGING_CONFIG.clear()
        aging.AGING_CONFIG.update({"ea_ev": None, "t_ref_c": None})
        cur = _FakeCursor()
        n = aj.process_trafo_phase(cur, "2", "R", RESET_DATE, "fallback_config", "x", yesterday=date(2026, 9, 3))
        assert n == 0
        assert cur.rows == {}


class TestRunOnce:
    def test_returns_zero_and_touches_no_db_when_not_configured(self):
        aging.AGING_CONFIG.clear()
        aging.AGING_CONFIG.update({"ea_ev": None, "t_ref_c": None})

        class _ExplodingPool:
            def connection(self):
                raise AssertionError("should not attempt DB connection when not configured")

        result = aj.run_once(_ExplodingPool())
        assert result == 0
