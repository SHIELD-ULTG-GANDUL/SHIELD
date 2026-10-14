"""Test server/ambient.py — fungsi murni (floor_to_slot, validasi rentang,
ambient_at()). Tanpa akses jaringan, tanpa DB (cursor di-mock).

Jalankan dari folder server/:
    pytest tests/test_ambient.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ambient  # noqa: E402


class TestFloorToSlot:
    def test_rounds_down_within_slot(self):
        ts = datetime(2026, 10, 14, 10, 37, 22, tzinfo=timezone.utc)
        assert ambient.floor_to_slot(ts) == datetime(2026, 10, 14, 10, 30, tzinfo=timezone.utc)

    def test_exact_slot_boundary_unchanged(self):
        ts = datetime(2026, 10, 14, 10, 0, 1, tzinfo=timezone.utc)
        assert ambient.floor_to_slot(ts) == datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)

    def test_different_timezone_normalized_to_utc(self):
        from datetime import timedelta

        wib = timezone(timedelta(hours=7))
        ts = datetime(2026, 10, 14, 17, 37, 0, tzinfo=wib)  # == 10:37 UTC
        assert ambient.floor_to_slot(ts) == datetime(2026, 10, 14, 10, 30, tzinfo=timezone.utc)


class TestIsValidAmbientTemperature:
    def test_positive_within_range(self):
        assert ambient.is_valid_ambient_temperature(29.5) is True

    def test_negative_none(self):
        assert ambient.is_valid_ambient_temperature(None) is False

    def test_negative_too_high(self):
        assert ambient.is_valid_ambient_temperature(999) is False

    def test_negative_too_low(self):
        assert ambient.is_valid_ambient_temperature(-50) is False


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def execute(self, sql, params):
        pass

    def fetchone(self):
        return self._row


class TestAmbientAt:
    def test_positive_found_within_window(self):
        cur = _FakeCursor((28.3,))
        value, source = ambient.ambient_at(
            cur, datetime(2026, 10, 14, 10, 30, tzinfo=timezone.utc), max_age_minutes=30
        )
        assert value == 28.3
        assert source == "ambient_logs"

    def test_negative_falls_back_to_default_when_not_found(self):
        cur = _FakeCursor(None)
        value, source = ambient.ambient_at(
            cur,
            datetime(2026, 10, 14, 10, 30, tzinfo=timezone.utc),
            max_age_minutes=30,
            fallback_default_c=30.0,
        )
        assert value == 30.0
        assert source == "default"

    def test_negative_no_fallback_given_returns_none(self):
        cur = _FakeCursor(None)
        value, source = ambient.ambient_at(
            cur, datetime(2026, 10, 14, 10, 30, tzinfo=timezone.utc), max_age_minutes=30
        )
        assert value is None
        assert source == "default"
