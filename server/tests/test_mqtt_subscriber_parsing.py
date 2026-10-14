"""Test parsing & NULL-handling di mqtt_subscriber.py (T2, cabang A).

Tanpa akses jaringan, tanpa DB produksi — DB layer (_pool) di-mock.
Jalankan dari folder server/:
    pytest tests/test_mqtt_subscriber_parsing.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import mqtt_subscriber as sub  # noqa: E402


# ── _to_float_or_none: fungsi murni, inti dari perbaikan T2 ──────────────

class TestToFloatOrNone:
    def test_missing_key_is_none(self):
        assert sub._to_float_or_none({}, "ia") is None

    def test_explicit_null_is_none(self):
        assert sub._to_float_or_none({"ia": None}, "ia") is None

    def test_zero_is_preserved_not_none(self):
        """Inti bug T2: 0 adalah bacaan sah (mis. CB terbuka), bukan 'tidak ada data'."""
        assert sub._to_float_or_none({"ia": 0}, "ia") == 0.0

    def test_invalid_string_is_none(self):
        assert sub._to_float_or_none({"ia": "bukan-angka"}, "ia") is None

    def test_valid_numeric_string_converts(self):
        assert sub._to_float_or_none({"ia": "12.5"}, "ia") == 12.5

    def test_negative_value_preserved(self):
        assert sub._to_float_or_none({"ia": -3.2}, "ia") == -3.2


# ── save_to_database: DB layer di-mock, cek parameter INSERT yang dikirim ──

class _FakeCursor:
    def __init__(self, sink: dict):
        self._sink = sink

    def execute(self, query, params):
        if "trafo_load" in query:
            self._sink["load_params"] = params
        elif "trafo_temperature" in query:
            self._sink["temp_params"] = params

    def fetchone(self):
        return (1,)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, sink: dict):
        self._sink = sink

    def cursor(self):
        return _FakeCursor(self._sink)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.sink: dict = {}

    def connection(self):
        return _FakeConn(self.sink)


@pytest.fixture
def fake_pool(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(sub, "_pool", pool)
    return pool


class TestSaveToDatabase:
    def test_missing_fields_become_null_not_zero(self, fake_pool):
        ok = sub.save_to_database({"trafo_id": "2", "ia": 5})
        assert ok is True

        load_params = fake_pool.sink["load_params"]
        # (timestamp, trafo_id, ia, ib, ic, current_max, kw_total, ...)
        assert load_params[2] == 5.0      # ia ada
        assert load_params[3] is None     # ib hilang -> None, bukan 0
        assert load_params[4] is None     # ic hilang -> None

        temp_params = fake_pool.sink["temp_params"]
        # (timestamp, trafo_id, r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah, temp_max, temp_avg)
        assert all(v is None for v in temp_params[2:8]), "semua suhu hilang harus None"
        assert temp_params[8] is None, "temp_max harus None kalau tidak ada suhu valid"
        assert temp_params[9] is None, "temp_avg harus None kalau tidak ada suhu valid"

    def test_zero_reading_is_preserved_and_counted(self, fake_pool):
        ok = sub.save_to_database({
            "trafo_id": "2",
            "r_atas": 0, "r_bawah": 55, "s_atas": 50, "s_bawah": 50,
            "t_atas": 50, "t_bawah": 50,
        })
        assert ok is True
        temp_params = fake_pool.sink["temp_params"]
        assert temp_params[2] == 0.0, "r_atas=0 harus tetap tersimpan sebagai 0.0"
        assert temp_params[8] == 55.0, "temp_max tetap benar walau salah satu titik 0"

    def test_current_max_ignores_missing_phases(self, fake_pool):
        ok = sub.save_to_database({"trafo_id": "2", "ia": 10, "ic": 7})
        assert ok is True
        load_params = fake_pool.sink["load_params"]
        assert load_params[5] == 10.0  # current_max = max(10, 7), ib diabaikan karena None

    def test_empty_payload_does_not_crash(self, fake_pool):
        ok = sub.save_to_database({})
        assert ok is True
        temp_params = fake_pool.sink["temp_params"]
        assert all(v is None for v in temp_params[2:10])


# ── quality_flags (T5): dihitung per-sampel saat tulis, bitmask di kolom terakhir ──

class TestQualityFlagsWiring:
    def test_all_valid_normal_readings_have_zero_flags(self, fake_pool):
        ok = sub.save_to_database({
            "trafo_id": "2", "ia": 500, "ib": 510, "ic": 490,
            "r_atas": 55, "r_bawah": 54, "s_atas": 56,
            "s_bawah": 53, "t_atas": 57, "t_bawah": 52,
        })
        assert ok is True
        assert fake_pool.sink["load_params"][-1] == 0
        assert fake_pool.sink["temp_params"][-1] == 0

    def test_missing_current_sets_missing_bit(self, fake_pool):
        ok = sub.save_to_database({"trafo_id": "2", "ib": 510, "ic": 490})
        assert ok is True
        import data_quality as dq
        assert fake_pool.sink["load_params"][-1] == dq.flags_to_bitmask({"MISSING"})

    def test_zero_temperature_sets_zero_reading_bit_not_missing(self, fake_pool):
        ok = sub.save_to_database({
            "trafo_id": "2", "r_atas": 0, "r_bawah": 54, "s_atas": 56,
            "s_bawah": 53, "t_atas": 57, "t_bawah": 52,
        })
        assert ok is True
        import data_quality as dq
        assert fake_pool.sink["temp_params"][-1] == dq.flags_to_bitmask({"ZERO_READING"})

    def test_out_of_range_temperature_sets_bit(self, fake_pool):
        ok = sub.save_to_database({
            "trafo_id": "2", "r_atas": 999, "r_bawah": 54, "s_atas": 56,
            "s_bawah": 53, "t_atas": 57, "t_bawah": 52,
        })
        assert ok is True
        import data_quality as dq
        assert fake_pool.sink["temp_params"][-1] == dq.flags_to_bitmask({"OUT_OF_RANGE"})

    def test_all_missing_payload_sets_missing_on_both_rows(self, fake_pool):
        ok = sub.save_to_database({"trafo_id": "2"})
        assert ok is True
        import data_quality as dq
        expected = dq.flags_to_bitmask({"MISSING"})
        assert fake_pool.sink["load_params"][-1] == expected
        assert fake_pool.sink["temp_params"][-1] == expected
