"""Test server/api.py — tanpa akses jaringan, tanpa DB produksi (DB layer
di-mock). Memanggil fungsi route FastAPI langsung (bukan lewat HTTP/
TestClient) supaya tidak butuh httpx terpasang -- cukup fastapi+psycopg
(sudah ada di requirements.txt).

Jalankan dari folder server/:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import api  # noqa: E402


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        self._last_query = query

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self, row_factory=None):
        return _FakeCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, rows):
        self._rows = rows

    def connection(self):
        return _FakeConn(self._rows)


@pytest.fixture
def fake_rows(monkeypatch):
    """Helper: pasang baris palsu yang akan dikembalikan _pool.connection()."""
    def _set(rows):
        monkeypatch.setattr(api, "_pool", _FakePool(rows))
    return _set


class TestHealth:
    def test_health_ok(self):
        result = api.health()
        assert result["status"] == "ok"


class TestTemperatureLogsBounds:
    def test_bounds_with_data(self, fake_rows):
        now = datetime.now(timezone.utc)
        fake_rows([{"min_ts": now, "max_ts": now}])
        result = api.temperature_logs_bounds(trafo=2, x_api_key=None)
        assert result["trafo"] == 2
        assert result["min"] == now.isoformat()
        assert result["max"] == now.isoformat()

    def test_bounds_no_data(self, fake_rows):
        fake_rows([{"min_ts": None, "max_ts": None}])
        result = api.temperature_logs_bounds(trafo=2, x_api_key=None)
        assert result["min"] is None
        assert result["max"] is None


class TestTemperatureLogs:
    def test_no_longer_returns_ambient_field(self, fake_rows):
        """T2/T3: field 'ambient' (dulu dari temp_ambient, selalu 30 -- lihat
        audit Fase 0) tidak boleh ada lagi di respons endpoint ini."""
        now = datetime.now(timezone.utc)
        fake_rows([{
            "trafo": "2", "R_Atas": 55.0, "R_Bawah": 54.0,
            "S_Atas": 50.0, "S_Bawah": 50.0, "T_Atas": 52.0, "T_Bawah": 51.0,
            "quality_flags": None, "ts": now,
        }])
        result = api.temperature_logs(trafo=2, limit=10, from_=None, to_=None, x_api_key=None)
        assert result["count"] == 1
        row = result["rows"][0]
        assert "ambient" not in row
        assert row["timestamp"] == now.isoformat()
        assert "ts" not in row

    def test_null_temperature_passed_through_as_null(self, fake_rows):
        """Field suhu yang NULL di DB harus tetap None di respons JSON,
        bukan dikonversi jadi 0 di lapisan API."""
        now = datetime.now(timezone.utc)
        fake_rows([{
            "trafo": "3", "R_Atas": None, "R_Bawah": 54.0,
            "S_Atas": 50.0, "S_Bawah": 50.0, "T_Atas": 52.0, "T_Bawah": 51.0,
            "quality_flags": None, "ts": now,
        }])
        result = api.temperature_logs(trafo=3, limit=10, from_=None, to_=None, x_api_key=None)
        assert result["rows"][0]["R_Atas"] is None

    def test_empty_result(self, fake_rows):
        fake_rows([])
        result = api.temperature_logs(trafo=2, limit=10, from_=None, to_=None, x_api_key=None)
        assert result["count"] == 0
        assert result["rows"] == []


class TestBebanLogs:
    def test_no_coalesce_null_stays_null(self, fake_rows):
        """T2: COALESCE(...,0) sudah dihapus dari query -- NULL di DB harus
        tetap None di respons, bukan 0."""
        now = datetime.now(timezone.utc)
        fake_rows([{
            "trafo": "2", "I_R": 100.0, "I_S": 98.0, "I_T": 101.0,
            "V_R": None, "V_S": 220.0, "V_T": 220.0,
            "V_RS": None, "V_ST": None, "V_TR": None,
            "kW": 50.0, "kVAR": None, "kVA": 55.0, "PF": 0.9,
            "current_max": 101.0, "freq": 50.0, "quality_flags": None, "ts": now,
        }])
        result = api.beban_logs(trafo=2, limit=10, from_=None, to_=None, x_api_key=None)
        row = result["rows"][0]
        assert row["V_R"] is None, "V_R NULL harus tetap None, bukan 0 (COALESCE dihapus)"
        assert row["kVAR"] is None, "kVAR NULL harus tetap None, bukan 0"
        assert row["I_R"] == 100.0


class TestApiKeyAuth:
    def test_check_api_key_disabled_by_default(self):
        # API_KEY module-level default kosong (auth dimatikan) -- panggilan
        # dengan x_api_key apa pun (termasuk None) tidak boleh raise.
        api._check_api_key(None)  # tidak raise = lolos

    def test_check_api_key_rejects_wrong_key_when_enabled(self, monkeypatch):
        monkeypatch.setattr(api, "API_KEY", "rahasia")
        with pytest.raises(Exception):
            api._check_api_key("salah")

    def test_check_api_key_accepts_correct_key_when_enabled(self, monkeypatch):
        monkeypatch.setattr(api, "API_KEY", "rahasia")
        api._check_api_key("rahasia")  # tidak raise = lolos
