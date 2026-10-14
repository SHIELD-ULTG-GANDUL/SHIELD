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
from datetime import datetime, timedelta, timezone
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


# ── /api/ambient-logs (T3, Cabang B) ─────────────────────────────────────

class _FakeAmbientCursor:
    """Simulasi INSERT ... ON CONFLICT (slot_10min) DO NOTHING RETURNING id
    -- `seen_slots` dibagi antar panggilan dalam satu test lewat referensi."""

    def __init__(self, seen_slots: set):
        self._seen_slots = seen_slots
        self._last_row = None

    def execute(self, query, params=None):
        if "INSERT INTO ambient_logs" in query:
            slot = params[1]
            if slot in self._seen_slots:
                self._last_row = None
            else:
                self._seen_slots.add(slot)
                self._last_row = {"id": 1}
        elif "FROM ambient_logs" in query and "SELECT \"timestamp\"" in query:
            self._last_row = None

    def fetchone(self):
        return self._last_row

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAmbientConn:
    def __init__(self, seen_slots: set):
        self._seen_slots = seen_slots

    def cursor(self, row_factory=None):
        return _FakeAmbientCursor(self._seen_slots)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAmbientPool:
    def __init__(self):
        self._seen_slots: set = set()

    def connection(self):
        return _FakeAmbientConn(self._seen_slots)


@pytest.fixture
def fake_ambient_pool(monkeypatch):
    pool = _FakeAmbientPool()
    monkeypatch.setattr(api, "_pool", pool)
    return pool


class TestCreateAmbientLog:
    def test_valid_ok_true_is_stored(self, fake_ambient_pool):
        body = api.AmbientLogIn(timestamp="2026-10-14T10:03:00+00:00", temperature_c=29.5, ok=True)
        result = api.create_ambient_log(body, x_api_key=None)
        assert result["stored"] is True

    def test_duplicate_slot_is_not_stored_twice(self, fake_ambient_pool):
        body1 = api.AmbientLogIn(timestamp="2026-10-14T10:03:00+00:00", temperature_c=29.5, ok=True)
        body2 = api.AmbientLogIn(timestamp="2026-10-14T10:07:00+00:00", temperature_c=30.0, ok=True)
        api.create_ambient_log(body1, x_api_key=None)
        result = api.create_ambient_log(body2, x_api_key=None)
        assert result["stored"] is False, "slot 10-menit yang sama harus dedup, bukan ditimpa"

    def test_ok_false_with_null_temperature_is_accepted(self, fake_ambient_pool):
        """ok=false (fetch Open-Meteo gagal di browser) adalah sinyal kualitas
        data yang sah -- TIDAK ditolak, TIDAK dikarang nilainya."""
        body = api.AmbientLogIn(timestamp="2026-10-14T10:23:00+00:00", temperature_c=None, ok=False)
        result = api.create_ambient_log(body, x_api_key=None)
        assert result["stored"] is True

    def test_out_of_range_temperature_rejected(self, fake_ambient_pool):
        from fastapi import HTTPException

        body = api.AmbientLogIn(timestamp="2026-10-14T10:33:00+00:00", temperature_c=999.0, ok=True)
        with pytest.raises(HTTPException) as exc_info:
            api.create_ambient_log(body, x_api_key=None)
        assert exc_info.value.status_code == 400


class TestAmbientLogsGet:
    def test_empty_result(self, fake_ambient_pool):
        result = api.ambient_logs(limit=10, from_=None, to_=None, x_api_key=None)
        assert result["count"] == 0
        assert result["rows"] == []


# ── /api/lntai-logs, /api/lntai-latest (T4) ──────────────────────────────

class TestLntaiLogs:
    def test_rows_renamed_ts_to_timestamp(self, fake_rows):
        now = datetime.now(timezone.utc)
        fake_rows([{
            "ts": now, "model_version": "lntai-v2", "lntai": 12.5, "level": "NORMAL",
            "s_temp": 5.0, "s_delta_t": 3.0, "s_res": 4.5, "hottest_phase": "R",
            "t_contact": 55.0, "t_ambient": 29.0, "ambient_source": "ambient_logs",
            "delta_t_real": 26.0, "delta_t_pred": 25.0, "residual": 1.0, "z": 0.5,
            "i_eff_norm": 0.3, "delta_phase": 2.0, "dab_max": 1.0, "per_phase": {},
            "quality_flags": 0, "paired": True, "error": None,
        }])
        result = api.lntai_logs(trafo=2, limit=10, from_=None, to_=None, x_api_key=None)
        assert result["count"] == 1
        row = result["rows"][0]
        assert row["timestamp"] == now.isoformat()
        assert "ts" not in row
        assert row["lntai"] == 12.5

    def test_empty_result(self, fake_rows):
        fake_rows([])
        result = api.lntai_logs(trafo=2, limit=10, from_=None, to_=None, x_api_key=None)
        assert result["count"] == 0


class _FakeDataQualityCursor:
    def __init__(self, temp_rows, load_rows):
        self._temp_rows = temp_rows
        self._load_rows = load_rows
        self._last = []

    def execute(self, sql, params=None):
        self._last = self._temp_rows if "trafo_temperature" in sql else self._load_rows

    def fetchall(self):
        return self._last

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDataQualityConn:
    def __init__(self, temp_rows, load_rows):
        self._temp_rows = temp_rows
        self._load_rows = load_rows

    def cursor(self, row_factory=None):
        return _FakeDataQualityCursor(self._temp_rows, self._load_rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDataQualityPool:
    def __init__(self, temp_rows, load_rows):
        self._temp_rows = temp_rows
        self._load_rows = load_rows

    def connection(self):
        return _FakeDataQualityConn(self._temp_rows, self._load_rows)


class TestDataQualitySummary:
    def test_counts_flags_per_bit(self, monkeypatch):
        import data_quality as dq

        temp_rows = [
            {"quality_flags": 0},
            {"quality_flags": dq.flags_to_bitmask({"MISSING"})},
            {"quality_flags": dq.flags_to_bitmask({"ZERO_READING", "OUT_OF_RANGE"})},
        ]
        load_rows = [{"quality_flags": 0}, {"quality_flags": None}]
        monkeypatch.setattr(api, "_pool", _FakeDataQualityPool(temp_rows, load_rows))

        result = api.data_quality_summary(trafo=2, window=1440, x_api_key=None)
        assert result["temperature"]["total_rows"] == 3
        assert result["temperature"]["flag_counts"]["MISSING"] == 1
        assert result["temperature"]["flag_counts"]["ZERO_READING"] == 1
        assert result["temperature"]["flag_counts"]["OUT_OF_RANGE"] == 1
        assert result["load"]["total_rows"] == 2
        assert result["load"]["flag_counts"]["MISSING"] == 0

    def test_null_quality_flags_counts_as_clean(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeDataQualityPool([{"quality_flags": None}], []))
        result = api.data_quality_summary(trafo=2, window=60, x_api_key=None)
        assert result["temperature"]["total_rows"] == 1
        assert all(v == 0 for v in result["temperature"]["flag_counts"].values())

    def test_empty_window_returns_zero_counts(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeDataQualityPool([], []))
        result = api.data_quality_summary(trafo=2, window=60, x_api_key=None)
        assert result["temperature"]["total_rows"] == 0
        assert result["load"]["total_rows"] == 0


class _FakeMaintenanceCursor:
    def __init__(self, list_rows):
        self._list_rows = list_rows
        self._last = None

    def execute(self, sql, params=None):
        if "INSERT INTO maintenance_windows" in sql:
            self._last = ("row", {"id": 1})
        elif "SELECT id, trafo_id" in sql:
            self._last = ("rows", self._list_rows)

    def fetchone(self):
        return self._last[1] if self._last and self._last[0] == "row" else None

    def fetchall(self):
        return self._last[1] if self._last and self._last[0] == "rows" else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeMaintenanceConn:
    def __init__(self, list_rows):
        self._list_rows = list_rows

    def cursor(self, row_factory=None):
        return _FakeMaintenanceCursor(self._list_rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeMaintenancePool:
    def __init__(self, list_rows=None):
        self._list_rows = list_rows or []

    def connection(self):
        return _FakeMaintenanceConn(self._list_rows)


class TestCreateMaintenanceWindow:
    def test_valid_window_is_created(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMaintenancePool())
        now = datetime.now(timezone.utc)
        body = api.MaintenanceWindowIn(
            trafo_id="2", start_time=now, end_time=now + timedelta(hours=1), reason="kalibrasi sensor"
        )
        result = api.create_maintenance_window(body, x_api_key=None)
        assert result["id"] == 1

    def test_end_before_start_rejected(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMaintenancePool())
        now = datetime.now(timezone.utc)
        body = api.MaintenanceWindowIn(
            trafo_id="2", start_time=now, end_time=now - timedelta(hours=1), reason="invalid"
        )
        with pytest.raises(Exception):
            api.create_maintenance_window(body, x_api_key=None)

    def test_trafo_id_none_allowed_for_all_trafo(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMaintenancePool())
        now = datetime.now(timezone.utc)
        body = api.MaintenanceWindowIn(
            trafo_id=None, start_time=now, end_time=now + timedelta(hours=1), reason="gangguan semua trafo"
        )
        result = api.create_maintenance_window(body, x_api_key=None)
        assert result["id"] == 1


class TestMaintenanceWindowsList:
    def test_list_returns_rows_with_isoformat_timestamps(self, monkeypatch):
        now = datetime.now(timezone.utc)
        rows = [{
            "id": 1, "trafo_id": "2", "start_time": now, "end_time": now,
            "reason": "test", "created_by": None, "archived": False, "created_at": now,
        }]
        monkeypatch.setattr(api, "_pool", _FakeMaintenancePool(rows))
        result = api.maintenance_windows(trafo=2, x_api_key=None)
        assert result["count"] == 1
        assert result["rows"][0]["start_time"] == now.isoformat()

    def test_empty_list(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMaintenancePool([]))
        result = api.maintenance_windows(trafo=None, x_api_key=None)
        assert result["count"] == 0


class TestLntaiLatest:
    def test_returns_none_row_when_no_data(self, fake_rows):
        fake_rows([])
        result = api.lntai_latest(trafo=2, x_api_key=None)
        assert result["row"] is None

    def test_returns_latest_row(self, fake_rows):
        now = datetime.now(timezone.utc)
        fake_rows([{
            "ts": now, "model_version": "lntai-v2", "lntai": 20.0, "level": "WARNING",
            "s_temp": 8.0, "s_delta_t": 4.0, "s_res": 8.0, "hottest_phase": "S",
            "t_contact": 60.0, "t_ambient": 30.0, "ambient_source": "default",
            "delta_t_real": 30.0, "delta_t_pred": 28.0, "residual": 2.0, "z": 1.0,
            "i_eff_norm": 0.4, "delta_phase": 3.0, "dab_max": 2.0, "per_phase": {},
            "quality_flags": 0, "paired": True, "error": None,
        }])
        result = api.lntai_latest(trafo=2, x_api_key=None)
        assert result["row"]["level"] == "WARNING"
        assert result["row"]["timestamp"] == now.isoformat()


# ── /api/ml/models (Stage 2, Part B) ─────────────────────────────────────

class _FakeMlModelsCursor:
    def __init__(self, list_rows, known_ids: set):
        self._list_rows = list_rows
        self._known_ids = known_ids
        self._last = None

    def execute(self, sql, params=None):
        if "FROM ml_models" in sql and sql.strip().startswith("SELECT id, trafo, phase, model_type, version"):
            self._last = ("rows", self._list_rows)
        elif "SELECT trafo, phase FROM ml_models" in sql or "SELECT id, trafo, phase FROM ml_models" in sql:
            model_id = params[0]
            self._last = ("row", {"id": model_id, "trafo": "2", "phase": "R"}) if model_id in self._known_ids else ("row", None)
        elif "UPDATE ml_models" in sql:
            self._last = None

    def fetchone(self):
        return self._last[1] if self._last and self._last[0] == "row" else None

    def fetchall(self):
        return self._last[1] if self._last and self._last[0] == "rows" else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeMlModelsConn:
    def __init__(self, list_rows, known_ids):
        self._list_rows = list_rows
        self._known_ids = known_ids

    def cursor(self, row_factory=None):
        return _FakeMlModelsCursor(self._list_rows, self._known_ids)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeMlModelsPool:
    def __init__(self, list_rows=None, known_ids=None):
        self._list_rows = list_rows or []
        self._known_ids = known_ids or set()

    def connection(self):
        return _FakeMlModelsConn(self._list_rows, self._known_ids)


class TestMlModelsList:
    def test_list_returns_rows_with_isoformat_timestamps(self, monkeypatch):
        now = datetime.now(timezone.utc)
        rows = [{
            "id": 1, "trafo": "2", "phase": "R", "model_type": "physics_rc", "version": "v1",
            "trained_at": now, "healthy_from": now, "healthy_to": now,
            "features": {}, "metrics": {"sigma": 0.3}, "artifact_path": "ml/artifacts/x.json",
            "sigma": 0.3, "mu": 0.0, "is_active": False, "is_frozen_reference": False, "notes": None,
        }]
        monkeypatch.setattr(api, "_pool", _FakeMlModelsPool(list_rows=rows))
        result = api.ml_models_list(trafo=2, x_api_key=None)
        assert result["count"] == 1
        assert result["rows"][0]["trained_at"] == now.isoformat()

    def test_empty_list(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMlModelsPool())
        result = api.ml_models_list(trafo=None, x_api_key=None)
        assert result["count"] == 0


class TestMlModelActivate:
    def test_activate_known_id(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMlModelsPool(known_ids={1}))
        result = api.ml_model_activate(1, x_api_key=None)
        assert result["is_active"] is True
        assert result["trafo"] == "2" and result["phase"] == "R"

    def test_activate_unknown_id_raises_404(self, monkeypatch):
        from fastapi import HTTPException

        monkeypatch.setattr(api, "_pool", _FakeMlModelsPool(known_ids=set()))
        with pytest.raises(HTTPException) as exc_info:
            api.ml_model_activate(999, x_api_key=None)
        assert exc_info.value.status_code == 404


class TestMlModelFreezeReference:
    def test_freeze_known_id(self, monkeypatch):
        monkeypatch.setattr(api, "_pool", _FakeMlModelsPool(known_ids={1}))
        result = api.ml_model_freeze_reference(1, x_api_key=None)
        assert result["is_frozen_reference"] is True

    def test_freeze_unknown_id_raises_404(self, monkeypatch):
        from fastapi import HTTPException

        monkeypatch.setattr(api, "_pool", _FakeMlModelsPool(known_ids=set()))
        with pytest.raises(HTTPException) as exc_info:
            api.ml_model_freeze_reference(999, x_api_key=None)
        assert exc_info.value.status_code == 404


# ── /api/aging/latest, /api/aging/daily (Stage 2, Part C) ────────────────

class _FakeAgingCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAgingConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self, row_factory=None):
        return _FakeAgingCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAgingPool:
    def __init__(self, rows=None):
        self._rows = rows or []

    def connection(self):
        return _FakeAgingConn(self._rows)


class _ExplodingPool:
    def connection(self):
        raise AssertionError("should not touch DB when aging is not_configured")


def _make_aging_row():
    now = datetime.now(timezone.utc)
    return {
        "phase": "R", "date": now.date(), "f_mean": 1.3, "f_peak": 1.5,
        "t_hot_mean_c": 60.0, "t_hot_peak_c": 65.0, "eq_days_cum": 12.5,
        "calendar_days": 13.0, "reset_date": now, "reset_source": "fallback_config",
        "n_samples": 288, "n_gaps_excluded": 0,
    }


def _set_aging_configured(configured: bool):
    """aging.is_configured()'s default `cfg` param is bound at function-def
    time to the ORIGINAL AGING_CONFIG dict object -- monkeypatch.setattr
    swapping in a NEW dict would NOT be observed by that default arg.
    Mutate the SAME dict object in place instead (like aging_job.py's own
    tests do)."""
    api.aging.AGING_CONFIG.clear()
    if configured:
        api.aging.AGING_CONFIG.update({"ea_ev": 0.5, "t_ref_c": 55.0, "max_gap_minutes_for_integration": 60})
    else:
        api.aging.AGING_CONFIG.update({"ea_ev": None, "t_ref_c": None})


class TestAgingLatest:
    def test_not_configured_returns_empty_without_touching_db(self, monkeypatch):
        _set_aging_configured(False)
        monkeypatch.setattr(api, "_pool", _ExplodingPool())
        result = api.aging_latest(trafo=2, x_api_key=None)
        assert result["status"] == "not_configured"
        assert result["rows"] == []
        assert "contact-finger" in result["scope"]

    def test_configured_returns_rows_with_isoformat_dates(self, monkeypatch):
        _set_aging_configured(True)
        try:
            row = _make_aging_row()
            expected_date_iso = row["date"].isoformat()  # capture BEFORE the endpoint mutates row in place
            monkeypatch.setattr(api, "_pool", _FakeAgingPool([row]))
            result = api.aging_latest(trafo=2, x_api_key=None)
            assert result["status"] == "ok"
            assert result["rows"][0]["date"] == expected_date_iso
        finally:
            _set_aging_configured(False)  # restore shipped default (ea_ev/t_ref_c null) for other tests


class TestAgingDaily:
    def test_not_configured_returns_empty_without_touching_db(self, monkeypatch):
        _set_aging_configured(False)
        monkeypatch.setattr(api, "_pool", _ExplodingPool())
        result = api.aging_daily_endpoint(trafo=2, from_=None, to_=None, x_api_key=None)
        assert result["status"] == "not_configured"

    def test_configured_returns_count_and_rows(self, monkeypatch):
        _set_aging_configured(True)
        try:
            row = _make_aging_row()
            monkeypatch.setattr(api, "_pool", _FakeAgingPool([row]))
            result = api.aging_daily_endpoint(trafo=2, from_=None, to_=None, x_api_key=None)
            assert result["status"] == "ok"
            assert result["count"] == 1
        finally:
            _set_aging_configured(False)
