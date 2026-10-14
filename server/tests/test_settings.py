"""Test server/settings.py — tanpa akses jaringan, tanpa DB produksi.

Jalankan dari folder server/:
    pytest tests/test_settings.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Pastikan tiap test mulai dari environment bersih — tidak ada env var
    settings.py yang bocor dari shell tempat pytest dijalankan."""
    for name in [
        "DATABASE_URL", "API_KEY", "ALLOWED_ORIGINS",
        "HIVEMQ_BROKER", "HIVEMQ_PORT", "HIVEMQ_USER", "HIVEMQ_PASS",
        "MQTT_BROKER", "MQTT_PORT", "MQTT_TOPIC", "MQTT_LOCAL_USER", "MQTT_LOCAL_PASS",
        "GANDUL_LATITUDE", "GANDUL_LONGITUDE",
        "OUTBOUND_HTTPS_PROXY", "OUTBOUND_CA_BUNDLE",
        "AMBIENT_WORKER_INTERVAL_SECONDS", "LNTAI_WORKER_INTERVAL_SECONDS", "WRITE_INTERVAL_SECONDS",
    ]:
        monkeypatch.delenv(name, raising=False)


class TestDatabaseSettings:
    def test_missing_database_url_raises(self):
        with pytest.raises(settings.MissingSettingError, match="DATABASE_URL"):
            settings.load_database_settings()

    def test_present_database_url_loads(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")
        result = settings.load_database_settings()
        assert result.url == "postgresql://u:p@host:5432/db"

    def test_empty_string_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "")
        with pytest.raises(settings.MissingSettingError):
            settings.load_database_settings()


class TestApiSettings:
    def test_default_api_key_is_empty_auth_disabled(self):
        result = settings.load_api_settings()
        assert result.api_key == ""

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret123")
        assert settings.load_api_settings().api_key == "secret123"

    def test_default_allowed_origins(self):
        result = settings.load_api_settings()
        assert result.allowed_origins == ["https://shield-ultg-gandul.github.io"]

    def test_allowed_origins_splits_on_comma_and_trims(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", " https://a.com, https://b.com ,")
        result = settings.load_api_settings()
        assert result.allowed_origins == ["https://a.com", "https://b.com"]


class TestHiveMqSettings:
    def test_missing_all_raises_on_broker_first(self):
        with pytest.raises(settings.MissingSettingError, match="HIVEMQ_BROKER"):
            settings.load_hivemq_settings()

    def test_missing_user_raises_even_if_broker_set(self, monkeypatch):
        monkeypatch.setenv("HIVEMQ_BROKER", "x.hivemq.cloud")
        with pytest.raises(settings.MissingSettingError, match="HIVEMQ_USER"):
            settings.load_hivemq_settings()

    def test_no_default_falls_back_to_real_broker_hostname(self, monkeypatch):
        """Regresi: pastikan tidak ada jalur yang diam-diam mengembalikan
        hostname/password HiveMQ produksi lama kalau env var kosong."""
        monkeypatch.setenv("HIVEMQ_BROKER", "x.hivemq.cloud")
        monkeypatch.setenv("HIVEMQ_USER", "u")
        monkeypatch.setenv("HIVEMQ_PASS", "p")
        result = settings.load_hivemq_settings()
        assert result.broker == "x.hivemq.cloud"
        assert result.password == "p"
        assert "webshield" not in result.user.lower()
        assert "webShield123" != result.password

    def test_port_default(self, monkeypatch):
        monkeypatch.setenv("HIVEMQ_BROKER", "x")
        monkeypatch.setenv("HIVEMQ_USER", "u")
        monkeypatch.setenv("HIVEMQ_PASS", "p")
        assert settings.load_hivemq_settings().port == 8884


class TestLocalMqttSettings:
    def test_defaults_no_env_needed(self):
        result = settings.load_local_mqtt_settings()
        assert result.broker == "127.0.0.1"
        assert result.port == 1883
        assert result.topic == "shield/trafo/#"
        assert result.user is None
        assert result.password is None

    def test_overrides_from_env(self, monkeypatch):
        monkeypatch.setenv("MQTT_BROKER", "10.0.0.5")
        monkeypatch.setenv("MQTT_PORT", "1884")
        monkeypatch.setenv("MQTT_LOCAL_USER", "shield")
        monkeypatch.setenv("MQTT_LOCAL_PASS", "pw")
        result = settings.load_local_mqtt_settings()
        assert result.broker == "10.0.0.5"
        assert result.port == 1884
        assert result.user == "shield"
        assert result.password == "pw"


class TestAmbientLocationSettings:
    def test_default_matches_gandul_weather_point(self):
        """HARUS sama persis dengan GANDUL_WEATHER_POINT di index.html --
        lihat catatan sinkronisasi di settings.py/.env.example."""
        result = settings.load_ambient_location_settings()
        assert result.latitude == -6.346667
        assert result.longitude == 106.7875

    def test_override_from_env(self, monkeypatch):
        monkeypatch.setenv("GANDUL_LATITUDE", "-6.5")
        monkeypatch.setenv("GANDUL_LONGITUDE", "107.0")
        result = settings.load_ambient_location_settings()
        assert result.latitude == -6.5
        assert result.longitude == 107.0


class TestNetworkSettings:
    def test_defaults_are_none(self):
        result = settings.load_network_settings()
        assert result.https_proxy is None
        assert result.ca_bundle_path is None

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OUTBOUND_HTTPS_PROXY", "http://proxy:8080")
        monkeypatch.setenv("OUTBOUND_CA_BUNDLE", "/etc/ssl/corp-ca.pem")
        result = settings.load_network_settings()
        assert result.https_proxy == "http://proxy:8080"
        assert result.ca_bundle_path == "/etc/ssl/corp-ca.pem"


class TestWorkerSettings:
    def test_defaults(self):
        result = settings.load_worker_settings()
        assert result.ambient_interval_seconds == 600
        assert result.lntai_interval_seconds == 60
        assert result.write_interval_seconds == 60

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("AMBIENT_WORKER_INTERVAL_SECONDS", "120")
        monkeypatch.setenv("LNTAI_WORKER_INTERVAL_SECONDS", "30")
        monkeypatch.setenv("WRITE_INTERVAL_SECONDS", "15")
        result = settings.load_worker_settings()
        assert result.ambient_interval_seconds == 120
        assert result.lntai_interval_seconds == 30
        assert result.write_interval_seconds == 15
