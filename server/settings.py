"""server/settings.py — satu sumber konfigurasi untuk semua proses di server/.

Semua nilai dibaca dari environment variable, TIDAK ADA nilai rahasia
(password, API key, token broker) yang di-hardcode sebagai default di sini
atau di proses mana pun yang mengimpor modul ini. Kalau env var yang WAJIB
tidak diset, proses gagal start dengan pesan jelas (lihat MissingSettingError)
alih-alih diam-diam memakai kredensial contoh/lama.

Setiap "loader" (load_xxx_settings) dipanggil LAZY oleh skrip yang benar-benar
butuh -- bukan dieksekusi otomatis saat modul ini diimpor. Ini penting: skrip
yang cuma butuh DATABASE_URL (mis. api.py) tidak boleh ikut gagal start hanya
karena HIVEMQ_* belum diisi, walau sama-sama mengimpor settings.py.

Opsional: kalau `python-dotenv` terpasang (lihat server/requirements.txt) dan
file `.env` ada di folder yang sama dengan skrip yang dijalankan, isinya
otomatis dimuat ke environment proses sebelum dibaca oleh loader di bawah.
Kalau python-dotenv TIDAK terpasang, modul ini tetap berfungsi normal selama
env var sudah di-export manual sebelum menjalankan skrip (lihat
server/README.md) -- python-dotenv bukan dependency wajib untuk modul ini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore

    _env_file = Path(__file__).resolve().parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # python-dotenv opsional -- lihat docstring modul di atas


class MissingSettingError(RuntimeError):
    """Environment variable wajib tidak diset atau kosong."""


def _require(name: str, example: str = "") -> str:
    value = os.environ.get(name)
    if not value:
        hint = f" Contoh: {name}={example}" if example else ""
        raise MissingSettingError(
            f"Environment variable {name} wajib diset, tidak boleh kosong.{hint} "
            f"Lihat server/.env.example."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _optional_or_none(name: str) -> Optional[str]:
    value = os.environ.get(name, "")
    return value or None


def _optional_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def _optional_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    return float(raw)


def _optional_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class DatabaseSettings:
    url: str


@dataclass(frozen=True)
class ApiSettings:
    # String kosong = autentikasi API key dimatikan -- ini perilaku LAMA
    # api.py yang sengaja dipertahankan (lihat _check_api_key di sana), bukan
    # kelonggaran baru yang ditambahkan oleh settings.py.
    api_key: str
    allowed_origins: list[str]


@dataclass(frozen=True)
class HiveMqSettings:
    """Kredensial broker MQTT cloud (HiveMQ). Hanya dipakai
    mqtt_ingester.py -- skrip itu DEPRECATED (service dinonaktifkan di
    server, lihat audit Fase 0 Tahap 1), tapi kredensialnya tetap wajib
    diisi eksplisit lewat env var kalau skrip ini pernah dijalankan lagi,
    tidak ada default yang jatuh ke broker produksi asli."""

    broker: str
    port: int
    user: str
    password: str


@dataclass(frozen=True)
class LocalMqttSettings:
    """Broker MQTT -- dipakai mqtt_subscriber.py. Nama "Local" historis dari
    desain awal (broker Mosquitto di server yang sama, tanpa TLS/auth) --
    MQTT_BROKER saat ini (Okt 2026) dialihkan langsung ke HiveMQ Cloud
    (audit Fase 2 menemukan ini tidak sesuai desain awal: HIVEMQ_* seharusnya
    dipakai, bukan MQTT_*), jadi `use_tls` ditambahkan supaya skrip yang SAMA
    bisa dipakai untuk broker lokal (TLS off, default lama) maupun cloud
    (TLS wajib -- HiveMQ Cloud menolak koneksi plaintext). user/password
    jatuh ke MQTT_USER/MQTT_PASS kalau MQTT_LOCAL_USER/MQTT_LOCAL_PASS
    kosong, supaya kredensial HiveMQ yang sudah ada di .env ikut terbaca."""

    broker: str
    port: int
    topic: str
    user: Optional[str]
    password: Optional[str]
    use_tls: bool


@dataclass(frozen=True)
class AmbientLocationSettings:
    """Lokasi GI Gandul untuk pengambilan suhu ambient (Open-Meteo).
    Nilai default SAMA PERSIS dengan GANDUL_WEATHER_POINT di index.html --
    kalau salah satu diubah, yang lain WAJIB ikut diubah (belum ada
    mekanisme sinkronisasi otomatis, lihat catatan di server/README.md)."""

    latitude: float
    longitude: float


@dataclass(frozen=True)
class NetworkSettings:
    """Proxy keluar opsional. Audit Fase 0 menemukan server ini di belakang
    proxy yang meng-intercept TLS (curl ke api.open-meteo.com gagal dengan
    'self-signed certificate in certificate chain'). Kalau tim IT memasang
    root CA korporat ke trust store server (opsi jangka panjang -- lihat
    laporan risiko T3), proxy/CA bundle bisa dikonfigurasi di sini TANPA
    menonaktifkan verifikasi SSL di kode mana pun -- dilarang keras oleh
    Aturan kerja Tahap 1."""

    https_proxy: Optional[str]
    ca_bundle_path: Optional[str]


@dataclass(frozen=True)
class WorkerSettings:
    # TODO(verifikasi): ambient_interval_seconds disalin dari
    # GANDUL_WEATHER_POINT.refreshMs (10 menit) di index.html, belum tentu
    # optimal untuk worker server-side (T3). lntai_interval_seconds (60
    # detik) adalah DEFAULT dari spesifikasi Tahap 1, worker-nya (T4) belum
    # dibuat -- bukan hasil pengukuran/kalibrasi.
    ambient_interval_seconds: int
    lntai_interval_seconds: int
    write_interval_seconds: int


def load_database_settings() -> DatabaseSettings:
    return DatabaseSettings(
        url=_require(
            "DATABASE_URL",
            "postgresql://shield:ganti-password-ini@localhost:5432/shield_data",
        ),
    )


def load_api_settings() -> ApiSettings:
    origins_raw = _optional("ALLOWED_ORIGINS", "https://shield-ultg-gandul.github.io")
    return ApiSettings(
        api_key=_optional("API_KEY", ""),
        allowed_origins=[o.strip() for o in origins_raw.split(",") if o.strip()],
    )


def load_hivemq_settings() -> HiveMqSettings:
    return HiveMqSettings(
        broker=_require("HIVEMQ_BROKER", "xxxxxxxxxxxxxxxx.s1.eu.hivemq.cloud"),
        port=_optional_int("HIVEMQ_PORT", 8884),
        user=_require("HIVEMQ_USER", "ganti-user"),
        password=_require("HIVEMQ_PASS", "ganti-password"),
    )


def load_local_mqtt_settings() -> LocalMqttSettings:
    # use_tls default True kalau broker BUKAN localhost/127.0.0.1 (broker
    # cloud seperti HiveMQ selalu butuh TLS) -- tetap bisa dipaksa lewat
    # MQTT_USE_TLS kalau ada broker non-cloud yang justru butuh TLS, atau
    # broker lokal yang kebetulan dikonfigurasi TLS.
    broker = _optional("MQTT_BROKER", "127.0.0.1")
    is_probably_local = broker in ("127.0.0.1", "localhost", "::1")
    return LocalMqttSettings(
        broker=broker,
        port=_optional_int("MQTT_PORT", 1883),
        topic=_optional("MQTT_TOPIC", "shield/trafo/#"),
        user=_optional_or_none("MQTT_LOCAL_USER") or _optional_or_none("MQTT_USER"),
        password=_optional_or_none("MQTT_LOCAL_PASS") or _optional_or_none("MQTT_PASS"),
        use_tls=_optional_bool("MQTT_USE_TLS", default=not is_probably_local),
    )


def load_ambient_location_settings() -> AmbientLocationSettings:
    return AmbientLocationSettings(
        latitude=_optional_float("GANDUL_LATITUDE", -6.346667),
        longitude=_optional_float("GANDUL_LONGITUDE", 106.7875),
    )


def load_network_settings() -> NetworkSettings:
    return NetworkSettings(
        https_proxy=_optional_or_none("OUTBOUND_HTTPS_PROXY"),
        ca_bundle_path=_optional_or_none("OUTBOUND_CA_BUNDLE"),
    )


def load_worker_settings() -> WorkerSettings:
    return WorkerSettings(
        ambient_interval_seconds=_optional_int("AMBIENT_WORKER_INTERVAL_SECONDS", 600),
        lntai_interval_seconds=_optional_int("LNTAI_WORKER_INTERVAL_SECONDS", 60),
        write_interval_seconds=_optional_int("WRITE_INTERVAL_SECONDS", 60),
    )
