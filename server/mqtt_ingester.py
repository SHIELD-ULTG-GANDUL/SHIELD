"""
SHIELD — MQTT -> TimescaleDB ingester.

Berlangganan topik suhu yang sama dengan dashboard (Suhu/Trafo/2, /3, /4)
di broker HiveMQ, lalu menulis tiap pembacaan ke TimescaleDB. Berjalan
independen dari browser — data tetap tercatat walau tidak ada yang membuka
dashboard, beda dengan pendekatan lama yang menulis dari dalam browser.

Konfigurasi lewat environment variable (lihat .env.example):
    MQTT_BROKER, MQTT_PORT, MQTT_USER, MQTT_PASS   -- sama seperti di Shield.html
    DATABASE_URL                                    -- postgresql://user:pass@host:5432/shield

Jalankan manual untuk tes:
    pip install -r requirements.txt
    export $(cat .env | xargs)   # atau isi env var manual
    python mqtt_ingester.py

Untuk produksi, jalankan sebagai systemd service — lihat shield-ingester.service.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import ssl
import sys
import time
from typing import Any

import paho.mqtt.client as mqtt
import psycopg
from psycopg_pool import ConnectionPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("shield-ingester")

MQTT_BROKER = os.environ.get("MQTT_BROKER", "d36cf3116e2a4d93bf531ebf8a1ffa38.s1.eu.hivemq.cloud")
# Port native MQTT (8883) terbukti tidak bisa dijangkau dari beberapa jaringan saat
# didiagnosis. Pakai port WebSocket (8884) — jalur yang sama dan sudah terbukti selalu
# berhasil dipakai dashboard (Shield.html) sepanjang pengembangan proyek ini.
MQTT_PORT = int(os.environ.get("MQTT_PORT", "8884"))
MQTT_USER = os.environ.get("MQTT_USER", "webshield")
MQTT_PASS = os.environ.get("MQTT_PASS", "webShield123")
MQTT_TOPICS = ["Suhu/Trafo/2", "Suhu/Trafo/3", "Suhu/Trafo/4"]

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    log.error("DATABASE_URL belum diset. Contoh: postgresql://shield:password@localhost:5432/shield")
    sys.exit(1)

# Throttle sama seperti dashboard lama: maksimal 1 baris tersimpan per trafo per menit,
# supaya volume data tidak meledak kalau device publish tiap detik.
WRITE_INTERVAL_SECONDS = int(os.environ.get("WRITE_INTERVAL_SECONDS", "60"))
_last_write_at: dict[str, float] = {}

_pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5, open=True)

FIELD_KEYS = ["R_Atas", "R_Bawah", "S_Atas", "S_Bawah", "T_Atas", "T_Bawah"]


def _trafo_from_topic(topic: str) -> str | None:
    m = re.search(r"(\d)$", topic)
    return m.group(1) if m and m.group(1) in ("2", "3", "4") else None


def _insert_row(trafo: str, payload: dict[str, Any]) -> None:
    values = {k: float(payload.get(k) or 0) for k in FIELD_KEYS}
    device_ts = payload.get("timestamp") or payload.get("ts")

    with _pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO temperature_logs
                (trafo, r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah, device_ts, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'mqtt_ingester')
            """,
            (
                int(trafo),
                values["R_Atas"], values["R_Bawah"],
                values["S_Atas"], values["S_Bawah"],
                values["T_Atas"], values["T_Bawah"],
                str(device_ts) if device_ts is not None else None,
            ),
        )


def on_connect(client: mqtt.Client, userdata, flags, rc, properties=None):
    if rc == 0:
        log.info("Terhubung ke broker MQTT %s:%s", MQTT_BROKER, MQTT_PORT)
        for topic in MQTT_TOPICS:
            client.subscribe(topic)
            log.info("Subscribe: %s", topic)
    else:
        log.error("Gagal konek ke broker MQTT, rc=%s", rc)


def on_disconnect(client: mqtt.Client, userdata, rc, *args):
    log.warning("Terputus dari broker MQTT (rc=%s) — paho akan mencoba reconnect otomatis.", rc)


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    trafo = _trafo_from_topic(msg.topic)
    if not trafo:
        return
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.warning("Payload bukan JSON valid dari topik %s: %r", msg.topic, msg.payload[:200])
        return

    if not any(float(payload.get(k) or 0) != 0 for k in FIELD_KEYS):
        return  # paket kosong/semua nol, lewati

    now = time.time()
    if now - _last_write_at.get(trafo, 0) < WRITE_INTERVAL_SECONDS:
        return
    _last_write_at[trafo] = now

    try:
        _insert_row(trafo, payload)
        log.info("Tersimpan: Trafo %s @ %s", trafo, payload.get("timestamp") or payload.get("ts") or "-")
    except Exception:
        log.exception("Gagal menyimpan baris ke database untuk Trafo %s", trafo)
        _last_write_at[trafo] = 0  # izinkan retry pada pesan berikutnya


def main() -> None:
    client = mqtt.Client(
        client_id="shield_ingester_" + str(int(time.time())),
        transport="websockets",
        protocol=mqtt.MQTTv311,
    )
    client.ws_set_options(path="/mqtt")
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _shutdown(signum, frame):
        log.info("Menerima signal berhenti, menutup koneksi...")
        client.disconnect()
        _pool.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
