#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MQTT Subscriber - Shield Data
Receives data from the local MQTT broker and stores it in TimescaleDB.

Timezone note: earlier versions of this script inserted
datetime.now().isoformat() (a naive local WIB timestamp, no UTC
offset) into a TIMESTAMPTZ column. Postgres's session timezone here
is Etc/UTC, so a naive "18:28:37" string got stored AS 18:28:37 UTC —
7 hours ahead of the true UTC-equivalent time, which showed up on the
dashboard as data appearing to arrive around midnight. Fixed by always
inserting a timezone-AWARE datetime (UTC), ignoring any timestamp the
payload itself may carry, since this subscriber writes essentially at
the moment the message arrives — the "timestamp" is this process's own
wall clock, not a value that needs interpreting from the payload.

T2 (Tahap 1, Oktober 2026) — perubahan dari versi sebelumnya:
  - Field yang hilang/null/tidak numerik dari payload disimpan sebagai
    SQL NULL, BUKAN 0. Nilai 0 yang memang dikirim device tetap disimpan
    apa adanya (0.0 adalah bacaan yang sah, bukan "tidak ada data").
    temp_max/temp_avg dihitung hanya dari nilai suhu yang non-NULL.
  - Pindah dari psycopg2 (buka koneksi baru tiap pesan) ke psycopg3 +
    connection pool (pola yang sama dengan api.py/mqtt_ingester.py).
  - print() diganti logging; tambah reconnect_delay_set + on_disconnect
    supaya status putus-sambung MQTT tercatat, bukan diam-diam hilang.
  - Kredensial/host dibaca dari settings.py (lihat T1), tidak ada lagi
    hardcode di file ini.

T5 (Tahap 1, Oktober 2026) — kolom quality_flags kini diisi dengan aturan
PER-SAMPEL (MISSING, ZERO_READING, OUT_OF_RANGE) lewat data_quality.py.
Aturan berbasis jendela (STUCK, SPIKE, PAIRING_GAP, TIME_GAP) butuh
riwayat/pasangan beban-suhu yang tidak tersedia di sini (satu pesan MQTT
pada satu waktu) -- itu dihitung belakangan oleh lntai_worker.py (T4) dan
di-OR-kan ke quality_flags baris yang sama.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import paho.mqtt.client as mqtt
from psycopg_pool import ConnectionPool

import data_quality as dq
from settings import load_database_settings, load_local_mqtt_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mqtt-subscriber")

# =====================================================
# CONFIGURATION
# =====================================================
_db_settings = load_database_settings()
_mqtt_settings = load_local_mqtt_settings()

DATABASE_URL = _db_settings.url

MQTT_BROKER = _mqtt_settings.broker
MQTT_PORT = _mqtt_settings.port
MQTT_TOPIC = _mqtt_settings.topic
MQTT_LOCAL_USER = _mqtt_settings.user
MQTT_LOCAL_PASS = _mqtt_settings.password
MQTT_CLIENT_ID = "shield_subscriber"

_pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5, open=True)


# =====================================================
# PARSING — NULL untuk hilang/tidak valid, bukan 0
# =====================================================
def _to_float_or_none(data: dict[str, Any], key: str) -> Optional[float]:
    """Kembalikan None kalau field `key` hilang dari payload, bernilai null,
    atau tidak bisa dikonversi ke angka -- BUKAN 0. Nilai 0 yang memang ada
    di payload (mis. ia=0 saat CB terbuka) tetap dikembalikan sebagai 0.0,
    karena itu bacaan yang sah, bukan data yang hilang."""
    value = data.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =====================================================
# SAVE TO DATABASE
# =====================================================
def save_to_database(data: dict[str, Any]) -> bool:
    try:
        trafo_id = str(data.get('trafo_id', '2'))
        # Always use this process's own current time, timezone-aware (UTC),
        # so psycopg binds it to the TIMESTAMPTZ column unambiguously.
        # See the timezone note in the module docstring for why the
        # payload's own "timestamp" field is intentionally not used here.
        timestamp = datetime.now(timezone.utc)

        ia = _to_float_or_none(data, 'ia')
        ib = _to_float_or_none(data, 'ib')
        ic = _to_float_or_none(data, 'ic')
        _currents = [v for v in (ia, ib, ic) if v is not None]
        current_max = max(_currents) if _currents else None

        kw_total = _to_float_or_none(data, 'kw_total')
        kva_total = _to_float_or_none(data, 'kva_total')
        kvar_total = _to_float_or_none(data, 'kvar_total')
        pf_total = _to_float_or_none(data, 'pf_total')
        freq = _to_float_or_none(data, 'freq')

        # Phase-to-neutral and phase-to-phase voltage.
        van = _to_float_or_none(data, 'van')
        vbn = _to_float_or_none(data, 'vbn')
        vcn = _to_float_or_none(data, 'vcn')
        vab = _to_float_or_none(data, 'vab')
        vbc = _to_float_or_none(data, 'vbc')
        vca = _to_float_or_none(data, 'vca')

        r_atas = _to_float_or_none(data, 'r_atas')
        r_bawah = _to_float_or_none(data, 'r_bawah')
        s_atas = _to_float_or_none(data, 's_atas')
        s_bawah = _to_float_or_none(data, 's_bawah')
        t_atas = _to_float_or_none(data, 't_atas')
        t_bawah = _to_float_or_none(data, 't_bawah')

        all_temps = [r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah]
        valid_temps = [t for t in all_temps if t is not None]
        temp_max = max(valid_temps) if valid_temps else None
        temp_avg = (sum(valid_temps) / len(valid_temps)) if valid_temps else None

        # T5: flag kualitas per-sampel (MISSING/ZERO_READING/OUT_OF_RANGE).
        # Aturan berbasis jendela (STUCK/SPIKE/PAIRING_GAP/TIME_GAP) tidak
        # bisa dihitung di sini -- lihat docstring modul.
        load_flags = dq.combine_row_flags([
            dq.classify_current_value(ia),
            dq.classify_current_value(ib),
            dq.classify_current_value(ic),
        ])
        load_quality_flags = dq.flags_to_bitmask(load_flags)

        temp_flags = dq.combine_row_flags([
            dq.classify_temperature_value(r_atas),
            dq.classify_temperature_value(r_bawah),
            dq.classify_temperature_value(s_atas),
            dq.classify_temperature_value(s_bawah),
            dq.classify_temperature_value(t_atas),
            dq.classify_temperature_value(t_bawah),
        ])
        temp_quality_flags = dq.flags_to_bitmask(temp_flags)

        with _pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trafo_load (
                    timestamp, trafo_id, ia, ib, ic, current_max,
                    kw_total, kva_total, kvar_total, pf_total, freq,
                    van, vbn, vcn, vab, vbc, vca, quality_flags
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (timestamp, trafo_id, ia, ib, ic, current_max,
                 kw_total, kva_total, kvar_total, pf_total, freq,
                 van, vbn, vcn, vab, vbc, vca, load_quality_flags),
            )
            load_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO trafo_temperature (
                    timestamp, trafo_id,
                    r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah,
                    temp_max, temp_avg, quality_flags
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (timestamp, trafo_id, r_atas, r_bawah, s_atas, s_bawah,
                 t_atas, t_bawah, temp_max, temp_avg, temp_quality_flags),
            )
            temp_id = cur.fetchone()[0]

        log.info("Trafo-%s saved (load_id=%s, temp_id=%s)", trafo_id, load_id, temp_id)
        return True

    except Exception:
        log.exception("Gagal menyimpan data ke database")
        return False


# =====================================================
# MQTT CALLBACKS
# =====================================================
def on_connect(client: mqtt.Client, userdata, flags, rc):
    if rc == 0:
        log.info("Connected to %s:%s", MQTT_BROKER, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        log.info("Subscribed to %s", MQTT_TOPIC)
    else:
        log.error("Connection failed (rc=%s)", rc)


def on_disconnect(client: mqtt.Client, userdata, rc):
    if rc != 0:
        log.warning("Unexpected disconnect (rc=%s) -- paho akan mencoba reconnect otomatis", rc)
    else:
        log.info("Disconnected")


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    try:
        payload = json.loads(msg.payload.decode())
        log.debug("Pesan diterima di topik %s", msg.topic)
        save_to_database(payload)
    except Exception:
        log.exception("Gagal memproses pesan MQTT dari topik %s", msg.topic)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MQTT Subscriber - Shield Data")
    log.info("Broker: %s:%s", MQTT_BROKER, MQTT_PORT)
    log.info("Topic:  %s", MQTT_TOPIC)
    log.info("DB:     %s", DATABASE_URL.rsplit('/', 1)[-1])
    log.info("=" * 60)

    client = mqtt.Client(MQTT_CLIENT_ID)
    if MQTT_LOCAL_USER and MQTT_LOCAL_PASS:
        # Broker lokal saat ini belum pakai autentikasi (lihat audit Fase 0) --
        # disiapkan di sini supaya tinggal jalan begitu ACL/autentikasi
        # ditambahkan (rekomendasi keamanan T1), tanpa perlu ubah kode lagi.
        client.username_pw_set(MQTT_LOCAL_USER, MQTT_LOCAL_PASS)
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception:
        log.exception("Failed to connect")
        sys.exit(1)

    try:
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        log.info("Stopped")
        client.disconnect()
        sys.exit(0)
