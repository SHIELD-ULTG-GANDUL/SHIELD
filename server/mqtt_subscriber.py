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
"""

import paho.mqtt.client as mqtt
import psycopg2
import psycopg2.extras
import json
import sys
import time
from datetime import datetime, timezone

# =====================================================
# CONFIGURATION
# =====================================================
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'shield_data',
    'user': 'postgres',
    'password': 'postgres'
}

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "shield/trafo/#"
MQTT_CLIENT_ID = "shield_subscriber"

# =====================================================
# SAVE TO DATABASE
# =====================================================
def save_to_database(data):
    try:
        trafo_id = str(data.get('trafo_id', '2'))
        # Always use this process's own current time, timezone-aware (UTC),
        # so psycopg2 binds it to the TIMESTAMPTZ column unambiguously.
        # See the timezone note in the module docstring for why the
        # payload's own "timestamp" field is intentionally not used here.
        timestamp = datetime.now(timezone.utc)

        ia = float(data.get('ia', 0))
        ib = float(data.get('ib', 0))
        ic = float(data.get('ic', 0))
        current_max = max(ia, ib, ic)
        kw_total = float(data.get('kw_total', 0))
        kva_total = float(data.get('kva_total', 0))
        pf_total = float(data.get('pf_total', 0))
        freq = float(data.get('freq', 0))

        r_atas = float(data.get('r_atas', 0))
        r_bawah = float(data.get('r_bawah', 0))
        s_atas = float(data.get('s_atas', 0))
        s_bawah = float(data.get('s_bawah', 0))
        t_atas = float(data.get('t_atas', 0))
        t_bawah = float(data.get('t_bawah', 0))

        all_temps = [r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah]
        valid_temps = [t for t in all_temps if t > 0]
        temp_max = max(valid_temps) if valid_temps else 0
        temp_avg = sum(valid_temps) / len(valid_temps) if valid_temps else 0

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO trafo_load (
                timestamp, trafo_id, ia, ib, ic, current_max,
                kw_total, kva_total, pf_total, freq
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (timestamp, trafo_id, ia, ib, ic, current_max,
              kw_total, kva_total, pf_total, freq))
        load_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO trafo_temperature (
                timestamp, trafo_id,
                r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah,
                temp_max, temp_avg
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (timestamp, trafo_id, r_atas, r_bawah, s_atas, s_bawah,
              t_atas, t_bawah, temp_max, temp_avg))
        temp_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Trafo-{trafo_id} saved (load_id={load_id}, temp_id={temp_id})")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =====================================================
# MQTT CALLBACKS
# =====================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to {MQTT_TOPIC}")
        print("=" * 50)
        print("Waiting for data...")
        print("=" * 50)
    else:
        print(f"❌ Connection failed (rc={rc})")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg.topic}")
        save_to_database(payload)
    except Exception as e:
        print(f"❌ Error: {e}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MQTT Subscriber - Shield Data")
    print("=" * 60)
    print(f"  Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Topic:  {MQTT_TOPIC}")
    print(f"  DB:     {DB_CONFIG['database']}")
    print("=" * 60)

    client = mqtt.Client(MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        sys.exit(1)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped")
        client.disconnect()
        sys.exit(0)
