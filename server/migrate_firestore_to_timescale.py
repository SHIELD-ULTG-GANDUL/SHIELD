#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-time backfill: export historical Firestore documents (the data Shield.html
used before the TimescaleDB migration) into shield_data (trafo_temperature /
trafo_load), so the same history becomes available for ML training and for
the dashboard's Data Logger via /api/temperature-logs and /api/beban-logs.

Sources (read-only, nothing in Firestore is modified or deleted):
    devices/acrel_atp007/trafo_{2,3,4}_temperature_logs   -> trafo_temperature
    beban_realtime/Trafo-{2,3,4}/logs                     -> trafo_load
    (beban_realtime/Cimanggis_Trf-2 is a different site and is skipped)

Firestore's "timestamp" field is a naive string ("YYYY-MM-DD HH:MM:SS") with
no timezone marker. The device that writes it runs on local wall-clock time
(Asia/Jakarta, UTC+7) — the same assumption already used for live MQTT data
after the 7-hour timezone bug fix — so it is localized to Asia/Jakarta here
before being converted to UTC for the TIMESTAMPTZ columns.

Only rows older than the first live (MQTT-ingested) row are inserted, and a
unique (trafo_id, timestamp) index makes re-running this script a no-op
instead of creating duplicates.

Run on the server (has firebase_admin + psycopg2 already installed):
    /home/shield_gandul/trafo_early_warning/venv311/bin/python \
        /home/shield_gandul/SHIELD/server/migrate_firestore_to_timescale.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import firebase_admin
import psycopg2
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT_KEY = "/home/shield_gandul/trafo_early_warning/serviceAccountKey.json"
JAKARTA = ZoneInfo("Asia/Jakarta")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "shield_data",
    "user": "postgres",
    "password": "postgres",
}

TRAFOS = ["2", "3", "4"]
# Cutoff: only migrate Firestore rows strictly older than this instant.
# Chosen just before the earliest live MQTT-ingested row (2026-09-03 07:18 WIB),
# so this backfill never overlaps with real-time data already in Postgres.
CUTOFF_WIB = datetime(2026, 9, 3, 0, 0, 0, tzinfo=JAKARTA)


def parse_wib(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=JAKARTA)


def migrate_temperature(db, conn) -> int:
    inserted = 0
    with conn.cursor() as cur:
        for trafo in TRAFOS:
            col = (
                db.collection("devices")
                .document("acrel_atp007")
                .collection(f"trafo_{trafo}_temperature_logs")
            )
            batch = []
            for doc in col.stream():
                d = doc.to_dict()
                ts = parse_wib(d.get("timestamp"))
                if ts is None or ts >= CUTOFF_WIB:
                    continue
                r_atas = float(d.get("R_Atas", 0) or 0)
                r_bawah = float(d.get("R_Bawah", 0) or 0)
                s_atas = float(d.get("S_Atas", 0) or 0)
                s_bawah = float(d.get("S_Bawah", 0) or 0)
                t_atas = float(d.get("T_Atas", 0) or 0)
                t_bawah = float(d.get("T_Bawah", 0) or 0)
                temps = [t for t in (r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah) if t > 0]
                temp_max = max(temps) if temps else 0
                temp_avg = sum(temps) / len(temps) if temps else 0
                batch.append((ts, trafo, r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah, temp_max, temp_avg))

            for row in batch:
                cur.execute(
                    """
                    INSERT INTO trafo_temperature (
                        timestamp, trafo_id, r_atas, r_bawah, s_atas, s_bawah,
                        t_atas, t_bawah, temp_max, temp_avg
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trafo_id, timestamp) DO NOTHING
                    """,
                    row,
                )
            conn.commit()
            print(f"  trafo {trafo}: {len(batch)} baris suhu diproses dari Firestore")
            inserted += len(batch)
    return inserted


def migrate_beban(db, conn) -> int:
    inserted = 0
    doc_id_by_trafo = {"2": "Trafo-2", "3": "Trafo-3", "4": "Trafo-4"}
    with conn.cursor() as cur:
        for trafo, doc_id in doc_id_by_trafo.items():
            logs_col = db.collection("beban_realtime").document(doc_id).collection("logs")
            batch = []
            for doc in logs_col.stream():
                d = doc.to_dict()
                ts = parse_wib(d.get("timestamp"))
                if ts is None or ts >= CUTOFF_WIB:
                    continue
                ia = float(d.get("ia", 0) or 0)
                ib = float(d.get("ib", 0) or 0)
                ic = float(d.get("ic", 0) or 0)
                current_max = max(ia, ib, ic)
                kw_total = float(d.get("kw_total", 0) or 0)
                kva_total = float(d.get("kva_total", 0) or 0)
                kvar_total = float(d.get("kvar_total", 0) or 0)
                pf_total = float(d.get("pf_total", 0) or 0)
                freq = float(d.get("freq", 0) or 0)
                van = float(d.get("van", 0) or 0)
                vbn = float(d.get("vbn", 0) or 0)
                vcn = float(d.get("vcn", 0) or 0)
                vab = float(d.get("vab", 0) or 0)
                vbc = float(d.get("vbc", 0) or 0)
                vca = float(d.get("vca", 0) or 0)
                batch.append((
                    ts, trafo, ia, ib, ic, current_max, kw_total, kva_total,
                    kvar_total, pf_total, freq, van, vbn, vcn, vab, vbc, vca,
                ))

            for row in batch:
                cur.execute(
                    """
                    INSERT INTO trafo_load (
                        timestamp, trafo_id, ia, ib, ic, current_max,
                        kw_total, kva_total, kvar_total, pf_total, freq,
                        van, vbn, vcn, vab, vbc, vca
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trafo_id, timestamp) DO NOTHING
                    """,
                    row,
                )
            conn.commit()
            print(f"  trafo {trafo} ({doc_id}): {len(batch)} baris beban diproses dari Firestore")
            inserted += len(batch)
    return inserted


def main() -> None:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_trafo_temperature_trafo_ts "
                "ON trafo_temperature (trafo_id, timestamp)"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_trafo_load_trafo_ts "
                "ON trafo_load (trafo_id, timestamp)"
            )
        conn.commit()

        print("Migrasi data suhu (temperature_logs)...")
        n_temp = migrate_temperature(db, conn)
        print(f"Total baris suhu di-insert (setelah dedup): {n_temp}")

        print("Migrasi data beban (beban_realtime/*/logs)...")
        n_beban = migrate_beban(db, conn)
        print(f"Total baris beban di-insert (setelah dedup): {n_beban}")
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
