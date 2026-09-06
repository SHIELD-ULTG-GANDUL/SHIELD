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
unique (trafo_id, timestamp) index makes re-running this script safe.

Run on the server (has firebase_admin + psycopg2 already installed):
    /home/shield_gandul/trafo_early_warning/venv311/bin/python \
        /home/shield_gandul/SHIELD/server/migrate_firestore_to_timescale.py [source] [trafo]

Firestore's free-tier plan caps reads at 50,000 documents/day, and each
trafo's history is tens of thousands of documents — one run will not finish
everything in a single day. To make repeated runs (e.g. from a daily cron
job) converge to 100% migrated without wasting quota:
  - Progress is checkpointed as a cursor (the last Firestore document
    processed) after every CHUNK_SIZE documents, written to
    .migration_markers/<source>_<trafo>.cursor. A run resumes from that
    cursor with Firestore's start_after(), instead of re-reading from the
    start of the collection.
  - A (source, trafo) pair is marked fully done (.done marker) only once a
    stream reaches the actual end of the collection with no error.
  - A quota error (or any error) simply stops that pair for this run; already
    -read documents are already committed, and the next run resumes from the
    checkpoint.

Optional CLI args scope a run to just one piece:
    source: "temp", "beban", or omitted for both
    trafo:  "2", "3", "4", or omitted for all three
Examples:
    migrate_firestore_to_timescale.py temp 4      # only trafo 4 temperature
    migrate_firestore_to_timescale.py beban        # beban, all trafos
"""

from __future__ import annotations

import os
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
CHUNK_SIZE = 2000  # documents per commit + checkpoint

# Cutoff: only migrate Firestore rows strictly older than this instant.
# Chosen just before the earliest live MQTT-ingested row (2026-09-03 07:18 WIB),
# so this backfill never overlaps with real-time data already in Postgres.
CUTOFF_WIB = datetime(2026, 9, 3, 0, 0, 0, tzinfo=JAKARTA)

# Firestore's free-tier quota (50k reads/day) means a full backfill spans
# several days. Marker/cursor files record progress so a later cron-driven
# run resumes instead of re-reading (and re-spending quota on) history
# already migrated.
MARKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".migration_markers")


def _done_path(source: str, trafo: str) -> str:
    return os.path.join(MARKER_DIR, f"{source}_{trafo}.done")


def _cursor_path(source: str, trafo: str) -> str:
    return os.path.join(MARKER_DIR, f"{source}_{trafo}.cursor")


def _is_done(source: str, trafo: str) -> bool:
    return os.path.exists(_done_path(source, trafo))


def _mark_done(source: str, trafo: str) -> None:
    os.makedirs(MARKER_DIR, exist_ok=True)
    with open(_done_path(source, trafo), "w") as f:
        f.write(datetime.now(JAKARTA).isoformat())
    cursor = _cursor_path(source, trafo)
    if os.path.exists(cursor):
        os.remove(cursor)


def _read_cursor(source: str, trafo: str) -> str | None:
    path = _cursor_path(source, trafo)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip() or None


def _write_cursor(source: str, trafo: str, doc_id: str) -> None:
    os.makedirs(MARKER_DIR, exist_ok=True)
    with open(_cursor_path(source, trafo), "w") as f:
        f.write(doc_id)


def parse_wib(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return naive.replace(tzinfo=JAKARTA)


def _stream_from_cursor(collection, source: str, trafo: str):
    """Yield (doc_id, data) ordered by document id (chronological — ids are
    zero-padded "YYYYMMDD_HHMMSS..." strings), resuming after the last
    checkpointed document if one exists. Resuming costs one extra read (to
    fetch the checkpointed document's snapshot, the form start_after()
    expects), which is negligible next to the size of these collections."""
    query = collection.order_by("__name__")
    cursor = _read_cursor(source, trafo)
    if cursor:
        snap = collection.document(cursor).get()
        if snap.exists:
            query = query.start_after(snap)
    for doc in query.stream():
        yield doc.id, doc.to_dict()


def migrate_temperature(db, conn, trafos: list[str]) -> int:
    inserted = 0
    with conn.cursor() as cur:
        for trafo in trafos:
            if _is_done("temp", trafo):
                print(f"  trafo {trafo}: sudah selesai (marker ada), dilewati")
                continue
            col = (
                db.collection("devices")
                .document("acrel_atp007")
                .collection(f"trafo_{trafo}_temperature_logs")
            )
            trafo_total = 0
            completed = False
            batch = []
            last_doc_id = None
            try:
                for doc_id, d in _stream_from_cursor(col, "temp", trafo):
                    last_doc_id = doc_id
                    ts = parse_wib(d.get("timestamp"))
                    if ts is not None and ts < CUTOFF_WIB:
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

                    if len(batch) >= CHUNK_SIZE:
                        _insert_temp_batch(cur, batch)
                        conn.commit()
                        _write_cursor("temp", trafo, last_doc_id)
                        trafo_total += len(batch)
                        batch = []
                completed = True
            except Exception as e:
                print(f"  trafo {trafo}: berhenti karena error ({e}); progres sampai sini sudah tersimpan")

            if batch:
                _insert_temp_batch(cur, batch)
                conn.commit()
                if last_doc_id:
                    _write_cursor("temp", trafo, last_doc_id)
                trafo_total += len(batch)

            print(f"  trafo {trafo}: {trafo_total} baris suhu di-insert pada run ini")
            inserted += trafo_total
            if completed:
                _mark_done("temp", trafo)
    return inserted


def _insert_temp_batch(cur, batch) -> None:
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


def migrate_beban(db, conn, trafos: list[str]) -> int:
    inserted = 0
    doc_id_by_trafo = {"2": "Trafo-2", "3": "Trafo-3", "4": "Trafo-4"}
    with conn.cursor() as cur:
        for trafo in trafos:
            if _is_done("beban", trafo):
                print(f"  trafo {trafo}: sudah selesai (marker ada), dilewati")
                continue
            doc_id = doc_id_by_trafo[trafo]
            logs_col = db.collection("beban_realtime").document(doc_id).collection("logs")
            trafo_total = 0
            completed = False
            batch = []
            last_doc_id = None
            try:
                for fdoc_id, d in _stream_from_cursor(logs_col, "beban", trafo):
                    last_doc_id = fdoc_id
                    ts = parse_wib(d.get("timestamp"))
                    if ts is not None and ts < CUTOFF_WIB:
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

                    if len(batch) >= CHUNK_SIZE:
                        _insert_beban_batch(cur, batch)
                        conn.commit()
                        _write_cursor("beban", trafo, last_doc_id)
                        trafo_total += len(batch)
                        batch = []
                completed = True
            except Exception as e:
                print(f"  trafo {trafo}: berhenti karena error ({e}); progres sampai sini sudah tersimpan")

            if batch:
                _insert_beban_batch(cur, batch)
                conn.commit()
                if last_doc_id:
                    _write_cursor("beban", trafo, last_doc_id)
                trafo_total += len(batch)

            print(f"  trafo {trafo} ({doc_id}): {trafo_total} baris beban di-insert pada run ini")
            inserted += trafo_total
            if completed:
                _mark_done("beban", trafo)
    return inserted


def _insert_beban_batch(cur, batch) -> None:
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


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "both"
    trafos = [sys.argv[2]] if len(sys.argv) > 2 else TRAFOS
    if source not in ("temp", "beban", "both"):
        print(f"source tidak dikenal: {source!r} (pakai: temp, beban, atau kosongkan)")
        sys.exit(1)
    for t in trafos:
        if t not in TRAFOS:
            print(f"trafo tidak dikenal: {t!r} (pakai: 2, 3, atau 4)")
            sys.exit(1)

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

        all_done = True

        if source in ("temp", "both"):
            print(f"Migrasi data suhu (temperature_logs) untuk trafo {trafos}...")
            n_temp = migrate_temperature(db, conn, trafos)
            print(f"Total baris suhu di-insert pada run ini: {n_temp}")
            all_done = all_done and all(_is_done("temp", t) for t in TRAFOS)

        if source in ("beban", "both"):
            print(f"Migrasi data beban (beban_realtime/*/logs) untuk trafo {trafos}...")
            n_beban = migrate_beban(db, conn, trafos)
            print(f"Total baris beban di-insert pada run ini: {n_beban}")
            all_done = all_done and all(_is_done("beban", t) for t in TRAFOS)

        if source == "both" and all_done:
            print("SEMUA data Firestore (suhu + beban, trafo 2/3/4) sudah 100% dipindah ke TimescaleDB.")
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
