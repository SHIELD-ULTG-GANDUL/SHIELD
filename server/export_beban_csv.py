#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-time export: dump ALL historical load (beban) documents from Firestore
(beban_realtime/Trafo-{2,3,4}/logs) into plain CSV files, one row per
document, so the full history can be downloaded/analyzed outside Firestore.

This does NOT touch TimescaleDB and does NOT filter by date — unlike
migrate_firestore_to_timescale.py, it dumps every document, including ones
already migrated or already live in TimescaleDB. Nothing in Firestore is
modified or deleted.

Run on the server (has firebase_admin already installed):
    /home/shield_gandul/trafo_early_warning/venv311/bin/python \
        /home/shield_gandul/SHIELD/server/export_beban_csv.py [trafo] [output_dir]

Optional CLI args:
    trafo:      "2", "3", "4", or omitted for all three
    output_dir: folder to write CSVs into (default: ./beban_csv_export)

Output:
    <output_dir>/beban_trafo_2.csv
    <output_dir>/beban_trafo_3.csv
    <output_dir>/beban_trafo_4.csv

Firestore's free-tier plan caps reads at 50,000 documents/day, and each
trafo's history can be tens of thousands of documents — one run may not
finish everything in a single day. Re-running this script is safe and
resumes automatically:
  - Progress is checkpointed as a cursor (the last Firestore document
    processed) after every CHUNK_SIZE documents, written to
    .csv_export_markers/beban_<trafo>.cursor. A run resumes from that
    cursor with Firestore's start_after(), instead of re-reading from the
    start of the collection.
  - New rows are appended to the CSV (the header is only written once,
    when the file doesn't exist yet).
  - A trafo is marked fully done (.done marker) only once a stream reaches
    the actual end of the collection with no error.
  - A quota error (or any error) simply stops that trafo for this run;
    already-read rows are already written to the CSV, and the next run
    resumes from the checkpoint.

Example:
    export_beban_csv.py 4                  # only trafo 4, all history
    export_beban_csv.py "" /tmp/beban_out  # all trafos, custom output dir
"""

from __future__ import annotations

import csv
import os
import sys

import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT_KEY = "/home/shield_gandul/trafo_early_warning/serviceAccountKey.json"

TRAFOS = ["2", "3", "4"]
CHUNK_SIZE = 2000  # documents per CSV flush + checkpoint

FIELDS = [
    "doc_id", "timestamp", "trafo",
    "ia", "ib", "ic",
    "van", "vbn", "vcn", "vab", "vbc", "vca",
    "kw_total", "kva_total", "kvar_total", "pf_total", "freq",
]

MARKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".csv_export_markers")


def _done_path(trafo: str) -> str:
    return os.path.join(MARKER_DIR, f"beban_{trafo}.done")


def _cursor_path(trafo: str) -> str:
    return os.path.join(MARKER_DIR, f"beban_{trafo}.cursor")


def _is_done(trafo: str) -> bool:
    return os.path.exists(_done_path(trafo))


def _mark_done(trafo: str) -> None:
    os.makedirs(MARKER_DIR, exist_ok=True)
    with open(_done_path(trafo), "w") as f:
        f.write("done")


def _read_cursor(trafo: str) -> str | None:
    path = _cursor_path(trafo)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip() or None


def _write_cursor(trafo: str, doc_id: str) -> None:
    os.makedirs(MARKER_DIR, exist_ok=True)
    with open(_cursor_path(trafo), "w") as f:
        f.write(doc_id)


def _stream_from_cursor(collection, trafo: str):
    """Yield (doc_id, data) ordered by document id (chronological — ids are
    zero-padded "YYYYMMDD_HHMMSS..." strings), resuming after the last
    checkpointed document if one exists."""
    query = collection.order_by("__name__")
    cursor = _read_cursor(trafo)
    if cursor:
        snap = collection.document(cursor).get()
        if snap.exists:
            query = query.start_after(snap)
    for doc in query.stream():
        yield doc.id, doc.to_dict()


def _row_from_doc(doc_id: str, d: dict, trafo: str) -> dict:
    get = lambda k: d.get(k, "")
    return {
        "doc_id": doc_id,
        "timestamp": get("timestamp"),
        "trafo": trafo,
        "ia": get("ia"), "ib": get("ib"), "ic": get("ic"),
        "van": get("van"), "vbn": get("vbn"), "vcn": get("vcn"),
        "vab": get("vab"), "vbc": get("vbc"), "vca": get("vca"),
        "kw_total": get("kw_total"), "kva_total": get("kva_total"),
        "kvar_total": get("kvar_total"), "pf_total": get("pf_total"),
        "freq": get("freq"),
    }


def export_trafo(db, trafo: str, out_path: str) -> int:
    if _is_done(trafo):
        print(f"  trafo {trafo}: sudah selesai (marker ada), dilewati")
        return 0

    doc_id_by_trafo = {"2": "Trafo-2", "3": "Trafo-3", "4": "Trafo-4"}
    logs_col = db.collection("beban_realtime").document(doc_id_by_trafo[trafo]).collection("logs")

    file_exists = os.path.exists(out_path)
    trafo_total = 0
    completed = False
    batch = []
    last_doc_id = None

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()

        try:
            for doc_id, d in _stream_from_cursor(logs_col, trafo):
                last_doc_id = doc_id
                batch.append(_row_from_doc(doc_id, d, trafo))

                if len(batch) >= CHUNK_SIZE:
                    writer.writerows(batch)
                    f.flush()
                    _write_cursor(trafo, last_doc_id)
                    trafo_total += len(batch)
                    batch = []
            completed = True
        except Exception as e:
            print(f"  trafo {trafo}: berhenti karena error ({e}); progres sampai sini sudah tersimpan di CSV")

        if batch:
            writer.writerows(batch)
            f.flush()
            if last_doc_id:
                _write_cursor(trafo, last_doc_id)
            trafo_total += len(batch)

    print(f"  trafo {trafo}: {trafo_total} baris ditulis ke {out_path} pada run ini")
    if completed:
        _mark_done(trafo)
    return trafo_total


def main() -> None:
    trafo_arg = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
    out_dir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "beban_csv_export"
    )
    trafos = [trafo_arg] if trafo_arg else TRAFOS
    for t in trafos:
        if t not in TRAFOS:
            print(f"trafo tidak dikenal: {t!r} (pakai: 2, 3, atau 4)")
            sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    total = 0
    all_done = True
    for trafo in trafos:
        print(f"Export beban Trafo {trafo}...")
        out_path = os.path.join(out_dir, f"beban_trafo_{trafo}.csv")
        total += export_trafo(db, trafo, out_path)
        all_done = all_done and _is_done(trafo)

    print(f"Total baris ditulis pada run ini: {total}")
    if trafo_arg is None and all_done:
        print(f"SEMUA data beban (trafo 2/3/4) sudah 100% ter-export ke CSV di: {out_dir}")
    else:
        print("Belum selesai semua — jalankan ulang perintah yang sama untuk melanjutkan dari checkpoint.")


if __name__ == "__main__":
    sys.exit(main())
