#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/backfill_lntai.py — SHIELD Tahap 1, T4.

Hitung ulang LNTAI v2 untuk rentang waktu historis SATU trafo, dari data
trafo_temperature + trafo_load yang sudah ada di database -- dipakai sekali
setelah lntai_worker.py (T4) pertama kali di-deploy, supaya histori lama
(sebelum worker berjalan) juga punya hasil LNTAI server-side, bukan hanya
data baru ke depan.

TIDAK menyentuh worker_state (checkpoint milik lntai_worker.py) -- backfill
dan worker berjalan independen; idempotent lewat PK (trafo_id, ts,
model_version) di lntai_results (ON CONFLICT DO NOTHING, lihat migrasi 004),
jadi backfill yang sama bisa dijalankan ulang tanpa duplikasi atau menimpa
hasil yang sudah ada.

Pemakaian:
    python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30
    python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30 --dry-run

--dry-run: jalankan seluruh logika (pairing, data_quality, ambient_at(),
lntai_step()) dan cetak ringkasan (jumlah baris, jumlah DATA_INVALID, dll.)
TANPA menulis apa pun ke lntai_results -- aman dipakai untuk mengecek hasil
sebelum benar-benar menulis ke database.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lntai_calculator import LNTAICalculator  # noqa: E402

from lntai_worker import (  # noqa: E402
    _AMBIENT_DEFAULT,
    _AMBIENT_MAX_AGE_MINUTES,
    _MODEL_VERSION,
    _PAIRING_TOLERANCE_MINUTES,
    build_sample,
    combined_quality_flags,
    insert_result,
    should_compute,
)
import ambient  # noqa: E402
from settings import load_database_settings  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill-lntai")


def _fetch_temperature_range(cur, trafo: str, ts_from: datetime, ts_to: datetime) -> list[dict]:
    cur.execute(
        """
        SELECT "timestamp", r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah, quality_flags
        FROM trafo_temperature
        WHERE trafo_id = %s AND "timestamp" >= %s AND "timestamp" <= %s
        ORDER BY "timestamp" ASC
        """,
        (trafo, ts_from, ts_to),
    )
    return cur.fetchall()


def _find_paired_load_row(cur, trafo: str, ts: datetime, tolerance_minutes: float) -> dict | None:
    from datetime import timedelta

    delta = timedelta(minutes=tolerance_minutes)
    cur.execute(
        """
        SELECT ia, ib, ic, quality_flags
        FROM trafo_load
        WHERE trafo_id = %s AND "timestamp" BETWEEN %s AND %s
        ORDER BY ABS(EXTRACT(EPOCH FROM ("timestamp" - %s)))
        LIMIT 1
        """,
        (trafo, ts - delta, ts + delta, ts),
    )
    return cur.fetchone()


def backfill(cur, trafo: str, ts_from: datetime, ts_to: datetime, dry_run: bool) -> dict:
    calculator = LNTAICalculator()
    rows = _fetch_temperature_range(cur, trafo, ts_from, ts_to)

    stats = {"total": len(rows), "computed": 0, "data_invalid": 0, "not_paired": 0}

    for row in rows:
        ts = row["timestamp"]
        load_row = _find_paired_load_row(cur, trafo, ts, _PAIRING_TOLERANCE_MINUTES)
        paired = load_row is not None
        if not paired:
            stats["not_paired"] += 1
        quality_flags = combined_quality_flags(row, load_row, paired)

        if should_compute(quality_flags, paired):
            amb_value, amb_source = ambient.ambient_at(
                cur, ts, int(_AMBIENT_MAX_AGE_MINUTES), fallback_default_c=_AMBIENT_DEFAULT
            )
            sample = build_sample(row, load_row, amb_value, amb_source, ts)
            result = calculator.update(trafo, sample)
            stats["computed"] += 1
        else:
            reason = "PAIRING_GAP: tidak ada baris beban dalam toleransi waktu" if not paired \
                else "Diblokir data_quality (MISSING/OUT_OF_RANGE/STUCK) -- lihat quality_flags"
            result = {
                "lntai": None, "level": "DATA_INVALID", "error": reason,
                "S_Temp": None, "S_DeltaT": None, "S_RES": None, "hottest_phase": None,
                "t_contact": None, "t_ambient": None, "ambient_source": None,
                "delta_T_real": None, "delta_T_pred": None, "residual": None, "z": None,
                "i_eff_norm": None, "delta_phase": None, "dAB_max": None, "per_phase": None,
            }
            stats["data_invalid"] += 1

        if not dry_run:
            insert_result(cur, trafo, ts, _MODEL_VERSION, result, quality_flags, paired)

    return stats


def _parse_date(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trafo", required=True, choices=["2", "3", "4"])
    parser.add_argument("--from", dest="from_", required=True, help="ISO8601, mis. 2026-09-01")
    parser.add_argument("--to", required=True, help="ISO8601, mis. 2026-09-30")
    parser.add_argument("--dry-run", action="store_true", help="Jangan menulis ke lntai_results, hanya cetak ringkasan")
    args = parser.parse_args()

    ts_from = _parse_date(args.from_)
    ts_to = _parse_date(args.to)

    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        stats = backfill(cur, args.trafo, ts_from, ts_to, args.dry_run)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()

    log.info("=" * 60)
    log.info("Backfill trafo-%s %s..%s (%s)", args.trafo, ts_from.isoformat(), ts_to.isoformat(),
              "DRY-RUN, tidak ditulis" if args.dry_run else "DITULIS ke lntai_results")
    log.info("Total baris suhu: %d", stats["total"])
    log.info("Dihitung (LNTAI valid): %d", stats["computed"])
    log.info("DATA_INVALID (diblokir quality/pairing): %d", stats["data_invalid"])
    log.info("Tidak ada pasangan beban: %d", stats["not_paired"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
