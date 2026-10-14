#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/backfill_lntai.py — SHIELD Tahap 1 (T4) + Stage 2 Part A (v3).

Hitung ulang LNTAI untuk rentang waktu historis SATU trafo, dari data
trafo_temperature + trafo_load yang sudah ada di database -- dipakai sekali
setelah lntai_worker.py pertama kali di-deploy (atau setelah LNTAI v3
ditambahkan), supaya histori lama juga punya hasil LNTAI server-side, bukan
hanya data baru ke depan.

TIDAK menyentuh worker_state (checkpoint milik lntai_worker.py) -- backfill
dan worker berjalan independen; idempotent lewat PK (trafo_id, ts,
model_version) di lntai_results (ON CONFLICT DO NOTHING, lihat migrasi 004),
jadi backfill yang sama bisa dijalankan ulang tanpa duplikasi atau menimpa
hasil yang sudah ada.

Pemakaian:
    python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30
    python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30 --dry-run
    python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30 --model-version lntai-v3 --dry-run

--model-version (default lntai-v2): model yang DITULIS ke lntai_results.
v2 SELALU dihitung terlebih dahulu (v3 butuh v2_result sebagai input,
lihat lntai_calculator.lntai_v3_step()), tapi kalau --model-version
lntai-v3 dipilih, HANYA baris v3 yang di-INSERT -- baris v2 untuk rentang
yang sama TIDAK disentuh sama sekali oleh pemanggilan ini (jalankan
backfill terpisah dengan --model-version lntai-v2, default, kalau baris
v2 historis itu juga belum ada).

--dry-run: jalankan seluruh logika (pairing, data_quality, ambient_at(),
lntai_step()/lntai_v3_step()) dan cetak ringkasan (jumlah baris, jumlah
DATA_INVALID, dll.) TANPA menulis apa pun ke lntai_results -- aman dipakai
untuk mengecek hasil sebelum benar-benar menulis ke database. CATATAN:
--dry-run TIDAK default aktif (harus diketik eksplisit), SAMA untuk
--model-version lntai-v2 maupun lntai-v3 -- flag ini sengaja tidak dibuat
berperilaku beda per model supaya tidak membingungkan; skrip mencetak
peringatan tambahan kalau --model-version lntai-v3 dipakai TANPA --dry-run,
menyarankan dry-run lebih dulu karena baseline v3 belum terkalibrasi.
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


def backfill(cur, trafo: str, ts_from: datetime, ts_to: datetime, dry_run: bool,
             model_version: str = _MODEL_VERSION) -> dict:
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

        if model_version == calculator.v3_model_version:
            # v2 `result` di atas HANYA dipakai sebagai input v3 di sini --
            # TIDAK pernah di-INSERT ketika --model-version lntai-v3 dipilih
            # (lihat docstring modul).
            v3_result = calculator.update_v3(trafo, result, ts.timestamp())
            if not dry_run:
                insert_result(
                    cur, trafo, ts, calculator.v3_model_version, v3_result, quality_flags, paired,
                    baseline_id=v3_result.get("baseline_id"), calibrated=v3_result.get("calibrated"),
                    lntai_rolling_median_30min=v3_result.get("lntai_rolling_median_30min"),
                )
        else:
            if not dry_run:
                insert_result(cur, trafo, ts, model_version, result, quality_flags, paired)

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
    parser.add_argument(
        "--model-version", default=_MODEL_VERSION, choices=[_MODEL_VERSION, "lntai-v3"],
        help="Model yang DITULIS (default: lntai-v2). lntai-v3 TIDAK menulis ulang baris v2.",
    )
    args = parser.parse_args()

    if args.model_version == "lntai-v3" and not args.dry_run:
        log.warning(
            "--model-version lntai-v3 TANPA --dry-run: baseline v3 BELUM terkalibrasi "
            "(lihat lntai_v3_config.json baseline.calibrated=false). Disarankan jalankan "
            "dulu dengan --dry-run untuk meninjau hasilnya sebelum menulis ke database."
        )

    ts_from = _parse_date(args.from_)
    ts_to = _parse_date(args.to)

    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        stats = backfill(cur, args.trafo, ts_from, ts_to, args.dry_run, model_version=args.model_version)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()

    log.info("=" * 60)
    log.info("Backfill trafo-%s %s..%s model_version=%s (%s)", args.trafo, ts_from.isoformat(), ts_to.isoformat(),
              args.model_version, "DRY-RUN, tidak ditulis" if args.dry_run else "DITULIS ke lntai_results")
    log.info("Total baris suhu: %d", stats["total"])
    log.info("Dihitung (LNTAI valid): %d", stats["computed"])
    log.info("DATA_INVALID (diblokir quality/pairing): %d", stats["data_invalid"])
    log.info("Tidak ada pasangan beban: %d", stats["not_paired"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
