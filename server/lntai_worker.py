#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/lntai_worker.py — SHIELD Tahap 1, T4.

Menghitung LNTAI v2 di SERVER, bukan di browser, dengan menggunakan ULANG
lntai_calculator.LNTAICalculator (root proyek) dan lntai_config.json — modul
ini TIDAK menulis ulang formula LNTAI, hanya memasok data (pairing suhu +
beban, ambient_at(), quality_flags) dan menulis hasilnya ke lntai_results
(lihat migrations/004_create_lntai_results.sql).

Checkpoint per trafo disimpan di worker_state (migrations/003), supaya
restart worker lanjut dari baris terakhir yang sudah diproses, bukan dari
awal histori.

Alur per batch, per trafo:
  1. Baca checkpoint (last_processed_ts) dari worker_state.
  2. Ambil baris trafo_temperature dengan timestamp > checkpoint, urut naik.
  3. Untuk setiap baris, cari baris trafo_load terdekat dalam
     pairing_tolerance_minutes (lntai_config.json) -- kalau tidak ada,
     paired=False dan baris ditandai PAIRING_GAP (lihat data_quality.py).
  4. Gabungkan quality_flags baris suhu + baris beban (OR bitmask). Kalau
     data_quality.should_skip_lntai() True (MISSING/OUT_OF_RANGE/STUCK) ATAU
     paired=False, LNTAI TIDAK dihitung dari sampel itu (level=DATA_INVALID,
     state filter lag termal TIDAK diupdate dari sampel ini) -- sesuai T5.
  5. Kalau lolos, ambil ambient_at() (fallback ke LNTAI_CONFIG["ambient"]
     ["default"] kalau tidak ada baris ambient_logs dalam jendela), panggil
     LNTAICalculator.update(trafo, sample).
  6. Tulis satu baris ke lntai_results (ON CONFLICT DO NOTHING -- idempotent,
     lihat komentar PK di migrasi 004).
  7. Setelah batch selesai, maju checkpoint ke timestamp baris TERAKHIR yang
     diproses (bukan yang berhasil dihitung -- baris DATA_INVALID tetap maju
     checkpoint supaya worker tidak terus mencoba memproses baris yang sama).

Stage 2, Part A — LNTAI v3 ditulis BERDAMPINGAN dengan v2 di atas (baris
terpisah, model_version='lntai-v3', PK sama (trafo_id, ts, model_version)
pada tabel yang sama), dari v2_result yang SUDAH dihitung untuk sampel yang
sama (lihat lntai_calculator.lntai_v3_step()) -- TIDAK menjalankan filter
I^2 kedua kalinya, hanya mengubah basis normalisasi arus + baseline. v2
TIDAK diubah/dihapus sama sekali (lihat lntai_config.json vs
lntai_v3_config.json, dua file konfigurasi terpisah).

Dijalankan sebagai service (lihat shield-lntai.service, README.md):
    python3 lntai_worker.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from lntai_calculator import LNTAICalculator  # noqa: E402

import ambient  # noqa: E402
import data_quality as dq  # noqa: E402
from settings import load_database_settings, load_worker_settings  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lntai-worker")

WORKER_NAME = "lntai_worker"
TRAFO_IDS: tuple[str, ...] = ("2", "3", "4")
BATCH_LIMIT = 500

_PAIRING_TOLERANCE_MINUTES: float = config.LNTAI_CONFIG["pairing_tolerance_minutes"]
_AMBIENT_DEFAULT: float = config.LNTAI_CONFIG["ambient"]["default"]
_AMBIENT_MAX_AGE_MINUTES: float = config.LNTAI_CONFIG["ambient"]["max_age_minutes"]
_MODEL_VERSION: str = config.LNTAI_CONFIG["model_version"]


# ════════════════════════════════════════════════════════════════════════
# DB helpers — satu fungsi satu query, supaya mudah dites dengan cursor palsu.
# ════════════════════════════════════════════════════════════════════════

def get_checkpoint(cur, trafo: str) -> Optional[datetime]:
    cur.execute(
        "SELECT last_processed_ts FROM worker_state WHERE worker_name = %s AND trafo_id = %s",
        (WORKER_NAME, trafo),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row["last_processed_ts"] if isinstance(row, dict) else row[0]


def set_checkpoint(cur, trafo: str, ts: datetime) -> None:
    cur.execute(
        """
        INSERT INTO worker_state (worker_name, trafo_id, last_processed_ts, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (worker_name, trafo_id)
        DO UPDATE SET last_processed_ts = EXCLUDED.last_processed_ts, updated_at = now()
        """,
        (WORKER_NAME, trafo, ts),
    )


def fetch_unprocessed_temperature_rows(cur, trafo: str, since: Optional[datetime], limit: int = BATCH_LIMIT) -> list[dict]:
    cur.execute(
        """
        SELECT "timestamp", r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah, quality_flags
        FROM trafo_temperature
        WHERE trafo_id = %s
          AND (%s::timestamptz IS NULL OR "timestamp" > %s)
        ORDER BY "timestamp" ASC
        LIMIT %s
        """,
        (trafo, since, since, limit),
    )
    return cur.fetchall()


def find_paired_load_row(cur, trafo: str, ts: datetime, tolerance_minutes: float) -> Optional[dict]:
    """Baris trafo_load TERDEKAT dengan `ts`, dalam jarak `tolerance_minutes`."""
    delta = timedelta(minutes=tolerance_minutes)
    cur.execute(
        """
        SELECT ia, ib, ic, quality_flags
        FROM trafo_load
        WHERE trafo_id = %s
          AND "timestamp" BETWEEN %s AND %s
        ORDER BY ABS(EXTRACT(EPOCH FROM ("timestamp" - %s)))
        LIMIT 1
        """,
        (trafo, ts - delta, ts + delta, ts),
    )
    return cur.fetchone()


def insert_result(cur, trafo: str, ts: datetime, model_version: str, result: dict[str, Any],
                   quality_flags: int, paired: bool,
                   baseline_id: Optional[str] = None, calibrated: Optional[bool] = None,
                   lntai_rolling_median_30min: Optional[float] = None) -> None:
    """Tulis satu baris lntai_results. `baseline_id`/`calibrated`/
    `lntai_rolling_median_30min` (Stage 2 Part A, migrasi 008) default None
    -- dibiarkan NULL untuk baris v2 (T4), diisi eksplisit oleh caller v3."""
    import json as _json

    cur.execute(
        """
        INSERT INTO lntai_results (
            trafo_id, ts, model_version, lntai, level, s_temp, s_delta_t, s_res,
            hottest_phase, t_contact, t_ambient, ambient_source, delta_t_real,
            delta_t_pred, residual, z, i_eff_norm, delta_phase, dab_max,
            per_phase, quality_flags, paired, error,
            baseline_id, calibrated, lntai_rolling_median_30min
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (trafo_id, ts, model_version) DO NOTHING
        """,
        (
            trafo, ts, model_version, result.get("lntai"), result.get("level"),
            result.get("S_Temp"), result.get("S_DeltaT"), result.get("S_RES"),
            result.get("hottest_phase"), result.get("t_contact"), result.get("t_ambient"),
            result.get("ambient_source"), result.get("delta_T_real"),
            result.get("delta_T_pred"), result.get("residual"), result.get("z"),
            result.get("i_eff_norm"), result.get("delta_phase"), result.get("dAB_max"),
            _json.dumps(result.get("per_phase")) if result.get("per_phase") is not None else None,
            quality_flags, paired, result.get("error"),
            baseline_id, calibrated, lntai_rolling_median_30min,
        ),
    )


# ════════════════════════════════════════════════════════════════════════
# Logika murni (testable tanpa DB) — membangun sample untuk lntai_step()
# dan menentukan apakah sampel boleh dihitung.
# ════════════════════════════════════════════════════════════════════════

def build_sample(temp_row: dict, load_row: Optional[dict], ambient_value: Optional[float],
                  ambient_source: str, ts: datetime) -> dict[str, Any]:
    return {
        "R_Atas": temp_row["r_atas"], "R_Bawah": temp_row["r_bawah"],
        "S_Atas": temp_row["s_atas"], "S_Bawah": temp_row["s_bawah"],
        "T_Atas": temp_row["t_atas"], "T_Bawah": temp_row["t_bawah"],
        "I_R": load_row["ia"] if load_row else None,
        "I_S": load_row["ib"] if load_row else None,
        "I_T": load_row["ic"] if load_row else None,
        "t_ambient": ambient_value,
        "ambient_source": ambient_source,
        "timestamp": ts.timestamp(),
    }


def combined_quality_flags(temp_row: dict, load_row: Optional[dict], paired: bool) -> int:
    """OR bitmask flag baris suhu + baris beban (kalau ada pasangan),
    ditambah PAIRING_GAP kalau tidak ada beban dalam toleransi waktu."""
    bitmask = (temp_row.get("quality_flags") or 0) | ((load_row or {}).get("quality_flags") or 0)
    if not paired:
        bitmask |= dq.flags_to_bitmask({"PAIRING_GAP"})
    return bitmask


def should_compute(quality_flags: int, paired: bool) -> bool:
    """False kalau quality_flags memblokir (MISSING/OUT_OF_RANGE/STUCK, lihat
    data_quality.should_skip_lntai()) ATAU tidak ada baris beban terpasang --
    LNTAI tidak bisa dihitung tanpa arus, PAIRING_GAP sendiri sengaja TIDAK
    termasuk bit yang diblokir should_skip_lntai() (lihat T5), jadi dicek
    terpisah di sini."""
    if not paired:
        return False
    return not dq.should_skip_lntai(quality_flags)


# ════════════════════════════════════════════════════════════════════════
# Orkestrasi — satu batch per trafo, dipanggil oleh loop utama & backfill.
# ════════════════════════════════════════════════════════════════════════

def process_trafo_batch(cur, calculator: LNTAICalculator, trafo: str) -> int:
    """Proses satu batch (<=BATCH_LIMIT baris) untuk satu trafo. Return jumlah
    baris yang diproses (0 berarti tidak ada data baru)."""
    since = get_checkpoint(cur, trafo)
    rows = fetch_unprocessed_temperature_rows(cur, trafo, since)
    if not rows:
        return 0

    last_ts = None
    for row in rows:
        ts = row["timestamp"]
        load_row = find_paired_load_row(cur, trafo, ts, _PAIRING_TOLERANCE_MINUTES)
        paired = load_row is not None
        quality_flags = combined_quality_flags(row, load_row, paired)

        if should_compute(quality_flags, paired):
            amb_value, amb_source = ambient.ambient_at(
                cur, ts, int(_AMBIENT_MAX_AGE_MINUTES), fallback_default_c=_AMBIENT_DEFAULT
            )
            sample = build_sample(row, load_row, amb_value, amb_source, ts)
            result = calculator.update(trafo, sample)
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

        insert_result(cur, trafo, ts, _MODEL_VERSION, result, quality_flags, paired)

        # Stage 2 Part A: LNTAI v3 ditulis BERDAMPINGAN dengan v2 di atas,
        # dari v2_result yang SAMA (reuse i_eff, tidak menjalankan filter
        # I^2 lagi) -- quality_flags/paired SAMA supaya kedua model_version
        # konsisten untuk baris trafo_id+ts yang sama (lihat lntai_v3_step()
        # untuk kenapa input DATA_INVALID/tidak-terpasang otomatis membuat
        # v3 juga DATA_INVALID, tanpa logika gating terpisah di sini).
        v3_result = calculator.update_v3(trafo, result, ts.timestamp())
        insert_result(
            cur, trafo, ts, calculator.v3_model_version, v3_result, quality_flags, paired,
            baseline_id=v3_result.get("baseline_id"), calibrated=v3_result.get("calibrated"),
            lntai_rolling_median_30min=v3_result.get("lntai_rolling_median_30min"),
        )

        last_ts = ts

    if last_ts is not None:
        set_checkpoint(cur, trafo, last_ts)
    return len(rows)


def run_once(pool: ConnectionPool, calculator: LNTAICalculator) -> int:
    total = 0
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for trafo in TRAFO_IDS:
            n = process_trafo_batch(cur, calculator, trafo)
            total += n
            if n:
                log.info("Trafo-%s: %d baris diproses", trafo, n)
    return total


def main() -> None:
    db_settings = load_database_settings()
    worker_settings = load_worker_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=5, open=True)
    calculator = LNTAICalculator()

    log.info("=" * 60)
    log.info("LNTAI Worker -- model_versions=%s,%s", _MODEL_VERSION, calculator.v3_model_version)
    log.info("Interval: %ss", worker_settings.lntai_interval_seconds)
    log.info("=" * 60)

    while True:
        try:
            n = run_once(pool, calculator)
            if n == 0:
                log.debug("Tidak ada data baru di semua trafo")
        except Exception:
            log.exception("Gagal menjalankan batch LNTAI worker")
        time.sleep(worker_settings.lntai_interval_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped")
        sys.exit(0)
