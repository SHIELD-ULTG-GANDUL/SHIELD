#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/aging_job.py — SHIELD Stage 2, Part C.

Job HARIAN (bukan service long-running): untuk tiap (trafo, phase), hitung
F(t)/EqAge hari-hari yang belum ada baris di aging_daily, dari hari setelah
baris terakhir (atau dari reset_date kalau belum ada baris sama sekali)
sampai KEMARIN (hari ini belum lengkap datanya, selalu dilewati).

Reset point = fallback_reset_date (aging_config.json) selama Part E (log
pemeliharaan/resistance-test dengan event resets_baseline=true) belum ada
-- lihat get_reset_date(). Kalau reset_date BERUBAH (event reset baru, atau
fallback_reset_date diubah manual) dibanding baris terakhir yang tersimpan,
akumulasi eq_days_cum/calendar_days RESTART dari 0 pada hari reset itu,
bukan melanjutkan angka lama (lihat process_trafo_phase()). CATATAN: ini
TIDAK menghapus/menghitung ulang baris LAMA yang sudah tersimpan dari
sebelum reset -- hanya baris BARU yang dihitung ulang dengan reset point
terkini. Backfill ulang histori lama kalau diperlukan ada di luar scope
job ini (lihat server/README.md kalau ingin menambahkannya nanti).

Kalau aging_config.json belum terkonfigurasi (ea_ev/t_ref_c null), job ini
TIDAK menulis apa pun -- bukan error, hanya belum ada angka yang bisa
dihitung (lihat server/aging.py is_configured()).

Dijalankan harian lewat cron (lihat README.md), SETELAH archive_job.py
kalau dijadwalkan di hari yang sama (tidak ada dependensi langsung,
hanya konvensi urutan):
    python3 aging_job.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import aging
import data_quality as dq
from settings import load_database_settings

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aging-job")

TRAFO_IDS: tuple[str, ...] = ("2", "3", "4")
PHASES: tuple[str, ...] = ("R", "S", "T")
_PHASE_COLUMNS: dict[str, tuple[str, str]] = {
    "R": ("r_atas", "r_bawah"), "S": ("s_atas", "s_bawah"), "T": ("t_atas", "t_bawah"),
}


def config_hash(cfg: dict[str, Any] = aging.AGING_CONFIG) -> str:
    """Hash pendek isi aging_config.json (minus key `_*` berkomentar) --
    dipakai mendeteksi baris lama yang dihitung dengan konfigurasi beda."""
    canonical = json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def get_reset_date(cfg: dict[str, Any] = aging.AGING_CONFIG) -> tuple[datetime, str]:
    """Part E (resets_baseline=true) belum ada -- SELALU fallback_config
    untuk saat ini. Fungsi ini sengaja dipisah supaya Part E nanti tinggal
    mengganti isinya (query tabel event pemeliharaan) tanpa mengubah
    pemanggilnya."""
    raw = cfg["fallback_reset_date"]
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt, "fallback_config"


def fetch_last_row(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT date, eq_days_cum, calendar_days, reset_date, config_hash
        FROM aging_daily
        WHERE trafo = %s AND phase = %s
        ORDER BY date DESC LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def fetch_day_samples(cur, trafo: str, phase: str, day: date) -> list[tuple[float, float]]:
    """Pembacaan suhu hotspot fasa `phase` (max Atas/Bawah) untuk SATU hari
    UTC, MISSING/OUT_OF_RANGE dibuang (ZERO_READING SENGAJA tidak dibuang,
    konsisten dengan T2/T5 -- 0°C literal nyaris mustahil untuk trafo
    distribusi tropis, tapi filosofi proyek ini tidak membuang ZERO_READING
    di mana pun kecuali eksplisit diminta, lihat ml/dataset.py untuk
    satu-satunya tempat itu dilakukan, untuk alasan training model)."""
    atas_col, bawah_col = _PHASE_COLUMNS[phase]
    day_start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    cur.execute(
        f"""
        SELECT "timestamp", {atas_col} AS atas, {bawah_col} AS bawah, quality_flags
        FROM trafo_temperature
        WHERE trafo_id = %s AND "timestamp" >= %s AND "timestamp" < %s
        ORDER BY "timestamp" ASC
        """,
        (trafo, day_start, day_end),
    )
    rows = cur.fetchall()

    samples: list[tuple[float, float]] = []
    for row in rows:
        flags = set(dq.bitmask_to_flags(row.get("quality_flags")))
        if "MISSING" in flags or "OUT_OF_RANGE" in flags:
            continue
        atas, bawah = row["atas"], row["bawah"]
        if atas is None or bawah is None:
            continue
        samples.append((row["timestamp"].timestamp(), max(atas, bawah)))
    return samples


def insert_row(
    cur, trafo: str, phase: str, day: date, result: dict[str, Any],
    eq_days_cum: float, calendar_days: float, cfg_hash: str, reset_date: datetime, reset_source: str,
) -> None:
    cur.execute(
        """
        INSERT INTO aging_daily (
            trafo, phase, date, f_mean, f_peak, t_hot_mean_c, t_hot_peak_c,
            eq_days_cum, calendar_days, config_hash, reset_date, reset_source,
            n_samples, n_gaps_excluded
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trafo, phase, date) DO NOTHING
        """,
        (
            trafo, phase, day, result["f_mean"], result["f_peak"],
            result["t_hot_mean_c"], result["t_hot_peak_c"],
            eq_days_cum, calendar_days, cfg_hash, reset_date, reset_source,
            result["n_samples"], result["n_gaps_excluded"],
        ),
    )


def process_trafo_phase(
    cur, trafo: str, phase: str, reset_date: datetime, reset_source: str,
    cfg_hash: str, yesterday: date,
) -> int:
    """Proses hari demi hari untuk (trafo, phase) sampai `yesterday`. Return
    jumlah baris baru ditulis."""
    last = fetch_last_row(cur, trafo, phase)
    if last is not None and last["reset_date"] == reset_date:
        start_day = last["date"] + timedelta(days=1)
        eq_days_cum = last["eq_days_cum"] or 0.0
        calendar_days = last["calendar_days"] or 0.0
    else:
        # Baris terakhir tidak ada, ATAU reset_date berubah (event reset
        # baru) -- restart akumulasi dari reset_date, bukan melanjutkan
        # angka lama.
        start_day = reset_date.date()
        eq_days_cum = 0.0
        calendar_days = 0.0

    n_written = 0
    day = start_day
    while day <= yesterday:
        samples = fetch_day_samples(cur, trafo, phase, day)
        result = aging.integrate_eq_days(samples)
        if result["status"] == "not_configured":
            break  # tidak ada gunanya lanjut ke hari berikutnya kalau belum terkonfigurasi
        eq_days_cum += result["eq_days"]
        calendar_days += 1.0
        insert_row(cur, trafo, phase, day, result, eq_days_cum, calendar_days, cfg_hash, reset_date, reset_source)
        n_written += 1
        day += timedelta(days=1)

    return n_written


def run_once(pool: ConnectionPool) -> int:
    if not aging.is_configured():
        log.warning(
            "aging_config.json belum terkonfigurasi (ea_ev/t_ref_c null, lihat TODO(calibration) "
            "di file itu) -- job tidak menulis apa pun."
        )
        return 0

    cfg_hash = config_hash()
    reset_date, reset_source = get_reset_date()
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    total = 0
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for trafo in TRAFO_IDS:
            for phase in PHASES:
                n = process_trafo_phase(cur, trafo, phase, reset_date, reset_source, cfg_hash, yesterday)
                total += n
                if n:
                    log.info("Trafo-%s fasa %s: %d hari ditulis", trafo, phase, n)
    return total


def main() -> None:
    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)
    log.info("=" * 60)
    log.info("Aging Job (Stage 2 Part C) -- mulai")
    n = run_once(pool)
    log.info("Aging Job selesai -- %d baris (trafo x fasa x hari) ditulis", n)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
