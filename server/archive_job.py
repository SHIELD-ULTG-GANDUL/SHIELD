#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/archive_job.py — SHIELD Tahap 1, T6 (DIREVISI, lihat README.md).

Rencana awal T6 memakai continuous aggregate + retention policy
TimescaleDB. Audit Fase 0 mengonfirmasi database ini PostgreSQL polos
(TimescaleDB TIDAK terpasang) -- desain diganti: job harian biasa yang
menyalin baris trafo_temperature/trafo_load yang jatuh di dalam SATU
maintenance_windows (migrasi 005) ke tabel *_archive (migrasi 006),
SEBELUM retensi/penghapusan apa pun (migrasi 007, TIDAK AKTIF, keputusan
dipegang pengguna) dijalankan di masa depan.

Job ini TIDAK PERNAH menghapus baris asli -- hanya menyalin (INSERT ...
SELECT ... ON CONFLICT (id) DO NOTHING). Idempotent: menjalankan ulang
job yang sama (atau restart di tengah jalan) tidak menduplikasi baris
arsip, dan maintenance_windows.archived=true membuat window yang sudah
diproses dilewati pada run berikutnya.

Dijalankan harian (lihat README.md untuk contoh entri cron):
    python3 archive_job.py
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from settings import load_database_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("archive-job")


def fetch_pending_windows(cur) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, trafo_id, start_time, end_time, reason
        FROM maintenance_windows
        WHERE archived = FALSE
        ORDER BY start_time ASC
        """
    )
    return cur.fetchall()


def archive_window(cur, window: dict[str, Any]) -> tuple[int, int]:
    """Salin baris trafo_temperature/trafo_load dalam jendela `window` ke
    tabel *_archive. Return (jumlah_suhu_disalin, jumlah_beban_disalin) --
    psycopg tidak mengembalikan rowcount dari INSERT...SELECT lewat
    fetchall, jadi dihitung lewat RETURNING id."""
    trafo_id = window["trafo_id"]

    if trafo_id is None:
        cur.execute(
            """
            INSERT INTO trafo_temperature_archive
            SELECT * FROM trafo_temperature
            WHERE "timestamp" BETWEEN %s AND %s
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            (window["start_time"], window["end_time"]),
        )
    else:
        cur.execute(
            """
            INSERT INTO trafo_temperature_archive
            SELECT * FROM trafo_temperature
            WHERE "timestamp" BETWEEN %s AND %s AND trafo_id = %s
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            (window["start_time"], window["end_time"], trafo_id),
        )
    temp_copied = len(cur.fetchall())

    if trafo_id is None:
        cur.execute(
            """
            INSERT INTO trafo_load_archive
            SELECT * FROM trafo_load
            WHERE "timestamp" BETWEEN %s AND %s
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            (window["start_time"], window["end_time"]),
        )
    else:
        cur.execute(
            """
            INSERT INTO trafo_load_archive
            SELECT * FROM trafo_load
            WHERE "timestamp" BETWEEN %s AND %s AND trafo_id = %s
            ON CONFLICT (id) DO NOTHING
            RETURNING id
            """,
            (window["start_time"], window["end_time"], trafo_id),
        )
    load_copied = len(cur.fetchall())

    cur.execute("UPDATE maintenance_windows SET archived = TRUE WHERE id = %s", (window["id"],))
    return temp_copied, load_copied


def run_once(pool: ConnectionPool) -> int:
    """Proses semua maintenance_windows yang belum diarsipkan. Return
    jumlah window yang diproses."""
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        windows = fetch_pending_windows(cur)
        for window in windows:
            temp_n, load_n = archive_window(cur, window)
            log.info(
                "Window #%s (trafo=%s, %s..%s, '%s'): %d baris suhu + %d baris beban disalin ke archive",
                window["id"], window["trafo_id"] or "SEMUA", window["start_time"], window["end_time"],
                window["reason"], temp_n, load_n,
            )
    return len(windows)


def main() -> None:
    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)
    log.info("=" * 60)
    log.info("Archive Job (T6) -- mulai")
    n = run_once(pool)
    log.info("Archive Job (T6) selesai -- %d maintenance window diproses", n)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
