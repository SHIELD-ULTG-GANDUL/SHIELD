"""server/ambient.py — fungsi murni untuk suhu ambient (T3, Cabang B).

Cabang B (dikonfirmasi Fase 0): server TIDAK bisa memanggil Open-Meteo
langsung (proxy keluar meng-intercept TLS). Baris ambient_logs ditulis oleh
BROWSER lewat POST /api/ambient-logs (lihat api.py), bukan oleh worker
server-side. Modul ini hanya berisi fungsi bantu murni (pembulatan slot,
validasi rentang, query "ambient terdekat") yang dipakai bersama oleh
api.py (T3) dan lntai_worker.py (T4) -- tidak ada I/O jaringan di sini.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# TODO(verifikasi): rentang suhu ambient yang dianggap valid. -10..60°C
# adalah DEFAULT masuk akal untuk iklim Indonesia, BUKAN hasil kalibrasi.
AMBIENT_MIN_C = -10.0
AMBIENT_MAX_C = 60.0

SLOT_MINUTES = 10


def floor_to_slot(ts: datetime, slot_minutes: int = SLOT_MINUTES) -> datetime:
    """Bulatkan `ts` ke bawah ke kelipatan `slot_minutes` -- kunci dedup
    satu baris per slot (lihat ambient_logs_slot_unique di migrasi 002).
    `ts` harus timezone-aware; dikonversi ke UTC dulu."""
    ts_utc = ts.astimezone(timezone.utc)
    minute_bucket = (ts_utc.minute // slot_minutes) * slot_minutes
    return ts_utc.replace(minute=minute_bucket, second=0, microsecond=0)


def is_valid_ambient_temperature(value: Optional[float]) -> bool:
    """True kalau `value` ada DAN dalam rentang AMBIENT_MIN_C..AMBIENT_MAX_C.
    None (fetch gagal) secara eksplisit BUKAN valid, tapi juga bukan error --
    itu kasus ok=false yang sah, lihat modul docstring."""
    if value is None:
        return False
    return AMBIENT_MIN_C <= value <= AMBIENT_MAX_C


def ambient_at(
    cur,
    ts: datetime,
    max_age_minutes: int,
    fallback_default_c: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """Cari suhu ambient (°C) ter-OK dan ter-DEKAT dengan `ts`, dalam jarak
    maksimal `max_age_minutes`. `cur` adalah psycopg cursor yang SUDAH
    terbuka pada koneksi aktif (dependency injection, modul ini tidak
    membuka koneksi sendiri -- tetap murni/testable dengan cursor palsu).

    Return (temperature_c, source):
      - (nilai, "ambient_logs")   kalau ditemukan baris ok=true dalam jendela
      - (fallback_default_c, "default") kalau tidak ada -- TIDAK PERNAH
        menginterpolasi/mengarang nilai; fallback_default_c harus berasal
        dari config (lihat lntai_worker.py, T4), bukan dikarang di sini.
    """
    cur.execute(
        """
        SELECT temperature_c
        FROM ambient_logs
        WHERE ok = TRUE
          AND "timestamp" BETWEEN %s AND %s
        ORDER BY ABS(EXTRACT(EPOCH FROM ("timestamp" - %s)))
        LIMIT 1
        """,
        (
            ts - timedelta(minutes=max_age_minutes),
            ts + timedelta(minutes=max_age_minutes),
            ts,
        ),
    )
    row = cur.fetchone()
    if row is not None:
        value = row[0] if not isinstance(row, dict) else row.get("temperature_c")
        if value is not None:
            return float(value), "ambient_logs"
    return fallback_default_c, "default"
