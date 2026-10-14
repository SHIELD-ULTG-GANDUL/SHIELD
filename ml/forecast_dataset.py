"""ml/forecast_dataset.py — SHIELD Stage 2, Part F.

Seri beban (arus) PER JAM dari database, untuk ml/train_load_forecast.py
dan server/forecast_job.py. Resample ke rata-rata per jam (bukan sampel
mentah tiap ~5 menit) karena forecast di sini per-jam (task brief:
"hourly forecasts for 24h and 7 days"), BUKAN realtime streaming seperti
LNTAI v2/v3.

MISSING/OUT_OF_RANGE (lihat server/data_quality.py) dibuang SEBELUM
dirata-ratakan per jam (bitmask check inline di SQL, bukan ditarik mentah
lalu difilter di Python) -- ZERO_READING SENGAJA tidak dibuang, konsisten
dengan filosofi proyek ini (0A adalah bacaan sah, bukan data hilang).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "server"))

import data_quality as dq  # noqa: E402

_PHASE_CURRENT_COLUMN: dict[str, str] = {"R": "ia", "S": "ib", "T": "ic"}
_BLOCKING_BITMASK: int = dq.flags_to_bitmask({"MISSING", "OUT_OF_RANGE"})


def build_hourly_load_series(cur, trafo: str, phase: str, ts_from: datetime, ts_to: datetime) -> list[tuple[datetime, float]]:
    """Return [(hour_start_utc, mean_current_amps), ...] terurut naik --
    HANYA jam yang punya >= 1 sampel valid (quality_flags tidak
    mengandung MISSING/OUT_OF_RANGE) ikut dihitung; jam tanpa sampel
    valid sama sekali TIDAK muncul di hasil (bukan diisi 0/NULL)."""
    current_col = _PHASE_CURRENT_COLUMN[phase]
    cur.execute(
        f"""
        SELECT date_trunc('hour', "timestamp") AS hour_start,
               AVG({current_col}) AS mean_current
        FROM trafo_load
        WHERE trafo_id = %s
          AND "timestamp" >= %s AND "timestamp" < %s
          AND {current_col} IS NOT NULL
          AND (quality_flags IS NULL OR (quality_flags & %s) = 0)
        GROUP BY hour_start
        ORDER BY hour_start ASC
        """,
        (trafo, ts_from, ts_to, _BLOCKING_BITMASK),
    )
    rows = cur.fetchall()
    return [(r["hour_start"], float(r["mean_current"])) for r in rows]


def to_epoch_pairs(hourly_series: list[tuple[datetime, float]]) -> list[tuple[float, float]]:
    return [(ts.timestamp(), val) for ts, val in hourly_series]
