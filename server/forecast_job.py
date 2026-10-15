#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/forecast_job.py — SHIELD Stage 2, Part F.

Job PER JAM (bukan service long-running): untuk tiap (trafo, phase) yang
punya model forecasting beban AKTIF (ml_models model_type IN
seasonal_naive/gbm_calendar/poly_calendar, is_active=true), hasilkan forecast 168 jam
(7 hari) ke depan dari "sekarang" -- cakupan 24h DAN 7d (task brief)
keduanya diambil dari SATU seri yang sama (24 titik pertama = horizon
24h, keseluruhan 168 titik = horizon 7d; lihat GET /api/forecast).

Untuk tiap target_ts (per jam):
  1. load_p10/p50/p90 dari model forecasting aktif.
  2. healthy ΔT dari model baseline AKTIF (Part B, bukan referensi beku)
     pada load_p50 -- fallback ke fisika legacy v3 kalau tidak ada model
     aktif (konsisten dengan fallback Part B di lntai_worker.py).
  3. excess dari r_relative_weekly TERBARU berstatus 'ok' (Part D) atau
     median residual lntai_results beberapa hari terakhir (fallback,
     method dicatat -- lihat server/forecast.py compute_excess()).
  4. LNTAI_pred P10/P50/P90 (server/forecast.compute_lntai_quantiles()).
  5. aging_f_pred dari server/aging.arrhenius_factor() (Part C, None
     kalau belum terkonfigurasi).

Dijalankan per jam lewat cron (lihat README.md), BUKAN service:
    python3 forecast_job.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ml"))

import config  # noqa: E402
import aging  # noqa: E402
import forecast as fc  # noqa: E402
from settings import load_database_settings  # noqa: E402

from models import load_model_from_artifact  # noqa: E402
from forecast_models import load_forecast_model_from_artifact  # noqa: E402
import forecast_dataset as fds  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forecast-job")

TRAFO_IDS: tuple[str, ...] = ("2", "3", "4")
PHASES: tuple[str, ...] = ("R", "S", "T")
HORIZON_HOURS = 168  # 7 hari -- 24h adalah prefix 24 titik pertama dari seri yang sama
RECENT_RESIDUAL_LOOKBACK_DAYS = 7


def fetch_active_forecast_model(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, model_type, artifact_path
        FROM ml_models
        WHERE trafo = %s AND phase = %s AND is_active = TRUE
          AND model_type IN ('seasonal_naive', 'gbm_calendar', 'poly_calendar')
        ORDER BY trained_at DESC LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def fetch_active_baseline_model(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, model_type, artifact_path
        FROM ml_models
        WHERE trafo = %s AND phase = %s AND is_active = TRUE
          AND model_type IN ('physics_quadratic', 'physics_rc', 'ml_gbm', 'hybrid')
        ORDER BY trained_at DESC LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def fetch_latest_r_relative(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT r_ratio_median_pct, a_ref, i_ref, status
        FROM r_relative_weekly
        WHERE trafo = %s AND phase = %s
        ORDER BY week_start DESC LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def fetch_recent_residual_median(cur, trafo: str, phase: str, now: datetime) -> Optional[float]:
    """Median `residual` (lntai_results, model_version='lntai-v2') dalam
    RECENT_RESIDUAL_LOOKBACK_DAYS terakhir -- fallback saat r_relative_weekly
    belum 'ok' untuk (trafo, phase) ini."""
    since = now - timedelta(days=RECENT_RESIDUAL_LOOKBACK_DAYS)
    cur.execute(
        """
        SELECT residual
        FROM lntai_results
        WHERE trafo_id = %s AND model_version = 'lntai-v2' AND ts >= %s
          AND residual IS NOT NULL AND level != 'DATA_INVALID'
        """,
        (trafo, since),
    )
    rows = cur.fetchall()
    values = sorted(r["residual"] for r in rows if r["residual"] is not None)
    if not values:
        return None
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2.0


def fetch_aging_rates(cur, trafo: str, phase: str, now: datetime) -> tuple[Optional[float], Optional[float]]:
    """(last_7d_total_eq_days_increment, last_30d_daily_avg_eq_days) --
    dihitung dari eq_days_cum harian (Part C aging_daily), BUKAN ulang
    integrasi F(t) di sini."""
    since_30d = (now - timedelta(days=30)).date()
    cur.execute(
        "SELECT date, eq_days_cum FROM aging_daily WHERE trafo = %s AND phase = %s "
        "AND date >= %s ORDER BY date ASC",
        (trafo, phase, since_30d),
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        return None, None

    cum_values = [(r["date"], r["eq_days_cum"]) for r in rows if r["eq_days_cum"] is not None]
    if len(cum_values) < 2:
        return None, None

    total_increment_30d = cum_values[-1][1] - cum_values[0][1]
    span_days_30d = (cum_values[-1][0] - cum_values[0][0]).days or 1
    daily_avg_30d = total_increment_30d / span_days_30d

    since_7d = (now - timedelta(days=7)).date()
    last_7d = [v for d, v in cum_values if d >= since_7d]
    total_7d = (last_7d[-1] - last_7d[0]) if len(last_7d) >= 2 else None

    return total_7d, daily_avg_30d


def _get_baseline_params(
    cur, trafo: str, phase: str,
) -> tuple[float, float, float, Optional[int]]:
    """(a, b, i_ref, ml_models_id_or_None) dari model baseline AKTIF
    (Part B) -- fallback ke fisika legacy v3 kalau tidak ada model aktif
    ATAU model aktif tidak punya koefisien A langsung (ml_gbm, lihat Part D
    untuk pola 'pinjam physics_rc' yang sama -- di sini kita AMBIL JALAN
    LEBIH SEDERHANA: ml_gbm aktif berarti healthy ΔT dipakai dari fisika
    legacy v3 langsung, bukan meminjam physics_rc lain, karena task brief
    Part F hanya bilang 'the active baseline' tanpa menyebut pola pinjaman
    seperti Part D -- didokumentasikan sebagai simplifikasi)."""
    row = fetch_active_baseline_model(cur, trafo, phase)
    if row is None:
        return config.LNTAI_V3_A, config.LNTAI_V3_B, config.LNTAI_V3_I_REF, None

    model_type = row["model_type"]
    with open(_ROOT / row["artifact_path"], "r", encoding="utf-8") as f:
        artifact = json.load(f)
    model = load_model_from_artifact(model_type, artifact)

    if model_type == "physics_quadratic" or model_type == "physics_rc":
        return model.a, model.b, model.i_ref, row["id"]
    if model_type == "hybrid":
        return model.physics.a, model.physics.b, model.physics.i_ref, row["id"]
    # ml_gbm aktif -- tidak punya A/B/i_ref sendiri, lihat catatan docstring.
    return config.LNTAI_V3_A, config.LNTAI_V3_B, config.LNTAI_V3_I_REF, row["id"]


def compute_forecast_series(cur, trafo: str, phase: str, now: datetime) -> Optional[list[dict[str, Any]]]:
    """Return list 168 dict (satu per jam, HORIZON_HOURS ke depan), atau
    None kalau tidak ada model forecasting beban aktif untuk (trafo,phase)
    ini -- caller TIDAK menulis apa pun ke `forecasts` pada kasus ini."""
    forecast_row = fetch_active_forecast_model(cur, trafo, phase)
    if forecast_row is None:
        return None

    with open(_ROOT / forecast_row["artifact_path"], "r", encoding="utf-8") as f:
        artifact = json.load(f)
    forecast_model = load_forecast_model_from_artifact(forecast_row["model_type"], artifact)

    lookup: dict[float, float] = {}
    if forecast_row["model_type"] == "seasonal_naive":
        history_from = now - timedelta(days=8)  # 7 hari lookback + sedikit margin
        hourly = fds.build_hourly_load_series(cur, trafo, phase, history_from, now)
        lookup = {ts.timestamp(): val for ts, val in hourly}

    a, b, i_ref, _baseline_model_id = _get_baseline_params(cur, trafo, phase)

    r_rel = fetch_latest_r_relative(cur, trafo, phase)
    r_over_r_ref = a_ref = r_i_ref = None
    if r_rel is not None and r_rel["status"] == "ok" and r_rel["r_ratio_median_pct"] is not None:
        r_over_r_ref = r_rel["r_ratio_median_pct"] / 100.0
        a_ref = r_rel["a_ref"]
        r_i_ref = r_rel["i_ref"]
    recent_residual_median = fetch_recent_residual_median(cur, trafo, phase, now)

    aging_configured = aging.is_configured()

    points: list[dict[str, Any]] = []
    for h in range(1, HORIZON_HOURS + 1):
        target_ts = now + timedelta(hours=h)
        target_epoch = target_ts.timestamp()

        if forecast_row["model_type"] == "seasonal_naive":
            load_p10, load_p50, load_p90 = forecast_model.predict_quantiles(target_epoch, lookup)
        else:
            load_p10, load_p50, load_p90 = forecast_model.predict_quantiles(target_epoch)

        lntai_p10 = lntai_p50 = lntai_p90 = None
        excess_method = "unavailable"
        t_hot_pred = None
        f_pred = None

        if load_p50 is not None:
            healthy_dt = fc.compute_healthy_delta_t(a, b, i_ref, load_p50)
            excess, excess_method = fc.compute_excess(r_over_r_ref, a_ref, r_i_ref, load_p50, recent_residual_median)
            t_hot_pred = fc.compute_t_hot_pred(healthy_dt, excess)
            lntai_p10, lntai_p50, lntai_p90, _ = fc.compute_lntai_quantiles(
                load_p10, load_p50, load_p90, r_over_r_ref, a_ref, r_i_ref, recent_residual_median,
            )
            if t_hot_pred is not None and aging_configured:
                f_pred = aging.arrhenius_factor(t_hot_pred)

        points.append({
            "target_ts": target_ts, "load_p10_a": load_p10, "load_p50_a": load_p50, "load_p90_a": load_p90,
            "t_hot_pred_c": t_hot_pred, "lntai_pred_p10": lntai_p10, "lntai_pred_p50": lntai_p50,
            "lntai_pred_p90": lntai_p90, "aging_f_pred": f_pred,
            "method": forecast_row["model_type"], "excess_method": excess_method,
            "forecast_model_id": forecast_row["id"],
        })

    return points


def insert_forecast_point(cur, trafo: str, phase: str, point: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO forecasts (
            trafo, phase, target_ts, load_p10_a, load_p50_a, load_p90_a, t_hot_pred_c,
            lntai_pred_p10, lntai_pred_p50, lntai_pred_p90, aging_f_pred, method,
            excess_method, forecast_model_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trafo, phase, target_ts) DO UPDATE SET
            load_p10_a = EXCLUDED.load_p10_a, load_p50_a = EXCLUDED.load_p50_a,
            load_p90_a = EXCLUDED.load_p90_a, t_hot_pred_c = EXCLUDED.t_hot_pred_c,
            lntai_pred_p10 = EXCLUDED.lntai_pred_p10, lntai_pred_p50 = EXCLUDED.lntai_pred_p50,
            lntai_pred_p90 = EXCLUDED.lntai_pred_p90, aging_f_pred = EXCLUDED.aging_f_pred,
            method = EXCLUDED.method, excess_method = EXCLUDED.excess_method,
            forecast_model_id = EXCLUDED.forecast_model_id, created_at = now()
        """,
        (
            trafo, phase, point["target_ts"], point["load_p10_a"], point["load_p50_a"], point["load_p90_a"],
            point["t_hot_pred_c"], point["lntai_pred_p10"], point["lntai_pred_p50"], point["lntai_pred_p90"],
            point["aging_f_pred"], point["method"], point["excess_method"], point["forecast_model_id"],
        ),
    )


def run_once(pool: ConnectionPool) -> int:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    total = 0
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for trafo in TRAFO_IDS:
            for phase in PHASES:
                points = compute_forecast_series(cur, trafo, phase, now)
                if points is None:
                    continue
                for point in points:
                    insert_forecast_point(cur, trafo, phase, point)
                total += len(points)
                log.info("Trafo-%s fasa %s: %d titik forecast ditulis/diperbarui", trafo, phase, len(points))
    return total


def main() -> None:
    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)
    log.info("=" * 60)
    log.info("Forecast Job (Stage 2 Part F) -- mulai")
    n = run_once(pool)
    log.info("Forecast Job selesai -- %d titik ditulis/diperbarui", n)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
