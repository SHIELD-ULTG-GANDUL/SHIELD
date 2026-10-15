#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ml/train_load_forecast.py — SHIELD Stage 2, Part F (+ poly_calendar,
go-live 2026-10-15 rework of the "Forecast Realtime" tab).

Melatih dan membandingkan 3 model forecasting BEBAN untuk SATU
(trafo, phase): seasonal_naive (sama jam minggu lalu), gbm_calendar
(gradient boosting atas fitur kalender), poly_calendar (regresi
polinomial atas fitur kalender yang sama -- pembanding yang lebih
sederhana/interpretable terhadap GBM). Split waktu-berurutan: rentang
historis awal = "training" (gbm_calendar.fit()/poly_calendar.fit(),
seasonal_naive hanya butuh data mentah, tidak benar-benar "dilatih"),
--backtest-days hari TERAKHIR = holdout untuk kalibrasi P10/P50/P90
(kuantil residual empiris) dan metrik (MAPE, mean pinball loss).

TIDAK PERNAH mengaktifkan model (is_active selalu FALSE saat training,
sama seperti ml/train_baseline.py Part B) -- aktivasi manual lewat
POST /api/ml/models/{id}/activate setelah meninjau tabel perbandingan.

Pemakaian:
    python3 ml/train_load_forecast.py --trafo 2 --phase R \\
        --historical-from 2026-07-01 --historical-to 2026-09-01

    python3 ml/train_load_forecast.py --trafo 2 --phase R \\
        --historical-from 2026-07-01 --historical-to 2026-09-01 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast_dataset as fds  # noqa: E402
import forecast_models as fm  # noqa: E402
from settings import load_database_settings  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-load-forecast")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_BACKTEST_DAYS = 7


def _version_string(trained_at: datetime) -> str:
    return "v" + trained_at.strftime("%Y%m%dT%H%M%S")


def train_and_backtest(
    hourly_series: list[tuple[Any, float]], backtest_days: int,
) -> list[dict[str, Any]]:
    """`hourly_series`: histori LENGKAP (training+backtest), terurut naik
    waktu. Return list hasil per model_type (status ok/failed)."""
    if len(hourly_series) < 24 * (backtest_days + 7):
        raise ValueError(
            f"Hanya {len(hourly_series)} jam data -- butuh minimal "
            f"{24 * (backtest_days + 7)} jam (>= {backtest_days} hari backtest + 7 hari training "
            f"minimum untuk seasonal_naive punya lookback)."
        )

    epoch_pairs = fds.to_epoch_pairs(hourly_series)
    cutoff_ts = hourly_series[-1][0] - timedelta(days=backtest_days)
    train_pairs = [(t, v) for t, v in epoch_pairs if t < cutoff_ts.timestamp()]
    backtest_pairs = [(t, v) for t, v in epoch_pairs if t >= cutoff_ts.timestamp()]
    full_lookup = {t: v for t, v in epoch_pairs}

    results: list[dict[str, Any]] = []

    try:
        sn = fm.SeasonalNaiveForecastModel()
        info = sn.calibrate(backtest_pairs, full_lookup)
        p10s, p50s, p90s, actuals = [], [], [], []
        for t, actual in backtest_pairs:
            q10, q50, q90 = sn.predict_quantiles(t, full_lookup)
            if q50 is not None:
                p10s.append(q10); p50s.append(q50); p90s.append(q90); actuals.append(actual)
        pinball = fm.mean_pinball_loss(actuals, p10s, p50s, p90s)
        results.append({
            "model_type": "seasonal_naive", "status": "ok", "model": sn,
            "fit_info": info, "metrics": {"mape": info["mape"], "pinball_loss": pinball, "n_backtest": info["n_backtest"]},
        })
    except Exception as exc:  # noqa: BLE001
        results.append({"model_type": "seasonal_naive", "status": "failed", "error": str(exc), "model": None, "fit_info": None, "metrics": None})

    try:
        gbm = fm.GBMCalendarForecastModel()
        fit_info = gbm.fit(train_pairs)
        bt_info = gbm.calibrate(backtest_pairs)
        p10s, p50s, p90s, actuals = [], [], [], []
        for t, actual in backtest_pairs:
            q10, q50, q90 = gbm.predict_quantiles(t)
            p10s.append(q10); p50s.append(q50); p90s.append(q90); actuals.append(actual)
        pinball = fm.mean_pinball_loss(actuals, p10s, p50s, p90s)
        results.append({
            "model_type": "gbm_calendar", "status": "ok", "model": gbm,
            "fit_info": {**fit_info, **bt_info}, "metrics": {"mape": bt_info["mape"], "pinball_loss": pinball, "n_backtest": bt_info["n_backtest"]},
        })
    except Exception as exc:  # noqa: BLE001
        results.append({"model_type": "gbm_calendar", "status": "failed", "error": str(exc), "model": None, "fit_info": None, "metrics": None})

    try:
        poly = fm.PolynomialCalendarForecastModel()
        fit_info = poly.fit(train_pairs)
        bt_info = poly.calibrate(backtest_pairs)
        p10s, p50s, p90s, actuals = [], [], [], []
        for t, actual in backtest_pairs:
            q10, q50, q90 = poly.predict_quantiles(t)
            p10s.append(q10); p50s.append(q50); p90s.append(q90); actuals.append(actual)
        pinball = fm.mean_pinball_loss(actuals, p10s, p50s, p90s)
        results.append({
            "model_type": "poly_calendar", "status": "ok", "model": poly,
            "fit_info": {**fit_info, **bt_info}, "metrics": {"mape": bt_info["mape"], "pinball_loss": pinball, "n_backtest": bt_info["n_backtest"]},
        })
    except Exception as exc:  # noqa: BLE001
        results.append({"model_type": "poly_calendar", "status": "failed", "error": str(exc), "model": None, "fit_info": None, "metrics": None})

    return results


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    header = f"{'model_type':<18} {'status':<8} {'mape(%)':>9} {'pinball':>9} {'n_backtest':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "ok":
            print(f"{r['model_type']:<18} {'FAILED':<8} {r['error']}")
            continue
        m = r["metrics"]
        mape_str = f"{m['mape']:.3f}" if m["mape"] is not None else "n/a"
        pinball_str = f"{m['pinball_loss']:.3f}" if m["pinball_loss"] is not None else "n/a"
        print(f"{r['model_type']:<18} {'ok':<8} {mape_str:>9} {pinball_str:>9} {m['n_backtest']:>11}")


def save_artifact_and_register(
    cur, result: dict[str, Any], trafo: str, phase: str, trained_at: datetime,
    historical_from: datetime, historical_to: datetime, backtest_days: int,
) -> int:
    version = _version_string(trained_at)
    artifact = result["model"].to_artifact()
    filename = f"{trafo}_{phase}_{result['model_type']}_{version}.json"
    artifact_path = ARTIFACTS_DIR / filename
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    metrics = result["metrics"]
    cur.execute(
        """
        INSERT INTO ml_models (
            trafo, phase, model_type, version, trained_at, healthy_from, healthy_to,
            features, metrics, artifact_path, sigma, mu, is_active, is_frozen_reference, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, FALSE, %s)
        RETURNING id
        """,
        (
            trafo, phase, result["model_type"], version, trained_at, historical_from, historical_to,
            json.dumps({"backtest_days": backtest_days, "fit_info": result["fit_info"]}),
            json.dumps(metrics), str(Path("ml/artifacts") / filename),
            None, None, None,
        ),
    )
    return cur.fetchone()["id"]


def _parse_date(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trafo", required=True, choices=["2", "3", "4"])
    parser.add_argument("--phase", required=True, choices=["R", "S", "T"])
    parser.add_argument("--historical-from", dest="historical_from", required=True)
    parser.add_argument("--historical-to", dest="historical_to", required=True)
    parser.add_argument("--backtest-days", type=int, default=DEFAULT_BACKTEST_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts_from = _parse_date(args.historical_from)
    ts_to = _parse_date(args.historical_to)

    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        hourly_series = fds.build_hourly_load_series(cur, args.trafo, args.phase, ts_from, ts_to)
        log.info("Seri per jam: %d jam (dari %d jam kalender dalam rentang)", len(hourly_series),
                  int((ts_to - ts_from).total_seconds() // 3600))

        results = train_and_backtest(hourly_series, args.backtest_days)
        print_comparison_table(results)

        if args.dry_run:
            log.info("--dry-run: TIDAK menyimpan artefak atau menulis ml_models")
            conn.rollback()
            return

        trained_at = datetime.now(timezone.utc)
        saved_ids = []
        for r in results:
            if r["status"] != "ok":
                continue
            model_id = save_artifact_and_register(cur, r, args.trafo, args.phase, trained_at, ts_from, ts_to, args.backtest_days)
            saved_ids.append((r["model_type"], model_id))
        conn.commit()

    log.info("=" * 60)
    for model_type, model_id in saved_ids:
        log.info("Tersimpan: %s -> ml_models.id=%s (is_active=false)", model_type, model_id)
    log.info("Aktifkan model lewat: POST /api/ml/models/{id}/activate")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
