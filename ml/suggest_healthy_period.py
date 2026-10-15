#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ml/suggest_healthy_period.py — Go-live helper (READ-ONLY, no writes
anywhere: no artifact, no ml_models row, no config file touched).

For each (trafo, phase), RANKS candidate 28-day windows (default; see
--window-days/--step-days) since 2026-06-02 (default; see --since) as
candidate healthy_from/healthy_to arguments for ml/train_baseline.py.
Does NOT choose for you -- prints the top 3 windows per (trafo, phase)
with their raw metrics, you decide. A window ranking well here is a
PROVISIONAL suggestion, not a confirmed healthy period -- it has not
been checked against a contact-resistance test (Part E) and should be
trained with ml/train_baseline.py's --provisional/--provisional-reason
flags, never silently treated as final.

Ranking metrics per window (all computed from data ALREADY in the DB,
nothing invented):
  - completeness: temperature samples actually present / samples
    expected at the nominal sampling interval (data_quality_config.json).
  - flagged_share: fraction of those samples carrying a quality_flags
    bit in ml_config.json's dataset.drop_flags (MISSING/OUT_OF_RANGE/
    STUCK/ZERO_READING) -- the SAME flags ml/dataset.py itself drops
    when building a training dataset, not a separately invented list.
  - load_range_coverage: this window's peak current / this phase's
    peak current over its ENTIRE available history -- a window that
    never saw peak load is not representative for a baseline meant to
    predict healthy behavior AT peak load too.
  - residual_stability: 1 / (1 + MAD(residual)), residual computed
    against the LEGACY v3 physics baseline (lntai_v3_config.json
    baseline A/B/I_ref/ambient_c -- the SAME uncalibrated baseline the
    live LNTAI v3 panel already compares against, not a new one) over
    the window. A window with wildly unstable residual likely contains
    a real anomaly/fault, not just thermal noise.

composite_score is a SIMPLE weighted average of the four (0.3/0.3/0.2/0.2)
for ORDERING ONLY -- read all four raw metrics yourself before picking a
window, don't trust the single number.

Run:
    cd ml && python3 suggest_healthy_period.py
    cd ml && python3 suggest_healthy_period.py --trafo 2 --window-days 28 --step-days 7
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import data_quality as dq  # noqa: E402
import dataset as ds  # noqa: E402
from settings import load_database_settings  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

DEFAULT_SINCE = datetime(2026, 6, 2, tzinfo=timezone.utc)
DEFAULT_WINDOW_DAYS = 28
DEFAULT_STEP_DAYS = 7
TRAFOS = ("2", "3", "4")

DROP_FLAGS: list[str] = ds.ML_CONFIG["dataset"]["drop_flags"]
NOMINAL_INTERVAL_S: float = dq.DATA_QUALITY_CONFIG["nominal_sample_interval_seconds"]

PROVISIONAL_NOTICE = (
    "Every window below is a PROVISIONAL suggestion only -- it has NOT been "
    "confirmed by a contact-resistance test (Part E). If you train from one "
    "of these windows, use ml/train_baseline.py --provisional --provisional-reason "
    "\"...\" so the UI shows a 'Provisional baseline' badge until confirmed."
)


def _hotspot_c(row: dict[str, Any], phase: str) -> Optional[float]:
    atas = row.get(f"{phase.lower()}_atas")
    bawah = row.get(f"{phase.lower()}_bawah")
    vals = [v for v in (atas, bawah) if v is not None]
    return max(vals) if vals else None


def _legacy_residual_series(
    temp_rows: list[dict[str, Any]], paired_load: list[Optional[dict[str, Any]]], phase: str,
) -> list[float]:
    """Residual (°C) terhadap baseline fisika legacy v3 (lntai_v3_config.json),
    SATU-SATUNYA baseline yang dipakai di sini -- lihat docstring modul."""
    current_col = ds._PHASE_CURRENT_COLUMN[phase]
    ts_epoch = [r["timestamp"].timestamp() for r in temp_rows]
    currents = [lr[current_col] if lr else None for lr in paired_load]

    i2_eff_series = ds.compute_i2_eff_series(
        ts_epoch, currents, config.LNTAI_V2_TAU_MINUTES, config.LNTAI_V2_MAX_GAP_MINUTES,
    )

    residuals: list[float] = []
    for row, i2_eff in zip(temp_rows, i2_eff_series):
        if i2_eff is None:
            continue
        hotspot_c = _hotspot_c(row, phase)
        if hotspot_c is None:
            continue
        i_eff_norm = sqrt(i2_eff) / config.LNTAI_V3_I_REF
        delta_t_pred = config.LNTAI_V3_A * (i_eff_norm ** 2) + config.LNTAI_V3_B
        delta_t_real = hotspot_c - config.LNTAI_V3_AMBIENT_C
        residuals.append(delta_t_real - delta_t_pred)
    return residuals


def _mad(values: list[float]) -> Optional[float]:
    if not values:
        return None
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def _window_metrics(
    temp_rows_window: list[dict[str, Any]], paired_load_window: list[Optional[dict[str, Any]]],
    phase: str, window_days: int, overall_max_current: Optional[float],
) -> dict[str, Any]:
    n = len(temp_rows_window)
    expected_n = max(1, int((window_days * 86400) / NOMINAL_INTERVAL_S))
    completeness = min(1.0, n / expected_n)

    if n == 0:
        flagged_share = 1.0
    else:
        flagged = sum(1 for r in temp_rows_window if ds._flags_to_drop(r.get("quality_flags") or 0, DROP_FLAGS))
        flagged_share = flagged / n

    current_col = ds._PHASE_CURRENT_COLUMN[phase]
    currents_in_window = [lr[current_col] for lr in paired_load_window if lr and lr.get(current_col) is not None]
    window_max_current = max(currents_in_window) if currents_in_window else None
    if window_max_current is not None and overall_max_current:
        load_range_coverage = min(1.0, window_max_current / overall_max_current)
    else:
        load_range_coverage = 0.0

    residuals = _legacy_residual_series(temp_rows_window, paired_load_window, phase)
    mad = _mad(residuals)
    residual_stability = 1.0 / (1.0 + mad) if mad is not None else 0.0
    n_residual_samples = len(residuals)

    composite_score = (
        0.3 * completeness + 0.3 * (1.0 - flagged_share) + 0.2 * load_range_coverage + 0.2 * residual_stability
    )

    return {
        "n_samples": n, "completeness": completeness, "flagged_share": flagged_share,
        "window_max_current_a": window_max_current, "load_range_coverage": load_range_coverage,
        "residual_mad_c": mad, "residual_stability": residual_stability, "n_residual_samples": n_residual_samples,
        "composite_score": composite_score,
    }


def suggest_for_phase(
    cur, trafo: str, phase: str, since: datetime, until: datetime, window_days: int, step_days: int,
) -> list[dict[str, Any]]:
    """Satu fetch untuk SELURUH rentang [since, until], lalu iterasi
    jendela kandidat in-memory (jauh lebih cepat daripada query ulang per
    jendela)."""
    temp_rows = ds._fetch_temperature_rows(cur, trafo, since, until)
    load_rows = ds._fetch_load_rows(cur, trafo, since, until)
    paired = ds._pair_nearest(temp_rows, load_rows, config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES)

    current_col = ds._PHASE_CURRENT_COLUMN[phase]
    all_currents = [lr[current_col] for lr in paired if lr and lr.get(current_col) is not None]
    overall_max_current = max(all_currents) if all_currents else None

    candidates: list[dict[str, Any]] = []
    window = timedelta(days=window_days)
    step = timedelta(days=step_days)
    start = since
    while start + window <= until:
        end = start + window
        idx_rows = [(r, p) for r, p in zip(temp_rows, paired) if start <= r["timestamp"] <= end]
        window_temp_rows = [r for r, _ in idx_rows]
        window_paired = [p for _, p in idx_rows]

        metrics = _window_metrics(window_temp_rows, window_paired, phase, window_days, overall_max_current)
        candidates.append({
            "trafo": trafo, "phase": phase,
            "healthy_from": start.date().isoformat(), "healthy_to": end.date().isoformat(),
            **metrics,
        })
        start += step

    candidates.sort(key=lambda c: c["composite_score"], reverse=True)
    return candidates[:3]


def _print_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        print("  (no candidate windows found in range -- not enough history yet)")
        return
    for i, c in enumerate(candidates, start=1):
        print(
            f"  #{i}  {c['healthy_from']} -> {c['healthy_to']}  "
            f"composite={c['composite_score']:.3f}  "
            f"n={c['n_samples']}  completeness={c['completeness']*100:.1f}%  "
            f"flagged={c['flagged_share']*100:.1f}%  "
            f"load_coverage={c['load_range_coverage']*100:.1f}% (peak={c['window_max_current_a']})  "
            f"residual_MAD={('%.3f' % c['residual_mad_c']) if c['residual_mad_c'] is not None else 'n/a'}°C "
            f"(n={c['n_residual_samples']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trafo", choices=["2", "3", "4"], default=None, help="Default: all 3 trafos")
    parser.add_argument("--since", default=DEFAULT_SINCE.date().isoformat(), help="ISO8601 date, default 2026-06-02")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    args = parser.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    until = datetime.now(timezone.utc)
    trafos = [args.trafo] if args.trafo else list(TRAFOS)

    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)

    print("=" * 72)
    print("ml/suggest_healthy_period.py -- READ-ONLY, no writes.")
    print(PROVISIONAL_NOTICE)
    print(f"Scanning {since.date()} .. {until.date()}, window={args.window_days}d, step={args.step_days}d")
    print("=" * 72)

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for trafo in trafos:
            for phase in ds.PHASES:
                print(f"\n--- Trafo {trafo}, Phase {phase} ---")
                candidates = suggest_for_phase(cur, trafo, phase, since, until, args.window_days, args.step_days)
                _print_candidates(candidates)

    print("\n" + "=" * 72)
    print("No window was chosen automatically. You decide, then run e.g.:")
    print("  python3 train_baseline.py --trafo 2 --phase R \\")
    print("      --healthy-from <chosen healthy_from> --healthy-to <chosen healthy_to> \\")
    print("      --dry-run --provisional --provisional-reason \"Suggested by ml/suggest_healthy_period.py, not yet confirmed by a contact-resistance test.\"")
    print("=" * 72)


if __name__ == "__main__":
    main()
