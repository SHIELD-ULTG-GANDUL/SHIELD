#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ml/evaluate_fault_injection.py — SHIELD Stage 2, Part I.

Validasi SATU baseline model (Part B, ml_models) lewat SYNTHETIC FAULT
INJECTION pada periode "sehat" yang DIPISAH (held-out) dari training --
BUKAN periode training yang sama (supaya hasil deteksi tidak bias oleh
overfitting terhadap data yang sama dipakai melatih model itu).

Fault diinjeksikan sebagai PENAMBAHAN suhu (°C) proporsional terhadap I²
(arus dinormalisasi ke arus PUNCAK pada periode held-out itu sendiri,
"typical peak load") -- ditambahkan ke T_hot_aktual, lalu dijalankan
lewat prediksi model yang SUDAH ADA (tidak melatih ulang apa pun).

Dua detektor diuji, KEDUANYA memakai ulang formula/konfigurasi yang SAMA
dengan bagian lain Stage 2 (bukan nilai duplikat):
  - z-score + median bergulir 30 menit: mu/sigma dari config.py
    (LNTAI_V3_MU/SIGMA, Part A), ambang dari config.LNTAI_V3_THRESHOLDS
    ["caution"].
  - CUSUM: server/explain.py cusum_step() (Part G), parameter dari
    server/explain_config.json.

TIDAK ADA angka acak (RNG) di mana pun dalam injeksi/deteksi -- fault
step/ramp dan kedua detektor murni deterministik dari residual yang
sudah ada, jadi hasil SELALU reproducible tanpa perlu seed eksplisit
(lihat test "results are reproducible with a fixed seed" -- dipenuhi
secara konstruksi, bukan lewat RNG yang di-seed).

Pemakaian:
    python3 ml/evaluate_fault_injection.py --model-id 42 \\
        --held-out-from 2026-09-01 --held-out-to 2026-09-30

    python3 ml/evaluate_fault_injection.py --model-id 42 \\
        --held-out-from 2026-09-01 --held-out-to 2026-09-30 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import dataset as ds  # noqa: E402
from models import load_model_from_artifact  # noqa: E402
from explain import cusum_step, EXPLAIN_CONFIG  # noqa: E402
from settings import load_database_settings  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate-fault-injection")

_CONFIG_PATH = Path(__file__).resolve().parent / "fault_injection_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    FAULT_INJECTION_CONFIG: dict[str, Any] = json.load(_f)

Z_SCORE_THRESHOLD: float = config.LNTAI_V3_THRESHOLDS["caution"]
Z_SCORE_MU: float = config.LNTAI_V3_MU
Z_SCORE_SIGMA: float = config.LNTAI_V3_SIGMA


# ════════════════════════════════════════════════════════════════════════
# Injeksi fault -- murni, tanpa DB, tanpa RNG.
# ════════════════════════════════════════════════════════════════════════

def inject_step_fault(
    currents: list[float], i_peak: float, magnitude_c: float, fault_start_idx: int,
) -> list[float]:
    """extra_dT[i] = magnitude_c * (I[i]/I_peak)^2 untuk i >= fault_start_idx,
    else 0 -- lompatan LANGSUNG ke magnitude penuh (step fault)."""
    if i_peak == 0:
        return [0.0] * len(currents)
    return [
        (magnitude_c * (I / i_peak) ** 2) if i >= fault_start_idx else 0.0
        for i, I in enumerate(currents)
    ]


def inject_ramp_fault(
    currents: list[float], i_peak: float, magnitude_c: float, fault_start_idx: int,
    ramp_duration_samples: int,
) -> list[float]:
    """Sama seperti inject_step_fault() tapi magnitude naik LINEAR dari 0
    ke penuh selama ramp_duration_samples sampel pertama setelah
    fault_start_idx, lalu tetap di magnitude penuh."""
    if i_peak == 0:
        return [0.0] * len(currents)
    out = []
    for i, I in enumerate(currents):
        if i < fault_start_idx:
            out.append(0.0)
            continue
        progress = min(1.0, (i - fault_start_idx) / ramp_duration_samples) if ramp_duration_samples > 0 else 1.0
        out.append(progress * magnitude_c * (I / i_peak) ** 2)
    return out


# ════════════════════════════════════════════════════════════════════════
# Detektor -- murni, tanpa DB. Keduanya memakai ULANG formula/konfigurasi
# dari bagian lain Stage 2 (lihat docstring modul).
# ════════════════════════════════════════════════════════════════════════

def _zscore_series(residuals: list[float], mu: float = Z_SCORE_MU, sigma: float = Z_SCORE_SIGMA) -> list[float]:
    if not sigma:
        return [0.0] * len(residuals)
    return [(r - mu) / sigma for r in residuals]


def detect_zscore_median_first(
    residuals: list[float], window_samples: int, threshold: float = Z_SCORE_THRESHOLD,
) -> Optional[int]:
    """Indeks sampel PERTAMA di mana median bergulir z-score >= threshold.
    None kalau tidak pernah terdeteksi."""
    z = _zscore_series(residuals)
    buf: list[float] = []
    for i, zi in enumerate(z):
        buf.append(zi)
        if len(buf) > window_samples:
            buf.pop(0)
        if len(buf) == window_samples and median(buf) >= threshold:
            return i
    return None


def count_zscore_median_alarms(residuals: list[float], window_samples: int, threshold: float = Z_SCORE_THRESHOLD) -> int:
    """Jumlah episode (BUKAN jumlah sampel) median bergulir >= threshold --
    debounce: satu episode dihitung sekali sampai turun kembali di bawah
    threshold sebelum naik lagi dihitung sebagai episode baru."""
    z = _zscore_series(residuals)
    buf: list[float] = []
    count = 0
    above = False
    for zi in z:
        buf.append(zi)
        if len(buf) > window_samples:
            buf.pop(0)
        if len(buf) == window_samples:
            is_above = median(buf) >= threshold
            if is_above and not above:
                count += 1
            above = is_above
    return count


def detect_cusum_first(residuals: list[float], timestamps: list[float], cfg: dict[str, Any] = EXPLAIN_CONFIG) -> Optional[int]:
    """Indeks sampel PERTAMA di mana CUSUM (server/explain.py) triggered.
    None kalau tidak pernah terdeteksi."""
    state: dict[str, Any] = {}
    for i, (ts, r) in enumerate(zip(timestamps, residuals)):
        state, triggered, _ = cusum_step(state, r, ts, cfg)
        if triggered:
            return i
    return None


def count_cusum_alarms(residuals: list[float], timestamps: list[float], cfg: dict[str, Any] = EXPLAIN_CONFIG) -> int:
    """Jumlah episode CUSUM triggered (debounce: episode baru hanya
    dihitung setelah kembali tidak-triggered lebih dulu)."""
    state: dict[str, Any] = {}
    count = 0
    was_triggered = False
    for ts, r in zip(timestamps, residuals):
        state, triggered, _ = cusum_step(state, r, ts, cfg)
        if triggered and not was_triggered:
            count += 1
        was_triggered = triggered
    return count


# ════════════════════════════════════════════════════════════════════════
# Orkestrasi evaluasi -- murni (terima `rows` sudah dibangun, tidak akses
# DB sendiri) supaya gampang diuji dengan data sintetis.
# ════════════════════════════════════════════════════════════════════════

def evaluate_detector_grid(
    model: Any, held_out_rows: list[dict[str, Any]], cfg: dict[str, Any] = FAULT_INJECTION_CONFIG,
) -> dict[str, Any]:
    """`held_out_rows`: skema SAMA dengan ml/dataset.py build_phase_dataset()
    (ts_epoch, T, current_amps, ambient_c, i2_eff_default_tau, load_roc_1h,
    hour_sin, hour_cos, weekday, ...), TERURUT NAIK waktu."""
    n = len(held_out_rows)
    if n < 10:
        raise ValueError(f"Hanya {n} baris held-out -- terlalu sedikit untuk evaluasi fault injection yang bermakna.")

    currents = [r["current_amps"] for r in held_out_rows]
    timestamps = [r["ts_epoch"] for r in held_out_rows]
    i_peak = max(currents)
    measured_t_hot = [r["T"] for r in held_out_rows]
    predicted = model.predict(held_out_rows)

    sample_interval_s = cfg["nominal_sample_interval_seconds"]
    window_samples = max(1, round(cfg["z_score_rolling_median_minutes"] * 60 / sample_interval_s))

    clean_residuals = [measured_t_hot[i] - predicted[i] for i in range(n)]
    period_weeks = max((timestamps[-1] - timestamps[0]) / (7 * 86400.0), 1e-9)

    false_alarms_per_week = {
        "z_score_median": count_zscore_median_alarms(clean_residuals, window_samples) / period_weeks,
        "cusum": count_cusum_alarms(clean_residuals, timestamps) / period_weeks,
    }

    fault_start_idx = int(n * cfg["fault_start_fraction"])
    fault_start_ts = timestamps[fault_start_idx]
    ramp_duration_samples = max(1, round(cfg["ramp_duration_days"] * 86400 / sample_interval_s))

    fault_results: list[dict[str, Any]] = []
    for fault_type in cfg["fault_types"]:
        for magnitude in cfg["magnitudes_c"]:
            if fault_type == "step":
                extra = inject_step_fault(currents, i_peak, magnitude, fault_start_idx)
            else:
                extra = inject_ramp_fault(currents, i_peak, magnitude, fault_start_idx, ramp_duration_samples)

            injected_residuals = [clean_residuals[i] + extra[i] for i in range(n)]

            det_z = detect_zscore_median_first(injected_residuals, window_samples)
            det_c = detect_cusum_first(injected_residuals, timestamps)

            delay_z = (timestamps[det_z] - fault_start_ts) / 3600.0 if det_z is not None and det_z >= fault_start_idx else None
            delay_c = (timestamps[det_c] - fault_start_ts) / 3600.0 if det_c is not None and det_c >= fault_start_idx else None

            fault_results.append({
                "fault_type": fault_type, "magnitude_c": magnitude,
                "z_score_median": {"detected": delay_z is not None, "detection_delay_hours": delay_z},
                "cusum": {"detected": delay_c is not None, "detection_delay_hours": delay_c},
            })

    min_detectable_magnitude_c: dict[str, dict[str, Optional[float]]] = {}
    for detector_key in ("z_score_median", "cusum"):
        min_detectable_magnitude_c[detector_key] = {}
        for fault_type in cfg["fault_types"]:
            detected = [
                r["magnitude_c"] for r in fault_results
                if r["fault_type"] == fault_type and r[detector_key]["detected"]
            ]
            min_detectable_magnitude_c[detector_key][fault_type] = min(detected) if detected else None

    return {
        "n_samples": n, "fault_start_idx": fault_start_idx,
        "false_alarms_per_week": false_alarms_per_week,
        "fault_results": fault_results,
        "min_detectable_magnitude_c": min_detectable_magnitude_c,
    }


def print_report(results: dict[str, Any]) -> None:
    print(f"n_samples={results['n_samples']}  fault_start_idx={results['fault_start_idx']}")
    print(f"false_alarms_per_week: z_score_median={results['false_alarms_per_week']['z_score_median']:.3f}  "
          f"cusum={results['false_alarms_per_week']['cusum']:.3f}")
    print()
    header = f"{'fault_type':<8} {'magnitude_c':>11} {'z_detected':>11} {'z_delay_h':>10} {'cusum_detected':>14} {'cusum_delay_h':>13}"
    print(header)
    print("-" * len(header))
    for r in results["fault_results"]:
        z, c = r["z_score_median"], r["cusum"]
        z_delay = f"{z['detection_delay_hours']:.1f}" if z["detection_delay_hours"] is not None else "n/a"
        c_delay = f"{c['detection_delay_hours']:.1f}" if c["detection_delay_hours"] is not None else "n/a"
        print(f"{r['fault_type']:<8} {r['magnitude_c']:>11} {str(z['detected']):>11} {z_delay:>10} "
              f"{str(c['detected']):>14} {c_delay:>13}")
    print()
    print("min_detectable_magnitude_c:", results["min_detectable_magnitude_c"])


def _parse_date(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=int, required=True, help="ml_models.id yang akan divalidasi")
    parser.add_argument("--held-out-from", dest="held_out_from", required=True)
    parser.add_argument("--held-out-to", dest="held_out_to", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Cetak laporan, jangan tulis ke ml_models.metrics")
    args = parser.parse_args()

    ts_from = _parse_date(args.held_out_from)
    ts_to = _parse_date(args.held_out_to)

    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT trafo, phase, model_type, artifact_path, metrics FROM ml_models WHERE id = %s", (args.model_id,))
        row = cur.fetchone()
        if row is None:
            log.error("ml_models id=%s tidak ditemukan", args.model_id)
            return

        with open(_ROOT / row["artifact_path"], "r", encoding="utf-8") as f:
            artifact = json.load(f)
        model = load_model_from_artifact(row["model_type"], artifact)

        held_out_rows, dataset_metadata = ds.build_phase_dataset(cur, row["trafo"], row["phase"], ts_from, ts_to)
        log.info("Held-out dataset: %d baris bersih (%s)", len(held_out_rows), json.dumps(dataset_metadata))

        results = evaluate_detector_grid(model, held_out_rows)
        print_report(results)

        if args.dry_run:
            log.info("--dry-run: TIDAK menulis ke ml_models.metrics")
            conn.rollback()
            return

        existing_metrics = row["metrics"] or {}
        existing_metrics["fault_injection"] = results
        cur.execute("UPDATE ml_models SET metrics = %s WHERE id = %s", (json.dumps(existing_metrics), args.model_id))
        conn.commit()

    log.info("ml_models.id=%s metrics diperbarui dengan hasil fault_injection", args.model_id)


if __name__ == "__main__":
    main()
