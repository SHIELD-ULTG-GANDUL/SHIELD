"""server/degradation.py — SHIELD Stage 2, Part H (beta).

Proyeksi Bayesian tren LINEAR R/R_ref mingguan (Part D) per fasa --
konjugat Normal, NUMPY SAJA (tanpa scipy/sklearn, sesuai task brief).
Prior LEMAH (weakly informative, degradation_config.json), σ² noise
di-plug-in dari estimasi OLS atas data itu sendiri (empirical Bayes) --
BUKAN model Normal-Inverse-Gamma penuh dengan σ² sebagai variabel acak
sendiri; disederhanakan dengan sengaja, karena itu label "(beta)" di
judul task ini.

Kalibrasi terhadap resistance_test (Part E, maintenance_events), KALAU
ADA: setiap event resistance_test dikonversi jadi titik (x minggu, y
persen) TAMBAHAN di regresi YANG SAMA, dengan BOBOT lebih tinggi
(weighted least squares, config resistance_test_weight) daripada titik
r_relative_weekly mingguan -- karena resistance_test adalah PENGUKURAN
LANGSUNG (micro-ohmmeter), r_relative_weekly adalah ESTIMASI TIDAK
LANGSUNG dari residual termal (Part D, juga "(beta)"). y_persen = nilai
resistance_uohm event itu dibagi resistance_test PALING AWAL (trafo,
phase) yang sama (titik referensi 100%), dikali 100.

Sesuai task brief tugas #4 (dibaca APA ADANYA, bukan ditafsirkan
longgar): proyeksi HANYA dijalankan kalau DUA syarat berikut TERPENUHI
SEKALIGUS -- (a) jumlah minggu valid >= min_valid_weeks, DAN (b) minimal
ADA SATU event resistance_test untuk (trafo, phase) ini. Kurang dari
salah satu -> status='insufficient_data', tidak ada proyeksi sama
sekali. Ini berarti proyeksi SELALU butuh minimal satu pengukuran
resistance_test riil, bukan hanya data r_relative_weekly -- interpretasi
ini DIAMBIL SECARA EKSPLISIT dari kata "or" di task brief ("fewer than N
valid weeks OR zero resistance tests -> insufficient_data"), dan
dilaporkan jelas ke pengguna (lihat laporan Part H) supaya bisa dikoreksi
kalau yang dimaksud berbeda.

numpy BUKAN dependency wajib untuk modul lain di proyek ini -- kalau
belum terpasang, fit_bayesian_trend()/project_crossing() melempar
RuntimeError yang jelas (sama seperti pola _try_import_gbm_backend() di
ml/models.py untuk LightGBM/scikit-learn), bukan diam-diam gagal.
is_sufficient_data() dan konfigurasi TIDAK butuh numpy sama sekali.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import resistance as rz  # noqa: E402 -- satu sumber crossing_threshold_pct (guidance Part D)

_CONFIG_PATH = Path(__file__).resolve().parent / "degradation_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    DEGRADATION_CONFIG: dict[str, Any] = json.load(_f)

CROSSING_THRESHOLD_PCT: float = (
    DEGRADATION_CONFIG["crossing_threshold_pct"]
    if DEGRADATION_CONFIG["crossing_threshold_pct"] is not None
    else rz.RESISTANCE_CONFIG["guidance"]["fair_max_pct"]
)


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise RuntimeError(
            "numpy tidak terpasang -- jalankan `pip install numpy` sebelum memakai proyeksi "
            "degradasi (Stage 2 Part H). is_sufficient_data() dan muatan degradation_config.json "
            "tidak terpengaruh, hanya fit_bayesian_trend()/project_crossing() yang butuh ini."
        )


def is_sufficient_data(
    n_valid_weeks: int, n_resistance_tests: int, cfg: dict[str, Any] = DEGRADATION_CONFIG,
) -> tuple[bool, Optional[str]]:
    """Lihat docstring modul untuk kenapa KEDUA syarat wajib terpenuhi
    (bukan salah satu saja)."""
    if n_valid_weeks < cfg["min_valid_weeks"]:
        return False, (
            f"Only {n_valid_weeks} valid weekly R/R_ref points (minimum "
            f"{cfg['min_valid_weeks']} required, see Part D r_relative_weekly)."
        )
    if n_resistance_tests == 0:
        return False, (
            "No resistance_test events available to calibrate the projection "
            "(see Part E maintenance_events)."
        )
    return True, None


def fit_bayesian_trend(
    x: list[float], y: list[float], weights: list[float], cfg: dict[str, Any] = DEGRADATION_CONFIG,
) -> dict[str, Any]:
    """Regresi Bayesian linear y = b0 + b1·x, konjugat Normal (σ² known/
    plug-in dari OLS residual variance atas data ini sendiri), WEIGHTED
    (weighted least squares -- baris dikalikan √weight sebelum fit, setara
    WLS untuk σ² bersama/homoskedastik atas bobot). Return
    {"beta_mean": np.ndarray[2], "beta_cov": np.ndarray[2,2], "sigma2": float, "n": int}."""
    _require_numpy()

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    n = len(x_arr)

    sw = np.sqrt(w_arr)
    X = np.column_stack([np.ones(n), x_arr]) * sw[:, None]
    yw = y_arr * sw

    if n > 2:
        beta_ols, *_ = np.linalg.lstsq(X, yw, rcond=None)
        residuals = yw - X @ beta_ols
        dof = max(n - 2, 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)
        if sigma2 <= 0:
            sigma2 = cfg["prior"]["sigma2_fallback"]
    else:
        sigma2 = cfg["prior"]["sigma2_fallback"]

    beta_0 = np.array(cfg["prior"]["beta_0"], dtype=float)
    Sigma_0 = np.diag(np.array(cfg["prior"]["beta_var_0"], dtype=float))
    Lambda_0 = np.linalg.inv(Sigma_0)

    Lambda_n = Lambda_0 + (X.T @ X) / sigma2
    Sigma_n = np.linalg.inv(Lambda_n)
    beta_n = Sigma_n @ (Lambda_0 @ beta_0 + (X.T @ yw) / sigma2)

    return {"beta_mean": beta_n, "beta_cov": Sigma_n, "sigma2": sigma2, "n": n}


def project_crossing(
    fit: dict[str, Any], threshold_pct: float, horizon_x: float, mc_samples: int,
    rng: Optional[Any] = None,
) -> dict[str, Any]:
    """Monte Carlo lewat posterior (intercept, slope) -- untuk tiap sampel
    garis, cari x di mana garis itu melewati `threshold_pct` (HANYA kalau
    slope>0 DAN 0 < x_cross <= horizon_x, selain itu dianggap "tidak
    melewati" untuk sampel itu). P10/P50/P90 dari kuantil empiris x_cross
    di antara sampel yang BENAR-BENAR melewati. `rng` opsional (numpy
    Generator) -- kalau None, dibuat baru TANPA seed (acak asli); test
    memasok rng berseed untuk hasil deterministik."""
    _require_numpy()

    if rng is None:
        rng = np.random.default_rng()

    samples = rng.multivariate_normal(fit["beta_mean"], fit["beta_cov"], size=mc_samples)
    b0, b1 = samples[:, 0], samples[:, 1]

    crossing_x = np.full(mc_samples, np.nan)
    positive_slope = b1 > 1e-9
    crossing_x[positive_slope] = (threshold_pct - b0[positive_slope]) / b1[positive_slope]

    valid = positive_slope & (crossing_x > 0) & (crossing_x <= horizon_x)
    n_crossing = int(np.sum(valid))

    if n_crossing == 0:
        return {"status": "no_crossing_projected", "n_crossing_samples": 0, "n_total_samples": mc_samples}

    x_p10, x_p50, x_p90 = np.percentile(crossing_x[valid], [10, 50, 90])
    return {
        "status": "ok", "x_p10": float(x_p10), "x_p50": float(x_p50), "x_p90": float(x_p90),
        "n_crossing_samples": n_crossing, "n_total_samples": mc_samples,
    }


# ════════════════════════════════════════════════════════════════════════
# Orkestrasi (butuh DB cursor) -- tidak ada job/tabel terpisah (Part H
# TIDAK meminta tabel baru), dihitung LIVE setiap GET /api/degradation/projection.
# ════════════════════════════════════════════════════════════════════════

def fetch_weekly_points(cur, trafo: str, phase: str) -> list[tuple[date, float]]:
    """[(week_start, r_ratio_median_pct), ...] HANYA baris status='ok' (Part D)."""
    cur.execute(
        "SELECT week_start, r_ratio_median_pct FROM r_relative_weekly "
        "WHERE trafo = %s AND phase = %s AND status = 'ok' AND r_ratio_median_pct IS NOT NULL "
        "ORDER BY week_start ASC",
        (trafo, phase),
    )
    rows = cur.fetchall()
    return [(r["week_start"], r["r_ratio_median_pct"]) for r in rows]


def fetch_resistance_test_points(cur, trafo: str, phase: str) -> list[tuple[datetime, float]]:
    """[(event_ts, resistance_uohm), ...] event_type='resistance_test',
    terurut naik waktu -- phase IS NULL di event berarti berlaku semua
    fasa trafo itu (ikut dicocokkan, sama seperti Part E/C)."""
    cur.execute(
        "SELECT event_ts, resistance_uohm FROM maintenance_events "
        "WHERE trafo = %s AND event_type = 'resistance_test' AND resistance_uohm IS NOT NULL "
        "AND (phase = %s OR phase IS NULL) ORDER BY event_ts ASC",
        (trafo, phase),
    )
    rows = cur.fetchall()
    return [(r["event_ts"], r["resistance_uohm"]) for r in rows]


def build_projection(
    cur, trafo: str, phase: str, cfg: dict[str, Any] = DEGRADATION_CONFIG, rng: Optional[Any] = None,
) -> dict[str, Any]:
    weekly_points = fetch_weekly_points(cur, trafo, phase)
    resistance_points = fetch_resistance_test_points(cur, trafo, phase)

    sufficient, reason = is_sufficient_data(len(weekly_points), len(resistance_points), cfg)
    if not sufficient:
        return {
            "status": "insufficient_data", "reason": reason,
            "n_valid_weeks": len(weekly_points), "n_resistance_tests": len(resistance_points),
        }

    _require_numpy()

    reference_date = weekly_points[0][0]
    x_weekly = [(w - reference_date).days / 7.0 for w, _ in weekly_points]
    y_weekly = [v for _, v in weekly_points]
    weights_weekly = [1.0] * len(weekly_points)

    r_ref_uohm = resistance_points[0][1]
    x_resistance = [(ts.date() - reference_date).days / 7.0 for ts, _ in resistance_points]
    y_resistance = [(val / r_ref_uohm) * 100.0 for _, val in resistance_points]
    weights_resistance = [cfg["resistance_test_weight"]] * len(resistance_points)

    x = x_weekly + x_resistance
    y = y_weekly + y_resistance
    weights = weights_weekly + weights_resistance

    fit = fit_bayesian_trend(x, y, weights, cfg)
    projection = project_crossing(fit, CROSSING_THRESHOLD_PCT, cfg["horizon_weeks"], cfg["mc_samples"], rng)

    result: dict[str, Any] = {
        "status": "ok",
        "n_valid_weeks": len(weekly_points), "n_resistance_tests": len(resistance_points),
        "calibrated_with_resistance_test": len(resistance_points) > 0,
        "threshold_pct": CROSSING_THRESHOLD_PCT,
        "beta_mean": fit["beta_mean"].tolist(),
        "reference_date": reference_date.isoformat(),
    }

    if projection["status"] == "no_crossing_projected":
        result["crossing"] = "No crossing projected within horizon"
        return result

    result["crossing_date_p10"] = (reference_date + timedelta(weeks=projection["x_p10"])).isoformat()
    result["crossing_date_p50"] = (reference_date + timedelta(weeks=projection["x_p50"])).isoformat()
    result["crossing_date_p90"] = (reference_date + timedelta(weeks=projection["x_p90"])).isoformat()
    result["n_crossing_samples"] = projection["n_crossing_samples"]
    result["n_total_samples"] = projection["n_total_samples"]
    return result
