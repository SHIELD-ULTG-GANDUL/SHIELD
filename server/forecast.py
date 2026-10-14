"""server/forecast.py — SHIELD Stage 2, Part F.

Menghubungkan forecast beban (ml/forecast_models.py, dijalankan
server/forecast_job.py) dengan prediksi suhu hotspot dan skor LNTAI,
memakai baseline AKTIF (Part B -- model "adaptif", BUKAN referensi beku
Part D/E) + drift resistansi relatif TERKINI (Part D, kalau tersedia):

    healthy ΔT = A·(I_forecast/I_ref)² + B           [baseline aktif]
    excess     = (R/R_ref − 1)·A_ref·(I_forecast/I_ref)²
                 ATAU median residual TERKINI kalau R/R_ref tidak
                 tersedia/tidak 'ok' (lihat compute_excess(), method
                 dicatat eksplisit -- lihat Stage 2 aturan "fallback
                 method labelled").
    LNTAI_pred = (excess − μ) / σ                     [μ/σ v3 legacy, config.py]
    t_hot_pred = ambient_default + healthy ΔT + excess

μ/σ DISENGAJA memakai config.LNTAI_V3_MU/SIGMA (bukan parameter baru) --
satu sumber yang sama dipakai LNTAI v3 (Part A) untuk menormalisasi
residual jadi skor, supaya LNTAI_pred (forecast) dan LNTAI v3 (realtime)
berada di skala yang SAMA dan bisa dibandingkan langsung.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parent / "forecast_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    FORECAST_CONFIG: dict[str, Any] = json.load(_f)

LEVEL_RANK: dict[str, int] = {"Normal": 0, "Caution": 1, "Alert": 2, "Critical": 3}


def compute_healthy_delta_t(a: float, b: float, i_ref: float, i_forecast_amps: float) -> float:
    """A·(I_forecast/I_ref)² + B -- fisika murni pada SATU titik beban
    forecast (bukan I_eff dengan filter lag termal: forecast per jam
    tidak punya riwayat sub-jam untuk difilter, I_forecast dianggap
    representatif untuk jam itu langsung)."""
    if i_ref == 0:
        return b
    return a * (i_forecast_amps / i_ref) ** 2 + b


def compute_excess(
    r_over_r_ref: Optional[float], a_ref: Optional[float], i_ref: Optional[float],
    i_forecast_amps: float, recent_residual_median: Optional[float],
) -> tuple[Optional[float], str]:
    """Return (excess, method). method='r_relative' kalau r_over_r_ref
    (dan a_ref/i_ref pasangannya dari Part D) tersedia DAN valid;
    'recent_residual_median' kalau tidak -- dipakai APA ADANYA sebagai
    excess (tanpa disesuaikan beban forecast), didokumentasikan sebagai
    perkiraan LEBIH KASAR daripada jalur r_relative. 'unavailable' kalau
    KEDUANYA tidak ada -- excess=None, caller tidak bisa menghitung
    LNTAI_pred/t_hot_pred sama sekali untuk titik ini."""
    if r_over_r_ref is not None and a_ref is not None and i_ref:
        excess = (r_over_r_ref - 1.0) * a_ref * (i_forecast_amps / i_ref) ** 2
        return excess, "r_relative"
    if recent_residual_median is not None:
        return recent_residual_median, "recent_residual_median"
    return None, "unavailable"


def compute_lntai_pred(
    excess: Optional[float], mu: float = config.LNTAI_V3_MU, sigma: float = config.LNTAI_V3_SIGMA,
) -> Optional[float]:
    if excess is None or not sigma:
        return None
    return (excess - mu) / sigma


def compute_t_hot_pred(
    healthy_delta_t: float, excess: Optional[float], ambient_c: float = config.LNTAI_V3_AMBIENT_C,
) -> Optional[float]:
    if excess is None:
        return None
    return ambient_c + healthy_delta_t + excess


def compute_lntai_quantiles(
    load_p10: Optional[float], load_p50: Optional[float], load_p90: Optional[float],
    r_over_r_ref: Optional[float], a_ref: Optional[float], a_ref_i_ref: Optional[float],
    recent_residual_median: Optional[float],
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    """LNTAI_pred untuk P10/P50/P90 beban -- DIHITUNG TERPISAH per kuantil
    beban lalu DIURUTKAN NAIK (bukan dipetakan langsung P10_beban->P10_LNTAI
    dst.) supaya P10<=P50<=P90 TERJAMIN secara konstruksi: excess sebagai
    fungsi beban BISA naik atau turun tergantung tanda (R/R_ref − 1) --
    kalau negatif (R/R_ref<1, jarang tapi mungkin dari noise), LNTAI_pred
    justru TURUN seiring beban naik, yang akan MEMBALIK urutan P10/P90
    kalau dipetakan langsung dari urutan beban."""
    raw: list[Optional[float]] = []
    method = "unavailable"
    for load in (load_p10, load_p50, load_p90):
        if load is None:
            raw.append(None)
            continue
        excess, method = compute_excess(r_over_r_ref, a_ref, a_ref_i_ref, load, recent_residual_median)
        raw.append(compute_lntai_pred(excess))

    valid = [v for v in raw if v is not None]
    if len(valid) < 3:
        return None, None, None, method
    s = sorted(valid)
    return s[0], s[1], s[2], method


def _level_of(score: Optional[float], thresholds: dict[str, float]) -> Optional[str]:
    if score is None:
        return None
    if score >= thresholds["critical"]:
        return "Critical"
    if score >= thresholds["alert"]:
        return "Alert"
    if score >= thresholds["caution"]:
        return "Caution"
    return "Normal"


def find_level_crossing(
    forecast_points: list[dict[str, Any]], thresholds: dict[str, float] = config.LNTAI_V3_THRESHOLDS,
) -> Optional[dict[str, Any]]:
    """Titik PERTAMA dalam `forecast_points` (terurut waktu naik, tiap
    dict punya key 'lntai_pred_p50') di mana level NAIK dibanding titik
    sebelumnya (mis. Normal->Caution, Caution->Alert) -- bukan setiap
    titik yang levelnya tinggi, HANYA crossing pertamanya. None kalau
    tidak ada crossing (level stabil atau turun terus) dalam horizon ini."""
    prev_level: Optional[str] = None
    for i, point in enumerate(forecast_points):
        cur_level = _level_of(point.get("lntai_pred_p50"), thresholds)
        if cur_level is None:
            continue
        if prev_level is not None and LEVEL_RANK[cur_level] > LEVEL_RANK[prev_level]:
            return {
                "crossed_at_index": i, "from_level": prev_level, "to_level": cur_level,
                "point": point,
            }
        prev_level = cur_level
    return None


def suggested_action(level: str, cfg: dict[str, Any] = FORECAST_CONFIG) -> Optional[str]:
    return cfg["suggested_actions"].get(level)


def compare_aging_rate(
    last_7d_total_eq_days: Optional[float], last_30d_daily_avg_eq_days: Optional[float],
) -> Optional[dict[str, float]]:
    """Bandingkan laju penuaan (Part C) 7 hari TERAKHIR vs rata-rata 30
    hari -- None kalau salah satu input tidak tersedia (mis. aging belum
    terkonfigurasi, lihat server/aging.py is_configured())."""
    if last_7d_total_eq_days is None or last_30d_daily_avg_eq_days is None or last_30d_daily_avg_eq_days == 0:
        return None
    last_7d_daily_avg = last_7d_total_eq_days / 7.0
    return {
        "last_7d_daily_avg_eq_days": last_7d_daily_avg,
        "last_30d_daily_avg_eq_days": last_30d_daily_avg_eq_days,
        "ratio": last_7d_daily_avg / last_30d_daily_avg_eq_days,
    }


def build_warning(
    crossing: dict[str, Any], aging_comparison: Optional[dict[str, float]],
) -> dict[str, Any]:
    point = crossing["point"]
    return {
        "triggered": True,
        "from_level": crossing["from_level"],
        "to_level": crossing["to_level"],
        "target_ts": point.get("target_ts"),
        "expected_peak_load_a": point.get("load_p50_a"),
        "predicted_hotspot_c": point.get("t_hot_pred_c"),
        "lntai_pred_interval": [point.get("lntai_pred_p10"), point.get("lntai_pred_p50"), point.get("lntai_pred_p90")],
        "aging_f_pred_at_peak": point.get("aging_f_pred"),
        "aging_rate_comparison": aging_comparison,
        "suggested_action": suggested_action(crossing["to_level"]),
    }
