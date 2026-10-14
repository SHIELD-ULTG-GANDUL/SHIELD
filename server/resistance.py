"""server/resistance.py — SHIELD Stage 2, Part D (beta).

Estimasi DRIFT resistansi kontak RELATIF, dari residual termal terhadap
SATU model baseline BEKU (ml_models.is_frozen_reference=true, lihat Part
B/E) -- BUKAN pengganti resistance_test aktual (micro-ohmmeter, dicatat
manual lewat maintenance_events, Part E). Label "(beta)" di judul task
ini disengaja: formulanya pendekatan fisis orde-1, belum divalidasi
silang terhadap resistance_test riil.

    R/R_ref ≈ 1 + Residual_ref / (A_ref · (I_eff/I_ref)²)

Residual_ref = T_hot_aktual − prediksi_absolut_model_beku (ambient yang
dipakai untuk prediksi SECARA ALJABAR saling meniadakan di pengurangan
ini -- lihat server/resistance_job.py untuk kenapa nilai ambient yang
dipakai tidak relevan terhadap hasil akhir).

A_ref: koefisien "A" milik model beku itu sendiri (physics_quadratic/
physics_rc/komponen physics_rc di dalam hybrid). Untuk model beku yang
BUKAN berbasis fisika (ml_gbm, tidak punya koefisien A sama sekali),
A_ref "dipinjam" dari model physics_rc LAIN yang terdaftar untuk
(trafo, phase) yang sama (apa pun status aktif/bekunya) -- lihat
server/resistance_job.py. Kalau tidak ada physics_rc terdaftar sama
sekali, Part D tidak bisa dihitung untuk kombinasi itu (status
'no_physics_rc_available', bukan dikarang).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_CONFIG_PATH = Path(__file__).resolve().parent / "resistance_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    RESISTANCE_CONFIG: dict[str, Any] = json.load(_f)

GUIDANCE_CITATION: str = RESISTANCE_CONFIG["guidance"]["_citation"]


def is_low_load(current_amps: Optional[float], i_rated: float, cfg: dict[str, Any] = RESISTANCE_CONFIG) -> bool:
    """True kalau sampel ini HARUS dibuang karena beban terlalu rendah
    (lihat _min_load_fraction_note di resistance_config.json). current_amps
    None (data hilang) diperlakukan sebagai low-load (dibuang), bukan
    diam-diam dilewatkan."""
    if current_amps is None:
        return True
    threshold = cfg["min_load_fraction"] * i_rated
    return current_amps < threshold


def is_unsteady(di_dt_a_per_min: Optional[float], cfg: dict[str, Any] = RESISTANCE_CONFIG) -> bool:
    """True kalau beban dianggap TIDAK stabil (atau tidak bisa dipastikan
    stabil -- None diperlakukan sebagai tidak stabil, dibuang, bukan
    diasumsikan stabil)."""
    if di_dt_a_per_min is None:
        return True
    return abs(di_dt_a_per_min) > cfg["steady_load"]["max_abs_di_dt_a_per_min"]


def estimate_r_ratio(residual_c: float, a_ref: float, i_eff_norm: float) -> Optional[float]:
    """R/R_ref ≈ 1 + Residual_ref / (A_ref · i_eff_norm²). None kalau
    denominator nol (i_eff_norm=0, mis. arus persis nol -- walau
    semestinya sudah tersaring is_low_load() sebelum sampai sini)."""
    denom = a_ref * (i_eff_norm ** 2)
    if denom == 0:
        return None
    return 1.0 + residual_c / denom


def _percentile(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = p * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def weekly_stats(ratios_pct: list[float]) -> dict[str, Optional[float]]:
    """Median + IQR (Q3-Q1) atas satu daftar rasio R/R_ref (dalam PERSEN,
    sudah dikali 100 oleh caller) untuk satu minggu."""
    if not ratios_pct:
        return {"median": None, "q1": None, "q3": None, "iqr": None}
    s = sorted(ratios_pct)
    q1 = _percentile(s, 0.25)
    q3 = _percentile(s, 0.75)
    return {"median": _percentile(s, 0.5), "q1": q1, "q3": q3, "iqr": q3 - q1}


def phase_relative_ratio(phase_median_pct: Optional[float], all_phase_medians_pct: list[float]) -> Optional[float]:
    """Rasio SATU fasa terhadap median ketiga fasa minggu yang sama
    (dalam persen juga) -- menonjolkan fasa mana yang menyimpang RELATIF
    terhadap saudaranya, menghilangkan efek bersama (mis. ambient/beban
    trafo keseluruhan) yang memengaruhi ketiga fasa sama rata. None kalau
    fasa ini sendiri tidak punya median (insufficient_data dll.), atau
    kurang dari semua fasa punya median untuk minggu itu."""
    if phase_median_pct is None or len(all_phase_medians_pct) < 3:
        return None
    baseline = sorted(all_phase_medians_pct)[1]  # median dari 3 nilai = nilai tengah
    if baseline == 0:
        return None
    return (phase_median_pct / baseline) * 100.0


def guidance_label(ratio_pct: Optional[float], cfg: dict[str, Any] = RESISTANCE_CONFIG) -> Optional[str]:
    """'good' / 'fair' / 'poor' sesuai kriteria KEMA (lihat GUIDANCE_CITATION).
    None kalau ratio_pct None (tidak ada data)."""
    if ratio_pct is None:
        return None
    if ratio_pct < cfg["guidance"]["good_max_pct"]:
        return "good"
    if ratio_pct <= cfg["guidance"]["fair_max_pct"]:
        return "fair"
    return "poor"
