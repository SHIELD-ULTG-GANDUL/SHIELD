"""server/aging.py — SHIELD Stage 2, Part C.

Faktor percepatan penuaan Arrhenius, RELATIF terhadap satu suhu referensi
(bukan model usia absolut):

    F(T_hot) = exp( (Ea/k_B) * (1/T_ref_K - 1/T_hot_K) )          [Kelvin, internal]

F=1 ketika T_hot == T_ref (laju penuaan sama seperti referensi). F>1
berarti lebih cepat dari referensi, F<1 lebih lambat -- BUKAN usia dalam
satuan absolut apa pun.

EqAge (usia ekivalen, satuan HARI) = integral F(t) dt sejak event reset
terakhir -- usia "efektif" seandainya komponen selalu persis berada di
T_ref, dipakai untuk membandingkan laju penuaan relatif antar periode/
trafo, BUKAN estimasi sisa umur pakai.

CAKUPAN: HANYA sambungan kontak finger (hotspot yang diukur sensor Acrel
ATP007) -- BUKAN interrupter vakum atau mekanisme penggerak CB. Lihat
AGING_SCOPE_TEXT, WAJIB disertakan di setiap respons API yang menampilkan
angka dari modul ini (lihat server/api.py endpoint /api/aging/*).

ea_ev/t_ref_c di aging_config.json defaultnya NULL (lihat TODO(calibration)
di file itu) -- selama NULL, is_configured() False dan semua fungsi di
sini yang bergantung padanya mengembalikan None/status "not_configured",
TIDAK PERNAH mengarang nilai Ea/T_ref sendiri.
"""

from __future__ import annotations

import json
from math import exp
from pathlib import Path
from typing import Any, Optional

_CONFIG_PATH = Path(__file__).resolve().parent / "aging_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    AGING_CONFIG: dict[str, Any] = json.load(_f)

# Konstanta FISIKA universal (CODATA 2018, Boltzmann dalam eV/K) -- BUKAN
# parameter kalibrasi, aman di-hardcode, sama untuk semua material/trafo.
KB_EV_PER_KELVIN: float = 8.617333262e-5

AGING_SCOPE_TEXT: str = (
    "Covers the contact-finger connection only — not the vacuum interrupter "
    "or operating mechanism."
)


def is_configured(cfg: dict[str, Any] = AGING_CONFIG) -> bool:
    return cfg.get("ea_ev") is not None and cfg.get("t_ref_c") is not None


def celsius_to_kelvin(t_c: float) -> float:
    return t_c + 273.15


def arrhenius_factor(t_hot_c: float, cfg: dict[str, Any] = AGING_CONFIG) -> Optional[float]:
    """F(T_hot) pada SATU titik waktu. None kalau belum terkonfigurasi
    (lihat is_configured()) -- caller HARUS memeriksa None, bukan
    mengasumsikan nilai default apa pun."""
    if not is_configured(cfg):
        return None
    ea_ev = cfg["ea_ev"]
    t_ref_k = celsius_to_kelvin(cfg["t_ref_c"])
    t_hot_k = celsius_to_kelvin(t_hot_c)
    exponent = (ea_ev / KB_EV_PER_KELVIN) * (1.0 / t_ref_k - 1.0 / t_hot_k)
    return exp(exponent)


def equivalent_days_increment(f_avg: float, delta_seconds: float) -> float:
    """Kontribusi satu interval waktu ke EqAge, dalam HARI ekivalen --
    F_rata-rata-interval * Δt, dikonversi dari detik ke hari."""
    return f_avg * (delta_seconds / 86400.0)


def integrate_eq_days(
    samples: list[tuple[float, float]], cfg: dict[str, Any] = AGING_CONFIG,
) -> dict[str, Any]:
    """Integral Arrhenius atas satu seri waktu (biasanya satu hari).

    `samples`: [(ts_epoch, t_hot_c), ...] TERURUT NAIK waktu, SATU fasa.
    Integrasi trapesium antar sampel BERTURUTAN; interval dengan jarak
    waktu > cfg["max_gap_minutes_for_integration"] DIKECUALIKAN dari
    integral (bukan diintegralkan dengan asumsi F konstan selama gap
    yang tidak diketahui kondisinya).

    Return dict:
      status: "not_configured" | "ok"
      f_mean, f_peak: rata-rata/puncak F pada sampel yang ada (None kalau
        tidak ada sampel sama sekali)
      t_hot_mean_c, t_hot_peak_c: suhu hotspot mentah untuk konteks UI
      eq_days: kontribusi HARI ekivalen dari seri INI SAJA (bukan kumulatif
        -- penjumlahan ke eq_days_cum adalah tanggung jawab caller/job)
      n_samples, n_gaps_excluded
    """
    if not is_configured(cfg):
        return {"status": "not_configured"}

    if not samples:
        return {
            "status": "ok", "f_mean": None, "f_peak": None,
            "t_hot_mean_c": None, "t_hot_peak_c": None,
            "eq_days": 0.0, "n_samples": 0, "n_gaps_excluded": 0,
        }

    temps = [t for _, t in samples]
    f_values = [arrhenius_factor(t, cfg) for t in temps]

    max_gap_s = cfg["max_gap_minutes_for_integration"] * 60.0
    eq_days = 0.0
    n_gaps_excluded = 0
    for i in range(len(samples) - 1):
        ts0, _ = samples[i]
        ts1, _ = samples[i + 1]
        dt = ts1 - ts0
        if dt <= 0:
            continue
        if dt > max_gap_s:
            n_gaps_excluded += 1
            continue
        f_avg = (f_values[i] + f_values[i + 1]) / 2.0
        eq_days += equivalent_days_increment(f_avg, dt)

    return {
        "status": "ok",
        "f_mean": sum(f_values) / len(f_values),
        "f_peak": max(f_values),
        "t_hot_mean_c": sum(temps) / len(temps),
        "t_hot_peak_c": max(temps),
        "eq_days": eq_days,
        "n_samples": len(samples),
        "n_gaps_excluded": n_gaps_excluded,
    }
