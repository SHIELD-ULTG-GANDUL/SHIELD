"""LNTAI v2 calculator untuk SHIELD — Load Normalized Thermal Anomaly Index.

LNTAI bukan alarm suhu absolut. LNTAI membandingkan kenaikan suhu kontak
aktual terhadap kenaikan suhu yang diprediksi dari arus beban (dengan lag
termal orde-1), lalu mengombinasikan tiga komponen skor (0-40) menjadi satu
skor komposit 0-40:

    S_Temp  — kenaikan suhu titik terpanas vs ambient, kurva eksponensial
               heuristik (terinspirasi kaidah Montsinger untuk isolasi;
               belum divalidasi untuk kontak finger).
    S_DeltaT — selisih suhu antar fasa (ketimpangan beban/kontak).
    S_RES   — residual ternormalisasi (z-score) antara kenaikan suhu aktual
               dan kenaikan suhu yang diprediksi dari arus efektif (setelah
               difilter lag termal), terhadap baseline A/B/sigma.

    LNTAI = w_temp*S_Temp + w_dT*S_DeltaT + w_res*S_RES   (hasil 0-40)

Semua konstanta (A, B, sigma, tau, ambang, bobot, dst.) berasal dari SATU
sumber: lntai_config.json (dimuat lewat config.LNTAI_CONFIG). Definisi yang
sama diimplementasikan ulang di Shield.html/index.html (lihat blok
<LNTAI-CORE> di sana) — tests/test_lntai_js.mjs memverifikasi kedua
implementasi menghasilkan angka yang identik dari vector yang sama
(lntai_test_vectors.json).

TODO(kalibrasi): baseline A/B/sigma di lntai_config.json adalah hasil fit
LNTAI v1 lama (basis arus 1000A) yang dikonversi secara matematis ke basis
1780A — BUKAN hasil fit baru dari data trafo 2/3/4 yang sebenarnya. Pakai
`fit_baseline()` pada histori data riil lalu tinjau hasilnya secara manual
sebelum menulis nilai baru ke lntai_config.json (baseline_calibrated masih
False sampai itu dilakukan). tau_minutes dan max_gap_minutes juga masih
asumsi rekayasa, bukan hasil kalibrasi.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import exp, isfinite, sqrt
from typing import Any, Optional

import config

PHASES: tuple[str, str, str] = ("R", "S", "T")


# ════════════════════════════════════════════════════════════════════════
# Fungsi murni — tidak menyentuh state kelas, supaya gampang diuji dan
# dicocokkan 1:1 dengan versi JS di <LNTAI-CORE>.
# ════════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def arrhenius_score(rise: float, ref: float, doubling_k: float, score_max: float) -> float:
    """Skor eksponensial: 0 pada rise<=0, ~separuh score_max pada rise=ref-doubling_k,
    tepat score_max pada rise=ref. Kaidah Montsinger/Arrhenius (risiko berlipat ganda
    setiap kenaikan doubling_k derajat) dipakai sebagai heuristik, bukan hukum fisik
    yang tervalidasi untuk kontak finger — lihat catatan di config dan UI."""
    if not isfinite(rise) or not isfinite(ref) or ref <= 0:
        return 0.0
    num = 2.0 ** (rise / doubling_k) - 1.0
    den = 2.0 ** (ref / doubling_k) - 1.0
    if not isfinite(num) or not isfinite(den) or den <= 0:
        return 0.0
    return _clamp((num / den) * score_max, 0.0, score_max)


def _is_finite_number(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


def lntai_step(
    state: dict[str, dict[str, Any]],
    sample: dict[str, Any],
    *,
    a: float,
    b: float,
    sigma: float,
    i_rated: float,
    tau_minutes: float,
    max_gap_minutes: float,
    z_floor: float,
    z_full: float,
    ref_temp_rise: float,
    ref_phase_delta: float,
    doubling_k: float,
    weights: dict[str, float],
    thresholds: dict[str, float],
    score_max: float,
    ambient_default: float,
    model_version: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Satu langkah LNTAI v2, murni (state masuk & keluar eksplisit, tidak ada
    efek samping tersembunyi) — dipakai baik untuk tick realtime (state
    bertahan antar panggilan) maupun fold atas histori (state lokal, dibuang
    setelah selesai). Lihat docstring modul untuk definisi lengkap.

    state: {"R": {"i2_eff": float|None, "last_time": float|None (epoch detik)},
            "S": {...}, "T": {...}}
    sample: {
        "R_Atas","R_Bawah","S_Atas","S_Bawah","T_Atas","T_Bawah": suhu (°C),
        "I_R","I_S","I_T": arus fasa (A),
        "t_ambient": float|None (ambient terukur, None -> pakai default),
        "ambient_source": str|None (opsional, override label sumber ambient),
        "timestamp": float epoch detik | None (None -> waktu sekarang UTC),
    }
    """
    timestamp_now = datetime.now(timezone.utc)
    ts_epoch = sample.get("timestamp")
    if ts_epoch is None:
        ts_epoch = timestamp_now.timestamp()
        ts_iso = timestamp_now.isoformat()
    else:
        ts_iso = datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).isoformat()

    temp_keys = ("R_Atas", "R_Bawah", "S_Atas", "S_Bawah", "T_Atas", "T_Bawah")
    current_keys = ("I_R", "I_S", "I_T")
    required = [sample.get(k) for k in temp_keys] + [sample.get(k) for k in current_keys]

    if not all(_is_finite_number(v) for v in required) or i_rated == 0 or sigma == 0:
        invalid = _invalid_result(ts_iso, model_version, "Data suhu/arus tidak lengkap atau tidak valid")
        return state, invalid

    temps = {k: float(sample[k]) for k in temp_keys}
    currents = {"R": float(sample["I_R"]), "S": float(sample["I_S"]), "T": float(sample["I_T"])}

    if sample.get("t_ambient") is not None and _is_finite_number(sample["t_ambient"]):
        t_ambient = float(sample["t_ambient"])
        ambient_source = sample.get("ambient_source") or "measured"
    else:
        t_ambient = float(ambient_default)
        ambient_source = sample.get("ambient_source") or "default"

    new_state: dict[str, dict[str, Any]] = {}
    per_phase: dict[str, dict[str, float]] = {}
    for p in PHASES:
        atas, bawah = temps[f"{p}_Atas"], temps[f"{p}_Bawah"]
        t_p = max(atas, bawah)
        dab_p = abs(atas - bawah)
        delta_t_p = t_p - t_ambient

        ph_state = state.get(p) or {"i2_eff": None, "last_time": None}
        i2_sample = currents[p] ** 2
        if ph_state["i2_eff"] is None or ph_state["last_time"] is None:
            i2_eff = i2_sample  # sampel pertama / state kosong: mulai dari nilai sesaat
        else:
            dt_minutes = max(0.0, (ts_epoch - ph_state["last_time"]) / 60.0)
            if dt_minutes > max_gap_minutes:
                i2_eff = i2_sample  # gap data > max_gap_minutes: reset filter, jangan bawa estimasi basi
            else:
                phi = exp(-dt_minutes / tau_minutes) if tau_minutes > 0 else 0.0
                i2_eff = phi * ph_state["i2_eff"] + (1.0 - phi) * i2_sample
        new_state[p] = {"i2_eff": i2_eff, "last_time": ts_epoch}

        i_eff_norm_p = sqrt(max(i2_eff, 0.0)) / i_rated
        delta_t_pred_p = a * (i_eff_norm_p ** 2) + b
        z_p = (delta_t_p - delta_t_pred_p) / sigma

        per_phase[p] = {
            "T": round(t_p, 3),
            "I": round(currents[p], 3),
            "dT": round(delta_t_p, 3),
            "dT_pred": round(delta_t_pred_p, 3),
            "z": round(z_p, 4),
            "dAB": round(dab_p, 3),
            "i_eff_norm": round(i_eff_norm_p, 5),
        }

    hottest_phase = max(PHASES, key=lambda p: per_phase[p]["T"])
    phase_temps = [per_phase[p]["T"] for p in PHASES]
    delta_phase = max(phase_temps) - min(phase_temps)
    dab_max = max(per_phase[p]["dAB"] for p in PHASES)
    max_delta_t = max(per_phase[p]["dT"] for p in PHASES)
    max_z = max(per_phase[p]["z"] for p in PHASES)

    s_temp = arrhenius_score(max_delta_t, ref_temp_rise, doubling_k, score_max)
    s_delta_t = arrhenius_score(delta_phase, ref_phase_delta, doubling_k, score_max)
    s_res = _clamp((max_z - z_floor) / (z_full - z_floor) * score_max, 0.0, score_max) if z_full != z_floor else 0.0

    lntai = weights["temp"] * s_temp + weights["delta_t"] * s_delta_t + weights["res"] * s_res
    if lntai >= thresholds["critical"]:
        level = "CRITICAL"
    elif lntai >= thresholds["warning"]:
        level = "WARNING"
    else:
        level = "NORMAL"

    hp = per_phase[hottest_phase]
    result = {
        "lntai": round(lntai, 3),
        "level": level,
        "S_Temp": round(s_temp, 3),
        "S_DeltaT": round(s_delta_t, 3),
        "S_RES": round(s_res, 3),
        "hottest_phase": hottest_phase,
        "t_contact": hp["T"],
        "t_ambient": round(t_ambient, 3),
        "ambient_source": ambient_source,
        "delta_T_real": hp["dT"],
        "delta_T_pred": hp["dT_pred"],
        "residual": round(hp["dT"] - hp["dT_pred"], 3),
        "z": hp["z"],
        "i_eff_norm": hp["i_eff_norm"],
        "delta_phase": round(delta_phase, 3),
        "dAB_max": round(dab_max, 3),
        "per_phase": per_phase,
        "timestamp": ts_iso,
        "model_version": model_version,
    }
    return new_state, result


def _invalid_result(timestamp: str, model_version: str, error: str) -> dict[str, Any]:
    return {
        "lntai": None,
        "level": "DATA_INVALID",
        "S_Temp": None,
        "S_DeltaT": None,
        "S_RES": None,
        "hottest_phase": None,
        "t_contact": None,
        "t_ambient": None,
        "ambient_source": None,
        "delta_T_real": None,
        "delta_T_pred": None,
        "residual": None,
        "z": None,
        "i_eff_norm": None,
        "delta_phase": None,
        "dAB_max": None,
        "per_phase": None,
        "timestamp": timestamp,
        "model_version": model_version,
        "error": error,
    }


# ════════════════════════════════════════════════════════════════════════
# Kelas stateful — pembungkus lntai_step() untuk pemakaian realtime (state
# bertahan per trafo/fasa) dan histori (fold dengan state lokal).
# ════════════════════════════════════════════════════════════════════════

class LNTAICalculator:
    """Kalkulator LNTAI v2 (Load Normalized Thermal Anomaly Index)."""

    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        c = cfg or config.LNTAI_CONFIG
        self.a = float(c["baseline"]["A"])
        self.b = float(c["baseline"]["B"])
        self.sigma = float(c["baseline"]["sigma"])
        self.i_rated = float(c["baseline"]["I_RATED"])
        self.tau_minutes = float(c["thermal_lag"]["tau_minutes"])
        self.max_gap_minutes = float(c["thermal_lag"]["max_gap_minutes"])
        self.z_floor = float(c["residual_score"]["z_floor"])
        self.z_full = float(c["residual_score"]["z_full"])
        self.ref_temp_rise = float(c["component_refs"]["ref_temp_rise"])
        self.ref_phase_delta = float(c["component_refs"]["ref_phase_delta"])
        self.doubling_k = float(c["component_refs"]["doubling_k"])
        self.weights = dict(c["weights"])
        self.thresholds = dict(c["thresholds"])
        self.score_max = float(c["score_max"])
        self.ambient_default = float(c["ambient"]["default"])
        self.model_version = str(c.get("model_version", "lntai-v2"))
        self._state: dict[str, dict[str, dict[str, Any]]] = {}

    def _kwargs(self) -> dict[str, Any]:
        return dict(
            a=self.a, b=self.b, sigma=self.sigma, i_rated=self.i_rated,
            tau_minutes=self.tau_minutes, max_gap_minutes=self.max_gap_minutes,
            z_floor=self.z_floor, z_full=self.z_full,
            ref_temp_rise=self.ref_temp_rise, ref_phase_delta=self.ref_phase_delta,
            doubling_k=self.doubling_k, weights=self.weights, thresholds=self.thresholds,
            score_max=self.score_max, ambient_default=self.ambient_default,
            model_version=self.model_version,
        )

    def reset(self, trafo: Optional[str] = None) -> None:
        """Buang state filter lag termal (mis. saat trafo berganti sumber data)."""
        if trafo is None:
            self._state.clear()
        else:
            self._state.pop(str(trafo), None)

    def update(self, trafo: str, sample: dict[str, Any]) -> dict[str, Any]:
        """Satu sampel realtime untuk `trafo` — state filter lag termal bertahan
        antar panggilan (disimpan di self._state[trafo])."""
        trafo_key = str(trafo)
        phase_state = self._state.get(trafo_key, {})
        new_state, result = lntai_step(phase_state, sample, **self._kwargs())
        self._state[trafo_key] = new_state
        return result

    def calculate_series(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Hitung LNTAI untuk histori `rows` (harus sudah terurut kronologis
        lama->baru). Memakai state LOKAL yang terpisah dari state realtime
        (`self._state`) — tidak mengganggu filter live, dan tidak membawa state
        dari satu pemanggilan calculate_series ke pemanggilan berikutnya."""
        local_state: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        for row in rows:
            local_state, result = lntai_step(local_state, row, **self._kwargs())
            results.append(result)
        return results

    def fit_baseline(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit least-squares A, B dari histori nyata: deltaT_real = A*i_eff_norm^2 + B,
        memakai i_eff (arus setelah filter lag termal), BUKAN arus sesaat. Sigma
        diestimasi dari residual std. TIDAK menulis ke lntai_config.json — review
        hasilnya secara manual (R^2, jumlah sampel) sebelum menulis nilai baru dan
        mengubah baseline_calibrated jadi true.

        rows: data historis terurut kronologis, format sama dengan `sample` di
        lntai_step() (temps + I_R/I_S/I_T + timestamp epoch detik).
        """
        local_state: dict[str, dict[str, Any]] = {}
        xs: list[float] = []  # i_eff_norm^2 (semua fasa, semua sampel)
        ys: list[float] = []  # delta_T_real (semua fasa, semua sampel)
        for row in rows:
            local_state, result = lntai_step(local_state, row, **self._kwargs())
            if result["level"] == "DATA_INVALID" or not result["per_phase"]:
                continue
            for p in PHASES:
                ph = result["per_phase"][p]
                xs.append(ph["i_eff_norm"] ** 2)
                ys.append(ph["dT"])

        n = len(xs)
        if n < 2:
            return {"A": None, "B": None, "sigma": None, "r2": None, "n": n,
                    "error": "Sampel valid tidak cukup untuk fit (butuh >= 2)"}

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        s_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        s_xx = sum((x - mean_x) ** 2 for x in xs)
        if s_xx == 0:
            return {"A": None, "B": None, "sigma": None, "r2": None, "n": n,
                    "error": "Variansi i_eff_norm^2 nol — tidak bisa fit slope"}

        a_fit = s_xy / s_xx
        b_fit = mean_y - a_fit * mean_x

        residuals = [y - (a_fit * x + b_fit) for x, y in zip(xs, ys)]
        ss_res = sum(r ** 2 for r in residuals)
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
        sigma_fit = sqrt(ss_res / (n - 2)) if n > 2 else None

        return {"A": round(a_fit, 4), "B": round(b_fit, 4),
                "sigma": round(sigma_fit, 4) if sigma_fit is not None else None,
                "r2": round(r2, 4) if r2 is not None else None, "n": n}
