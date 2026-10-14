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
from statistics import median
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

        # Presisi penuh (tidak dibulatkan) -- pembulatan untuk tampilan adalah
        # urusan lapisan display (mis. realtime_monitor.py print / UI .toFixed()),
        # BUKAN hasil komputasi. Membulatkan di sini akan membuat test parity
        # Python<->JS gagal pada toleransi ketat (noise pembulatan >> 1e-6).
        per_phase[p] = {
            "T": t_p,
            "I": currents[p],
            "dT": delta_t_p,
            "dT_pred": delta_t_pred_p,
            "z": z_p,
            "dAB": dab_p,
            "i_eff_norm": i_eff_norm_p,
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
        # Presisi penuh (lihat catatan di per_phase di atas) -- pembulatan
        # adalah urusan lapisan display, bukan hasil komputasi.
        "lntai": lntai,
        "level": level,
        "S_Temp": s_temp,
        "S_DeltaT": s_delta_t,
        "S_RES": s_res,
        "hottest_phase": hottest_phase,
        "t_contact": hp["T"],
        "t_ambient": t_ambient,
        "ambient_source": ambient_source,
        "delta_T_real": hp["dT"],
        "delta_T_pred": hp["dT_pred"],
        "residual": hp["dT"] - hp["dT_pred"],
        "z": hp["z"],
        "i_eff_norm": hp["i_eff_norm"],
        "delta_phase": delta_phase,
        "dAB_max": dab_max,
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
# LNTAI v3 (SHIELD Stage 2, Part A) — residual z-score PER FASA, memakai
# ULANG i_eff (hasil filter lag termal) yang SUDAH dihitung lntai_step()
# untuk v2 -- TIDAK menjalankan filter I^2 sendiri, hanya mengubah basis
# normalisasi arus (1780A -> 1000A) dan memakai baseline LEGACY v1
# (A/B/sigma/mu/i_ref/ambient dari lntai_v3_config.json, SEMUA belum
# dikalibrasi -- lihat baseline.calibrated=false di file itu).
#
# LNTAI_v3 = max_p z_p, z_p = (residual_p - mu) / sigma,
#   residual_p = (T_p - ambient_c) - (A * (I_eff_p/i_ref)^2 + B)
# "level" (Normal/Caution/Alert/Critical) berasal dari MEDIAN BERGULIR
# 30 menit atas LNTAI_v3 instan, BUKAN nilai instan itu sendiri -- supaya
# tidak gonta-ganti level hanya karena satu sampel noise. Kolom "lntai"
# tetap menyimpan nilai INSTAN; median tersimpan terpisah
# (lntai_rolling_median_30min, lihat migrations/008).
# ════════════════════════════════════════════════════════════════════════

def _v3_invalid_result(timestamp: Optional[str], model_version: str, error: str) -> dict[str, Any]:
    return {
        "lntai": None, "lntai_rolling_median_30min": None, "level": "DATA_INVALID",
        "hottest_phase": None, "per_phase": None,
        "baseline_id": None, "calibrated": None,
        "timestamp": timestamp, "model_version": model_version, "error": error,
    }


def lntai_v3_step(
    state: dict[str, Any],
    v2_result: dict[str, Any],
    ts_epoch: float,
    *,
    a: float,
    b: float,
    sigma: float,
    mu: float,
    i_ref: float,
    i_rated_v2: float,
    ambient_c: float,
    rolling_median_minutes: float,
    thresholds: dict[str, float],
    baseline_id: str,
    calibrated: bool,
    model_version: str,
    baseline_overrides: Optional[dict[str, tuple[float, str, bool]]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Satu langkah LNTAI v3, murni -- mengambil `v2_result` (hasil
    lntai_step() yang SAMA, SUDAH dihitung untuk sampel ini) sebagai input,
    bukan sample mentah. `state`: {"rolling_buffer": [(ts_epoch, instant), ...]}
    hanya untuk level (median 30 menit); TIDAK ada state filter I^2 di sini
    (dipakai ulang dari v2_result["per_phase"][p]["i_eff_norm"]).

    `baseline_overrides` (SHIELD Stage 2, Part B): {phase: (delta_t_pred,
    baseline_id, calibrated)} opsional -- kalau ADA untuk satu fasa, nilai
    itu DIPAKAI LANGSUNG sebagai delta_T_pred fasa itu (sudah dihitung
    ml/model_registry.py dari model aktif yang dilatih ml/train_baseline.py),
    MENGGANTI formula fisika legacy A/B HANYA untuk fasa itu -- fasa lain
    tanpa override tetap memakai fisika legacy. Ini mewujudkan catatan di
    Part A: "ΔT_pred,p = baseline_t,p(features) (until Part B: legacy
    physics ...)" -- sekarang Part B tersedia, baseline_t,p menjadi model
    aktif kalau ada, fisika legacy kalau tidak (lihat fallback di
    ml/model_registry.py predict_delta_t_step())."""
    rolling_buffer: list[tuple[float, float]] = list(state.get("rolling_buffer", []))
    overrides = baseline_overrides or {}

    if v2_result.get("level") == "DATA_INVALID" or not v2_result.get("per_phase"):
        new_state = {"rolling_buffer": rolling_buffer}
        reason = v2_result.get("error") or "Input v2 tidak valid (DATA_INVALID) -- v3 tidak bisa dihitung dari situ"
        return new_state, _v3_invalid_result(v2_result.get("timestamp"), model_version, reason)

    per_phase_v3: dict[str, dict[str, Any]] = {}
    for p in PHASES:
        v2p = v2_result["per_phase"][p]
        t_p = v2p["T"]
        # i_eff_norm (v2) dinormalisasi terhadap i_rated_v2 (1780A) -- kalikan
        # balik untuk dapat arus efektif riil (Ampere), lalu normalisasi ulang
        # terhadap i_ref basis legacy (1000A) milik baseline v3.
        i_eff_amps = v2p["i_eff_norm"] * i_rated_v2
        i_eff_norm_legacy = (i_eff_amps / i_ref) if i_ref else 0.0

        override = overrides.get(p)
        if override is not None and override[0] is not None:
            delta_t_pred, phase_baseline_id, phase_calibrated = override
        else:
            delta_t_pred = a * (i_eff_norm_legacy ** 2) + b
            phase_baseline_id, phase_calibrated = baseline_id, calibrated

        delta_t_real = t_p - ambient_c
        residual_c = delta_t_real - delta_t_pred
        z = (residual_c - mu) / sigma if sigma else 0.0

        per_phase_v3[p] = {
            "T": t_p, "delta_T_real": delta_t_real, "delta_T_pred": delta_t_pred,
            "residual_c": residual_c, "z": z,
            "baseline_id": phase_baseline_id, "calibrated": phase_calibrated,
        }

    hottest_phase_v3 = max(PHASES, key=lambda p: per_phase_v3[p]["z"])
    instant_lntai = per_phase_v3[hottest_phase_v3]["z"]

    rolling_buffer.append((ts_epoch, instant_lntai))
    cutoff = ts_epoch - rolling_median_minutes * 60.0
    rolling_buffer = [(t, v) for (t, v) in rolling_buffer if t >= cutoff]
    median_lntai = median(v for (_, v) in rolling_buffer)

    if median_lntai >= thresholds["critical"]:
        level = "Critical"
    elif median_lntai >= thresholds["alert"]:
        level = "Alert"
    elif median_lntai >= thresholds["caution"]:
        level = "Caution"
    else:
        level = "Normal"

    new_state = {"rolling_buffer": rolling_buffer}
    result = {
        "lntai": instant_lntai,
        "lntai_rolling_median_30min": median_lntai,
        "level": level,
        "hottest_phase": hottest_phase_v3,
        "per_phase": per_phase_v3,
        # baseline_id/calibrated di level baris = milik fasa TERPANAS (yang
        # menentukan skor "lntai" di atas) -- lihat per_phase[p]["baseline_id"]
        # untuk rincian per fasa kalau fasa lain memakai sumber berbeda
        # (mis. satu fasa punya model aktif, fasa lain masih fallback legacy).
        "baseline_id": per_phase_v3[hottest_phase_v3]["baseline_id"],
        "calibrated": per_phase_v3[hottest_phase_v3]["calibrated"],
        "timestamp": v2_result["timestamp"],
        "model_version": model_version,
        "error": None,
    }
    return new_state, result


# ════════════════════════════════════════════════════════════════════════
# Kelas stateful — pembungkus lntai_step() untuk pemakaian realtime (state
# bertahan per trafo/fasa) dan histori (fold dengan state lokal).
# ════════════════════════════════════════════════════════════════════════

class LNTAICalculator:
    """Kalkulator LNTAI v2 (Load Normalized Thermal Anomaly Index)."""

    def __init__(self, cfg: Optional[dict[str, Any]] = None, cfg_v3: Optional[dict[str, Any]] = None) -> None:
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

        # ── LNTAI v3 (Stage 2 Part A) — config TERPISAH, state TERPISAH
        # (self._state_v3), TIDAK memengaruhi field v2 di atas sama sekali.
        c3 = cfg_v3 or config.LNTAI_V3_CONFIG
        self.v3_a = float(c3["baseline"]["A"])
        self.v3_b = float(c3["baseline"]["B"])
        self.v3_sigma = float(c3["baseline"]["sigma"])
        self.v3_mu = float(c3["baseline"]["mu"])
        self.v3_i_ref = float(c3["baseline"]["i_ref"])
        self.v3_ambient_c = float(c3["baseline"]["ambient_c"])
        self.v3_baseline_id = str(c3["baseline"]["baseline_id"])
        self.v3_calibrated = bool(c3["baseline"]["calibrated"])
        self.v3_rolling_median_minutes = float(c3["rolling_median_minutes"])
        self.v3_thresholds = dict(c3["thresholds"])
        self.v3_model_version = str(c3.get("model_version", "lntai-v3"))
        self._state_v3: dict[str, dict[str, Any]] = {}

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

    def _v3_kwargs(self) -> dict[str, Any]:
        return dict(
            a=self.v3_a, b=self.v3_b, sigma=self.v3_sigma, mu=self.v3_mu,
            i_ref=self.v3_i_ref, i_rated_v2=self.i_rated, ambient_c=self.v3_ambient_c,
            rolling_median_minutes=self.v3_rolling_median_minutes, thresholds=self.v3_thresholds,
            baseline_id=self.v3_baseline_id, calibrated=self.v3_calibrated,
            model_version=self.v3_model_version,
        )

    def reset(self, trafo: Optional[str] = None) -> None:
        """Buang state filter lag termal v2 DAN buffer median bergulir v3
        (mis. saat trafo berganti sumber data)."""
        if trafo is None:
            self._state.clear()
            self._state_v3.clear()
        else:
            self._state.pop(str(trafo), None)
            self._state_v3.pop(str(trafo), None)

    def update(self, trafo: str, sample: dict[str, Any]) -> dict[str, Any]:
        """Satu sampel realtime untuk `trafo` — state filter lag termal bertahan
        antar panggilan (disimpan di self._state[trafo])."""
        trafo_key = str(trafo)
        phase_state = self._state.get(trafo_key, {})
        new_state, result = lntai_step(phase_state, sample, **self._kwargs())
        self._state[trafo_key] = new_state
        return result

    def update_v3(self, trafo: str, v2_result: dict[str, Any], ts_epoch: float,
                   baseline_overrides: Optional[dict[str, tuple[float, str, bool]]] = None) -> dict[str, Any]:
        """LNTAI v3 realtime untuk `trafo`, dari `v2_result` (output `update()`
        UNTUK SAMPEL YANG SAMA, dipanggil sebelum ini). Buffer median bergulir
        30 menit bertahan antar panggilan (self._state_v3[trafo]).
        `baseline_overrides` (Stage 2 Part B) -- lihat lntai_v3_step()."""
        trafo_key = str(trafo)
        v3_state = self._state_v3.get(trafo_key, {})
        new_state, result = lntai_v3_step(
            v3_state, v2_result, ts_epoch, baseline_overrides=baseline_overrides, **self._v3_kwargs()
        )
        self._state_v3[trafo_key] = new_state
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

    def calculate_series_v3(self, v2_results_with_ts: list[tuple[dict[str, Any], float]]) -> list[dict[str, Any]]:
        """Hitung LNTAI v3 untuk histori, dari pasangan (v2_result, ts_epoch)
        SUDAH terurut kronologis lama->baru (lihat calculate_series() untuk
        menghasilkan v2_result-nya). State LOKAL (buffer median), tidak
        menyentuh self._state_v3."""
        local_state: dict[str, Any] = {}
        results: list[dict[str, Any]] = []
        for v2_result, ts_epoch in v2_results_with_ts:
            local_state, result = lntai_v3_step(local_state, v2_result, ts_epoch, **self._v3_kwargs())
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
