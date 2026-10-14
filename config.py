"""Konfigurasi LNTAI untuk SHIELD thermal anomaly monitoring.

Berisi DUA generasi konfigurasi:
  - Konstanta top-level (A, B, SIGMA, T_AMBIENT, I_RATED, THRESHOLDS) adalah
    LNTAI v1 (legacy) — z-score residual, basis I_RATED=1000A. Dipertahankan
    apa adanya (tidak diubah) supaya kode/test lama yang masih memakainya
    tetap menghasilkan angka yang sama persis, dan supaya v1 bisa dipakai
    sebagai pembanding terhadap v2 di laporan migrasi.
  - `LNTAI_CONFIG` adalah LNTAI v2 (skor komposit 0-40), dimuat dari
    lntai_config.json — SATU sumber kebenaran yang juga disinkronkan secara
    manual ke `LNTAI_CFG` di Shield.html/index.html (lihat test parity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


# ════════════════════════════════════════════════════════════════════════
# LNTAI v1 (LEGACY) — z-score residual, basis I_RATED=1000A.
# Model regresi kuadratik delta suhu terhadap arus ternormalisasi.
# ════════════════════════════════════════════════════════════════════════
A: Final[float] = 8.2059
B: Final[float] = 26.86
SIGMA: Final[float] = 4.6761

T_AMBIENT: Final[float] = 30.0
I_RATED: Final[float] = 1000.0


@dataclass(frozen=True)
class LevelConfig:
    """Metadata level untuk output dashboard."""

    name: str
    color: str
    icon: str
    recommendation: str


# Ambang v1 (skala z-score residual, BUKAN skala 0-40 v2 — lihat
# LNTAI_CONFIG["thresholds"] di bawah untuk ambang v2: warning=18, critical=30).
THRESHOLDS: Final[dict[str, float]] = {
    "attention": 1.5,
    "alert": 3.0,
    "critical": 5.0,
}

LEVELS: Final[dict[str, LevelConfig]] = {
    "NORMAL": LevelConfig(
        name="NORMAL",
        color="green",
        icon="🟢",
        recommendation="Kondisi normal. Lanjutkan monitoring berkala.",
    ),
    "PERHATIAN": LevelConfig(
        name="PERHATIAN",
        color="yellow-orange",
        icon="🟡",
        recommendation="Pantau tren suhu dan arus. Verifikasi kondisi sensor dan beban.",
    ),
    "SIAGA": LevelConfig(
        name="SIAGA",
        color="orange-red",
        icon="🟠",
        recommendation="Lakukan inspeksi terjadwal pada kontak finger dan evaluasi pola pembebanan.",
    ),
    "KRITIS": LevelConfig(
        name="KRITIS",
        color="red",
        icon="🔴",
        recommendation="Tindakan segera, lakukan inspeksi kontak finger dan pertimbangkan pengurangan beban.",
    ),
    "DATA_INVALID": LevelConfig(
        name="DATA_INVALID",
        color="gray",
        icon="⚪",
        recommendation="Data sensor tidak valid. Periksa payload, koneksi sensor, dan mapping field.",
    ),
}


# ════════════════════════════════════════════════════════════════════════
# LNTAI v2 — dimuat dari lntai_config.json (satu sumber kebenaran, disalin
# manual ke LNTAI_CFG di Shield.html/index.html; test parity memverifikasi
# keduanya sama). Jangan hardcode ulang nilai-nilai ini di tempat lain.
# ════════════════════════════════════════════════════════════════════════
_LNTAI_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parent / "lntai_config.json"

with open(_LNTAI_CONFIG_PATH, "r", encoding="utf-8") as _f:
    LNTAI_CONFIG: Final[dict[str, Any]] = json.load(_f)

LNTAI_MODEL_VERSION: Final[str] = LNTAI_CONFIG["model_version"]
LNTAI_BASELINE_CALIBRATED: Final[bool] = LNTAI_CONFIG["baseline_calibrated"]

LNTAI_V2_A: Final[float] = LNTAI_CONFIG["baseline"]["A"]
LNTAI_V2_B: Final[float] = LNTAI_CONFIG["baseline"]["B"]
LNTAI_V2_SIGMA: Final[float] = LNTAI_CONFIG["baseline"]["sigma"]
LNTAI_V2_I_RATED: Final[float] = LNTAI_CONFIG["baseline"]["I_RATED"]

LNTAI_V2_TAU_MINUTES: Final[float] = LNTAI_CONFIG["thermal_lag"]["tau_minutes"]
LNTAI_V2_MAX_GAP_MINUTES: Final[float] = LNTAI_CONFIG["thermal_lag"]["max_gap_minutes"]

LNTAI_V2_PAIRING_TOLERANCE_MINUTES: Final[float] = LNTAI_CONFIG["pairing_tolerance_minutes"]

LNTAI_V2_Z_FLOOR: Final[float] = LNTAI_CONFIG["residual_score"]["z_floor"]
LNTAI_V2_Z_FULL: Final[float] = LNTAI_CONFIG["residual_score"]["z_full"]

LNTAI_V2_REF_TEMP_RISE: Final[float] = LNTAI_CONFIG["component_refs"]["ref_temp_rise"]
LNTAI_V2_REF_PHASE_DELTA: Final[float] = LNTAI_CONFIG["component_refs"]["ref_phase_delta"]
LNTAI_V2_DOUBLING_K: Final[float] = LNTAI_CONFIG["component_refs"]["doubling_k"]

LNTAI_V2_WEIGHTS: Final[dict[str, float]] = LNTAI_CONFIG["weights"]
LNTAI_V2_THRESHOLDS: Final[dict[str, float]] = LNTAI_CONFIG["thresholds"]
LNTAI_V2_SCORE_MAX: Final[float] = LNTAI_CONFIG["score_max"]

LNTAI_V2_AMBIENT_DEFAULT: Final[float] = LNTAI_CONFIG["ambient"]["default"]
LNTAI_V2_AMBIENT_MAX_AGE_MINUTES: Final[float] = LNTAI_CONFIG["ambient"]["max_age_minutes"]

LNTAI_V2_SMOOTHING_WINDOW: Final[int] = LNTAI_CONFIG["smoothing_window"]


# ════════════════════════════════════════════════════════════════════════
# LNTAI v3 (SHIELD Stage 2, Part A) — physics-informed residual z-score,
# PER PHASE, dimuat dari lntai_v3_config.json (file TERPISAH dari
# lntai_config.json di atas -- v2 tetap dihitung & ditulis apa adanya,
# tidak diubah oleh penambahan v3; lihat lntai_calculator.lntai_v3_step()).
# ════════════════════════════════════════════════════════════════════════
_LNTAI_V3_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parent / "lntai_v3_config.json"

with open(_LNTAI_V3_CONFIG_PATH, "r", encoding="utf-8") as _f:
    LNTAI_V3_CONFIG: Final[dict[str, Any]] = json.load(_f)

LNTAI_V3_MODEL_VERSION: Final[str] = LNTAI_V3_CONFIG["model_version"]

LNTAI_V3_BASELINE_ID: Final[str] = LNTAI_V3_CONFIG["baseline"]["baseline_id"]
LNTAI_V3_CALIBRATED: Final[bool] = LNTAI_V3_CONFIG["baseline"]["calibrated"]
LNTAI_V3_A: Final[float] = LNTAI_V3_CONFIG["baseline"]["A"]
LNTAI_V3_B: Final[float] = LNTAI_V3_CONFIG["baseline"]["B"]
LNTAI_V3_SIGMA: Final[float] = LNTAI_V3_CONFIG["baseline"]["sigma"]
LNTAI_V3_MU: Final[float] = LNTAI_V3_CONFIG["baseline"]["mu"]
LNTAI_V3_I_REF: Final[float] = LNTAI_V3_CONFIG["baseline"]["i_ref"]
LNTAI_V3_AMBIENT_C: Final[float] = LNTAI_V3_CONFIG["baseline"]["ambient_c"]

LNTAI_V3_ROLLING_MEDIAN_MINUTES: Final[float] = LNTAI_V3_CONFIG["rolling_median_minutes"]
LNTAI_V3_THRESHOLDS: Final[dict[str, float]] = LNTAI_V3_CONFIG["thresholds"]
