"""server/lntai_vs_aging.py — SHIELD Stage 2, Part J2.

Fungsi murni (tanpa DB) untuk panel "LNTAI vs Aging Factor": klasifikasi
kuadran (aging factor x, LNTAI y) per fasa per trafo. Query DB (ambil
baris lntai_results v3 + aging_daily terbaru, agregasi harian) ditulis
di server/api.py (GET /api/lntai-vs-aging), supaya modul ini bisa diuji
tanpa database -- sama seperti pola ml/evaluate_fault_injection.py (Part I).

Batas "tinggi" LNTAI DIPAKAI ULANG dari config.LNTAI_V3_THRESHOLDS
["caution"] (BUKAN angka independen) -- lihat lntai_vs_aging_config.json.
Batas "tinggi" aging factor berasal dari lntai_vs_aging_config.json
sendiri (TODO(calibration), belum ada sumber lain untuk nilai ini).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parent / "lntai_vs_aging_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    LNTAI_VS_AGING_CONFIG: dict[str, Any] = json.load(_f)

AGING_FACTOR_SPLIT: float = float(LNTAI_VS_AGING_CONFIG["aging_factor_split"])
LNTAI_SPLIT: float = float(config.LNTAI_V3_THRESHOLDS["caution"])

QUADRANT_HEALTHY = "healthy"
QUADRANT_HIGH_STRESS_NORMAL = "high_stress_normal"
QUADRANT_ANOMALOUS_LOW_STRESS = "anomalous_low_stress"
QUADRANT_ANOMALOUS_HIGH_STRESS = "anomalous_high_stress"
QUADRANT_UNKNOWN = "unknown"

QUADRANT_LABELS: dict[str, str] = {
    QUADRANT_HEALTHY: "Healthy",
    QUADRANT_HIGH_STRESS_NORMAL: "High stress, behaving normally",
    QUADRANT_ANOMALOUS_LOW_STRESS: "Anomalous at low stress — investigate",
    QUADRANT_ANOMALOUS_HIGH_STRESS: "Anomalous under high stress — priority",
    QUADRANT_UNKNOWN: "Unknown (missing data)",
}

QUADRANT_READING_NOTE = (
    "Each point is one phase of one trafo: x = thermal aging factor (today), "
    "y = LNTAI (residual z-score). A point in the top-right (high aging AND "
    "high LNTAI) is the priority case; top-left (high LNTAI at low aging) is "
    "worth investigating even though load-driven aging looks normal."
)


def classify_quadrant(
    aging_factor: Optional[float], lntai: Optional[float],
    aging_split: float = AGING_FACTOR_SPLIT, lntai_split: float = LNTAI_SPLIT,
) -> str:
    """Klasifikasi kuadran. Batas DIMASUKKAN ke sisi 'tinggi' (>=), bukan
    'rendah' -- dipilih supaya "tinggi" di sini konsisten dengan arti
    ">= caution" di panel LNTAI v3 (level dihitung dengan >= di
    lntai_calculator.py/forecast.py _level_of(), lihat catatan di sana)."""
    if aging_factor is None or lntai is None:
        return QUADRANT_UNKNOWN
    high_aging = aging_factor >= aging_split
    high_lntai = lntai >= lntai_split
    if not high_aging and not high_lntai:
        return QUADRANT_HEALTHY
    if high_aging and not high_lntai:
        return QUADRANT_HIGH_STRESS_NORMAL
    if not high_aging and high_lntai:
        return QUADRANT_ANOMALOUS_LOW_STRESS
    return QUADRANT_ANOMALOUS_HIGH_STRESS


def build_scatter_point(
    trafo: str, phase: str, lntai: Optional[float], aging_factor: Optional[float],
    aging_split: float = AGING_FACTOR_SPLIT, lntai_split: float = LNTAI_SPLIT,
) -> dict[str, Any]:
    return {
        "trafo": trafo, "phase": phase,
        "aging_factor": aging_factor, "lntai": lntai,
        "quadrant": classify_quadrant(aging_factor, lntai, aging_split, lntai_split),
    }
