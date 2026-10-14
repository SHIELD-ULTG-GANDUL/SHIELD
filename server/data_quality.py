"""server/data_quality.py — fungsi murni untuk menilai kualitas satu
pembacaan sensor (atau satu baris suhu/beban) dan menghasilkan bitmask
quality_flags (lihat migrations/001_add_quality_flags.sql).

Semua fungsi di sini MURNI (tidak ada I/O, tidak ada state tersembunyi) --
menerima nilai/riwayat sebagai argumen, mengembalikan flag sebagai set
string atau bitmask integer. Ini sengaja membuatnya gampang dites (lihat
server/tests/test_data_quality.py) dan reusable baik dari
mqtt_subscriber.py (aturan per-sampel: MISSING, ZERO_READING,
OUT_OF_RANGE) maupun lntai_worker.py (aturan berbasis jendela: STUCK,
SPIKE, PAIRING_GAP, TIME_GAP).

Definisi & ambang di server/data_quality_config.json -- TIDAK ada angka
yang dikarang di sini, semua dibaca dari config (lihat TODO(verifikasi)
di file itu).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_CONFIG_PATH = Path(__file__).resolve().parent / "data_quality_config.json"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    DATA_QUALITY_CONFIG: dict = json.load(_f)

FLAG_BITS: dict[str, int] = DATA_QUALITY_CONFIG["flags"]


# ════════════════════════════════════════════════════════════════════════
# Aturan PER-SAMPEL — tidak butuh riwayat, bisa dihitung saat sampel masuk
# (dipakai mqtt_subscriber.py, cabang A).
# ════════════════════════════════════════════════════════════════════════

def classify_temperature_value(value: Optional[float], cfg: dict = DATA_QUALITY_CONFIG) -> set[str]:
    """Flag untuk SATU pembacaan suhu (°C). value=None berarti field hilang
    dari payload -- lihat mqtt_subscriber.py _to_float_or_none()."""
    if value is None:
        return {"MISSING"}
    flags: set[str] = set()
    if value == 0:
        flags.add("ZERO_READING")
    lo = cfg["temperature_range_c"]["min"]
    hi = cfg["temperature_range_c"]["max"]
    if value < lo or value > hi:
        flags.add("OUT_OF_RANGE")
    return flags


def classify_current_value(value: Optional[float], cfg: dict = DATA_QUALITY_CONFIG) -> set[str]:
    """Flag untuk SATU pembacaan arus (A). value=None berarti field hilang."""
    if value is None:
        return {"MISSING"}
    flags: set[str] = set()
    if value == 0:
        flags.add("ZERO_READING")
    i_rated = cfg["current_range"]["i_rated"]
    max_allowed = i_rated * cfg["current_range"]["multiplier_of_rated"]
    if value < 0 or value > max_allowed:
        flags.add("OUT_OF_RANGE")
    return flags


def combine_row_flags(per_sensor_flag_sets: list[set[str]]) -> set[str]:
    """Gabungkan flag dari beberapa sensor dalam satu baris (mis. 6 titik
    suhu R/S/T Atas/Bawah) jadi satu set -- union: satu sensor bermasalah
    saja sudah cukup untuk menandai baris itu bermasalah."""
    combined: set[str] = set()
    for s in per_sensor_flag_sets:
        combined |= s
    return combined


# ════════════════════════════════════════════════════════════════════════
# Aturan BERBASIS JENDELA — butuh riwayat/konteks beberapa sampel
# (dipakai lntai_worker.py, T4).
# ════════════════════════════════════════════════════════════════════════

def detect_stuck(history: list[Optional[float]], cfg: dict = DATA_QUALITY_CONFIG) -> bool:
    """True kalau >= stuck_sample_count sampel TERAKHIR (termasuk sampel
    saat ini di posisi akhir `history`) bernilai identik dan bukan None.
    history diurutkan lama->baru."""
    n = cfg["stuck_sample_count"]
    if len(history) < n:
        return False
    tail = history[-n:]
    if any(v is None for v in tail):
        return False
    return len(set(tail)) == 1


def detect_spike(
    prev: Optional[float], curr: Optional[float], next_: Optional[float],
    cfg: dict = DATA_QUALITY_CONFIG,
) -> bool:
    """True kalau lompatan |curr-prev| melebihi spike_delta_c TAPI sampel
    berikutnya kembali dekat ke nilai sebelum lompatan -- pola khas noise
    sesaat (satu sampel aneh), BUKAN tren naik/turun yang sah (yang akan
    terus berlanjut di next_, bukan kembali)."""
    if prev is None or curr is None or next_ is None:
        return False
    jump = abs(curr - prev)
    revert = abs(next_ - prev)
    return jump > cfg["spike_delta_c"] and revert <= cfg["spike_delta_c"] / 2


def detect_time_gap(dt_seconds: Optional[float], cfg: dict = DATA_QUALITY_CONFIG) -> bool:
    """True kalau jarak waktu ke sampel sebelumnya melebihi
    nominal_sample_interval_seconds * time_gap_multiplier."""
    if dt_seconds is None:
        return False
    return dt_seconds > cfg["nominal_sample_interval_seconds"] * cfg["time_gap_multiplier"]


def detect_pairing_gap(paired: bool) -> bool:
    """paired=False berarti tidak ada baris beban (atau suhu, tergantung
    arah pairing) dalam pairing_tolerance_minutes dari baris ini."""
    return not paired


# ════════════════════════════════════════════════════════════════════════
# Konversi flag <-> bitmask
# ════════════════════════════════════════════════════════════════════════

def flags_to_bitmask(flag_names: set[str] | list[str], cfg: dict = DATA_QUALITY_CONFIG) -> int:
    bitmask = 0
    for name in flag_names:
        bitmask |= cfg["flags"][name]
    return bitmask


def bitmask_to_flags(bitmask: Optional[int], cfg: dict = DATA_QUALITY_CONFIG) -> list[str]:
    if bitmask is None:
        return []
    return sorted(name for name, bit in cfg["flags"].items() if bitmask & bit)


def should_skip_lntai(bitmask: Optional[int], cfg: dict = DATA_QUALITY_CONFIG) -> bool:
    """LNTAI TIDAK dihitung dari input dengan flag MISSING, OUT_OF_RANGE,
    atau STUCK (lihat spesifikasi T5) -- ZERO_READING/SPIKE/PAIRING_GAP/
    TIME_GAP TIDAK menghentikan perhitungan (zero reading tetap valid,
    lihat T2)."""
    if bitmask is None:
        return False
    blocking = cfg["flags"]["MISSING"] | cfg["flags"]["OUT_OF_RANGE"] | cfg["flags"]["STUCK"]
    return bool(bitmask & blocking)
