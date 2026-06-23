"""Konfigurasi LNTAI untuk SHIELD thermal anomaly monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# Model regresi kuadratik delta suhu terhadap arus ternormalisasi.
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
