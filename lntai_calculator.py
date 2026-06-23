"""LNTAI calculator untuk SHIELD.

LNTAI bukan hanya alarm suhu absolut. LNTAI membandingkan kenaikan suhu
aktual terhadap kenaikan suhu yang diprediksi dari arus beban. Semakin besar
residual positif, semakin besar indikasi anomali thermal akibat potensi
kenaikan tahanan kontak.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Optional

import config


class LNTAICalculator:
    """Kalkulator Local Normalized Thermal Anomaly Index."""

    def __init__(
        self,
        a: float = config.A,
        b: float = config.B,
        sigma: float = config.SIGMA,
        default_t_ambient: float = config.T_AMBIENT,
        default_i_rated: float = config.I_RATED,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.default_t_ambient = float(default_t_ambient)
        self.default_i_rated = float(default_i_rated)

    def calculate(
        self,
        t_contact: Any,
        i_load: Any,
        t_ambient: Optional[Any] = None,
        i_rated: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Hitung LNTAI dan kembalikan dictionary siap JSON."""

        timestamp = self._timestamp()
        try:
            t_contact_f = self._to_float(t_contact, "t_contact")
            i_load_f = self._to_float(i_load, "i_load")
            t_ambient_f = self._to_float(
                self.default_t_ambient if t_ambient is None else t_ambient,
                "t_ambient",
            )
            i_rated_f = self._to_float(
                self.default_i_rated if i_rated is None else i_rated,
                "i_rated",
            )
            if i_rated_f == 0:
                raise ValueError("i_rated tidak boleh bernilai 0")
            if self.sigma == 0:
                raise ValueError("sigma baseline tidak boleh bernilai 0")

            delta_t_real = t_contact_f - t_ambient_f
            i_norm = i_load_f / i_rated_f
            delta_t_pred = self.a * (i_norm**2) + self.b
            residual = delta_t_real - delta_t_pred
            lntai = residual / self.sigma
            level = self._classify(lntai)
            meta = config.LEVELS[level]

            return {
                "t_contact": self._round(t_contact_f),
                "t_ambient": self._round(t_ambient_f),
                "i_load": self._round(i_load_f),
                "i_rated": self._round(i_rated_f),
                "i_norm": round(i_norm, 4),
                "delta_T_real": self._round(delta_t_real),
                "delta_T_pred": self._round(delta_t_pred),
                "residual": self._round(residual),
                "lntai": self._round(lntai),
                "level": meta.name,
                "icon": meta.icon,
                "color": meta.color,
                "recommendation": meta.recommendation,
                "timestamp": timestamp,
            }
        except (TypeError, ValueError) as exc:
            meta = config.LEVELS["DATA_INVALID"]
            return {
                "t_contact": self._safe_value(t_contact),
                "t_ambient": self._safe_value(t_ambient if t_ambient is not None else self.default_t_ambient),
                "i_load": self._safe_value(i_load),
                "i_rated": self._safe_value(i_rated if i_rated is not None else self.default_i_rated),
                "i_norm": None,
                "delta_T_real": None,
                "delta_T_pred": None,
                "residual": None,
                "lntai": None,
                "level": meta.name,
                "icon": meta.icon,
                "color": meta.color,
                "recommendation": meta.recommendation,
                "timestamp": timestamp,
                "error": str(exc),
            }

    def calculate_batch(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Hitung LNTAI untuk banyak payload sensor."""

        if not isinstance(data_list, list):
            return [self._invalid_batch_result("data_list harus berupa list")]

        results: list[dict[str, Any]] = []
        for item in data_list:
            if not isinstance(item, dict):
                results.append(self._invalid_batch_result("item batch harus berupa dict"))
                continue
            results.append(
                self.calculate(
                    item.get("t_contact"),
                    item.get("i_load"),
                    item.get("t_ambient"),
                    item.get("i_rated"),
                )
            )
        return results

    def calculate_three_phase(self, data: dict[str, Any]) -> dict[str, Any]:
        """Hitung LNTAI dari rata-rata kontak finger tiga fasa."""

        timestamp = self._timestamp()
        try:
            if not isinstance(data, dict):
                raise ValueError("data tiga fasa harus berupa dict")

            r_upper = self._to_float(data.get("R_upper"), "R_upper")
            r_lower = self._to_float(data.get("R_lower"), "R_lower")
            s_upper = self._to_float(data.get("S_upper"), "S_upper")
            s_lower = self._to_float(data.get("S_lower"), "S_lower")
            t_upper = self._to_float(data.get("T_upper"), "T_upper")
            t_lower = self._to_float(data.get("T_lower"), "T_lower")
            i_r = self._to_float(data.get("I_R"), "I_R")
            i_s = self._to_float(data.get("I_S"), "I_S")
            i_t = self._to_float(data.get("I_T"), "I_T")

            phase_temps = {
                "R": (r_upper + r_lower) / 2,
                "S": (s_upper + s_lower) / 2,
                "T": (t_upper + t_lower) / 2,
            }
            hottest_phase = max(phase_temps, key=phase_temps.get)
            t_contact = phase_temps[hottest_phase]
            i_load = max(i_r, i_s, i_t)
            result = self.calculate(
                t_contact=t_contact,
                i_load=i_load,
                t_ambient=data.get("T_ambient"),
                i_rated=data.get("I_rated"),
            )

            return {
                "T_R": self._round(phase_temps["R"]),
                "T_S": self._round(phase_temps["S"]),
                "T_T": self._round(phase_temps["T"]),
                "hottest_phase": hottest_phase,
                "delta_phase": self._round(max(phase_temps.values()) - min(phase_temps.values())),
                "lntai_result": result,
                "timestamp": timestamp,
            }
        except (TypeError, ValueError) as exc:
            meta = config.LEVELS["DATA_INVALID"]
            return {
                "T_R": None,
                "T_S": None,
                "T_T": None,
                "hottest_phase": None,
                "delta_phase": None,
                "lntai_result": {
                    "level": meta.name,
                    "icon": meta.icon,
                    "color": meta.color,
                    "recommendation": meta.recommendation,
                    "timestamp": timestamp,
                    "error": str(exc),
                },
                "timestamp": timestamp,
            }

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _round(value: float) -> float:
        return round(value, 2)

    @staticmethod
    def _safe_value(value: Any) -> Any:
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _to_float(value: Any, field_name: str) -> float:
        if value is None:
            raise ValueError(f"{field_name} tidak boleh None")
        if isinstance(value, str) and value.strip() == "":
            raise ValueError(f"{field_name} tidak boleh string kosong")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} harus numerik") from exc
        if not isfinite(number):
            raise ValueError(f"{field_name} harus finite")
        return number

    @staticmethod
    def _classify(lntai: float) -> str:
        if lntai >= config.THRESHOLDS["critical"]:
            return "KRITIS"
        if lntai >= config.THRESHOLDS["alert"]:
            return "SIAGA"
        if lntai >= config.THRESHOLDS["attention"]:
            return "PERHATIAN"
        return "NORMAL"

    def _invalid_batch_result(self, message: str) -> dict[str, Any]:
        meta = config.LEVELS["DATA_INVALID"]
        return {
            "t_contact": None,
            "t_ambient": None,
            "i_load": None,
            "i_rated": None,
            "i_norm": None,
            "delta_T_real": None,
            "delta_T_pred": None,
            "residual": None,
            "lntai": None,
            "level": meta.name,
            "icon": meta.icon,
            "color": meta.color,
            "recommendation": meta.recommendation,
            "timestamp": self._timestamp(),
            "error": message,
        }
