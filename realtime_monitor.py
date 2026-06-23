"""Contoh monitoring realtime LNTAI.

Jalankan:
    python realtime_monitor.py
"""

from __future__ import annotations

import json
import time
from itertools import cycle
from typing import Any

from lntai_calculator import LNTAICalculator


def get_latest_sensor_data() -> dict[str, Any]:
    """Placeholder integrasi sensor/MQTT/Firestore.

    Ganti fungsi ini dengan pembacaan data real-time dari dashboard SHIELD.
    """

    samples = [
        {"t_contact": 58, "i_load": 780, "t_ambient": 30},
        {"t_contact": 72, "i_load": 850, "t_ambient": 30},
        {"t_contact": 84, "i_load": 620, "t_ambient": 30},
        {"t_contact": 90, "i_load": 1000, "t_ambient": 30},
    ]
    if not hasattr(get_latest_sensor_data, "_sample_cycle"):
        get_latest_sensor_data._sample_cycle = cycle(samples)  # type: ignore[attr-defined]
    return next(get_latest_sensor_data._sample_cycle)  # type: ignore[attr-defined]


def main(interval_seconds: float = 3.0) -> None:
    calculator = LNTAICalculator()
    print("SHIELD LNTAI realtime monitor aktif. Tekan Ctrl+C untuk berhenti.")

    try:
        while True:
            payload = get_latest_sensor_data()
            result = calculator.calculate(
                t_contact=payload.get("t_contact"),
                i_load=payload.get("i_load"),
                t_ambient=payload.get("t_ambient"),
                i_rated=payload.get("i_rated"),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if result.get("level") in {"SIAGA", "KRITIS"}:
                print(f"ALERT {result['icon']} {result['level']}: {result['recommendation']}")

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nMonitoring LNTAI dihentikan.")


if __name__ == "__main__":
    main()
