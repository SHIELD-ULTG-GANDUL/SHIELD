"""server/explain.py — SHIELD Stage 2, Part G.

CUSUM satu-sisi (mendeteksi KENAIKAN residual yang konsisten -- penurunan
TIDAK memicu, karena residual yang menurun bukan sinyal degradasi termal
yang diwaspadai di sini) + perakitan hasil "explain" (measured hotspot,
predicted healthy, unexplained residual, kontribusi per kategori -- lihat
ml/explain.py untuk formula kontribusi itu sendiri).

    S[k] = max(0, S[k-1] + (residual[k] - mu0 - k_slack))
    triggered = S[k] > h

Parameter (mu0, k_slack, h) di explain_config.json, TODO(calibration) --
tidak ada nilai yang dikarang di sini. CUSUM TIDAK disimpan sebagai state
persisten di tabel mana pun -- dihitung ulang (fold) dari histori residual
setiap kali GET /api/explain/latest diminta (lihat compute_cusum_over_history()),
cukup murah untuk lookback_days (config) hari data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_CONFIG_PATH = Path(__file__).resolve().parent / "explain_config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    EXPLAIN_CONFIG: dict[str, Any] = json.load(_f)


def cusum_step(
    state: dict[str, Any], residual: float, ts: Any, cfg: dict[str, Any] = EXPLAIN_CONFIG,
) -> tuple[dict[str, Any], bool, Optional[Any]]:
    """Satu langkah CUSUM murni. `state`: {"s": float, "start_date": Any|None}.
    Return (new_state, triggered, start_date_kalau_triggered_else_None)."""
    k_slack = cfg["cusum"]["k_slack"]
    h = cfg["cusum"]["h"]
    mu0 = cfg["cusum"]["mu0"]

    s_prev = state.get("s", 0.0)
    start_date_prev = state.get("start_date")

    s = max(0.0, s_prev + (residual - mu0 - k_slack))

    if s > 0.0 and s_prev == 0.0:
        start_date = ts  # satu run baru dimulai dari sini
    elif s == 0.0:
        start_date = None  # reset -- tidak ada run aktif
    else:
        start_date = start_date_prev

    triggered = s > h
    new_state = {"s": s, "start_date": start_date}
    return new_state, triggered, (start_date if triggered else None)


def compute_cusum_over_history(
    residual_history: list[tuple[Any, float]], cfg: dict[str, Any] = EXPLAIN_CONFIG,
) -> tuple[bool, Optional[Any]]:
    """`residual_history`: [(ts, residual), ...] TERURUT NAIK waktu.
    Return (triggered_di_akhir_seri, start_date_kalau_triggered)."""
    state: dict[str, Any] = {}
    triggered = False
    start_date = None
    for ts, residual in residual_history:
        state, triggered, candidate = cusum_step(state, residual, ts, cfg)
        start_date = candidate if triggered else None
    return triggered, start_date


def build_explanation(
    measured_hotspot_c: float, predicted_healthy_c: float, contributions_c: dict[str, float],
    contribution_method: str, cusum_triggered: bool, cusum_start_date: Optional[Any],
) -> dict[str, Any]:
    unexplained_residual_c = measured_hotspot_c - predicted_healthy_c
    cusum_message = None
    if cusum_triggered and cusum_start_date is not None:
        start_str = cusum_start_date.isoformat() if hasattr(cusum_start_date, "isoformat") else str(cusum_start_date)
        cusum_message = f"Consistent rise since {start_str}"
    return {
        "measured_hotspot_c": measured_hotspot_c,
        "predicted_healthy_c": predicted_healthy_c,
        "unexplained_residual_c": unexplained_residual_c,
        "contributions_c": contributions_c,
        "contribution_method": contribution_method,
        "cusum_triggered": cusum_triggered,
        "cusum_message": cusum_message,
    }
