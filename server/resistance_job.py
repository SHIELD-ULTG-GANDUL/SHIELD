#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""server/resistance_job.py — SHIELD Stage 2, Part D (beta).

Job MINGGUAN (bukan service long-running): untuk tiap (trafo, phase),
hitung drift resistansi kontak relatif (lihat server/resistance.py) untuk
minggu-minggu yang belum ada baris di r_relative_weekly, dari minggu
setelah baris terakhir sampai minggu LALU yang SUDAH SELESAI PENUH (minggu
berjalan selalu dilewati, datanya belum lengkap).

Alur per (trafo, phase, minggu):
  1. Cari model referensi BEKU (ml_models.is_frozen_reference=true,
     ambil yang paling baru kalau lebih dari satu -- seharusnya cuma satu
     per Part B, tapi tidak dipaksa lewat constraint DB). Tidak ada ->
     status='no_reference'. Ada tapi recalibration_note terisi (Part E)
     -> status='needs_recalibration', TIDAK dihitung sama sekali (angka
     dari model yang ditandai perlu dikalibrasi ulang tidak bisa dipercaya).
  2. Kalau model referensi itu ml_gbm (tidak punya koefisien A sendiri),
     "pinjam" A dari model physics_rc LAIN yang terdaftar untuk
     (trafo, phase) yang sama (versi mana pun, aktif atau tidak). Tidak
     ada -> status='no_physics_rc_available'.
  3. Pasangkan baris trafo_temperature+trafo_load minggu itu (two-pointer
     in-memory, sama seperti ml/dataset.py), fold berurutan lewat
     predict_step() model referensi (dan model physics_rc yang dipinjam,
     kalau ada, state TERPISAH) untuk dapat residual + i_eff_norm per
     sampel.
  4. Buang sampel beban rendah (is_low_load) atau tidak stabil
     (is_unsteady, dihitung dari dI/dt terhadap sampel ~tau menit lalu).
  5. Hitung R/R_ref (estimate_r_ratio) per sampel yang lolos, kumpulkan
     median+IQR (weekly_stats) dalam PERSEN. Kurang dari
     min_valid_samples_per_week -> status='insufficient_data'.
  6. Setelah SEMUA 3 fasa trafo itu diproses untuk minggu yang sama,
     hitung phase_relative_ratio_pct (butuh median ketiga fasa).

Dijalankan mingguan lewat cron (lihat README.md), BUKAN service:
    python3 resistance_job.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ml"))

import config  # noqa: E402
import resistance  # noqa: E402
from settings import load_database_settings  # noqa: E402

from dataset import _pair_nearest, _PHASE_CURRENT_COLUMN  # noqa: E402
from models import load_model_from_artifact  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("resistance-job")

TRAFO_IDS: tuple[str, ...] = ("2", "3", "4")
PHASES: tuple[str, ...] = ("R", "S", "T")
_PHASE_TEMP_COLUMNS: dict[str, tuple[str, str]] = {
    "R": ("r_atas", "r_bawah"), "S": ("s_atas", "s_bawah"), "T": ("t_atas", "t_bawah"),
}

_I_RATED: float = config.LNTAI_V2_I_RATED  # 1780A, basis "typical peak" untuk min_load_fraction
_PAIRING_TOLERANCE_MINUTES: float = config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES
_STEADY_TAU_MINUTES: float = resistance.RESISTANCE_CONFIG["steady_load"]["tau_minutes"]
_MIN_VALID_SAMPLES: int = resistance.RESISTANCE_CONFIG["min_valid_samples_per_week"]


def iso_week_start(d: date) -> date:
    """Senin (UTC) minggu yang memuat `d`."""
    return d - timedelta(days=d.weekday())


def _fetch_temperature_rows(cur, trafo: str, ts_from: datetime, ts_to: datetime) -> list[dict]:
    cur.execute(
        """
        SELECT "timestamp", r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah
        FROM trafo_temperature
        WHERE trafo_id = %s AND "timestamp" >= %s AND "timestamp" < %s
        ORDER BY "timestamp" ASC
        """,
        (trafo, ts_from, ts_to),
    )
    return cur.fetchall()


def _fetch_load_rows(cur, trafo: str, ts_from: datetime, ts_to: datetime) -> list[dict]:
    cur.execute(
        """
        SELECT "timestamp", ia, ib, ic
        FROM trafo_load
        WHERE trafo_id = %s AND "timestamp" >= %s AND "timestamp" < %s
        ORDER BY "timestamp" ASC
        """,
        (trafo, ts_from, ts_to),
    )
    return cur.fetchall()


def fetch_frozen_reference(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, model_type, artifact_path, recalibration_note
        FROM ml_models
        WHERE trafo = %s AND phase = %s AND is_frozen_reference = TRUE
        ORDER BY trained_at DESC LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def fetch_any_physics_rc(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, artifact_path
        FROM ml_models
        WHERE trafo = %s AND phase = %s AND model_type = 'physics_rc'
        ORDER BY trained_at DESC LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def _load_artifact(artifact_path: str) -> dict[str, Any]:
    with open(_ROOT / artifact_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _a_ref_and_i_eff_norm(
    model_type: str, frozen_model: Any, frozen_state: dict[str, Any],
    borrowed_model: Optional[Any], borrowed_state: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    """A_ref + i_eff_norm untuk DENOMINATOR estimate_r_ratio(), dari basis
    yang KONSISTEN dengan asal A_ref itu sendiri (lihat server/resistance.py
    docstring modul). TIDAK dipanggil untuk model_type='physics_quadratic'
    -- ditangani inline di compute_week() karena tidak punya state filter
    sama sekali (i_eff_norm = arus sesaat / i_ref langsung)."""
    if model_type == "physics_rc":
        i2_eff = frozen_state.get("i2_eff")
        i_eff_norm = (i2_eff ** 0.5) / frozen_model.i_ref if i2_eff is not None else None
        return frozen_model.a, i_eff_norm
    if model_type == "hybrid":
        physics_state = frozen_state.get("physics", {}) or {}
        i2_eff = physics_state.get("i2_eff")
        i_eff_norm = (i2_eff ** 0.5) / frozen_model.physics.i_ref if i2_eff is not None else None
        return frozen_model.physics.a, i_eff_norm
    if model_type == "ml_gbm":
        if borrowed_model is None:
            return None, None
        i2_eff = borrowed_state.get("i2_eff")
        i_eff_norm = (i2_eff ** 0.5) / borrowed_model.i_ref if i2_eff is not None else None
        return borrowed_model.a, i_eff_norm
    return None, None


def compute_week(cur, trafo: str, phase: str, week_start: date) -> dict[str, Any]:
    """Return dict siap di-INSERT (tanpa phase_relative_ratio_pct, yang
    dihitung belakangan setelah ketiga fasa diproses)."""
    base: dict[str, Any] = {
        "trafo": trafo, "phase": phase, "week_start": week_start,
        "r_ratio_median_pct": None, "r_ratio_q1_pct": None, "r_ratio_q3_pct": None, "r_ratio_iqr_pct": None,
        "n_samples_valid": 0, "n_samples_total": 0, "reference_model_id": None, "a_ref": None,
        "status": "ok", "reason": None,
    }

    ref_row = fetch_frozen_reference(cur, trafo, phase)
    if ref_row is None:
        base["status"] = "no_reference"
        base["reason"] = "No frozen reference baseline model exists for this trafo/phase yet (see Part B ml/train_baseline.py + POST /api/ml/models/{id}/freeze-reference)."
        return base

    base["reference_model_id"] = ref_row["id"]
    if ref_row["recalibration_note"]:
        base["status"] = "needs_recalibration"
        base["reason"] = ref_row["recalibration_note"]
        return base

    model_type = ref_row["model_type"]

    # Cek dulu ketersediaan physics_rc "pinjaman" SEBELUM memuat artefak
    # model beku -- kalau ml_gbm dan tidak ada physics_rc terdaftar,
    # Part D tidak bisa dihitung sama sekali untuk (trafo, phase) ini,
    # jadi tidak perlu membuka/mem-parsing artefak model beku dulu.
    borrowed_model = None
    if model_type == "ml_gbm":
        physics_row = fetch_any_physics_rc(cur, trafo, phase)
        if physics_row is None:
            base["status"] = "no_physics_rc_available"
            base["reason"] = "Reference model is ml_gbm (no A coefficient of its own) and no physics_rc model is registered for this trafo/phase to supply A_ref."
            return base
        borrowed_model = load_model_from_artifact("physics_rc", _load_artifact(physics_row["artifact_path"]))

    frozen_model = load_model_from_artifact(model_type, _load_artifact(ref_row["artifact_path"]))

    week_start_dt = datetime.combine(week_start, datetime.min.time(), tzinfo=timezone.utc)
    week_end_dt = week_start_dt + timedelta(days=7)

    temp_rows = _fetch_temperature_rows(cur, trafo, week_start_dt, week_end_dt)
    load_rows = _fetch_load_rows(cur, trafo, week_start_dt, week_end_dt)
    paired = _pair_nearest(temp_rows, load_rows, _PAIRING_TOLERANCE_MINUTES)

    current_col = _PHASE_CURRENT_COLUMN[phase]
    atas_col, bawah_col = _PHASE_TEMP_COLUMNS[phase]

    frozen_state: dict[str, Any] = {}
    borrowed_state: dict[str, Any] = {}
    roc_history: list[tuple[float, Optional[float]]] = []
    ratios_pct: list[float] = []
    n_total = 0
    a_ref_used: Optional[float] = None

    for t_row, l_row in zip(temp_rows, paired):
        if l_row is None:
            continue
        n_total += 1

        ts_epoch = t_row["timestamp"].timestamp()
        current_amps = l_row.get(current_col)
        atas, bawah = t_row.get(atas_col), t_row.get(bawah_col)
        t_hot = max(atas, bawah) if (atas is not None and bawah is not None) else None

        # dI/dt vs sampel ~tau menit lalu (two-pointer sederhana atas roc_history).
        target_ts = ts_epoch - _STEADY_TAU_MINUTES * 60.0
        past_current = None
        for hist_ts, hist_current in roc_history:
            if hist_ts <= target_ts:
                past_current = hist_current
            else:
                break
        di_dt = None
        if past_current is not None and current_amps is not None:
            di_dt = (current_amps - past_current) / _STEADY_TAU_MINUTES
        roc_history.append((ts_epoch, current_amps))
        cutoff = ts_epoch - (_STEADY_TAU_MINUTES + 10) * 60.0
        roc_history = [(t, c) for (t, c) in roc_history if t >= cutoff]

        if current_amps is None or t_hot is None:
            continue

        row_for_model = {
            "ts_epoch": ts_epoch, "current_amps": current_amps, "ambient_c": 0.0,
            "i2_eff_default_tau": 0.0, "load_roc_1h": 0.0, "hour_sin": 0.0, "hour_cos": 0.0, "weekday": 0,
        }

        if model_type == "physics_quadratic":
            frozen_state, pred = frozen_model.predict_step(frozen_state, row_for_model)
            a_ref = frozen_model.a
            i_eff_norm = current_amps / frozen_model.i_ref if frozen_model.i_ref else None
        else:
            frozen_state, pred = frozen_model.predict_step(frozen_state, row_for_model)
            if model_type == "ml_gbm" and borrowed_model is not None:
                borrowed_state, _ = borrowed_model.predict_step(borrowed_state, row_for_model)
            a_ref, i_eff_norm = _a_ref_and_i_eff_norm(model_type, frozen_model, frozen_state, borrowed_model, borrowed_state)

        if pred is None or a_ref is None or i_eff_norm is None:
            continue

        if resistance.is_low_load(current_amps, _I_RATED):
            continue
        if resistance.is_unsteady(di_dt):
            continue

        residual_c = t_hot - pred
        ratio = resistance.estimate_r_ratio(residual_c, a_ref, i_eff_norm)
        if ratio is None:
            continue

        ratios_pct.append(ratio * 100.0)
        a_ref_used = a_ref

    base["n_samples_total"] = n_total
    base["n_samples_valid"] = len(ratios_pct)
    base["a_ref"] = a_ref_used

    if len(ratios_pct) < _MIN_VALID_SAMPLES:
        base["status"] = "insufficient_data"
        base["reason"] = f"Only {len(ratios_pct)} valid samples this week (minimum {_MIN_VALID_SAMPLES} required, see resistance_config.json min_valid_samples_per_week)."
        return base

    stats = resistance.weekly_stats(ratios_pct)
    base["r_ratio_median_pct"] = stats["median"]
    base["r_ratio_q1_pct"] = stats["q1"]
    base["r_ratio_q3_pct"] = stats["q3"]
    base["r_ratio_iqr_pct"] = stats["iqr"]
    return base


def insert_week_row(cur, row: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO r_relative_weekly (
            trafo, phase, week_start, r_ratio_median_pct, r_ratio_q1_pct, r_ratio_q3_pct,
            r_ratio_iqr_pct, phase_relative_ratio_pct, n_samples_valid, n_samples_total,
            reference_model_id, a_ref, status, reason
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trafo, phase, week_start) DO NOTHING
        """,
        (
            row["trafo"], row["phase"], row["week_start"], row["r_ratio_median_pct"],
            row["r_ratio_q1_pct"], row["r_ratio_q3_pct"], row["r_ratio_iqr_pct"],
            row.get("phase_relative_ratio_pct"), row["n_samples_valid"], row["n_samples_total"],
            row["reference_model_id"], row["a_ref"], row["status"], row["reason"],
        ),
    )


def fetch_last_week(cur, trafo: str, phase: str) -> Optional[date]:
    cur.execute(
        "SELECT week_start FROM r_relative_weekly WHERE trafo = %s AND phase = %s ORDER BY week_start DESC LIMIT 1",
        (trafo, phase),
    )
    row = cur.fetchone()
    return row["week_start"] if row else None


def process_trafo(cur, trafo: str, last_complete_week: date) -> int:
    """Proses minggu demi minggu untuk SEMUA 3 fasa trafo ini bersamaan
    (phase_relative_ratio_pct butuh ketiga fasa minggu yang sama)."""
    starts = {phase: fetch_last_week(cur, trafo, phase) for phase in PHASES}
    # Mulai dari minggu TERTUA yang masih perlu diproses di antara ketiga fasa
    # (fasa yang sudah lebih maju akan diproses ulang untuk minggu itu --
    # tidak masalah, ON CONFLICT DO NOTHING membuatnya idempotent).
    week_start_candidates = [
        (starts[p] + timedelta(days=7)) if starts[p] is not None else None for p in PHASES
    ]
    known_starts = [w for w in week_start_candidates if w is not None]
    week = min(known_starts) if known_starts else None
    if week is None:
        # belum ada baris sama sekali untuk trafo ini -- mulai dari minggu
        # paling awal yang ADA datanya tidak diketahui di sini; pakai
        # last_complete_week - 1 minggu sebagai titik mulai konservatif
        # kalau tidak ada histori r_relative_weekly sama sekali (operator
        # bisa backfill manual lebih jauh kalau perlu).
        week = last_complete_week

    n_written = 0
    while week <= last_complete_week:
        week_rows: dict[str, dict[str, Any]] = {}
        for phase in PHASES:
            week_rows[phase] = compute_week(cur, trafo, phase, week)

        valid_medians = [
            r["r_ratio_median_pct"] for r in week_rows.values()
            if r["status"] == "ok" and r["r_ratio_median_pct"] is not None
        ]
        for phase in PHASES:
            r = week_rows[phase]
            if r["status"] == "ok":
                r["phase_relative_ratio_pct"] = resistance.phase_relative_ratio(
                    r["r_ratio_median_pct"], valid_medians
                )
            insert_week_row(cur, r)
            n_written += 1

        week = week + timedelta(days=7)

    return n_written


def run_once(pool: ConnectionPool) -> int:
    today = datetime.now(timezone.utc).date()
    last_complete_week = iso_week_start(today) - timedelta(days=7)

    total = 0
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for trafo in TRAFO_IDS:
            n = process_trafo(cur, trafo, last_complete_week)
            total += n
            if n:
                log.info("Trafo-%s: %d baris minggu ditulis", trafo, n)
    return total


def main() -> None:
    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)
    log.info("=" * 60)
    log.info("Resistance Job (Stage 2 Part D, beta) -- mulai")
    n = run_once(pool)
    log.info("Resistance Job selesai -- %d baris ditulis", n)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
