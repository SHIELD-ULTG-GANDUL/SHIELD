"""ml/dataset.py — SHIELD Stage 2, Part B.

Membangun dataset bersih PER TRAFO, PER FASA dari database (TIDAK dari
Firestore) untuk melatih baseline "perilaku sehat" (ml/train_baseline.py).

Kebijakan pembersihan di sini SENGAJA LEBIH KETAT daripada
data_quality.should_skip_lntai() yang dipakai lntai_worker.py (lihat T5):
sampel dengan flag MISSING/OUT_OF_RANGE/STUCK/**ZERO_READING** semuanya
dibuang (should_skip_lntai() tidak memblokir ZERO_READING, karena arus=0
adalah bacaan sah untuk SKOR LNTAI realtime -- tapi bukan representasi
"beban sehat normal" yang ingin dipelajari model, jadi dibuang di sini).

Fitur yang BERGANTUNG pada tau (I2_eff) TIDAK dihitung penuh di modul ini
-- hanya versi dengan tau DEFAULT (dari lntai_config.json, satu sumber
yang sama dipakai LNTAI v2/v3) yang disertakan sebagai kolom siap pakai.
physics_rc (ml/train_baseline.py) men-sweep beberapa tau lain lewat grid
search, dan untuk itu re-compute compute_i2_eff_series() dengan tau
kandidat berbeda dari current_amps+ts_epoch mentah yang juga disertakan
di tiap baris -- supaya tidak perlu mengulang query+pairing dari DB untuk
tiap kandidat tau.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from math import cos, exp, pi, sin
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

import config  # noqa: E402
import data_quality as dq  # noqa: E402
import ambient  # noqa: E402

# TIDAK mengimpor dari lntai_worker.py di sini -- lntai_worker.py (Stage 2
# Part B) mengimpor ml/dataset.py (untuk wib_hour_sin_cos_weekday dipakai
# ulang oleh baseline model aktif), jadi arah impor sebaliknya akan
# membuat import circular. Lihat data_quality.combined_quality_flags().
combined_quality_flags = dq.combined_quality_flags

_ML_CONFIG_PATH = Path(__file__).resolve().parent / "ml_config.json"
with open(_ML_CONFIG_PATH, "r", encoding="utf-8") as _f:
    ML_CONFIG: dict[str, Any] = json.load(_f)

PHASES: tuple[str, str, str] = ("R", "S", "T")
_PHASE_CURRENT_COLUMN = {"R": "ia", "S": "ib", "T": "ic"}

_WIB_OFFSET = timedelta(hours=7)  # WIB = UTC+7, tanpa DST (Stage 2 aturan #3)


# ════════════════════════════════════════════════════════════════════════
# Fungsi murni (tanpa DB) — mudah dites terpisah.
# ════════════════════════════════════════════════════════════════════════

def wib_hour_sin_cos_weekday(ts_epoch: float) -> tuple[float, float, int]:
    """Jam (sin/cos, siklik 24 jam) dan hari-dalam-minggu (0=Senin..6=Minggu)
    dalam WIB (UTC+7) -- pola beban/ambient mengikuti jam/hari LOKAL
    pengguna, bukan UTC (lihat Stage 2 aturan global #3)."""
    dt_utc = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
    dt_wib = dt_utc + _WIB_OFFSET
    hour_frac = dt_wib.hour + dt_wib.minute / 60.0 + dt_wib.second / 3600.0
    angle = 2 * pi * hour_frac / 24.0
    return sin(angle), cos(angle), dt_wib.weekday()


def i2_eff_step(
    i2_eff_prev: Optional[float], last_time_prev: Optional[float],
    current_amps: Optional[float], ts_epoch: float,
    tau_minutes: float, max_gap_minutes: float,
) -> tuple[Optional[float], Optional[float]]:
    """SATU langkah filter lag termal I^2 (phi = exp(-dt/tau), reset kalau
    gap > max_gap_minutes) -- rumus yang SAMA PERSIS dengan rekurensi di
    lntai_calculator.lntai_step() per fasa, diimplementasikan ulang di sini
    (bukan memanggil lntai_step) karena konteks pemakaian beda total (array
    historis untuk training/dataset vs satu-langkah-streaming untuk model
    aktif realtime di lntai_worker.py) -- KEDUA implementasi mengkodekan
    fisika yang sama dan harus disinkronkan manual kalau salah satunya
    berubah (lihat test parity di test_lntai_v3.py/test_ml_part_b.py).

    Return (i2_eff_baru, last_time_baru) -- simpan keduanya sebagai state
    untuk panggilan berikutnya (lihat compute_i2_eff_series() untuk versi
    batch, dan ml/models.py PhysicsRCModel.predict_step()/HybridModel
    untuk pemakaian streaming)."""
    if current_amps is None:
        return None, None
    i2_sample = current_amps ** 2
    if i2_eff_prev is None or last_time_prev is None:
        return i2_sample, ts_epoch
    dt_minutes = max(0.0, (ts_epoch - last_time_prev) / 60.0)
    if dt_minutes > max_gap_minutes:
        return i2_sample, ts_epoch
    phi = exp(-dt_minutes / tau_minutes) if tau_minutes > 0 else 0.0
    return phi * i2_eff_prev + (1.0 - phi) * i2_sample, ts_epoch


def compute_i2_eff_series(
    ts_epoch: list[float], currents: list[Optional[float]],
    tau_minutes: float, max_gap_minutes: float,
) -> list[Optional[float]]:
    """Versi BATCH (array) dari i2_eff_step() -- fold berurutan atas
    histori. `currents[i] is None` -> output None di posisi itu DAN
    mereset filter (sampel berikutnya mulai lagi dari nilai sesaatnya),
    sama seperti lntai_step memperlakukan sampel yang datanya tidak
    lengkap."""
    out: list[Optional[float]] = []
    i2_eff: Optional[float] = None
    last_time: Optional[float] = None
    for t, c in zip(ts_epoch, currents):
        i2_eff, last_time = i2_eff_step(i2_eff, last_time, c, t, tau_minutes, max_gap_minutes)
        out.append(i2_eff)
    return out


def _pair_nearest(
    temp_rows: list[dict], load_rows: list[dict], tolerance_minutes: float,
) -> list[Optional[dict]]:
    """Untuk tiap baris di `temp_rows` (terurut naik berdasar timestamp),
    cari baris di `load_rows` (juga terurut naik) yang PALING DEKAT waktu,
    dalam jarak `tolerance_minutes`. Two-pointer O(n+m) -- jauh lebih cepat
    daripada satu query SQL "ORDER BY ABS(...)" per baris suhu (dipakai
    lntai_worker.py untuk batch kecil realtime, tapi terlalu lambat untuk
    membangun dataset atas rentang waktu panjang di sini)."""
    tolerance_s = tolerance_minutes * 60.0
    paired: list[Optional[dict]] = []
    j = 0
    m = len(load_rows)
    for t in temp_rows:
        t_epoch = t["timestamp"].timestamp()
        while j + 1 < m and load_rows[j + 1]["timestamp"].timestamp() <= t_epoch:
            j += 1
        candidates = [load_rows[j]] if j < m else []
        if j + 1 < m:
            candidates.append(load_rows[j + 1])
        best = None
        best_dt = None
        for c in candidates:
            dt = abs(c["timestamp"].timestamp() - t_epoch)
            if best_dt is None or dt < best_dt:
                best, best_dt = c, dt
        paired.append(best if (best is not None and best_dt <= tolerance_s) else None)
    return paired


def _flags_to_drop(quality_flags: int, drop_flag_names: list[str]) -> bool:
    present = set(dq.bitmask_to_flags(quality_flags))
    return bool(present.intersection(drop_flag_names))


# ════════════════════════════════════════════════════════════════════════
# Orkestrasi (butuh DB cursor) — satu fungsi, satu query besar + pairing
# in-memory, bukan N query kecil per baris.
# ════════════════════════════════════════════════════════════════════════

def _fetch_temperature_rows(cur, trafo: str, ts_from: datetime, ts_to: datetime) -> list[dict]:
    cur.execute(
        """
        SELECT "timestamp", r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah, quality_flags
        FROM trafo_temperature
        WHERE trafo_id = %s AND "timestamp" >= %s AND "timestamp" <= %s
        ORDER BY "timestamp" ASC
        """,
        (trafo, ts_from, ts_to),
    )
    return cur.fetchall()


def _fetch_load_rows(cur, trafo: str, ts_from: datetime, ts_to: datetime) -> list[dict]:
    cur.execute(
        """
        SELECT "timestamp", ia, ib, ic, quality_flags
        FROM trafo_load
        WHERE trafo_id = %s AND "timestamp" >= %s AND "timestamp" <= %s
        ORDER BY "timestamp" ASC
        """,
        (trafo, ts_from, ts_to),
    )
    return cur.fetchall()


def _ambient_coverage(cur, ts_from: datetime, ts_to: datetime) -> float:
    """Proporsi slot 10-menit dalam [ts_from, ts_to] yang punya baris
    ambient_logs ok=true -- dipakai menentukan ambient_source dataset
    (lihat ML_CONFIG["dataset"]["ambient_coverage_threshold"])."""
    total_minutes = max((ts_to - ts_from).total_seconds() / 60.0, 1.0)
    total_slots = max(int(total_minutes // 10), 1)
    cur.execute(
        """
        SELECT COUNT(DISTINCT slot_10min) AS n
        FROM ambient_logs
        WHERE ok = TRUE AND "timestamp" BETWEEN %s AND %s
        """,
        (ts_from, ts_to),
    )
    row = cur.fetchone()
    covered = (row["n"] if row else 0) or 0
    return min(covered / total_slots, 1.0)


def build_phase_dataset(
    cur, trafo: str, phase: str, ts_from: datetime, ts_to: datetime,
    cfg: dict[str, Any] = ML_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Dataset bersih untuk satu (trafo, phase) dalam rentang
    [ts_from, ts_to]. Tiap baris hasil:
        ts_epoch, T (target), current_amps, i2_eff_default_tau,
        load_roc_1h, hour_sin, hour_cos, weekday, ambient_c, quality_flags
    """
    if phase not in PHASES:
        raise ValueError(f"phase harus salah satu dari {PHASES}, dapat {phase!r}")

    drop_flag_names = cfg["dataset"]["drop_flags"]
    roc_window_minutes = cfg["dataset"]["load_roc_window_minutes"]
    roc_tolerance_minutes = cfg["dataset"]["load_roc_match_tolerance_minutes"]
    coverage_threshold = cfg["dataset"]["ambient_coverage_threshold"]
    pairing_tolerance = config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES
    tau_default = config.LNTAI_V2_TAU_MINUTES
    max_gap_default = config.LNTAI_V2_MAX_GAP_MINUTES
    ambient_default_c = config.LNTAI_V3_AMBIENT_C
    current_col = _PHASE_CURRENT_COLUMN[phase]

    temp_rows = _fetch_temperature_rows(cur, trafo, ts_from, ts_to)
    load_rows = _fetch_load_rows(cur, trafo, ts_from, ts_to)
    n_temp_raw = len(temp_rows)

    paired_loads = _pair_nearest(temp_rows, load_rows, pairing_tolerance)

    coverage = _ambient_coverage(cur, ts_from, ts_to)
    ambient_source = "measured" if coverage >= coverage_threshold else "default"

    kept_ts: list[float] = []
    kept_current: list[Optional[float]] = []
    kept_temp_row: list[dict] = []
    kept_quality_flags: list[int] = []
    n_paired = 0
    n_dropped_by_flags = 0

    for t_row, l_row in zip(temp_rows, paired_loads):
        if l_row is None:
            continue
        n_paired += 1
        qf = combined_quality_flags(t_row, l_row, paired=True)
        if _flags_to_drop(qf, drop_flag_names):
            n_dropped_by_flags += 1
            continue
        kept_ts.append(t_row["timestamp"].timestamp())
        kept_current.append(l_row.get(current_col))
        kept_temp_row.append(t_row)
        kept_quality_flags.append(qf)

    i2_eff_series = compute_i2_eff_series(kept_ts, kept_current, tau_default, max_gap_default)

    # Ambient: satu sumber untuk SELURUH dataset (bukan campuran terukur/
    # default per baris) -- lihat catatan ambient_coverage_threshold.
    ambient_lookup_cache: dict[float, Optional[float]] = {}

    def _ambient_for(ts_epoch: float) -> float:
        if ambient_source == "default":
            return ambient_default_c
        if ts_epoch not in ambient_lookup_cache:
            ts_dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
            value, _source = ambient.ambient_at(
                cur, ts_dt, int(config.LNTAI_V3_CONFIG.get("rolling_median_minutes", 30)),
                fallback_default_c=ambient_default_c,
            )
            ambient_lookup_cache[ts_epoch] = value
        return ambient_lookup_cache[ts_epoch] if ambient_lookup_cache[ts_epoch] is not None else ambient_default_c

    load_roc_tolerance_s = roc_tolerance_minutes * 60.0
    roc_window_s = roc_window_minutes * 60.0

    rows: list[dict[str, Any]] = []
    n_dropped_missing_roc = 0
    j_roc = 0  # pointer untuk mencari sampel ~1 jam lalu, dua-pointer (kept_ts sudah terurut)

    for idx in range(len(kept_ts)):
        ts_epoch = kept_ts[idx]
        current_amps = kept_current[idx]
        i2_eff = i2_eff_series[idx]
        if current_amps is None or i2_eff is None:
            continue

        target_ts = ts_epoch - roc_window_s
        while j_roc + 1 < idx and kept_ts[j_roc + 1] <= target_ts:
            j_roc += 1
        roc_candidate_idx = j_roc if j_roc < idx else None
        if roc_candidate_idx is None or abs(kept_ts[roc_candidate_idx] - target_ts) > load_roc_tolerance_s:
            n_dropped_missing_roc += 1
            continue
        past_current = kept_current[roc_candidate_idx]
        if past_current is None:
            n_dropped_missing_roc += 1
            continue
        load_roc_1h = (current_amps - past_current) / (roc_window_minutes / 60.0)

        hour_sin, hour_cos, weekday = wib_hour_sin_cos_weekday(ts_epoch)
        temp_row = kept_temp_row[idx]
        t_target = max(temp_row[f"{phase.lower()}_atas"], temp_row[f"{phase.lower()}_bawah"])

        rows.append({
            "ts_epoch": ts_epoch,
            "trafo": trafo,
            "phase": phase,
            "T": t_target,
            "current_amps": current_amps,
            "i2_eff_default_tau": i2_eff,
            "load_roc_1h": load_roc_1h,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "weekday": weekday,
            "ambient_c": _ambient_for(ts_epoch),
            "quality_flags": kept_quality_flags[idx],
        })

    metadata = {
        "trafo": trafo, "phase": phase,
        "healthy_from": ts_from.isoformat(), "healthy_to": ts_to.isoformat(),
        "n_temp_rows_raw": n_temp_raw,
        "n_paired": n_paired,
        "n_dropped_by_flags": n_dropped_by_flags,
        "n_dropped_missing_roc_history": n_dropped_missing_roc,
        "n_final": len(rows),
        "drop_flags": drop_flag_names,
        "ambient_source": ambient_source,
        "ambient_coverage": coverage,
        "ambient_coverage_threshold": coverage_threshold,
        "tau_minutes_default": tau_default,
        "max_gap_minutes_default": max_gap_default,
        "feature_names": ["i2_eff_default_tau", "load_roc_1h", "hour_sin", "hour_cos", "weekday"],
    }
    return rows, metadata
