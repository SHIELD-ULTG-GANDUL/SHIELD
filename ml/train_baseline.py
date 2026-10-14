#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ml/train_baseline.py — SHIELD Stage 2, Part B.

Melatih dan membandingkan 4 tipe baseline "perilaku sehat" untuk SATU
(trafo, phase) dari data riil di database, dalam satu periode "sehat"
yang Anda tentukan (--healthy-from/--healthy-to -- rentang waktu yang
Anda tahu trafo dalam kondisi baik, BUKAN dipilih otomatis oleh skrip
ini). Split train/validasi TERURUT WAKTU di dalam periode itu (lebih
awal = train, lebih akhir = validasi) -- TIDAK PERNAH diacak, supaya
validasi mencerminkan "bisakah model ini memprediksi kondisi berikutnya
dari pola sebelumnya", bukan interpolasi dalam rentang training.

TIDAK PERNAH mengaktifkan model (is_active selalu FALSE saat training).
Aktivasi dilakukan manual lewat POST /api/ml/models/{id}/activate
(server/api.py) setelah Anda meninjau tabel perbandingan di bawah.

Pemakaian:
    python3 ml/train_baseline.py --trafo 2 --phase R \\
        --healthy-from 2026-07-01 --healthy-to 2026-08-01

    python3 ml/train_baseline.py --trafo 2 --phase R \\
        --healthy-from 2026-07-01 --healthy-to 2026-08-01 --dry-run

--dry-run: bangun dataset + latih semua model + cetak tabel perbandingan,
TANPA menyimpan artefak ke disk maupun menulis baris ke ml_models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as ds  # noqa: E402
import models as mo  # noqa: E402
from settings import load_database_settings  # noqa: E402

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train-baseline")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

# Field yang BOLEH dipakai sebagai fitur -- daftar-putih eksplisit supaya
# "no-leakage test" (lihat test_ml_part_b.py) bisa gagal jelas kalau nanti
# ada yang menambah fitur dari suhu/T masa lalu sensor target tanpa sadar.
ALLOWED_FEATURE_NAMES: frozenset[str] = frozenset(mo.FEATURE_NAMES)
_FORBIDDEN_FEATURE_SUBSTRINGS: tuple[str, ...] = ("temp", "_t_", "target")


def assert_no_target_leakage(feature_names: tuple[str, ...]) -> None:
    """Baseline model TIDAK BOLEH memakai nilai suhu masa lalu sensor
    target sebagai fitur (lihat Stage 2 aturan global #8). Fitur yang
    dipakai (i2_eff_default_tau, load_roc_1h, hour_sin/cos, weekday)
    semuanya berasal dari ARUS/BEBAN dan waktu kalender, BUKAN dari
    riwayat suhu -- dicek eksplisit di sini supaya penambahan fitur baru
    di masa depan tidak diam-diam melanggar aturan ini."""
    for name in feature_names:
        lowered = name.lower()
        if any(bad in lowered for bad in _FORBIDDEN_FEATURE_SUBSTRINGS):
            raise AssertionError(
                f"Potensi kebocoran target: fitur {name!r} tampak berasal dari suhu/target. "
                f"Fitur yang diizinkan: {sorted(ALLOWED_FEATURE_NAMES)}"
            )
        if name not in ALLOWED_FEATURE_NAMES:
            raise AssertionError(
                f"Fitur {name!r} tidak ada di daftar ALLOWED_FEATURE_NAMES -- tinjau dulu "
                f"apakah ini aman (tidak berasal dari suhu masa lalu sensor target) sebelum menambahkannya."
            )


def time_ordered_split(rows: list[dict], train_fraction: float) -> tuple[list[dict], list[dict]]:
    """Split TERURUT WAKTU (rows HARUS sudah terurut ts_epoch naik, lihat
    ml/dataset.py) -- bagian awal (lebih lama) = train, bagian akhir
    (lebih baru) = validasi. TIDAK PERNAH diacak (shuffle)."""
    n = len(rows)
    split_idx = max(1, min(n - 1, int(n * train_fraction))) if n >= 2 else n
    return rows[:split_idx], rows[split_idx:]


def assert_valid_time_ordered_split(train: list[dict], val: list[dict]) -> None:
    """Pastikan train seluruhnya SEBELUM validasi (tidak overlap) dan
    keduanya sendiri terurut naik -- gagal jelas kalau ada yang mengubah
    time_ordered_split() jadi memakai random/shuffle di masa depan."""
    def _is_sorted(rows: list[dict]) -> bool:
        return all(rows[i]["ts_epoch"] <= rows[i + 1]["ts_epoch"] for i in range(len(rows) - 1))

    assert _is_sorted(train), "train tidak terurut waktu"
    assert _is_sorted(val), "validasi tidak terurut waktu"
    if train and val:
        assert train[-1]["ts_epoch"] <= val[0]["ts_epoch"], (
            "train dan validasi overlap/tidak terurut -- train harus SELURUHNYA "
            "sebelum validasi (split waktu-berurutan, bukan acak)"
        )


def _version_string(trained_at: datetime) -> str:
    return "v" + trained_at.strftime("%Y%m%dT%H%M%S")


def train_all_models(
    train_rows: list[dict], val_rows: list[dict], trafo: str, phase: str,
) -> list[dict[str, Any]]:
    """Latih keempat tipe model, return list hasil (satu dict per model,
    termasuk yang gagal -- status dicatat, tidak melempar exception ke
    atas supaya satu model gagal tidak menggagalkan yang lain)."""
    i_ref = ds.config.LNTAI_V3_I_REF  # basis legacy 1000A, satu sumber yang sama dipakai LNTAI v3
    results: list[dict[str, Any]] = []

    def _run(model_type: str, build_fn) -> None:
        try:
            model, fit_info = build_fn()
            preds = model.predict(val_rows)
            metrics = mo.compute_metrics([r["T"] for r in val_rows], preds)
            results.append({
                "model_type": model_type, "status": "ok", "model": model,
                "fit_info": fit_info, "metrics": metrics,
            })
        except Exception as exc:  # noqa: BLE001 -- sengaja lebar, satu model gagal != training lain gagal
            results.append({"model_type": model_type, "status": "failed", "error": str(exc),
                             "model": None, "fit_info": None, "metrics": None})

    _run("physics_quadratic", lambda: (
        lambda m: (m, m.fit(train_rows) or {"i_ref": i_ref})
    )(mo.PhysicsQuadraticModel(i_ref=i_ref)))

    def _fit_physics_rc():
        m = mo.PhysicsRCModel(i_ref=i_ref)
        info = m.fit(train_rows, validation_rows=val_rows)
        return m, info
    _run("physics_rc", _fit_physics_rc)

    def _fit_ml_gbm():
        m = mo.MLGBMModel()
        info = m.fit(train_rows)
        return m, info
    _run("ml_gbm", _fit_ml_gbm)

    def _fit_hybrid():
        m = mo.HybridModel(i_ref=i_ref)
        info = m.fit(train_rows, validation_rows=val_rows)
        return m, info
    _run("hybrid", _fit_hybrid)

    return results


def save_artifact_and_register(
    cur, result: dict[str, Any], trafo: str, phase: str, trained_at: datetime,
    healthy_from: datetime, healthy_to: datetime, dataset_metadata: dict[str, Any],
) -> Optional[int]:
    version = _version_string(trained_at)
    artifact = result["model"].to_artifact()
    filename = f"{trafo}_{phase}_{result['model_type']}_{version}.json"
    artifact_path = ARTIFACTS_DIR / filename
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    cur.execute(
        """
        INSERT INTO ml_models (
            trafo, phase, model_type, version, trained_at, healthy_from, healthy_to,
            features, metrics, artifact_path, sigma, mu, is_active, is_frozen_reference, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, FALSE, %s)
        RETURNING id
        """,
        (
            trafo, phase, result["model_type"], version, trained_at, healthy_from, healthy_to,
            json.dumps({**dataset_metadata, "fit_info": result["fit_info"]}),
            json.dumps(result["metrics"]), str(Path("ml/artifacts") / filename),
            result["metrics"]["sigma"], result["metrics"]["mu"],
            None,
        ),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    header = f"{'model_type':<20} {'status':<8} {'sigma(°C)':>10} {'mu(°C)':>9} {'mae(°C)':>9} {'r2':>7} {'n':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "ok":
            print(f"{r['model_type']:<20} {'FAILED':<8} {r['error']}")
            continue
        m = r["metrics"]
        print(f"{r['model_type']:<20} {'ok':<8} {m['sigma']:>10.4f} {m['mu']:>9.4f} "
              f"{m['mae']:>9.4f} {('%.4f' % m['r2']) if m['r2'] is not None else 'n/a':>7} {m['n']:>6}")


def _parse_date(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trafo", required=True, choices=["2", "3", "4"])
    parser.add_argument("--phase", required=True, choices=["R", "S", "T"])
    parser.add_argument("--healthy-from", dest="healthy_from", required=True, help="ISO8601, mis. 2026-07-01")
    parser.add_argument("--healthy-to", dest="healthy_to", required=True, help="ISO8601, mis. 2026-08-01")
    parser.add_argument("--dry-run", action="store_true", help="Latih & cetak tabel, jangan simpan artefak/baris ml_models")
    args = parser.parse_args()

    ts_from = _parse_date(args.healthy_from)
    ts_to = _parse_date(args.healthy_to)

    db_settings = load_database_settings()
    pool = ConnectionPool(conninfo=db_settings.url, min_size=1, max_size=2, open=True)

    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        rows, metadata = ds.build_phase_dataset(cur, args.trafo, args.phase, ts_from, ts_to)
        log.info("Dataset: %s", json.dumps(metadata, indent=2))

        if len(rows) < 20:
            log.error("Hanya %d baris bersih setelah pembersihan -- terlalu sedikit untuk melatih "
                       "model yang bermakna (minimal yang masuk akal jauh lebih besar dari ini; "
                       "cek rentang healthy-from/to atau konektivitas sensor).", len(rows))
            return

        assert_no_target_leakage(tuple(metadata["feature_names"]))

        train_fraction = ds.ML_CONFIG["split"]["train_fraction"]
        train_rows, val_rows = time_ordered_split(rows, train_fraction)
        assert_valid_time_ordered_split(train_rows, val_rows)
        log.info("Split: %d train, %d validasi (fraction=%.2f)", len(train_rows), len(val_rows), train_fraction)

        results = train_all_models(train_rows, val_rows, args.trafo, args.phase)
        print_comparison_table(results)

        if args.dry_run:
            log.info("--dry-run: TIDAK menyimpan artefak atau menulis ml_models")
            conn.rollback()
            return

        trained_at = datetime.now(timezone.utc)
        saved_ids = []
        for r in results:
            if r["status"] != "ok":
                continue
            model_id = save_artifact_and_register(cur, r, args.trafo, args.phase, trained_at, ts_from, ts_to, metadata)
            saved_ids.append((r["model_type"], model_id))
        conn.commit()

    log.info("=" * 60)
    for model_type, model_id in saved_ids:
        log.info("Tersimpan: %s -> ml_models.id=%s (is_active=false)", model_type, model_id)
    log.info("Aktifkan model lewat: POST /api/ml/models/{id}/activate")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
