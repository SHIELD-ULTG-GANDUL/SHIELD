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


def time_series_folds(
    rows: list[dict], n_folds: int = 4, gap_minutes: float = 60.0,
) -> list[tuple[list[dict], list[dict]]]:
    """M12 (autopilot 2026-11): expanding-window time-series CV -- fold k
    pakai SEMUA baris sebelum blok validasi fold itu sebagai train (makin
    besar tiap fold berikutnya), BUKAN sliding window ukuran tetap,
    mencerminkan skenario nyata "makin banyak histori makin baik". Antara
    akhir train dan awal validasi ADA jeda >= gap_minutes (menghindari
    leakage dari titik yang nyaris bersebelahan waktu, mis. fitur I2_eff
    berjendela yang "mengintip" beberapa menit ke depan/belakang).

    `rows` HARUS sudah terurut ts_epoch naik (sama seperti
    time_ordered_split()). Return list (train_rows, val_rows) -- fold
    dengan validasi kosong (semua baris di blok itu kena potong gap)
    DIBUANG, bukan disertakan dengan val_rows=[]."""
    n = len(rows)
    if n < n_folds + 1 or n_folds < 1:
        return []
    block_size = n // (n_folds + 1)
    gap_seconds = gap_minutes * 60.0
    folds: list[tuple[list[dict], list[dict]]] = []
    for k in range(1, n_folds + 1):
        train_end = block_size * k
        val_block_end = block_size * (k + 1) if k < n_folds else n
        train_rows = rows[:train_end]
        if not train_rows:
            continue
        cutoff = train_rows[-1]["ts_epoch"] + gap_seconds
        val_rows = [r for r in rows[train_end:val_block_end] if r["ts_epoch"] >= cutoff]
        if val_rows:
            folds.append((train_rows, val_rows))
    return folds


def assert_folds_no_overlap(folds: list[tuple[list[dict], list[dict]]]) -> None:
    """Tiap fold: train seluruhnya sebelum validasi fold itu sendiri (tidak
    overlap) -- tidak membandingkan ANTAR fold (fold berikutnya WAJAR
    overlap dengan train fold sebelumnya, itu intinya expanding window)."""
    for train_rows, val_rows in folds:
        assert_valid_time_ordered_split(train_rows, val_rows)


def _version_string(trained_at: datetime) -> str:
    return "v" + trained_at.strftime("%Y%m%dT%H%M%S")


# ════════════════════════════════════════════════════════════════════════
# naive_median — M12 (autopilot 2026-11): baseline TRIVIAL untuk tabel
# perbandingan (ΔT_pred = median ΔT training, konstan, tidak peduli
# beban/waktu sama sekali). TIDAK disimpan ke ml_models (lihat skip_save
# di train_all_models() -- model_type ini TIDAK ada di CHECK constraint
# ml_models.model_type dan memang tidak dimaksudkan bisa diaktifkan,
# murni patokan "seberapa jauh model sungguhan lebih baik dari tebakan
# paling sederhana yang masuk akal").
# ════════════════════════════════════════════════════════════════════════

class NaiveMedianModel:
    model_type = "naive_median"

    def __init__(self) -> None:
        self.median_delta_t: Optional[float] = None

    def fit(self, rows: list[dict]) -> dict[str, Any]:
        if not rows:
            raise ValueError("Butuh >= 1 baris untuk median ΔT")
        delta_ts = sorted(r["T"] - r["ambient_c"] for r in rows)
        n = len(delta_ts)
        self.median_delta_t = delta_ts[n // 2] if n % 2 else (delta_ts[n // 2 - 1] + delta_ts[n // 2]) / 2.0
        return {"median_delta_t": self.median_delta_t, "n_train": n}

    def predict(self, rows: list[dict]) -> list[float]:
        if self.median_delta_t is None:
            raise RuntimeError("Model belum di-fit")
        return [self.median_delta_t + row["ambient_c"] for row in rows]

    def to_artifact(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "median_delta_t": self.median_delta_t}

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "NaiveMedianModel":
        m = cls()
        m.median_delta_t = artifact["median_delta_t"]
        return m


def _build_model(model_type: str, i_ref: float):
    if model_type == "physics_quadratic":
        return mo.PhysicsQuadraticModel(i_ref=i_ref)
    if model_type == "physics_rc":
        return mo.PhysicsRCModel(i_ref=i_ref)
    if model_type == "ml_gbm":
        return mo.MLGBMModel()
    if model_type == "hybrid":
        return mo.HybridModel(i_ref=i_ref)
    if model_type == "naive_median":
        return NaiveMedianModel()
    raise ValueError(f"model_type tidak dikenal: {model_type!r}")


def _fit_model(
    model_type: str, model: Any, train_rows: list[dict],
    validation_rows: Optional[list[dict]] = None, gbm_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Satu titik panggilan `.fit()` yang tahu signature tiap model_type --
    dipakai baik untuk fit "headline" (train_all_models) maupun tiap fold
    CV (_cv_metrics), supaya keduanya PASTI konsisten caranya memanggil
    fit, bukan dua implementasi yang bisa diam-diam menyimpang."""
    if model_type == "physics_quadratic":
        return model.fit(train_rows) or {}
    if model_type in ("physics_rc", "hybrid"):
        return model.fit(train_rows, validation_rows=validation_rows)
    if model_type == "ml_gbm":
        return model.fit(train_rows, **(gbm_overrides or {}))
    if model_type == "naive_median":
        return model.fit(train_rows)
    raise ValueError(f"model_type tidak dikenal: {model_type!r}")


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n if n > 1 else 0.0
    return mean, var ** 0.5


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def cv_metrics(
    model_type: str, i_ref: float, rows: list[dict], n_folds: int = 4, gap_minutes: float = 60.0,
    gbm_overrides: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """M12: mean/std/median sigma & MAE lewat time_series_folds() -- fold
    yang gagal fit/predict (mis. terlalu sedikit data untuk model_type
    tertentu) DILEWATI, bukan menggagalkan fold lain; None kalau TIDAK ADA
    fold yang berhasil sama sekali (bukan 0/kosong yang menyesatkan)."""
    folds = time_series_folds(rows, n_folds=n_folds, gap_minutes=gap_minutes)
    sigmas, maes = [], []
    n_failed = 0
    for train_rows, val_rows in folds:
        try:
            model = _build_model(model_type, i_ref)
            _fit_model(model_type, model, train_rows, validation_rows=val_rows, gbm_overrides=gbm_overrides)
            preds = model.predict(val_rows)
            m = mo.compute_metrics([r["T"] for r in val_rows], preds)
            if m["sigma"] is not None:
                sigmas.append(m["sigma"])
                maes.append(m["mae"])
            else:
                n_failed += 1
        except Exception:  # noqa: BLE001 -- satu fold gagal TIDAK boleh menggagalkan fold lain
            n_failed += 1
    if not sigmas:
        return None
    sigma_mean, sigma_std = _mean_std(sigmas)
    mae_mean, mae_std = _mean_std(maes)
    return {
        "n_folds_total": len(folds), "n_folds_ok": len(sigmas), "n_folds_failed": n_failed,
        "sigma_mean": sigma_mean, "sigma_std": sigma_std, "sigma_median": _median(sigmas),
        "mae_mean": mae_mean, "mae_std": mae_std, "mae_median": _median(maes),
    }


# M12 --tune: grid GBM -- TODO(verifikasi), tiga sumbu dipilih sebagai
# rentang masuk akal di sekitar default ml_config.json (max_depth=4,
# learning_rate=0.05), bukan hasil studi/eksperimen sebelumnya.
GBM_TUNE_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_samples_leaf": [20, 40, 80],
}


def tune_gbm_hyperparameters(
    train_rows: list[dict], i_ref: float, n_folds: int = 4, gap_minutes: float = 60.0,
) -> dict[str, Any]:
    """Grid search max_depth x learning_rate x min_samples_leaf (27
    kombinasi) lewat CV pada TRAIN SAJA (val_rows "headline" tidak pernah
    disentuh tuning -- tetap jadi ukuran akhir yang independen dari proses
    pemilihan hyperparameter). Pilih kombinasi dengan MEDIAN sigma CV
    TERKECIL (median, bukan mean -- tahan terhadap satu fold outlier).
    ml_config.json TIDAK diubah oleh fungsi ini (override per-trial lewat
    kwargs MLGBMModel.fit(), lihat M12)."""
    import itertools

    trials: list[dict[str, Any]] = []
    for max_depth, lr, min_leaf in itertools.product(
        GBM_TUNE_GRID["max_depth"], GBM_TUNE_GRID["learning_rate"], GBM_TUNE_GRID["min_samples_leaf"],
    ):
        overrides = {"max_depth": max_depth, "learning_rate": lr, "min_samples_leaf": min_leaf}
        cv = cv_metrics("ml_gbm", i_ref, train_rows, n_folds=n_folds, gap_minutes=gap_minutes, gbm_overrides=overrides)
        trials.append({**overrides, "cv": cv})

    valid_trials = [t for t in trials if t["cv"] is not None]
    if not valid_trials:
        log.warning("--tune: semua %d kombinasi grid gagal CV (data CV terlalu sedikit?) -- pakai default ml_config.json", len(trials))
        return {"best_params": None, "all_trials": trials, "selected": "default (ml_config.json, grid gagal total)"}

    best = min(valid_trials, key=lambda t: t["cv"]["sigma_median"])
    best_params = {"max_depth": best["max_depth"], "learning_rate": best["learning_rate"], "min_samples_leaf": best["min_samples_leaf"]}
    return {"best_params": best_params, "all_trials": trials, "selected": best_params}


def train_all_models(
    train_rows: list[dict], val_rows: list[dict], trafo: str, phase: str,
    cv_n_folds: int = 4, cv_gap_minutes: float = 60.0, gbm_overrides: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Latih keempat tipe model baseline + naive_median (pembanding, lihat
    docstring NaiveMedianModel), return list hasil (satu dict per model,
    termasuk yang gagal -- status dicatat, tidak melempar exception ke
    atas supaya satu model gagal tidak menggagalkan yang lain). Tiap hasil
    "ok" juga dapat metrics["cv"] (mean/std/median sigma & MAE lewat
    time_series_folds() pada train_rows+val_rows gabungan -- M12)."""
    i_ref = ds.config.LNTAI_V3_I_REF  # basis legacy 1000A, satu sumber yang sama dipakai LNTAI v3
    all_rows_for_cv = train_rows + val_rows  # CV expanding-window pakai SELURUH rentang "sehat", bukan cuma train headline
    results: list[dict[str, Any]] = []

    def _run(model_type: str, skip_save: bool = False, gbm_overrides_for_fit: Optional[dict[str, Any]] = None) -> None:
        try:
            model = _build_model(model_type, i_ref)
            fit_info = _fit_model(model_type, model, train_rows, validation_rows=val_rows, gbm_overrides=gbm_overrides_for_fit)
            preds = model.predict(val_rows)
            metrics = mo.compute_metrics([r["T"] for r in val_rows], preds)
            metrics["cv"] = cv_metrics(
                model_type, i_ref, all_rows_for_cv, n_folds=cv_n_folds, gap_minutes=cv_gap_minutes,
                gbm_overrides=gbm_overrides_for_fit,
            )
            results.append({
                "model_type": model_type, "status": "ok", "model": model,
                "fit_info": fit_info, "metrics": metrics, "skip_save": skip_save,
            })
        except Exception as exc:  # noqa: BLE001 -- sengaja lebar, satu model gagal != training lain gagal
            results.append({"model_type": model_type, "status": "failed", "error": str(exc),
                             "model": None, "fit_info": None, "metrics": None, "skip_save": skip_save})

    _run("physics_quadratic")
    _run("physics_rc")
    _run("ml_gbm", gbm_overrides_for_fit=gbm_overrides)
    _run("hybrid")
    _run("naive_median", skip_save=True)

    return results


def save_artifact_and_register(
    cur, result: dict[str, Any], trafo: str, phase: str, trained_at: datetime,
    healthy_from: datetime, healthy_to: datetime, dataset_metadata: dict[str, Any],
    is_provisional: bool = False, provisional_reason: Optional[str] = None,
) -> Optional[int]:
    """`is_provisional`/`provisional_reason` (go-live, migration 017):
    diisi saat healthy_from/healthy_to berasal dari saran
    ml/suggest_healthy_period.py (peringkat otomatis, BUKAN keputusan
    operator berdasarkan pengetahuan riil trafo) dan belum dikonfirmasi
    resistance test -- lihat --provisional/--provisional-reason di main().
    `provisional_reason` WAJIB diisi kalau is_provisional=True (lihat
    validasi di main(), bukan di sini -- fungsi ini hanya menulis apa
    yang diberi caller)."""
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
            features, metrics, artifact_path, sigma, mu, is_active, is_frozen_reference, notes,
            is_provisional, provisional_reason
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, FALSE, %s, %s, %s)
        RETURNING id
        """,
        (
            trafo, phase, result["model_type"], version, trained_at, healthy_from, healthy_to,
            json.dumps({**dataset_metadata, "fit_info": result["fit_info"]}),
            json.dumps(result["metrics"]), str(Path("ml/artifacts") / filename),
            result["metrics"]["sigma"], result["metrics"]["mu"],
            None, is_provisional, provisional_reason,
        ),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    header = (f"{'model_type':<20} {'status':<8} {'sigma(°C)':>10} {'mu(°C)':>9} {'mae(°C)':>9} "
              f"{'r2':>7} {'n':>6}  {'cv_sigma(mean±std)':>22} {'cv_mae(mean±std)':>20} {'cv_ok/total':>12}")
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "ok":
            print(f"{r['model_type']:<20} {'FAILED':<8} {r['error']}")
            continue
        m = r["metrics"]
        save_note = "" if not r.get("skip_save") else " (pembanding, tidak disimpan)"
        cv = m.get("cv")
        if cv is not None:
            cv_sigma_str = f"{cv['sigma_mean']:.3f}±{cv['sigma_std']:.3f}"
            cv_mae_str = f"{cv['mae_mean']:.3f}±{cv['mae_std']:.3f}"
            cv_n_str = f"{cv['n_folds_ok']}/{cv['n_folds_total']}"
        else:
            cv_sigma_str = cv_mae_str = cv_n_str = "n/a"
        print(f"{r['model_type']:<20} {'ok':<8} {m['sigma']:>10.4f} {m['mu']:>9.4f} "
              f"{m['mae']:>9.4f} {('%.4f' % m['r2']) if m['r2'] is not None else 'n/a':>7} {m['n']:>6}  "
              f"{cv_sigma_str:>22} {cv_mae_str:>20} {cv_n_str:>12}{save_note}")


def _parse_date(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def validate_provisional_args(is_provisional: bool, provisional_reason: Optional[str]) -> Optional[str]:
    """Go-live rule: --provisional dan --provisional-reason WAJIB
    berpasangan (bukan optional independen) -- satu tanpa yang lain
    adalah kesalahan pemakaian, bukan default yang masuk akal. Return
    pesan error (str) kalau tidak valid, None kalau valid."""
    if is_provisional and not provisional_reason:
        return "--provisional requires --provisional-reason (shown verbatim in the UI banner)."
    if provisional_reason and not is_provisional:
        return "--provisional-reason requires --provisional (a reason with no flag set does nothing)."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trafo", required=True, choices=["2", "3", "4"])
    parser.add_argument("--phase", required=True, choices=["R", "S", "T"])
    parser.add_argument("--healthy-from", dest="healthy_from", required=True, help="ISO8601, mis. 2026-07-01")
    parser.add_argument("--healthy-to", dest="healthy_to", required=True, help="ISO8601, mis. 2026-08-01")
    parser.add_argument("--dry-run", action="store_true", help="Latih & cetak tabel, jangan simpan artefak/baris ml_models")
    parser.add_argument(
        "--provisional", action="store_true",
        help="Tandai ml_models.is_provisional=TRUE -- WAJIB dipasangkan dengan --provisional-reason "
             "(go-live: healthy period dari ml/suggest_healthy_period.py, belum dikonfirmasi resistance test).",
    )
    parser.add_argument(
        "--provisional-reason", dest="provisional_reason", default=None,
        help="Wajib diisi kalau --provisional dipakai -- kalimat yang ditampilkan di banner UI.",
    )
    parser.add_argument(
        "--tune", action="store_true", default=False,
        help="M12: grid search hyperparameter ml_gbm (max_depth x learning_rate x min_samples_leaf, "
             "27 kombinasi) lewat CV pada data training SAJA, pilih median sigma CV terkecil, SEBELUM "
             "fit final. Default OFF (pakai ml_config.json apa adanya) -- grid search menambah waktu "
             "latih signifikan (27x lipat fit ml_gbm per fold CV).",
    )
    args = parser.parse_args()

    error = validate_provisional_args(args.provisional, args.provisional_reason)
    if error:
        parser.error(error)

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

        gbm_overrides = None
        if args.tune:
            i_ref = ds.config.LNTAI_V3_I_REF
            log.info("--tune: menjalankan grid search ml_gbm (%d kombinasi) -- bisa makan waktu...", len(GBM_TUNE_GRID["max_depth"]) * len(GBM_TUNE_GRID["learning_rate"]) * len(GBM_TUNE_GRID["min_samples_leaf"]))
            tune_result = tune_gbm_hyperparameters(train_rows, i_ref)
            gbm_overrides = tune_result["best_params"]
            log.info("--tune: hyperparameter terpilih (median sigma CV terkecil): %s", gbm_overrides)

        results = train_all_models(train_rows, val_rows, args.trafo, args.phase, gbm_overrides=gbm_overrides)
        print_comparison_table(results)

        if args.dry_run:
            log.info("--dry-run: TIDAK menyimpan artefak atau menulis ml_models")
            conn.rollback()
            return

        trained_at = datetime.now(timezone.utc)
        saved_ids = []
        for r in results:
            if r["status"] != "ok" or r.get("skip_save"):
                continue
            model_id = save_artifact_and_register(
                cur, r, args.trafo, args.phase, trained_at, ts_from, ts_to, metadata,
                is_provisional=args.provisional, provisional_reason=args.provisional_reason,
            )
            saved_ids.append((r["model_type"], model_id))
        conn.commit()

    log.info("=" * 60)
    for model_type, model_id in saved_ids:
        log.info("Tersimpan: %s -> ml_models.id=%s (is_active=false)", model_type, model_id)
    log.info("Aktifkan model lewat: POST /api/ml/models/{id}/activate")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
