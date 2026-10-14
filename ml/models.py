"""ml/models.py — SHIELD Stage 2, Part B.

Empat tipe model baseline "perilaku sehat" (ΔT_pred dari fitur beban),
dilatih per (trafo, phase) oleh ml/train_baseline.py. Semua model punya
interface yang sama: `.fit(rows)`, `.predict(rows) -> list[float]`,
`.to_artifact() -> dict` (JSON-serializable, disimpan ke disk), dan
classmethod `.from_artifact(dict)` untuk memuat ulang (dipakai
ml/model_registry.py saat model diaktifkan).

`rows` adalah list[dict] dengan skema yang sama seperti keluaran
ml/dataset.py: ts_epoch, T, current_amps, i2_eff_default_tau, load_roc_1h,
hour_sin, hour_cos, weekday, ambient_c, quality_flags.

GBM/hybrid BUTUH scikit-learn (HistGradientBoostingRegressor) -- LightGBM
dicoba lebih dulu kalau terpasang (lihat _try_import_gbm_backend()), tapi
Python 3.14 di server ini terlalu baru untuk dipastikan LightGBM punya
wheel yang kompatibel (dicek di laporan audit Stage 2 kickoff: TIDAK ada
akses internet di sandbox pengembangan untuk memverifikasi `pip install
lightgbm` secara langsung). Kalau KEDUANYA tidak terpasang, MLGBMModel/
HybridModel melempar RuntimeError yang jelas saat .fit() dipanggil --
physics_quadratic dan physics_rc TETAP bisa dilatih tanpa dependency itu
sama sekali (murni rumus, tidak butuh library ML).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import ML_CONFIG, compute_i2_eff_series, i2_eff_step  # noqa: E402

FEATURE_NAMES: tuple[str, ...] = ("i2_eff_default_tau", "load_roc_1h", "hour_sin", "hour_cos", "weekday")


def _median_abs_deviation_sigma(residuals: list[float]) -> float:
    """σ robust = 1.4826 * MAD (Median Absolute Deviation) -- tahan
    terhadap outlier dibanding std() biasa, sesuai permintaan brief."""
    if not residuals:
        return 0.0
    sorted_r = sorted(residuals)
    n = len(sorted_r)
    median = sorted_r[n // 2] if n % 2 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2.0
    abs_dev = sorted([abs(r - median) for r in residuals])
    mad = abs_dev[n // 2] if n % 2 else (abs_dev[n // 2 - 1] + abs_dev[n // 2]) / 2.0
    return 1.4826 * mad


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """mu, sigma (robust MAD-based), MAE, R^2, bias -- semua dalam °C,
    dihitung dari residual = y_true - y_pred pada set VALIDASI."""
    n = len(y_true)
    if n == 0:
        return {"mu": None, "sigma": None, "mae": None, "r2": None, "bias": None, "n": 0}
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    mu = sum(residuals) / n
    sigma = _median_abs_deviation_sigma(residuals)
    mae = sum(abs(r) for r in residuals) / n
    y_mean = sum(y_true) / n
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
    return {"mu": mu, "sigma": sigma, "mae": mae, "r2": r2, "bias": mu, "n": n}


def _i_ref_norm(rows: list[dict], i_ref: float) -> list[float]:
    return [row["current_amps"] / i_ref if i_ref else 0.0 for row in rows]


# ════════════════════════════════════════════════════════════════════════
# physics_quadratic — A*(I/I_ref)^2 + B, fit least-squares atas I SESAAT
# (bukan I2_eff -- tidak ada filter lag termal di model ini, sesuai nama
# "quadratic" polos di brief).
# ════════════════════════════════════════════════════════════════════════

class PhysicsQuadraticModel:
    model_type = "physics_quadratic"

    def __init__(self, i_ref: float = 1000.0):
        self.i_ref = i_ref
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    def fit(self, rows: list[dict]) -> None:
        if len(rows) < 2:
            raise ValueError("Butuh >= 2 baris untuk fit least-squares")
        xs = [(row["current_amps"] / self.i_ref) ** 2 for row in rows]
        ys = [row["T"] - row["ambient_c"] for row in rows]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        s_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        s_xx = sum((x - mean_x) ** 2 for x in xs)
        if s_xx == 0:
            raise ValueError("Variansi (I/I_ref)^2 nol -- tidak bisa fit slope")
        self.a = s_xy / s_xx
        self.b = mean_y - self.a * mean_x

    def predict(self, rows: list[dict]) -> list[float]:
        if self.a is None or self.b is None:
            raise RuntimeError("Model belum di-fit")
        return [self.a * (row["current_amps"] / self.i_ref) ** 2 + self.b + row["ambient_c"] for row in rows]

    def predict_step(self, state: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], Optional[float]]:
        """Tidak butuh state (arus SESAAT, bukan I2_eff) -- state dilewatkan
        apa adanya supaya interface seragam dengan model lain (lihat
        ml/model_registry.py predict_delta_t_step())."""
        if self.a is None or self.b is None:
            return state, None
        pred = self.a * (row["current_amps"] / self.i_ref) ** 2 + self.b + row["ambient_c"]
        return state, pred

    def to_artifact(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "i_ref": self.i_ref, "a": self.a, "b": self.b}

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "PhysicsQuadraticModel":
        m = cls(i_ref=artifact["i_ref"])
        m.a, m.b = artifact["a"], artifact["b"]
        return m


# ════════════════════════════════════════════════════════════════════════
# physics_rc — sama seperti quadratic tapi pakai I2_eff (filter lag termal
# orde-1), tau dipilih lewat grid search (ml_config.json
# physics_rc_grid_search.tau_minutes_candidates) berdasar sigma validasi
# TERKECIL di antara kandidat.
# ════════════════════════════════════════════════════════════════════════

class PhysicsRCModel:
    model_type = "physics_rc"

    def __init__(self, i_ref: float = 1000.0, max_gap_minutes: float = 60.0):
        self.i_ref = i_ref
        self.max_gap_minutes = max_gap_minutes
        self.tau_minutes: Optional[float] = None
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    def _i2_eff_for_tau(self, rows: list[dict], tau_minutes: float) -> list[float]:
        ts = [row["ts_epoch"] for row in rows]
        currents = [row["current_amps"] for row in rows]
        return compute_i2_eff_series(ts, currents, tau_minutes, self.max_gap_minutes)

    def fit(self, rows: list[dict], validation_rows: Optional[list[dict]] = None,
            tau_candidates: Optional[list[float]] = None) -> dict[str, Any]:
        """Grid search tau atas (rows=train, validation_rows=validasi).
        Kalau validation_rows tidak diberikan, pakai train sendiri untuk
        memilih tau (kurang ideal, tapi tetap lebih baik daripada tau
        tunggal tanpa pilihan -- hanya dipakai saat caller tidak split)."""
        if len(rows) < 2:
            raise ValueError("Butuh >= 2 baris untuk fit least-squares")
        candidates = tau_candidates or ML_CONFIG["physics_rc_grid_search"]["tau_minutes_candidates"]
        val_rows = validation_rows if validation_rows is not None else rows

        best_tau, best_a, best_b, best_sigma = None, None, None, None
        grid_results = []
        for tau in candidates:
            i2_eff_train = self._i2_eff_for_tau(rows, tau)
            xs = [(i2 ** 0.5) / self.i_ref for i2 in i2_eff_train]
            xs2 = [x ** 2 for x in xs]
            ys = [row["T"] - row["ambient_c"] for row in rows]
            n = len(xs2)
            mean_x, mean_y = sum(xs2) / n, sum(ys) / n
            s_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs2, ys))
            s_xx = sum((x - mean_x) ** 2 for x in xs2)
            if s_xx == 0:
                continue
            a_fit = s_xy / s_xx
            b_fit = mean_y - a_fit * mean_x

            i2_eff_val = self._i2_eff_for_tau(val_rows, tau)
            preds = [a_fit * ((i2 ** 0.5) / self.i_ref) ** 2 + b_fit + row["ambient_c"]
                     for i2, row in zip(i2_eff_val, val_rows)]
            targets = [row["T"] for row in val_rows]
            metrics = compute_metrics(targets, preds)
            grid_results.append({"tau_minutes": tau, "sigma": metrics["sigma"]})
            if best_sigma is None or (metrics["sigma"] is not None and metrics["sigma"] < best_sigma):
                best_tau, best_a, best_b, best_sigma = tau, a_fit, b_fit, metrics["sigma"]

        if best_tau is None:
            raise ValueError("Grid search tau gagal untuk semua kandidat (variansi nol di semua tau)")

        self.tau_minutes, self.a, self.b = best_tau, best_a, best_b
        return {"tau_minutes_candidates": candidates, "tau_minutes_selected": best_tau,
                "grid_results": grid_results}

    def predict(self, rows: list[dict]) -> list[float]:
        if self.a is None or self.b is None or self.tau_minutes is None:
            raise RuntimeError("Model belum di-fit")
        i2_eff = self._i2_eff_for_tau(rows, self.tau_minutes)
        return [self.a * ((i2 ** 0.5) / self.i_ref) ** 2 + self.b + row["ambient_c"]
                for i2, row in zip(i2_eff, rows)]

    def predict_step(self, state: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], Optional[float]]:
        """BUTUH state streaming (i2_eff, last_time) -- tau model ini hasil
        grid search, BISA BEDA dari tau default (config.LNTAI_V2_TAU_MINUTES)
        yang dipakai fitur i2_eff_default_tau di ml/dataset.py, jadi TIDAK
        bisa memakai i_eff yang sudah dihitung v2/LNTAI -- harus difilter
        ulang dengan tau MILIK MODEL INI sendiri, state demi state."""
        if self.a is None or self.b is None or self.tau_minutes is None:
            return state, None
        i2_eff_prev, last_time_prev = state.get("i2_eff"), state.get("last_time")
        i2_eff, last_time = i2_eff_step(
            i2_eff_prev, last_time_prev, row["current_amps"], row["ts_epoch"],
            self.tau_minutes, self.max_gap_minutes,
        )
        new_state = {"i2_eff": i2_eff, "last_time": last_time}
        if i2_eff is None:
            return new_state, None
        pred = self.a * ((i2_eff ** 0.5) / self.i_ref) ** 2 + self.b + row["ambient_c"]
        return new_state, pred

    def to_artifact(self) -> dict[str, Any]:
        return {"model_type": self.model_type, "i_ref": self.i_ref, "max_gap_minutes": self.max_gap_minutes,
                "tau_minutes": self.tau_minutes, "a": self.a, "b": self.b}

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "PhysicsRCModel":
        m = cls(i_ref=artifact["i_ref"], max_gap_minutes=artifact["max_gap_minutes"])
        m.tau_minutes, m.a, m.b = artifact["tau_minutes"], artifact["a"], artifact["b"]
        return m


# ════════════════════════════════════════════════════════════════════════
# Backend GBM — LightGBM dicoba lebih dulu, fallback ke scikit-learn.
# ════════════════════════════════════════════════════════════════════════

def _try_import_gbm_backend() -> tuple[str, Any]:
    """Return (nama_library, kelas_regressor). Melempar RuntimeError kalau
    KEDUANYA tidak terpasang -- TIDAK PERNAH diam-diam memilih model lain
    tanpa sepengetahuan caller."""
    try:
        import lightgbm  # type: ignore
        return "lightgbm", lightgbm.LGBMRegressor
    except ImportError:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
        return "sklearn.HistGradientBoostingRegressor", HistGradientBoostingRegressor
    except ImportError:
        pass
    raise RuntimeError(
        "Tidak ada backend GBM terpasang (lightgbm maupun scikit-learn). "
        "Jalankan `pip install scikit-learn` (atau lightgbm kalau tersedia untuk "
        "Python 3.14 di server ini) sebelum melatih model_type=ml_gbm/hybrid. "
        "physics_quadratic dan physics_rc tidak terpengaruh -- keduanya tidak "
        "butuh library ML sama sekali."
    )


def _rows_to_feature_matrix(rows: list[dict]) -> list[list[float]]:
    return [[row[name] for name in FEATURE_NAMES] for row in rows]


# ════════════════════════════════════════════════════════════════════════
# ml_gbm — gradient boosting langsung atas fitur penuh.
# ════════════════════════════════════════════════════════════════════════

class MLGBMModel:
    model_type = "ml_gbm"

    def __init__(self):
        self.backend_name: Optional[str] = None
        self._regressor: Any = None

    def fit(self, rows: list[dict]) -> dict[str, Any]:
        backend_name, regressor_cls = _try_import_gbm_backend()
        self.backend_name = backend_name
        gbm_cfg = ML_CONFIG["gbm"]
        if "lightgbm" in backend_name:
            self._regressor = regressor_cls(
                n_estimators=gbm_cfg["max_iter"], max_depth=gbm_cfg["max_depth"],
                learning_rate=gbm_cfg["learning_rate"],
            )
        else:
            self._regressor = regressor_cls(
                max_iter=gbm_cfg["max_iter"], max_depth=gbm_cfg["max_depth"],
                learning_rate=gbm_cfg["learning_rate"],
            )
        X = _rows_to_feature_matrix(rows)
        y = [row["T"] for row in rows]
        self._regressor.fit(X, y)
        return {"backend": backend_name}

    def predict(self, rows: list[dict]) -> list[float]:
        if self._regressor is None:
            raise RuntimeError("Model belum di-fit")
        return list(self._regressor.predict(_rows_to_feature_matrix(rows)))

    def predict_step(self, state: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], Optional[float]]:
        """Tidak butuh state sendiri -- `row` HARUS sudah berisi
        i2_eff_default_tau (dihitung ulang dari i_eff v2/LNTAI yang sudah
        ada di worker, tau SAMA dengan saat training, lihat
        ml/model_registry.py) + load_roc_1h/hour_sin/hour_cos/weekday."""
        if self._regressor is None:
            return state, None
        pred = self.predict([row])[0]
        return state, pred

    def to_artifact(self) -> dict[str, Any]:
        import pickle
        import base64
        return {
            "model_type": self.model_type, "backend": self.backend_name,
            "regressor_pickle_b64": base64.b64encode(pickle.dumps(self._regressor)).decode("ascii"),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "MLGBMModel":
        import pickle
        import base64
        m = cls()
        m.backend_name = artifact["backend"]
        m._regressor = pickle.loads(base64.b64decode(artifact["regressor_pickle_b64"]))
        return m


# ════════════════════════════════════════════════════════════════════════
# hybrid — physics_rc + GBM atas RESIDUAL physics_rc (bukan atas target
# langsung) -- menangkap efek non-fisik (hour/weekday/dll.) di atas basis
# fisis yang sudah masuk akal.
# ════════════════════════════════════════════════════════════════════════

class HybridModel:
    model_type = "hybrid"

    def __init__(self, i_ref: float = 1000.0, max_gap_minutes: float = 60.0):
        self.physics = PhysicsRCModel(i_ref=i_ref, max_gap_minutes=max_gap_minutes)
        self.residual_model = MLGBMModel()

    def fit(self, rows: list[dict], validation_rows: Optional[list[dict]] = None,
            tau_candidates: Optional[list[float]] = None) -> dict[str, Any]:
        physics_info = self.physics.fit(rows, validation_rows=validation_rows, tau_candidates=tau_candidates)
        physics_preds_train = self.physics.predict(rows)
        residuals_train = [row["T"] - pred for row, pred in zip(rows, physics_preds_train)]
        rows_with_residual_target = [{**row, "T": resid} for row, resid in zip(rows, residuals_train)]
        gbm_info = self.residual_model.fit(rows_with_residual_target)
        return {"physics": physics_info, "residual_gbm": gbm_info}

    def predict(self, rows: list[dict]) -> list[float]:
        physics_preds = self.physics.predict(rows)
        residual_preds = self.residual_model.predict(rows)
        return [p + r for p, r in zip(physics_preds, residual_preds)]

    def predict_step(self, state: dict[str, Any], row: dict[str, Any]) -> tuple[dict[str, Any], Optional[float]]:
        """State streaming HANYA untuk komponen physics_rc (tau milik model
        ini sendiri); residual_model (GBM) stateless, memakai
        row["i2_eff_default_tau"] yang sama seperti saat training (BUKAN
        i2_eff dari physics_rc di atas -- lihat HybridModel.fit(), residual
        GBM dilatih di atas fitur default-tau, bukan tau physics_rc)."""
        physics_state = state.get("physics", {})
        new_physics_state, physics_pred = self.physics.predict_step(physics_state, row)
        new_state = {"physics": new_physics_state}
        if physics_pred is None:
            return new_state, None
        residual_pred = self.residual_model.predict([row])[0]
        return new_state, physics_pred + residual_pred

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "physics": self.physics.to_artifact(),
            "residual_gbm": self.residual_model.to_artifact(),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "HybridModel":
        m = cls()
        m.physics = PhysicsRCModel.from_artifact(artifact["physics"])
        m.residual_model = MLGBMModel.from_artifact(artifact["residual_gbm"])
        return m


MODEL_CLASSES: dict[str, Any] = {
    "physics_quadratic": PhysicsQuadraticModel,
    "physics_rc": PhysicsRCModel,
    "ml_gbm": MLGBMModel,
    "hybrid": HybridModel,
}


def load_model_from_artifact(model_type: str, artifact: dict[str, Any]):
    cls = MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"model_type tidak dikenal: {model_type!r}")
    return cls.from_artifact(artifact)
