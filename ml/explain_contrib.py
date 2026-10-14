"""ml/explain_contrib.py — SHIELD Stage 2, Part G.

(Nama file sengaja BUKAN "explain.py" -- bentrok dengan server/explain.py
yang menangani CUSUM + perakitan hasil untuk endpoint yang sama; dua modul
berbeda isi, keduanya dipakai bersamaan oleh GET /api/explain/latest,
jadi harus bisa diimpor berdampingan tanpa ambiguitas nama modul.)

Kontribusi per-fitur (°C) atas prediksi ΔT_sehat model baseline AKTIF
(Part B), dikelompokkan ke 4 kategori TETAP (sesuai task brief, bukan
daftar terbuka):

    "Phase load (filtered)"   <- I_eff (arus setelah filter lag termal)
    "Evening peak hour"       <- hour_sin, hour_cos (jam siklik WIB)
    "Load ramp in last hour"  <- load_roc_1h
    "Baseline offset"         <- intercept model (B fisika / bias GBM)
                                 PLUS weekday -- task brief hanya menyebut
                                 4 kategori, weekday (dimensi kalender
                                 lain, bukan "jam") tidak eksplisit masuk
                                 salah satunya; disatukan ke "Baseline
                                 offset" sebagai simplifikasi yang
                                 didokumentasikan, bukan dibuang diam-diam.

Metode BEDA per tipe model baseline:
  - physics_quadratic / physics_rc: method="exact_physics" -- formula
    A·(I_eff/I_ref)²+B SUDAH BERUPA dekomposisi, tidak ada aproksimasi
    sama sekali. i_eff_norm untuk physics_rc HARUS dipasok caller
    (i_eff_norm_override) dari basis tau MILIK MODEL ITU sendiri (lihat
    server/explain.py) -- bukan i2_eff_default_tau di feature_row (itu
    basis tau DEFAULT, beda kalau physics_rc hasil grid search tau lain).
  - ml_gbm (backend LightGBM): method="exact_treeshap" -- pred_contrib=True
    (Tree SHAP eksak, dijamin sum(kontribusi)+bias == prediksi mentah).
  - ml_gbm (backend sklearn HistGradientBoostingRegressor, fallback kita
    saat LightGBM tidak terpasang -- lihat ml/models.py
    _try_import_gbm_backend()): method="approximation_waterfall" --
    TIDAK ADA API kontribusi native di sklearn untuk model ini, jadi
    dipakai dekomposisi "waterfall" bertingkat dengan urutan fitur TETAP
    (FEATURE_NAMES) dari satu titik dasar (semua fitur = 0) ke titik
    aktual, kontribusi fitur ke-i = Δprediksi saat fitur ke-i diubah dari
    basis ke nilai aktual (fitur sebelumnya SUDAH di nilai aktual). Ini
    BUKAN nilai Shapley simetris (bergantung urutan), TAPI dijamin
    sum(kontribusi)+f(basis) == f(aktual) TEPAT (teleskopik), dan
    diberi label method eksplisit -- TIDAK PERNAH disamakan diam-diam
    dengan TreeSHAP.
  - hybrid: physics (eksak) + GBM residual (metode di atas, tergantung
    backend residual_model), dijumlahkan per kategori. method yang
    dilaporkan mengikuti method bagian GBM (bagian yang berpotensi
    aproksimasi), karena bagian physics selalu eksak.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import FEATURE_NAMES  # noqa: E402

CONTRIBUTION_GROUPS: tuple[str, ...] = (
    "Phase load (filtered)", "Evening peak hour", "Load ramp in last hour", "Baseline offset",
)

_FEATURE_TO_GROUP: dict[str, str] = {
    "i2_eff_default_tau": "Phase load (filtered)",
    "load_roc_1h": "Load ramp in last hour",
    "hour_sin": "Evening peak hour",
    "hour_cos": "Evening peak hour",
    "weekday": "Baseline offset",
}

_BASELINE_FEATURE_VALUES: dict[str, float] = {
    "i2_eff_default_tau": 0.0, "load_roc_1h": 0.0, "hour_sin": 0.0, "hour_cos": 0.0, "weekday": 0.0,
}


def _empty_contributions() -> dict[str, float]:
    return {g: 0.0 for g in CONTRIBUTION_GROUPS}


def physics_contributions(a: float, b: float, i_eff_norm: float) -> dict[str, float]:
    """EKSAK -- A·i_eff_norm² + B adalah dekomposisinya sendiri."""
    c = _empty_contributions()
    c["Phase load (filtered)"] = a * (i_eff_norm ** 2)
    c["Baseline offset"] = b
    return c


def _waterfall_contributions(
    regressor: Any, baseline_vec: list[float], actual_vec: list[float], feature_names: tuple[str, ...],
) -> tuple[dict[str, float], float]:
    """Dekomposisi teleskopik: sum(hasil)+f(baseline_vec) == f(actual_vec)
    TEPAT, apa pun urutan fitur-nya (dijamin aljabar, bukan pendekatan
    numerik) -- urutan hanya memengaruhi ATRIBUSI per fitur (bukan
    totalnya), karena itu diberi label "approximation" (bukan Shapley
    simetris)."""
    contributions: dict[str, float] = {}
    current = list(baseline_vec)
    prev_pred = float(regressor.predict([current])[0])
    for i, name in enumerate(feature_names):
        current[i] = actual_vec[i]
        new_pred = float(regressor.predict([current])[0])
        contributions[name] = new_pred - prev_pred
        prev_pred = new_pred
    return contributions, prev_pred


def gbm_contributions(gbm_model: Any, feature_row: dict[str, float]) -> tuple[dict[str, float], str]:
    actual_vec = [feature_row[name] for name in FEATURE_NAMES]
    backend = gbm_model.backend_name or ""

    if "lightgbm" in backend:
        raw = gbm_model._regressor.predict([actual_vec], pred_contrib=True)[0]
        feature_contribs = dict(zip(FEATURE_NAMES, raw[:-1]))
        bias = float(raw[-1])
        method = "exact_treeshap"
    else:
        baseline_vec = [_BASELINE_FEATURE_VALUES[name] for name in FEATURE_NAMES]
        feature_contribs, _ = _waterfall_contributions(gbm_model._regressor, baseline_vec, actual_vec, FEATURE_NAMES)
        bias = float(gbm_model._regressor.predict([baseline_vec])[0])
        method = "approximation_waterfall"

    contributions = _empty_contributions()
    for name, value in feature_contribs.items():
        contributions[_FEATURE_TO_GROUP[name]] += float(value)
    contributions["Baseline offset"] += bias
    return contributions, method


def explain_model(
    model_type: str, model: Any, feature_row: dict[str, float],
    i_eff_norm_override: Optional[float] = None,
) -> tuple[dict[str, float], str]:
    """Titik masuk utama. `feature_row` wajib berisi semua FEATURE_NAMES,
    plus "current_amps" kalau model_type='physics_quadratic'.
    `i_eff_norm_override` WAJIB diisi caller untuk physics_rc/hybrid
    (lihat docstring modul -- basis tau model itu sendiri, bukan tau
    default di feature_row['i2_eff_default_tau'])."""
    if model_type == "physics_quadratic":
        i_eff_norm = (feature_row["current_amps"] / model.i_ref) if model.i_ref else 0.0
        return physics_contributions(model.a, model.b, i_eff_norm), "exact_physics"
    if model_type == "physics_rc":
        i_eff_norm = i_eff_norm_override if i_eff_norm_override is not None else 0.0
        return physics_contributions(model.a, model.b, i_eff_norm), "exact_physics"
    if model_type == "ml_gbm":
        return gbm_contributions(model, feature_row)
    if model_type == "hybrid":
        i_eff_norm = i_eff_norm_override if i_eff_norm_override is not None else 0.0
        physics_c = physics_contributions(model.physics.a, model.physics.b, i_eff_norm)
        gbm_c, gbm_method = gbm_contributions(model.residual_model, feature_row)
        combined = {g: physics_c[g] + gbm_c[g] for g in CONTRIBUTION_GROUPS}
        return combined, gbm_method
    raise ValueError(f"model_type tidak dikenal untuk explain: {model_type!r}")
