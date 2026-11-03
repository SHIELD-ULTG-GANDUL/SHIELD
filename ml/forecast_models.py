"""ml/forecast_models.py — SHIELD Stage 2, Part F (+ go-live rework of the
"Forecast Realtime" tab, 2026-10-15 — see docs/STAGE2_FINAL_REPORT.md).

Model forecasting BEBAN (arus), per (trafo, phase):

  - seasonal_naive: titik forecast = nilai jam yang sama TEPAT 7 hari
    sebelumnya. Butuh data historis terbaru pada saat forecast dibuat
    (lihat `lookup` di predict_quantiles()) -- TIDAK menyimpan seri
    historis di dalam artefak (akan basi begitu disimpan), hanya offset
    kalibrasi P10/P50/P90 (lihat calibrate()).
  - gbm_calendar: gradient boosting atas fitur KALENDER MURNI (jam
    siklik, hari-dalam-minggu WIB) -- BUKAN lag beban masa lalu, sesuai
    task brief ("gradient boosting on calendar features") -- supaya satu
    model terlatih bisa forecast ke 24 jam maupun 7 hari ke depan tanpa
    butuh observasi beban terbaru sebagai input sama sekali (beda dari
    seasonal_naive).
  - poly_calendar: regresi polinomial (least squares, numpy) atas
    harmonik jam-siklik (hour_sin, hour_cos, pangkat sampai `degree`)
    + hari-dalam-minggu (one-hot) -- metode klasik untuk kurva beban
    harian/mingguan periodik (dirujuk luas di literatur short-term load
    forecasting/STLF, mis. regresi harmonik/polinomial sebagai
    baseline pembanding GBM). Ditambahkan saat merombak tab "Forecast
    Realtime" (sebelumnya Holt-Winters di BROWSER tanpa validasi nyata,
    kadang jatuh ke satu sampel sesaat -- lihat commit message) supaya
    ada metode kalender murni KEDUA yang lebih sederhana/interpretable
    dibanding GBM untuk dibandingkan lewat ml/train_load_forecast.py.

Kuantil P10/P50/P90 KEDUANYA dari OFFSET RESIDUAL EMPIRIS hasil backtest
(bukan tiga model quantile-loss terpisah) -- offset diurutkan naik saat
kalibrasi supaya P10<=P50<=P90 TERJAMIN secara konstruksi, apa pun bentuk
distribusi residualnya.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import _try_import_gbm_backend  # noqa: E402 -- reuse Part B's lightgbm/sklearn fallback


def _require_numpy() -> None:
    """Pola yang sama dengan server/degradation.py Part H -- hanya
    PolynomialCalendarForecastModel.fit() yang butuh numpy (least
    squares); seasonal_naive dan gbm_calendar sama sekali tidak
    terpengaruh kalau numpy belum terpasang."""
    if not _HAS_NUMPY:
        raise RuntimeError(
            "numpy tidak terpasang -- jalankan `pip install numpy` sebelum melatih "
            "model_type=poly_calendar. seasonal_naive dan gbm_calendar tidak terpengaruh."
        )

_WIB_OFFSET = timedelta(hours=7)
SEVEN_DAYS_SECONDS = 7 * 86400.0

FORECAST_FEATURE_NAMES: tuple[str, ...] = ("hour_sin", "hour_cos", "weekday")


def hour_sin_cos_weekday(ts_epoch: float) -> tuple[float, float, int]:
    """Sama seperti ml/dataset.wib_hour_sin_cos_weekday() -- diduplikasi
    di sini (bukan diimpor) supaya ml/forecast_models.py tidak perlu
    bergantung pada ml/dataset.py (yang punya dependency server/ sendiri,
    lihat catatan sirkularitas di server/data_quality.py)."""
    dt_utc = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
    dt_wib = dt_utc + _WIB_OFFSET
    hour_frac = dt_wib.hour + dt_wib.minute / 60.0
    angle = 2 * pi * hour_frac / 24.0
    return sin(angle), cos(angle), dt_wib.weekday()


def _sorted_quantile_offsets(residuals: list[float]) -> tuple[float, float, float]:
    """Offset P10/P50/P90 dari kuantil empiris residual (actual - pred),
    DIURUTKAN NAIK -- menjamin p10<=p50<=p90 secara konstruksi."""
    if not residuals:
        return 0.0, 0.0, 0.0
    s = sorted(residuals)
    n = len(s)

    def _pct(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac

    offsets = sorted([_pct(0.10), _pct(0.50), _pct(0.90)])
    return offsets[0], offsets[1], offsets[2]


# M11 (autopilot 2026-11): offset kuantil TERPISAH per bucket jam WIB,
# bukan satu offset global -- beban (dan sebarannya) biasanya beda
# signifikan siang vs malam, jadi band P10/P90 yang sama untuk kedua
# kondisi itu cenderung terlalu lebar di jam sepi dan/atau terlalu
# sempit di jam padat. Default bucket & ambang sampel minimum ada di
# ml_config.json (`forecast_quantile_buckets`).

_MIN_BUCKET_RESIDUALS = 20  # TODO(verifikasi): ambang minimum sampel per bucket sebelum jatuh ke offset global


def _wib_hour_of(ts_epoch: float) -> float:
    dt_wib = datetime.fromtimestamp(ts_epoch, tz=timezone.utc) + _WIB_OFFSET
    return dt_wib.hour + dt_wib.minute / 60.0


def _bucket_index(hour: float, buckets: list[tuple[int, int]]) -> Optional[int]:
    """`buckets`: [(lo, hi), ...] jam WIB, hi EKSKLUSIF. Mendukung bucket
    yang membungkus tengah malam (lo > hi, mis. (20, 4)). None kalau
    `hour` tidak masuk bucket mana pun (konfigurasi bucket tidak lengkap)."""
    for i, (lo, hi) in enumerate(buckets):
        if lo <= hi:
            if lo <= hour < hi:
                return i
        elif hour >= lo or hour < hi:
            return i
    return None


def _quantile_offsets_by_bucket(
    residual_hour_pairs: list[tuple[float, float]],
    buckets: list[tuple[int, int]],
    global_offsets: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """`residual_hour_pairs`: [(residual, wib_hour), ...] dari backtest.
    Return satu tuple offset (p10,p50,p90) per bucket (urutan sama dengan
    `buckets`) -- bucket dengan residual < _MIN_BUCKET_RESIDUALS jatuh ke
    `global_offsets` (sampel terlalu sedikit untuk kuantil per-jam yang
    stabil secara statistik)."""
    grouped: dict[int, list[float]] = {i: [] for i in range(len(buckets))}
    for residual, hour in residual_hour_pairs:
        idx = _bucket_index(hour, buckets)
        if idx is not None:
            grouped[idx].append(residual)
    return [
        _sorted_quantile_offsets(grouped[i]) if len(grouped[i]) >= _MIN_BUCKET_RESIDUALS else global_offsets
        for i in range(len(buckets))
    ]


def _quantiles_for_bucket(
    target_ts_epoch: float, point: float, buckets: list[tuple[int, int]],
    bucket_offsets: list[tuple[float, float, float]], global_offsets: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Pilih offset (p10,p50,p90) bucket jam WIB dari `target_ts_epoch`,
    jatuh ke `global_offsets` kalau buckets/bucket_offsets kosong (artefak
    lama sebelum M11) atau jamnya tidak masuk bucket mana pun."""
    if buckets and bucket_offsets:
        idx = _bucket_index(_wib_hour_of(target_ts_epoch), buckets)
        if idx is not None and idx < len(bucket_offsets):
            offs = bucket_offsets[idx]
            return point + offs[0], point + offs[1], point + offs[2]
    return point + global_offsets[0], point + global_offsets[1], point + global_offsets[2]


class _QuantileBucketMixin:
    """M11: logika offset-per-bucket-jam BERSAMA ketiga model_type -- tiap
    class tetap punya p10/50/90_offset milik sendiri sebagai offset GLOBAL
    (fallback & kompatibel dengan artefak lama sebelum M11), ditambah
    `buckets`/`bucket_offsets` opsional. Disimpan di DALAM artefak (bukan
    dibaca ulang dari ml_config.json saat load) supaya model lama tetap
    bisa dimuat dan diprediksi persis sama walau config berubah di masa
    depan."""

    buckets: list[tuple[int, int]]
    bucket_offsets: list[tuple[float, float, float]]
    p10_offset: float
    p50_offset: float
    p90_offset: float

    def _init_bucket_fields(self) -> None:
        self.buckets = []
        self.bucket_offsets = []

    def _calibrate_bucket_offsets(self, residual_hour_pairs: list[tuple[float, float]]) -> None:
        from models import ML_CONFIG  # lazy import, sama pola dengan GBMCalendarForecastModel.fit()

        self.buckets = [tuple(b) for b in ML_CONFIG.get("forecast_quantile_buckets", [])]
        global_offsets = (self.p10_offset, self.p50_offset, self.p90_offset)
        self.bucket_offsets = (
            _quantile_offsets_by_bucket(residual_hour_pairs, self.buckets, global_offsets)
            if self.buckets else []
        )

    def _quantiles(self, target_ts_epoch: float, point: float) -> tuple[float, float, float]:
        return _quantiles_for_bucket(
            target_ts_epoch, point, self.buckets, self.bucket_offsets,
            (self.p10_offset, self.p50_offset, self.p90_offset),
        )

    def _bucket_artifact_fields(self) -> dict[str, Any]:
        return {
            "buckets": [list(b) for b in self.buckets],
            "bucket_offsets": [list(o) for o in self.bucket_offsets],
        }

    def _load_bucket_artifact_fields(self, artifact: dict[str, Any]) -> None:
        # get(..., []) -- artefak dari SEBELUM M11 tidak punya key ini sama
        # sekali; predict_quantiles() lewat _quantiles() otomatis jatuh ke
        # offset global p10/50/90_offset kalau buckets kosong.
        self.buckets = [tuple(b) for b in artifact.get("buckets", [])]
        self.bucket_offsets = [tuple(o) for o in artifact.get("bucket_offsets", [])]


def coverage_p10_p90(
    actuals: list[float], p10s: list[Optional[float]], p90s: list[Optional[float]],
) -> Optional[float]:
    """M11: proporsi actual yang jatuh di DALAM [p10, p90] -- kalibrasi
    yang baik idealnya mendekati 0.80 (interval 10%-90%). Dipakai sebagai
    metrik ringkas di tabel perbandingan ml/train_load_forecast.py."""
    pairs = [(a, lo, hi) for a, lo, hi in zip(actuals, p10s, p90s) if lo is not None and hi is not None]
    if not pairs:
        return None
    hits = sum(1 for a, lo, hi in pairs if lo <= a <= hi)
    return hits / len(pairs)


def mape(actuals: list[float], preds: list[float]) -> Optional[float]:
    """Mean Absolute Percentage Error (%) -- sampel dengan actual=0
    DIBUANG (pembagian oleh nol tidak terdefinisi), bukan dipaksakan."""
    pairs = [(a, p) for a, p in zip(actuals, preds) if a != 0]
    if not pairs:
        return None
    return sum(abs((a - p) / a) for a, p in pairs) / len(pairs) * 100.0


def pinball_loss(actual: float, pred: float, quantile: float) -> float:
    diff = actual - pred
    return max(quantile * diff, (quantile - 1.0) * diff)


def mean_pinball_loss(actuals: list[float], p10: list[float], p50: list[float], p90: list[float]) -> Optional[float]:
    """Rata-rata pinball loss atas ketiga kuantil (0.10/0.50/0.90),
    dipakai sebagai satu angka ringkasan kalibrasi kuantil (lihat task
    brief: 'Metrics (MAPE, pinball loss)')."""
    losses = []
    for a, q10, q50, q90 in zip(actuals, p10, p50, p90):
        if q10 is None or q50 is None or q90 is None:
            continue
        losses.append(pinball_loss(a, q10, 0.10))
        losses.append(pinball_loss(a, q50, 0.50))
        losses.append(pinball_loss(a, q90, 0.90))
    if not losses:
        return None
    return sum(losses) / len(losses)


# ════════════════════════════════════════════════════════════════════════
# seasonal_naive
# ════════════════════════════════════════════════════════════════════════

class SeasonalNaiveForecastModel(_QuantileBucketMixin):
    model_type = "seasonal_naive"

    def __init__(self) -> None:
        self.p10_offset = 0.0
        self.p50_offset = 0.0
        self.p90_offset = 0.0
        self._init_bucket_fields()

    @staticmethod
    def point_forecast(target_ts_epoch: float, lookup: dict[float, float]) -> Optional[float]:
        """`lookup`: {hour_epoch: mean_current_amps, ...} -- harus mencakup
        (target_ts_epoch - 7 hari); caller (server/forecast_job.py)
        bertanggung jawab memasok data historis yang cukup segar."""
        return lookup.get(target_ts_epoch - SEVEN_DAYS_SECONDS)

    def predict_quantiles(
        self, target_ts_epoch: float, lookup: dict[float, float],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        point = self.point_forecast(target_ts_epoch, lookup)
        if point is None:
            return None, None, None
        return self._quantiles(target_ts_epoch, point)

    def calibrate(self, backtest_pairs: list[tuple[float, float]], lookup: dict[float, float]) -> dict[str, Any]:
        """`backtest_pairs`: [(target_ts_epoch, actual_current_amps), ...]
        dari rentang holdout (lebih baru dari data training)."""
        residuals = []
        preds = []
        residual_hour_pairs = []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch, lookup)
            if pred is not None:
                residuals.append(actual - pred)
                preds.append(pred)
                residual_hour_pairs.append((actual - pred, _wib_hour_of(target_ts_epoch)))
        self.p10_offset, self.p50_offset, self.p90_offset = _sorted_quantile_offsets(residuals)
        self._calibrate_bucket_offsets(residual_hour_pairs)
        actuals = [a for _, a in backtest_pairs if self.point_forecast(_, lookup) is not None]
        p10s, p50s, p90s = [], [], []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch, lookup)
            if pred is not None:
                q10, q50, q90 = self._quantiles(target_ts_epoch, pred)
                p10s.append(q10); p50s.append(q50); p90s.append(q90)
        return {
            "n_backtest": len(residuals),
            "mape": mape(actuals, preds),
            "coverage_p10_p90": coverage_p10_p90(actuals, p10s, p90s),
        }

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "p10_offset": self.p10_offset, "p50_offset": self.p50_offset, "p90_offset": self.p90_offset,
            **self._bucket_artifact_fields(),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "SeasonalNaiveForecastModel":
        m = cls()
        m.p10_offset = artifact["p10_offset"]
        m.p50_offset = artifact["p50_offset"]
        m.p90_offset = artifact["p90_offset"]
        m._load_bucket_artifact_fields(artifact)
        return m


# ════════════════════════════════════════════════════════════════════════
# gbm_calendar
# ════════════════════════════════════════════════════════════════════════

class GBMCalendarForecastModel(_QuantileBucketMixin):
    model_type = "gbm_calendar"

    def __init__(self) -> None:
        self.backend_name: Optional[str] = None
        self._regressor: Any = None
        self.p10_offset = 0.0
        self.p50_offset = 0.0
        self.p90_offset = 0.0
        self._init_bucket_fields()

    @staticmethod
    def _features(ts_epoch: float) -> list[float]:
        hour_sin, hour_cos, weekday = hour_sin_cos_weekday(ts_epoch)
        return [hour_sin, hour_cos, float(weekday)]

    def fit(self, hourly_series: list[tuple[float, float]]) -> dict[str, Any]:
        """`hourly_series`: [(ts_epoch, mean_current_amps), ...] (data
        TRAINING -- rentang lebih awal dari backtest holdout)."""
        from models import ML_CONFIG  # lazy import (hindari beban impor saat hanya predict)

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
        X = [self._features(ts) for ts, _ in hourly_series]
        y = [current for _, current in hourly_series]
        self._regressor.fit(X, y)
        return {"backend": backend_name, "n_train": len(hourly_series)}

    def point_forecast(self, target_ts_epoch: float) -> Optional[float]:
        if self._regressor is None:
            return None
        return float(self._regressor.predict([self._features(target_ts_epoch)])[0])

    def predict_quantiles(self, target_ts_epoch: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
        point = self.point_forecast(target_ts_epoch)
        if point is None:
            return None, None, None
        return self._quantiles(target_ts_epoch, point)

    def calibrate(self, backtest_pairs: list[tuple[float, float]]) -> dict[str, Any]:
        residuals, preds, actuals = [], [], []
        residual_hour_pairs = []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch)
            if pred is not None:
                residuals.append(actual - pred)
                preds.append(pred)
                actuals.append(actual)
                residual_hour_pairs.append((actual - pred, _wib_hour_of(target_ts_epoch)))
        self.p10_offset, self.p50_offset, self.p90_offset = _sorted_quantile_offsets(residuals)
        self._calibrate_bucket_offsets(residual_hour_pairs)
        p10s, p50s, p90s = [], [], []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch)
            if pred is not None:
                q10, q50, q90 = self._quantiles(target_ts_epoch, pred)
                p10s.append(q10); p50s.append(q50); p90s.append(q90)
        return {
            "n_backtest": len(residuals), "mape": mape(actuals, preds),
            "coverage_p10_p90": coverage_p10_p90(actuals, p10s, p90s),
        }

    def to_artifact(self) -> dict[str, Any]:
        import base64
        import pickle

        return {
            "model_type": self.model_type, "backend": self.backend_name,
            "p10_offset": self.p10_offset, "p50_offset": self.p50_offset, "p90_offset": self.p90_offset,
            "regressor_pickle_b64": base64.b64encode(pickle.dumps(self._regressor)).decode("ascii"),
            **self._bucket_artifact_fields(),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "GBMCalendarForecastModel":
        import base64
        import pickle

        m = cls()
        m.backend_name = artifact["backend"]
        m.p10_offset = artifact["p10_offset"]
        m.p50_offset = artifact["p50_offset"]
        m.p90_offset = artifact["p90_offset"]
        m._regressor = pickle.loads(base64.b64decode(artifact["regressor_pickle_b64"]))
        m._load_bucket_artifact_fields(artifact)
        return m


# ════════════════════════════════════════════════════════════════════════
# poly_calendar
# ════════════════════════════════════════════════════════════════════════

class PolynomialCalendarForecastModel(_QuantileBucketMixin):
    """Regresi polinomial (least squares) atas fitur kalender -- metode
    KEDUA (selain gbm_calendar) untuk tab "Forecast Realtime" yang
    dirombak 2026-10-15, supaya ada pembanding yang lebih sederhana/
    interpretable terhadap GBM, bukan cuma satu metode. Fitur: harmonik
    jam-siklik (hour_sin, hour_cos, dan pangkatnya sampai `degree`) +
    hari-dalam-minggu one-hot (6 kolom dummy, Senin=referensi) -- SAMA
    PERSIS filosofi "kalender murni, bukan lag beban" dengan
    gbm_calendar, cuma regressor-nya beda (linear least squares atas
    fitur polinomial, bukan pohon boosting)."""

    model_type = "poly_calendar"

    def __init__(self, degree: int = 3) -> None:
        self.degree = degree
        self.coefficients: Optional[list[float]] = None
        self.p10_offset = 0.0
        self.p50_offset = 0.0
        self.p90_offset = 0.0
        self._init_bucket_fields()

    def _features(self, ts_epoch: float) -> list[float]:
        hour_sin, hour_cos, weekday = hour_sin_cos_weekday(ts_epoch)
        feats = [1.0]
        for d in range(1, self.degree + 1):
            feats.append(hour_sin ** d)
            feats.append(hour_cos ** d)
        # One-hot hari-dalam-minggu, Senin (0) jadi referensi (dibuang,
        # tersirat lewat intercept) -- standar dummy-variable encoding.
        for wd in range(1, 7):
            feats.append(1.0 if weekday == wd else 0.0)
        return feats

    def fit(self, hourly_series: list[tuple[float, float]]) -> dict[str, Any]:
        """`hourly_series`: [(ts_epoch, mean_current_amps), ...] (data
        TRAINING -- rentang lebih awal dari backtest holdout)."""
        _require_numpy()
        X = np.array([self._features(ts) for ts, _ in hourly_series], dtype=float)
        y = np.array([current for _, current in hourly_series], dtype=float)
        coeffs, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        self.coefficients = coeffs.tolist()
        return {"degree": self.degree, "n_train": len(hourly_series), "matrix_rank": int(rank)}

    def point_forecast(self, target_ts_epoch: float) -> Optional[float]:
        if self.coefficients is None:
            return None
        feats = self._features(target_ts_epoch)
        return float(sum(c * f for c, f in zip(self.coefficients, feats)))

    def predict_quantiles(self, target_ts_epoch: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
        point = self.point_forecast(target_ts_epoch)
        if point is None:
            return None, None, None
        return self._quantiles(target_ts_epoch, point)

    def calibrate(self, backtest_pairs: list[tuple[float, float]]) -> dict[str, Any]:
        residuals, preds, actuals = [], [], []
        residual_hour_pairs = []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch)
            if pred is not None:
                residuals.append(actual - pred)
                preds.append(pred)
                actuals.append(actual)
                residual_hour_pairs.append((actual - pred, _wib_hour_of(target_ts_epoch)))
        self.p10_offset, self.p50_offset, self.p90_offset = _sorted_quantile_offsets(residuals)
        self._calibrate_bucket_offsets(residual_hour_pairs)
        p10s, p50s, p90s = [], [], []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch)
            if pred is not None:
                q10, q50, q90 = self._quantiles(target_ts_epoch, pred)
                p10s.append(q10); p50s.append(q50); p90s.append(q90)
        return {
            "n_backtest": len(residuals), "mape": mape(actuals, preds),
            "coverage_p10_p90": coverage_p10_p90(actuals, p10s, p90s),
        }

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type, "degree": self.degree, "coefficients": self.coefficients,
            "p10_offset": self.p10_offset, "p50_offset": self.p50_offset, "p90_offset": self.p90_offset,
            **self._bucket_artifact_fields(),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "PolynomialCalendarForecastModel":
        m = cls(degree=artifact["degree"])
        m.coefficients = artifact["coefficients"]
        m.p10_offset = artifact["p10_offset"]
        m.p50_offset = artifact["p50_offset"]
        m.p90_offset = artifact["p90_offset"]
        m._load_bucket_artifact_fields(artifact)
        return m


FORECAST_MODEL_CLASSES: dict[str, Any] = {
    "seasonal_naive": SeasonalNaiveForecastModel,
    "gbm_calendar": GBMCalendarForecastModel,
    "poly_calendar": PolynomialCalendarForecastModel,
}


def load_forecast_model_from_artifact(model_type: str, artifact: dict[str, Any]):
    cls = FORECAST_MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"model_type forecast tidak dikenal: {model_type!r}")
    return cls.from_artifact(artifact)
