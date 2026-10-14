"""ml/forecast_models.py — SHIELD Stage 2, Part F.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import _try_import_gbm_backend  # noqa: E402 -- reuse Part B's lightgbm/sklearn fallback

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

class SeasonalNaiveForecastModel:
    model_type = "seasonal_naive"

    def __init__(self) -> None:
        self.p10_offset = 0.0
        self.p50_offset = 0.0
        self.p90_offset = 0.0

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
        return point + self.p10_offset, point + self.p50_offset, point + self.p90_offset

    def calibrate(self, backtest_pairs: list[tuple[float, float]], lookup: dict[float, float]) -> dict[str, Any]:
        """`backtest_pairs`: [(target_ts_epoch, actual_current_amps), ...]
        dari rentang holdout (lebih baru dari data training)."""
        residuals = []
        preds = []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch, lookup)
            if pred is not None:
                residuals.append(actual - pred)
                preds.append(pred)
        self.p10_offset, self.p50_offset, self.p90_offset = _sorted_quantile_offsets(residuals)
        actuals = [a for _, a in backtest_pairs if self.point_forecast(_, lookup) is not None]
        return {
            "n_backtest": len(residuals),
            "mape": mape(actuals, preds),
        }

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "p10_offset": self.p10_offset, "p50_offset": self.p50_offset, "p90_offset": self.p90_offset,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "SeasonalNaiveForecastModel":
        m = cls()
        m.p10_offset = artifact["p10_offset"]
        m.p50_offset = artifact["p50_offset"]
        m.p90_offset = artifact["p90_offset"]
        return m


# ════════════════════════════════════════════════════════════════════════
# gbm_calendar
# ════════════════════════════════════════════════════════════════════════

class GBMCalendarForecastModel:
    model_type = "gbm_calendar"

    def __init__(self) -> None:
        self.backend_name: Optional[str] = None
        self._regressor: Any = None
        self.p10_offset = 0.0
        self.p50_offset = 0.0
        self.p90_offset = 0.0

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
        return point + self.p10_offset, point + self.p50_offset, point + self.p90_offset

    def calibrate(self, backtest_pairs: list[tuple[float, float]]) -> dict[str, Any]:
        residuals, preds, actuals = [], [], []
        for target_ts_epoch, actual in backtest_pairs:
            pred = self.point_forecast(target_ts_epoch)
            if pred is not None:
                residuals.append(actual - pred)
                preds.append(pred)
                actuals.append(actual)
        self.p10_offset, self.p50_offset, self.p90_offset = _sorted_quantile_offsets(residuals)
        return {"n_backtest": len(residuals), "mape": mape(actuals, preds)}

    def to_artifact(self) -> dict[str, Any]:
        import base64
        import pickle

        return {
            "model_type": self.model_type, "backend": self.backend_name,
            "p10_offset": self.p10_offset, "p50_offset": self.p50_offset, "p90_offset": self.p90_offset,
            "regressor_pickle_b64": base64.b64encode(pickle.dumps(self._regressor)).decode("ascii"),
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
        return m


FORECAST_MODEL_CLASSES: dict[str, Any] = {
    "seasonal_naive": SeasonalNaiveForecastModel,
    "gbm_calendar": GBMCalendarForecastModel,
}


def load_forecast_model_from_artifact(model_type: str, artifact: dict[str, Any]):
    cls = FORECAST_MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(f"model_type forecast tidak dikenal: {model_type!r}")
    return cls.from_artifact(artifact)
