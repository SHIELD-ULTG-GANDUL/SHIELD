"""Test SHIELD Stage 2, Part F + go-live rework — ml/forecast_models.py
PolynomialCalendarForecastModel (new model_type added when replacing the
browser-side Holt-Winters "Forecast Realtime" tab with the real,
validated Part F forecast system, 2026-10-15). Tanpa DB/network.

Run: python3 test_ml_forecast_models.py
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "ml"))
sys.path.insert(0, str(_ROOT / "server"))

import forecast_models as fm  # noqa: E402


def _synthetic_hourly_series(n_hours: int, noise_sd: float = 2.0, seed: int = 1) -> list[tuple[float, float]]:
    """Pola kalender yang DIKETAHUI: beban naik siang (hour_sin dominan)
    dan sedikit lebih tinggi di hari kerja -- supaya bisa dicek apakah
    model polinomial benar-benar menangkap pola ini, bukan cuma
    menebak rata-rata."""
    random.seed(seed)
    series = []
    for i in range(n_hours):
        ts = i * 3600.0
        hour_sin, hour_cos, weekday = fm.hour_sin_cos_weekday(ts)
        weekday_bump = 20.0 if weekday < 5 else 0.0  # hari kerja lebih tinggi
        value = 800.0 + 150.0 * hour_sin + weekday_bump + random.gauss(0, noise_sd)
        series.append((ts, value))
    return series


class TestPolynomialCalendarForecastModelFit(unittest.TestCase):
    def test_fit_recovers_known_calendar_pattern(self):
        """Residual setelah fit HARUS kecil (model menangkap pola, bukan
        cuma rata-rata) -- ini bedanya dengan 'menebak-nebak'."""
        series = _synthetic_hourly_series(24 * 30, noise_sd=1.0)
        m = fm.PolynomialCalendarForecastModel(degree=3)
        m.fit(series)
        preds = [m.point_forecast(ts) for ts, _ in series]
        errors = [abs(v - p) for (_, v), p in zip(series, preds)]
        mean_abs_error = sum(errors) / len(errors)
        self.assertLess(mean_abs_error, 5.0, "model polinomial harus menangkap pola kalender dengan error kecil")

    def test_point_forecast_before_fit_is_none(self):
        m = fm.PolynomialCalendarForecastModel()
        self.assertIsNone(m.point_forecast(0.0))

    def test_too_few_samples_still_fits_without_crash(self):
        # Hanya 20 sampel -- matrix rank-deficient mungkin terjadi, tapi
        # TIDAK boleh melempar exception tak tertangani.
        series = _synthetic_hourly_series(20, noise_sd=1.0)
        m = fm.PolynomialCalendarForecastModel(degree=3)
        info = m.fit(series)
        self.assertEqual(info["n_train"], 20)


class TestPolynomialCalendarForecastModelCalibrate(unittest.TestCase):
    def test_quantiles_are_monotonic(self):
        series = _synthetic_hourly_series(24 * 20, noise_sd=3.0)
        train, backtest = series[:-24 * 7], series[-24 * 7:]
        m = fm.PolynomialCalendarForecastModel(degree=3)
        m.fit(train)
        m.calibrate(backtest)
        for ts, _ in backtest[:20]:
            q10, q50, q90 = m.predict_quantiles(ts)
            self.assertLessEqual(q10, q50)
            self.assertLessEqual(q50, q90)

    def test_calibrate_reports_mape(self):
        series = _synthetic_hourly_series(24 * 20, noise_sd=1.0)
        train, backtest = series[:-24 * 7], series[-24 * 7:]
        m = fm.PolynomialCalendarForecastModel(degree=3)
        m.fit(train)
        info = m.calibrate(backtest)
        self.assertIsNotNone(info["mape"])
        self.assertLess(info["mape"], 10.0)


class TestPolynomialCalendarForecastModelArtifact(unittest.TestCase):
    def test_roundtrip_preserves_predictions(self):
        series = _synthetic_hourly_series(24 * 15, noise_sd=1.0)
        m = fm.PolynomialCalendarForecastModel(degree=2)
        m.fit(series)
        m.calibrate(series[-50:])
        artifact = m.to_artifact()
        self.assertEqual(artifact["model_type"], "poly_calendar")

        restored = fm.PolynomialCalendarForecastModel.from_artifact(artifact)
        for ts in (0.0, 3600.0 * 50, 3600.0 * 200):
            self.assertAlmostEqual(m.point_forecast(ts), restored.point_forecast(ts), places=6)

    def test_registered_in_model_classes_and_loader(self):
        self.assertIn("poly_calendar", fm.FORECAST_MODEL_CLASSES)
        m = fm.PolynomialCalendarForecastModel(degree=1)
        m.fit(_synthetic_hourly_series(24 * 10))
        restored = fm.load_forecast_model_from_artifact("poly_calendar", m.to_artifact())
        self.assertIsInstance(restored, fm.PolynomialCalendarForecastModel)


if __name__ == "__main__":
    unittest.main(verbosity=2)
