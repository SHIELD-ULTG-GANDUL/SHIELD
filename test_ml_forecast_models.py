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


def _synthetic_hourly_series_heteroscedastic(
    n_hours: int, day_noise_sd: float, night_noise_sd: float, seed: int = 1,
) -> list[tuple[float, float]]:
    """M11 (autopilot 2026-11): sama seperti _synthetic_hourly_series, tapi
    noise SIANG (WIB 06:00-18:00) dan MALAM beda standar deviasinya --
    dipakai membuktikan offset kuantil PER-BUCKET lebih lebar di jam
    dengan sebaran residual yang memang lebih lebar, bukan satu band
    global yang sama untuk kedua kondisi."""
    random.seed(seed)
    series = []
    for i in range(n_hours):
        ts = i * 3600.0
        hour_sin, hour_cos, weekday = fm.hour_sin_cos_weekday(ts)
        wib_hour = fm._wib_hour_of(ts)
        is_day = 6 <= wib_hour < 18
        noise_sd = day_noise_sd if is_day else night_noise_sd
        weekday_bump = 20.0 if weekday < 5 else 0.0
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


class TestQuantileOffsetsByBucket(unittest.TestCase):
    """M11 (autopilot 2026-11): offset P10/P90 per bucket jam WIB."""

    def test_day_band_wider_than_night_band_with_heteroscedastic_noise(self):
        # Noise siang 3x malam -- band P10-P90 bucket siang HARUS lebih
        # lebar daripada bucket malam, kalau bucketing benar-benar jalan
        # (bukan satu offset global yang sama untuk semua jam).
        series = _synthetic_hourly_series_heteroscedastic(
            24 * 40, day_noise_sd=9.0, night_noise_sd=3.0,
        )
        train, backtest = series[:-24 * 14], series[-24 * 14:]
        m = fm.PolynomialCalendarForecastModel(degree=3)
        m.fit(train)
        m.calibrate(backtest)

        self.assertTrue(m.buckets, "buckets harus terisi dari ml_config.json setelah calibrate()")
        buckets = m.buckets
        day_idxs = [i for i, (lo, hi) in enumerate(buckets) if lo >= 6 and hi <= 18]
        night_idxs = [i for i in range(len(buckets)) if i not in day_idxs]
        self.assertTrue(day_idxs and night_idxs, "bucket default harus mencakup jam siang & malam terpisah")

        day_widths = [m.bucket_offsets[i][2] - m.bucket_offsets[i][0] for i in day_idxs]
        night_widths = [m.bucket_offsets[i][2] - m.bucket_offsets[i][0] for i in night_idxs]
        self.assertGreater(
            min(day_widths), max(night_widths),
            f"band siang (lebar={day_widths}) harus lebih lebar dari band malam (lebar={night_widths})",
        )

    def test_coverage_p10_p90_in_reasonable_range(self):
        series = _synthetic_hourly_series(24 * 40, noise_sd=5.0)
        train, backtest = series[:-24 * 14], series[-24 * 14:]
        m = fm.PolynomialCalendarForecastModel(degree=3)
        m.fit(train)
        info = m.calibrate(backtest)
        self.assertIsNotNone(info["coverage_p10_p90"])
        self.assertGreaterEqual(info["coverage_p10_p90"], 0.70)
        self.assertLessEqual(info["coverage_p10_p90"], 0.90)

    def test_quantiles_still_monotonic_with_bucket_offsets(self):
        series = _synthetic_hourly_series_heteroscedastic(24 * 30, day_noise_sd=8.0, night_noise_sd=1.0)
        train, backtest = series[:-24 * 10], series[-24 * 10:]
        m = fm.PolynomialCalendarForecastModel(degree=3)
        m.fit(train)
        m.calibrate(backtest)
        for ts, _ in backtest[:30]:
            q10, q50, q90 = m.predict_quantiles(ts)
            self.assertLessEqual(q10, q50)
            self.assertLessEqual(q50, q90)

    def test_old_artifact_without_bucket_keys_falls_back_to_global_offset(self):
        # Artefak dari SEBELUM M11 tidak punya key buckets/bucket_offsets
        # sama sekali -- from_artifact() & predict_quantiles() harus tetap
        # jalan (fallback ke offset global), bukan KeyError/crash.
        series = _synthetic_hourly_series(24 * 15, noise_sd=1.0)
        m = fm.PolynomialCalendarForecastModel(degree=2)
        m.fit(series)
        m.calibrate(series[-50:])
        artifact = m.to_artifact()
        del artifact["buckets"]
        del artifact["bucket_offsets"]

        restored = fm.PolynomialCalendarForecastModel.from_artifact(artifact)
        self.assertEqual(restored.buckets, [])
        q10, q50, q90 = restored.predict_quantiles(series[-1][0])
        self.assertIsNotNone(q50)
        self.assertLessEqual(q10, q50)
        self.assertLessEqual(q50, q90)
        # Harus persis sama dengan offset global (fallback), bukan 0/hilang.
        expected_q50 = restored.point_forecast(series[-1][0]) + restored.p50_offset
        self.assertAlmostEqual(q50, expected_q50, places=6)

    def test_sparse_bucket_falls_back_to_global_offset(self):
        # Backtest sangat pendek (< _MIN_BUCKET_RESIDUALS per bucket) --
        # bucket_offsets harus SAMA DENGAN offset global untuk tiap bucket,
        # bukan kuantil tak-stabil dari segelintir sampel.
        series = _synthetic_hourly_series(24 * 10, noise_sd=2.0)
        m = fm.PolynomialCalendarForecastModel(degree=2)
        m.fit(series[:-10])
        info = m.calibrate(series[-10:])
        self.assertLess(info["n_backtest"], fm._MIN_BUCKET_RESIDUALS)
        global_offsets = (m.p10_offset, m.p50_offset, m.p90_offset)
        for offs in m.bucket_offsets:
            self.assertEqual(tuple(offs), global_offsets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
