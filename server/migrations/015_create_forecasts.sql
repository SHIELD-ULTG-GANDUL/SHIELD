-- 015_create_forecasts.sql
--
-- SHIELD Stage 2, Part F — hasil forecast per jam, ditulis
-- server/forecast_job.py dari model forecasting beban AKTIF (ml_models
-- model_type IN ('seasonal_naive','gbm_calendar'), is_active=true, lihat
-- migrasi 014). Satu baris per (trafo, phase, target_ts) -- method
-- mencatat metode mana yang MENGHASILKAN baris ini (bukan dua baris
-- untuk dibandingkan; perbandingan metode ada di ml_models.metrics saat
-- training/backtest, lihat ml/train_load_forecast.py).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/015_create_forecasts.sql
--
-- Rollback: server/migrations/015_create_forecasts_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS forecasts (
    trafo                TEXT NOT NULL,
    phase                TEXT NOT NULL CHECK (phase IN ('R', 'S', 'T')),
    target_ts            TIMESTAMPTZ NOT NULL,
    load_p10_a           DOUBLE PRECISION NULL,
    load_p50_a           DOUBLE PRECISION NULL,
    load_p90_a           DOUBLE PRECISION NULL,
    t_hot_pred_c         DOUBLE PRECISION NULL,
    lntai_pred_p10       DOUBLE PRECISION NULL,
    lntai_pred_p50       DOUBLE PRECISION NULL,
    lntai_pred_p90       DOUBLE PRECISION NULL,
    aging_f_pred         DOUBLE PRECISION NULL,
    method               TEXT NOT NULL,
    excess_method        TEXT NOT NULL,
    forecast_model_id    BIGINT NULL REFERENCES ml_models(id),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trafo, phase, target_ts)
);

COMMENT ON TABLE forecasts IS
  'Forecast per jam (Stage 2 Part F) -- load (arus) P10/P50/P90, suhu hotspot prediksi, skor LNTAI prediksi (P10/P50/P90, DIJAMIN terurut naik -- lihat server/forecast.py untuk kenapa diurutkan ulang, bukan diturunkan langsung dari load P10/P50/P90), dan faktor penuaan Arrhenius prediksi (Part C, pada beban P50).';
COMMENT ON COLUMN forecasts.method IS
  'Model forecasting beban yang dipakai: "seasonal_naive" atau "gbm_calendar" (lihat ml_models.model_type). Baris TIDAK ditulis sama sekali kalau belum ada model aktif untuk (trafo,phase) ini -- bukan NULL/fallback diam-diam.';
COMMENT ON COLUMN forecasts.excess_method IS
  '"r_relative" (dari r_relative_weekly terbaru, Part D) atau "recent_residual_median" (fallback kalau R/R_ref belum tersedia/tidak ok) -- lihat server/forecast.py compute_excess().';
COMMENT ON COLUMN forecasts.aging_f_pred IS
  'Faktor Arrhenius F (Part C) pada t_hot_pred_c -- NULL kalau aging_config.json belum terkonfigurasi (ea_ev/t_ref_c null), sama seperti server/aging.py.';

CREATE INDEX IF NOT EXISTS idx_forecasts_trafo_phase_ts ON forecasts (trafo, phase, target_ts);

COMMIT;
