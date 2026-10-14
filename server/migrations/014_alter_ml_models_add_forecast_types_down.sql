-- Rollback 014_alter_ml_models_add_forecast_types.sql
--
-- PERINGATAN: kalau ada baris ml_models dengan model_type='seasonal_naive'
-- atau 'gbm_calendar' SAAT rollback ini dijalankan, ALTER TABLE di bawah
-- akan GAGAL (melanggar constraint lama) -- hapus/ubah baris itu dulu
-- kalau memang ingin rollback penuh ke Part B saja.
--
--   psql "$DATABASE_URL" -f server/migrations/014_alter_ml_models_add_forecast_types_down.sql

BEGIN;

ALTER TABLE ml_models DROP CONSTRAINT IF EXISTS ml_models_model_type_check;
ALTER TABLE ml_models ADD CONSTRAINT ml_models_model_type_check
    CHECK (model_type IN ('physics_quadratic', 'physics_rc', 'ml_gbm', 'hybrid'));

COMMIT;
