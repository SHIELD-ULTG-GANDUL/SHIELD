-- 014_alter_ml_models_add_forecast_types.sql
--
-- SHIELD Stage 2, Part F — menambah 2 model_type baru ke ml_models (yang
-- SAMA dipakai Part B) untuk model FORECASTING BEBAN: 'seasonal_naive'
-- (sama jam minggu lalu) dan 'gbm_calendar' (gradient boosting atas fitur
-- kalender -- jam/hari, BUKAN lag beban masa lalu). Registry dan mekanisme
-- activate/freeze-reference (server/api.py) dipakai ULANG apa adanya --
-- is_active=true di sini berarti "dipakai server/forecast_job.py", bukan
-- "dipakai lntai_worker.py" (itu tetap khusus model_type Part B, tidak
-- tertukar karena keduanya dibedakan trafo+phase+model_type, bukan
-- is_active saja).
--
-- Constraint lama (dari migrasi 009) diberi nama eksplisit lebih dulu
-- (Postgres menamainya otomatis "ml_models_model_type_check" untuk CHECK
-- inline tanpa nama -- di-DROP pakai nama itu, lalu diganti yang baru).
--
-- Jalankan manual (idempotent -- aman dijalankan ulang, lihat DO block):
--   psql "$DATABASE_URL" -f server/migrations/014_alter_ml_models_add_forecast_types.sql
--
-- Rollback: server/migrations/014_alter_ml_models_add_forecast_types_down.sql

BEGIN;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ml_models_model_type_check'
    ) THEN
        ALTER TABLE ml_models DROP CONSTRAINT ml_models_model_type_check;
    END IF;
END $$;

ALTER TABLE ml_models ADD CONSTRAINT ml_models_model_type_check CHECK (model_type IN (
    'physics_quadratic', 'physics_rc', 'ml_gbm', 'hybrid',
    'seasonal_naive', 'gbm_calendar'
));

COMMENT ON COLUMN ml_models.model_type IS
  'physics_quadratic/physics_rc/ml_gbm/hybrid = baseline suhu sehat (Part B). seasonal_naive/gbm_calendar = model forecasting BEBAN (Part F) -- tabel/mekanisme registry SAMA, dibedakan oleh model_type ini, bukan tabel terpisah.';

COMMIT;
