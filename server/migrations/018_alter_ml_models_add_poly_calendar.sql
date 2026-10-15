-- 018_alter_ml_models_add_poly_calendar.sql
--
-- Go-live (2026-10-15), rework of the "Forecast Realtime" tab -- adds a
-- third forecast model_type, 'poly_calendar' (polynomial regression over
-- calendar features, ml/forecast_models.py PolynomialCalendarForecastModel),
-- alongside the 'seasonal_naive'/'gbm_calendar' pair added by migration 014.
-- Same registry/table, same activate/freeze-reference mechanism -- this
-- migration only widens the model_type CHECK constraint.
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/018_alter_ml_models_add_poly_calendar.sql
--
-- Rollback: server/migrations/018_alter_ml_models_add_poly_calendar_down.sql

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
    'seasonal_naive', 'gbm_calendar', 'poly_calendar'
));

COMMENT ON COLUMN ml_models.model_type IS
  'physics_quadratic/physics_rc/ml_gbm/hybrid = baseline suhu sehat (Part B). seasonal_naive/gbm_calendar/poly_calendar = model forecasting BEBAN (Part F + go-live rework) -- tabel/mekanisme registry SAMA, dibedakan oleh model_type ini, bukan tabel terpisah.';

COMMIT;
