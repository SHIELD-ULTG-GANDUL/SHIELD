-- Rollback 018_alter_ml_models_add_poly_calendar.sql
--   psql "$DATABASE_URL" -f server/migrations/018_alter_ml_models_add_poly_calendar_down.sql
--
-- WARNING: only run this if no ml_models row has model_type='poly_calendar'
-- yet -- restoring the narrower constraint will fail (or orphan rows) if
-- any exist. Delete those rows first if you really need to roll back.

BEGIN;

ALTER TABLE ml_models DROP CONSTRAINT IF EXISTS ml_models_model_type_check;

ALTER TABLE ml_models ADD CONSTRAINT ml_models_model_type_check CHECK (model_type IN (
    'physics_quadratic', 'physics_rc', 'ml_gbm', 'hybrid',
    'seasonal_naive', 'gbm_calendar'
));

COMMIT;
