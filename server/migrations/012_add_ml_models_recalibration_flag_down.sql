-- Rollback 012_add_ml_models_recalibration_flag.sql
--   psql "$DATABASE_URL" -f server/migrations/012_add_ml_models_recalibration_flag_down.sql

BEGIN;

ALTER TABLE ml_models DROP COLUMN IF EXISTS recalibration_note;

COMMIT;
