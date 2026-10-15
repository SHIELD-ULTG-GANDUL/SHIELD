-- Rollback 017_add_ml_models_provisional_flag.sql
--   psql "$DATABASE_URL" -f server/migrations/017_add_ml_models_provisional_flag_down.sql

BEGIN;

ALTER TABLE ml_models DROP COLUMN IF EXISTS is_provisional;
ALTER TABLE ml_models DROP COLUMN IF EXISTS provisional_reason;

COMMIT;
