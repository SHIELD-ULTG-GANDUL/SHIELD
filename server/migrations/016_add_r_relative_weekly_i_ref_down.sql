-- Rollback 016_add_r_relative_weekly_i_ref.sql
--   psql "$DATABASE_URL" -f server/migrations/016_add_r_relative_weekly_i_ref_down.sql

BEGIN;

ALTER TABLE r_relative_weekly DROP COLUMN IF EXISTS i_ref;

COMMIT;
