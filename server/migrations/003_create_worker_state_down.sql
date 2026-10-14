-- Rollback 003_create_worker_state.sql
--   psql "$DATABASE_URL" -f server/migrations/003_create_worker_state_down.sql

BEGIN;

DROP TABLE IF EXISTS worker_state;

COMMIT;
