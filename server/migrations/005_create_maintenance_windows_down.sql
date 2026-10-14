-- Rollback 005_create_maintenance_windows.sql
--   psql "$DATABASE_URL" -f server/migrations/005_create_maintenance_windows_down.sql

BEGIN;

DROP TABLE IF EXISTS maintenance_windows;

COMMIT;
