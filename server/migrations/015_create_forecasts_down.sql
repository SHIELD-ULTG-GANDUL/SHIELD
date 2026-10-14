-- Rollback 015_create_forecasts.sql
--
-- PERINGATAN: menghapus seluruh histori forecast. Backup dulu kalau perlu:
--   pg_dump -t forecasts "$DATABASE_URL" > forecasts_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/015_create_forecasts_down.sql

BEGIN;

DROP TABLE IF EXISTS forecasts;

COMMIT;
