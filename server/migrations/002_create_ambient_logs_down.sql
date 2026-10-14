-- Rollback 002_create_ambient_logs.sql
--
-- Jalankan manual:
--   psql "$DATABASE_URL" -f server/migrations/002_create_ambient_logs_down.sql
--
-- PERINGATAN: ini MENGHAPUS seluruh histori ambient_logs. Backup dulu kalau
-- perlu: pg_dump -t ambient_logs "$DATABASE_URL" > ambient_logs_backup.sql

BEGIN;

DROP TABLE IF EXISTS ambient_logs;

COMMIT;
