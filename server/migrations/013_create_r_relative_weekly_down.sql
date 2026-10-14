-- Rollback 013_create_r_relative_weekly.sql
--
-- PERINGATAN: menghapus seluruh histori drift resistansi relatif mingguan.
-- Backup dulu kalau perlu:
--   pg_dump -t r_relative_weekly "$DATABASE_URL" > r_relative_weekly_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/013_create_r_relative_weekly_down.sql

BEGIN;

DROP TABLE IF EXISTS r_relative_weekly;

COMMIT;
