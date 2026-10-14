-- Rollback 010_create_aging_daily.sql
--
-- PERINGATAN: menghapus seluruh histori penuaan harian. Backup dulu kalau
-- perlu: pg_dump -t aging_daily "$DATABASE_URL" > aging_daily_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/010_create_aging_daily_down.sql

BEGIN;

DROP TABLE IF EXISTS aging_daily;

COMMIT;
