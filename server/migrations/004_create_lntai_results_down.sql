-- Rollback 004_create_lntai_results.sql
--
-- PERINGATAN: ini menghapus seluruh histori LNTAI yang sudah dihitung
-- server-side. Backup dulu kalau perlu:
--   pg_dump -t lntai_results "$DATABASE_URL" > lntai_results_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/004_create_lntai_results_down.sql

BEGIN;

DROP TABLE IF EXISTS lntai_results;

COMMIT;
