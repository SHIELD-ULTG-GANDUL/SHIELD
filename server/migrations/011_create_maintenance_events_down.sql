-- Rollback 011_create_maintenance_events.sql
--
-- PERINGATAN: menghapus seluruh log pemeliharaan/resistance-test, dan
-- aging_job.py (Part E) akan jatuh kembali ke aging_config.json
-- fallback_reset_date untuk SEMUA trafo/fasa setelah ini. Backup dulu:
--   pg_dump -t maintenance_events "$DATABASE_URL" > maintenance_events_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/011_create_maintenance_events_down.sql

BEGIN;

DROP TABLE IF EXISTS maintenance_events;

COMMIT;
