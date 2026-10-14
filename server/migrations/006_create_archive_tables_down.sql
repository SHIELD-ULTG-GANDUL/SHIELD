-- Rollback 006_create_archive_tables.sql
--
-- PERINGATAN: menghapus seluruh salinan arsip. Baris ASLI di
-- trafo_temperature/trafo_load TIDAK terpengaruh (archive adalah salinan,
-- bukan pemindahan). Backup dulu kalau perlu:
--   pg_dump -t trafo_temperature_archive -t trafo_load_archive "$DATABASE_URL" > archive_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/006_create_archive_tables_down.sql

BEGIN;

DROP TABLE IF EXISTS trafo_temperature_archive;
DROP TABLE IF EXISTS trafo_load_archive;

COMMIT;
