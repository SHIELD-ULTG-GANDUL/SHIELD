-- 001_add_quality_flags_down.sql — rollback untuk 001_add_quality_flags.sql
--
-- Menghapus kolom quality_flags dari trafo_temperature dan trafo_load.
-- PERINGATAN: kalau T5 sudah berjalan dan mengisi kolom ini, data di
-- kolom tersebut akan HILANG permanen. Backup dulu kalau ragu:
--   psql "$DATABASE_URL" -c "\copy (SELECT id, quality_flags FROM trafo_temperature WHERE quality_flags IS NOT NULL) TO 'quality_flags_temp_backup.csv' CSV HEADER"
--   psql "$DATABASE_URL" -c "\copy (SELECT id, quality_flags FROM trafo_load WHERE quality_flags IS NOT NULL) TO 'quality_flags_load_backup.csv' CSV HEADER"
--
-- Jalankan manual:
--   psql "$DATABASE_URL" -f server/migrations/001_add_quality_flags_down.sql

BEGIN;

ALTER TABLE trafo_temperature DROP COLUMN IF EXISTS quality_flags;
ALTER TABLE trafo_load        DROP COLUMN IF EXISTS quality_flags;

COMMIT;
