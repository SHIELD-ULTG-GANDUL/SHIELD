-- Rollback 008_add_lntai_v3_columns.sql
--
-- Menghapus 3 kolom v3 -- baris v3 (model_version='lntai-v3') yang sudah
-- ada TETAP ADA sebagai baris, hanya kehilangan 3 kolom ini. Kalau Anda
-- juga ingin membuang baris v3 itu sendiri, jalankan terpisah:
--   DELETE FROM lntai_results WHERE model_version = 'lntai-v3';
-- (tidak termasuk di file ini -- rollback kolom dan penghapusan baris
-- adalah dua keputusan berbeda).
--
--   psql "$DATABASE_URL" -f server/migrations/008_add_lntai_v3_columns_down.sql

BEGIN;

ALTER TABLE lntai_results DROP COLUMN IF EXISTS baseline_id;
ALTER TABLE lntai_results DROP COLUMN IF EXISTS calibrated;
ALTER TABLE lntai_results DROP COLUMN IF EXISTS lntai_rolling_median_30min;

COMMIT;
