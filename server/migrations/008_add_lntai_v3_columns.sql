-- 008_add_lntai_v3_columns.sql
--
-- SHIELD Stage 2, Part A — LNTAI v3 (physics-informed residual z-score,
-- per fasa). Menambah 3 kolom NULLABLE ke lntai_results (TIDAK membuat
-- tabel baru -- v3 memakai PK gabungan (trafo_id, ts, model_version) yang
-- sama, dengan model_version='lntai-v3' membedakan baris dari 'lntai-v2').
-- Baris v2 yang sudah ada TIDAK berubah (3 kolom baru otomatis NULL untuk
-- baris lama, sesuai makna "tidak berlaku untuk model ini").
--
-- level/per_phase/quality_flags/paired/error/lntai/ts/model_version yang
-- SUDAH ADA dipakai ulang apa adanya oleh v3 (level v3 memakai label
-- berbeda -- Normal/Caution/Alert/Critical alih-alih
-- NORMAL/WARNING/CRITICAL/DATA_INVALID v2 -- tapi kolomnya TEXT biasa,
-- tidak perlu diubah).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/008_add_lntai_v3_columns.sql
--
-- Rollback: server/migrations/008_add_lntai_v3_columns_down.sql

BEGIN;

ALTER TABLE lntai_results ADD COLUMN IF NOT EXISTS baseline_id TEXT NULL;
ALTER TABLE lntai_results ADD COLUMN IF NOT EXISTS calibrated BOOLEAN NULL;
ALTER TABLE lntai_results ADD COLUMN IF NOT EXISTS lntai_rolling_median_30min DOUBLE PRECISION NULL;

COMMENT ON COLUMN lntai_results.baseline_id IS
  'Identitas baseline yang dipakai (T4: v2 selalu NULL; Stage 2 Part A: v3 diisi "legacy-v1-physics-uncalibrated", lihat lntai_v3_config.json).';
COMMENT ON COLUMN lntai_results.calibrated IS
  'false = baseline belum di-fit dari data trafo 2/3/4 riil (lihat lntai_v3_config.json baseline.calibrated). NULL untuk baris v2 (T4) -- kolom ini tidak berlaku untuk model itu.';
COMMENT ON COLUMN lntai_results.lntai_rolling_median_30min IS
  'Median bergulir 30 menit atas nilai lntai INSTAN (kolom "lntai") -- dipakai untuk menentukan "level" supaya tidak berubah-ubah karena satu sampel noise. NULL untuk baris v2 (T4), yang levelnya berasal langsung dari nilai instan.';

COMMIT;
