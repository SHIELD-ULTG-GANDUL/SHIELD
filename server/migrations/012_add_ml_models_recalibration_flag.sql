-- 012_add_ml_models_recalibration_flag.sql
--
-- SHIELD Stage 2, Part E — kolom nullable di ml_models: diisi otomatis oleh
-- POST /api/maintenance-events (server/api.py) saat sebuah event
-- resets_baseline=true dibuat untuk trafo (dan fasa, kalau event itu
-- spesifik satu fasa) yang punya baris is_frozen_reference=true. TIDAK
-- menghapus atau mengubah model yang sudah ada -- hanya menandai perlu
-- ditinjau ulang (keputusan re-kalibrasi tetap manual, lewat
-- ml/train_baseline.py + POST .../freeze-reference seperti biasa).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/012_add_ml_models_recalibration_flag.sql
--
-- Rollback: server/migrations/012_add_ml_models_recalibration_flag_down.sql

BEGIN;

ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS recalibration_note TEXT NULL;

COMMENT ON COLUMN ml_models.recalibration_note IS
  'Diisi otomatis ("Needs re-calibration after maintenance on <tanggal>") saat POST /api/maintenance-events membuat event resets_baseline=true untuk (trafo, phase) model beku (is_frozen_reference=true) ini. NULL = belum ada event semacam itu sejak model ini dibekukan/dibuat. Dikosongkan manual (lewat psql atau endpoint di masa depan) setelah model ditinjau/dilatih ulang.';

COMMIT;
