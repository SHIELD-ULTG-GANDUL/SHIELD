-- 003_create_worker_state.sql
--
-- SHIELD Tahap 1, T4 — tabel checkpoint generik untuk worker batch
-- (lntai_worker.py, dan worker lain di masa depan). Satu baris per
-- (worker_name, trafo_id): "sampai timestamp berapa worker ini sudah
-- memproses data trafo ini". Dipakai supaya restart worker tidak
-- memproses ulang dari awal, dan backfill_lntai.py bisa membaca/menulis
-- checkpoint yang sama lewat --dry-run tanpa mengganggu worker live
-- (baca saja, lihat backfill_lntai.py).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/003_create_worker_state.sql
--
-- Rollback: server/migrations/003_create_worker_state_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS worker_state (
    worker_name         TEXT NOT NULL,
    trafo_id            TEXT NOT NULL,
    last_processed_ts   TIMESTAMPTZ NULL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (worker_name, trafo_id)
);

COMMENT ON TABLE worker_state IS
  'Checkpoint generik per worker batch + trafo. last_processed_ts = baris trafo_temperature terakhir yang SUDAH diproses (exclusive -- worker lanjut dari > last_processed_ts). NULL berarti belum pernah berjalan untuk trafo ini.';

COMMIT;
