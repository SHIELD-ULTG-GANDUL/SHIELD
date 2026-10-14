-- 001_add_quality_flags.sql
--
-- SHIELD Tahap 1, T2 — tambah kolom quality_flags (bitmask INTEGER, nullable)
-- ke trafo_temperature dan trafo_load.
--
-- Definisi tiap bit (MISSING, ZERO_READING, OUT_OF_RANGE, STUCK, SPIKE,
-- PAIRING_GAP, TIME_GAP) ditulis di server/data_quality_config.json dan
-- dihitung oleh server/data_quality.py (sub-tugas T5) -- kolom ini sengaja
-- dibiarkan NULL/tidak diisi sampai T5 selesai. Aman dijalankan sekarang:
-- tidak mengubah data yang sudah ada, tidak mengunci tabel lama (ADD COLUMN
-- dengan default NULL adalah operasi metadata-only di PostgreSQL modern,
-- tidak menulis ulang tabel).
--
-- Database saat audit: PostgreSQL 18.4 polos (TimescaleDB TIDAK terpasang).
-- Sintaks di bawah adalah SQL standar, tidak memakai fitur TimescaleDB apa
-- pun.
--
-- Jalankan manual (idempotent — aman dijalankan berkali-kali):
--   psql "$DATABASE_URL" -f server/migrations/001_add_quality_flags.sql
--
-- Rollback: server/migrations/001_add_quality_flags_down.sql

BEGIN;

ALTER TABLE trafo_temperature ADD COLUMN IF NOT EXISTS quality_flags INTEGER NULL;
ALTER TABLE trafo_load        ADD COLUMN IF NOT EXISTS quality_flags INTEGER NULL;

COMMENT ON COLUMN trafo_temperature.quality_flags IS
  'Bitmask kualitas data (lihat server/data_quality_config.json). NULL = belum pernah dihitung (baris lama / T5 belum berjalan).';
COMMENT ON COLUMN trafo_load.quality_flags IS
  'Bitmask kualitas data (lihat server/data_quality_config.json). NULL = belum pernah dihitung (baris lama / T5 belum berjalan).';

COMMIT;
