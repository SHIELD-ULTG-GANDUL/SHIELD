-- 006_create_archive_tables.sql
--
-- SHIELD Tahap 1, T6 (DIREVISI) — trafo_temperature_archive /
-- trafo_load_archive: salinan baris yang jatuh di dalam
-- maintenance_windows (migrasi 005), ditulis oleh server/archive_job.py.
--
-- "LIKE ... INCLUDING INDEXES INCLUDING CONSTRAINTS" (TANPA "INCLUDING
-- DEFAULTS") sengaja: kita MAU PRIMARY KEY + CHECK constraint ikut
-- tersalin (supaya ON CONFLICT (id) DO NOTHING di archive_job.py bisa
-- idempotent), tapi TIDAK MAU kolom id punya default nextval(...) yang
-- menunjuk ke SEQUENCE milik tabel asli -- archive_job.py selalu menyalin
-- nilai id yang sudah ada (bukan membuat baris baru), jadi default itu
-- tidak pernah dipakai dan hanya berisiko membingungkan di kemudian hari
-- kalau ada yang INSERT manual tanpa id.
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/006_create_archive_tables.sql
--
-- Rollback: server/migrations/006_create_archive_tables_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS trafo_temperature_archive (
    LIKE trafo_temperature INCLUDING INDEXES INCLUDING CONSTRAINTS
);
ALTER TABLE trafo_temperature_archive ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS trafo_load_archive (
    LIKE trafo_load INCLUDING INDEXES INCLUDING CONSTRAINTS
);
ALTER TABLE trafo_load_archive ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ NOT NULL DEFAULT now();

COMMENT ON TABLE trafo_temperature_archive IS
  'Salinan baris trafo_temperature yang jatuh di dalam satu maintenance_windows (T6), ditulis server/archive_job.py. Baris ASLI di trafo_temperature TIDAK dihapus oleh job ini -- lihat migrasi 007 (retention, tidak aktif) untuk penghapusan, yang terpisah dan butuh keputusan eksplisit.';
COMMENT ON TABLE trafo_load_archive IS
  'Salinan baris trafo_load yang jatuh di dalam satu maintenance_windows (T6), ditulis server/archive_job.py. Lihat catatan di trafo_temperature_archive.';

COMMIT;
