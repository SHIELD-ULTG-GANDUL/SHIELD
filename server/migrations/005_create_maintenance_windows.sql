-- 005_create_maintenance_windows.sql
--
-- SHIELD Tahap 1, T6 (DIREVISI) — rencana awal T6 memakai TimescaleDB
-- (hypertable, continuous aggregate, retention policy otomatis). Audit
-- Fase 0 mengonfirmasi database ini PostgreSQL 18.4 POLOS, TimescaleDB
-- TIDAK terpasang -- rencana itu dibatalkan dan diganti desain di bawah
-- (tabel biasa + job terjadwal, bukan fitur TimescaleDB apa pun).
--
-- maintenance_windows: jendela waktu yang ditandai manual sebagai
-- "periode maintenance/kalibrasi/gangguan" untuk satu trafo (atau semua
-- trafo kalau trafo_id NULL). Dipakai oleh server/archive_job.py (lihat
-- migrasi 006) untuk menyalin baris trafo_temperature/trafo_load dalam
-- jendela itu ke tabel *_archive SEBELUM retention apa pun berjalan --
-- retention sendiri TIDAK diaktifkan di Tahap 1 ini (lihat migrasi 007,
-- sengaja dikomentari, keputusan dipegang pengguna).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/005_create_maintenance_windows.sql
--
-- Rollback: server/migrations/005_create_maintenance_windows_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS maintenance_windows (
    id          BIGSERIAL PRIMARY KEY,
    trafo_id    TEXT NULL,
    start_time  TIMESTAMPTZ NOT NULL,
    end_time    TIMESTAMPTZ NOT NULL,
    reason      TEXT NOT NULL,
    created_by  TEXT NULL,
    archived    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT maintenance_windows_valid_range CHECK (end_time > start_time)
);

COMMENT ON TABLE maintenance_windows IS
  'Jendela waktu maintenance/kalibrasi/gangguan ditandai manual (T6). trafo_id NULL = berlaku semua trafo. archived=true setelah server/archive_job.py menyalin baris dalam jendela ini ke *_archive (lihat migrasi 006).';
COMMENT ON COLUMN maintenance_windows.trafo_id IS
  'NULL = jendela berlaku untuk SEMUA trafo (2, 3, 4), bukan hanya satu.';
COMMENT ON COLUMN maintenance_windows.archived IS
  'true setelah archive_job.py berhasil menyalin baris trafo_temperature/trafo_load dalam jendela ini ke tabel *_archive -- idempotent, job melewati baris archived=true.';

CREATE INDEX IF NOT EXISTS idx_maintenance_windows_time ON maintenance_windows (start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_maintenance_windows_pending ON maintenance_windows (archived) WHERE archived = FALSE;

COMMIT;
