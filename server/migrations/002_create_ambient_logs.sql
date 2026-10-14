-- 002_create_ambient_logs.sql
--
-- SHIELD Tahap 1, T3 — tabel baru ambient_logs: suhu ambient (cuaca) di
-- sekitar GI Gandul, dipakai sebagai baseline LNTAI (T4), TERPISAH dari
-- kolom trafo_temperature.temp_ambient (yang dicadangkan untuk sensor
-- ambient di DALAM panel trafo di masa depan, lihat komentar kolom itu
-- di 001_add_quality_flags.sql dan docs/DATA_DICTIONARY.md).
--
-- Cabang B (dikonfirmasi lewat audit Fase 0: curl dari server ke
-- api.open-meteo.com gagal dengan "self-signed certificate in certificate
-- chain" -- proxy keluar meng-intercept TLS, server TIDAK bisa memanggil
-- Open-Meteo langsung). Karena itu baris di tabel ini ditulis lewat
-- POST /api/ambient-logs oleh BROWSER (yang sudah berhasil fetch
-- Open-Meteo langsung dari sisi klien), bukan oleh worker server-side.
-- Konsekuensi yang didokumentasikan: histori ambient hanya terisi selama
-- ada dashboard yang terbuka di browser -- ADA JEDA saat tidak ada
-- browser aktif. Ini bukan bug, tapi keterbatasan arsitektur Cabang B yang
-- disengaja (lihat server/README.md).
--
-- ok=false, temperature_c=NULL ditulis saat browser gagal fetch
-- Open-Meteo -- TIDAK PERNAH diisi dengan nilai yang dikarang/diinterpolasi.
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/002_create_ambient_logs.sql
--
-- Rollback: server/migrations/002_create_ambient_logs_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS ambient_logs (
    id              BIGSERIAL PRIMARY KEY,
    "timestamp"     TIMESTAMPTZ NOT NULL,
    slot_10min      TIMESTAMPTZ NOT NULL,
    temperature_c   DOUBLE PRECISION NULL,
    ok              BOOLEAN NOT NULL,
    source          TEXT NOT NULL DEFAULT 'open-meteo-browser-relay',
    latitude        DOUBLE PRECISION NOT NULL,
    longitude       DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ambient_logs_slot_unique UNIQUE (slot_10min)
);

COMMENT ON TABLE ambient_logs IS
  'Suhu ambient (cuaca) sekitar GI Gandul dari Open-Meteo, direlay browser (Cabang B, T3) -- bukan sensor di dalam panel trafo. Satu baris per slot 10 menit (dedup server-side via slot_10min UNIQUE).';
COMMENT ON COLUMN ambient_logs."timestamp" IS
  'Waktu sebenarnya saat browser menerima respons Open-Meteo (UTC).';
COMMENT ON COLUMN ambient_logs.slot_10min IS
  'timestamp dibulatkan ke bawah ke kelipatan 10 menit -- kunci dedup, maksimal satu baris tersimpan per slot per definisi Cabang B.';
COMMENT ON COLUMN ambient_logs.temperature_c IS
  'Suhu ambient (°C). NULL kalau ok=false (fetch Open-Meteo gagal) -- TIDAK PERNAH diisi nilai interpolasi/dikarang.';
COMMENT ON COLUMN ambient_logs.ok IS
  'false kalau browser gagal fetch Open-Meteo untuk slot ini (lihat temperature_c).';

CREATE INDEX IF NOT EXISTS idx_ambient_logs_timestamp ON ambient_logs ("timestamp" DESC);

COMMIT;
