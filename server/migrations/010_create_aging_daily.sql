-- 010_create_aging_daily.sql
--
-- SHIELD Stage 2, Part C — hasil harian faktor percepatan penuaan Arrhenius
-- per (trafo, phase), ditulis server/aging_job.py. Lihat server/aging.py
-- untuk definisi F(t)/EqAge dan catatan cakupan (HANYA kontak finger,
-- BUKAN interrupter vakum/mekanisme penggerak).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/010_create_aging_daily.sql
--
-- Rollback: server/migrations/010_create_aging_daily_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS aging_daily (
    trafo           TEXT NOT NULL,
    phase           TEXT NOT NULL CHECK (phase IN ('R', 'S', 'T')),
    date            DATE NOT NULL,
    f_mean          DOUBLE PRECISION NULL,
    f_peak          DOUBLE PRECISION NULL,
    t_hot_mean_c    DOUBLE PRECISION NULL,
    t_hot_peak_c    DOUBLE PRECISION NULL,
    eq_days_cum     DOUBLE PRECISION NULL,
    calendar_days   DOUBLE PRECISION NULL,
    config_hash     TEXT NOT NULL,
    reset_date      TIMESTAMPTZ NOT NULL,
    reset_source    TEXT NOT NULL,
    n_samples       INTEGER NOT NULL DEFAULT 0,
    n_gaps_excluded INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trafo, phase, date)
);

COMMENT ON TABLE aging_daily IS
  'Agregat harian faktor penuaan Arrhenius (Stage 2 Part C) -- CAKUPAN HANYA sambungan kontak finger, BUKAN interrupter vakum/mekanisme penggerak (lihat server/aging.AGING_SCOPE_TEXT). f_mean/f_peak/eq_days_cum NULL kalau aging_config.json belum terkonfigurasi (ea_ev/t_ref_c null) pada hari itu.';
COMMENT ON COLUMN aging_daily.eq_days_cum IS
  'Usia ekivalen KUMULATIF (hari) sejak reset_date -- BUKAN usia kalender, lihat server/aging.py. Reset ke 0 kalau reset_date baris ini lebih baru dari baris hari sebelumnya (event reset baru).';
COMMENT ON COLUMN aging_daily.reset_source IS
  'Asal titik reset: "fallback_config" (aging_config.json fallback_reset_date, dipakai selagi Part E belum ada) atau "maintenance_event" (Part E, baris resets_baseline=true) di masa depan.';
COMMENT ON COLUMN aging_daily.config_hash IS
  'Hash SHA-256 aging_config.json saat baris ini dihitung -- dipakai mendeteksi baris lama yang dihitung dengan ea_ev/t_ref_c BERBEDA dari konfigurasi saat ini (lihat server/aging_job.py).';

CREATE INDEX IF NOT EXISTS idx_aging_daily_trafo_phase_date ON aging_daily (trafo, phase, date DESC);

COMMIT;
