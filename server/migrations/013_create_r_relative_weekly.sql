-- 013_create_r_relative_weekly.sql
--
-- SHIELD Stage 2, Part D (beta) — drift resistansi kontak RELATIF
-- mingguan per (trafo, phase), dihitung server/resistance_job.py dari
-- residual termal terhadap SATU model baseline BEKU (ml_models.
-- is_frozen_reference=true). Lihat server/resistance.py untuk formula
-- dan server/resistance_job.py untuk alur penuh.
--
-- week_start = Senin (UTC) minggu itu. status menjelaskan kenapa baris
-- tidak punya nilai kalau bukan 'ok' -- TIDAK PERNAH mengarang angka
-- untuk mengisi kekosongan (lihat kolom reason).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/013_create_r_relative_weekly.sql
--
-- Rollback: server/migrations/013_create_r_relative_weekly_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS r_relative_weekly (
    trafo                     TEXT NOT NULL,
    phase                     TEXT NOT NULL CHECK (phase IN ('R', 'S', 'T')),
    week_start                DATE NOT NULL,
    r_ratio_median_pct        DOUBLE PRECISION NULL,
    r_ratio_q1_pct            DOUBLE PRECISION NULL,
    r_ratio_q3_pct            DOUBLE PRECISION NULL,
    r_ratio_iqr_pct           DOUBLE PRECISION NULL,
    phase_relative_ratio_pct  DOUBLE PRECISION NULL,
    n_samples_valid           INTEGER NOT NULL DEFAULT 0,
    n_samples_total           INTEGER NOT NULL DEFAULT 0,
    reference_model_id        BIGINT NULL REFERENCES ml_models(id),
    a_ref                     DOUBLE PRECISION NULL,
    status                    TEXT NOT NULL CHECK (status IN (
                                  'ok', 'insufficient_data', 'no_reference',
                                  'needs_recalibration', 'no_physics_rc_available'
                              )),
    reason                    TEXT NULL,
    created_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trafo, phase, week_start)
);

COMMENT ON TABLE r_relative_weekly IS
  'Estimasi mingguan drift resistansi kontak RELATIF (Stage 2 Part D, beta) -- BUKAN pengganti resistance_test aktual (maintenance_events, Part E). status != ''ok'' berarti kolom nilai (r_ratio_*, phase_relative_ratio_pct) NULL dengan alasan di `reason`, bukan angka dikarang.';
COMMENT ON COLUMN r_relative_weekly.a_ref IS
  'Koefisien A milik model referensi (atau "dipinjam" dari model physics_rc lain kalau referensi adalah ml_gbm, lihat server/resistance.py). Dicatat di sini untuk audit/reproduksi, bukan untuk dipakai ulang langsung.';
COMMENT ON COLUMN r_relative_weekly.phase_relative_ratio_pct IS
  'Rasio median fasa ini terhadap median KETIGA fasa minggu yang sama (persen) -- NULL kalau salah satu dari ketiga fasa tidak punya median minggu itu (lihat server/resistance.phase_relative_ratio()).';

CREATE INDEX IF NOT EXISTS idx_r_relative_weekly_trafo_phase_week ON r_relative_weekly (trafo, phase, week_start DESC);

COMMIT;
