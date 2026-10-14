-- 011_create_maintenance_events.sql
--
-- SHIELD Stage 2, Part E — log pemeliharaan & resistance-test per trafo
-- (opsional per fasa). Menggantikan aging_config.json fallback_reset_date
-- sebagai sumber titik reset EqAge (lihat server/aging_job.py get_reset_date())
-- dan memicu tanda "perlu dikalibrasi ulang" pada model baseline beku
-- (ml_models.is_frozen_reference=true, lihat migrasi 012) saat
-- resets_baseline=true.
--
-- `trafo` SENGAJA bertipe TEXT TANPA batasan nilai 2/3/4 (beda dari
-- endpoint lain) -- data historis mencakup Trafo 1 (sudah tidak dipantau
-- sistem ini, tapi tetap bagian dari log pemeliharaan, lihat
-- server/scripts/import_ml_r_contact.py).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/011_create_maintenance_events.sql
--
-- Rollback: server/migrations/011_create_maintenance_events_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS maintenance_events (
    id              BIGSERIAL PRIMARY KEY,
    trafo           TEXT NOT NULL,
    phase           TEXT NULL CHECK (phase IN ('R', 'S', 'T')),
    event_type      TEXT NOT NULL CHECK (event_type IN (
                        'inspection', 'cleaning', 'tightening', 'replacement',
                        'resistance_test', 'thermovision'
                    )),
    event_ts        TIMESTAMPTZ NOT NULL,
    resistance_uohm DOUBLE PRECISION NULL,
    test_current_a  DOUBLE PRECISION NULL,
    notes           TEXT NULL,
    resets_baseline BOOLEAN NOT NULL DEFAULT FALSE,
    created_by      TEXT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE maintenance_events IS
  'Log pemeliharaan & resistance-test (Stage 2 Part E). phase NULL = berlaku untuk trafo secara keseluruhan (semua fasa), bukan satu fasa spesifik. resets_baseline=true memicu reset EqAge (server/aging_job.py) dan menandai model baseline beku (ml_models.is_frozen_reference=true) untuk trafo/fasa ini sebagai perlu dikalibrasi ulang (lihat ml_models.recalibration_note, migrasi 012).';
COMMENT ON COLUMN maintenance_events.trafo IS
  'TANPA batasan 2/3/4 (beda dari tabel lain) -- mencakup trafo lama/sudah tidak dipantau (mis. Trafo 1) untuk keperluan histori.';
COMMENT ON COLUMN maintenance_events.resistance_uohm IS
  'Resistansi kontak SATU fasa, mikro-ohm (µΩ) -- NULL untuk event_type selain resistance_test.';

CREATE INDEX IF NOT EXISTS idx_maintenance_events_trafo_ts ON maintenance_events (trafo, event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_maintenance_events_resets_baseline
    ON maintenance_events (trafo, phase, event_ts DESC) WHERE resets_baseline = TRUE;

COMMIT;
