-- 004_create_lntai_results.sql
--
-- SHIELD Tahap 1, T4 — tabel lntai_results: hasil LNTAI v2 dihitung di
-- SERVER (lntai_worker.py / backfill_lntai.py) dari data trafo_temperature
-- + trafo_load yang sudah dipasangkan, dipakai sebagai sumber utama panel
-- LNTAI di dashboard (browser hanya fallback -- lihat README.md).
--
-- PRIMARY KEY (trafo_id, ts, model_version) BUKAN sembarang pilihan: ini
-- membuat INSERT ... ON CONFLICT DO NOTHING (dipakai lntai_worker.py DAN
-- backfill_lntai.py) idempotent -- backfill ulang rentang yang sudah
-- pernah dihitung untuk model_version yang sama tidak menduplikasi atau
-- menimpa baris, dan mengganti model (model_version baru) tidak kolisi
-- dengan hasil model lama (keduanya tetap tersimpan berdampingan untuk
-- perbandingan).
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/004_create_lntai_results.sql
--
-- Rollback: server/migrations/004_create_lntai_results_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS lntai_results (
    trafo_id        TEXT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    model_version   TEXT NOT NULL,
    lntai           DOUBLE PRECISION NULL,
    level           TEXT NOT NULL,
    s_temp          DOUBLE PRECISION NULL,
    s_delta_t       DOUBLE PRECISION NULL,
    s_res           DOUBLE PRECISION NULL,
    hottest_phase   TEXT NULL,
    t_contact       DOUBLE PRECISION NULL,
    t_ambient       DOUBLE PRECISION NULL,
    ambient_source  TEXT NULL,
    delta_t_real    DOUBLE PRECISION NULL,
    delta_t_pred    DOUBLE PRECISION NULL,
    residual        DOUBLE PRECISION NULL,
    z               DOUBLE PRECISION NULL,
    i_eff_norm      DOUBLE PRECISION NULL,
    delta_phase     DOUBLE PRECISION NULL,
    dab_max         DOUBLE PRECISION NULL,
    per_phase       JSONB NULL,
    quality_flags   INTEGER NULL,
    paired          BOOLEAN NOT NULL DEFAULT TRUE,
    error           TEXT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trafo_id, ts, model_version)
);

COMMENT ON TABLE lntai_results IS
  'Hasil LNTAI v2 per trafo per timestamp, dihitung server-side (T4). level=DATA_INVALID + error terisi kalau input diblokir data_quality.should_skip_lntai() atau tidak ada baris beban terpasang (paired=false) dalam pairing_tolerance_minutes.';
COMMENT ON COLUMN lntai_results.quality_flags IS
  'Bitmask gabungan (OR) dari quality_flags baris suhu + baris beban yang dipasangkan, ditambah PAIRING_GAP kalau paired=false. Lihat server/data_quality_config.json.';
COMMENT ON COLUMN lntai_results.per_phase IS
  'Rincian per fasa (R/S/T): T, I, dT, dT_pred, z, dAB, i_eff_norm -- sama seperti per_phase di lntai_calculator.lntai_step().';

CREATE INDEX IF NOT EXISTS idx_lntai_results_trafo_ts ON lntai_results (trafo_id, ts DESC);

COMMIT;
