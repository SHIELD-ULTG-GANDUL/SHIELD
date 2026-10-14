-- 009_create_ml_models.sql
--
-- SHIELD Stage 2, Part B — registry model baseline "perilaku sehat" per
-- (trafo, phase). Satu baris per model yang dilatih (ml/train_baseline.py);
-- is_active/is_frozen_reference diubah lewat
-- POST /api/ml/models/{id}/activate dan /freeze-reference (server/api.py),
-- TIDAK PERNAH diset otomatis saat training (lihat ml/train_baseline.py --
-- "Never auto-activate").
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/009_create_ml_models.sql
--
-- Rollback: server/migrations/009_create_ml_models_down.sql

BEGIN;

CREATE TABLE IF NOT EXISTS ml_models (
    id                  BIGSERIAL PRIMARY KEY,
    trafo               TEXT NOT NULL,
    phase               TEXT NOT NULL CHECK (phase IN ('R', 'S', 'T')),
    model_type          TEXT NOT NULL CHECK (model_type IN ('physics_quadratic', 'physics_rc', 'ml_gbm', 'hybrid')),
    version             TEXT NOT NULL,
    trained_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    healthy_from        TIMESTAMPTZ NOT NULL,
    healthy_to          TIMESTAMPTZ NOT NULL,
    features            JSONB NOT NULL,
    metrics             JSONB NOT NULL,
    artifact_path       TEXT NOT NULL,
    sigma               DOUBLE PRECISION NULL,
    mu                  DOUBLE PRECISION NULL,
    is_active           BOOLEAN NOT NULL DEFAULT FALSE,
    is_frozen_reference BOOLEAN NOT NULL DEFAULT FALSE,
    notes               TEXT NULL,
    UNIQUE (trafo, phase, model_type, version)
);

COMMENT ON TABLE ml_models IS
  'Registry model baseline (Stage 2 Part B). is_active=true = dipakai lntai_worker.py sebagai sumber delta_T_pred ADAPTIF untuk (trafo,phase) ini (fallback ke fisika legacy kalau tidak ada model aktif). is_frozen_reference=true = dipakai Part D/H sebagai referensi TETAP (tidak berubah walau model adaptif baru diaktifkan) -- dua flag ini independen, satu model BOLEH keduanya sekaligus.';
COMMENT ON COLUMN ml_models.artifact_path IS
  'Path relatif ke file artefak (JSON, lihat ml/models.py .to_artifact()) di dalam ml/artifacts/, mis. "2_R_physics_rc_v1.json".';
COMMENT ON COLUMN ml_models.features IS
  'Metadata dataset dari ml/dataset.py (healthy_from/to, drop_flags, ambient_source, n_final, dst.) -- BUKAN data training itu sendiri.';
COMMENT ON COLUMN ml_models.metrics IS
  'Metrik validasi: mu, sigma (1.4826*MAD, robust), mae, r2, bias, n -- semua dalam satuan °C kecuali n dan r2.';

-- Hanya SATU model boleh is_active=true per (trafo, phase) -- partial unique
-- index (bukan constraint biasa) supaya banyak baris is_active=false bebas
-- berdampingan.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_models_one_active_per_trafo_phase
    ON ml_models (trafo, phase) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_ml_models_trafo_phase ON ml_models (trafo, phase);

COMMIT;
