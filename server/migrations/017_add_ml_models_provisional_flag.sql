-- 017_add_ml_models_provisional_flag.sql
--
-- SHIELD Stage 2, go-live (provisional operation) — lets a baseline model
-- be trained and activated from a healthy period chosen via
-- ml/suggest_healthy_period.py (a RANKED SUGGESTION, not a confirmed
-- measurement) before any contact-resistance test exists to confirm it.
-- Does NOT change training/activation logic itself -- purely a label
-- ml/train_baseline.py writes and GET /api/ml/models exposes, so the UI
-- can show "Provisional baseline" instead of silently presenting an
-- unconfirmed healthy period as if it were settled.
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/017_add_ml_models_provisional_flag.sql
--
-- Rollback: server/migrations/017_add_ml_models_provisional_flag_down.sql

BEGIN;

ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS is_provisional BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS provisional_reason TEXT NULL;

COMMENT ON COLUMN ml_models.is_provisional IS
  'TRUE kalau model ini dilatih dari healthy_from/healthy_to yang berasal dari ml/suggest_healthy_period.py (saran peringkat otomatis) dan BELUM dikonfirmasi oleh resistance test riil (Part E) pada periode itu. Default FALSE -- model yang healthy period-nya dipilih manual oleh operator berdasarkan pengetahuan riil trafo TETAP FALSE, is_provisional bukan default untuk semua model baru.';
COMMENT ON COLUMN ml_models.provisional_reason IS
  'Wajib diisi (oleh ml/train_baseline.py --provisional-reason) kalau is_provisional=TRUE -- kalimat yang ditampilkan di banner UI, mis. "Healthy period suggested by ml/suggest_healthy_period.py from completeness/load-coverage/residual-stability ranking, not yet confirmed by a contact-resistance test." NULL kalau is_provisional=FALSE.';

COMMIT;
