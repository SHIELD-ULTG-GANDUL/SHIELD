-- Rollback 009_create_ml_models.sql
--
-- PERINGATAN: menghapus seluruh registry model (bukan file artefak di
-- ml/artifacts/ -- itu tetap ada di disk, hanya referensinya di DB yang
-- hilang). Backup dulu kalau perlu:
--   pg_dump -t ml_models "$DATABASE_URL" > ml_models_backup.sql
--
--   psql "$DATABASE_URL" -f server/migrations/009_create_ml_models_down.sql

BEGIN;

DROP TABLE IF EXISTS ml_models;

COMMIT;
