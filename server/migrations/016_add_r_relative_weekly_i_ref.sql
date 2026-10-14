-- 016_add_r_relative_weekly_i_ref.sql
--
-- SHIELD Stage 2, Part F — menambah kolom i_ref ke r_relative_weekly
-- (Part D). server/forecast.py compute_excess() butuh A_ref DAN i_ref
-- yang SEPASANG (basis yang sama persis A_ref itu berasal) untuk
-- menghitung (I_forecast/I_ref)^2 -- tanpa kolom ini, server/forecast_job.py
-- harus menelusuri ulang model physics_rc "pinjaman" yang resistance_job.py
-- pakai (kalau referensi adalah ml_gbm), yang TIDAK tercatat di mana pun
-- selain di dalam resistance_job.py sendiri saat run itu. Menyimpan i_ref
-- langsung di sini jauh lebih sederhana dan tidak rawan error.
--
-- Jalankan manual (idempotent):
--   psql "$DATABASE_URL" -f server/migrations/016_add_r_relative_weekly_i_ref.sql
--
-- Rollback: server/migrations/016_add_r_relative_weekly_i_ref_down.sql

BEGIN;

ALTER TABLE r_relative_weekly ADD COLUMN IF NOT EXISTS i_ref DOUBLE PRECISION NULL;

COMMENT ON COLUMN r_relative_weekly.i_ref IS
  'I_ref (Ampere) berpasangan dengan a_ref -- basis arus yang sama persis dipakai saat A_ref dihitung/dipinjam (lihat server/resistance_job.py). NULL kalau a_ref juga NULL (status != ok).';

COMMIT;
