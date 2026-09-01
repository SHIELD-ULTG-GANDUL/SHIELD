-- SHIELD — skema TimescaleDB untuk log historis suhu.
-- Jalankan sekali di server Ubuntu setelah TimescaleDB terpasang:
--   sudo -u postgres psql -d shield -f schema.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS temperature_logs (
    id            BIGSERIAL,
    trafo         SMALLINT         NOT NULL,           -- 2, 3, atau 4
    r_atas        DOUBLE PRECISION NOT NULL DEFAULT 0,
    r_bawah       DOUBLE PRECISION NOT NULL DEFAULT 0,
    s_atas        DOUBLE PRECISION NOT NULL DEFAULT 0,
    s_bawah       DOUBLE PRECISION NOT NULL DEFAULT 0,
    t_atas        DOUBLE PRECISION NOT NULL DEFAULT 0,
    t_bawah       DOUBLE PRECISION NOT NULL DEFAULT 0,
    device_ts     TEXT,                                -- timestamp asli dari payload MQTT device (kalau ada)
    source        TEXT             NOT NULL DEFAULT 'mqtt_ingester',
    ts            TIMESTAMPTZ      NOT NULL DEFAULT now(),  -- kolom waktu hypertable
    PRIMARY KEY (trafo, ts, id)
);

-- Jadikan hypertable TimescaleDB, dipartisi otomatis per 1 hari berdasarkan kolom ts.
SELECT create_hypertable('temperature_logs', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index untuk query "N data terbaru per trafo" (pola akses utama dashboard).
CREATE INDEX IF NOT EXISTS idx_temperature_logs_trafo_ts
    ON temperature_logs (trafo, ts DESC);

-- ── Retensi otomatis (opsional tapi disarankan) ─────────────────────────────
-- Hapus otomatis data yang lebih tua dari 180 hari supaya storage tidak
-- membengkak tanpa batas. Sesuaikan interval sesuai kebutuhan retensi Anda.
SELECT add_retention_policy('temperature_logs', INTERVAL '180 days', if_not_exists => TRUE);

-- ── (Opsional) Continuous aggregate — rata-rata per jam, untuk chart histori
-- jangka panjang tanpa perlu scan jutaan baris mentah. Timeline LNTAI Score
-- di dashboard bisa memakai ini kalau datanya sudah sangat banyak.
CREATE MATERIALIZED VIEW IF NOT EXISTS temperature_logs_hourly
WITH (timescaledb.continuous) AS
SELECT
    trafo,
    time_bucket('1 hour', ts) AS bucket,
    avg(r_atas) AS r_atas_avg, avg(r_bawah) AS r_bawah_avg,
    avg(s_atas) AS s_atas_avg, avg(s_bawah) AS s_bawah_avg,
    avg(t_atas) AS t_atas_avg, avg(t_bawah) AS t_bawah_avg,
    max(greatest(r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah)) AS max_temp,
    count(*) AS sample_count
FROM temperature_logs
GROUP BY trafo, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('temperature_logs_hourly',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);
