"""
SHIELD — REST API baca-saja di atas TimescaleDB, dipanggil oleh Shield.html
(browser tidak bisa konek langsung ke PostgreSQL).

Jalankan manual untuk tes:
    pip install -r requirements.txt
    export DATABASE_URL=postgresql://shield:password@localhost:5432/shield
    export API_KEY=ganti-dengan-string-acak-panjang   # opsional tapi disarankan
    uvicorn api:app --host 0.0.0.0 --port 8000

Untuk produksi, jalankan sebagai systemd service (shield-api.service) di
belakang reverse proxy nginx dengan HTTPS — lihat README.md.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "ml"))

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import ambient
import aging
import resistance
import degradation as dg
import forecast as fc
import explain as xp
import data_quality as dq
import lntai_vs_aging as lva
from settings import load_ambient_location_settings, load_api_settings, load_database_settings

import explain_contrib  # noqa: E402 -- ml/explain_contrib.py (kontribusi per-fitur, Part G)
from forecast_job import fetch_active_baseline_model  # noqa: E402
from forecast_models import hour_sin_cos_weekday  # noqa: E402
from models import load_model_from_artifact  # noqa: E402

_db_settings = load_database_settings()
_api_settings = load_api_settings()
_ambient_location = load_ambient_location_settings()

# Opsional: kalau API_KEY diisi (lewat settings.py/env var), semua request wajib
# menyertakan header X-API-Key yang cocok. Kosongkan kalau API ini hanya dipakai
# secara internal/read-only publik -- perilaku ini dipertahankan dari versi lama,
# bukan diubah oleh migrasi ke settings.py.
API_KEY = _api_settings.api_key

app = FastAPI(title="SHIELD Temperature Log API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_settings.allowed_origins,
    # T3: POST ditambah untuk /api/ambient-logs (browser mengirim laporan
    # ambient langsung) -- endpoint lain tetap GET-only seperti sebelumnya.
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

_pool = ConnectionPool(conninfo=_db_settings.url, min_size=1, max_size=5, open=True)


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key tidak valid")


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/temperature-logs/bounds")
def temperature_logs_bounds(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Rentang waktu data suhu yang tersedia untuk satu trafo (MIN/MAX
    timestamp) — query murah (pakai index yang sama dengan endpoint utama),
    dipakai frontend untuk membagi unduhan rentang penuh menjadi beberapa
    bagian waktu yang bisa diambil paralel, bukan menebak dari mana histori
    dimulai lewat trial-and-error."""
    _check_api_key(x_api_key)
    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            'SELECT min("timestamp") AS min_ts, max("timestamp") AS max_ts '
            "FROM trafo_temperature WHERE trafo_id = %s",
            (str(trafo),),
        )
        row = cur.fetchone()
    return {
        "trafo": trafo,
        "min": row["min_ts"].isoformat() if row and row["min_ts"] else None,
        "max": row["max_ts"].isoformat() if row and row["max_ts"] else None,
    }


@app.get("/api/temperature-logs")
def temperature_logs(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    limit: int = Query(300, ge=1, le=20000),
    from_: Optional[datetime] = Query(default=None, alias="from", description="Batas awal ISO8601, opsional"),
    to_: Optional[datetime] = Query(default=None, alias="to", description="Batas akhir ISO8601, opsional — dipakai untuk paginasi mundur saat mengunduh rentang penuh"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Mengembalikan N pembacaan suhu terbaru untuk satu trafo, terurut dari
    yang paling baru. Bentuk field sengaja disamakan dengan skema lama di
    Firestore supaya sisi frontend (Shield.html) tinggal memetakan langsung.

    T2/T3 (Tahap 1, Oktober 2026): field "ambient" yang dulu diambil dari
    kolom temp_ambient DIHAPUS dari respons ini -- audit Fase 0 mengonfirmasi
    kolom itu selalu bernilai 30 (default statis, bukan pengukuran) untuk
    SEMUA baris yang ada. Suhu ambient yang sebenarnya akan datang dari
    ambient_logs (T3), lewat endpoint /api/ambient-logs terpisah, bukan
    menumpang di endpoint ini. Kolom temp_ambient sendiri TETAP ADA di tabel,
    dicadangkan untuk sensor ambient di dalam panel trafo di masa depan
    (lihat docs/DATA_DICTIONARY.md, T9)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT trafo_id AS trafo, r_atas AS "R_Atas", r_bawah AS "R_Bawah",
                   s_atas AS "S_Atas", s_bawah AS "S_Bawah",
                   t_atas AS "T_Atas", t_bawah AS "T_Bawah",
                   quality_flags,
                   timestamp AS ts
            FROM trafo_temperature
            WHERE trafo_id = %s
              AND (%s::timestamptz IS NULL OR "timestamp" >= %s)
              AND (%s::timestamptz IS NULL OR "timestamp" <= %s)
            ORDER BY "timestamp" DESC
            LIMIT %s
            """,
            (str(trafo), from_, from_, to_, to_, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        row["timestamp"] = row["ts"].isoformat()
        del row["ts"]

    return {"trafo": trafo, "count": len(rows), "rows": rows}


# =====================================================
# AMBIENT LOGS (T3, Cabang B) — ditulis oleh browser, bukan worker server
# =====================================================

class AmbientLogIn(BaseModel):
    """Body POST /api/ambient-logs. Dikirim oleh browser (Shield.html) yang
    SUDAH berhasil fetch Open-Meteo langsung dari sisi klien -- lihat
    catatan Cabang B di migrations/002_create_ambient_logs.sql."""

    timestamp: datetime = Field(..., description="Waktu respons Open-Meteo diterima browser (ISO8601)")
    temperature_c: Optional[float] = Field(None, description="Suhu ambient (°C); null kalau ok=false")
    ok: bool = Field(..., description="false kalau browser gagal fetch Open-Meteo untuk slot ini")
    latitude: Optional[float] = Field(None, description="Default: lokasi GI Gandul dari server jika tidak dikirim")
    longitude: Optional[float] = Field(None, description="Default: lokasi GI Gandul dari server jika tidak dikirim")


@app.post("/api/ambient-logs")
def create_ambient_log(
    body: AmbientLogIn,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Terima satu laporan suhu ambient dari browser, dedup per slot 10
    menit (ambient_logs_slot_unique, migrasi 002). Kalau browser mengirim
    dua kali untuk slot yang sama (mis. dua tab terbuka), baris pertama yang
    menang -- ON CONFLICT DO NOTHING, bukan overwrite, supaya satu klien
    yang telat/gagal tidak menimpa laporan ok=true yang sudah tersimpan.

    ok=false WAJIB diterima dan disimpan apa adanya (temperature_c=NULL) --
    endpoint ini TIDAK PERNAH menolak atau mengarang nilai untuk kegagalan
    fetch; itu adalah sinyal kualitas data yang sah untuk ambient_at() (T4)."""
    _check_api_key(x_api_key)

    if body.ok and not ambient.is_valid_ambient_temperature(body.temperature_c):
        raise HTTPException(
            status_code=400,
            detail=f"temperature_c di luar rentang valid ({ambient.AMBIENT_MIN_C}..{ambient.AMBIENT_MAX_C}) atau hilang padahal ok=true",
        )

    slot = ambient.floor_to_slot(body.timestamp)
    lat = body.latitude if body.latitude is not None else _ambient_location.latitude
    lon = body.longitude if body.longitude is not None else _ambient_location.longitude

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO ambient_logs (timestamp, slot_10min, temperature_c, ok, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (slot_10min) DO NOTHING
            RETURNING id
            """,
            (body.timestamp, slot, body.temperature_c if body.ok else None, body.ok, lat, lon),
        )
        row = cur.fetchone()

    return {"stored": row is not None, "slot": slot.isoformat()}


@app.get("/api/ambient-logs")
def ambient_logs(
    limit: int = Query(300, ge=1, le=20000),
    from_: Optional[datetime] = Query(default=None, alias="from"),
    to_: Optional[datetime] = Query(default=None, alias="to"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Histori suhu ambient GI Gandul -- lihat catatan jeda data (Cabang B)
    di docstring modul ambient.py / server/README.md: hanya terisi selama
    ada dashboard terbuka di browser."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT "timestamp", temperature_c, ok, source, latitude, longitude
            FROM ambient_logs
            WHERE (%s::timestamptz IS NULL OR "timestamp" >= %s)
              AND (%s::timestamptz IS NULL OR "timestamp" <= %s)
            ORDER BY "timestamp" DESC
            LIMIT %s
            """,
            (from_, from_, to_, to_, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        row["timestamp"] = row["timestamp"].isoformat()

    return {"count": len(rows), "rows": rows}


# =====================================================
# LNTAI LOGS (T4) — hasil dihitung server-side oleh lntai_worker.py
# =====================================================

@app.get("/api/lntai-logs")
def lntai_logs(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    limit: int = Query(300, ge=1, le=20000),
    from_: Optional[datetime] = Query(default=None, alias="from"),
    to_: Optional[datetime] = Query(default=None, alias="to"),
    model_version: Optional[str] = Query(
        default=None, description="Filter opsional, mis. 'lntai-v3' -- kosong = semua versi (perilaku lama, dipertahankan untuk caller yang sudah ada)."
    ),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Histori LNTAI v2 dihitung server (T4) untuk satu trafo. Frontend
    memakai ini sebagai sumber UTAMA panel LNTAI; kalkulasi di browser
    (_lntaiStep di index.html) jadi fallback saat endpoint ini tidak
    terjangkau -- lihat badge "source" di panel LNTAI.

    `model_version` (Stage 2 Part J2): worker menulis SATU baris v2 DAN
    SATU baris v3 per tick dengan `ts` yang sama -- tanpa filter ini,
    rentang waktu campur v2/v3 tidak terurai per model. Opsional dan
    default None supaya caller lama (panel LNTAI v2) tidak berubah."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ts, model_version, lntai, level, s_temp, s_delta_t, s_res,
                   hottest_phase, t_contact, t_ambient, ambient_source,
                   delta_t_real, delta_t_pred, residual, z, i_eff_norm,
                   delta_phase, dab_max, per_phase, quality_flags, paired, error
            FROM lntai_results
            WHERE trafo_id = %s
              AND (%s::timestamptz IS NULL OR ts >= %s)
              AND (%s::timestamptz IS NULL OR ts <= %s)
              AND (%s::text IS NULL OR model_version = %s)
            ORDER BY ts DESC
            LIMIT %s
            """,
            (str(trafo), from_, from_, to_, to_, model_version, model_version, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        row["timestamp"] = row.pop("ts").isoformat()

    return {"trafo": trafo, "count": len(rows), "rows": rows}


@app.get("/api/lntai-latest")
def lntai_latest(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    model_version: Optional[str] = Query(
        default=None, description="Filter opsional, mis. 'lntai-v3' -- kosong = tidak difilter (perilaku lama, dipertahankan untuk caller yang sudah ada)."
    ),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Hasil LNTAI TERBARU untuk satu trafo -- dipakai tick realtime
    panel LNTAI (polling ringan, tidak perlu mengambil histori penuh).

    `model_version` (Stage 2 Part J2): tanpa filter ini, baris v2 dan v3
    yang ditulis pada `ts` yang sama tidak punya urutan tie-break yang
    dijamin -- caller yang butuh SPESIFIK v3 (mis. panel J1/J2) HARUS
    mengisi parameter ini. Opsional dan default None supaya caller lama
    (panel LNTAI v2) tidak berubah perilakunya."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT ts, model_version, lntai, level, s_temp, s_delta_t, s_res,
                   hottest_phase, t_contact, t_ambient, ambient_source,
                   delta_t_real, delta_t_pred, residual, z, i_eff_norm,
                   delta_phase, dab_max, per_phase, quality_flags, paired, error,
                   baseline_id, calibrated, lntai_rolling_median_30min
            FROM lntai_results
            WHERE trafo_id = %s
              AND (%s::text IS NULL OR model_version = %s)
            ORDER BY ts DESC
            LIMIT 1
            """,
            (str(trafo), model_version, model_version),
        )
        row = cur.fetchone()

    if row is None:
        return {"trafo": trafo, "row": None}

    row["timestamp"] = row.pop("ts").isoformat()
    return {"trafo": trafo, "row": row}


# =====================================================
# DATA QUALITY SUMMARY (T5) — ringkasan bitmask quality_flags
# =====================================================

def _count_flags(rows: list[dict], key: str = "quality_flags") -> dict[str, int]:
    counts = {name: 0 for name in dq.FLAG_BITS}
    for row in rows:
        for name in dq.bitmask_to_flags(row.get(key)):
            counts[name] += 1
    return counts


@app.get("/api/data-quality/summary")
def data_quality_summary(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    window: int = Query(1440, ge=1, le=43200, description="Jendela waktu dalam menit (default 1440 = 24 jam)"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Ringkasan jumlah baris per flag data_quality (T5) dalam `window` menit
    terakhir, untuk kartu "Kualitas Data" di dashboard. Dihitung dari kolom
    quality_flags trafo_temperature/trafo_load -- TIDAK menghitung ulang
    aturan data_quality.py di sini, hanya membaca bitmask yang sudah ditulis
    mqtt_subscriber.py (T5) saat insert.

    `completeness_pct`/`longest_gap_minutes`/`ambient` (Stage 2 Part J3):
    tambahan ADITIF -- completeness_pct = baris yang benar-benar ada /
    baris yang SEHARUSNYA ada kalau sampling persis setiap
    nominal_sample_interval_seconds (data_quality_config.json) tanpa jeda
    sama sekali; longest_gap_minutes dihitung dari selisih timestamp
    berurutan (baris yang HILANG SAMA SEKALI, bukan dari bit TIME_GAP di
    quality_flags -- bit itu menandai satu baris yang terlambat, bukan
    ukuran jeda itu sendiri)."""
    _check_api_key(x_api_key)
    since = datetime.now(timezone.utc) - timedelta(minutes=window)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            'SELECT "timestamp", quality_flags FROM trafo_temperature WHERE trafo_id = %s AND "timestamp" >= %s ORDER BY "timestamp" ASC',
            (str(trafo), since),
        )
        temp_rows = cur.fetchall()
        cur.execute(
            'SELECT quality_flags FROM trafo_load WHERE trafo_id = %s AND "timestamp" >= %s',
            (str(trafo), since),
        )
        load_rows = cur.fetchall()
        cur.execute(
            'SELECT "timestamp", ok, source FROM ambient_logs WHERE "timestamp" >= %s ORDER BY "timestamp" ASC',
            (since,),
        )
        ambient_rows = cur.fetchall()

    nominal_interval_s = dq.DATA_QUALITY_CONFIG["nominal_sample_interval_seconds"]
    expected_rows = max(1, int((window * 60) / nominal_interval_s))
    completeness_pct = min(100.0, round(len(temp_rows) / expected_rows * 100.0, 1))

    longest_gap_minutes = None
    if len(temp_rows) >= 2:
        gaps = [
            (temp_rows[i]["timestamp"] - temp_rows[i - 1]["timestamp"]).total_seconds() / 60.0
            for i in range(1, len(temp_rows))
        ]
        longest_gap_minutes = round(max(gaps), 1)

    ambient_ok_rows = [r for r in ambient_rows if r["ok"]]
    ambient_longest_gap_minutes = None
    if len(ambient_ok_rows) >= 2:
        ambient_gaps = [
            (ambient_ok_rows[i]["timestamp"] - ambient_ok_rows[i - 1]["timestamp"]).total_seconds() / 60.0
            for i in range(1, len(ambient_ok_rows))
        ]
        ambient_longest_gap_minutes = round(max(ambient_gaps), 1)
    latest_ambient_source = ambient_ok_rows[-1]["source"] if ambient_ok_rows else None

    return {
        "trafo": trafo,
        "window_minutes": window,
        "temperature": {"total_rows": len(temp_rows), "flag_counts": _count_flags(temp_rows)},
        "load": {"total_rows": len(load_rows), "flag_counts": _count_flags(load_rows)},
        "completeness_pct": completeness_pct,
        "longest_gap_minutes": longest_gap_minutes,
        "ambient": {
            "total_reports": len(ambient_rows),
            "ok_reports": len(ambient_ok_rows),
            "latest_source": latest_ambient_source,
            "longest_gap_minutes": ambient_longest_gap_minutes,
        },
    }


# =====================================================
# MAINTENANCE WINDOWS (T6) — ditandai manual, dipakai archive_job.py
# =====================================================

class MaintenanceWindowIn(BaseModel):
    trafo_id: Optional[str] = Field(None, description="NULL/tidak diisi = berlaku semua trafo")
    start_time: datetime
    end_time: datetime
    reason: str
    created_by: Optional[str] = None


@app.get("/api/maintenance-windows")
def maintenance_windows(
    trafo: Optional[int] = Query(default=None, ge=2, le=4),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Daftar maintenance window (T6). Tanpa `trafo`, kembalikan semua
    window (termasuk yang trafo_id=NULL, berlaku untuk semua trafo)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if trafo is None:
            cur.execute(
                "SELECT id, trafo_id, start_time, end_time, reason, created_by, archived, created_at "
                "FROM maintenance_windows ORDER BY start_time DESC"
            )
        else:
            cur.execute(
                "SELECT id, trafo_id, start_time, end_time, reason, created_by, archived, created_at "
                "FROM maintenance_windows WHERE trafo_id IS NULL OR trafo_id = %s ORDER BY start_time DESC",
                (str(trafo),),
            )
        rows = cur.fetchall()

    for row in rows:
        row["start_time"] = row["start_time"].isoformat()
        row["end_time"] = row["end_time"].isoformat()
        row["created_at"] = row["created_at"].isoformat()

    return {"count": len(rows), "rows": rows}


@app.post("/api/maintenance-windows")
def create_maintenance_window(
    body: MaintenanceWindowIn,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Tandai satu jendela waktu sebagai maintenance/kalibrasi/gangguan
    (T6) -- menulis ke database dan mempengaruhi archive_job.py, jadi
    tetap melalui _check_api_key seperti endpoint lain (ditolak kalau
    API_KEY diset dan header X-API-Key tidak cocok)."""
    _check_api_key(x_api_key)
    if body.end_time <= body.start_time:
        raise HTTPException(status_code=400, detail="end_time harus setelah start_time")

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO maintenance_windows (trafo_id, start_time, end_time, reason, created_by)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (body.trafo_id, body.start_time, body.end_time, body.reason, body.created_by),
        )
        row = cur.fetchone()

    return {"id": row["id"]}


@app.get("/api/beban-logs/bounds")
def beban_logs_bounds(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Rentang waktu data beban yang tersedia untuk satu trafo (MIN/MAX
    timestamp) — lihat catatan di temperature_logs_bounds()."""
    _check_api_key(x_api_key)
    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            'SELECT min("timestamp") AS min_ts, max("timestamp") AS max_ts '
            "FROM trafo_load WHERE trafo_id = %s",
            (str(trafo),),
        )
        row = cur.fetchone()
    return {
        "trafo": trafo,
        "min": row["min_ts"].isoformat() if row and row["min_ts"] else None,
        "max": row["max_ts"].isoformat() if row and row["max_ts"] else None,
    }


@app.get("/api/beban-logs")
def beban_logs(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    limit: int = Query(300, ge=1, le=20000),
    from_: Optional[datetime] = Query(default=None, alias="from", description="Batas awal ISO8601, opsional"),
    to_: Optional[datetime] = Query(default=None, alias="to", description="Batas akhir ISO8601, opsional — dipakai untuk paginasi mundur saat mengunduh rentang penuh"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Mengembalikan N pembacaan beban (arus/tegangan/daya) terbaru untuk satu
    trafo, dari tabel trafo_load (shield_data). Nama field disamakan dengan
    alias yang dikenali parseBebanDoc() di Shield.html (I_R, kW, dst.) supaya
    frontend tidak perlu mapping ulang. V_R/V_S/V_T adalah tegangan fasa-netral
    (van/vbn/vcn); V_RS/V_ST/V_TR adalah tegangan fasa-fasa (vab/vbc/vca).

    T2 (Tahap 1, Oktober 2026): COALESCE(..., 0) pada kolom tegangan/kVAR
    DIHAPUS -- sebelumnya API ini sendiri yang mengubah NULL (data hilang)
    jadi 0 sebelum sampai ke frontend, padahal mqtt_subscriber.py (T2) sudah
    diperbaiki supaya tidak lagi menulis 0 untuk data yang hilang. Frontend
    (parseBebanDoc, dll.) sekarang bertanggung jawab memperlakukan null
    sebagai "data tidak tersedia", bukan API yang menyamarkannya jadi 0."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT trafo_id AS trafo, ia AS "I_R", ib AS "I_S", ic AS "I_T",
                   van AS "V_R", vbn AS "V_S", vcn AS "V_T",
                   vab AS "V_RS", vbc AS "V_ST", vca AS "V_TR",
                   kw_total AS "kW", kvar_total AS "kVAR", kva_total AS "kVA", pf_total AS "PF",
                   current_max, freq, quality_flags,
                   timestamp AS ts
            FROM trafo_load
            WHERE trafo_id = %s
              AND (%s::timestamptz IS NULL OR "timestamp" >= %s)
              AND (%s::timestamptz IS NULL OR "timestamp" <= %s)
            ORDER BY "timestamp" DESC
            LIMIT %s
            """,
            (str(trafo), from_, from_, to_, to_, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        row["timestamp"] = row["ts"].isoformat()
        del row["ts"]

    return {"trafo": trafo, "count": len(rows), "rows": rows}


# =====================================================
# ML MODELS (Stage 2, Part B) — registry baseline "perilaku sehat"
# =====================================================

@app.get("/api/ml/models")
def ml_models_list(
    trafo: Optional[int] = Query(default=None, ge=2, le=4),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Daftar model baseline (ml/train_baseline.py) -- tanpa `trafo`,
    kembalikan semua. Dipakai UI untuk menampilkan tabel perbandingan dan
    tombol activate/freeze-reference."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if trafo is None:
            cur.execute(
                "SELECT id, trafo, phase, model_type, version, trained_at, healthy_from, healthy_to, "
                "features, metrics, artifact_path, sigma, mu, is_active, is_frozen_reference, notes "
                "FROM ml_models ORDER BY trafo, phase, model_type, trained_at DESC"
            )
        else:
            cur.execute(
                "SELECT id, trafo, phase, model_type, version, trained_at, healthy_from, healthy_to, "
                "features, metrics, artifact_path, sigma, mu, is_active, is_frozen_reference, notes "
                "FROM ml_models WHERE trafo = %s ORDER BY phase, model_type, trained_at DESC",
                (str(trafo),),
            )
        rows = cur.fetchall()

    for row in rows:
        row["trained_at"] = row["trained_at"].isoformat()
        row["healthy_from"] = row["healthy_from"].isoformat()
        row["healthy_to"] = row["healthy_to"].isoformat()

    return {"count": len(rows), "rows": rows}


@app.post("/api/ml/models/{model_id}/activate")
def ml_model_activate(
    model_id: int,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Jadikan model ini AKTIF untuk (trafo, phase)-nya -- lntai_worker.py
    memakainya sebagai sumber delta_T_pred v3 dalam <= model_registry.
    ACTIVE_MODEL_CACHE_TTL_SECONDS detik ke depan (worker proses TERPISAH,
    tidak bisa diberi tahu instan, lihat ml/model_registry.py). Model lain
    untuk (trafo, phase) yang SAMA otomatis di-nonaktifkan (hanya satu
    is_active=true per trafo+phase, lihat migrasi 009 partial unique index)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT trafo, phase FROM ml_models WHERE id = %s", (model_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"ml_models id={model_id} tidak ditemukan")

        cur.execute(
            "UPDATE ml_models SET is_active = FALSE WHERE trafo = %s AND phase = %s AND id != %s",
            (row["trafo"], row["phase"], model_id),
        )
        cur.execute("UPDATE ml_models SET is_active = TRUE WHERE id = %s", (model_id,))

    return {"id": model_id, "trafo": row["trafo"], "phase": row["phase"], "is_active": True}


@app.post("/api/ml/models/{model_id}/freeze-reference")
def ml_model_freeze_reference(
    model_id: int,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Tandai model ini sebagai REFERENSI TETAP (dipakai Part D/H nanti) --
    independen dari is_active (satu model BOLEH keduanya sekaligus, dan
    BERBEDA dari model yang sedang is_active=true -- referensi tetap
    sengaja TIDAK ikut berganti walau model adaptif baru diaktifkan)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, trafo, phase FROM ml_models WHERE id = %s", (model_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"ml_models id={model_id} tidak ditemukan")

        cur.execute("UPDATE ml_models SET is_frozen_reference = TRUE WHERE id = %s", (model_id,))

    return {"id": model_id, "trafo": row["trafo"], "phase": row["phase"], "is_frozen_reference": True}


# =====================================================
# AGING (Stage 2, Part C) — faktor percepatan penuaan Arrhenius (relatif)
# =====================================================

@app.get("/api/aging/latest")
def aging_latest(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Baris aging_daily TERBARU untuk tiap fasa (R/S/T) trafo ini.
    status='not_configured' (tanpa angka apa pun) kalau aging_config.json
    belum punya ea_ev/t_ref_c -- lihat server/aging.py is_configured()."""
    _check_api_key(x_api_key)

    if not aging.is_configured():
        return {"trafo": trafo, "status": "not_configured", "scope": aging.AGING_SCOPE_TEXT, "rows": []}

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (phase) phase, date, f_mean, f_peak, t_hot_mean_c, t_hot_peak_c,
                   eq_days_cum, calendar_days, reset_date, reset_source, n_samples, n_gaps_excluded
            FROM aging_daily
            WHERE trafo = %s
            ORDER BY phase, date DESC
            """,
            (str(trafo),),
        )
        rows = cur.fetchall()

    for row in rows:
        row["date"] = row["date"].isoformat()
        row["reset_date"] = row["reset_date"].isoformat()

    return {"trafo": trafo, "status": "ok", "scope": aging.AGING_SCOPE_TEXT, "rows": rows}


@app.get("/api/aging/daily")
def aging_daily_endpoint(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    from_: Optional[datetime] = Query(default=None, alias="from"),
    to_: Optional[datetime] = Query(default=None, alias="to"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Histori harian (semua fasa) untuk satu trafo dalam rentang [from, to]
    (tanggal, bukan timestamp -- dibandingkan sebagai DATE)."""
    _check_api_key(x_api_key)

    if not aging.is_configured():
        return {"trafo": trafo, "status": "not_configured", "scope": aging.AGING_SCOPE_TEXT, "rows": []}

    from_date = from_.date() if from_ else None
    to_date = to_.date() if to_ else None

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT phase, date, f_mean, f_peak, t_hot_mean_c, t_hot_peak_c,
                   eq_days_cum, calendar_days, reset_date, reset_source, n_samples, n_gaps_excluded
            FROM aging_daily
            WHERE trafo = %s
              AND (%s::date IS NULL OR date >= %s)
              AND (%s::date IS NULL OR date <= %s)
            ORDER BY phase, date ASC
            """,
            (str(trafo), from_date, from_date, to_date, to_date),
        )
        rows = cur.fetchall()

    for row in rows:
        row["date"] = row["date"].isoformat()
        row["reset_date"] = row["reset_date"].isoformat()

    return {"trafo": trafo, "status": "ok", "scope": aging.AGING_SCOPE_TEXT, "count": len(rows), "rows": rows}


# =====================================================
# MAINTENANCE EVENTS (Stage 2, Part E) — pemeliharaan & resistance-test
# =====================================================

EVENT_TYPES: tuple[str, ...] = (
    "inspection", "cleaning", "tightening", "replacement", "resistance_test", "thermovision",
)


class MaintenanceEventIn(BaseModel):
    trafo: str = Field(..., description="Not restricted to 2/3/4 — historical records may reference retired units")
    phase: Optional[Literal["R", "S", "T"]] = Field(None, description="Null = applies to the whole transformer, all phases")
    event_type: Literal[
        "inspection", "cleaning", "tightening", "replacement", "resistance_test", "thermovision",
    ]
    event_ts: datetime
    resistance_uohm: Optional[float] = Field(None, description="Contact resistance, microohms (µΩ) — required for resistance_test")
    test_current_a: Optional[float] = Field(None, description="Test current, amperes")
    notes: Optional[str] = None
    resets_baseline: bool = False
    created_by: Optional[str] = None


@app.get("/api/maintenance-events")
def maintenance_events_list(
    trafo: Optional[str] = Query(default=None, description="Not restricted to 2/3/4"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        if trafo is None:
            cur.execute(
                "SELECT id, trafo, phase, event_type, event_ts, resistance_uohm, test_current_a, "
                "notes, resets_baseline, created_by, created_at FROM maintenance_events "
                "ORDER BY trafo, event_ts DESC"
            )
        else:
            cur.execute(
                "SELECT id, trafo, phase, event_type, event_ts, resistance_uohm, test_current_a, "
                "notes, resets_baseline, created_by, created_at FROM maintenance_events "
                "WHERE trafo = %s ORDER BY event_ts DESC",
                (trafo,),
            )
        rows = cur.fetchall()

    for row in rows:
        row["event_ts"] = row["event_ts"].isoformat()
        row["created_at"] = row["created_at"].isoformat()

    return {"count": len(rows), "rows": rows}


@app.post("/api/maintenance-events")
def maintenance_events_create(
    body: MaintenanceEventIn,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Create one maintenance/resistance-test log entry. When
    resets_baseline=true: aging.py's cumulative EqAge for this (trafo,
    phase) restarts from event_ts (lihat aging_job.get_reset_date(), yang
    membaca tabel ini -- menggantikan fallback_reset_date di
    aging_config.json, lihat Part E tugas #5), and any frozen-reference
    baseline models (ml_models.is_frozen_reference=true) for this
    (trafo, phase) are flagged as needing re-calibration."""
    _check_api_key(x_api_key)

    if body.event_type == "resistance_test" and body.resistance_uohm is None:
        raise HTTPException(
            status_code=400,
            detail="resistance_uohm is required when event_type is 'resistance_test'",
        )
    if body.resistance_uohm is not None and body.resistance_uohm <= 0:
        raise HTTPException(status_code=400, detail="resistance_uohm must be a positive number")
    if body.test_current_a is not None and body.test_current_a <= 0:
        raise HTTPException(status_code=400, detail="test_current_a must be a positive number")

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO maintenance_events (
                trafo, phase, event_type, event_ts, resistance_uohm, test_current_a,
                notes, resets_baseline, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                body.trafo, body.phase, body.event_type, body.event_ts, body.resistance_uohm,
                body.test_current_a, body.notes, body.resets_baseline, body.created_by,
            ),
        )
        new_id = cur.fetchone()["id"]

        flagged_model_ids: list[int] = []
        if body.resets_baseline:
            note = f"Needs re-calibration after maintenance on {body.event_ts.date().isoformat()}."
            if body.phase is None:
                cur.execute(
                    "UPDATE ml_models SET recalibration_note = %s "
                    "WHERE trafo = %s AND is_frozen_reference = TRUE "
                    "RETURNING id",
                    (note, body.trafo),
                )
            else:
                cur.execute(
                    "UPDATE ml_models SET recalibration_note = %s "
                    "WHERE trafo = %s AND phase = %s AND is_frozen_reference = TRUE "
                    "RETURNING id",
                    (note, body.trafo, body.phase),
                )
            flagged_model_ids = [r["id"] for r in cur.fetchall()]

    return {"id": new_id, "resets_baseline": body.resets_baseline, "flagged_model_ids": flagged_model_ids}


# =====================================================
# RESISTANCE (Stage 2, Part D, beta) — relative contact-resistance drift
# =====================================================

@app.get("/api/resistance/relative")
def resistance_relative(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    from_: Optional[datetime] = Query(default=None, alias="from"),
    to_: Optional[datetime] = Query(default=None, alias="to"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Weekly relative contact-resistance drift (beta) per phase, from
    server/resistance_job.py. Rows with status != 'ok' carry no ratio
    values (see `reason`) -- no frozen reference, reference needs
    re-calibration, or insufficient valid (steady, adequately-loaded)
    samples that week. `guidance` is static metadata (KEMA criteria),
    not computed per row -- see resistance.GUIDANCE_CITATION."""
    _check_api_key(x_api_key)

    from_date = from_.date() if from_ else None
    to_date = to_.date() if to_ else None

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT phase, week_start, r_ratio_median_pct, r_ratio_q1_pct, r_ratio_q3_pct,
                   r_ratio_iqr_pct, phase_relative_ratio_pct, n_samples_valid, n_samples_total,
                   reference_model_id, a_ref, i_ref, status, reason
            FROM r_relative_weekly
            WHERE trafo = %s
              AND (%s::date IS NULL OR week_start >= %s)
              AND (%s::date IS NULL OR week_start <= %s)
            ORDER BY phase, week_start ASC
            """,
            (str(trafo), from_date, from_date, to_date, to_date),
        )
        rows = cur.fetchall()

    for row in rows:
        row["week_start"] = row["week_start"].isoformat()
        row["guidance_label"] = resistance.guidance_label(row["r_ratio_median_pct"])

    return {
        "trafo": trafo,
        "count": len(rows),
        "rows": rows,
        "guidance": {
            "good_max_pct": resistance.RESISTANCE_CONFIG["guidance"]["good_max_pct"],
            "fair_max_pct": resistance.RESISTANCE_CONFIG["guidance"]["fair_max_pct"],
            "citation": resistance.GUIDANCE_CITATION,
        },
    }


# =====================================================
# LNTAI vs AGING FACTOR (Stage 2, Part J2)
# =====================================================

@app.get("/api/lntai-vs-aging")
def lntai_vs_aging_endpoint(
    trafo: Optional[int] = Query(default=None, ge=2, le=4, description="Nomor trafo tunggal: 2, 3, atau 4"),
    all_trafos: bool = Query(default=False, alias="all", description="true = semua trafo (2,3,4) sekaligus; mengabaikan `trafo` kalau diisi bersamaan"),
    days: int = Query(
        default=lva.LNTAI_VS_AGING_CONFIG["daily_days_default"], ge=1,
        le=lva.LNTAI_VS_AGING_CONFIG["daily_days_max"],
    ),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Scatter kuadran (titik TERBARU per fasa per trafo: x=faktor
    penuaan, y=LNTAI v3) + tren harian 'days' hari terakhir (LNTAI
    maksimum harian & faktor penuaan rata-rata harian), untuk panel
    "LNTAI vs Aging Factor" (Part J2). Kuadran dihitung
    server/lntai_vs_aging.py (fungsi murni, diuji tanpa DB) -- endpoint
    ini hanya mengambil baris lalu memanggilnya. Titik/hari tanpa data
    cukup (fasa belum ada baris v3, atau aging belum terkonfigurasi)
    MUNCUL dengan nilai null / quadrant='unknown', BUKAN dihilangkan --
    supaya frontend bisa menampilkan empty state yang jelas per titik."""
    _check_api_key(x_api_key)

    if trafo is None and not all_trafos:
        raise HTTPException(status_code=400, detail="Provide either `trafo` (2-4) or `all=true`.")

    trafos = [2, 3, 4] if all_trafos else [trafo]

    scatter: list[dict[str, Any]] = []
    daily: dict[str, dict[str, Any]] = {}

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for t in trafos:
            t_str = str(t)

            cur.execute(
                """
                SELECT per_phase FROM lntai_results
                WHERE trafo_id = %s AND model_version = 'lntai-v3'
                ORDER BY ts DESC LIMIT 1
                """,
                (t_str,),
            )
            lntai_row = cur.fetchone()
            per_phase_lntai = (lntai_row or {}).get("per_phase") or {}

            cur.execute(
                """
                SELECT DISTINCT ON (phase) phase, f_mean
                FROM aging_daily
                WHERE trafo = %s
                ORDER BY phase, date DESC
                """,
                (t_str,),
            )
            f_mean_by_phase = {r["phase"]: r["f_mean"] for r in cur.fetchall()}

            for p in ("R", "S", "T"):
                phase_entry = per_phase_lntai.get(p) if per_phase_lntai else None
                lntai_z = phase_entry.get("z") if phase_entry else None
                aging_factor = f_mean_by_phase.get(p)
                scatter.append(lva.build_scatter_point(t_str, p, lntai_z, aging_factor))

            cur.execute(
                """
                SELECT date_trunc('day', ts) AS day, MAX(lntai) AS max_lntai
                FROM lntai_results
                WHERE trafo_id = %s AND model_version = 'lntai-v3'
                  AND ts >= now() - (%s::text || ' days')::interval
                GROUP BY day ORDER BY day ASC
                """,
                (t_str, days),
            )
            lntai_daily_rows = cur.fetchall()

            cur.execute(
                """
                SELECT date, AVG(f_mean) AS mean_aging_factor
                FROM aging_daily
                WHERE trafo = %s
                  AND date >= (now() - (%s::text || ' days')::interval)::date
                GROUP BY date ORDER BY date ASC
                """,
                (t_str, days),
            )
            aging_daily_rows = cur.fetchall()

            daily[t_str] = {
                "lntai": [
                    {"date": r["day"].date().isoformat(), "max_lntai": r["max_lntai"]}
                    for r in lntai_daily_rows
                ],
                "aging_factor": [
                    {"date": r["date"].isoformat(), "mean_aging_factor": r["mean_aging_factor"]}
                    for r in aging_daily_rows
                ],
            }

    return {
        "trafos": trafos,
        "days": days,
        "quadrant_splits": {
            "aging_factor_split": lva.AGING_FACTOR_SPLIT,
            "lntai_split": lva.LNTAI_SPLIT,
        },
        "quadrant_labels": lva.QUADRANT_LABELS,
        "reading_note": lva.QUADRANT_READING_NOTE,
        "scatter": scatter,
        "daily": daily,
    }


# =====================================================
# FORECAST & EARLY WARNING (Stage 2, Part F)
# =====================================================

_HORIZON_HOURS = {"24h": 24, "7d": 168}


@app.get("/api/forecast")
def forecast_endpoint(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    horizon: str = Query("24h", pattern="^(24h|7d)$", description="'24h' atau '7d'"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Forecast per jam (Stage 2 Part F) untuk ketiga fasa trafo ini,
    plus early warning (titik PERTAMA di mana LNTAI_pred P50 melewati
    ambang level, kalau ada, dalam horizon yang diminta). Baris
    forecasts ditulis server/forecast_job.py -- endpoint ini hanya
    membaca + menyusun ringkasan warning, tidak menghitung ulang."""
    _check_api_key(x_api_key)

    n_hours = _HORIZON_HOURS[horizon]
    now = datetime.now(timezone.utc)
    horizon_end = now + timedelta(hours=n_hours)

    result: dict[str, Any] = {"trafo": trafo, "horizon": horizon, "phases": {}}

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        for phase in ("R", "S", "T"):
            cur.execute(
                """
                SELECT target_ts, load_p10_a, load_p50_a, load_p90_a, t_hot_pred_c,
                       lntai_pred_p10, lntai_pred_p50, lntai_pred_p90, aging_f_pred,
                       method, excess_method
                FROM forecasts
                WHERE trafo = %s AND phase = %s AND target_ts > %s AND target_ts <= %s
                ORDER BY target_ts ASC
                """,
                (str(trafo), phase, now, horizon_end),
            )
            rows = cur.fetchall()
            for row in rows:
                row["target_ts"] = row["target_ts"].isoformat()

            warning = None
            crossing = fc.find_level_crossing(rows)
            if crossing is not None:
                from forecast_job import fetch_aging_rates

                last_7d, avg_30d = fetch_aging_rates(cur, str(trafo), phase, now)
                aging_comparison = fc.compare_aging_rate(last_7d, avg_30d)
                warning = fc.build_warning(crossing, aging_comparison)

            result["phases"][phase] = {
                "count": len(rows), "points": rows, "warning": warning,
                "method": rows[0]["method"] if rows else None,
            }

    return result


# =====================================================
# EXPLAIN & ANOMALY PATTERN (Stage 2, Part G)
# =====================================================

@app.get("/api/explain/latest")
def explain_latest(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    phase: Literal["R", "S", "T"] = Query(..., description="Fasa: R, S, atau T"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Jelaskan prediksi model baseline AKTIF (Part B) untuk sampel
    TERBARU (trafo, phase): suhu hotspot terukur, ΔT sehat yang
    diprediksi, residual tak-terjelaskan, dan kontribusi per kategori
    (lihat ml/explain_contrib.py -- metode berbeda per tipe model,
    SELALU dilaporkan eksplisit, dijamin sum(kontribusi)==prediksi).
    Juga menjalankan CUSUM (server/explain.py) atas histori residual fasa
    ini untuk mendeteksi kenaikan yang konsisten ("consistent rise since
    <date>").

    CATATAN: i_eff_norm untuk fitur "Phase load (filtered)" DIAMBIL ULANG
    dari i_eff_norm yang SUDAH DIHITUNG lntai_worker.py (v2, basis
    1780A) -- bukan state filter lag termal terpisah dengan tau MILIK
    model aktif sendiri (yang butuh state streaming yang tidak tersedia
    di proses API ini) -- diskalakan ke basis i_ref model aktif. Sama
    seperti simplifikasi yang sudah diambil di Part F untuk alasan yang
    sama (lihat server/forecast_job.py)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        baseline_row = fetch_active_baseline_model(cur, str(trafo), phase)
        if baseline_row is None:
            return {"trafo": trafo, "phase": phase, "status": "no_active_model",
                    "reason": "No active baseline model for this trafo/phase (see Part B ml/train_baseline.py + POST /api/ml/models/{id}/activate)."}

        cur.execute(
            """
            SELECT ts, per_phase FROM lntai_results
            WHERE trafo_id = %s AND model_version = 'lntai-v2' AND level != 'DATA_INVALID'
            ORDER BY ts DESC LIMIT 1
            """,
            (str(trafo),),
        )
        latest = cur.fetchone()
        if latest is None or not latest.get("per_phase") or phase not in latest["per_phase"]:
            return {"trafo": trafo, "phase": phase, "status": "no_data",
                    "reason": "No valid lntai_results (v2) sample available yet for this trafo/phase."}

        phase_data = latest["per_phase"][phase]
        measured_hotspot_c = phase_data["T"]
        i_eff_norm_v2 = phase_data["i_eff_norm"]
        ts = latest["ts"]

        with open(_ROOT / baseline_row["artifact_path"], "r", encoding="utf-8") as f:
            artifact = json.load(f)
        model = load_model_from_artifact(baseline_row["model_type"], artifact)

        i_eff_amps = i_eff_norm_v2 * config.LNTAI_V2_I_RATED
        hour_sin, hour_cos, weekday = hour_sin_cos_weekday(ts.timestamp())

        # load_roc_1h: sampel ~1 jam lalu, query sekali (bukan state
        # streaming -- lihat catatan docstring fungsi ini).
        cur.execute(
            """
            SELECT ia, ib, ic FROM trafo_load
            WHERE trafo_id = %s AND "timestamp" BETWEEN %s AND %s
            ORDER BY ABS(EXTRACT(EPOCH FROM ("timestamp" - %s))) LIMIT 1
            """,
            (str(trafo), ts - timedelta(minutes=70), ts - timedelta(minutes=50), ts - timedelta(hours=1)),
        )
        past_load_row = cur.fetchone()
        past_current = {"R": "ia", "S": "ib", "T": "ic"}
        past_current_val = past_load_row.get(past_current[phase]) if past_load_row else None
        load_roc_1h = (i_eff_amps - past_current_val) if past_current_val is not None else 0.0

        feature_row = {
            "current_amps": i_eff_amps, "i2_eff_default_tau": i_eff_amps ** 2,
            "load_roc_1h": load_roc_1h, "hour_sin": hour_sin, "hour_cos": hour_cos, "weekday": float(weekday),
        }

        model_type = baseline_row["model_type"]
        i_eff_norm_override = None
        if model_type == "physics_rc" and model.i_ref:
            i_eff_norm_override = i_eff_amps / model.i_ref
        elif model_type == "hybrid" and model.physics.i_ref:
            i_eff_norm_override = i_eff_amps / model.physics.i_ref

        contributions, method = explain_contrib.explain_model(model_type, model, feature_row, i_eff_norm_override)
        predicted_healthy_c = sum(contributions.values())

        lookback_days = xp.EXPLAIN_CONFIG["lookback_days"]
        cur.execute(
            """
            SELECT ts, per_phase FROM lntai_results
            WHERE trafo_id = %s AND model_version = 'lntai-v2' AND level != 'DATA_INVALID'
              AND ts >= %s
            ORDER BY ts ASC
            """,
            (str(trafo), ts - timedelta(days=lookback_days)),
        )
        history_rows = cur.fetchall()

    residual_history: list[tuple[Any, float]] = []
    for row in history_rows:
        pp = row.get("per_phase") or {}
        pd = pp.get(phase)
        if pd is None or pd.get("dT") is None or pd.get("dT_pred") is None:
            continue
        residual_history.append((row["ts"], pd["dT"] - pd["dT_pred"]))

    cusum_triggered, cusum_start = xp.compute_cusum_over_history(residual_history)
    explanation = xp.build_explanation(
        measured_hotspot_c, predicted_healthy_c, contributions, method, cusum_triggered, cusum_start,
    )

    return {
        "trafo": trafo, "phase": phase, "status": "ok", "timestamp": ts.isoformat(),
        "baseline_model_id": baseline_row["id"], "baseline_model_type": model_type,
        **explanation,
    }


# =====================================================
# DEGRADATION PROJECTION (Stage 2, Part H, beta)
# =====================================================

@app.get("/api/degradation/projection")
def degradation_projection(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    phase: Literal["R", "S", "T"] = Query(..., description="Fasa: R, S, atau T"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Proyeksi Bayesian (beta) tanggal P10/P50/P90 saat R/R_ref mingguan
    (Part D) diproyeksikan melewati ambang 'fair/poor' (120%, kriteria
    KEMA/Mensah 2011 -- sama dipakai Part D). status='insufficient_data'
    kalau belum ada >= min_valid_weeks titik mingguan ATAU belum ada SATU
    PUN event resistance_test (Part E) untuk kalibrasi -- lihat
    server/degradation.py docstring modul untuk penjelasan kenapa
    KEDUANYA wajib (bukan salah satu saja)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        result = dg.build_projection(cur, str(trafo), phase)

    return {"trafo": trafo, "phase": phase, **result}
