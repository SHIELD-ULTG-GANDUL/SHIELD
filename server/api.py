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

from datetime import datetime, timezone
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ambient
from settings import load_ambient_location_settings, load_api_settings, load_database_settings

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
