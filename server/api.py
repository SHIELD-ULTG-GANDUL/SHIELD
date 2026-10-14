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

from settings import load_api_settings, load_database_settings

_db_settings = load_database_settings()
_api_settings = load_api_settings()

# Opsional: kalau API_KEY diisi (lewat settings.py/env var), semua request wajib
# menyertakan header X-API-Key yang cocok. Kosongkan kalau API ini hanya dipakai
# secara internal/read-only publik -- perilaku ini dipertahankan dari versi lama,
# bukan diubah oleh migrasi ke settings.py.
API_KEY = _api_settings.api_key

app = FastAPI(title="SHIELD Temperature Log API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_api_settings.allowed_origins,
    allow_methods=["GET"],
    allow_headers=["X-API-Key"],
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
    Firestore supaya sisi frontend (Shield.html) tinggal memetakan langsung."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT trafo_id AS trafo, r_atas AS "R_Atas", r_bawah AS "R_Bawah",
                   s_atas AS "S_Atas", s_bawah AS "S_Bawah",
                   t_atas AS "T_Atas", t_bawah AS "T_Bawah",
                   temp_ambient AS "ambient",
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
    (van/vbn/vcn); V_RS/V_ST/V_TR adalah tegangan fasa-fasa (vab/vbc/vca)."""
    _check_api_key(x_api_key)

    with _pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT trafo_id AS trafo, ia AS "I_R", ib AS "I_S", ic AS "I_T",
                   COALESCE(van, 0) AS "V_R", COALESCE(vbn, 0) AS "V_S", COALESCE(vcn, 0) AS "V_T",
                   COALESCE(vab, 0) AS "V_RS", COALESCE(vbc, 0) AS "V_ST", COALESCE(vca, 0) AS "V_TR",
                   kw_total AS "kW", COALESCE(kvar_total, 0) AS "kVAR", kva_total AS "kVA", pf_total AS "PF",
                   current_max, freq,
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
