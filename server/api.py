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

import os
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL belum diset. Contoh: postgresql://shield:password@localhost:5432/shield")

# Opsional: kalau diisi, semua request wajib menyertakan header X-API-Key yang cocok.
# Kosongkan (jangan diset) kalau API ini hanya dipakai secara internal/read-only publik.
API_KEY = os.environ.get("API_KEY", "")

# Origin yang boleh memanggil API ini dari browser. Isi domain GitHub Pages Anda,
# tambahkan yang lain kalau perlu (mis. domain custom).
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://shield-ultg-gandul.github.io"
    ).split(",") if o.strip()
]

app = FastAPI(title="SHIELD Temperature Log API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["X-API-Key"],
)

_pool = psycopg2.pool.SimpleConnectionPool(1, 5, dsn=DATABASE_URL)


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key tidak valid")


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/temperature-logs")
def temperature_logs(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    limit: int = Query(300, ge=1, le=2000),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Mengembalikan N pembacaan suhu terbaru untuk satu trafo, terurut dari
    yang paling baru. Bentuk field sengaja disamakan dengan skema lama di
    Firestore supaya sisi frontend (Shield.html) tinggal memetakan langsung."""
    _check_api_key(x_api_key)

    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT trafo, r_atas AS "R_Atas", r_bawah AS "R_Bawah",
                       s_atas AS "S_Atas", s_bawah AS "S_Bawah",
                       t_atas AS "T_Atas", t_bawah AS "T_Bawah",
                       device_ts AS "deviceTimestamp", source,
                       ts
                FROM temperature_logs
                WHERE trafo = %s
                ORDER BY ts DESC
                LIMIT %s
                """,
                (trafo, limit),
            )
            rows = cur.fetchall()
    finally:
        _pool.putconn(conn)

    for row in rows:
        row["timestamp"] = row["ts"].isoformat()
        del row["ts"]

    return {"trafo": trafo, "count": len(rows), "rows": rows}
