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

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
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

_pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=5, open=True)


def _check_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key tidak valid")


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/temperature-logs")
def temperature_logs(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    limit: int = Query(300, ge=1, le=5000),
    from_: Optional[datetime] = Query(default=None, alias="from", description="Batas awal ISO8601, opsional"),
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
            WHERE trafo_id = %s AND (%s::timestamptz IS NULL OR "timestamp" >= %s)
            ORDER BY "timestamp" DESC
            LIMIT %s
            """,
            (str(trafo), from_, from_, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        row["timestamp"] = row["ts"].isoformat()
        del row["ts"]

    return {"trafo": trafo, "count": len(rows), "rows": rows}


@app.get("/api/beban-logs")
def beban_logs(
    trafo: int = Query(..., ge=2, le=4, description="Nomor trafo: 2, 3, atau 4"),
    limit: int = Query(300, ge=1, le=5000),
    from_: Optional[datetime] = Query(default=None, alias="from", description="Batas awal ISO8601, opsional"),
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
            WHERE trafo_id = %s AND (%s::timestamptz IS NULL OR "timestamp" >= %s)
            ORDER BY "timestamp" DESC
            LIMIT %s
            """,
            (str(trafo), from_, from_, limit),
        )
        rows = cur.fetchall()

    for row in rows:
        row["timestamp"] = row["ts"].isoformat()
        del row["ts"]

    return {"trafo": trafo, "count": len(rows), "rows": rows}
