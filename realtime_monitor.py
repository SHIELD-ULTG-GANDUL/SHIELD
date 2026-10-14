"""Monitoring realtime LNTAI v2.

Dua mode:
  - dummy (default) — data sampel berputar, tidak butuh koneksi apa pun.
        python realtime_monitor.py

  - timescale — baca baris suhu & beban TERBARU dari TimescaleDB (tabel
    trafo_temperature / trafo_load, sama seperti server/api.py). Kredensial
    TIDAK disimpan di kode — set env var DATABASE_URL dulu (lihat
    server/.env.example), lalu:
        export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shield_data
        python realtime_monitor.py --source timescale --trafo 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import cycle
from typing import Any

from lntai_calculator import LNTAICalculator


def get_dummy_sample() -> dict[str, Any]:
    """Sampel dummy berputar — bentuk payload sama dengan sample v2 asli
    (6 titik suhu + 3 arus fasa), supaya representatif tanpa perlu sensor."""

    samples = [
        {"R_Atas": 58, "R_Bawah": 56, "S_Atas": 55, "S_Bawah": 54, "T_Atas": 57, "T_Bawah": 55,
         "I_R": 780, "I_S": 780, "I_T": 780},
        {"R_Atas": 72, "R_Bawah": 70, "S_Atas": 68, "S_Bawah": 67, "T_Atas": 69, "T_Bawah": 68,
         "I_R": 850, "I_S": 850, "I_T": 850},
        {"R_Atas": 84, "R_Bawah": 80, "S_Atas": 75, "S_Bawah": 74, "T_Atas": 76, "T_Bawah": 75,
         "I_R": 620, "I_S": 620, "I_T": 620},
        {"R_Atas": 90, "R_Bawah": 86, "S_Atas": 85, "S_Bawah": 84, "T_Atas": 87, "T_Bawah": 86,
         "I_R": 1000, "I_S": 1000, "I_T": 1000},
    ]
    if not hasattr(get_dummy_sample, "_cycle"):
        get_dummy_sample._cycle = cycle(samples)  # type: ignore[attr-defined]
    return dict(next(get_dummy_sample._cycle))  # type: ignore[attr-defined]


def make_timescale_sample_fetcher(trafo: str):
    """Mengembalikan fungsi get_sample() yang membaca baris suhu & beban
    TERBARU untuk `trafo` dari TimescaleDB. DATABASE_URL wajib di env var
    (tidak ada default/kredensial tertanam di sini)."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "DATABASE_URL belum diset. Contoh:\n"
            "  export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shield_data\n"
            "Lihat server/.env.example untuk format lengkap."
        )

    import psycopg  # import lokal: dependency opsional, hanya dibutuhkan mode ini

    conn = psycopg.connect(database_url)

    def get_sample() -> dict[str, Any] | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah
                FROM trafo_temperature WHERE trafo_id = %s
                ORDER BY "timestamp" DESC LIMIT 1
                """,
                (trafo,),
            )
            temp_row = cur.fetchone()
            cur.execute(
                """
                SELECT ia, ib, ic FROM trafo_load WHERE trafo_id = %s
                ORDER BY "timestamp" DESC LIMIT 1
                """,
                (trafo,),
            )
            load_row = cur.fetchone()

        if temp_row is None or load_row is None:
            return None

        r_atas, r_bawah, s_atas, s_bawah, t_atas, t_bawah = temp_row
        i_r, i_s, i_t = load_row
        return {
            "R_Atas": r_atas, "R_Bawah": r_bawah,
            "S_Atas": s_atas, "S_Bawah": s_bawah,
            "T_Atas": t_atas, "T_Bawah": t_bawah,
            "I_R": i_r, "I_S": i_s, "I_T": i_t,
        }

    return get_sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["dummy", "timescale"], default="dummy",
                         help="Sumber data: 'dummy' (default, tanpa koneksi) atau 'timescale' (baca TimescaleDB live).")
    parser.add_argument("--trafo", default="2", help="Nomor trafo (dipakai hanya untuk --source timescale). Default: 2.")
    parser.add_argument("--interval", type=float, default=3.0, help="Jeda antar sampel (detik). Default: 3.")
    args = parser.parse_args()

    calculator = LNTAICalculator()

    if args.source == "timescale":
        get_sample = make_timescale_sample_fetcher(args.trafo)
        print(f"SHIELD LNTAI v2 realtime monitor aktif (sumber: TimescaleDB, trafo {args.trafo}). Tekan Ctrl+C untuk berhenti.")
    else:
        get_sample = get_dummy_sample
        print("SHIELD LNTAI v2 realtime monitor aktif (sumber: dummy). Tekan Ctrl+C untuk berhenti.")

    try:
        while True:
            sample = get_sample()
            if sample is None:
                print("Belum ada data suhu/beban untuk trafo ini di TimescaleDB, menunggu...")
                time.sleep(args.interval)
                continue

            result = calculator.update(args.trafo, sample)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if result.get("level") in {"WARNING", "CRITICAL"}:
                print(f"ALERT {result['level']}: LNTAI={result.get('lntai')} (hottest phase {result.get('hottest_phase')})")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring LNTAI dihentikan.")


if __name__ == "__main__":
    main()
