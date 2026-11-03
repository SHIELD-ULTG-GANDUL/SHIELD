# SHIELD Autopilot Log — autopilot/2026-11

Mulai: 2026-11-03. Branch dibuat dari `main` (commit `b932e08`, setelah
perbaikan TLS `mqtt_subscriber.py` di-commit terpisah ke `main`).

## Batasan environment (dicatat jujur, bukan diklaim lolos)

- **`node` tidak terpasang** di environment eksekusi ini. Gerbang kualitas
  `node tests/test_lntai_js.mjs` (parity JS LNTAI-CORE vs Python) **TIDAK
  BISA dijalankan/diverifikasi di sini** untuk tugas apa pun di bawah.
  Mitigasi: blok `<LNTAI-CORE>`...`</LNTAI-CORE>` di index.html TIDAK
  disentuh sama sekali oleh tugas apa pun (dicek lewat grep sebelum &
  sesudah setiap tugas yang menyentuh index.html). Manusia WAJIB
  menjalankan `node tests/test_lntai_js.mjs` sebelum merge.
- Gerbang yang BISA dijalankan & benar-benar ditegakkan: `pytest -q`
  (427 test lolos di baseline) dan `cmp index.html Shield.html`.

## Baseline (sebelum tugas pertama)

- `pytest -q` (root): 427 passed, 13 subtests passed
- `cmp index.html Shield.html`: IDENTIK
- `node tests/test_lntai_js.mjs`: TIDAK BISA DIJALANKAN (node tidak ada)

## Status Tugas

| ID | Status | Commit | Catatan & yang belum diverifikasi |
|---|---|---|---|
| S3 | DONE | d2d1295 | Targeted pass (bukan audit menyeluruh 46 lokasi innerHTML) -- cakup field yang disebut tugas (provisional_reason, healthy_from/to, notes-equivalent cells, pesan error server). LNTAI-CORE & narasi insight di sebelahnya sengaja tidak disentuh (field di situ enum/angka, bukan teks bebas). |
| M10 | DONE | ba0b65e | Koreksi vs brief: call site asli ada di server/api.py (GET /api/forecast), bukan forecast_job.py -- forecast_job.py cuma menghitung & menulis titik forecast, deteksi crossing jalan saat endpoint dibaca. elevated_at_start() dipasang di api.py. |
| M11 | DONE | 22001f3 | Dibuktikan dengan noise heteroscedastic (siang 3x malam) bahwa band bucket siang lebih lebar dari malam -- bukan diasumsikan. Backward compat artefak lama diverifikasi test eksplisit. |
| M12 | DONE | bb22ca8 | --tune hanya mencakup ml_gbm, bukan residual GBM di dalam hybrid (scoping, dicatat di commit). naive_median TIDAK ditulis ke ml_models (bukan di CHECK constraint, memang bukan model teraktivasi -- skip_save flag). |
| U7 | DONE | 00909b0 | suhu/beban freshness FLAT (bukan per-trafo), sesuai bentuk objek di task. Tab Forecast dibiarkan pakai indikator sendiri (fc-last-update-tab), tidak dipetakan ke 4 kategori. Tidak bisa diverifikasi visual di browser (tidak ada akses) -- cek manual disarankan. |
| U5 | PARTIAL | 1bfee23 | Status strip + ikon Lucide + badge skala v2/v3 selesai. "Ganti label OT Normalized/Load Factor" SENGAJA tidak dikerjakan -- tumpang tindih dengan rencana unifikasi field LNTAI v2/v3 yang belum selesai dari sesi sebelumnya (S_TLR->S_RES dst). Tidak bisa diverifikasi visual (tidak ada browser). |
| U4 | TODO | | |
| F1 | TODO | | |
| F2 | TODO | | |
| F3 | TODO | | |
| F4 | TODO | | |
| F5 | TODO | | |
| F6 | TODO | | |
| F7 | TODO | | |
| U6 | TODO | | |

## Tidak dikerjakan (butuh manusia, sesuai instruksi)

- Ganti password Firebase yang tertanam di HTML
- Firestore Security Rules & verifikasi Firebase ID token server (S2)
- Fitur ringkasan alarm berbasis LLM
- Retraining model dengan data produksi & aktivasi model
- Push ke remote bila kredensial tidak tersedia (di sini kredensial GitHub
  ADA dan sudah terbukti jalan di sesi ini -- push akan dicoba di akhir)
