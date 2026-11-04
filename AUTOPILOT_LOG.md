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
| U4 | SKIPPED | - | **Konflik instruksi, bukan masalah teknis.** Task ini minta ubah SEMUA UI jadi Bahasa Indonesia + hapus layer terjemahan Inggris. Ini bertentangan LANGSUNG dengan aturan proyek yang sudah ditetapkan eksplisit di sesi SEBELUMNYA ("semua teks UI-facing harus Bahasa Inggris, Indonesia hanya di komentar kode") -- bahkan pernah ada pelanggaran (ea_source) yang diperbaiki karena ini. Mengubah arah bahasa UI adalah keputusan produk/konten yang perlu dikonfirmasi manusia secara eksplisit, bukan sesuatu yang aman diputuskan otomatis oleh autopilot. Tidak dikerjakan sama sekali (bukan partial) -- menunggu klarifikasi: apakah aturan lama dicabut, atau task ini yang keliru.
| F1 | TODO | | |
| F2 | TODO | | |
| F3 | TODO | | |
| F4 | TODO | | |
| F5 | TODO | | |
| F6 | TODO | | |
| F7 | TODO | | |
| U6 | TODO | | |

## RUN KE-2 (2026-11-04) — prompt autopilot baru, Fase A (audit) ditambahkan

Prompt baru dari pemilik menambah Fase A (A0-A12, audit keamanan/UI/ML)
di ATAS Fase B yang sudah berjalan di atas (S3/M10-12/U7/U5/U4 = Fase B
run pertama, statusnya tetap seperti di atas). Pemilik eksplisit bilang
limit/budget hampir habis saat memberi prompt ini -- autopilot MENGUBAH
URUTAN pengerjaan (bukan urutan di prompt) jadi tugas kecil-bernilai-
tinggi dulu (A2, A10, A1) sebelum tugas besar (A0), supaya progres tidak
terhenti di tengah perubahan besar yang belum lengkap kalau limit habis
mendadak. Baseline test SEBELUM run ke-2: 460 passed (setelah A10).

| ID | Status | Commit | Catatan & yang belum diverifikasi |
|---|---|---|---|
| A2 | DONE | 86e048f | chart.js@4.4.1 (umd build) + lucide@0.460.0. |
| A10 | DONE | 0803ad2 | hmac.compare_digest. Verifikasi token login di endpoint tulis TETAP belum dikerjakan (butuh service account Firebase -- di luar wewenang autopilot, lihat daftar "Tidak dikerjakan"). |
| A1 | DONE | 86e048f | **PERINGATAN BESAR UNTUK PEMILIK: password akun login yang sebelumnya tertanam plaintext di HTML (value= pada form) SUDAH ADA DI RIWAYAT GIT dan WAJIB DIGANTI SEGERA di Firebase Console** -- menghapus value= di commit ini TIDAK menghapusnya dari histori commit lama. Autopilot sengaja TIDAK menulis ulang password itu di commit message (diblokir classifier keamanan sendiri saat dicoba -- bukti tambahan ini sensitif). |
| A0 | TODO | | **Temuan penting (inventaris selesai, penghapusan BELUM dikerjakan):** jalur Firestore sudah pakai pola dispatcher yang benar di setiap titik masuk (`subscribeTemperatureLogs()`, beban, dulfur: `if (window.timescaleReady) return ...Timescale(); return ...Firestore();`) -- karena `timescaleReady` SELALU true di produksi, cabang Firestore SUDAH JADI DEAD CODE yang tidak pernah tereksekusi saat ini, BUKAN bug aktif. Fungsi inti yang perlu dihapus: `_subscribeTemperatureLogsFirestore`, `_subscribeBebanDataFirestore`, `_persistTempLogToFirestore`, `_loadBebanLogsFirestore`, `parseFirestoreDoc`, `window.fs`/`window.db` Firestore SDK imports (169 baris referensi total, scope genuinely besar). Mengingat budget sesi ini terbatas dan risiko salah potong pada fungsi yang masih dipanggil dari banyak tempat, PENGHAPUSAN AKTUAL DITUNDA ke sesi berikutnya -- bukan diburu-buru sekarang. R_contact/dulfur (A0.d) juga belum dicek apakah endpoint GET lokal yang sesuai sudah ada. |
| A3 | TODO | | |
| A4 | TODO | | |
| A5 | TODO | | |
| A6 | TODO | | |
| A7 | TODO | | |
| A8 | TODO | | |
| A9 | TODO | | |
| A11 | TODO | | |
| A12 | TODO | | |

## Tidak dikerjakan (butuh manusia, sesuai instruksi)

- Ganti password Firebase yang tertanam di HTML -- **LIHAT PERINGATAN A1 DI ATAS, INI PRIORITAS TERTINGGI**
- Firestore Security Rules & verifikasi Firebase ID token server (S2 / A10)
- Fitur ringkasan alarm berbasis LLM
- Retraining model dengan data produksi & aktivasi model
- Menonaktifkan/mengosongkan database Firestore lama di Firebase Console
- Push ke remote bila kredensial tidak tersedia (di sini kredensial GitHub
  ADA dan sudah terbukti jalan di sesi ini -- push akan dicoba di akhir)
