# SHIELD — Kamus Data

Dokumen ini mendefinisikan setiap tabel dan kolom di database PostgreSQL
SHIELD setelah **SHIELD Tahap 1 — Fondasi Data** (Oktober 2026). Database
ini **PostgreSQL 18.4 polos — TimescaleDB TIDAK terpasang** (dikonfirmasi
audit Fase 0); abaikan referensi hypertable/continuous-aggregate di
dokumentasi lama.

Konvensi NULL di seluruh dokumen ini: **NULL berarti data tidak
tersedia/hilang/tidak valid, BUKAN nol.** Lihat SHIELD Tahap 1 T2.

---

## `trafo_temperature`

Satu baris per pembacaan suhu dari satu trafo. Ditulis oleh
`mqtt_subscriber.py` (lihat `server/README.md` §1 untuk kenapa ini
ditetapkan sebagai writer aktif, bukan `mqtt_ingester.py`).

| Kolom | Tipe | Unit | Arti NULL | Catatan |
|---|---|---|---|---|
| `id` | BIGSERIAL (PK) | — | — | |
| `timestamp` | TIMESTAMPTZ | — | tidak pernah NULL | UTC. Waktu PROSES SUBSCRIBER saat pesan diterima, **bukan** timestamp dari payload MQTT (lihat catatan timezone di `mqtt_subscriber.py`). |
| `trafo_id` | TEXT | — | tidak pernah NULL | `"2"`, `"3"`, atau `"4"`. |
| `r_atas`, `r_bawah`, `s_atas`, `s_bawah`, `t_atas`, `t_bawah` | DOUBLE PRECISION | °C | sensor itu tidak mengirim data / nilai tidak valid pada sampel ini | 6 titik ukur (fasa R/S/T, posisi atas/bawah kontak finger). |
| `temp_max` | DOUBLE PRECISION | °C | SEMUA 6 titik di atas NULL pada sampel ini | `max()` dari titik yang non-NULL saja. |
| `temp_avg` | DOUBLE PRECISION | °C | SEMUA 6 titik di atas NULL pada sampel ini | Rata-rata titik yang non-NULL saja. |
| `temp_ambient` | DOUBLE PRECISION | °C | — | **BUKAN suhu cuaca/ambient.** Selalu bernilai statis `30` di semua baris yang pernah ditulis sebelum Tahap 1 (dikonfirmasi audit) — dicadangkan untuk sensor ambient DI DALAM panel trafo di masa depan. Suhu cuaca ada di tabel `ambient_logs`, bukan di sini. API (`/api/temperature-logs`) tidak lagi mengembalikan kolom ini sebagai field "ambient". |
| `quality_flags` | INTEGER | bitmask | baris lama sebelum T5 berjalan, ATAU aturan berbasis-jendela (STUCK/SPIKE/dll.) belum dihitung | Lihat **Bitmask quality_flags** di bawah. Diisi `mqtt_subscriber.py` untuk aturan per-sampel (MISSING/ZERO_READING/OUT_OF_RANGE) saat INSERT; aturan berbasis jendela (STUCK/SPIKE/PAIRING_GAP/TIME_GAP) dihitung terpisah oleh `lntai_worker.py` dan tercermin di `lntai_results.quality_flags`, BUKAN di-OR balik ke kolom ini. |

## `trafo_load`

Satu baris per pembacaan beban (arus/tegangan/daya) dari satu trafo.
Ditulis bersamaan (dalam transaksi yang sama) dengan baris
`trafo_temperature` yang berpasangan waktu, oleh `mqtt_subscriber.py`.

| Kolom | Tipe | Unit | Arti NULL | Catatan |
|---|---|---|---|---|
| `id` | BIGSERIAL (PK) | — | — | |
| `timestamp` | TIMESTAMPTZ | — | tidak pernah NULL | UTC, sama seperti `trafo_temperature.timestamp`. |
| `trafo_id` | TEXT | — | tidak pernah NULL | `"2"`, `"3"`, atau `"4"`. |
| `ia`, `ib`, `ic` | DOUBLE PRECISION | A | field tidak dikirim/tidak valid | Arus fasa R/S/T. `0` yang memang dikirim (CB terbuka) **tersimpan sebagai 0, bukan NULL** — lihat `ZERO_READING` di bawah. |
| `current_max` | DOUBLE PRECISION | A | ia, ib, ic SEMUA NULL | `max()` dari fasa yang non-NULL. |
| `kw_total`, `kva_total`, `kvar_total`, `pf_total`, `freq` | DOUBLE PRECISION | kW / kVA / kVAR / — / Hz | field tidak dikirim/tidak valid | |
| `van`, `vbn`, `vcn` | DOUBLE PRECISION | V | field tidak dikirim/tidak valid | Tegangan fasa-netral. |
| `vab`, `vbc`, `vca` | DOUBLE PRECISION | V | field tidak dikirim/tidak valid | Tegangan fasa-fasa. |
| `quality_flags` | INTEGER | bitmask | baris lama sebelum T5 | Sama seperti `trafo_temperature.quality_flags`, dihitung dari `ia`/`ib`/`ic` saja (MISSING/ZERO_READING/OUT_OF_RANGE terhadap kelipatan arus nominal). |

## Bitmask `quality_flags`

Didefinisikan di `server/data_quality_config.json` (satu sumber kebenaran,
dibaca `server/data_quality.py`) — **jangan hardcode ulang nilai bit di
tempat lain.**

| Bit | Nilai | Arti | Dihitung oleh | Menghentikan LNTAI? |
|---|---|---|---|---|
| `MISSING` | 1 | Field hilang dari payload / tidak bisa dikonversi ke angka | `mqtt_subscriber.py` (per-sampel) | **Ya** |
| `ZERO_READING` | 2 | Nilai persis 0 — bacaan SAH (bukan data hilang), hanya ditandai untuk audit | `mqtt_subscriber.py` (per-sampel) | Tidak |
| `OUT_OF_RANGE` | 4 | Di luar rentang fisik wajar (suhu: lihat `temperature_range_c`; arus: > kelipatan `i_rated`, TODO(verifikasi)) | `mqtt_subscriber.py` (per-sampel) | **Ya** |
| `STUCK` | 8 | ≥30 sampel terakhir identik dan bukan NULL — sensor diduga macet | `lntai_worker.py` (berbasis jendela, hasil tercermin di `lntai_results.quality_flags`) | **Ya** |
| `SPIKE` | 16 | Lompatan >15°C yang kembali di sampel berikutnya — noise sesaat, bukan tren naik sah | `lntai_worker.py` | Tidak |
| `PAIRING_GAP` | 32 | Tidak ada baris beban (atau suhu) dalam `pairing_tolerance_minutes` dari baris ini | `lntai_worker.py` | Tidak secara langsung (tapi `lntai_worker.py` TETAP tidak menghitung LNTAI kalau `paired=false`, lihat `lntai_results.paired`) |
| `TIME_GAP` | 64 | Jarak waktu ke sampel sebelumnya > 3× interval nominal | `lntai_worker.py` | Tidak |

Semua ambang (rentang suhu, kelipatan arus, jumlah sampel STUCK, delta
SPIKE, dst.) ada di `server/data_quality_config.json` dengan komentar
`TODO(verifikasi)` — ini DEFAULT dari spesifikasi Tahap 1, **belum**
divalidasi dari data riil trafo 2/3/4.

## `ambient_logs`

Suhu ambient (cuaca) sekitar GI Gandul dari Open-Meteo, **ditulis
browser** lewat relay (Cabang B, T3 — lihat `server/README.md` §8 untuk
alasan arsitekturnya). **Bukan** sensor di dalam panel trafo.

| Kolom | Tipe | Unit | Arti NULL | Catatan |
|---|---|---|---|---|
| `id` | BIGSERIAL (PK) | — | — | |
| `timestamp` | TIMESTAMPTZ | — | tidak pernah NULL | Waktu browser menerima respons Open-Meteo (UTC). |
| `slot_10min` | TIMESTAMPTZ | — | tidak pernah NULL | `timestamp` dibulatkan ke bawah ke kelipatan 10 menit. **UNIQUE** — kunci dedup, maksimal satu baris per slot. |
| `temperature_c` | DOUBLE PRECISION | °C | `ok=false` (fetch Open-Meteo gagal di browser) | **Tidak pernah** diisi nilai interpolasi/rekaan. |
| `ok` | BOOLEAN | — | tidak pernah NULL | `false` = fetch gagal untuk slot ini (lihat `temperature_c`). |
| `source` | TEXT | — | tidak pernah NULL | Default `'open-meteo-browser-relay'`. |
| `latitude`, `longitude` | DOUBLE PRECISION | derajat | tidak pernah NULL | Default lokasi GI Gandul dari `GANDUL_LATITUDE`/`GANDUL_LONGITUDE` (settings.py) kalau browser tidak mengirim eksplisit. |

**Keterbatasan yang disengaja:** histori hanya terisi selama ada dashboard
terbuka di browser — ini konsekuensi arsitektur Cabang B, bukan bug.

## `lntai_results`

Hasil LNTAI v2 dihitung **server-side** oleh `lntai_worker.py`/
`backfill_lntai.py` (T4), memakai ULANG `lntai_calculator.LNTAICalculator`
(root proyek) — formula LNTAI didefinisikan di sana dan di blok
`<LNTAI-CORE>` index.html, **bukan** di tabel ini.

| Kolom | Tipe | Unit | Arti NULL | Catatan |
|---|---|---|---|---|
| `trafo_id`, `ts`, `model_version` | TEXT, TIMESTAMPTZ, TEXT | — | tidak pernah NULL | **PRIMARY KEY gabungan** — idempotent untuk backfill ulang; model_version berbeda tersimpan berdampingan (bukan menimpa). |
| `lntai` | DOUBLE PRECISION | skor 0–40 | `level='DATA_INVALID'` (input diblokir quality/pairing) | |
| `level` | TEXT | — | tidak pernah NULL | `NORMAL`/`WARNING`/`CRITICAL`/`DATA_INVALID`. |
| `s_temp`, `s_delta_t`, `s_res` | DOUBLE PRECISION | skor 0–40 | sama seperti `lntai` | Tiga komponen skor komposit. |
| `hottest_phase` | TEXT | — | sama seperti `lntai` | `R`/`S`/`T`. |
| `t_contact`, `t_ambient` | DOUBLE PRECISION | °C | sama seperti `lntai` | `t_ambient` dari `ambient_at()` (T3) atau default `lntai_config.json`. |
| `ambient_source` | TEXT | — | sama seperti `lntai` | `"ambient_logs"` atau `"default"` — lihat `server/ambient.py`. |
| `delta_t_real`, `delta_t_pred`, `residual`, `z`, `i_eff_norm`, `delta_phase`, `dab_max` | DOUBLE PRECISION | — | sama seperti `lntai` | Lihat `lntai_calculator.py` untuk definisi matematis tiap field. |
| `per_phase` | JSONB | — | sama seperti `lntai` | Rincian per fasa R/S/T: `{T, I, dT, dT_pred, z, dAB, i_eff_norm}`. |
| `quality_flags` | INTEGER | bitmask | — | OR gabungan flag baris suhu + baris beban yang dipasangkan, ditambah `PAIRING_GAP` kalau `paired=false`. |
| `paired` | BOOLEAN | — | tidak pernah NULL | `false` = tidak ada baris beban dalam `pairing_tolerance_minutes` — LNTAI TIDAK dihitung dari sampel ini sama sekali. |
| `error` | TEXT | — | input valid & dihitung normal | Alasan `DATA_INVALID` (mis. "PAIRING_GAP: tidak ada baris beban..."). |

## `worker_state`

Checkpoint generik per worker batch (saat ini hanya dipakai
`lntai_worker.py`, PK `(worker_name, trafo_id)`). `last_processed_ts`
NULL = worker belum pernah berjalan untuk trafo itu.

## `maintenance_windows`

Jendela waktu ditandai manual (lewat `POST /api/maintenance-windows`)
sebagai periode maintenance/kalibrasi/gangguan. `trafo_id` NULL = berlaku
semua trafo. `archived=true` setelah `archive_job.py` menyalin baris
dalam jendela itu ke `*_archive` (lihat di bawah).

## `trafo_temperature_archive`, `trafo_load_archive`

Salinan baris `trafo_temperature`/`trafo_load` yang jatuh di dalam SATU
`maintenance_windows`, ditulis `archive_job.py`. Skema kolom SAMA PERSIS
dengan tabel asal (lihat di atas) ditambah `archived_at` (waktu disalin).
**Baris asli TIDAK dihapus** oleh proses penyalinan ini — penghapusan
(retensi) terpisah dan sengaja TIDAK AKTIF (lihat
`server/migrations/007_retention_policy_INACTIVE.sql`).

---

## Tabel/skrip LEGACY (di luar database yang didokumentasikan di atas)

- `shield_data` (Firestore) — sumber data SEBELUM migrasi ke Postgres;
  `migrate_firestore_to_timescale.py` (DEPRECATED) pernah menyalin dari
  sini, sekarang 0 baris baru per-run.
- `temperature_logs` (tabel Postgres lama) — ditulis `mqtt_ingester.py`
  (DEPRECATED, service dinonaktifkan). **Jangan dibaca** sebagai sumber
  data aktif — gunakan `trafo_temperature`.
