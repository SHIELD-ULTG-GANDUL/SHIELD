# SHIELD — Backend server/

Backend PostgreSQL (**bukan TimescaleDB** — lihat catatan di bawah) untuk
dashboard SHIELD. Ditulis ulang besar-besaran di **SHIELD Tahap 1 — Fondasi
Data** (Oktober 2026); dokumen ini menggantikan README lama yang masih
mengasumsikan TimescaleDB terpasang dan `mqtt_ingester.py` sebagai penulis
aktif — keduanya sudah tidak benar sejak audit Fase 0.

Untuk definisi tiap tabel/kolom (unit, arti NULL, arti bit quality_flags),
lihat **[docs/DATA_DICTIONARY.md](../docs/DATA_DICTIONARY.md)**.

## 1. Arsitektur

```
Perangkat sensor (ESP32/dkk) ──MQTT──▶ Mosquitto (broker lokal, server ini)
                                              │
                                              ▼
                                    mqtt_subscriber.py (AKTIF)
                                    - parsing NULL-safe (T2)
                                    - hitung quality_flags per-sampel (T5)
                                    - tulis ke trafo_temperature, trafo_load
                                              │
                                              ▼
                                         PostgreSQL 18.4 polos
                                   (TimescaleDB TIDAK terpasang)
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ▼                     ▼                     ▼
                 lntai_worker.py       archive_job.py          api.py
              (baca temp+load,      (cron harian, salin      (REST read,
               pairing+ambient_at,   baris maintenance_       + 2 endpoint
               tulis lntai_results)  windows ke *_archive)     tulis: ambient-
                                                                logs, maintenance-
                                                                windows)
                                              │
                                              ▼
                                   Browser (Shield.html / index.html)
                            - fetch Open-Meteo sendiri, relay ke
                              POST /api/ambient-logs (Cabang B, T3)
                            - panel LNTAI: server dulu, browser fallback
```

**Proses yang SUDAH TIDAK AKTIF** (jangan dihidupkan ulang tanpa meninjau
kredensial & isinya — lihat catatan DEPRECATED di masing-masing file):
- `mqtt_ingester.py` — broker HiveMQ cloud lama, service dinonaktifkan di
  server, ditemukan bug restart-loop saat audit Fase 0.
- `migrate_firestore_to_timescale.py` — migrasi satu-kali Firestore→Postgres,
  sudah selesai (0 baris baru per-run sejak lama).

## 2. Konfigurasi (`settings.py`, `.env`)

Semua kredensial/host dibaca dari environment variable lewat `settings.py` —
**tidak ada default rahasia** di kode. Variabel wajib akan membuat proses
gagal start dengan pesan jelas kalau belum diisi.

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env   # isi DATABASE_URL, API_KEY, MQTT_*, GANDUL_LATITUDE/LONGITUDE, dst.
```

`python-dotenv` (ada di requirements.txt) memuat `.env` otomatis kalau
terpasang; kalau tidak, export manual env var sebelum menjalankan skrip.

**Catatan keamanan:** kredensial HiveMQ lama sempat ada di riwayat git
sebelum Tahap 1 — rotasi password itu di sisi HiveMQ kalau belum dilakukan,
terlepas dari status `mqtt_ingester.py` yang sekarang nonaktif.

## 3. Migrasi database

**Jangan pernah dijalankan otomatis oleh asisten/skrip apa pun — selalu
manual, setelah backup.** Semua file di `migrations/` idempotent (`IF NOT
EXISTS`), aman dijalankan ulang.

Urutan (jalankan sesuai nomor, bukan diacak):

```bash
# Backup dulu, SELALU, sebelum migrasi apa pun:
pg_dump "$DATABASE_URL" > backup_sebelum_migrasi_$(date +%Y%m%d).sql

psql "$DATABASE_URL" -f migrations/001_add_quality_flags.sql           # T2/T5: kolom quality_flags
psql "$DATABASE_URL" -f migrations/002_create_ambient_logs.sql         # T3: tabel ambient_logs
psql "$DATABASE_URL" -f migrations/003_create_worker_state.sql         # T4: checkpoint worker
psql "$DATABASE_URL" -f migrations/004_create_lntai_results.sql        # T4: hasil LNTAI server-side
psql "$DATABASE_URL" -f migrations/005_create_maintenance_windows.sql  # T6: jendela maintenance
psql "$DATABASE_URL" -f migrations/006_create_archive_tables.sql       # T6: tabel *_archive
# migrations/007_retention_policy_INACTIVE.sql TIDAK dijalankan -- lihat
# isi file itu, semua DELETE dikomentari, periode retensi adalah keputusan
# Anda, bukan nilai yang dikarang.
```

**Rollback** satu migrasi (dari yang TERAKHIR dijalankan ke belakang, bukan
sembarang urutan):

```bash
psql "$DATABASE_URL" -f migrations/006_create_archive_tables_down.sql
psql "$DATABASE_URL" -f migrations/005_create_maintenance_windows_down.sql
psql "$DATABASE_URL" -f migrations/004_create_lntai_results_down.sql
psql "$DATABASE_URL" -f migrations/003_create_worker_state_down.sql
psql "$DATABASE_URL" -f migrations/002_create_ambient_logs_down.sql
psql "$DATABASE_URL" -f migrations/001_add_quality_flags_down.sql
```

Beberapa rollback (002, 004, 006) menghapus data historis permanen — pesan
peringatan dan perintah `pg_dump` spesifik ada di komentar masing-masing
file `_down.sql`, baca sebelum menjalankan.

## 4. Jalankan sebagai systemd service (produksi)

```bash
sudo useradd -r -s /bin/false shield || true
sudo mkdir -p /opt/shield/server
sudo chown -R shield:shield /opt/shield
cp -r server/* /opt/shield/server/
# (venv dibuat ulang di server produksi, jangan copy venv/ dari laptop)

sudo cp server/shield-api.service server/shield-lntai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now shield-api shield-lntai
sudo systemctl status shield-api shield-lntai
```

`mqtt_subscriber.py` dijalankan sebagai service terpisah (lihat setup yang
sudah berjalan di server produksi saat ini — belum ada file `.service` baru
dibuat untuknya di Tahap 1 karena **tidak diubah perilaku proses-nya**,
hanya isi `save_to_database()`, lihat Aturan kerja #4).

`archive_job.py` **bukan** service long-running — jalankan sebagai cron
harian:

```bash
# crontab -e (user shield)
0 3 * * * cd /opt/shield/server && venv/bin/python3 archive_job.py >> /var/log/shield-archive.log 2>&1
```

## 5. Backfill LNTAI historis (sekali, setelah deploy T4 pertama kali)

`lntai_worker.py` hanya memproses data BARU (sejak checkpoint pertama kali
berjalan). Untuk histori sebelum itu:

```bash
source venv/bin/activate
export $(grep -v '^#' .env | xargs)

# Cek dulu tanpa menulis apa pun:
python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30 --dry-run

# Kalau hasilnya sesuai ekspektasi, jalankan sungguhan (idempotent, aman diulang):
python3 backfill_lntai.py --trafo 2 --from 2026-09-01 --to 2026-09-30
python3 backfill_lntai.py --trafo 3 --from 2026-09-01 --to 2026-09-30
python3 backfill_lntai.py --trafo 4 --from 2026-09-01 --to 2026-09-30
```

## 6. Sambungkan ke Shield.html / index.html

Konstanta `TIMESCALE_API_BASE_URL`/`TIMESCALE_API_KEY` di dalam file HTML
harus menunjuk ke API ini. Kalau diisi, browser otomatis: (a) membaca
log suhu/beban/LNTAI dari sini, (b) merelay hasil fetch Open-Meteo-nya
sendiri ke `POST /api/ambient-logs` (lihat §8).

## 7. Ekspos API ke internet lewat nginx + HTTPS

`shield-api.service` bind ke `127.0.0.1:8000`. Pasang reverse proxy nginx
dengan HTTPS (Let's Encrypt) di depannya:

```nginx
server {
    listen 443 ssl;
    server_name api.domain-anda.com;
    ssl_certificate     /etc/letsencrypt/live/api.domain-anda.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.domain-anda.com/privkey.pem;
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
```

## 8. Catatan arsitektur penting

- **TimescaleDB TIDAK terpasang.** Rencana retensi/continuous-aggregate awal
  (hypertable, retention policy otomatis) DIBATALKAN setelah audit Fase 0
  mengonfirmasi ini PostgreSQL polos. Diganti: `maintenance_windows` +
  `archive_job.py` (§4) + migrasi retensi yang sengaja TIDAK AKTIF
  (`007_retention_policy_INACTIVE.sql`).
- **Ambient (cuaca) ditulis BROWSER, bukan server (Cabang B, T3).** Server
  ini berada di belakang proxy keluar yang meng-intercept TLS (curl ke
  `api.open-meteo.com` gagal dengan *self-signed certificate in certificate
  chain*) — server tidak bisa memanggil Open-Meteo langsung. Konsekuensinya:
  histori `ambient_logs` hanya terisi selama ada dashboard terbuka di
  browser. Ini bukan bug; kalau tim IT memasang root CA korporat ke trust
  store server, `ambient_worker.py` (server-side) bisa ditambahkan nanti
  tanpa mengubah skema tabel.
- **Kolom `trafo_temperature.temp_ambient` BUKAN suhu cuaca.** Kolom itu
  selalu bernilai statis `30` di semua baris (dikonfirmasi audit) dan
  dicadangkan untuk sensor ambient DI DALAM panel trafo di masa depan.
  Suhu cuaca ada di tabel `ambient_logs` yang terpisah.
- **LNTAI: server dulu, browser fallback.** `lntai_worker.py` adalah sumber
  utama (`GET /api/lntai-latest`); kalkulasi di browser (`_lntaiStep` di
  index.html) hanya jalan kalau baris server tidak ada/basi (>10 menit)/
  `DATA_INVALID`. Badge "source" di panel LNTAI menunjukkan yang mana
  sedang aktif.
- **Data hilang = NULL, bukan 0.** `mqtt_subscriber.py` (T2) menulis NULL
  untuk field yang tidak dikirim/tidak valid; 0 yang memang dikirim sensor
  (mis. arus 0 saat CB terbuka) tetap tersimpan sebagai 0. API ini TIDAK
  lagi melakukan `COALESCE(..., 0)` — frontend yang menangani NULL.

## 9. Testing

```bash
source venv/bin/activate
pytest tests/ -v
```

**Catatan sandbox pengembangan (Oktober 2026):** semua test di atas ditulis
gaya pytest standar tapi diverifikasi MANUAL (stub-injection `paho`/
`psycopg_pool`, panggilan fungsi langsung tanpa `fastapi.testclient`) karena
sandbox tempat kode ini ditulis tidak punya akses internet untuk
`pip install pytest httpx`. Di server/mesin dengan internet, jalankan
`pytest tests/ -v` seperti biasa untuk re-verifikasi — hasilnya harus sama
(94–107 test lolos, bertambah sesuai sub-tugas terakhir yang digabung).

Test parity LNTAI Python↔JS (`lntai_calculator.py` vs blok `<LNTAI-CORE>`
di index.html):

```bash
cd ..   # ke root proyek
python3 test_lntai.py              # 21 test
node tests/test_lntai_js.mjs        # butuh Node.js (tidak tersedia di sandbox ini)
```

## 10. Variabel lingkungan (ringkasan — lihat `.env.example` untuk contoh)

| Variabel | Dipakai oleh | Wajib? |
|---|---|---|
| `DATABASE_URL` | semua skrip yang akses DB | **Wajib** |
| `API_KEY` | `api.py` (opsional — kalau kosong, auth dimatikan) | Opsional |
| `ALLOWED_ORIGINS` | `api.py` (CORS) | Opsional |
| `MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC`, `MQTT_LOCAL_USER`, `MQTT_LOCAL_PASS` | `mqtt_subscriber.py` | Opsional (ada default broker lokal) |
| `HIVEMQ_BROKER`, `HIVEMQ_PORT`, `HIVEMQ_USER`, `HIVEMQ_PASS` | `mqtt_ingester.py` (DEPRECATED) | Wajib HANYA kalau skrip itu dijalankan lagi |
| `GANDUL_LATITUDE`, `GANDUL_LONGITUDE` | `ambient.py`, `lntai_worker.py` (default ambient) | Opsional (default GI Gandul) |
| `AMBIENT_WORKER_INTERVAL_SECONDS`, `LNTAI_WORKER_INTERVAL_SECONDS`, `WRITE_INTERVAL_SECONDS` | worker settings | Opsional |
| `OUTBOUND_HTTPS_PROXY`, `OUTBOUND_CA_BUNDLE` | disiapkan untuk proxy keluar (lihat §8 ambient) | Opsional |
