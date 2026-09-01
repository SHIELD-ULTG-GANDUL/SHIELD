# SHIELD — Backend TimescaleDB (Ubuntu server)

Menggantikan Firestore sebagai penyimpan log historis suhu. Terdiri dari dua
proses independen:

- **`mqtt_ingester.py`** — berlangganan broker MQTT yang sama dengan dashboard
  (topik `Suhu/Trafo/2|3|4`) dan menulis tiap pembacaan ke TimescaleDB. Jalan
  terus di server, tidak tergantung ada-tidaknya browser yang membuka
  dashboard — beda dengan pendekatan lama yang menulis dari dalam browser.
- **`api.py`** — REST API baca-saja (`GET /api/temperature-logs`) yang
  dipanggil Shield.html dari browser, karena browser tidak bisa konek
  langsung ke PostgreSQL/TimescaleDB.

## 1. Instalasi TimescaleDB (Ubuntu)

Ikuti panduan resmi: https://docs.timescale.com/self-hosted/latest/install/installation-linux/
Ringkasnya untuk Ubuntu 22.04/24.04:

```bash
sudo apt install -y gnupg postgresql-common apt-transport-https lsb-release
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update
sudo apt install -y timescaledb-2-postgresql-16 postgresql-client-16
sudo timescaledb-tune --quiet --yes
sudo systemctl restart postgresql
```

Buat database & user:

```bash
sudo -u postgres psql -c "CREATE DATABASE shield;"
sudo -u postgres psql -c "CREATE USER shield WITH PASSWORD 'ganti-password-ini';"
sudo -u postgres psql -d shield -c "GRANT ALL PRIVILEGES ON DATABASE shield TO shield;"
sudo -u postgres psql -d shield -f schema.sql
```

## 2. Siapkan aplikasi Python

```bash
sudo mkdir -p /opt/shield/server
sudo chown $USER:$USER /opt/shield/server
cp server/*.py server/schema.sql server/requirements.txt server/.env.example /opt/shield/server/
cd /opt/shield/server

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env   # isi DATABASE_URL dengan password yang tadi dibuat, generate API_KEY, dst.
```

Tes manual dulu sebelum dijadikan service:

```bash
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
python mqtt_ingester.py        # biarkan jalan, cek log "Tersimpan: Trafo 2 @ ..."
# di terminal lain:
uvicorn api:app --host 0.0.0.0 --port 8000
curl "http://localhost:8000/api/temperature-logs?trafo=2&limit=5"
```

## 3. Jalankan sebagai systemd service (produksi)

```bash
sudo useradd -r -s /bin/false shield || true
sudo chown -R shield:shield /opt/shield

sudo cp server/shield-ingester.service server/shield-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now shield-ingester shield-api
sudo systemctl status shield-ingester shield-api
```

## 4. Ekspos API ke internet lewat nginx + HTTPS

`shield-api.service` sengaja hanya bind ke `127.0.0.1:8000` (tidak langsung
ke publik). Pasang nginx sebagai reverse proxy dengan HTTPS (Let's Encrypt),
supaya trafik browser ke API terenkripsi:

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

(`sudo apt install certbot python3-certbot-nginx && sudo certbot --nginx` untuk
mengurus sertifikat otomatis.)

## 5. Sambungkan ke Shield.html

Di `Shield.html`, cari konstanta `TIMESCALE_API_BASE_URL` (dekat konfigurasi
MQTT_BROKER) dan isi dengan URL publik API Anda, misalnya:

```js
const TIMESCALE_API_BASE_URL = "https://api.domain-anda.com/api";
const TIMESCALE_API_KEY = "isi-sama-dengan-API_KEY-di-.env-kalau-diisi";
```

Selama `TIMESCALE_API_BASE_URL` masih kosong/placeholder, dashboard otomatis
tetap memakai Firestore seperti sebelumnya — jadi aman diisi kapan saja
tanpa perlu koordinasi waktu deploy.

## Catatan retensi & ukuran data

`schema.sql` sudah menyertakan retention policy (hapus otomatis data > 180
hari) dan continuous aggregate per jam, supaya storage tidak membengkak
tanpa batas dan query histori jangka panjang tetap cepat. Sesuaikan interval
retensi di `schema.sql` kalau kebutuhan Anda berbeda.
