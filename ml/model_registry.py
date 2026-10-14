"""ml/model_registry.py — SHIELD Stage 2, Part B.

Memuat model AKTIF (ml_models.is_active=true) untuk satu (trafo, phase)
dari database + artefak di disk, dipakai lntai_worker.py untuk mengganti
baseline fisika legacy LNTAI v3 dengan model adaptif yang sudah dilatih --
TANPA model aktif, selalu fallback ke fisika legacy (lihat
lntai_calculator.lntai_v3_step() baseline_overrides, Part A: "ΔT_pred,p =
baseline_t,p(features) (until Part B: legacy physics ...)").

Cache in-memory per (trafo, phase) dengan TTL pendek -- supaya worker
tidak query DB di SETIAP sampel (interval worker jauh lebih cepat dari
TTL cache ini). CATATAN PENTING: api.py (endpoint activate/freeze) dan
lntai_worker.py berjalan sebagai PROSES TERPISAH (dua systemd service
berbeda) -- keduanya TIDAK berbagi memori Python, jadi
POST /api/ml/models/{id}/activate TIDAK BISA langsung memanggil
invalidate_cache() di proses worker. Aktivasi tetap "switches the
worker's model" seperti diminta Stage 2 Part B, tapi dengan jeda
maksimal ACTIVE_MODEL_CACHE_TTL_SECONDS (bukan instan) -- worker membaca
ulang ml_models dari DB sendiri setiap kali cache-nya kedaluwarsa.
invalidate_cache() di modul ini berguna untuk test (satu proses) dan
untuk pemakaian in-process lain, BUKAN untuk dipanggil dari api.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models import load_model_from_artifact  # noqa: E402

ACTIVE_MODEL_CACHE_TTL_SECONDS = 30.0
_ARTIFACTS_ROOT = _ROOT  # artifact_path di ml_models relatif ke root proyek (mis. "ml/artifacts/...")

class _CacheEntry:
    __slots__ = ("loaded_at", "model", "metadata", "step_state")

    def __init__(self, loaded_at: float, model: Optional[Any], metadata: Optional[dict[str, Any]]):
        self.loaded_at = loaded_at
        self.model = model
        self.metadata = metadata
        # Dibuat FRESH setiap kali entry ini dibuat (mis. model baru dimuat
        # setelah aktivasi berganti) -- state predict_step() TIDAK PERNAH
        # dibawa dari model LAMA ke model BARU (tau/bobot bisa beda total).
        self.step_state: dict[str, Any] = {}


_cache: dict[tuple[str, str], _CacheEntry] = {}


def _fetch_active_row(cur, trafo: str, phase: str) -> Optional[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, trafo, phase, model_type, version, artifact_path, sigma, mu
        FROM ml_models
        WHERE trafo = %s AND phase = %s AND is_active = TRUE
        LIMIT 1
        """,
        (trafo, phase),
    )
    return cur.fetchone()


def _load_entry(cur, trafo: str, phase: str, now: float) -> _CacheEntry:
    row = _fetch_active_row(cur, trafo, phase)
    if row is None:
        return _CacheEntry(now, None, None)

    artifact_path = _ARTIFACTS_ROOT / row["artifact_path"]
    with open(artifact_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    model = load_model_from_artifact(row["model_type"], artifact)
    metadata = {"id": row["id"], "model_type": row["model_type"], "version": row["version"],
                "sigma": row["sigma"], "mu": row["mu"]}
    return _CacheEntry(now, model, metadata)


def load_active_model(cur, trafo: str, phase: str, now: Optional[float] = None) -> tuple[Optional[Any], Optional[dict[str, Any]]]:
    """Return (model_instance, row_metadata) kalau ada model aktif untuk
    (trafo, phase), atau (None, None) kalau tidak ada -- caller (lntai_worker)
    HARUS menangani None sebagai sinyal "pakai fisika legacy" (lihat
    Stage 2 Part B test requirement "Fallback works with no active model").
    `cur` dipakai hanya saat cache kedaluwarsa (lihat ACTIVE_MODEL_CACHE_TTL_SECONDS)."""
    now = time.monotonic() if now is None else now
    key = (str(trafo), str(phase))
    entry = _cache.get(key)
    if entry is None or (now - entry.loaded_at) >= ACTIVE_MODEL_CACHE_TTL_SECONDS:
        entry = _load_entry(cur, trafo, phase, now)
        _cache[key] = entry
    return entry.model, entry.metadata


def invalidate_cache(trafo: Optional[str] = None, phase: Optional[str] = None) -> None:
    """Paksa load_active_model() membaca ulang dari DB (dan reset
    predict_step() state) pada panggilan berikutnya -- dipanggil oleh
    POST /api/ml/models/{id}/activate supaya aktivasi tidak menunggu TTL
    cache habis ("Activation switches the worker's model" dalam satu
    panggilan berikutnya, bukan sampai 30 detik lagi)."""
    if trafo is None or phase is None:
        _cache.clear()
    else:
        _cache.pop((str(trafo), str(phase)), None)


def predict_delta_t_step(cur, trafo: str, phase: str, row: dict[str, Any], ambient_c_v3: float,
                          now: Optional[float] = None) -> tuple[Optional[float], Optional[dict[str, Any]]]:
    """Satu langkah prediksi ΔT_pred (kenaikan suhu, BUKAN suhu absolut)
    dari model AKTIF untuk (trafo, phase) -- STREAMING (state filter I^2
    milik model, kalau ada, bertahan antar panggilan, lihat ml/models.py
    predict_step() tiap kelas). Return (delta_t_pred, model_metadata):
    delta_t_pred=None (model_metadata=None juga) berarti TIDAK ADA model
    aktif -- caller (lntai_worker.py) HARUS fallback ke fisika legacy v3.

    `row` berisi fitur TANPA "ambient_c" (ts_epoch, current_amps,
    i2_eff_default_tau, load_roc_1h, hour_sin, hour_cos, weekday, dst.) --
    `ambient_c_v3` (konstanta legacy v3, config.LNTAI_V3_AMBIENT_C) di-set
    SEBAGAI "ambient_c" di sini SEBELUM dipanggil ke model.predict_step()
    (yang mengembalikan SUHU ABSOLUT = formula + ambient_c), lalu
    dikurangi balik supaya hasil akhirnya ΔT_pred relatif terhadap
    ambient_c yang SAMA dipakai lntai_v3_step() untuk ΔT_real -- model
    mana pun yang aktif (dilatih dengan ambient measured ATAU default,
    lihat ml/dataset.py) tetap menghasilkan ΔT_pred yang konsisten
    dijumlahkan terhadap ambient_c_v3 yang tetap, bukan ambient yang
    dipakai saat training (lihat docstring modul untuk alasan lengkap)."""
    now = time.monotonic() if now is None else now
    key = (str(trafo), str(phase))
    entry = _cache.get(key)
    if entry is None or (now - entry.loaded_at) >= ACTIVE_MODEL_CACHE_TTL_SECONDS:
        entry = _load_entry(cur, trafo, phase, now)
        _cache[key] = entry

    if entry.model is None:
        return None, None

    row_with_ambient = {**row, "ambient_c": ambient_c_v3}
    new_state, absolute_pred = entry.model.predict_step(entry.step_state, row_with_ambient)
    entry.step_state = new_state
    if absolute_pred is None:
        return None, entry.metadata
    delta_t_pred = absolute_pred - ambient_c_v3
    return delta_t_pred, entry.metadata
