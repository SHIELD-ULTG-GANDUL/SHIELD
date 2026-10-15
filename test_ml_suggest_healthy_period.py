"""Test SHIELD go-live — ml/suggest_healthy_period.py (window ranking,
no DB -- windows/pairing built in-memory). Tanpa DB/network.

Run: python3 test_ml_suggest_healthy_period.py
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "ml"))
sys.path.insert(0, str(_ROOT / "server"))
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
import data_quality as dq  # noqa: E402
import suggest_healthy_period as shp  # noqa: E402


def _clean_rows(n_days: int, base: datetime, current: float = 800.0, noise_sd: float = 0.0, seed: int = 1):
    random.seed(seed)
    i_norm = current / config.LNTAI_V3_I_REF
    dT = config.LNTAI_V3_A * i_norm ** 2 + config.LNTAI_V3_B
    temp_rows, load_rows = [], []
    for i in range(n_days * 288):
        ts = base + timedelta(minutes=5 * i)
        hotspot = config.LNTAI_V3_AMBIENT_C + dT + random.gauss(0, noise_sd)
        temp_rows.append({"timestamp": ts, "r_atas": hotspot, "r_bawah": hotspot - 1.0, "quality_flags": 0})
        load_rows.append({"timestamp": ts, "ia": current, "quality_flags": 0})
    return temp_rows, load_rows


class TestHotspotC(unittest.TestCase):
    def test_takes_max_of_atas_bawah(self):
        row = {"r_atas": 50.0, "r_bawah": 55.0}
        self.assertEqual(shp._hotspot_c(row, "R"), 55.0)

    def test_none_when_both_missing(self):
        self.assertIsNone(shp._hotspot_c({"r_atas": None, "r_bawah": None}, "R"))

    def test_uses_whichever_is_present(self):
        self.assertEqual(shp._hotspot_c({"r_atas": None, "r_bawah": 40.0}, "R"), 40.0)


class TestMad(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(shp._mad([]))

    def test_constant_series_is_zero(self):
        self.assertEqual(shp._mad([5.0] * 10), 0.0)

    def test_known_mad(self):
        # median=3, deviations=[2,1,0,1,2], median of those = 1
        self.assertEqual(shp._mad([1.0, 2.0, 3.0, 4.0, 5.0]), 1.0)


class TestWindowMetrics(unittest.TestCase):
    def test_clean_full_data_has_near_perfect_metrics(self):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        temp_rows, load_rows = _clean_rows(7, base, current=800.0, noise_sd=0.0)
        paired = shp.ds._pair_nearest(temp_rows, load_rows, config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES)
        m = shp._window_metrics(temp_rows, paired, "R", 7, overall_max_current=800.0)
        self.assertGreaterEqual(m["completeness"], 0.95)
        self.assertEqual(m["flagged_share"], 0.0)
        self.assertAlmostEqual(m["residual_mad_c"], 0.0, places=6)
        self.assertEqual(m["load_range_coverage"], 1.0)
        self.assertAlmostEqual(m["composite_score"], 1.0, places=2)

    def test_flagged_samples_reduce_flagged_share_proportionally(self):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        temp_rows, load_rows = _clean_rows(7, base, current=800.0, noise_sd=0.0)
        for i in range(0, len(temp_rows), 10):
            temp_rows[i]["quality_flags"] = dq.flags_to_bitmask({"STUCK"})
        paired = shp.ds._pair_nearest(temp_rows, load_rows, config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES)
        m = shp._window_metrics(temp_rows, paired, "R", 7, overall_max_current=800.0)
        self.assertAlmostEqual(m["flagged_share"], 0.1, delta=0.01)

    def test_low_peak_vs_history_reduces_load_range_coverage(self):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        temp_rows, load_rows = _clean_rows(7, base, current=800.0, noise_sd=0.0)
        paired = shp.ds._pair_nearest(temp_rows, load_rows, config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES)
        m = shp._window_metrics(temp_rows, paired, "R", 7, overall_max_current=1780.0)
        self.assertLess(m["load_range_coverage"], 1.0)
        self.assertAlmostEqual(m["load_range_coverage"], 800.0 / 1780.0, places=6)

    def test_noisy_residual_increases_mad_and_lowers_stability(self):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        clean_rows, load_rows = _clean_rows(7, base, current=800.0, noise_sd=0.0)
        noisy_rows, _ = _clean_rows(7, base, current=800.0, noise_sd=3.0, seed=7)
        paired_clean = shp.ds._pair_nearest(clean_rows, load_rows, config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES)
        paired_noisy = shp.ds._pair_nearest(noisy_rows, load_rows, config.LNTAI_V2_PAIRING_TOLERANCE_MINUTES)
        m_clean = shp._window_metrics(clean_rows, paired_clean, "R", 7, overall_max_current=800.0)
        m_noisy = shp._window_metrics(noisy_rows, paired_noisy, "R", 7, overall_max_current=800.0)
        self.assertGreater(m_noisy["residual_mad_c"], m_clean["residual_mad_c"])
        self.assertLess(m_noisy["residual_stability"], m_clean["residual_stability"])
        self.assertLess(m_noisy["composite_score"], m_clean["composite_score"])

    def test_empty_window_is_worst_case_not_crash(self):
        m = shp._window_metrics([], [], "R", 7, overall_max_current=1780.0)
        self.assertEqual(m["completeness"], 0.0)
        self.assertEqual(m["flagged_share"], 1.0)
        self.assertEqual(m["load_range_coverage"], 0.0)
        self.assertIsNone(m["residual_mad_c"])
        self.assertEqual(m["residual_stability"], 0.0)


class TestSuggestForPhaseRanking(unittest.TestCase):
    def test_top_candidates_sorted_descending_by_composite_score(self):
        """Fake cursor nge-return seri 56 hari: 28 hari pertama bersih,
        28 hari berikutnya penuh sampel STUCK -- jendela 28-hari di
        paruh bersih harus rank lebih tinggi."""
        base = datetime(2026, 6, 2, tzinfo=timezone.utc)
        clean_rows, load_rows = _clean_rows(28, base, current=800.0, noise_sd=0.1, seed=3)
        dirty_rows, dirty_load = _clean_rows(28, base + timedelta(days=28), current=800.0, noise_sd=0.1, seed=4)
        for r in dirty_rows:
            r["quality_flags"] = dq.flags_to_bitmask({"STUCK"})
        all_temp = clean_rows + dirty_rows
        all_load = load_rows + dirty_load

        class _FakeCursor:
            def execute(self, *a, **k):
                self._last_sql = a[0]

            def fetchall(self):
                if "trafo_temperature" in self._last_sql:
                    return all_temp
                return all_load

        cur = _FakeCursor()
        until = base + timedelta(days=56)
        candidates = shp.suggest_for_phase(cur, "2", "R", base, until, window_days=28, step_days=28)
        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0]["composite_score"], candidates[1]["composite_score"])
        self.assertEqual(candidates[0]["healthy_from"], base.date().isoformat())

    def test_no_windows_fit_returns_empty_list(self):
        class _FakeCursor:
            def execute(self, *a, **k):
                pass

            def fetchall(self):
                return []

        base = datetime(2026, 6, 2, tzinfo=timezone.utc)
        candidates = shp.suggest_for_phase(_FakeCursor(), "2", "R", base, base + timedelta(days=5), window_days=28, step_days=7)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
