"""Test LNTAI v3 (SHIELD Stage 2, Part A) -- lntai_calculator.lntai_v3_step()
and LNTAICalculator.update_v3()/calculate_series_v3(). Pure functions, no
DB/network. Companion to test_lntai.py (v2, untouched by this file).

Run: python3 test_lntai_v3.py
"""

from __future__ import annotations

import unittest

import config
from lntai_calculator import LNTAICalculator, PHASES, lntai_v3_step

V3_KWARGS = dict(
    a=config.LNTAI_V3_A, b=config.LNTAI_V3_B, sigma=config.LNTAI_V3_SIGMA, mu=config.LNTAI_V3_MU,
    i_ref=config.LNTAI_V3_I_REF, i_rated_v2=1780.0, ambient_c=config.LNTAI_V3_AMBIENT_C,
    rolling_median_minutes=config.LNTAI_V3_ROLLING_MEDIAN_MINUTES, thresholds=config.LNTAI_V3_THRESHOLDS,
    baseline_id=config.LNTAI_V3_BASELINE_ID, calibrated=config.LNTAI_V3_CALIBRATED,
    model_version=config.LNTAI_V3_MODEL_VERSION,
)


def _fake_v2_result(temps: dict[str, float], i_eff_norm: float = 0.0, level: str = "NORMAL",
                     ts_iso: str = "2026-10-14T10:00:00+00:00") -> dict:
    per_phase = {p: {"T": temps[p], "i_eff_norm": i_eff_norm} for p in PHASES}
    return {"level": level, "per_phase": per_phase, "timestamp": ts_iso, "error": None}


def _temp_for_z(z_target: float) -> float:
    """T sedemikian hingga z = z_target, dengan i_eff_norm=0 (delta_T_pred=B)."""
    return config.LNTAI_V3_AMBIENT_C + config.LNTAI_V3_B + z_target * config.LNTAI_V3_SIGMA


class TestDataInvalidPassthrough(unittest.TestCase):
    def test_v2_data_invalid_produces_v3_data_invalid(self):
        v2_invalid = {"level": "DATA_INVALID", "per_phase": None, "timestamp": "t0", "error": "no data"}
        new_state, result = lntai_v3_step({}, v2_invalid, 1000.0, **V3_KWARGS)
        self.assertEqual(result["level"], "DATA_INVALID")
        self.assertIsNone(result["lntai"])
        self.assertEqual(new_state["rolling_buffer"], [])

    def test_missing_per_phase_also_treated_invalid(self):
        v2_bad = {"level": "NORMAL", "per_phase": None, "timestamp": "t0", "error": None}
        _, result = lntai_v3_step({}, v2_bad, 1000.0, **V3_KWARGS)
        self.assertEqual(result["level"], "DATA_INVALID")


class TestPerPhaseRescaling(unittest.TestCase):
    def test_zero_current_predicts_B_only(self):
        """i_eff_norm=0 (v2 basis) -> i_eff_norm_legacy=0 -> delta_T_pred == B."""
        v2 = _fake_v2_result({"R": 60.0, "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        _, result = lntai_v3_step({}, v2, 0.0, **V3_KWARGS)
        self.assertAlmostEqual(result["per_phase"]["R"]["delta_T_pred"], config.LNTAI_V3_B)

    def test_rescaling_from_1780_basis_to_1000_basis(self):
        """i_eff_norm=0.5 (basis 1780A) -> I_eff=890A -> basis 1000A -> 0.89."""
        v2 = _fake_v2_result({"R": 60.0, "S": 20.0, "T": 20.0}, i_eff_norm=0.5)
        _, result = lntai_v3_step({}, v2, 0.0, **V3_KWARGS)
        expected_pred = config.LNTAI_V3_A * (0.89 ** 2) + config.LNTAI_V3_B
        self.assertAlmostEqual(result["per_phase"]["R"]["delta_T_pred"], expected_pred, places=6)

    def test_hottest_phase_by_z_not_by_raw_temperature(self):
        """Fasa dengan z tertinggi bisa BEDA dari fasa bersuhu tertinggi kalau
        i_eff per fasa berbeda -- v3 memilih berdasar z, bukan T mentah."""
        per_phase = {
            "R": {"T": 70.0, "i_eff_norm": 0.6},   # suhu tertinggi, tapi arus juga tinggi -> z moderat
            "S": {"T": 50.0, "i_eff_norm": 0.0},   # suhu rendah TAPI arus nyaris nol -> residual tinggi jika dT_pred rendah
            "T": {"T": 50.0, "i_eff_norm": 0.0},
        }
        v2 = {"level": "NORMAL", "per_phase": per_phase, "timestamp": "t0", "error": None}
        _, result = lntai_v3_step({}, v2, 0.0, **V3_KWARGS)
        # Pastikan hottest_phase dipilih dari max(z), konsisten dengan nilai z yang dihitung
        z_values = {p: result["per_phase"][p]["z"] for p in PHASES}
        self.assertEqual(result["hottest_phase"], max(z_values, key=z_values.get))


class TestLevelBoundaries(unittest.TestCase):
    def _level_for_z(self, z_target: float) -> str:
        v2 = _fake_v2_result({"R": _temp_for_z(z_target), "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        _, result = lntai_v3_step({}, v2, 2000.0, **V3_KWARGS)
        return result["level"]

    def test_just_below_caution_is_normal(self):
        self.assertEqual(self._level_for_z(1.499), "Normal")

    def test_exactly_caution_boundary(self):
        self.assertEqual(self._level_for_z(1.5), "Caution")

    def test_just_below_alert_is_caution(self):
        self.assertEqual(self._level_for_z(2.999), "Caution")

    def test_exactly_alert_boundary(self):
        self.assertEqual(self._level_for_z(3.0), "Alert")

    def test_just_below_critical_is_alert(self):
        self.assertEqual(self._level_for_z(4.999), "Alert")

    def test_exactly_critical_boundary(self):
        self.assertEqual(self._level_for_z(5.0), "Critical")

    def test_zero_is_normal(self):
        self.assertEqual(self._level_for_z(0.0), "Normal")


class TestRollingMedian(unittest.TestCase):
    def test_median_of_single_sample_equals_instant(self):
        v2 = _fake_v2_result({"R": _temp_for_z(4.0), "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        _, result = lntai_v3_step({}, v2, 0.0, **V3_KWARGS)
        self.assertAlmostEqual(result["lntai"], result["lntai_rolling_median_30min"])

    def test_median_lags_behind_sudden_spike(self):
        """Satu sampel instan tinggi TIDAK langsung menaikkan level -- median
        gabungan dengan histori sebelumnya yang rendah."""
        state = {}
        low = _fake_v2_result({"R": _temp_for_z(0.0), "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        for minute in (0, 5, 10):
            state, _ = lntai_v3_step(state, low, minute * 60.0, **V3_KWARGS)
        high = _fake_v2_result({"R": _temp_for_z(7.0), "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        state, result = lntai_v3_step(state, high, 15 * 60.0, **V3_KWARGS)
        self.assertGreater(result["lntai"], 5.0)  # instan sudah CRITICAL
        self.assertLess(result["lntai_rolling_median_30min"], 1.5)  # median masih Normal
        self.assertEqual(result["level"], "Normal")

    def test_samples_older_than_30min_are_pruned(self):
        state = {}
        zero = _fake_v2_result({"R": _temp_for_z(0.0), "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        state, _ = lntai_v3_step(state, zero, 0.0, **V3_KWARGS)  # t=0min, akan dibuang nanti
        four = _fake_v2_result({"R": _temp_for_z(4.0), "S": 20.0, "T": 20.0}, i_eff_norm=0.0)
        for minute in (5, 10, 15, 20, 25, 30):
            state, _ = lntai_v3_step(state, four, minute * 60.0, **V3_KWARGS)
        # Pada t=35min, cutoff=5min -> sampel t=0 (z=0) harus terbuang,
        # median murni dari nilai z=4.0 semua.
        state, result = lntai_v3_step(state, four, 35 * 60.0, **V3_KWARGS)
        self.assertAlmostEqual(result["lntai_rolling_median_30min"], 4.0, places=6)


class TestBaselineFlags(unittest.TestCase):
    def test_baseline_id_and_calibrated_present_on_valid_result(self):
        v2 = _fake_v2_result({"R": 55.0, "S": 50.0, "T": 50.0}, i_eff_norm=0.2)
        _, result = lntai_v3_step({}, v2, 0.0, **V3_KWARGS)
        self.assertEqual(result["baseline_id"], "legacy-v1-physics-uncalibrated")
        self.assertFalse(result["calibrated"])

    def test_config_flagged_uncalibrated(self):
        self.assertFalse(config.LNTAI_V3_CALIBRATED)
        self.assertEqual(config.LNTAI_V3_BASELINE_ID, "legacy-v1-physics-uncalibrated")


class TestCalculatorIntegration(unittest.TestCase):
    """LNTAICalculator.update()/update_v3() -- v2 TIDAK berubah oleh adanya v3."""

    def test_v2_still_produced_unchanged(self):
        calc = LNTAICalculator()
        sample = {
            "R_Atas": 55, "R_Bawah": 54, "S_Atas": 56, "S_Bawah": 53, "T_Atas": 57, "T_Bawah": 52,
            "I_R": 800, "I_S": 800, "I_T": 800, "timestamp": 0.0,
        }
        v2_result = calc.update("2", sample)
        self.assertIn(v2_result["level"], ("NORMAL", "WARNING", "CRITICAL"))
        self.assertIn("S_Temp", v2_result)  # field v2 khas, harus tetap ada

    def test_update_v3_uses_separate_state_from_v2(self):
        calc = LNTAICalculator()
        sample = {
            "R_Atas": 55, "R_Bawah": 54, "S_Atas": 56, "S_Bawah": 53, "T_Atas": 57, "T_Bawah": 52,
            "I_R": 800, "I_S": 800, "I_T": 800, "timestamp": 0.0,
        }
        v2_result = calc.update("2", sample)
        v3_result = calc.update_v3("2", v2_result, 0.0)
        self.assertIn(v3_result["level"], ("Normal", "Caution", "Alert", "Critical"))
        self.assertIn("2", calc._state)       # state v2 terisi
        self.assertIn("2", calc._state_v3)    # state v3 terisi TERPISAH

    def test_reset_clears_both_v2_and_v3_state(self):
        calc = LNTAICalculator()
        sample = {
            "R_Atas": 55, "R_Bawah": 54, "S_Atas": 56, "S_Bawah": 53, "T_Atas": 57, "T_Bawah": 52,
            "I_R": 800, "I_S": 800, "I_T": 800, "timestamp": 0.0,
        }
        v2_result = calc.update("2", sample)
        calc.update_v3("2", v2_result, 0.0)
        calc.reset("2")
        self.assertNotIn("2", calc._state)
        self.assertNotIn("2", calc._state_v3)

    def test_calculate_series_v3_uses_local_state_not_live_state(self):
        calc = LNTAICalculator()
        sample = {
            "R_Atas": 55, "R_Bawah": 54, "S_Atas": 56, "S_Bawah": 53, "T_Atas": 57, "T_Bawah": 52,
            "I_R": 800, "I_S": 800, "I_T": 800,
        }
        rows = [{**sample, "timestamp": float(i * 300)} for i in range(3)]
        v2_series = calc.calculate_series(rows)
        pairs = list(zip(v2_series, (r["timestamp"] for r in rows)))
        v3_series = calc.calculate_series_v3(pairs)
        self.assertEqual(len(v3_series), 3)
        self.assertEqual(calc._state_v3, {})  # tidak mencemari state live


if __name__ == "__main__":
    unittest.main(verbosity=2)
