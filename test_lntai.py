"""Unit test LNTAI v2.

Jalankan:
    python test_lntai.py

Memakai lntai_test_vectors.json sebagai golden vectors — vector yang sama
dipakai ulang oleh tests/test_lntai_js.mjs untuk memverifikasi implementasi
JS di Shield.html/index.html menghasilkan angka yang identik (parity).
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from lntai_calculator import LNTAICalculator, arrhenius_score

VECTORS_PATH = Path(__file__).resolve().parent / "lntai_test_vectors.json"
FLOAT_FIELDS = (
    "lntai", "S_Temp", "S_DeltaT", "S_RES", "t_contact", "t_ambient",
    "delta_T_real", "delta_T_pred", "residual", "z", "i_eff_norm",
    "delta_phase", "dAB_max",
)


def _run_vector(vector: dict) -> dict:
    calc = LNTAICalculator()
    result = None
    for step in vector["steps"]:
        result = calc.update("2", step)
    return result


class TestLNTAIGoldenVectors(unittest.TestCase):
    """Replay setiap golden vector dan bandingkan hasil runtime terhadap
    'expected' yang tersimpan di lntai_test_vectors.json. Ini juga berfungsi
    sebagai regression test: kalau lntai_calculator.py berubah tanpa sengaja
    mengubah hasil, test ini gagal."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(VECTORS_PATH, encoding="utf-8") as f:
            cls.data = json.load(f)
        cls.vectors = {v["id"]: v for v in cls.data["vectors"]}

    def _assert_matches_expected(self, vector_id: str) -> dict:
        vector = self.vectors[vector_id]
        actual = _run_vector(vector)
        expected = vector["expected"]
        self.assertEqual(actual["level"], expected["level"], f"{vector_id}: level mismatch")
        if expected["level"] != "DATA_INVALID":
            for field in FLOAT_FIELDS:
                self.assertIsNotNone(actual[field], f"{vector_id}: {field} should not be None")
                self.assertAlmostEqual(
                    actual[field], expected[field], places=6,
                    msg=f"{vector_id}: field '{field}' mismatch",
                )
            self.assertEqual(actual["hottest_phase"], expected["hottest_phase"], f"{vector_id}: hottest_phase mismatch")
            self.assertEqual(actual["ambient_source"], expected["ambient_source"], f"{vector_id}: ambient_source mismatch")
        return actual

    def test_all_vectors_match_golden_output(self) -> None:
        for vector_id in self.vectors:
            with self.subTest(vector=vector_id):
                self._assert_matches_expected(vector_id)

    # ── Assertion kualitatif tambahan per skenario (dokumentasi intent, bukan hanya replay) ──

    def test_regresi_bug_v1_beban_rendah_suhu_normal(self) -> None:
        actual = self._assert_matches_expected("low_load_normal_temp_regression")
        self.assertEqual(actual["level"], "NORMAL")

    def test_full_load_high_temp_lebih_tinggi_dari_normal(self) -> None:
        normal = _run_vector(self.vectors["normal_baseline"])
        full_load = self._assert_matches_expected("full_load_high_temp")
        self.assertGreater(full_load["lntai"], normal["lntai"])

    def test_single_point_hotspot_terdeteksi_di_fasa_yang_benar(self) -> None:
        actual = self._assert_matches_expected("single_point_hotspot")
        self.assertEqual(actual["hottest_phase"], "R")
        self.assertGreater(actual["dAB_max"], 10)

    def test_phase_imbalance_menaikkan_s_delta_t(self) -> None:
        actual = self._assert_matches_expected("phase_imbalance")
        self.assertGreater(actual["S_DeltaT"], 5)

    def test_zero_current_delta_t_pred_sama_dengan_b(self) -> None:
        actual = self._assert_matches_expected("zero_current")
        import config
        self.assertAlmostEqual(actual["delta_T_pred"], config.LNTAI_V2_B, places=6)

    def test_invalid_data_tidak_mengandung_angka(self) -> None:
        actual = self._assert_matches_expected("invalid_data")
        self.assertEqual(actual["level"], "DATA_INVALID")
        self.assertIsNone(actual["lntai"])
        self.assertIn("error", actual)

    def test_ambient_measured_dipakai_apa_adanya(self) -> None:
        actual = self._assert_matches_expected("ambient_measured")
        self.assertEqual(actual["ambient_source"], "measured")
        self.assertAlmostEqual(actual["t_ambient"], 34.5, places=6)

    def test_ambient_default_fallback_dipakai_saat_tidak_ada_pengukuran(self) -> None:
        actual = self._assert_matches_expected("ambient_default_fallback")
        import config
        self.assertEqual(actual["ambient_source"], "default")
        self.assertAlmostEqual(actual["t_ambient"], config.LNTAI_V2_AMBIENT_DEFAULT, places=6)

    def test_gap_data_mereset_filter_lag_termal(self) -> None:
        actual = self._assert_matches_expected("gap_reset_filter")
        # Sampel ke-2: arus 200A setelah gap 120 menit (> max_gap_minutes) dari
        # sampel ke-1 yang 1780A -- filter HARUS reset, bukan tercampur histori lama.
        self.assertAlmostEqual(actual["i_eff_norm"], 200.0 / 1780.0, places=4)

    def test_lonjakan_beban_mendadak_diredam_filter_lag(self) -> None:
        actual = self._assert_matches_expected("sudden_load_increase_lag")
        # Arus sesaat sampel ke-2 = rated (i_norm=1.0), tapi i_eff_norm harus jauh
        # di bawah itu karena baru 5 menit berlalu (tau=20 menit).
        self.assertLess(actual["i_eff_norm"], 0.7)
        self.assertGreater(actual["i_eff_norm"], 0.1)

    def test_dekat_ambang_warning_dan_critical_konsisten(self) -> None:
        near_warning = self._assert_matches_expected("near_warning_threshold")
        near_critical = self._assert_matches_expected("near_critical_threshold")
        self.assertLess(near_warning["lntai"], 18.5)
        self.assertGreater(near_critical["lntai"], 25.0)


class TestLNTAIStateIsolation(unittest.TestCase):
    """Verifikasi state filter lag termal realtime terisolasi per trafo dan
    terpisah dari state lokal calculate_series()."""

    def _sample(self, tc=60, i=1000, ts=0.0):
        return dict(R_Atas=tc, R_Bawah=tc, S_Atas=tc, S_Bawah=tc, T_Atas=tc, T_Bawah=tc,
                    I_R=i, I_S=i, I_T=i, t_ambient=30, timestamp=ts)

    def test_update_state_terpisah_antar_trafo(self) -> None:
        calc = LNTAICalculator()
        r2 = calc.update("2", self._sample(i=1780, ts=0.0))
        r3 = calc.update("3", self._sample(i=100, ts=0.0))
        self.assertNotAlmostEqual(r2["i_eff_norm"], r3["i_eff_norm"], places=3)

    def test_calculate_series_tidak_mengubah_state_realtime(self) -> None:
        calc = LNTAICalculator()
        calc.update("2", self._sample(i=1780, ts=0.0))
        state_before = json.loads(json.dumps(calc._state))  # deep copy via round-trip
        calc.calculate_series([self._sample(i=50, ts=10.0), self._sample(i=50, ts=20.0)])
        self.assertEqual(calc._state, state_before)

    def test_calculate_series_urutan_sama_dengan_update_berurutan(self) -> None:
        rows = [self._sample(tc=55, i=400, ts=0.0), self._sample(tc=60, i=1200, ts=300.0)]
        series_result = LNTAICalculator().calculate_series(rows)[-1]

        calc = LNTAICalculator()
        calc.update("2", rows[0])
        update_result = calc.update("2", rows[1])

        self.assertAlmostEqual(series_result["lntai"], update_result["lntai"], places=6)
        self.assertAlmostEqual(series_result["i_eff_norm"], update_result["i_eff_norm"], places=6)


class TestArrheniusScore(unittest.TestCase):
    """Unit test fungsi kurva skor eksponensial secara terisolasi."""

    def test_nol_pada_rise_nol_atau_negatif(self) -> None:
        self.assertEqual(arrhenius_score(0, 55, 10, 40), 0.0)
        self.assertEqual(arrhenius_score(-5, 55, 10, 40), 0.0)

    def test_maksimum_tepat_pada_rise_sama_dengan_ref(self) -> None:
        self.assertAlmostEqual(arrhenius_score(55, 55, 10, 40), 40.0, places=6)

    def test_melebihi_ref_tetap_clamp_ke_score_max(self) -> None:
        self.assertEqual(arrhenius_score(200, 55, 10, 40), 40.0)

    def test_monoton_naik(self) -> None:
        a = arrhenius_score(10, 55, 10, 40)
        b = arrhenius_score(30, 55, 10, 40)
        c = arrhenius_score(50, 55, 10, 40)
        self.assertLess(a, b)
        self.assertLess(b, c)


class TestFitBaseline(unittest.TestCase):
    """fit_baseline() TIDAK boleh menulis config -- hanya mengembalikan hasil
    fit untuk ditinjau manual."""

    def test_fit_baseline_mengembalikan_r2_dan_n(self) -> None:
        calc = LNTAICalculator()
        rows = []
        # Data sintetis yang sengaja mengikuti baseline config persis, supaya
        # fit_baseline() diharapkan mendekati A/B config itu sendiri (sanity check
        # metode least-squares-nya benar, BUKAN klaim ini kalibrasi data nyata).
        import config
        a, b = config.LNTAI_V2_A, config.LNTAI_V2_B
        for step_i, i_amp in enumerate([0, 400, 800, 1200, 1600, 1780]):
            i_norm = i_amp / config.LNTAI_V2_I_RATED
            delta_t = a * (i_norm ** 2) + b
            tc = 30 + delta_t
            rows.append(dict(R_Atas=tc, R_Bawah=tc, S_Atas=tc, S_Bawah=tc, T_Atas=tc, T_Bawah=tc,
                              I_R=i_amp, I_S=i_amp, I_T=i_amp, t_ambient=30, timestamp=float(step_i * 3600)))
        fit = calc.fit_baseline(rows)
        self.assertIsNotNone(fit["A"])
        self.assertGreater(fit["n"], 0)
        self.assertGreater(fit["r2"], 0.99)
        self.assertAlmostEqual(fit["A"], a, delta=0.5)
        self.assertAlmostEqual(fit["B"], b, delta=0.5)

    def test_fit_baseline_tidak_menulis_config(self) -> None:
        import config
        a_before = config.LNTAI_V2_A
        calc = LNTAICalculator()
        calc.fit_baseline([dict(R_Atas=60, R_Bawah=60, S_Atas=60, S_Bawah=60, T_Atas=60, T_Bawah=60,
                                 I_R=500, I_S=500, I_T=500, t_ambient=30, timestamp=0.0)])
        self.assertEqual(config.LNTAI_V2_A, a_before)


if __name__ == "__main__":
    unittest.main()
