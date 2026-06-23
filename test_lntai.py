"""Unit test LNTAI.

Jalankan:
    python test_lntai.py
"""

from __future__ import annotations

import unittest

import config
from lntai_calculator import LNTAICalculator


class TestLNTAICalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = LNTAICalculator()

    def _contact_for_lntai(self, target_lntai: float, i_load: float = 1000.0) -> float:
        i_norm = i_load / config.I_RATED
        delta_t_pred = config.A * (i_norm**2) + config.B
        return config.T_AMBIENT + delta_t_pred + (target_lntai * config.SIGMA)

    def test_suhu_normal_beban_normal(self) -> None:
        result = self.calculator.calculate(t_contact=58, i_load=800, t_ambient=30)
        self.assertEqual(result["level"], "NORMAL")
        self.assertLess(result["lntai"], 1.5)

    def test_suhu_tinggi_beban_tinggi(self) -> None:
        result = self.calculator.calculate(t_contact=90, i_load=1000, t_ambient=30)
        self.assertEqual(result["level"], "KRITIS")
        self.assertGreaterEqual(result["lntai"], 5.0)

    def test_suhu_tinggi_beban_rendah(self) -> None:
        result = self.calculator.calculate(t_contact=82, i_load=100, t_ambient=30)
        self.assertEqual(result["level"], "KRITIS")
        self.assertGreaterEqual(result["residual"], 20)

    def test_arus_nol(self) -> None:
        result = self.calculator.calculate(t_contact=58, i_load=0, t_ambient=30)
        self.assertEqual(result["i_norm"], 0.0)
        self.assertEqual(result["level"], "NORMAL")

    def test_input_tidak_valid(self) -> None:
        result = self.calculator.calculate(t_contact="", i_load="abc", t_ambient=None)
        self.assertEqual(result["level"], "DATA_INVALID")
        self.assertIn("error", result)

    def test_boundary_1_5(self) -> None:
        t_contact = self._contact_for_lntai(1.5)
        result = self.calculator.calculate(t_contact=t_contact, i_load=1000, t_ambient=30)
        self.assertEqual(result["level"], "PERHATIAN")

    def test_boundary_3_0(self) -> None:
        t_contact = self._contact_for_lntai(3.0)
        result = self.calculator.calculate(t_contact=t_contact, i_load=1000, t_ambient=30)
        self.assertEqual(result["level"], "SIAGA")

    def test_boundary_5_0(self) -> None:
        t_contact = self._contact_for_lntai(5.0)
        result = self.calculator.calculate(t_contact=t_contact, i_load=1000, t_ambient=30)
        self.assertEqual(result["level"], "KRITIS")

    def test_batch(self) -> None:
        results = self.calculator.calculate_batch(
            [
                {"t_contact": 75, "i_load": 800, "t_ambient": 30},
                {"t_contact": 90, "i_load": 1000, "t_ambient": 30},
            ]
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["level"], "KRITIS")

    def test_three_phase(self) -> None:
        result = self.calculator.calculate_three_phase(
            {
                "R_upper": 50,
                "R_lower": 51,
                "S_upper": 52,
                "S_lower": 53,
                "T_upper": 70,
                "T_lower": 72,
                "I_R": 800,
                "I_S": 810,
                "I_T": 805,
                "T_ambient": 30,
            }
        )
        self.assertEqual(result["hottest_phase"], "T")
        self.assertAlmostEqual(result["T_T"], 71.0)
        self.assertIn("lntai", result["lntai_result"])


if __name__ == "__main__":
    unittest.main()
