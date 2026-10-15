"""Test server/aging.py — faktor percepatan penuaan Arrhenius (Stage 2,
Part C). Fungsi murni, tanpa DB/network.

Jalankan dari folder server/:
    pytest tests/test_aging.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")

import aging  # noqa: E402

TEST_CFG = {"ea_ev": 0.5, "t_ref_c": 55.0, "max_gap_minutes_for_integration": 60}


class TestIsConfigured:
    def test_default_shipped_config_is_not_configured(self):
        """aging_config.json yang dikirim defaultnya ea_ev/t_ref_c null --
        lihat TODO(calibration) di file itu."""
        assert aging.is_configured() is False

    def test_fully_configured_dict_is_configured(self):
        assert aging.is_configured(TEST_CFG) is True

    def test_missing_ea_ev_is_not_configured(self):
        assert aging.is_configured({"ea_ev": None, "t_ref_c": 55.0}) is False

    def test_missing_t_ref_c_is_not_configured(self):
        assert aging.is_configured({"ea_ev": 0.5, "t_ref_c": None}) is False


class TestGetCalibrationStatus:
    """Go-live (provisional operation): 'status'/'ea_source' di
    aging_config.json SELALU divalidasi ulang di sini, tidak pernah
    dipercaya apa adanya -- lihat docstring get_calibration_status()."""

    def test_default_shipped_config_is_not_configured(self):
        result = aging.get_calibration_status()
        assert result == {"status": "not_configured", "ea_source": None}

    def test_fully_valid_provisional_is_accepted(self):
        cfg = {"ea_ev": 0.5, "t_ref_c": 55.0, "status": "provisional", "ea_source": "IEC TR 60943 Table 2"}
        result = aging.get_calibration_status(cfg)
        assert result == {"status": "provisional", "ea_source": "IEC TR 60943 Table 2"}

    def test_fully_valid_calibrated_is_accepted(self):
        cfg = {"ea_ev": 0.5, "t_ref_c": 55.0, "status": "calibrated", "ea_source": "Site resistance test 2026-09-01"}
        result = aging.get_calibration_status(cfg)
        assert result["status"] == "calibrated"

    def test_provisional_without_ea_source_downgrades_to_not_configured(self):
        """Inti aturan go-live: TANPA sitasi sumber, status provisional
        TIDAK dipercaya, walau ea_ev/t_ref_c sudah berupa angka asli."""
        cfg = {"ea_ev": 0.5, "t_ref_c": 55.0, "status": "provisional", "ea_source": ""}
        result = aging.get_calibration_status(cfg)
        assert result == {"status": "not_configured", "ea_source": None}

    def test_provisional_with_missing_ea_source_key_downgrades(self):
        cfg = {"ea_ev": 0.5, "t_ref_c": 55.0, "status": "provisional"}
        result = aging.get_calibration_status(cfg)
        assert result["status"] == "not_configured"

    def test_provisional_without_ea_ev_downgrades_to_not_configured(self):
        """ea_source saja tidak cukup -- ea_ev/t_ref_c tetap wajib angka."""
        cfg = {"ea_ev": None, "t_ref_c": 55.0, "status": "provisional", "ea_source": "IEC TR 60943"}
        result = aging.get_calibration_status(cfg)
        assert result == {"status": "not_configured", "ea_source": None}

    def test_declared_status_not_in_allowed_set_downgrades(self):
        cfg = {"ea_ev": 0.5, "t_ref_c": 55.0, "status": "definitely_calibrated_trust_me", "ea_source": "x"}
        result = aging.get_calibration_status(cfg)
        assert result["status"] == "not_configured"

    def test_missing_status_field_entirely_is_not_configured(self):
        cfg = {"ea_ev": 0.5, "t_ref_c": 55.0, "ea_source": "IEC TR 60943"}
        result = aging.get_calibration_status(cfg)
        assert result["status"] == "not_configured"


class TestCelsiusKelvinConversion:
    def test_zero_celsius_is_273_15_kelvin(self):
        assert aging.celsius_to_kelvin(0.0) == 273.15

    def test_absolute_zero_celsius_is_zero_kelvin(self):
        assert aging.celsius_to_kelvin(-273.15) == 0.0

    def test_positive_temperature(self):
        assert aging.celsius_to_kelvin(55.0) == 328.15


class TestArrheniusFactor:
    def test_not_configured_returns_none(self):
        assert aging.arrhenius_factor(80.0) is None

    def test_f_equals_one_at_reference_temperature(self):
        f = aging.arrhenius_factor(TEST_CFG["t_ref_c"], TEST_CFG)
        assert abs(f - 1.0) < 1e-9

    def test_monotonic_increase_with_temperature(self):
        f60 = aging.arrhenius_factor(60.0, TEST_CFG)
        f70 = aging.arrhenius_factor(70.0, TEST_CFG)
        f80 = aging.arrhenius_factor(80.0, TEST_CFG)
        assert f60 < f70 < f80

    def test_below_reference_gives_factor_less_than_one(self):
        f = aging.arrhenius_factor(40.0, TEST_CFG)
        assert f < 1.0

    def test_above_reference_gives_factor_greater_than_one(self):
        f = aging.arrhenius_factor(70.0, TEST_CFG)
        assert f > 1.0


class TestIntegrateEqDays:
    def test_not_configured_returns_status_not_configured(self):
        result = aging.integrate_eq_days([(0.0, 55.0)])
        assert result["status"] == "not_configured"

    def test_empty_samples(self):
        result = aging.integrate_eq_days([], TEST_CFG)
        assert result["status"] == "ok"
        assert result["eq_days"] == 0.0
        assert result["n_samples"] == 0

    def test_constant_at_reference_for_one_day_gives_one_equivalent_day(self):
        samples = [(i * 300.0, 55.0) for i in range(289)]  # 86400s = 1 day
        result = aging.integrate_eq_days(samples, TEST_CFG)
        assert abs(result["eq_days"] - 1.0) < 1e-6
        assert abs(result["f_mean"] - 1.0) < 1e-9

    def test_constant_hot_temperature_scales_eq_days_by_f(self):
        f70 = aging.arrhenius_factor(70.0, TEST_CFG)
        samples = [(i * 300.0, 70.0) for i in range(289)]
        result = aging.integrate_eq_days(samples, TEST_CFG)
        assert abs(result["eq_days"] - f70) < 1e-6

    def test_large_gap_is_excluded_from_integral(self):
        samples = [(0.0, 55.0), (7200.0, 55.0)]  # 2h gap > max_gap_minutes_for_integration=60
        result = aging.integrate_eq_days(samples, TEST_CFG)
        assert result["eq_days"] == 0.0
        assert result["n_gaps_excluded"] == 1

    def test_reports_mean_and_peak_hotspot_temperature(self):
        samples = [(0.0, 50.0), (300.0, 60.0), (600.0, 70.0)]
        result = aging.integrate_eq_days(samples, TEST_CFG)
        assert result["t_hot_peak_c"] == 70.0
        assert abs(result["t_hot_mean_c"] - 60.0) < 1e-9


class TestScopeText:
    def test_scope_text_mentions_contact_finger_only(self):
        assert "contact-finger" in aging.AGING_SCOPE_TEXT
        assert "vacuum interrupter" in aging.AGING_SCOPE_TEXT
