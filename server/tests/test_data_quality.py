"""Test server/data_quality.py — tiap flag minimal 2 kasus (positif & negatif).
Tanpa akses jaringan, tanpa DB.

Jalankan dari folder server/:
    pytest tests/test_data_quality.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data_quality as dq  # noqa: E402


class TestMissing:
    def test_positive_none_value(self):
        assert dq.classify_temperature_value(None) == {"MISSING"}
        assert dq.classify_current_value(None) == {"MISSING"}

    def test_negative_present_value(self):
        assert "MISSING" not in dq.classify_temperature_value(55.0)
        assert "MISSING" not in dq.classify_current_value(100.0)


class TestZeroReading:
    def test_positive_zero_is_flagged_but_not_missing(self):
        flags = dq.classify_temperature_value(0.0)
        assert flags == {"ZERO_READING"}
        flags_i = dq.classify_current_value(0)
        assert flags_i == {"ZERO_READING"}

    def test_negative_nonzero_not_flagged(self):
        assert "ZERO_READING" not in dq.classify_temperature_value(55.0)
        assert "ZERO_READING" not in dq.classify_current_value(100.0)


class TestOutOfRange:
    def test_positive_temperature_too_high(self):
        assert "OUT_OF_RANGE" in dq.classify_temperature_value(250.0)

    def test_positive_temperature_too_low(self):
        assert "OUT_OF_RANGE" in dq.classify_temperature_value(-50.0)

    def test_positive_current_exceeds_rated_multiplier(self):
        # i_rated=1780, multiplier=3 -> batas 5340A
        assert "OUT_OF_RANGE" in dq.classify_current_value(6000.0)

    def test_positive_current_negative(self):
        assert "OUT_OF_RANGE" in dq.classify_current_value(-5.0)

    def test_negative_temperature_within_range(self):
        assert "OUT_OF_RANGE" not in dq.classify_temperature_value(55.0)

    def test_negative_current_within_range(self):
        assert "OUT_OF_RANGE" not in dq.classify_current_value(1000.0)


class TestStuck:
    def test_positive_30_identical_samples(self):
        history = [55.0] * 30
        assert dq.detect_stuck(history) is True

    def test_negative_varying_samples(self):
        history = [55.0 + i * 0.1 for i in range(30)]
        assert dq.detect_stuck(history) is False

    def test_negative_fewer_than_threshold(self):
        history = [55.0] * 29
        assert dq.detect_stuck(history) is False

    def test_negative_contains_none(self):
        history = [55.0] * 29 + [None]
        assert dq.detect_stuck(history) is False


class TestSpike:
    def test_positive_spike_then_revert(self):
        assert dq.detect_spike(55.0, 90.0, 56.0) is True

    def test_negative_genuine_rising_trend(self):
        # lompatan besar tapi TIDAK kembali -- tren naik sah, bukan noise
        assert dq.detect_spike(55.0, 90.0, 92.0) is False

    def test_negative_small_jump(self):
        assert dq.detect_spike(55.0, 58.0, 56.0) is False

    def test_negative_missing_values(self):
        assert dq.detect_spike(None, 90.0, 56.0) is False
        assert dq.detect_spike(55.0, None, 56.0) is False
        assert dq.detect_spike(55.0, 90.0, None) is False


class TestTimeGap:
    def test_positive_large_gap(self):
        # nominal 300s * multiplier 3 = 900s ambang
        assert dq.detect_time_gap(1200) is True

    def test_negative_normal_gap(self):
        assert dq.detect_time_gap(300) is False

    def test_negative_none(self):
        assert dq.detect_time_gap(None) is False


class TestPairingGap:
    def test_positive_not_paired(self):
        assert dq.detect_pairing_gap(False) is True

    def test_negative_paired(self):
        assert dq.detect_pairing_gap(True) is False


class TestCombineRowFlags:
    def test_union_of_multiple_sensors(self):
        result = dq.combine_row_flags([{"ZERO_READING"}, {"OUT_OF_RANGE"}, set()])
        assert result == {"ZERO_READING", "OUT_OF_RANGE"}

    def test_empty_list(self):
        assert dq.combine_row_flags([]) == set()


class TestBitmaskConversion:
    def test_flags_to_bitmask_single(self):
        assert dq.flags_to_bitmask({"MISSING"}) == 1

    def test_flags_to_bitmask_combined(self):
        # MISSING(1) | ZERO_READING(2) | STUCK(8) = 11
        assert dq.flags_to_bitmask({"MISSING", "ZERO_READING", "STUCK"}) == 11

    def test_bitmask_to_flags_roundtrip(self):
        original = {"ZERO_READING", "SPIKE"}
        bitmask = dq.flags_to_bitmask(original)
        assert set(dq.bitmask_to_flags(bitmask)) == original

    def test_bitmask_to_flags_none(self):
        assert dq.bitmask_to_flags(None) == []

    def test_bitmask_to_flags_zero(self):
        assert dq.bitmask_to_flags(0) == []


class TestShouldSkipLntai:
    def test_positive_missing_blocks(self):
        bitmask = dq.flags_to_bitmask({"MISSING"})
        assert dq.should_skip_lntai(bitmask) is True

    def test_positive_out_of_range_blocks(self):
        bitmask = dq.flags_to_bitmask({"OUT_OF_RANGE"})
        assert dq.should_skip_lntai(bitmask) is True

    def test_positive_stuck_blocks(self):
        bitmask = dq.flags_to_bitmask({"STUCK"})
        assert dq.should_skip_lntai(bitmask) is True

    def test_negative_zero_reading_does_not_block(self):
        """T2: 0 adalah bacaan sah -- tidak boleh menghentikan LNTAI."""
        bitmask = dq.flags_to_bitmask({"ZERO_READING"})
        assert dq.should_skip_lntai(bitmask) is False

    def test_negative_spike_pairing_timegap_do_not_block(self):
        bitmask = dq.flags_to_bitmask({"SPIKE", "PAIRING_GAP", "TIME_GAP"})
        assert dq.should_skip_lntai(bitmask) is False

    def test_negative_none_bitmask(self):
        assert dq.should_skip_lntai(None) is False
