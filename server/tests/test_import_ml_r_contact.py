"""Test server/scripts/import_ml_r_contact.py — parsing & anomaly flagging.
Tanpa DB/network, tanpa membaca index.html sungguhan (fixture string).

Jalankan dari folder server/:
    pytest tests/test_import_ml_r_contact.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import import_ml_r_contact as script  # noqa: E402

FIXTURE_HTML = """
<html><script>
        let ML_R_CONTACT = [
            {date:'2022-01-18',trafo:1,R:[293.9,226.3,150.6],I:100,source:'baseline_lokal'},
            {date:'2022-04-07',trafo:1,R:[293.9,226.3,150.6],I:100,source:'baseline_lokal'},
            {date:'2022-06-08',trafo:3,R:[166.1,100.0,92.4], I:100,source:'baseline_lokal'},
            {date:'2022-07-07',trafo:2,R:[162.5,147.8,126.5],I:100,source:'baseline_lokal'},
            {date:'2024-06-07',trafo:2,R:[17.0, 29.0, 10.0], I:100,source:'baseline_lokal'},
            {date:'2026-01-17',trafo:2,R:[124.47,115.67,143.77],I:100,source:'baseline_lokal'}
        ];
</script></html>
"""


class TestExtractMlRContact:
    def test_extracts_all_entries_regardless_of_whitespace_variation(self):
        """Regresi: bug nyata ditemukan saat benar-benar menjalankan skrip
        ini atas index.html -- beberapa baris punya spasi sebelum 'I:'
        ('], I:100' alih-alih '],I:100'), regex lama melewatkannya diam-diam."""
        entries = script.extract_ml_r_contact(FIXTURE_HTML)
        assert len(entries) == 6

    def test_parses_fields_correctly(self):
        entries = script.extract_ml_r_contact(FIXTURE_HTML)
        first = entries[0]
        assert first["date"] == "2022-01-18"
        assert first["trafo"] == 1
        assert first["r"] == [293.9, 226.3, 150.6]
        assert first["i"] == 100.0
        assert first["source"] == "baseline_lokal"

    def test_missing_array_raises(self):
        import pytest

        with pytest.raises(ValueError):
            script.extract_ml_r_contact("<html>no array here</html>")


class TestFindDuplicates:
    def test_detects_identical_readings_on_different_dates_same_trafo(self):
        entries = script.extract_ml_r_contact(FIXTURE_HTML)
        duplicates = script.find_duplicates(entries)
        assert len(duplicates) == 1
        a, b = duplicates[0]
        assert a["trafo"] == b["trafo"] == 1
        assert {a["date"], b["date"]} == {"2022-01-18", "2022-04-07"}

    def test_no_false_positive_across_different_trafos(self):
        entries = [
            {"date": "2022-01-01", "trafo": 2, "r": [100.0, 100.0, 100.0], "i": 100.0, "source": "x"},
            {"date": "2022-01-02", "trafo": 3, "r": [100.0, 100.0, 100.0], "i": 100.0, "source": "x"},
        ]
        assert script.find_duplicates(entries) == []


class TestFindOutliers:
    def test_detects_the_specific_pattern_from_the_task_brief(self):
        """Trafo 2's 2024 entry (10-29 µΩ) must be flagged as inconsistent
        with 2022/2026 (115-165 µΩ range)."""
        entries = script.extract_ml_r_contact(FIXTURE_HTML)
        outliers = script.find_outliers(entries)
        flagged_dates = {o["date"] for o in outliers if o["trafo"] == 2}
        assert "2024-06-07" in flagged_dates
        assert "2026-01-17" not in flagged_dates

    def test_consistent_readings_are_not_flagged(self):
        entries = [
            {"date": "2022-01-01", "trafo": 3, "r": [100.0, 105.0, 95.0], "i": 100.0, "source": "x"},
            {"date": "2023-01-01", "trafo": 3, "r": [98.0, 102.0, 100.0], "i": 100.0, "source": "x"},
            {"date": "2024-01-01", "trafo": 3, "r": [101.0, 99.0, 103.0], "i": 100.0, "source": "x"},
        ]
        assert script.find_outliers(entries) == []

    def test_single_entry_trafo_has_nothing_to_compare_against(self):
        entries = [{"date": "2022-01-01", "trafo": 4, "r": [50.0, 50.0, 50.0], "i": 100.0, "source": "x"}]
        assert script.find_outliers(entries) == []


class TestGenerateInserts:
    def test_produces_three_rows_per_entry(self):
        entries = script.extract_ml_r_contact(FIXTURE_HTML)
        sql = script.generate_inserts(entries, script.find_duplicates(entries), script.find_outliers(entries))
        assert sql.count("INSERT INTO maintenance_events") == len(entries) * 3

    def test_flags_appear_as_comments_above_affected_rows(self):
        entries = script.extract_ml_r_contact(FIXTURE_HTML)
        sql = script.generate_inserts(entries, script.find_duplicates(entries), script.find_outliers(entries))
        assert "REVIEW REQUIRED" in sql
        assert "DUPLICATE" in sql
        assert "OUTLIER" in sql

    def test_script_never_imports_a_db_driver(self):
        """Safety assertion: this script only does regex parsing and string
        formatting -- it must never import psycopg/ConnectionPool, since it
        is explicitly meant to print SQL for review, not execute it."""
        import inspect

        source = inspect.getsource(script)
        assert "psycopg" not in source
        assert "ConnectionPool" not in source
