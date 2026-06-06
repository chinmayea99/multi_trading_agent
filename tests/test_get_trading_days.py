"""
tests/test_get_trading_days.py — Unit tests for Day 25 script.

Run from project root:
    python -m pytest tests/test_get_trading_days.py -v

Or without pytest:
    python tests/test_get_trading_days.py
"""

import sys
import logging
import datetime
import tempfile
import unittest
from pathlib import Path

# ── Add project root to path ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Silence logger output during tests
logging.disable(logging.CRITICAL)

from data.get_trading_days import (
    load_holidays,
    generate_trading_days,
    validate_trading_days,
)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def make_logger() -> logging.Logger:
    """Return a silent logger for testing."""
    logger = logging.getLogger("test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    return logger


def write_temp_holidays(rows: list[str]) -> str:
    """Write a temp holidays CSV and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp.write("date,year,holiday_name\n")
    for row in rows:
        tmp.write(row + "\n")
    tmp.close()
    return tmp.name


# ════════════════════════════════════════════════════════════════════════════
# TEST SUITE
# ════════════════════════════════════════════════════════════════════════════

class TestLoadHolidays(unittest.TestCase):

    def test_missing_file_returns_empty_set(self):
        """load_holidays should return an empty set when file doesn't exist."""
        logger = make_logger()
        result = load_holidays("/nonexistent/path.csv", logger)
        self.assertIsInstance(result, set)
        self.assertEqual(len(result), 0)

    def test_valid_csv_returns_dates(self):
        """load_holidays should parse dates correctly from a valid CSV."""
        logger = make_logger()
        path = write_temp_holidays([
            "2024-01-26,2024,Republic Day",
            "2024-03-25,2024,Holi",
            "2025-01-26,2025,Republic Day",
        ])
        result = load_holidays(path, logger)
        self.assertEqual(len(result), 3)
        self.assertIn(datetime.date(2024, 1, 26), result)
        self.assertIn(datetime.date(2025, 1, 26), result)

    def test_empty_csv_returns_empty_set(self):
        """load_holidays should handle an empty CSV gracefully."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        tmp.write("date,year,holiday_name\n")
        tmp.close()
        logger = make_logger()
        result = load_holidays(tmp.name, logger)
        self.assertIsInstance(result, set)


class TestGenerateTradingDays(unittest.TestCase):

    def setUp(self):
        self.logger = make_logger()

    def test_no_weekends(self):
        """Output should contain no Saturdays or Sundays."""
        df = generate_trading_days("2024-01-01", "2024-01-31", set(), self.logger)
        self.assertNotIn("Saturday", df["day_of_week"].values)
        self.assertNotIn("Sunday", df["day_of_week"].values)

    def test_holidays_excluded(self):
        """A known holiday date should not appear in the output."""
        republic_day = datetime.date(2024, 1, 26)   # Friday — would normally be a trading day
        holidays = {republic_day}
        df = generate_trading_days("2024-01-01", "2024-01-31", holidays, self.logger)
        self.assertNotIn("2024-01-26", df["date"].values)

    def test_january_2024_count(self):
        """January 2024 has 23 weekdays. With Republic Day (26th, Friday) excluded → 22."""
        republic_day = datetime.date(2024, 1, 26)
        df = generate_trading_days("2024-01-01", "2024-01-31", {republic_day}, self.logger)
        self.assertEqual(len(df), 22)

    def test_metadata_columns_present(self):
        """All expected metadata columns should be in output."""
        df = generate_trading_days("2024-01-01", "2024-01-07", set(), self.logger)
        for col in ["date", "day_of_week", "week_number", "year", "month",
                    "is_month_start", "is_month_end"]:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_month_start_end_flags(self):
        """is_month_start should be True only for the first trading day of a month."""
        df = generate_trading_days("2024-01-01", "2024-01-31", set(), self.logger)
        month_starts = df[df["is_month_start"]]
        self.assertEqual(len(month_starts), 1)
        self.assertEqual(month_starts.iloc[0]["date"], "2024-01-01")

    def test_full_range_count(self):
        """
        Full 2024+2025 range with no holidays should yield ~522 trading days
        (261 per year is the maximum possible weekdays).
        """
        df = generate_trading_days("2024-01-01", "2025-12-31", set(), self.logger)
        self.assertGreater(len(df), 500)
        self.assertLess(len(df), 530)

    def test_both_years_present(self):
        """Both 2024 and 2025 should appear in the year column."""
        df = generate_trading_days("2024-01-01", "2025-12-31", set(), self.logger)
        self.assertIn(2024, df["year"].values)
        self.assertIn(2025, df["year"].values)

    def test_date_order_ascending(self):
        """Dates should be in ascending chronological order."""
        df = generate_trading_days("2024-01-01", "2024-03-31", set(), self.logger)
        dates = list(df["date"])
        self.assertEqual(dates, sorted(dates))


class TestValidateTradingDays(unittest.TestCase):

    def setUp(self):
        self.logger = make_logger()

    def test_clean_dataframe_passes(self):
        """A cleanly generated DataFrame should pass all validation checks."""
        df = generate_trading_days("2024-01-01", "2025-12-31", set(), self.logger)
        result = validate_trading_days(df, self.logger)
        self.assertTrue(result)

    def test_duplicate_dates_fail(self):
        """Duplicate dates should cause validate to return False."""
        import pandas as pd
        df = generate_trading_days("2024-01-01", "2024-01-31", set(), self.logger)
        df_duped = pd.concat([df, df.head(1)], ignore_index=True)
        result = validate_trading_days(df_duped, self.logger)
        self.assertFalse(result)


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
