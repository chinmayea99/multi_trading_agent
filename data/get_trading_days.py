"""
data/get_trading_days.py — NSE Trading Calendar Generator

PURPOSE:
    Generate the complete, authoritative list of NSE trading days for
    2024 and 2025. This list is the backbone of the backtest loop —
    every agent decision in Phase 3 onward will iterate over exactly
    these dates.

APPROACH:
    1. Load market_holidays.csv (NSE official holidays for both years)
    2. Generate all calendar days from 2024-01-01 to 2025-12-31
    3. Exclude weekends (Saturday = 5, Sunday = 6)
    4. Exclude NSE holidays loaded from CSV
    5. Save the resulting trading day list as trading_days.csv

INPUT:
    data/events/market_holidays.csv
        Columns: date (YYYY-MM-DD), year (int), holiday_name (str)

OUTPUT:
    data/events/trading_days.csv
        Columns: date (YYYY-MM-DD), day_of_week (str), week_number (int),
                 year (int), month (int), is_month_start (bool),
                 is_month_end (bool)

USAGE:
    python data/get_trading_days.py

AUTHOR:  M.Tech Project — Multi-Agent LLM Trading System (Indian Market)
DAY:     25 of 90
"""

import sys
import logging
import datetime
from pathlib import Path

import pandas as pd

# ── Make sure project root is on sys.path so we can import config ────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
    START_DATE   = config.START_DATE          # "2024-01-01"
    END_DATE     = config.END_DATE            # "2025-12-31"
    HOLIDAYS_CSV = f"{config.EVENTS_DIR}/market_holidays.csv"
    TRADING_DAYS_CSV = f"{config.EVENTS_DIR}/trading_days.csv"
except ImportError:
    # Graceful fallback if config is unavailable
    START_DATE   = "2024-01-01"
    END_DATE     = "2025-12-31"
    HOLIDAYS_CSV = "data/events/market_holidays.csv"
    TRADING_DAYS_CSV = "data/events/trading_days.csv"


# ════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """
    Configure and return the module logger.

    Outputs to both stdout and a rotating log file under logs/.
    Level: INFO for normal operation, DEBUG for troubleshooting.
    """
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "get_trading_days.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logger initialised → %s", log_file)
    return logger


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD NSE HOLIDAYS
# ════════════════════════════════════════════════════════════════════════════

def load_holidays(holidays_csv: str, logger: logging.Logger) -> set:
    """
    Load NSE market holidays from CSV into a set of datetime.date objects.

    The CSV must have at minimum a 'date' column in YYYY-MM-DD format.
    Optional columns 'year' and 'holiday_name' are accepted but not required.

    If the file is missing, logs a warning and returns an empty set so the
    rest of the pipeline can still run (weekends are still excluded).

    Args:
        holidays_csv: Path to market_holidays.csv
        logger:       Active logger

    Returns:
        set of datetime.date objects representing NSE holidays
    """
    path = Path(holidays_csv)

    if not path.exists():
        logger.warning(
            "Holidays CSV not found: %s\n"
            "  ➜ Only weekends will be excluded. NSE holidays will NOT be excluded.\n"
            "  ➜ Please create %s with columns: date, year, holiday_name\n"
            "  ➜ Source: https://www.nseindia.com/products-services/equity-market-holidays",
            holidays_csv, holidays_csv
        )
        return set()

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", holidays_csv, exc)
        return set()

    if df.empty:
        logger.warning("Holidays CSV is empty: %s", holidays_csv)
        return set()

    if "date" not in df.columns:
        logger.error(
            "Holidays CSV missing required 'date' column. "
            "Found columns: %s", list(df.columns)
        )
        return set()

    # Parse dates — accept YYYY-MM-DD or DD-MM-YYYY via pandas inference
    try:
        parsed = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    except Exception as exc:
        logger.error("Date parsing failed: %s", exc)
        return set()

    invalid_count = parsed.isna().sum()
    if invalid_count > 0:
        logger.warning("%d holiday date(s) could not be parsed and will be skipped.", invalid_count)

    holidays = set(parsed.dropna().dt.date)

    # Log per-year breakdown for quick sanity check
    for year in [2024, 2025]:
        year_holidays = {d for d in holidays if d.year == year}
        logger.info("  %d NSE holidays loaded for %d", len(year_holidays), year)

    logger.info("✓ Total NSE holidays loaded: %d from %s", len(holidays), holidays_csv)
    return holidays


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — GENERATE TRADING DAYS
# ════════════════════════════════════════════════════════════════════════════

def generate_trading_days(
    start: str,
    end: str,
    holidays: set,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Generate all valid NSE trading days between start and end (inclusive).

    A valid trading day satisfies ALL three conditions:
        1. Falls within [start, end]
        2. Is a weekday (Monday through Friday)
        3. Is NOT in the NSE holidays set

    Each resulting date is enriched with calendar metadata columns that
    are useful for the backtest loop and thesis reporting.

    Args:
        start:    Start date string "YYYY-MM-DD"
        end:      End date string "YYYY-MM-DD"
        holidays: Set of datetime.date objects for NSE holidays
        logger:   Active logger

    Returns:
        DataFrame with columns:
            date           (str)  YYYY-MM-DD
            day_of_week    (str)  e.g. "Monday"
            week_number    (int)  ISO week number 1–53
            year           (int)
            month          (int)  1–12
            is_month_start (bool) True if first trading day of the month
            is_month_end   (bool) True if last trading day of the month
    """
    start_dt = datetime.date.fromisoformat(start)
    end_dt   = datetime.date.fromisoformat(end)

    logger.info("Generating trading days: %s → %s", start, end)
    logger.info("Total calendar days in range: %d", (end_dt - start_dt).days + 1)

    # ── Iterate every calendar day and apply filters ─────────────────────────
    trading_dates = []
    current = start_dt

    while current <= end_dt:
        is_weekday = current.weekday() < 5          # Mon=0 … Fri=4
        is_holiday = current in holidays

        if is_weekday and not is_holiday:
            trading_dates.append(current)

        current += datetime.timedelta(days=1)

    if not trading_dates:
        logger.error("No trading days generated! Check your date range and holidays file.")
        return pd.DataFrame()

    logger.info("✓ Raw trading days before metadata: %d", len(trading_dates))

    # ── Build base DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame({"date": trading_dates})
    df["date"] = pd.to_datetime(df["date"])

    # ── Calendar metadata columns ────────────────────────────────────────────
    df["day_of_week"]  = df["date"].dt.day_name()
    df["week_number"]  = df["date"].dt.isocalendar().week.astype(int)
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month

    # is_month_start: first trading day of each calendar month
    # is_month_end:   last trading day of each calendar month
    df["_ym"] = df["date"].dt.to_period("M")
    first_per_month = df.groupby("_ym")["date"].transform("min")
    last_per_month  = df.groupby("_ym")["date"].transform("max")
    df["is_month_start"] = (df["date"] == first_per_month)
    df["is_month_end"]   = (df["date"] == last_per_month)
    df.drop(columns=["_ym"], inplace=True)

    # ── Convert date back to string for clean CSV output ─────────────────────
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # ── Per-year summary log ─────────────────────────────────────────────────
    for year in [2024, 2025]:
        count = (df["year"] == year).sum()
        logger.info("  Trading days in %d: %d", year, count)

    logger.info("✓ Total trading days generated: %d", len(df))
    return df


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — VALIDATE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def validate_trading_days(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Run basic sanity checks on the generated trading days DataFrame.

    Checks performed:
        - No duplicate dates
        - No weekends present (Saturday/Sunday)
        - Both 2024 and 2025 are represented
        - Count is plausible (NSE typically has ~245 trading days per year)

    Args:
        df:     DataFrame from generate_trading_days()
        logger: Active logger

    Returns:
        True if all checks pass, False if any critical check fails.
    """
    logger.info("Running validation checks …")
    all_passed = True

    # Check 1: No duplicates
    dup_count = df["date"].duplicated().sum()
    if dup_count > 0:
        logger.error("FAIL — %d duplicate dates found!", dup_count)
        all_passed = False
    else:
        logger.info("  ✓ No duplicate dates")

    # Check 2: No weekends
    weekend_mask = df["day_of_week"].isin(["Saturday", "Sunday"])
    if weekend_mask.any():
        logger.error("FAIL — %d weekend dates found!", weekend_mask.sum())
        all_passed = False
    else:
        logger.info("  ✓ No weekends present")

    # Check 3: Both years present
    years_present = set(df["year"].unique())
    for year in [2024, 2025]:
        if year not in years_present:
            logger.error("FAIL — Year %d has no trading days!", year)
            all_passed = False
        else:
            logger.info("  ✓ Year %d is present", year)

    # Check 4: Plausible count per year (NSE typically 240–252 trading days)
    MIN_DAYS, MAX_DAYS = 230, 260
    for year in [2024, 2025]:
        count = (df["year"] == year).sum()
        if not (MIN_DAYS <= count <= MAX_DAYS):
            logger.warning(
                "  ⚠ Year %d has %d trading days — outside expected range %d–%d. "
                "Check holidays CSV.",
                year, count, MIN_DAYS, MAX_DAYS
            )
        else:
            logger.info("  ✓ Year %d count %d is plausible", year, count)

    if all_passed:
        logger.info("✓ All validation checks passed.")
    else:
        logger.error("✗ One or more validation checks FAILED. Review logs above.")

    return all_passed


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — SAVE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def save_trading_days(df: pd.DataFrame, output_csv: str, logger: logging.Logger) -> None:
    """
    Save the trading days DataFrame to CSV.

    Creates parent directories automatically if they don't exist.

    Args:
        df:         DataFrame from generate_trading_days()
        output_csv: Destination file path
        logger:     Active logger
    """
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(out_path, index=False)
        logger.info("✓ Saved %d trading days → %s", len(df), out_path)
    except Exception as exc:
        logger.error("Failed to save output: %s", exc)
        raise


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — PRINT SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, holidays: set, logger: logging.Logger) -> None:
    """
    Print a human-readable summary of the generated trading calendar.
    This output can be pasted directly into thesis Chapter 3 (Data Description).

    Args:
        df:       Generated trading days DataFrame
        holidays: Set of holiday dates loaded earlier
        logger:   Active logger
    """
    sep  = "─" * 60
    sep2 = "═" * 60

    print(f"\n{sep2}")
    print("  NSE TRADING CALENDAR SUMMARY — 2024 & 2025")
    print(sep2)

    print(f"\n  Holidays loaded from CSV : {len(holidays)}")
    print(f"  Date range              : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    print(f"  Total trading days      : {len(df)}")

    print(f"\n{sep}")
    print(f"  {'Year':<8} {'Trading Days':<16} {'Holidays Excluded'}")
    print(sep)

    for year in [2024, 2025]:
        td   = (df["year"] == year).sum()
        hols = len([d for d in holidays if d.year == year])
        print(f"  {year:<8} {td:<16} {hols}")

    print(f"\n{sep}")
    print(f"  {'Month':<12} {'2024':>8} {'2025':>8}")
    print(sep)

    months = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May",     6: "June",     7: "July",   8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    for m in range(1, 13):
        c24 = ((df["year"] == 2024) & (df["month"] == m)).sum()
        c25 = ((df["year"] == 2025) & (df["month"] == m)).sum()
        print(f"  {months[m]:<12} {c24:>8} {c25:>8}")

    print(f"\n{sep}")
    print(f"  First 5 trading days of 2024:")
    for d in df[df["year"] == 2024]["date"].head(5):
        print(f"    {d}")

    print(f"\n  First 5 trading days of 2025:")
    for d in df[df["year"] == 2025]["date"].head(5):
        print(f"    {d}")

    print(f"\n  Last 5 trading days of 2025:")
    for d in df[df["year"] == 2025]["date"].tail(5):
        print(f"    {d}")

    print(f"\n{sep2}\n")


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orchestrate the full trading day generation pipeline:
        1. Load NSE holidays from CSV
        2. Generate trading days (exclude weekends + holidays)
        3. Validate the result
        4. Save to CSV
        5. Print summary
    """
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("  NSE Trading Calendar Generator")
    logger.info("  Date range: %s → %s", START_DATE, END_DATE)
    logger.info("=" * 60)

    # ── Step 1: Load holidays ────────────────────────────────────────────────
    holidays = load_holidays(HOLIDAYS_CSV, logger)

    # ── Step 2: Generate trading days ────────────────────────────────────────
    df = generate_trading_days(START_DATE, END_DATE, holidays, logger)

    if df.empty:
        logger.critical("Trading day generation produced no output. Exiting.")
        sys.exit(1)

    # ── Step 3: Validate ─────────────────────────────────────────────────────
    passed = validate_trading_days(df, logger)
    if not passed:
        logger.warning("Validation issues detected. Output saved anyway for inspection.")

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    save_trading_days(df, TRADING_DAYS_CSV, logger)

    # ── Step 5: Print summary ────────────────────────────────────────────────
    print_summary(df, holidays, logger)

    logger.info("Done. Trading calendar ready at: %s", TRADING_DAYS_CSV)


if __name__ == "__main__":
    main()
