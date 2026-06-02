"""
data/validate_prices.py — Day 12: Price data validation script.

Checks every stock CSV in data/prices/ for:
  1. File existence
  2. Expected date range coverage
  3. Missing trading days (gaps that are NOT NSE holidays)
  4. NaN values in OHLCV columns
  5. Zero or negative volume days
  6. Zero or negative Close price
  7. OHLC consistency (High >= Low, High >= Open/Close, Low <= Open/Close)
  8. Suspiciously large single-day price moves (>20%)

Outputs:
  - Coloured summary to terminal
  - data/validation_report.txt  — machine-readable full report
  - data/events/market_holidays.csv — created as an empty template if missing,
    with a reminder to fill it in from nseindia.com

Usage:
    python data/validate_prices.py

Requires:
    pip install pandas numpy colorama
"""

import os
import sys
import datetime
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np

# ── Try importing colorama for coloured terminal output ──────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    RED    = Fore.RED
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    RED = GREEN = YELLOW = BOLD = RESET = ""

# ── Import project config ────────────────────────────────────────────────────
# Add project root to path so this script can be run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import config
    STOCKS      = config.STOCKS
    STOCK_NAMES = config.STOCK_NAMES
    START_DATE  = config.START_DATE
    END_DATE    = config.END_DATE
    PRICES_DIR  = config.PRICES_DIR
    HOLIDAYS_CSV = "data/events/market_holidays.csv"   # updated name per Day 12
except ImportError:
    # Fallback — hardcode if config.py is not available
    print(f"{YELLOW}⚠  Could not import config.py. Using fallback values.{RESET}")
    STOCKS = [
        "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","HDFCBANK.NS",
        "ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","RELIANCE.NS","ONGC.NS",
        "POWERGRID.NS","NTPC.NS","HINDUNILVR.NS","ITC.NS","NESTLEIND.NS",
        "BRITANNIA.NS","MARUTI.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","M&M.NS",
    ]
    STOCK_NAMES = {s: s for s in STOCKS}
    START_DATE  = "2024-01-01"
    END_DATE    = "2025-12-31"
    PRICES_DIR  = "data/prices"
    HOLIDAYS_CSV = "data/events/market_holidays.csv"

# ── Constants ────────────────────────────────────────────────────────────────
LARGE_MOVE_THRESHOLD = 0.20      # Flag single-day moves > 20%
REQUIRED_OHLCV_COLS  = ["Open", "High", "Low", "Close", "Volume"]
REPORT_PATH          = "data/validation_report.txt"

# ── Separator helpers ────────────────────────────────────────────────────────
SEP  = "─" * 72
SEP2 = "═" * 72


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD HOLIDAYS
# ════════════════════════════════════════════════════════════════════════════

def load_holidays(holidays_csv: str) -> set:
    """
    Load NSE holidays from CSV and return as a set of datetime.date objects.
    If the file doesn't exist, creates an empty template and returns an empty set.
    """
    path = Path(holidays_csv)

    if not path.exists():
        # Create parent directories and an empty template
        path.parent.mkdir(parents=True, exist_ok=True)
        template = pd.DataFrame(columns=["date", "year", "holiday_name"])
        template.to_csv(path, index=False)
        print(f"{YELLOW}⚠  {holidays_csv} not found. Created empty template.")
        print(f"   Fill it in from nseindia.com > Market > Holiday List")
        print(f"   Columns: date (YYYY-MM-DD), year (int), holiday_name (str){RESET}")
        return set()

    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        print(f"{YELLOW}⚠  {holidays_csv} is empty or missing 'date' column. "
              f"Holiday checks will be skipped.{RESET}")
        return set()

    holidays = set(pd.to_datetime(df["date"], dayfirst=True).dt.date)
    print(f"✓  Loaded {len(holidays)} NSE holidays from {holidays_csv}")
    return holidays


# ════════════════════════════════════════════════════════════════════════════
# 2. GENERATE EXPECTED TRADING DAYS
# ════════════════════════════════════════════════════════════════════════════

def get_expected_trading_days(start: str, end: str, holidays: set) -> pd.DatetimeIndex:
    """
    Return business days (Mon–Fri) between start and end, excluding known
    NSE holidays. This is an approximation — the gold standard is the actual
    NSE holiday list you populate in market_holidays.csv.
    """
    all_bdays = pd.bdate_range(start=start, end=end)
    trading_days = pd.DatetimeIndex(
        [d for d in all_bdays if d.date() not in holidays]
    )
    return trading_days


# ════════════════════════════════════════════════════════════════════════════
# 3. VALIDATE A SINGLE STOCK FILE
# ════════════════════════════════════════════════════════════════════════════

def validate_stock(ticker: str, expected_days: pd.DatetimeIndex) -> dict:
    """
    Run all checks on one stock's price CSV.

    Returns a dict with keys:
      ticker, file_exists, row_count, issues (list of dicts),
      warnings (list of dicts), passed (bool)
    """
    result = {
        "ticker":      ticker,
        "file_exists": False,
        "row_count":   0,
        "issues":      [],   # hard failures
        "warnings":    [],   # soft flags worth investigating
        "passed":      False,
    }

    # ── 1. File existence ────────────────────────────────────────────────────
    csv_name = ticker.replace(".NS", "").replace("/", "_")   # TCS.NS → TCS, M&M.NS → M&M
    csv_path = Path(PRICES_DIR) / f"{csv_name}.csv"

    if not csv_path.exists():
        result["issues"].append({
            "check": "file_exists",
            "detail": f"File not found: {csv_path}",
        })
        return result   # can't do further checks

    result["file_exists"] = True

    # ── 2. Load CSV ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    except KeyError:
        # Some yfinance saves use 'Datetime' or the date as index without name
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index.name = "Date"
        except Exception as e:
            result["issues"].append({"check": "load", "detail": str(e)})
            return result

    df.index = pd.to_datetime(df.index).normalize()   # strip time component
    result["row_count"] = len(df)

    # ── 3. Required columns ──────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
    if missing_cols:
        result["issues"].append({
            "check": "columns",
            "detail": f"Missing columns: {missing_cols}",
        })
        return result   # can't run price checks without OHLCV

    # ── 4. Date range coverage ───────────────────────────────────────────────
    actual_start = df.index.min().date()
    actual_end   = df.index.max().date()
    expected_start = pd.to_datetime(START_DATE).date()
    expected_end   = pd.to_datetime(END_DATE).date()

    if actual_start > expected_start:
        result["issues"].append({
            "check": "date_range_start",
            "detail": (f"Data starts {actual_start} — expected on or before "
                       f"{expected_start}. Missing early data."),
        })
    if actual_end < expected_end:
        # Only flag as issue if more than 5 business days short
        # (end of range data often has a small lag)
        lag_days = (expected_end - actual_end).days
        if lag_days > 7:
            result["issues"].append({
                "check": "date_range_end",
                "detail": (f"Data ends {actual_end} — expected near "
                           f"{expected_end} ({lag_days} days short)."),
            })
        else:
            result["warnings"].append({
                "check": "date_range_end",
                "detail": (f"Data ends {actual_end} ({lag_days} days before "
                           f"expected end {expected_end}). Likely a download lag."),
            })

    # ── 5. Missing trading days ──────────────────────────────────────────────
    actual_days    = set(df.index.normalize())
    expected_set   = set(expected_days)
    missing_days   = sorted(expected_set - actual_days)

    # Filter to only days within the actual data range
    missing_in_range = [
        d for d in missing_days
        if pd.Timestamp(actual_start) <= d <= pd.Timestamp(actual_end)
    ]

    if missing_in_range:
        # Group consecutive missing days for readability
        sample = [str(d.date()) for d in missing_in_range[:10]]
        suffix = f"... and {len(missing_in_range) - 10} more" if len(missing_in_range) > 10 else ""
        result["issues"].append({
            "check": "missing_trading_days",
            "detail": (f"{len(missing_in_range)} missing trading days "
                       f"(not in holiday list): {', '.join(sample)}{suffix}"),
        })

    # ── 6. NaN values in OHLCV ──────────────────────────────────────────────
    for col in REQUIRED_OHLCV_COLS:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_dates = df[df[col].isna()].index[:5].strftime("%Y-%m-%d").tolist()
            result["issues"].append({
                "check": f"nan_{col.lower()}",
                "detail": (f"{nan_count} NaN in '{col}' column. "
                           f"First occurrences: {nan_dates}"),
            })

    # ── 7. Zero or negative volume ───────────────────────────────────────────
    zero_vol = df[df["Volume"] <= 0]
    if len(zero_vol) > 0:
        dates = zero_vol.index[:5].strftime("%Y-%m-%d").tolist()
        result["warnings"].append({
            "check": "zero_volume",
            "detail": (f"{len(zero_vol)} day(s) with zero/negative volume: "
                       f"{dates}{'...' if len(zero_vol) > 5 else ''}"),
        })

    # ── 8. Zero or negative Close ────────────────────────────────────────────
    bad_close = df[df["Close"] <= 0]
    if len(bad_close) > 0:
        result["issues"].append({
            "check": "invalid_close",
            "detail": (f"{len(bad_close)} row(s) with Close ≤ 0. "
                       f"Dates: {bad_close.index[:5].strftime('%Y-%m-%d').tolist()}"),
        })

    # ── 9. OHLC internal consistency ─────────────────────────────────────────
    # High should be the highest of O, H, L, C
    # Low should be the lowest
    bad_high = df[df["High"] < df[["Open", "Close"]].max(axis=1)]
    bad_low  = df[df["Low"]  > df[["Open", "Close"]].min(axis=1)]

    if len(bad_high) > 0:
        result["issues"].append({
            "check": "ohlc_high",
            "detail": (f"{len(bad_high)} row(s) where High < max(Open, Close). "
                       f"Dates: {bad_high.index[:5].strftime('%Y-%m-%d').tolist()}"),
        })
    if len(bad_low) > 0:
        result["issues"].append({
            "check": "ohlc_low",
            "detail": (f"{len(bad_low)} row(s) where Low > min(Open, Close). "
                       f"Dates: {bad_low.index[:5].strftime('%Y-%m-%d').tolist()}"),
        })

    # ── 10. Large single-day moves ───────────────────────────────────────────
    df_sorted = df.sort_index()
    df_sorted["pct_change"] = df_sorted["Close"].pct_change().abs()
    large_moves = df_sorted[df_sorted["pct_change"] > LARGE_MOVE_THRESHOLD]

    if len(large_moves) > 0:
        details = [
            f"{d.strftime('%Y-%m-%d')} ({row['pct_change']:.1%})"
            for d, row in large_moves.iterrows()
        ]
        result["warnings"].append({
            "check": "large_move",
            "detail": (f"{len(large_moves)} day(s) with >{LARGE_MOVE_THRESHOLD:.0%} "
                       f"close-to-close move: {', '.join(details[:5])}"
                       f"{'...' if len(details) > 5 else ''}"),
        })

    result["passed"] = len(result["issues"]) == 0
    return result


# ════════════════════════════════════════════════════════════════════════════
# 4. PRINT & SAVE REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_stock_result(r: dict) -> None:
    """Print one stock's result to terminal."""
    ticker      = r["ticker"]
    name        = STOCK_NAMES.get(ticker, ticker)
    status_icon = f"{GREEN}✓{RESET}" if r["passed"] else f"{RED}✗{RESET}"
    warn_icon   = f" {YELLOW}⚠ {len(r['warnings'])} warning(s){RESET}" if r["warnings"] else ""
    rows        = f"  [{r['row_count']} rows]" if r["file_exists"] else ""

    print(f"  {status_icon}  {BOLD}{ticker:<18}{RESET} {name}{rows}{warn_icon}")

    for issue in r["issues"]:
        print(f"       {RED}ISSUE [{issue['check']}]{RESET}: {issue['detail']}")
    for warn in r["warnings"]:
        print(f"       {YELLOW}WARN  [{warn['check']}]{RESET}: {warn['detail']}")


def build_text_report(results: list, holidays: set,
                      expected_days: pd.DatetimeIndex) -> str:
    """Build a plain-text report string for saving to file."""
    lines = []
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(SEP2)
    lines.append(f"  PRICE DATA VALIDATION REPORT")
    lines.append(f"  Generated : {now}")
    lines.append(f"  Date range: {START_DATE} → {END_DATE}")
    lines.append(f"  Expected trading days: {len(expected_days)}")
    lines.append(f"  NSE holidays loaded: {len(holidays)}")
    lines.append(SEP2)

    passed   = [r for r in results if r["passed"]]
    failed   = [r for r in results if not r["passed"]]
    warnings = [r for r in results if r["warnings"]]

    lines.append(f"\nSUMMARY")
    lines.append(f"  Total stocks checked : {len(results)}")
    lines.append(f"  Passed (no issues)   : {len(passed)}")
    lines.append(f"  Failed (has issues)  : {len(failed)}")
    lines.append(f"  Has warnings         : {len(warnings)}")

    if failed:
        lines.append(f"\n{SEP}")
        lines.append("STOCKS WITH ISSUES")
        lines.append(SEP)
        for r in failed:
            lines.append(f"\n  [{r['ticker']}]  {STOCK_NAMES.get(r['ticker'], '')}")
            for issue in r["issues"]:
                wrapped = textwrap.fill(
                    f"  ISSUE [{issue['check']}]: {issue['detail']}",
                    width=70, subsequent_indent="    "
                )
                lines.append(wrapped)
            for warn in r["warnings"]:
                wrapped = textwrap.fill(
                    f"  WARN  [{warn['check']}]: {warn['detail']}",
                    width=70, subsequent_indent="    "
                )
                lines.append(wrapped)

    if warnings:
        lines.append(f"\n{SEP}")
        lines.append("STOCKS WITH WARNINGS ONLY")
        lines.append(SEP)
        for r in warnings:
            if r["passed"]:   # only those that passed issues but have warnings
                lines.append(f"\n  [{r['ticker']}]  {STOCK_NAMES.get(r['ticker'], '')}")
                for warn in r["warnings"]:
                    wrapped = textwrap.fill(
                        f"  WARN  [{warn['check']}]: {warn['detail']}",
                        width=70, subsequent_indent="    "
                    )
                    lines.append(wrapped)

    lines.append(f"\n{SEP}")
    lines.append("ALL STOCKS — QUICK REFERENCE")
    lines.append(SEP)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        warn_n = len(r["warnings"])
        rows   = r["row_count"]
        lines.append(
            f"  {status:<5}  {r['ticker']:<18}  {rows:>4} rows"
            f"  {warn_n} warning(s)"
        )

    lines.append(f"\n{SEP2}")
    lines.append("END OF REPORT")
    lines.append(SEP2)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{BOLD}{SEP2}{RESET}")
    print(f"{BOLD}  PRICE DATA VALIDATION — Indian Multi-Agent Trading System{RESET}")
    print(f"  Date range : {START_DATE} → {END_DATE}")
    print(f"  Prices dir : {PRICES_DIR}")
    print(f"{BOLD}{SEP2}{RESET}\n")

    # ── Step 1: Load holidays ────────────────────────────────────────────────
    holidays = load_holidays(HOLIDAYS_CSV)

    # ── Step 2: Build expected trading days ──────────────────────────────────
    expected_days = get_expected_trading_days(START_DATE, END_DATE, holidays)
    print(f"  Expected trading days in range: {len(expected_days)}")
    print(f"  ({START_DATE} → {END_DATE}, excl. weekends + {len(holidays)} holidays)\n")

    # ── Step 3: Validate each stock ──────────────────────────────────────────
    print(f"{BOLD}{SEP}{RESET}")
    print(f"{BOLD}  Per-Stock Results{RESET}")
    print(f"{BOLD}{SEP}{RESET}")

    results = []
    for ticker in STOCKS:
        r = validate_stock(ticker, expected_days)
        results.append(r)
        print_stock_result(r)

    # ── Step 4: Summary ──────────────────────────────────────────────────────
    passed_n  = sum(1 for r in results if r["passed"])
    failed_n  = len(results) - passed_n
    warning_n = sum(1 for r in results if r["warnings"])

    print(f"\n{BOLD}{SEP}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{SEP}{RESET}")
    print(f"  Stocks checked : {len(results)}")
    print(f"  {GREEN}Passed{RESET}         : {passed_n}")
    print(f"  {RED}Failed{RESET}         : {failed_n}")
    print(f"  {YELLOW}Warnings{RESET}       : {warning_n} stock(s) have soft warnings")

    if failed_n == 0:
        print(f"\n  {GREEN}{BOLD}✓ All stocks passed validation. Data pipeline is clean.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}✗ {failed_n} stock(s) failed. Fix issues before proceeding to Day 13.{RESET}")
        print(f"  Most gaps are NSE holidays — add them to {HOLIDAYS_CSV}")
        print(f"  and re-run this script. Genuine gaps need manual investigation on nseindia.com.")

    if warning_n > 0:
        print(f"\n  {YELLOW}⚠  Large price moves flagged as warnings are usually correct{RESET}")
        print(f"  (earnings surprises, budget day, policy announcements).")
        print(f"  Cross-check each against your event calendar.")

    # ── Step 5: Save text report ─────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    report_text = build_text_report(results, holidays, expected_days)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  Full report saved → {REPORT_PATH}")
    print(f"{BOLD}{SEP2}{RESET}\n")


if __name__ == "__main__":
    main()
