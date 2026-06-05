"""
data/validate_all.py — Day 23: Full Data Pipeline Validation Script.

Runs a comprehensive end-to-end check across your entire dataset before
you start building agents.

What this script checks:
  CHECK 1 — Price files
      • data/prices/<TICKER>.csv exists for all 20 stocks
        Rule: strip .NS, keep hyphen → BAJAJ-AUTO.csv, EICHERMOT.csv
      • Date range spans 2024-01-01 → 2025-12-31
        Handles both YYYY-MM-DD and DD-MM-YYYY (Indian format) date columns
      • OHLCV columns present (Open, High, Low, Close, Volume)
      • No NaN in Close
      • All 10 indicator columns present:
        SMA_20, SMA_50, RSI_14, MACD, MACD_Signal, MACD_Hist,
        BB_Upper, BB_Lower, Change_1D_pct, Change_5D_pct

  CHECK 2 — News files
      • data/news/<TICKER_SAFE>_news.csv exists for all 20 stocks
        Rule: '.' → '_', '&' → 'AND' (so M&M.NS → MANDM_NS_news.csv)
      • Required columns: date, headline, source, summary
      • Date coverage spans both 2024 and 2025

  CHECK 3 — Look-ahead bias
      • 50 random (stock, date) pairs run through get_news_context()
      • Zero violations tolerated

  CHECK 4 — Event calendar & holidays
      • data/events/market_holidays.csv        — must exist
      • data/events/earnings_calendar_2024_2025.csv — WIDE format
        (columns: ticker, company, year, q1_date, q2_date, q3_date, q4_date)
        40 rows = 20 stocks × 2 years → 40 rows × 4 quarters = 160 dates ✓
      • data/events/indian_events.csv          — must have 2024 + 2025 entries

Outputs:
  • Colour-coded terminal summary
  • data/validation_full_report.txt
  • Exit code 0 = all passed, 1 = failures

Usage:
    python data/validate_all.py
    python data/validate_all.py --skip-lookahead

Requirements:
    pip install pandas colorama
"""

import os
import sys
import random
import argparse
import datetime
from pathlib import Path

import pandas as pd

# ── Colour helpers ────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    RED    = Fore.RED
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    RED = GREEN = YELLOW = CYAN = BOLD = RESET = ""

# ── Add project root + data/ to sys.path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER  = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_FOLDER))

# ── Import config (catch EnvironmentError when API keys are absent) ───────────
try:
    import config
    STOCKS      = config.STOCKS
    STOCK_NAMES = config.STOCK_NAMES
    START_DATE  = config.START_DATE
    END_DATE    = config.END_DATE
    PRICES_DIR  = config.PRICES_DIR          # "data/prices"
    NEWS_DIR    = config.NEWS_DIR            # "data/news"
    EVENTS_DIR  = config.EVENTS_DIR          # "data/events"
except (ImportError, EnvironmentError):
    print(f"{YELLOW}⚠  config.py not importable (API keys missing or absent).")
    print(f"   Using built-in defaults — all paths still checked correctly.{RESET}\n")
    STOCKS = [
        "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS",
        "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS",
        "RELIANCE.NS","ONGC.NS","POWERGRID.NS","NTPC.NS",
        "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS",
        "MARUTI.NS","EICHERMOT.NS","BAJAJ-AUTO.NS","M&M.NS",
    ]
    STOCK_NAMES = {
        "TCS.NS":"Tata Consultancy Services","INFY.NS":"Infosys",
        "WIPRO.NS":"Wipro","HCLTECH.NS":"HCL Technologies",
        "HDFCBANK.NS":"HDFC Bank","ICICIBANK.NS":"ICICI Bank",
        "KOTAKBANK.NS":"Kotak Mahindra Bank","SBIN.NS":"State Bank of India",
        "RELIANCE.NS":"Reliance Industries","ONGC.NS":"ONGC",
        "POWERGRID.NS":"Power Grid Corporation","NTPC.NS":"NTPC",
        "HINDUNILVR.NS":"Hindustan Unilever","ITC.NS":"ITC",
        "NESTLEIND.NS":"Nestle India","BRITANNIA.NS":"Britannia Industries",
        "MARUTI.NS":"Maruti Suzuki","EICHERMOT.NS":"Eicher Motors",
        "BAJAJ-AUTO.NS":"Bajaj Auto","M&M.NS":"Mahindra & Mahindra",
    }
    START_DATE = "2024-01-01"
    END_DATE   = "2025-12-31"
    PRICES_DIR = "data/prices"
    NEWS_DIR   = "data/news"
    EVENTS_DIR = "data/events"

# ── Event file paths (your actual filenames) ──────────────────────────────────
HOLIDAYS_CSV = os.path.join(EVENTS_DIR, "market_holidays.csv")
EARNINGS_CSV = os.path.join(EVENTS_DIR, "earnings_calendar_2024_2025.csv")
EVENTS_CSV   = os.path.join(EVENTS_DIR, "indian_events.csv")
REPORT_PATH  = "data/validation_full_report.txt"

# ── Exact indicator column names written by add_indicators.py ─────────────────
REQUIRED_INDICATORS = [
    "SMA_20", "SMA_50",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower",
    "Change_1D_pct", "Change_5D_pct",
]
REQUIRED_OHLCV     = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED_NEWS_COLS = ["date", "headline", "source", "summary"]
LOOKAHEAD_SAMPLE   = 50

SEP  = "─" * 72
SEP2 = "═" * 72


# ════════════════════════════════════════════════════════════════════════════
# FILENAME HELPERS  ← the single source of truth for name → filename mapping
# ════════════════════════════════════════════════════════════════════════════

def price_filename(ticker: str) -> str:
    """
    Converts a ticker to its price CSV filename.

    Rule (matches add_indicators.py exactly):
        strip '.NS'  →  replace '/' with '_'  →  append '.csv'
        '-' and '&' are kept as-is.

    Examples:
        TCS.NS        →  TCS.csv
        BAJAJ-AUTO.NS →  BAJAJ-AUTO.csv
        EICHERMOT.NS  →  EICHERMOT.csv
        M&M.NS        →  M&M.csv
    """
    return ticker.replace(".NS", "").replace("/", "_") + ".csv"


def news_filename(ticker: str) -> str:
    """
    Converts a ticker to its merged news CSV filename.

    Rule (matches get_news_context.py AND your actual files):
        '.' → '_'
        '&' → 'AND'   ← NOT '_' — your file is MANDM_NS not M_M_NS
        '-' kept as-is
        append '_news.csv'

    Examples:
        TCS.NS        →  TCS_NS_news.csv
        BAJAJ-AUTO.NS →  BAJAJ-AUTO_NS_news.csv
        EICHERMOT.NS  →  EICHERMOT_NS_news.csv
        M&M.NS        →  MANDM_NS_news.csv        ← '&' → 'AND', '.' → '_'
    """
    safe = ticker.replace(".", "_").replace("&", "AND")
    return f"{safe}_news.csv"


# ════════════════════════════════════════════════════════════════════════════
# PRICE CSV LOADER  — handles both YYYY-MM-DD and DD-MM-YYYY date formats
# ════════════════════════════════════════════════════════════════════════════

def _load_price_csv(path: Path):
    """
    Load a price CSV with robust date parsing.

    Some files (e.g. EICHERMOT.csv) store dates in Indian DD-MM-YYYY format.
    pandas' default parser reads them as MM-DD-YYYY and fails when day > 12.
    We try three strategies in order:
      1. Standard parse_dates (works for YYYY-MM-DD files)
      2. dayfirst=True on the index (works for DD-MM-YYYY files)
      3. Mixed format inference (fallback for anything else)

    Returns a DataFrame with a normalised DatetimeIndex, or raises on failure.
    """
    # Strategy 1 — standard (YYYY-MM-DD, most stocks)
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = df.index.normalize()
            return df
    except Exception:
        pass

    # Strategy 2 — dayfirst (DD-MM-YYYY, e.g. EICHERMOT)
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, dayfirst=True)
        df.index = df.index.normalize()
        return df
    except Exception:
        pass

    # Strategy 3 — mixed format inference
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, format="mixed", dayfirst=True)
    df.index = df.index.normalize()
    return df


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ok_icon(passed: bool) -> str:
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{SEP}{RESET}")


# ════════════════════════════════════════════════════════════════════════════
# CHECK 1 — PRICE FILES
# ════════════════════════════════════════════════════════════════════════════

def check_price_files() -> dict:
    results        = {}
    expected_start = pd.Timestamp(START_DATE)
    expected_end   = pd.Timestamp(END_DATE)

    for ticker in STOCKS:
        fname = price_filename(ticker)
        path  = Path(PRICES_DIR) / fname
        r = {
            "ticker": ticker, "file": str(path),
            "issues": [], "warnings": [], "rows": 0,
        }

        # ── Existence ────────────────────────────────────────────────────────
        if not path.exists():
            r["issues"].append(
                f"File not found: {path}\n"
                f"         Fix: python data/fetch_prices.py"
            )
            results[ticker] = r
            continue

        # ── Load (handles DD-MM-YYYY dates) ──────────────────────────────────
        try:
            df = _load_price_csv(path)
        except Exception as e:
            r["issues"].append(
                f"Cannot load CSV: {e}\n"
                f"         File: {path}"
            )
            results[ticker] = r
            continue

        r["rows"] = len(df)

        # ── Date range ───────────────────────────────────────────────────────
        actual_start = df.index.min()
        actual_end   = df.index.max()

        if actual_start > expected_start + pd.Timedelta(days=10):
            r["issues"].append(
                f"Data starts {actual_start.date()} — "
                f"expected on/before {START_DATE}"
            )
        if actual_end < expected_end - pd.Timedelta(days=30):
            r["issues"].append(
                f"Data ends {actual_end.date()} — "
                f"expected near {END_DATE} "
                f"({(expected_end - actual_end).days} days short)"
            )

        # ── OHLCV columns ────────────────────────────────────────────────────
        missing_ohlcv = [c for c in REQUIRED_OHLCV if c not in df.columns]
        if missing_ohlcv:
            r["issues"].append(
                f"Missing OHLCV columns: {missing_ohlcv}\n"
                f"         Fix: python data/fetch_prices.py"
            )

        # ── NaN in Close ─────────────────────────────────────────────────────
        if "Close" in df.columns:
            nan_count = int(df["Close"].isna().sum())
            if nan_count > 0:
                r["issues"].append(
                    f"{nan_count} NaN value(s) in Close column"
                )

        # ── Technical indicators ─────────────────────────────────────────────
        missing_ind = [c for c in REQUIRED_INDICATORS if c not in df.columns]
        if missing_ind:
            r["issues"].append(
                f"Missing indicators: {missing_ind}\n"
                f"         Fix: python data/add_indicators.py"
            )

        # ── Row count sanity (2 yrs ≈ 488 trading days) ──────────────────────
        if r["rows"] < 400:
            r["warnings"].append(
                f"Only {r['rows']} rows — expected ~488 for a 2-year window"
            )

        results[ticker] = r

    return results


# ════════════════════════════════════════════════════════════════════════════
# CHECK 2 — NEWS FILES
# ════════════════════════════════════════════════════════════════════════════

def check_news_files() -> dict:
    results = {}

    for ticker in STOCKS:
        fname = news_filename(ticker)
        path  = Path(NEWS_DIR) / fname
        r = {
            "ticker": ticker, "file": str(path),
            "issues": [], "warnings": [], "rows": 0,
        }

        # ── Existence ────────────────────────────────────────────────────────
        if not path.exists():
            r["issues"].append(
                f"File not found: {path}\n"
                f"         Fix: python data/merge_news.py"
            )
            results[ticker] = r
            continue

        # ── Load ─────────────────────────────────────────────────────────────
        try:
            df = pd.read_csv(path)
            r["rows"] = len(df)
        except Exception as e:
            r["issues"].append(f"Cannot load CSV: {e}")
            results[ticker] = r
            continue

        # ── Required columns ─────────────────────────────────────────────────
        missing_cols = [c for c in REQUIRED_NEWS_COLS if c not in df.columns]
        if missing_cols:
            r["issues"].append(
                f"Missing columns: {missing_cols} — "
                "re-run merge_news.py to rebuild"
            )

        # ── Empty ────────────────────────────────────────────────────────────
        if r["rows"] == 0:
            r["issues"].append("CSV is empty — BSE scraper returned nothing")
            results[ticker] = r
            continue

        if r["rows"] < 10:
            r["warnings"].append(
                f"Only {r['rows']} news items — very sparse. "
                "Check BSE scraper for this stock."
            )

        # ── Date coverage ────────────────────────────────────────────────────
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            if len(df) > 0:
                min_d = df["date"].min()
                max_d = df["date"].max()
                if min_d > pd.Timestamp("2024-06-01"):
                    r["warnings"].append(
                        f"Earliest news is {min_d.date()} — missing H1 2024"
                    )
                if max_d < pd.Timestamp("2025-01-01"):
                    r["warnings"].append(
                        f"Latest news is {max_d.date()} — no 2025 coverage"
                    )

        results[ticker] = r

    return results


# ════════════════════════════════════════════════════════════════════════════
# CHECK 3 — LOOK-AHEAD BIAS
# ════════════════════════════════════════════════════════════════════════════

def check_lookahead_bias(sample_size: int = LOOKAHEAD_SAMPLE) -> dict:
    """
    Calls get_news_context() on random (stock, date) pairs.
    get_news_context.py lives in the same data/ folder.
    The news_filename() function here uses '&' → 'AND', which must
    match what get_news_context.py uses for its file lookup too.
    """
    try:
        from get_news_context import get_news_context
    except ImportError:
        try:
            from data.get_news_context import get_news_context
        except ImportError as e:
            return {
                "tested": 0, "passed": 0, "violations": [], "skipped": 0,
                "error": f"Cannot import get_news_context: {e}",
            }

    all_bdays = pd.bdate_range(start="2024-06-01", end=END_DATE)
    random.seed(42)
    pairs = [
        (random.choice(STOCKS), random.choice(all_bdays).strftime("%Y-%m-%d"))
        for _ in range(sample_size)
    ]

    tested = passed_cnt = skipped = 0
    violations = []

    for stock, date_str in pairs:
        try:
            items    = get_news_context(stock, date_str, max_items=3)
            tested  += 1
            query_dt = pd.Timestamp(date_str)
            bad      = [
                i for i in items
                if pd.Timestamp(str(i["date"])) >= query_dt
            ]
            if bad:
                violations.append({
                    "stock": stock, "date": date_str,
                    "bad_items": [
                        {"item_date": str(b["date"])[:10],
                         "headline":  str(b["headline"])[:70]}
                        for b in bad
                    ],
                })
            else:
                passed_cnt += 1
        except FileNotFoundError:
            skipped += 1
        except AssertionError as e:
            violations.append({
                "stock": stock, "date": date_str,
                "bad_items": [{"assertion_error": str(e)[:120]}],
            })
            tested += 1
        except Exception as e:
            violations.append({
                "stock": stock, "date": date_str,
                "bad_items": [{"unexpected_error": str(e)[:120]}],
            })
            tested += 1

    return {
        "tested": tested, "passed": passed_cnt,
        "violations": violations, "skipped": skipped,
    }


# ════════════════════════════════════════════════════════════════════════════
# CHECK 4 — EVENT CALENDAR & HOLIDAYS
# ════════════════════════════════════════════════════════════════════════════

def check_event_files() -> dict:
    """
    Checks three files in data/events/.

    earnings_calendar_2024_2025.csv is WIDE format:
        columns: ticker, company, year, q1_date, q2_date, q3_date, q4_date
        40 rows = 20 stocks × 2 years
        Each row has 4 quarter dates → 40 × 4 = 160 quarterly events total
    The old check compared 40 rows against a 160-row threshold and incorrectly
    flagged it as insufficient. Fixed: count rows × 4 for the actual date count.
    """
    r = {
        "issues": [], "warnings": [],
        "holidays_rows":    0,
        "earnings_rows":    0,   # raw row count (wide format)
        "earnings_dates":   0,   # actual date count = rows × 4
        "events_rows_2024": 0,
        "events_rows_2025": 0,
    }

    # ── 1. market_holidays.csv ────────────────────────────────────────────────
    h_path = Path(HOLIDAYS_CSV)
    if not h_path.exists():
        r["issues"].append(
            f"Holidays file missing: {HOLIDAYS_CSV}\n"
            "         Create with NSE holiday dates — see Day 12."
        )
    else:
        try:
            hdf = pd.read_csv(h_path)
            r["holidays_rows"] = len(hdf)
            if r["holidays_rows"] == 0:
                r["warnings"].append(f"{HOLIDAYS_CSV} is empty")
            elif r["holidays_rows"] < 15:
                r["warnings"].append(
                    f"Only {r['holidays_rows']} holiday rows — "
                    "NSE has ~20 holidays/year; expect 30–45 for 2024+2025"
                )
        except Exception as e:
            r["issues"].append(f"Cannot load {HOLIDAYS_CSV}: {e}")

    # ── 2. earnings_calendar_2024_2025.csv (WIDE format) ─────────────────────
    e_path = Path(EARNINGS_CSV)
    if not e_path.exists():
        r["warnings"].append(
            f"Earnings calendar missing: {EARNINGS_CSV}\n"
            "         Build with: python data/build_earnings_calendar.py"
        )
    else:
        try:
            edf = pd.read_csv(e_path)
            r["earnings_rows"]  = len(edf)
            # Wide format: each row has 4 quarter date columns
            quarter_cols = [c for c in edf.columns if c in
                            ("q1_date","q2_date","q3_date","q4_date")]
            r["earnings_dates"] = r["earnings_rows"] * len(quarter_cols)

            if r["earnings_rows"] == 0:
                r["warnings"].append(f"{EARNINGS_CSV} is empty")
            else:
                # 20 stocks × 2 years = 40 rows is correct for wide format
                expected_rows = len(STOCKS) * 2
                if r["earnings_rows"] < expected_rows * 0.8:   # allow 20% gap
                    r["warnings"].append(
                        f"Only {r['earnings_rows']} rows in wide-format earnings "
                        f"— expected ~{expected_rows} "
                        f"(20 stocks × 2 years). "
                        f"Some stocks may be missing."
                    )
                # This is informational, not a failure
                print(
                    f"  ℹ  Earnings calendar: {r['earnings_rows']} rows "
                    f"(wide format) = "
                    f"{r['earnings_dates']} quarterly dates total"
                )
        except Exception as e:
            r["warnings"].append(f"Cannot load {EARNINGS_CSV}: {e}")

    # ── 3. indian_events.csv ─────────────────────────────────────────────────
    ev_path = Path(EVENTS_CSV)
    if not ev_path.exists():
        r["issues"].append(
            f"Indian events calendar missing: {EVENTS_CSV}\n"
            "         Create from Day 7 instructions — "
            "RBI MPC dates + Budget dates for 2024 and 2025."
        )
        return r

    try:
        evdf = pd.read_csv(ev_path)
    except Exception as e:
        r["issues"].append(f"Cannot load {EVENTS_CSV}: {e}")
        return r

    if evdf.empty:
        r["issues"].append(f"{EVENTS_CSV} is completely empty")
        return r

    # Find date column
    date_col = next(
        (c for c in evdf.columns if "date" in c.lower()), None
    )
    if not date_col:
        r["issues"].append(
            f"No date column in {EVENTS_CSV}. "
            f"Found columns: {list(evdf.columns)}"
        )
        return r

    evdf[date_col] = pd.to_datetime(evdf[date_col], errors="coerce")
    evdf = evdf.dropna(subset=[date_col])

    r["events_rows_2024"] = int((evdf[date_col].dt.year == 2024).sum())
    r["events_rows_2025"] = int((evdf[date_col].dt.year == 2025).sum())

    if r["events_rows_2024"] == 0:
        r["issues"].append(
            "No 2024 events found — add RBI MPC + Budget 2024 dates"
        )
    if r["events_rows_2025"] == 0:
        r["issues"].append(
            "No 2025 events found — add RBI MPC + Budget 2025 dates"
        )
    if len(evdf) < 15:
        r["warnings"].append(
            f"Only {len(evdf)} events total — "
            "expected ≥15 (12 RBI MPC dates + 3 Budget dates)"
        )

    return r


# ════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_stock_rows(results: dict) -> tuple[int, int]:
    passed = failed = 0
    for ticker, r in results.items():
        ok     = len(r["issues"]) == 0
        warn_n = len(r["warnings"])
        rows   = r.get("rows", 0)
        name   = STOCK_NAMES.get(ticker, ticker)

        row_lbl  = f"  [{rows} rows]" if rows else ""
        warn_lbl = (f"  {YELLOW}⚠ {warn_n} warning(s){RESET}"
                    if warn_n else "")

        print(f"  {ok_icon(ok)}  {BOLD}{ticker:<18}{RESET}"
              f" {name}{row_lbl}{warn_lbl}")

        for issue in r["issues"]:
            for line in issue.split("\n"):
                print(f"       {RED}{line}{RESET}")
        for w in r["warnings"]:
            print(f"       {YELLOW}WARN: {w}{RESET}")

        passed += ok
        failed += (not ok)
    return passed, failed


# ════════════════════════════════════════════════════════════════════════════
# MASTER SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_master_summary(
    price_results: dict,
    news_results:  dict,
    lookahead:     dict,
    events:        dict,
    skipped_la:    bool,
) -> bool:
    section("MASTER SUMMARY")

    checks = []

    # Price existence + date range
    p_fail = sum(1 for r in price_results.values() if r["issues"])
    checks.append({
        "label": "Price files (existence + OHLCV + date range)",
        "ok":    p_fail == 0,
        "note":  f"{len(STOCKS) - p_fail}/{len(STOCKS)} stocks OK",
    })

    # Indicators
    ind_fail = sum(
        1 for r in price_results.values()
        if any("Missing indicators" in i for i in r["issues"])
    )
    checks.append({
        "label": "Technical indicators (SMA/RSI/MACD/BB/Change%)",
        "ok":    ind_fail == 0,
        "note":  f"{len(STOCKS) - ind_fail}/{len(STOCKS)} stocks have all indicators",
    })

    # News
    n_fail = sum(1 for r in news_results.values() if r["issues"])
    checks.append({
        "label": "News files (existence + columns + date coverage)",
        "ok":    n_fail == 0,
        "note":  f"{len(STOCKS) - n_fail}/{len(STOCKS)} stocks OK",
    })

    # Look-ahead
    if skipped_la:
        la_ok   = None
        la_note = "Skipped (--skip-lookahead flag)"
    elif "error" in lookahead:
        la_ok   = False
        la_note = lookahead["error"].split("\n")[0][:65]
    else:
        v_count = len(lookahead["violations"])
        la_ok   = v_count == 0
        la_note = (
            f"{lookahead['passed']}/{lookahead['tested']} pairs clean"
            + (f"  |  {lookahead['skipped']} skipped (no CSV)"
               if lookahead.get("skipped") else "")
            + (f"  |  {RED}{v_count} VIOLATION(S){RESET}"
               if v_count else "")
        )
    checks.append({
        "label": "Look-ahead bias test (50 random pairs)",
        "ok":    la_ok,
        "note":  la_note,
    })

    # Events
    ev_ok = len(events["issues"]) == 0
    checks.append({
        "label": "Event calendar (holidays + indian_events + earnings)",
        "ok":    ev_ok,
        "note":  (
            f"Holidays: {events['holidays_rows']} rows  |  "
            f"Earnings: {events['earnings_rows']} rows "
            f"({events['earnings_dates']} quarterly dates)  |  "
            f"Events 2024: {events['events_rows_2024']}  "
            f"Events 2025: {events['events_rows_2025']}"
        ),
    })

    all_passed = True
    for c in checks:
        if c["ok"] is None:
            flag = f"{YELLOW}–{RESET}"
        elif c["ok"]:
            flag = f"{GREEN}✓{RESET}"
        else:
            flag = f"{RED}✗{RESET}"
            all_passed = False

        print(f"  {flag}  {c['label']}")
        print(f"       {c['note']}")

    print()
    if all_passed:
        print(
            f"  {GREEN}{BOLD}✓ ALL CHECKS PASSED.{RESET} "
            "Dataset is clean and complete.\n"
        )
    else:
        print(
            f"  {RED}{BOLD}✗ SOME CHECKS FAILED.{RESET} "
            "Fix the issues above before building agents.\n"
        )
        print("  Quick fix guide:")
        print("    Price missing      →  python data/fetch_prices.py")
        print("    Indicators missing →  python data/add_indicators.py")
        print("    News missing       →  python data/merge_news.py")
        print("    Look-ahead fail    →  check data/get_news_context.py date filter")
        print("    Events missing     →  create data/events/indian_events.csv (Day 7)")

    return all_passed


# ════════════════════════════════════════════════════════════════════════════
# REPORT FILE
# ════════════════════════════════════════════════════════════════════════════

def build_report(
    price_results: dict,
    news_results:  dict,
    lookahead:     dict,
    events:        dict,
    skipped_la:    bool,
) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        SEP2,
        "  FULL DATA PIPELINE VALIDATION REPORT",
        f"  Generated : {now}",
        f"  Date range: {START_DATE} → {END_DATE}",
        SEP2, "",
    ]

    lines.append("── PRICE FILES " + "─" * 57)
    p_p = p_f = 0
    for ticker, r in price_results.items():
        ok = len(r["issues"]) == 0
        lines.append(
            f"  {'PASS' if ok else 'FAIL'}  "
            f"{ticker:<18}  {r['rows']} rows  →  {r['file']}"
        )
        for i in r["issues"]:   lines.append(f"         ISSUE: {i}")
        for w in r["warnings"]: lines.append(f"         WARN : {w}")
        p_p += ok; p_f += (not ok)
    lines.append(f"\n  Price: {p_p} passed / {p_f} failed\n")

    lines.append("── NEWS FILES " + "─" * 58)
    n_p = n_f = 0
    for ticker, r in news_results.items():
        ok = len(r["issues"]) == 0
        lines.append(
            f"  {'PASS' if ok else 'FAIL'}  "
            f"{ticker:<18}  {r['rows']} articles  →  {r['file']}"
        )
        for i in r["issues"]:   lines.append(f"         ISSUE: {i}")
        for w in r["warnings"]: lines.append(f"         WARN : {w}")
        n_p += ok; n_f += (not ok)
    lines.append(f"\n  News: {n_p} passed / {n_f} failed\n")

    lines.append("── LOOK-AHEAD BIAS " + "─" * 52)
    if skipped_la:
        lines.append("  SKIPPED")
    elif "error" in lookahead:
        lines.append(f"  ERROR: {lookahead['error']}")
    else:
        lines.append(
            f"  Tested: {lookahead['tested']}  "
            f"Passed: {lookahead['passed']}  "
            f"Violations: {len(lookahead['violations'])}  "
            f"Skipped: {lookahead['skipped']}"
        )
        for v in lookahead.get("violations", []):
            lines.append(f"  VIOLATION  {v['stock']}  on  {v['date']}")
            for b in v["bad_items"]: lines.append(f"    {b}")
    lines.append("")

    lines.append("── EVENT CALENDAR & HOLIDAYS " + "─" * 43)
    lines.append(
        f"  Holidays file     : {HOLIDAYS_CSV}  "
        f"({events['holidays_rows']} rows)"
    )
    lines.append(
        f"  Earnings file     : {EARNINGS_CSV}  "
        f"({events['earnings_rows']} rows wide = "
        f"{events['earnings_dates']} quarterly dates)"
    )
    lines.append(f"  Indian events 2024: {events['events_rows_2024']}")
    lines.append(f"  Indian events 2025: {events['events_rows_2025']}")
    for i in events["issues"]:  lines.append(f"  ISSUE: {i}")
    for w in events["warnings"]:lines.append(f"  WARN : {w}")

    lines += ["", SEP2, "END OF REPORT", SEP2]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(skip_lookahead: bool = False) -> int:
    print(f"\n{BOLD}{SEP2}{RESET}")
    print(f"{BOLD}  FULL DATA PIPELINE VALIDATION{RESET}")
    print(f"  Date range    : {START_DATE}  →  {END_DATE}")
    print(f"  Stocks        : {len(STOCKS)}")
    print(f"  Prices dir    : {PRICES_DIR}")
    print(f"  News dir      : {NEWS_DIR}")
    print(f"  Events dir    : {EVENTS_DIR}")
    print(f"{BOLD}{SEP2}{RESET}")

    # ── 1. Price files ───────────────────────────────────────────────────────
    section("CHECK 1 — PRICE FILES (existence · date range · OHLCV · indicators)")
    print(f"  Filename rule: strip .NS  →  keep hyphen  (e.g. BAJAJ-AUTO.csv)")
    print(f"  Date parsing : handles both YYYY-MM-DD and DD-MM-YYYY formats\n")
    price_results = check_price_files()
    p_pass, p_fail = print_stock_rows(price_results)
    print(f"\n  Result: {GREEN}{p_pass} passed{RESET}  {RED}{p_fail} failed{RESET}")

    # ── 2. News files ────────────────────────────────────────────────────────
    section("CHECK 2 — NEWS FILES (existence · columns · date coverage)")
    print(f"  Filename rule: '.' → '_', '&' → 'AND'")
    print(f"  e.g. M&M.NS → MANDM_NS_news.csv   "
          f"BAJAJ-AUTO.NS → BAJAJ-AUTO_NS_news.csv\n")
    news_results = check_news_files()
    n_pass, n_fail = print_stock_rows(news_results)
    print(f"\n  Result: {GREEN}{n_pass} passed{RESET}  {RED}{n_fail} failed{RESET}")

    # ── 3. Look-ahead bias ───────────────────────────────────────────────────
    section("CHECK 3 — LOOK-AHEAD BIAS TEST")
    if skip_lookahead:
        print(f"  {YELLOW}Skipped (--skip-lookahead flag).{RESET}")
        lookahead = {}
    else:
        print(f"  Running {LOOKAHEAD_SAMPLE} random (stock, date) pairs ...\n")
        lookahead = check_lookahead_bias()
        if "error" in lookahead:
            print(f"  {RED}ERROR: {lookahead['error']}{RESET}")
        else:
            la_ok = len(lookahead["violations"]) == 0
            print(
                f"  {ok_icon(la_ok)}  "
                f"Tested: {lookahead['tested']}  "
                f"Passed: {lookahead['passed']}  "
                f"Violations: {len(lookahead['violations'])}  "
                f"Skipped (no CSV): {lookahead['skipped']}"
            )
            if lookahead["violations"]:
                print(f"\n  {RED}{BOLD}⚠  LOOK-AHEAD VIOLATIONS — fix immediately!{RESET}")
                for v in lookahead["violations"]:
                    print(f"  {RED}Stock: {v['stock']}  Date: {v['date']}{RESET}")
                    for b in v["bad_items"]:
                        print(f"    {b}")

    # ── 4. Event calendar ────────────────────────────────────────────────────
    section("CHECK 4 — EVENT CALENDAR & HOLIDAY FILES")
    print(f"  Checking:")
    print(f"    {HOLIDAYS_CSV}")
    print(f"    {EARNINGS_CSV}  ← wide format (rows × 4 quarters = total dates)")
    print(f"    {EVENTS_CSV}\n")
    events = check_event_files()
    ev_ok  = len(events["issues"]) == 0

    print(
        f"  {ok_icon(events['holidays_rows'] > 0)}  "
        f"market_holidays.csv              "
        f"{events['holidays_rows']} rows"
    )
    print(
        f"  {ok_icon(events['earnings_rows'] > 0)}  "
        f"earnings_calendar_2024_2025.csv  "
        f"{events['earnings_rows']} rows  "
        f"({events['earnings_dates']} quarterly dates)"
    )
    print(
        f"  {ok_icon(events['events_rows_2024'] > 0 and events['events_rows_2025'] > 0)}  "
        f"indian_events.csv                "
        f"2024: {events['events_rows_2024']}  |  "
        f"2025: {events['events_rows_2025']}"
    )
    for issue in events["issues"]:
        for line in issue.split("\n"):
            print(f"       {RED}{line}{RESET}")
    for w in events["warnings"]:
        print(f"       {YELLOW}WARN: {w}{RESET}")

    # ── Master summary ───────────────────────────────────────────────────────
    all_ok = print_master_summary(
        price_results, news_results, lookahead, events, skip_lookahead
    )

    # ── Save report ──────────────────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)
    report = build_report(
        price_results, news_results, lookahead, events, skip_lookahead
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Full report saved → {REPORT_PATH}")
    print(f"{BOLD}{SEP2}{RESET}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full data pipeline validation"
    )
    parser.add_argument(
        "--skip-lookahead", action="store_true",
        help="Skip look-ahead test (useful if news CSVs not ready yet)"
    )
    args = parser.parse_args()
    sys.exit(main(skip_lookahead=args.skip_lookahead))
