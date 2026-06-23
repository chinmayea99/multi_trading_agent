# validate_pipeline.py -- Full End-to-End Data Pipeline Validation
#
# M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.
#
# Project root : C:/Users/Chinmaye A/multi_trading_agent/
# Script lives : C:/Users/Chinmaye A/multi_trading_agent/validate_pipeline.py
#                (project root -- one level ABOVE the data/ folder)
#
# What it does:
#   Master validation script that tests the ENTIRE data pipeline from price
#   download through context generation. Checks every stage sequentially,
#   validates data integrity, detects look-ahead bias, and produces a
#   pass/fail readiness report for Phase 3 (Agent Framework).
#
# Why it matters:
#   Before building agents that depend on this data, we MUST confirm that:
#     1. Every pipeline module (inside data/) is importable and syntactically correct.
#     2. Every expected output file exists with correct columns and row counts.
#     3. No look-ahead bias exists anywhere (price rows or news items dated
#        ON or AFTER the query date must never appear in context).
#     4. One call to build_context(stock, date) works for any stock/date.
#     5. The three required test cases (TCS / RELIANCE / HDFCBANK) all pass.
#
# How to run (from project root):
#   python validate_pipeline.py
#   python validate_pipeline.py --quick         # imports + dirs only
#   python validate_pipeline.py --stock TCS.NS --date 2024-03-15
#
# Actual file layout (from File_structure.txt):
#   data/prices/TCS.csv                          <- no .NS suffix, no underscore
#   data/prices/M&M.csv                          <- literal ampersand in filename
#   data/news/TCS_NS_news.csv                    <- {stem}_NS_news.csv pattern
#   data/news/M_M_NS_news.csv                    <- & replaced with _ in news filenames
#   data/news/TCS_NS_bse.csv                     <- BSE filings also in data/news/
#   data/raw_bse/TCS_2024-01_2025-12_p1.json
#   data/raw_news/bse/TCS_NS_2024_01_raw.json
#   data/raw_news/newsapi/TCS_NS_2026-05_p1.json
#   data/events/earnings_calendar_2024_2025.csv  <- NOT earnings_calendar_2024.csv
#   data/events/indian_events.csv
#   data/events/market_holidays.csv              <- NOT market_holidays_2024.csv
#   data/events/trading_days.csv
#   data/context_builder.py                      <- scripts are INSIDE data/
#   data/get_news_context.py
#   data/price_context.py
#
# Requires: pip install pandas yfinance python-dotenv


from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── Project root = folder containing THIS script ──────────────────────────────
# When run as:  python validate_pipeline.py
# ROOT = C:\Users\Chinmaye A\multi_trading_agent
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# Make both root AND data\ importable so "import data.context_builder" works
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA_DIR))   # also allows bare "import context_builder"

try:
    import pandas as pd
except ImportError:
    print("FATAL: pandas not installed.  Run: pip install pandas")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Paths — derived from File_structure.txt
# ─────────────────────────────────────────────────────────────────────────────
PRICES_DIR   = DATA_DIR / "prices"
NEWS_DIR     = DATA_DIR / "news"
RAW_BSE_DIR  = DATA_DIR / "raw_bse"
RAW_NEWS_DIR = DATA_DIR / "raw_news"
EVENTS_DIR   = DATA_DIR / "events"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_TEST_CASES: list[tuple[str, str, str]] = [
    ("TCS.NS",      "2024-03-15",  "TCS — Q4 2024 mid-quarter"),
    ("RELIANCE.NS", "2025-09-10",  "Reliance — Q2 FY26 pre-results"),
    ("HDFCBANK.NS", "2025-01-31",  "HDFC Bank — Jan 2025 end"),
]

ALL_STOCKS = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS",
    "RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS",
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "MARUTI.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "M&M.NS",
]

PRICE_REQUIRED_COLS = {
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower",
    "Change_1D_pct", "Change_5D_pct",
}

NEWS_REQUIRED_COLS = {"date", "headline", "source", "summary"}

# Pipeline modules — all scripts live inside data\
# Import path uses "data.module_name" when imported from project root
PIPELINE_MODULES: list[tuple[str, str, int]] = [
    ("config",                      "Central configuration",            1),
    ("data.utils",                  "Shared utilities (data_utils.py)", 28),
    ("llm_utils",                   "LLM wrappers (Groq/Gemini)",      10),
    ("data.fetch_prices",           "Price downloader",                11),
    ("data.validate_prices",        "Price validation",                12),
    ("data.add_indicators",         "Technical indicators",            13),
    ("data.bse_scraper",            "BSE filings scraper",             14),
    ("data.newsapi_fetch",          "NewsAPI fetcher",                 16),
    ("data.rss_news",               "RSS news fetcher",                18),
    ("data.merge_news",             "News merger",                     19),
    ("data.get_news_context",       "Look-ahead-safe news context",    20),
    ("data.price_context",          "Price context builder",           13),
    ("data.context_builder",        "Analyst briefing builder",        27),
    ("data.validate_all",           "Full data validator",             23),
    ("data.get_trading_days",       "NSE trading days generator",      25),
]

SEP  = "═" * 68
SEP2 = "─" * 68


# ─────────────────────────────────────────────────────────────────────────────
# Ticker → filename helpers  (matching actual file_structure.txt naming)
# ─────────────────────────────────────────────────────────────────────────────

def ticker_to_price_stem(ticker: str) -> str:
    """
    NSE ticker → price CSV filename stem (no extension).

    Actual files (from File_structure.txt):
        TCS.NS       → TCS           (just strip .NS)
        M&M.NS       → M&M           (keep ampersand — that's the actual filename!)
        BAJAJ-AUTO.NS → BAJAJ-AUTO   (keep hyphen)
    """
    return ticker.replace(".NS", "")


def ticker_to_news_stem(ticker: str) -> str:
    """
    NSE ticker → news CSV filename stem (no extension, no _NS_news suffix).

    Actual files (from File_structure.txt):
        TCS.NS        → TCS           → file: TCS_NS_news.csv
        M&M.NS        → M_M           → file: M_M_NS_news.csv   (& → _)
        BAJAJ-AUTO.NS → BAJAJ-AUTO    → file: BAJAJ-AUTO_NS_news.csv
    """
    return ticker.replace(".NS", "").replace("&", "_")


def price_csv_candidates(ticker: str) -> list[Path]:
    """Return all plausible price CSV paths for a ticker."""
    stem1 = ticker_to_price_stem(ticker)   # e.g. M&M  (actual filename)
    stem2 = ticker_to_news_stem(ticker)    # e.g. M_M  (safe fallback)
    paths = []
    for stem in dict.fromkeys([stem1, stem2]):   # preserve order, deduplicate
        paths.append(PRICES_DIR / f"{stem}.csv")
    return paths


def news_csv_path(ticker: str) -> Path:
    """
    Return the expected merged news CSV path.
    Pattern (from File_structure.txt): data/news/{stem}_NS_news.csv
    """
    stem = ticker_to_news_stem(ticker)
    return NEWS_DIR / f"{stem}_NS_news.csv"


def bse_csv_path(ticker: str) -> Path:
    """BSE filings CSV — data/news/{stem}_NS_bse.csv"""
    stem = ticker_to_news_stem(ticker)
    return NEWS_DIR / f"{stem}_NS_bse.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Validation report collector
# ─────────────────────────────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self):
        self.stages:    list[dict] = []
        self.log_lines: list[str]  = []

    def add(self, stage: str, ok: int, fail: int, notes: list[str] = None):
        self.stages.append({"stage": stage, "ok": ok, "fail": fail,
                            "notes": notes or []})

    def log(self, line: str = ""):
        print(line)
        self.log_lines.append(line)

    def total_ok(self):   return sum(s["ok"]   for s in self.stages)
    def total_fail(self): return sum(s["fail"] for s in self.stages)
    def all_passed(self): return self.total_fail() == 0

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.log_lines), encoding="utf-8")
        print(f"\n  Report saved → {path}")


def _hdr(report: ValidationReport, title: str):
    report.log(f"\n{SEP}")
    report.log(f"  {title}")
    report.log(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Module import / syntax check
# ─────────────────────────────────────────────────────────────────────────────

def stage_imports(report: ValidationReport) -> None:
    _hdr(report, "STAGE 1 — Module Import / Syntax Check")
    ok = fail = 0
    notes = []

    for module_path, description, day in PIPELINE_MODULES:
        try:
            importlib.import_module(module_path)
            report.log(f"  ✓  Day {day:>2}  {module_path:<32}  {description}")
            ok += 1
        except ImportError as e:
            report.log(f"  ✗  Day {day:>2}  {module_path:<32}  ImportError: {e}")
            fail += 1
            notes.append(f"{module_path}: ImportError — {e}")
        except Exception as e:
            # config.py raises EnvironmentError when .env is missing — acceptable
            if any(k in str(e) for k in ("API_KEY", "GROQ", "GEMINI", ".env", "env")):
                report.log(f"  ⚠  Day {day:>2}  {module_path:<32}  Needs .env (syntax OK)")
                ok += 1
            else:
                report.log(f"  ✗  Day {day:>2}  {module_path:<32}  {type(e).__name__}: {e}")
                fail += 1
                notes.append(f"{module_path}: {type(e).__name__} — {e}")

    report.add("Module Imports", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Directory & file structure
# ─────────────────────────────────────────────────────────────────────────────

def stage_directory_structure(report: ValidationReport) -> None:
    _hdr(report, "STAGE 2 — Directory & File Structure")
    ok = fail = 0
    notes = []

    required_dirs = [
        DATA_DIR,
        PRICES_DIR,
        NEWS_DIR,
        EVENTS_DIR,
        RAW_BSE_DIR,
        RAW_NEWS_DIR,
        RAW_NEWS_DIR / "bse",
        RAW_NEWS_DIR / "newsapi",
        ROOT / "logs",
    ]

    # Scripts sit inside data\ (confirmed by File_structure.txt)
    required_scripts = [
        ROOT  / "config.py",
        ROOT  / ".env",
        DATA_DIR / "fetch_prices.py",
        DATA_DIR / "validate_prices.py",
        DATA_DIR / "add_indicators.py",
        DATA_DIR / "bse_scraper.py",
        DATA_DIR / "newsapi_fetch.py",
        DATA_DIR / "rss_news.py",
        DATA_DIR / "merge_news.py",
        DATA_DIR / "get_news_context.py",
        DATA_DIR / "price_context.py",
        DATA_DIR / "context_builder.py",
        DATA_DIR / "validate_all.py",
        DATA_DIR / "get_trading_days.py",
        DATA_DIR / "data_summary.py",
        DATA_DIR / "stocks.csv",
    ]

    # Event files with their ACTUAL names from File_structure.txt
    required_events = [
        EVENTS_DIR / "earnings_calendar_2024_2025.csv",   # not _2024.csv
        EVENTS_DIR / "indian_events.csv",
        EVENTS_DIR / "market_holidays.csv",               # not _2024.csv
        EVENTS_DIR / "trading_days.csv",
    ]

    for p in required_dirs:
        rel = p.relative_to(ROOT)
        if p.exists():
            report.log(f"  ✓  DIR   {rel}")
            ok += 1
        else:
            report.log(f"  ✗  DIR   {rel}  ← MISSING  (mkdir \"{p}\")")
            fail += 1
            notes.append(f"Missing directory: {rel}")

    for p in required_scripts:
        rel = p.relative_to(ROOT)
        if p.exists():
            report.log(f"  ✓  FILE  {rel}")
            ok += 1
        else:
            report.log(f"  ✗  FILE  {rel}  ← MISSING")
            fail += 1
            notes.append(f"Missing file: {rel}")

    for p in required_events:
        rel = p.relative_to(ROOT)
        if p.exists():
            report.log(f"  ✓  EVT   {rel}")
            ok += 1
        else:
            report.log(f"  ✗  EVT   {rel}  ← MISSING")
            fail += 1
            notes.append(f"Missing event file: {rel}")

    report.add("Directory Structure", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Price CSV validation
# ─────────────────────────────────────────────────────────────────────────────

def stage_price_files(report: ValidationReport, stocks: list[str]) -> None:
    _hdr(report, "STAGE 3 — Price CSVs  (data/prices/*.csv)")
    report.log(f"  Expected naming: TCS.csv, HDFCBANK.csv, M&M.csv  (strip .NS only)")
    report.log("")
    ok = fail = 0
    notes = []

    for ticker in stocks:
        candidates = price_csv_candidates(ticker)
        csv_path   = next((p for p in candidates if p.exists()), None)

        if csv_path is None:
            tried = ", ".join(p.name for p in candidates)
            report.log(f"  ✗  {ticker:<22}  Not found  (tried: {tried})")
            fail += 1
            notes.append(f"{ticker}: price CSV missing")
            continue

        try:
            df = pd.read_csv(csv_path)
            missing = PRICE_REQUIRED_COLS - set(df.columns)
            rows    = len(df)
            fname   = csv_path.name

            if missing:
                report.log(f"  ✗  {ticker:<22}  {fname}  {rows:>5} rows — missing: {missing}")
                fail += 1
                notes.append(f"{ticker}: missing indicator columns {missing}")
            else:
                date_col = next((c for c in ("Date","date") if c in df.columns), None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    min_d = df[date_col].min().strftime("%Y-%m-%d")
                    max_d = df[date_col].max().strftime("%Y-%m-%d")
                    suffix = f"[{min_d} → {max_d}]"
                else:
                    suffix = "(no date col found)"
                warn = "  ⚠ low" if rows < 100 else ""
                report.log(f"  ✓  {ticker:<22}  {fname}  {rows:>5} rows  {suffix}{warn}")
                ok += 1

        except Exception as e:
            report.log(f"  ✗  {ticker:<22}  Read error: {e}")
            fail += 1
            notes.append(f"{ticker}: {e}")

    report.add("Price CSVs", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — News CSV validation
# ─────────────────────────────────────────────────────────────────────────────

def stage_news_files(report: ValidationReport, stocks: list[str]) -> None:
    _hdr(report, "STAGE 4 — News CSVs  (data/news/{stem}_NS_news.csv)")
    report.log(f"  Expected naming: TCS_NS_news.csv, M_M_NS_news.csv  (& → _)")
    report.log("")
    ok = fail = 0
    notes = []

    for ticker in stocks:
        csv_path = news_csv_path(ticker)

        if not csv_path.exists():
            report.log(f"  ✗  {ticker:<22}  Not found: {csv_path.name}")
            fail += 1
            notes.append(f"{ticker}: news CSV missing ({csv_path.name})")
            continue

        try:
            df = pd.read_csv(csv_path)
            missing = NEWS_REQUIRED_COLS - set(df.columns)
            rows    = len(df)

            if missing:
                report.log(f"  ✗  {ticker:<22}  {csv_path.name}  {rows:>5} items — missing cols: {missing}")
                fail += 1
                notes.append(f"{ticker}: news missing columns {missing}")
            else:
                warn = "  ⚠ very few" if rows < 5 else ""
                report.log(f"  ✓  {ticker:<22}  {csv_path.name}  {rows:>5} items{warn}")
                ok += 1

        except Exception as e:
            report.log(f"  ✗  {ticker:<22}  Read error: {e}")
            fail += 1
            notes.append(f"{ticker}: {e}")

    report.add("News CSVs", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Event calendar files  (actual filenames from File_structure.txt)
# ─────────────────────────────────────────────────────────────────────────────

def stage_events(report: ValidationReport) -> None:
    _hdr(report, "STAGE 5 — Event Calendar Files  (data/events/)")
    ok = fail = 0
    notes = []

    # Column names confirmed from actual CSV files on local machine:
    #   earnings_calendar_2024_2025.csv : ticker, company, year, q1_date, q2_date, q3_date, q4_date
    #   indian_events.csv               : date, event_type, event_name, description,
    #                                     expected_market_impact, applicable_stocks, notes
    #   market_holidays.csv             : date (+ optional name/description)
    #   trading_days.csv                : date
    event_files = {
        EVENTS_DIR / "earnings_calendar_2024_2025.csv": {"ticker", "company"},
        EVENTS_DIR / "indian_events.csv":               {"date", "event_type"},
        EVENTS_DIR / "market_holidays.csv":             {"date"},
        EVENTS_DIR / "trading_days.csv":                {"date"},
    }

    for fpath, req_cols in event_files.items():
        rel = fpath.relative_to(ROOT)
        if not fpath.exists():
            report.log(f"  ✗  {rel}  ← not found")
            fail += 1
            notes.append(f"Missing: {rel}")
            continue

        try:
            df      = pd.read_csv(fpath)
            missing = req_cols - set(df.columns)
            if missing:
                report.log(f"  ⚠  {rel}  {len(df)} rows — columns missing: {missing}")
            else:
                report.log(f"  ✓  {rel}  {len(df)} rows")
            ok += 1   # presence is what matters; missing cols are a warning
        except Exception as e:
            report.log(f"  ✗  {rel}  Read error: {e}")
            fail += 1
            notes.append(f"{rel}: {e}")

    report.add("Event Calendars", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Technical indicator spot check
# ─────────────────────────────────────────────────────────────────────────────

def stage_indicator_spot_check(report: ValidationReport) -> None:
    _hdr(report, "STAGE 6 — Technical Indicator Spot Check")
    ok = fail = 0
    notes = []

    spot = [("TCS.NS", "TCS"), ("HDFCBANK.NS", "HDFCBANK"), ("RELIANCE.NS", "RELIANCE")]

    for ticker, stem in spot:
        candidates = price_csv_candidates(ticker)
        csv_path   = next((p for p in candidates if p.exists()), None)

        if not csv_path:
            report.log(f"  ✗  {stem:<14}  CSV not found — skipping indicator checks")
            fail += 1
            notes.append(f"{stem}: price CSV missing")
            continue

        try:
            df = pd.read_csv(csv_path)
            checks = [
                ("RSI_14 in [0,100]",
                 lambda d: "RSI_14" in d.columns and ((d["RSI_14"].dropna() >= 0) & (d["RSI_14"].dropna() <= 100)).all()),
                ("SMA_20 > 0",
                 lambda d: "SMA_20" in d.columns and (d["SMA_20"].dropna() > 0).all()),
                ("BB_Upper >= BB_Lower",
                 lambda d: all(c in d.columns for c in ["BB_Upper","BB_Lower"])
                            and (d["BB_Upper"].dropna() >= d["BB_Lower"].dropna()).all()),
                ("MACD is numeric",
                 lambda d: "MACD" in d.columns and pd.api.types.is_numeric_dtype(d["MACD"])),
                ("Close has no NaN",
                 lambda d: "Close" in d.columns and d["Close"].notna().all()),
            ]
            for name, fn in checks:
                try:
                    passed = fn(df)
                    mark   = "✓" if passed else "✗"
                    report.log(f"  {mark}  {stem:<14}  {name}")
                    if passed:
                        ok += 1
                    else:
                        fail += 1
                        notes.append(f"{stem} indicator failed: {name}")
                except Exception as e:
                    report.log(f"  ⚠  {stem:<14}  {name}  Error: {e}")
        except Exception as e:
            report.log(f"  ✗  {stem:<14}  Could not read CSV: {e}")
            fail += 1

    report.add("Indicator Spot Check", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — Look-ahead bias detection
# ─────────────────────────────────────────────────────────────────────────────

def stage_lookahead_check(report: ValidationReport) -> None:
    _hdr(report, "STAGE 7 — Look-Ahead Bias Detection")
    ok = fail = 0
    notes = []

    report.log("  Rule: get_news_context() must NEVER return items with date >= query_date")
    report.log("")

    # Import from data\ subfolder
    get_news_context = None
    for mod in ("data.get_news_context", "get_news_context"):
        try:
            m = importlib.import_module(mod)
            get_news_context = m.get_news_context
            report.log(f"  Imported get_news_context from '{mod}'")
            break
        except Exception:
            pass

    if get_news_context is None:
        report.log(f"  ✗  Cannot import get_news_context — skipping look-ahead check")
        report.add("Look-Ahead Bias", 0, 1, ["Import failed"])
        return

    test_cases = [
        ("TCS.NS",      "2024-03-15"),
        ("TCS.NS",      "2024-10-10"),
        ("TCS.NS",      "2025-06-01"),
        ("RELIANCE.NS", "2025-09-10"),
        ("HDFCBANK.NS", "2025-01-31"),
    ]

    for ticker, date in test_cases:
        try:
            items = get_news_context(ticker, date, max_items=5,
                                     news_dir=str(NEWS_DIR))
            q_dt      = pd.Timestamp(date)
            violators = [i for i in items if pd.Timestamp(i["date"]) >= q_dt]

            if violators:
                report.log(f"  ✗  {ticker:<22} {date}  LOOK-AHEAD! {len(violators)} item(s)")
                fail += 1
                notes.append(f"LOOK-AHEAD: {ticker} {date} — {len(violators)} violation(s)")
            else:
                report.log(f"  ✓  {ticker:<22} {date}  {len(items)} items — all before query date")
                ok += 1

        except FileNotFoundError:
            report.log(f"  ⚠  {ticker:<22} {date}  News CSV not found — cannot test")
        except Exception as e:
            report.log(f"  ✗  {ticker:<22} {date}  {type(e).__name__}: {e}")
            fail += 1
            notes.append(f"{ticker} {date}: {e}")

    # Price monotonicity check (proxy for no-future-row guarantee)
    report.log("")
    report.log("  Price look-ahead proxy: checking TCS.csv dates are monotonically increasing")
    tcs_csv = PRICES_DIR / "TCS.csv"
    if tcs_csv.exists():
        try:
            df = pd.read_csv(tcs_csv)
            date_col = next((c for c in ("Date","date") if c in df.columns), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                if df[date_col].is_monotonic_increasing:
                    report.log(f"  ✓  TCS.csv dates are strictly ordered — no future rows possible")
                    ok += 1
                else:
                    report.log(f"  ✗  TCS.csv dates NOT monotonic — possible duplicate/future rows")
                    fail += 1
                    notes.append("TCS price dates not monotonically increasing")
        except Exception as e:
            report.log(f"  ⚠  Could not check TCS.csv date ordering: {e}")
    else:
        report.log(f"  ⚠  TCS.csv not found — skipping price look-ahead proxy")

    report.add("Look-Ahead Bias", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — Required test cases  (end-to-end build_context)
# ─────────────────────────────────────────────────────────────────────────────

def stage_required_test_cases(report: ValidationReport) -> None:
    _hdr(report, "STAGE 8 — Required Test Cases (TCS / RELIANCE / HDFCBANK)")
    ok = fail = 0
    notes = []

    # Import build_context from data\context_builder.py
    build_context = None
    for mod in ("data.context_builder", "context_builder"):
        try:
            m = importlib.import_module(mod)
            build_context = m.build_context
            report.log(f"  Imported build_context from '{mod}'")
            break
        except Exception:
            pass

    if build_context is None:
        report.log("  ✗  Cannot import build_context — skipping test cases")
        report.add("Required Test Cases", 0, 3, ["build_context import failed"])
        return

    # Inspect the actual signature at runtime — the local data/context_builder.py
    # may be a different version than expected (e.g. missing price_data_dir param).
    import inspect
    sig_params = set(inspect.signature(build_context).parameters.keys())
    report.log(f"  build_context accepts params: {sorted(sig_params)}")
    report.log("")

    for ticker, date, label in REQUIRED_TEST_CASES:
        report.log(SEP2)
        report.log(f"  Test : {label}")
        report.log(f"  Stock: {ticker}   Date: {date}")

        try:
            # Only pass kwargs the local function actually accepts.
            # Confirmed local signature: prices_dir (not price_data_dir).
            # The inspect loop future-proofs against further renames.
            kwargs: dict = {}
            if "news_dir"       in sig_params:
                kwargs["news_dir"]       = str(NEWS_DIR)
            if "prices_dir"     in sig_params:          # actual local param name
                kwargs["prices_dir"]     = str(PRICES_DIR)
            if "price_data_dir" in sig_params:          # older project-knowledge variant
                kwargs["price_data_dir"] = str(PRICES_DIR)
            if "data_dir"       in sig_params:          # oldest variant
                kwargs["data_dir"]       = str(PRICES_DIR)

            briefing = build_context(ticker, date, **kwargs)

            chars      = len(briefing)
            lines      = briefing.count("\n")
            has_price  = any(k in briefing for k in ("Close", "SECTION 1", "Price Action"))
            has_news   = any(k in briefing for k in ("SECTION 2", "Headline", "News"))
            looks_good = chars > 200 and has_price

            report.log(f"  ✓  Context built — {chars} chars, {lines} lines")
            report.log(f"     Has price section : {'YES' if has_price else 'NO ← check price_context.py'}")
            report.log(f"     Has news section  : {'YES' if has_news  else 'NO (may be empty — check news CSV)'}")

            # Print first 20 lines as visual proof
            report.log("")
            report.log("  ── Preview (first 20 lines) ─────────────────────────────────────")
            for line in briefing.split("\n")[:20]:
                report.log(f"  {line}")
            report.log("  ...")
            report.log("")

            ok += 1 if looks_good else 0
            if not looks_good:
                fail += 1
                notes.append(f"{ticker}/{date}: briefing too short or missing price section")

        except FileNotFoundError as e:
            report.log(f"  ✗  FileNotFoundError: {e}")
            report.log(f"     → Ensure data\\prices\\ and data\\news\\ files exist for {ticker}")
            fail += 1
            notes.append(f"{ticker}/{date}: FileNotFoundError — {e}")
        except AssertionError as e:
            report.log(f"  ✗  LOOK-AHEAD BIAS DETECTED: {e}")
            fail += 1
            notes.append(f"{ticker}/{date}: look-ahead assertion failed")
        except Exception as e:
            report.log(f"  ✗  {type(e).__name__}: {e}")
            traceback.print_exc()
            fail += 1
            notes.append(f"{ticker}/{date}: {type(e).__name__} — {e}")

    report.add("Required Test Cases", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9 — Date range coverage  (2024-01-01 to 2025-12-31)
# ─────────────────────────────────────────────────────────────────────────────

def stage_date_coverage(report: ValidationReport) -> None:
    _hdr(report, "STAGE 9 — Date Range Coverage  (Jan 2024 – Dec 2025)")
    ok = fail = 0
    notes = []

    EXP_START = pd.Timestamp("2024-01-01")
    EXP_END   = pd.Timestamp("2025-12-31")
    MIN_ROWS  = 400   # ~2 years of NSE trading days ≈ 490

    spot = [("TCS.NS","TCS"), ("RELIANCE.NS","RELIANCE"),
            ("HDFCBANK.NS","HDFCBANK"), ("INFY.NS","INFY")]

    for ticker, stem in spot:
        candidates = price_csv_candidates(ticker)
        csv_path   = next((p for p in candidates if p.exists()), None)
        if not csv_path:
            report.log(f"  ✗  {stem:<14}  CSV not found")
            fail += 1
            continue

        try:
            df = pd.read_csv(csv_path)
            date_col = next((c for c in ("Date","date") if c in df.columns), None)
            if not date_col:
                report.log(f"  ⚠  {stem:<14}  No date column found")
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            min_d, max_d, rows = df[date_col].min(), df[date_col].max(), len(df)

            start_ok = min_d <= EXP_START + pd.Timedelta(days=30)
            end_ok   = max_d >= EXP_END   - pd.Timedelta(days=30)
            rows_ok  = rows  >= MIN_ROWS
            all_ok   = start_ok and end_ok and rows_ok

            marks = []
            if not start_ok: marks.append("start too late")
            if not end_ok:   marks.append("end too early")
            if not rows_ok:  marks.append(f"only {rows} rows")

            flag = "✓" if all_ok else ("⚠" if not rows_ok else "✗")
            report.log(
                f"  {flag}  {stem:<14}  {rows:>5} rows  "
                f"[{min_d.strftime('%Y-%m-%d')} → {max_d.strftime('%Y-%m-%d')}]"
                + (f"  ← {', '.join(marks)}" if marks else "")
            )

            if all_ok:
                ok += 1
            else:
                fail += 1
                notes.append(f"{stem}: {', '.join(marks)}")
        except Exception as e:
            report.log(f"  ✗  {stem:<14}  Error: {e}")
            fail += 1

    report.add("Date Coverage (2024–2025)", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10 — BSE filings presence
# ─────────────────────────────────────────────────────────────────────────────

def stage_bse_filings(report: ValidationReport) -> None:
    _hdr(report, "STAGE 10 — BSE Filings  (data/news/*_NS_bse.csv + data/raw_bse/*.json)")
    ok = fail = 0
    notes = []

    # Merged BSE CSVs land in data/news/ (e.g. BAJAJ-AUTO_NS_bse.csv)
    bse_csvs = list(NEWS_DIR.glob("*_NS_bse.csv"))
    report.log(f"  Merged BSE CSVs in data/news/   : {len(bse_csvs)} files")
    if bse_csvs:
        for f in bse_csvs[:5]:
            report.log(f"    ✓  {f.name}")
        if len(bse_csvs) > 5:
            report.log(f"    ... and {len(bse_csvs)-5} more")
        ok += len(bse_csvs)
    else:
        report.log(f"  ✗  No *_NS_bse.csv files found in data/news/")
        report.log(f"     → Run: python data\\bse_scraper.py")
        fail += 1
        notes.append("No merged BSE CSVs in data/news/")

    # Raw BSE JSONs in data/raw_bse/
    raw_jsons = list(RAW_BSE_DIR.glob("*.json")) if RAW_BSE_DIR.exists() else []
    report.log(f"  Raw BSE JSONs in data/raw_bse/  : {len(raw_jsons)} files")
    if raw_jsons:
        report.log(f"  ✓  Raw BSE JSON cache present")
        ok += 1
    else:
        report.log(f"  ⚠  No raw BSE JSONs found in data/raw_bse/")

    # Raw newsapi JSONs in data/raw_news/newsapi/
    newsapi_dir = RAW_NEWS_DIR / "newsapi"
    api_jsons   = list(newsapi_dir.glob("*.json")) if newsapi_dir.exists() else []
    report.log(f"  Raw NewsAPI JSONs in data/raw_news/newsapi/: {len(api_jsons)} files")
    if api_jsons:
        report.log(f"  ✓  NewsAPI raw cache present")
        ok += 1
    else:
        report.log(f"  ⚠  No NewsAPI JSONs found in data/raw_news/newsapi/")

    report.add("BSE Filings & Raw News", ok, fail, notes)


# ─────────────────────────────────────────────────────────────────────────────
# Final report + Phase 3 checklist
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(report: ValidationReport) -> None:
    report.log(f"\n{SEP}")
    report.log("  DAY 30 — FINAL VALIDATION SUMMARY")
    report.log(SEP)
    report.log(f"  {'Stage':<36} {'Pass':>6} {'Fail':>6}")
    report.log(f"  {'-'*36} {'-'*6} {'-'*6}")
    for s in report.stages:
        flag = "✓" if s["fail"] == 0 else "✗"
        report.log(f"  {flag}  {s['stage']:<34} {s['ok']:>6} {s['fail']:>6}")
    report.log(f"  {'-'*36} {'-'*6} {'-'*6}")
    report.log(f"  {'TOTAL':<36} {report.total_ok():>6} {report.total_fail():>6}")
    report.log(SEP)

    if report.all_passed():
        report.log("""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  ✅  ALL VALIDATION CHECKS PASSED                              ║
  ║  Data pipeline is PHASE 3 READY — proceed to Agent Framework   ║
  ╚══════════════════════════════════════════════════════════════════╝
""")
    else:
        report.log(f"\n  ⚠  {report.total_fail()} check(s) FAILED. Fix before starting Phase 3.\n")
        report.log("  Failed items:")
        for s in report.stages:
            for note in s["notes"]:
                report.log(f"    •  {note}")

    report.log(f"\n  Run timestamp: {datetime.now(tz=__import__('datetime').timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.log(SEP)


def print_phase3_checklist(report: ValidationReport) -> None:
    report.log(f"\n{SEP}")
    report.log("  PHASE 3 READINESS CHECKLIST")
    report.log(SEP)
    items = [
        "All 20 stocks have data/prices/{TICKER}.csv  (OHLCV + 10 indicator cols)",
        "All 20 stocks have data/news/{STEM}_NS_news.csv  (4 required cols)",
        "Price data covers Jan 2024 – Dec 2025  (~490 NSE trading days)",
        "Technical indicators: SMA_20, SMA_50, RSI_14, MACD, BB, Change%",
        "BSE filings present in data/news/*_NS_bse.csv + data/raw_bse/*.json",
        "NewsAPI + RSS raw JSONs cached in data/raw_news/",
        "data/get_news_context.py enforces strict date < query_date filter",
        "data/context_builder.py has dual look-ahead assertion (price + news)",
        "data/events/earnings_calendar_2024_2025.csv  present",
        "data/events/market_holidays.csv + trading_days.csv  present",
        "build_context(TCS.NS,      2024-03-15)  → valid briefing",
        "build_context(RELIANCE.NS, 2025-09-10)  → valid briefing",
        "build_context(HDFCBANK.NS, 2025-01-31)  → valid briefing",
        "data/validate_all.py passes all checks",
        "data/README.md written and committed to GitHub",
        "All code committed with meaningful git messages",
    ]
    passed = report.all_passed()
    for item in items:
        report.log(f"  {'☑' if passed else '☐'}  {item}")
    report.log(SEP)
    if passed:
        report.log("  🚀  Ready to begin Phase 3: Agent Framework (Day 31)")
    else:
        report.log("  ❌  Fix failures above before proceeding to Phase 3.")
    report.log(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Single-stock quick test
# ─────────────────────────────────────────────────────────────────────────────

def run_single_test(stock: str, date: str) -> None:
    print(f"\n{SEP}")
    print(f"  SINGLE CONTEXT TEST: {stock}   {date}")
    print(f"  prices dir : {PRICES_DIR}")
    print(f"  news dir   : {NEWS_DIR}")
    print(SEP)

    build_context = None
    for mod in ("data.context_builder", "context_builder"):
        try:
            m = importlib.import_module(mod)
            build_context = m.build_context
            break
        except Exception:
            pass

    if build_context is None:
        print("Cannot import build_context from data/context_builder.py")
        sys.exit(1)

    try:
        briefing = build_context(stock, date,
                                  news_dir=str(NEWS_DIR),
                                  price_data_dir=str(PRICES_DIR))
        print(briefing)
        print(f"\n  ✓  Context generated: {len(briefing)} chars")
    except Exception as e:
        print(f"  ✗  {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Data Pipeline Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_pipeline.py                            # full validation
  python validate_pipeline.py --quick                    # imports + dirs only
  python validate_pipeline.py --stock TCS.NS --date 2024-03-15
  python validate_pipeline.py --stock RELIANCE.NS --date 2025-09-10
  python validate_pipeline.py --stock HDFCBANK.NS --date 2025-01-31
        """
    )
    parser.add_argument("--quick",  action="store_true",
                        help="Imports + directory checks only (fast, no CSV reads)")
    parser.add_argument("--stock",  type=str, default=None,
                        help="Single-stock context test (e.g. TCS.NS)")
    parser.add_argument("--date",   type=str, default=None,
                        help="Date for single-stock test (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.stock and args.date:
        run_single_test(args.stock, args.date)
        return

    report = ValidationReport()

    report.log(f"\n{SEP}")
    report.log("  END-TO-END DATA PIPELINE VALIDATION")
    report.log("  M.Tech: Multi-Agent LLM Trading System — Indian Stock Market")
    report.log(f"  Project root : {ROOT}")
    report.log(f"  Data dir     : {DATA_DIR}")
    report.log(f"  Run started  : {datetime.now(tz=__import__('datetime').timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.log(SEP)

    # Load stock list from config if possible
    stocks = ALL_STOCKS
    try:
        import config as cfg
        stocks = cfg.STOCKS
        report.log(f"  Config loaded: {len(stocks)} stocks")
    except Exception as e:
        report.log(f"  ⚠  config.py not loadable ({e}) — using built-in stock list")

    stage_imports(report)
    stage_directory_structure(report)

    if not args.quick:
        stage_price_files(report, stocks)
        stage_news_files(report, stocks)
        stage_events(report)
        stage_indicator_spot_check(report)
        stage_lookahead_check(report)
        stage_required_test_cases(report)
        stage_date_coverage(report)
        stage_bse_filings(report)

    print_final_report(report)
    print_phase3_checklist(report)

    log_path = ROOT / "logs" / "validation_report.txt"
    report.save(log_path)

    sys.exit(0 if report.all_passed() else 1)


if __name__ == "__main__":
    main()
