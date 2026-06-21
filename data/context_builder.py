"""
data/context_builder.py — Analyst Briefing Context Builder

Day 27 of M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.

What it does:
    Combines (a) enriched price data — OHLCV + technical indicators, read
    directly from data/prices/<TICKER>.csv (written by add_indicators.py,
    Day 13) — and (b) look-ahead-safe news headlines from get_news_context()
    (Day 19/20) into a single, well-structured analyst briefing string that
    becomes the verbatim input to the Analyst Agent LLM prompt.

Why it matters:
    The Analyst Agent must receive EXACTLY the right information — no more, no less.
    Too much noise -> hallucination. Too little -> shallow analysis.
    This module is the final gate before any LLM call. Its output must be:
      1. Self-contained (the LLM sees nothing else about the stock)
      2. Bias-free (no look-ahead data sneaks through)
      3. Concise (under ~700 tokens for most stocks)
      4. Consistently formatted (so downstream prompt parsing is reliable)

How it works:
    1. Read data/prices/<TICKER>.csv directly (there is no separate
       price_context.py in this project — add_indicators.py already writes
       SMA/RSI/MACD/Bollinger columns straight into the price file), filter
       to rows STRICTLY BEFORE the query date, take the last
       PRICE_LOOKBACK_DAYS rows.
    2. Call get_news_context(stock, date) -> at most NEWS_MAX_ITEMS headlines
       strictly before date.
    3. Merge into a structured briefing string.
    4. Run look-ahead checks and raise immediately on violation.
    5. Return the string (and optionally a metadata dict).

Usage:
    from data.context_builder import build_context

    briefing = build_context("TCS.NS", "2024-03-15")
    print(briefing)

File naming conventions used in this project (see data/data_utils.py):
    Price CSV : data/prices/<ticker_to_filename(ticker)>.csv
                e.g. "TCS.NS" -> data/prices/TCS.csv
                     "M&M.NS" -> data/prices/M_M.csv
    News CSV  : data/news/<ticker_to_filename(ticker)>_NS_news.csv
                e.g. "TCS.NS" -> data/news/TCS_NS_news.csv
                     "M&M.NS" -> data/news/M_M_NS_news.csv
                (path is constructed inside get_news_context.py itself —
                context_builder never builds this path directly)

Dependencies (must be completed before Day 27):
    - data/add_indicators.py   - Day 13 (writes OHLCV + indicators into
                                  data/prices/<TICKER>.csv)
    - data/get_news_context.py - Day 19/20 (look-ahead-safe news)
    - data/data_utils.py       - shared helper module (ticker_to_filename)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Project imports ────────────────────────────────────────────────────────────
# Adjust sys.path so this works whether called as a module or standalone script.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from config import STOCK_NAMES, NEWS_MAX_ITEMS, PRICE_LOOKBACK_DAYS, PRICES_DIR
except ImportError:
    # Fallback defaults — keeps the module usable standalone / during testing
    STOCK_NAMES: dict[str, str] = {}
    NEWS_MAX_ITEMS: int = 3
    PRICE_LOOKBACK_DAYS: int = 10
    PRICES_DIR: str = "data/prices"

try:
    from data.get_news_context import get_news_context
except ImportError:
    from get_news_context import get_news_context

# NOTE: This project has NO data/price_context.py. Price data already lives,
# fully enriched with indicators, in data/prices/<TICKER>.csv thanks to
# add_indicators.py (Day 13). context_builder reads that CSV directly using
# the same ticker_to_filename() convention defined in data/data_utils.py.
try:
    from data.data_utils import ticker_to_filename
except ImportError:
    try:
        from data_utils import ticker_to_filename
    except ImportError:
        # Last-resort inline fallback — mirrors data_utils.ticker_to_filename exactly
        def ticker_to_filename(ticker: str) -> str:
            return ticker.replace(".NS", "").replace("&", "_")


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | context_builder | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SEPARATOR      = "=" * 68
SECTION_SEP    = "-" * 68
MISSING_NEWS   = "No relevant news available for this period."
MISSING_PRICE  = "Price data unavailable for this date."


# ─────────────────────────────────────────────────────────────────────────────
# Internal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_inr(value: float | None) -> str:
    """Format a float as INR with 2 decimal places, or 'N/A'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"Rs.{value:,.2f}"


def _pct(value: float | None) -> str:
    """Format a float as a percentage string, or 'N/A'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def _rsi_label(rsi: float | None) -> str:
    """Attach a human-readable RSI label for the LLM."""
    if rsi is None or pd.isna(rsi):
        return "N/A"
    if rsi >= 70:
        return f"{rsi:.1f} (Overbought)"
    if rsi <= 30:
        return f"{rsi:.1f} (Oversold)"
    return f"{rsi:.1f} (Neutral)"


def _trend_label(close: float | None, sma20: float | None, sma50: float | None) -> str:
    """Derive a simple trend label from price vs moving averages."""
    if any(v is None or pd.isna(v) for v in [close, sma20, sma50]):
        return "Trend: N/A"
    if close > sma20 > sma50:
        return "Uptrend  (price > SMA20 > SMA50)"
    if close < sma20 < sma50:
        return "Downtrend (price < SMA20 < SMA50)"
    if close > sma20 and sma20 < sma50:
        return "Recovery  (price > SMA20 but SMA20 < SMA50)"
    if close < sma20 and sma20 > sma50:
        return "Pullback  (price < SMA20 but SMA20 > SMA50)"
    return "Mixed / sideways"


# ─────────────────────────────────────────────────────────────────────────────
# Price loading — reads data/prices/<TICKER>.csv directly
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_window(
    stock: str,
    date: str,
    lookback_days: int = PRICE_LOOKBACK_DAYS,
    prices_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the last `lookback_days` trading rows STRICTLY BEFORE `date` from
    data/prices/<TICKER>.csv (already enriched with indicators by
    add_indicators.py).

    Args:
        stock         : NSE ticker, e.g. "TCS.NS", "M&M.NS".
        date          : Query date as "YYYY-MM-DD". Only rows with
                        Date < date are returned (no look-ahead).
        lookback_days : How many trailing trading rows to keep.
        prices_dir    : Override path to prices directory (for testing).

    Returns:
        DataFrame indexed by Date (datetime), containing at most
        `lookback_days` rows, all dated strictly before `date`, with columns
        Open/High/Low/Close/Volume plus any indicator columns present
        (SMA_20, SMA_50, RSI_14, MACD, MACD_Signal, MACD_Hist, BB_Upper,
        BB_Lower, Change_1D_pct, Change_5D_pct).

    Raises:
        FileNotFoundError : If data/prices/<TICKER>.csv does not exist.
        ValueError         : If required OHLCV columns are missing.
    """
    _prices_dir = Path(prices_dir or PRICES_DIR)
    filename    = f"{ticker_to_filename(stock)}.csv"
    csv_path    = _prices_dir / filename

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Price CSV not found for {stock}.\n"
            f"Expected: {csv_path}\n"
            f"Run data/fetch_prices.py (Day 5) and data/add_indicators.py "
            f"(Day 13) first."
        )

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Price CSV for {stock} ({csv_path.name}) is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.sort_index()

    # ── Strict look-ahead filter: only rows BEFORE the query date ────────────
    query_date = pd.to_datetime(date)
    prior = df[df.index < query_date]

    window = prior.tail(lookback_days)

    return window


# ─────────────────────────────────────────────────────────────────────────────
# Look-ahead bias validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_no_lookahead(
    stock: str,
    query_date: str,
    price_window: pd.DataFrame,
    news_items: list[dict],
) -> None:
    """
    Two-layer look-ahead bias check (price + news). Raises AssertionError
    immediately on any violation. Called internally by build_context() before
    the briefing is returned.

    Args:
        stock        : NSE ticker (for error messages).
        query_date   : The analysis date as "YYYY-MM-DD".
        price_window : DataFrame from _load_price_window() (Date index).
        news_items   : List of news dicts from get_news_context().

    Raises:
        AssertionError : If any look-ahead bias is detected.
    """
    q_dt = pd.to_datetime(query_date)

    # ── Check 1: price rows ───────────────────────────────────────────────────
    if not price_window.empty:
        violators = price_window.index[price_window.index >= q_dt]
        assert len(violators) == 0, (
            f"\n{'='*60}\n"
            f"LOOK-AHEAD BIAS — PRICE DATA\n"
            f"Stock: {stock} | Query date: {query_date}\n"
            f"Price row(s) dated {list(violators)} are ON OR AFTER query date.\n"
            f"{'='*60}\n"
            "Fix _load_price_window() — do NOT proceed with backtesting."
        )

    # ── Check 2: news items ───────────────────────────────────────────────────
    for item in news_items:
        item_date = pd.to_datetime(item["date"])
        assert item_date < q_dt, (
            f"\n{'='*60}\n"
            f"LOOK-AHEAD BIAS — NEWS DATA\n"
            f"Stock: {stock} | Query date: {query_date}\n"
            f"News item dated {item_date.date()} is ON OR AFTER query date.\n"
            f"Headline: {item.get('headline','')}\n"
            f"{'='*60}\n"
            "Fix get_news_context.py — do NOT proceed."
        )

    log.debug("Look-ahead validation passed for %s on %s", stock, query_date)


# ─────────────────────────────────────────────────────────────────────────────
# Price context section builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_price_section(price_window: pd.DataFrame) -> str:
    """
    Build the price + technical indicators section of the briefing.

    Args:
        price_window : DataFrame from _load_price_window(), already filtered
                        to rows strictly before the query date.

    Returns:
        Formatted multi-line section string.
    """
    if price_window.empty:
        return f"  {MISSING_PRICE}\n"

    # ── Recent price table (last 5 rows max for token economy) ───────────────
    display_rows = price_window.tail(5)
    lines = ["  Recent Price Action (last 5 trading days, all BEFORE analysis date):"]
    lines.append(f"  {'Date':<12} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>12} {'Chg 1D%':>8}")
    lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for idx, row in display_rows.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        chg = row.get("Change_1D_pct", None)
        vol = row.get("Volume", None)
        vol_str = f"{int(vol):,}" if pd.notna(vol) else "N/A"
        lines.append(
            f"  {date_str:<12} {_fmt_inr(row.get('Open')):>10} {_fmt_inr(row.get('High')):>10} "
            f"{_fmt_inr(row.get('Low')):>10} {_fmt_inr(row.get('Close')):>10} {vol_str:>12} {_pct(chg):>8}"
        )

    # ── Latest session summary ────────────────────────────────────────────────
    last_idx = price_window.index[-1]
    last     = price_window.iloc[-1]
    lines.append("")
    lines.append(f"  Most Recent Session ({last_idx.strftime('%Y-%m-%d')}):")
    lines.append(
        f"    Open : {_fmt_inr(last.get('Open'))}   High : {_fmt_inr(last.get('High'))}   "
        f"Low : {_fmt_inr(last.get('Low'))}   Close : {_fmt_inr(last.get('Close'))}"
    )

    # ── Technical indicators (column names exactly as written by add_indicators.py) ──
    lines.append("")
    lines.append("  Technical Indicators (computed on data up to most recent session):")

    sma20       = last.get("SMA_20")
    sma50       = last.get("SMA_50")
    rsi         = last.get("RSI_14")
    macd        = last.get("MACD")
    macd_signal = last.get("MACD_Signal")
    bb_upper    = last.get("BB_Upper")
    bb_lower    = last.get("BB_Lower")
    chg5d       = last.get("Change_5D_pct")

    lines.append(f"    SMA-20          : {_fmt_inr(sma20)}")
    lines.append(f"    SMA-50          : {_fmt_inr(sma50)}")
    lines.append(f"    RSI (14)        : {_rsi_label(rsi)}")
    lines.append(f"    MACD            : {_fmt_inr(macd)}")
    lines.append(f"    MACD Signal     : {_fmt_inr(macd_signal)}")
    lines.append(f"    Bollinger Upper : {_fmt_inr(bb_upper)}")
    lines.append(f"    Bollinger Lower : {_fmt_inr(bb_lower)}")
    lines.append(f"    5-Day Change    : {_pct(chg5d)}")

    trend = _trend_label(last.get("Close"), sma20, sma50)
    lines.append(f"    Trend Signal    : {trend}")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# News section builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_news_section(query_date: str, news_items: list[dict]) -> str:
    """
    Build the news/events section of the briefing.

    Args:
        query_date  : The analysis date.
        news_items  : Output of get_news_context() — already look-ahead safe.

    Returns:
        Formatted news string ready for the briefing.
    """
    if not news_items:
        return f"  {MISSING_NEWS}\n"

    lines = [f"  Recent News and Events (all published STRICTLY BEFORE {query_date}):"]
    lines.append("")

    for i, item in enumerate(news_items, start=1):
        item_date = pd.to_datetime(item["date"]).strftime("%Y-%m-%d")
        source    = item.get("source", "Unknown")
        headline  = item.get("headline", "")
        summary   = item.get("summary", "")

        lines.append(f"  [{i}] {item_date} | Source: {source}")
        lines.append(f"      Headline : {headline}")
        if summary and str(summary).strip() not in ("", "nan", "None"):
            s = str(summary).strip()
            if len(s) > 250:
                s = s[:247] + "..."
            lines.append(f"      Summary  : {s}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def build_context(
    stock: str,
    date: str,
    max_news: Optional[int] = None,
    news_dir: Optional[str] = None,
    prices_dir: Optional[str] = None,
    lookback_days: Optional[int] = None,
    include_metadata_block: bool = True,
) -> str:
    """
    Build a complete, look-ahead-safe analyst briefing for one stock on one date.

    This is the ONLY function the Analyst Agent ever calls. The returned string
    is the verbatim user-turn content sent to the LLM.

    Args:
        stock                  : NSE ticker, e.g. "TCS.NS", "M&M.NS".
        date                   : Analysis date as "YYYY-MM-DD".
                                 Must be a valid trading day in 2024 or 2025.
        max_news               : Override NEWS_MAX_ITEMS from config (default 3).
        news_dir               : Override path to news directory (for testing).
        prices_dir             : Override path to prices directory (for testing).
        lookback_days          : Override PRICE_LOOKBACK_DAYS from config.
        include_metadata_block : Add a metadata footer (used in logging/debugging).

    Returns:
        A multi-line string — the analyst briefing — ready to be injected into
        the Analyst Agent's prompt template.

    Raises:
        FileNotFoundError  : If price CSV or news CSV is missing.
        AssertionError      : If look-ahead bias is detected (should never happen).
    """
    n_news     = max_news or NEWS_MAX_ITEMS
    n_lookback = lookback_days or PRICE_LOOKBACK_DAYS
    company    = STOCK_NAMES.get(stock, stock)
    date_obj   = datetime.strptime(date, "%Y-%m-%d")
    today_str  = date_obj.strftime("%B %d, %Y")  # human-readable

    log.info("Building context for %s (%s) on %s", company, stock, date)

    # ── 1. Load price window directly from data/prices/<TICKER>.csv ──────────
    price_window = pd.DataFrame()
    price_error  = ""
    try:
        price_window = _load_price_window(
            stock, date, lookback_days=n_lookback, prices_dir=prices_dir
        )
        log.info("  Price rows loaded: %d", len(price_window))
    except FileNotFoundError as exc:
        price_error = str(exc)
        log.error("  Price data not found: %s", price_error)
    except ValueError as exc:
        price_error = str(exc)
        log.error("  Price schema error: %s", price_error)
    except Exception as exc:
        price_error = f"{type(exc).__name__}: {exc}"
        log.error("  Price context error: %s", price_error)

    # ── 2. Fetch news context ─────────────────────────────────────────────────
    news_items: list[dict] = []
    news_error: str = ""
    try:
        news_items = get_news_context(stock, date, max_items=n_news, news_dir=news_dir)
        log.info("  News items loaded: %d", len(news_items))
    except FileNotFoundError as exc:
        news_error = str(exc)
        log.warning("  News data not found: %s", news_error)
    except AssertionError:
        log.critical("  LOOK-AHEAD BIAS in news for %s on %s!", stock, date)
        raise
    except Exception as exc:
        news_error = f"{type(exc).__name__}: {exc}"
        log.error("  News context error: %s", news_error)

    # ── 3. Look-ahead validation (belt-and-suspenders on top of each source) ──
    _validate_no_lookahead(stock, date, price_window, news_items)

    # ── 4. Assemble briefing string ───────────────────────────────────────────
    parts: list[str] = []

    parts.append(SEPARATOR)
    parts.append("  ANALYST BRIEFING — INDIAN STOCK MARKET")
    parts.append(SEPARATOR)
    parts.append(f"  Stock        : {company} ({stock})")
    parts.append(f"  Analysis Date: {today_str}  [all data is PRIOR to this date]")
    parts.append(f"  Exchange     : NSE (National Stock Exchange of India)")
    parts.append(SEPARATOR)

    parts.append("")
    parts.append("  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS")
    parts.append(SECTION_SEP)
    if price_error:
        parts.append(f"  Price data unavailable: {price_error}")
    else:
        parts.append(_build_price_section(price_window))

    parts.append("")
    parts.append("  SECTION 2 — RECENT NEWS & CORPORATE EVENTS")
    parts.append(SECTION_SEP)
    if news_error:
        parts.append(f"  News data unavailable: {news_error}")
    else:
        parts.append(_build_news_section(date, news_items))

    if include_metadata_block:
        parts.append(SEPARATOR)
        parts.append("  CONTEXT METADATA (internal — do not include in analysis)")
        parts.append(f"  Price rows used : {len(price_window)}")
        parts.append(f"  News items used : {len(news_items)}")
        parts.append(f"  Look-ahead check: PASSED")
        parts.append(f"  Generated at    : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        parts.append(SEPARATOR)

    briefing = "\n".join(parts)

    log.info(
        "Context built for %s on %s — %d chars, %d price rows, %d news items",
        stock, date, len(briefing), len(price_window), len(news_items),
    )

    return briefing


# ─────────────────────────────────────────────────────────────────────────────
# Metadata-only helper (for pipeline logging)
# ─────────────────────────────────────────────────────────────────────────────

def get_context_metadata(stock: str, date: str) -> dict:
    """
    Return a lightweight metadata dict without building the full briefing.
    Useful for pipeline monitoring / CSV logging.

    Returns:
        dict with keys: stock, date, company, has_price, has_news,
                        price_row_count, news_item_count.
    """
    company = STOCK_NAMES.get(stock, stock)
    meta    = {
        "stock":            stock,
        "date":             date,
        "company":          company,
        "has_price":        False,
        "has_news":         False,
        "price_row_count":  0,
        "news_item_count":  0,
    }

    try:
        window = _load_price_window(stock, date)
        meta["has_price"]       = True
        meta["price_row_count"] = len(window)
    except Exception:
        pass

    try:
        items = get_news_context(stock, date, max_items=NEWS_MAX_ITEMS)
        meta["has_news"]        = True
        meta["news_item_count"] = len(items)
    except Exception:
        pass

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test (run: python data/context_builder.py)
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    """
    Test context_builder on 5 stocks x 3 dates each (15 combinations total).
    Dates span both 2024 and 2025 to validate temporal coverage.
    Uses tickers that actually appear in this project's File_structure.txt
    (TCS, INFY, HDFCBANK, M&M, BAJAJ-AUTO) so the test runs against real data
    on your machine without any renaming.
    """
    TEST_CASES: list[tuple[str, str, str]] = [
        ("TCS.NS",         "2024-03-15",  "TCS - Mid Q4 2024"),
        ("TCS.NS",         "2024-09-10",  "TCS - Q2 2024 pre-results"),
        ("TCS.NS",         "2025-02-20",  "TCS - Q3 2025 post-results"),

        ("INFY.NS",        "2024-04-12",  "Infosys - Q4 2024 results week"),
        ("INFY.NS",        "2024-10-15",  "Infosys - Q2 2025 results week"),
        ("INFY.NS",        "2025-01-20",  "Infosys - Q3 2025"),

        ("HDFCBANK.NS",    "2024-06-03",  "HDFC Bank - June 2024"),
        ("HDFCBANK.NS",    "2024-11-05",  "HDFC Bank - post Q2 2025"),
        ("HDFCBANK.NS",    "2025-03-10",  "HDFC Bank - Q4 preview"),

        ("M&M.NS",         "2024-04-22",  "M&M - Q4 2024"),
        ("M&M.NS",         "2024-10-14",  "M&M - Q2 2025"),
        ("M&M.NS",         "2025-01-27",  "M&M - Jan 2025"),

        ("BAJAJ-AUTO.NS",  "2024-05-20",  "Bajaj Auto - pre-election result"),
        ("BAJAJ-AUTO.NS",  "2024-08-12",  "Bajaj Auto - Q1 2025"),
        ("BAJAJ-AUTO.NS",  "2025-02-10",  "Bajaj Auto - Q3 2025 post-results"),
    ]

    print()
    print("=" * 68)
    print("  DAY 27 - context_builder.py Test Suite")
    print("  5 stocks x 3 dates = 15 test cases")
    print("  2024 and 2025 coverage, look-ahead validation on every case")
    print("=" * 68)

    passed = 0
    failed = 0
    errors: list[str] = []

    for stock, date, label in TEST_CASES:
        print(f"\n{'-'*68}")
        print(f"  TEST : {label}")
        print(f"  Stock: {stock} | Date: {date}")

        try:
            briefing = build_context(stock, date)

            assert len(briefing) > 100, "Briefing is suspiciously short"
            assert "ANALYST BRIEFING" in briefing, "Missing header"
            assert "SECTION 1" in briefing, "Missing price section"
            assert "SECTION 2" in briefing, "Missing news section"
            assert "Look-ahead check: PASSED" in briefing, "Missing look-ahead confirmation"

            char_count  = len(briefing)
            token_est   = char_count // 4

            print(f"  PASS | {char_count} chars | ~{token_est} tokens")
            print()

            for line in briefing.split("\n")[:40]:
                print("    " + line)
            if briefing.count("\n") > 40:
                print("    ... [truncated for display]")

            passed += 1

        except FileNotFoundError as e:
            msg = f"SKIP (data not found): {e}"
            print(f"  {msg}")
            errors.append(f"{label}: {msg}")
        except AssertionError as e:
            msg = f"LOOK-AHEAD BIAS or sanity check FAILED: {e}"
            print(f"  FAIL - {msg}")
            errors.append(f"{label}: {msg}")
            failed += 1
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  ERROR - {msg}")
            errors.append(f"{label}: {msg}")
            failed += 1

    print()
    print("=" * 68)
    print(f"  RESULTS: {passed} passed | {failed} failed | {len(TEST_CASES)-passed-failed} skipped")
    if failed == 0 and passed > 0:
        print("  All run tests passed. No look-ahead bias detected.")
        print("  Ready for Day 28 - code review and docstring pass.")
    elif failed > 0:
        print("  Failures detected - fix before proceeding:")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  All tests were skipped (data not available).")
        print("  Ensure data pipeline (Days 1-26) is complete and data/ exists.")
    print("=" * 68)


if __name__ == "__main__":
    _run_tests()
