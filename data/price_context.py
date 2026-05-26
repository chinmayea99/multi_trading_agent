"""
data/price_context.py — Price + indicator context builder for the Analyst Agent.

Task: write a function that takes a stock symbol and a date and returns
the last PRICE_LOOKBACK_DAYS trading days of OHLCV + indicator data as a clean,
naturally readable string suitable for pasting directly into an LLM prompt.

Works for any date in 2024 or 2025 — no hardcoded year.

Usage:
    from data.price_context import get_price_context
    context = get_price_context("TCS.NS", "2025-03-15")
    print(context)
"""

import os
import pandas as pd
from datetime import datetime

# Import from central config — never hardcode paths or constants here
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PRICES_DIR, STOCK_NAMES, PRICE_LOOKBACK_DAYS


# ── Main function ─────────────────────────────────────────────────────────────

def get_price_context(stock: str, date: str, lookback: int = PRICE_LOOKBACK_DAYS) -> str:
    """
    Return the last `lookback` trading days of price + indicator data for a stock,
    strictly before `date`, formatted as a natural-language briefing string.

    Args:
        stock:    NSE ticker with .NS suffix, e.g. "TCS.NS"
        date:     Query date as "YYYY-MM-DD" string. Data returned is BEFORE this date.
        lookback: Number of prior trading days to include (default: PRICE_LOOKBACK_DAYS from config).

    Returns:
        A multi-line string describing recent price action and indicators.

    Raises:
        FileNotFoundError: If the enriched price CSV for the stock does not exist.
        ValueError: If fewer than 2 rows of prior data are available (not enough context).
    """
    ticker = stock.replace(".NS", "")
    csv_path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Enriched price file not found: {csv_path}\n"
            "Run data/fetch_prices.py and data/add_indicators.py first."
        )

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    query_date = pd.to_datetime(date)

    # Keep only rows strictly before the query date (no look-ahead)
    prior = df[df["Date"] < query_date].copy()

    if len(prior) < 2:
        raise ValueError(
            f"Not enough prior data for {stock} before {date}. "
            f"Only {len(prior)} rows found."
        )

    # Take the most recent `lookback` rows
    window = prior.tail(lookback).reset_index(drop=True)

    # ── Build the briefing string ─────────────────────────────────────────────
    company_name = STOCK_NAMES.get(stock, stock)
    first_date   = window["Date"].iloc[0].strftime("%d %b %Y")
    last_date    = window["Date"].iloc[-1].strftime("%d %b %Y")

    lines = [
        f"PRICE CONTEXT — {company_name} ({stock})",
        f"Period: {first_date} to {last_date}  ({len(window)} trading days before {query_date.strftime('%d %b %Y')})",
        "",
        f"{'Date':<14}{'Open':>10}{'High':>10}{'Low':>10}{'Close':>10}{'Volume':>14}{'Chg1D%':>9}{'Chg5D%':>9}",
        "─" * 86,
    ]

    for _, row in window.iterrows():
        date_str  = row["Date"].strftime("%d %b %Y")
        open_p    = _fmt(row, "Open")
        high_p    = _fmt(row, "High")
        low_p     = _fmt(row, "Low")
        close_p   = _fmt(row, "Close")
        volume    = _fmt_vol(row, "Volume")
        chg1d     = _fmt_pct(row, "Change_1D_pct")
        chg5d     = _fmt_pct(row, "Change_5D_pct")
        lines.append(
            f"{date_str:<14}{open_p:>10}{high_p:>10}{low_p:>10}{close_p:>10}{volume:>14}{chg1d:>9}{chg5d:>9}"
        )

    # ── Latest indicator snapshot (last row in window) ────────────────────────
    latest = window.iloc[-1]
    lines += [
        "",
        "TECHNICAL INDICATORS (as of last trading day in window):",
        f"  SMA 20:            {_fmt(latest, 'SMA_20')}",
        f"  SMA 50:            {_fmt(latest, 'SMA_50')}",
        f"  RSI 14:            {_fmt(latest, 'RSI_14', decimals=1)}",
        f"  MACD:              {_fmt(latest, 'MACD', decimals=4)}",
        f"  MACD Signal:       {_fmt(latest, 'MACD_signal', decimals=4)}",
        f"  Bollinger Upper:   {_fmt(latest, 'BB_upper')}",
        f"  Bollinger Lower:   {_fmt(latest, 'BB_lower')}",
        "",
        _trend_summary(window, latest),
    ]

    return "\n".join(lines)


# ── Trend summary helper ──────────────────────────────────────────────────────

def _trend_summary(window: pd.DataFrame, latest: pd.Series) -> str:
    """Generate a one-paragraph plain-English trend summary from the window data."""
    close_col = "Close"
    first_close = window[close_col].iloc[0]
    last_close  = window[close_col].iloc[-1]
    period_chg  = ((last_close - first_close) / first_close) * 100

    direction = "gained" if period_chg >= 0 else "fallen"

    sma20 = latest.get("SMA_20", float("nan"))
    rsi   = latest.get("RSI_14", float("nan"))

    # RSI interpretation
    if pd.notna(rsi):
        if rsi > 70:
            rsi_note = f"RSI at {rsi:.1f} indicates overbought conditions."
        elif rsi < 30:
            rsi_note = f"RSI at {rsi:.1f} indicates oversold conditions."
        else:
            rsi_note = f"RSI at {rsi:.1f} is in neutral territory."
    else:
        rsi_note = "RSI data unavailable."

    # Price vs SMA20
    if pd.notna(sma20):
        vs_sma = "above" if last_close > sma20 else "below"
        sma_note = f"Price is {vs_sma} the 20-day SMA ({sma20:.2f})."
    else:
        sma_note = "SMA data unavailable."

    return (
        f"TREND SUMMARY: Over the past {len(window)} trading days the stock has "
        f"{direction} {abs(period_chg):.2f}%, closing at {last_close:.2f}. "
        f"{sma_note} {rsi_note}"
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt(row_or_series, col: str, decimals: int = 2) -> str:
    """Format a numeric field; return 'N/A' if missing."""
    val = row_or_series[col] if col in row_or_series.index else float("nan")
    if pd.isna(val):
        return "N/A"
    return f"{val:,.{decimals}f}"


def _fmt_pct(row_or_series, col: str) -> str:
    val = row_or_series[col] if col in row_or_series.index else float("nan")
    if pd.isna(val):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _fmt_vol(row_or_series, col: str) -> str:
    val = row_or_series[col] if col in row_or_series.index else float("nan")
    if pd.isna(val):
        return "N/A"
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"{val/1_000:.1f}K"
    return str(int(val))


# ── Manual test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Day 14 manual test: run 5 stock/date combinations and print outputs.
    Covers both 2024 and 2025 dates to validate the two-year pipeline.
    """
    TEST_CASES = [
        ("TCS.NS",       "2024-03-15"),
        ("RELIANCE.NS",  "2024-07-24"),   # day after Budget 2024
        ("HDFCBANK.NS",  "2025-02-10"),
        ("INFY.NS",      "2025-09-01"),
        ("MARUTI.NS",    "2025-04-05"),
    ]

    print("=" * 86)
    print("Day 14 — price_context.py smoke test")
    print("=" * 86)

    passed = 0
    for stock, date in TEST_CASES:
        print(f"\n▶  {stock}  |  query date: {date}")
        print("-" * 86)
        try:
            ctx = get_price_context(stock, date)
            print(ctx)
            passed += 1
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
        except ValueError as e:
            print(f"  [ERROR] {e}")
        except Exception as e:
            print(f"  [UNEXPECTED ERROR] {e}")

    print("\n" + "=" * 86)
    print(f"Results: {passed}/{len(TEST_CASES)} tests produced output.")
    if passed < len(TEST_CASES):
        print("Skipped tests mean enriched CSVs don't exist yet.")
    print("=" * 86)
