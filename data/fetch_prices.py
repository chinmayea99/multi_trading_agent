"""
data/fetch_prices.py — Download OHLCV price data for all 20 NSE stocks.

Coverage : Jan 1 2024 – Dec 31 2025  (driven by config.START_DATE / END_DATE)
Output   : data/prices/<TICKER>.csv   — one file per stock
Run      : python data/fetch_prices.py

Dependencies: yfinance, pandas  (pip install yfinance pandas)
"""

import os
import time
import pandas as pd
import yfinance as yf

# Import everything from config — no hardcoded values here
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    STOCKS,
    STOCK_NAMES,
    START_DATE,
    END_DATE,
    PRICES_DIR,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def fetch_single_stock(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Download OHLCV data for one ticker via yfinance.

    yfinance's end date is exclusive, so we pass the day after END_DATE
    to ensure Dec 31 2025 is included.

    Returns a cleaned DataFrame or None on failure.
    """
    try:
        # Add 1 day to end so the end date itself is included
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        df = yf.download(
            ticker,
            start=start,
            end=end_exclusive,
            auto_adjust=True,      # adjusts for splits and dividends
            progress=False,
        )

        if df.empty:
            print(f"  ⚠  No data returned for {ticker}.")
            return None

        # yfinance returns a MultiIndex when downloading single ticker with
        # auto_adjust; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardise column names
        df = df.rename(columns={
            "Open":   "Open",
            "High":   "High",
            "Low":    "Low",
            "Close":  "Close",
            "Volume": "Volume",
        })

        # Keep only OHLCV columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Ensure index is named 'Date' and is a proper DatetimeIndex
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index)

        # Remove any rows where Close is NaN (rare, but can happen on
        # partial trading days or data gaps)
        before = len(df)
        df = df.dropna(subset=["Close"])
        dropped = before - len(df)
        if dropped:
            print(f"  ℹ  {ticker}: dropped {dropped} row(s) with NaN Close.")

        return df

    except Exception as e:
        print(f"  ✗  Error fetching {ticker}: {e}")
        return None


def save_stock_csv(df: pd.DataFrame, ticker: str, output_dir: str) -> str:
    """Save DataFrame to CSV and return the file path."""
    # Use ticker without .NS suffix as filename for cleanliness
    filename = ticker.replace(".NS", "") + ".csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    return filepath


# ── Main ──────────────────────────────────────────────────────────────────────

def fetch_all_prices(
    stocks: list[str] = STOCKS,
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: str = PRICES_DIR,
    delay: float = 0.5,          # seconds between requests — be polite to Yahoo
) -> dict:
    """
    Download OHLCV data for all stocks and save to CSV files.

    Args:
        stocks     : List of NSE ticker strings (e.g. ["TCS.NS", "INFY.NS"])
        start      : Start date string "YYYY-MM-DD"
        end        : End date string  "YYYY-MM-DD"
        output_dir : Directory to save CSV files
        delay      : Seconds to wait between each download (rate-limit courtesy)

    Returns:
        dict mapping ticker → {"status": "ok"|"failed", "rows": int, "path": str}
    """
    ensure_dir(output_dir)

    print("=" * 60)
    print("  fetch_prices.py — NSE Stock OHLCV Downloader")
    print("=" * 60)
    print(f"  Date range : {start}  →  {end}")
    print(f"  Stocks     : {len(stocks)}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)

    results = {}
    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(stocks, start=1):
        name = STOCK_NAMES.get(ticker, ticker)
        print(f"\n[{i:02d}/{len(stocks)}] {ticker}  ({name})")

        df = fetch_single_stock(ticker, start, end)

        if df is None:
            results[ticker] = {"status": "failed", "rows": 0, "path": None}
            fail_count += 1
        else:
            path = save_stock_csv(df, ticker, output_dir)
            row_count = len(df)
            date_min = df.index.min().strftime("%Y-%m-%d")
            date_max = df.index.max().strftime("%Y-%m-%d")

            print(f"  ✓  {row_count} trading days  |  {date_min} → {date_max}  |  saved: {path}")
            results[ticker] = {"status": "ok", "rows": row_count, "path": path}
            success_count += 1

        # Polite delay between requests
        if i < len(stocks):
            time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  ✓  Successful : {success_count}")
    print(f"  ✗  Failed     : {fail_count}")

    if fail_count > 0:
        failed = [t for t, r in results.items() if r["status"] == "failed"]
        print(f"\n  Failed tickers: {', '.join(failed)}")
        print("  → Re-run the script or check ticker symbols on NSE website.")

    # Verify file count
    csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    print(f"\n  CSV files in {output_dir}/: {len(csv_files)}")
    if len(csv_files) == len(stocks):
        print("  ✓  All 20 files present.")
    else:
        missing = len(stocks) - len(csv_files)
        print(f"  ⚠  {missing} file(s) missing — check failed tickers above.")

    print("=" * 60)
    return results


def print_sample(ticker: str = "TCS.NS", n: int = 5) -> None:
    """Print the first n rows of a downloaded file as a quick sanity check."""
    filename = ticker.replace(".NS", "") + ".csv"
    filepath = os.path.join(PRICES_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    print(f"\n  Sample — {ticker} (first {n} rows):")
    print(df.head(n).to_string())
    print(f"\n  Shape: {df.shape}  |  Columns: {list(df.columns)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = fetch_all_prices()

    # Print a quick sample from TCS to eyeball the data
    print_sample("TCS.NS")
