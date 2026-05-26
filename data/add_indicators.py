"""
data/add_indicators.py — Day 13: Add technical indicators to all price CSVs.

Indicators added per stock:
  - SMA 20 / SMA 50       (Simple Moving Averages)
  - RSI 14                (Relative Strength Index)
  - MACD line, Signal, Histogram
  - Bollinger Bands upper / lower (20-period, 2σ)
  - Change 1D %           (1-day price change percent)
  - Change 5D %           (5-day price change percent)

Usage:
    python data/add_indicators.py                    # enrich all stocks
    python data/add_indicators.py --spot-check       # enrich + spot-check TCS & Reliance

Reads  : data/prices/<TICKER>_prices.csv
Writes : data/prices/<TICKER>_prices.csv  (overwrites in place)

Requires:  pip install pandas ta
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import ta  # pip install ta

# ── Import config from project root ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from config import STOCKS, PRICES_DIR, START_DATE, END_DATE
except ImportError:
    # Fallback defaults if running standalone
    STOCKS = [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS",
        "RELIANCE.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS",
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
        "MARUTI.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "M&M.NS",
    ]
    PRICES_DIR = "data/prices"
    START_DATE = "2024-01-01"
    END_DATE   = "2025-12-31"

# ── Spot-check dates (pick 3 across both years) ──────────────────────────────
SPOT_CHECK_CONFIG = {
    "TCS.NS": [
        "2024-03-15",   # mid-year 2024
        "2024-10-04",   # Q2 results season
        "2025-06-13",   # 2025 mid-year
    ],
    "RELIANCE.NS": [
        "2024-04-19",   # post Q4 FY24
        "2024-11-29",   # Q2 FY25 results period
        "2025-03-07",   # early 2025
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Core: add all indicators to a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append all required technical indicators.

    Args:
        df: DataFrame with columns [Date, Open, High, Low, Close, Volume].
            Date must be the index (datetime).

    Returns:
        DataFrame with new indicator columns appended.
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # ── Simple Moving Averages ────────────────────────────────────────────────
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"] = ta.trend.sma_indicator(close, window=50)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI_14"] = ta.momentum.rsi(close, window=14)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_obj       = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]     = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()

    # ── Bollinger Bands (20-period, 2σ) ──────────────────────────────────────
    bb_obj         = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_Upper"] = bb_obj.bollinger_hband()
    df["BB_Lower"] = bb_obj.bollinger_lband()

    # ── Price change % ────────────────────────────────────────────────────────
    df["Change_1D_pct"] = close.pct_change(periods=1) * 100
    df["Change_5D_pct"] = close.pct_change(periods=5) * 100

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Process a single stock file
# ─────────────────────────────────────────────────────────────────────────────

def process_file(ticker: str, prices_dir: str) -> tuple[bool, str]:
    """
    Read price CSV for `ticker`, add indicators, write back.

    Returns:
        (success: bool, message: str)
    """
    csv_path = Path(prices_dir) / f"{ticker.replace('.NS', '').replace('/', '_')}.csv"

    if not csv_path.exists():
        return False, f"File not found: {csv_path}"

    try:
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    except Exception as e:
        return False, f"Read error: {e}"

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"

    df = add_indicators(df)
    df = df.sort_index()

    try:
        df.to_csv(csv_path)
    except Exception as e:
        return False, f"Write error: {e}"

    rows = len(df)
    return True, f"{rows} rows enriched → {csv_path.name}"


# ─────────────────────────────────────────────────────────────────────────────
# Spot check: print indicator values for given dates
# ─────────────────────────────────────────────────────────────────────────────

SPOT_COLS = [
    "Close", "SMA_20", "SMA_50", "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "Change_1D_pct", "Change_5D_pct",
]


def spot_check(prices_dir: str) -> None:
    """
    Print indicator values for TCS and Reliance on 3 spot-check dates each.
    Use these numbers to cross-verify against TradingView.
    """
    print("\n" + "═" * 72)
    print("SPOT CHECK  — Verify these values against TradingView")
    print("═" * 72)

    for ticker, dates in SPOT_CHECK_CONFIG.items():
        csv_path = Path(prices_dir) / f"{ticker.replace('.NS', '').replace('/', '_')}.csv"
        if not csv_path.exists():
            print(f"\n  ⚠  {ticker}: file missing, skipping spot check.")
            continue

        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        print(f"\n{'─' * 72}")
        print(f"  {ticker}")
        print(f"{'─' * 72}")

        for date_str in dates:
            target = pd.Timestamp(date_str)

            # Find closest available trading day on or before target
            available = df[df.index <= target]
            if available.empty:
                print(f"  {date_str}: no data on or before this date")
                continue

            row   = available.iloc[-1]
            actual_date = available.index[-1].strftime("%Y-%m-%d")
            label = f"  {date_str}" + (f" (nearest: {actual_date})" if actual_date != date_str else "")
            print(label)

            for col in SPOT_COLS:
                if col in row.index and pd.notna(row[col]):
                    print(f"    {col:<18} {row[col]:>12.4f}")
                else:
                    print(f"    {col:<18}      N/A")

    print("\n" + "═" * 72)
    print("Cross-check SMA_20, RSI_14, BB_Upper / BB_Lower on TradingView.")
    print("Values should match within ±0.5% (minor differences OK due to")
    print("adjusted-close vs raw-close and slightly different period starts).")
    print("═" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add technical indicators to all stock price CSVs."
    )
    parser.add_argument(
        "--spot-check",
        action="store_true",
        help="After enriching, print spot-check table for TCS and Reliance.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Process only this ticker (e.g. TCS.NS). Default: all stocks.",
    )
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else STOCKS

    print(f"\nAdding indicators to {len(tickers)} stock(s)")
    print(f"Date range in config: {START_DATE} → {END_DATE}")
    print(f"Prices directory    : {PRICES_DIR}\n")

    ok_count  = 0
    err_count = 0
    errors    = []

    for ticker in tickers:
        success, msg = process_file(ticker, PRICES_DIR)
        icon = "✓" if success else "✗"
        print(f"  {icon}  {ticker:<18}  {msg}")
        if success:
            ok_count += 1
        else:
            err_count += 1
            errors.append((ticker, msg))

    print(f"\n── Summary ────────────────────────────────────────────")
    print(f"  Enriched : {ok_count}")
    print(f"  Errors   : {err_count}")

    if errors:
        print("\n  Failed tickers:")
        for t, m in errors:
            print(f"    {t}: {m}")
        print("\n  Tip: Run data/fetch_prices.py first if files are missing.")

    if args.spot_check:
        spot_check(PRICES_DIR)

    if err_count == 0:
        print("\n✓ All files enriched.")
    else:
        print("\n⚠  Fix errors above before proceeding.")


if __name__ == "__main__":
    main()
