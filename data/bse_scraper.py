"""
data/bse_scraper.py — BSE Corporate Filings Scraper
Day 15 of the Indian Multi-Agent Trading System

Fetches corporate announcements from the BSE India API for all 20 stocks
across both 2024 and 2025. Saves raw JSON responses and parses them into
clean per-stock CSVs.

Usage:
    # Test on TCS for one month (Day 15 task):
    python bse_scraper.py --test

    # Full run on all 20 stocks, full date range (Day 16 task):
    python bse_scraper.py --full

    # Single stock:
    python bse_scraper.py --stock TCS.NS

Output:
    data/raw_news/bse/<TICKER>_<YEAR>_<MONTH>_raw.json   — raw API responses
    data/news/<TICKER>_bse.csv                            — parsed clean CSV
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# Import from config — all hardcoded values live there
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (
        BSE_CODES, STOCK_NAMES, STOCKS,
        START_DATE, END_DATE,
        BSE_REQUEST_DELAY, RAW_NEWS_DIR, NEWS_DIR,
    )
except ImportError:
    # Fallback defaults if running standalone (outside project root)
    print("⚠  Could not import config.py — using built-in defaults.")
    BSE_CODES = {
        "TCS.NS":        "532540",
        "INFY.NS":       "500209",
        "WIPRO.NS":      "507685",
        "HCLTECH.NS":    "532281",
        "HDFCBANK.NS":   "500180",
        "ICICIBANK.NS":  "532174",
        "KOTAKBANK.NS":  "500247",
        "SBIN.NS":       "500112",
        "RELIANCE.NS":   "500325",
        "ONGC.NS":       "500312",
        "POWERGRID.NS":  "532898",
        "NTPC.NS":       "532555",
        "HINDUNILVR.NS": "500696",
        "ITC.NS":        "500875",
        "NESTLEIND.NS":  "500790",
        "BRITANNIA.NS":  "500825",
        "MARUTI.NS":     "532500",
        "EICHERMOT.NS":  "505200",
        "BAJAJ-AUTO.NS": "532977",
        "M&M.NS":        "500520",
    }
    STOCK_NAMES = {k: k for k in BSE_CODES}
    STOCKS = list(BSE_CODES.keys())
    START_DATE = "2024-01-01"
    END_DATE   = "2025-12-31"
    BSE_REQUEST_DELAY = 2.0
    RAW_NEWS_DIR = "data/raw_news"
    NEWS_DIR     = "data/news"


# ── Constants ─────────────────────────────────────────────────────────────────

BSE_API_URL = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

# Headers that BSE expects — Referer is required or the API returns 403
BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":         "https://www.bseindia.com/",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://www.bseindia.com",
}

# BSE category codes — empty string means "all categories"
ALL_CATEGORIES = ""

RAW_BSE_DIR = os.path.join(RAW_NEWS_DIR, "bse")


# ── Directory setup ───────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [RAW_BSE_DIR, NEWS_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Date helpers ──────────────────────────────────────────────────────────────

def month_ranges(start: str, end: str):
    """
    Yield (from_date_str, to_date_str) tuples for each calendar month
    between start and end inclusive, in BSE date format (YYYYMMDD).
    """
    current = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    end_dt  = datetime.strptime(end,   "%Y-%m-%d")

    while current <= end_dt:
        month_start = current
        # Last day of month
        month_end   = (current + relativedelta(months=1)) - relativedelta(days=1)
        # Don't go past the configured end date
        if month_end > end_dt:
            month_end = end_dt

        yield month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")
        current += relativedelta(months=1)


# ── BSE API fetch ─────────────────────────────────────────────────────────────

def fetch_bse_month(bse_code: str, from_date: str, to_date: str) -> dict | None:
    """
    Fetch one month of BSE announcements for a given security code.

    Args:
        bse_code:  BSE security code, e.g. "532540"
        from_date: YYYYMMDD string
        to_date:   YYYYMMDD string

    Returns:
        Parsed JSON dict, or None on failure.
    """
    params = {
        "pageno":     "1",
        "strCat":     ALL_CATEGORIES,
        "strPrevDate": from_date,
        "strScrip":   bse_code,
        "strSearch":  "P",
        "strToDate":  to_date,
        "strType":    "C",
        "subcategory": "-1",
    }

    try:
        resp = requests.get(
            BSE_API_URL,
            params=params,
            headers=BSE_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    except requests.exceptions.HTTPError as e:
        print(f"    HTTP error {e.response.status_code} for code {bse_code} "
              f"({from_date}–{to_date})")
        return None
    except requests.exceptions.Timeout:
        print(f"    Timeout for code {bse_code} ({from_date}–{to_date}). Skipping.")
        return None
    except Exception as e:
        print(f"    Unexpected error: {e}")
        return None


# ── Raw save / load ───────────────────────────────────────────────────────────

def raw_path(ticker: str, from_date: str) -> str:
    year  = from_date[:4]
    month = from_date[4:6]
    safe  = ticker.replace(".", "_").replace("&", "AND")
    return os.path.join(RAW_BSE_DIR, f"{safe}_{year}_{month}_raw.json")


def save_raw(ticker: str, from_date: str, data: dict):
    path = raw_path(ticker, from_date)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_raw(ticker: str, from_date: str) -> dict | None:
    path = raw_path(ticker, from_date)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Parse JSON → rows ─────────────────────────────────────────────────────────

def parse_bse_response(data: dict, ticker: str) -> list[dict]:
    """
    Extract announcement rows from a BSE API JSON response.

    BSE API structure (as of 2024–2025):
        data["Table"]  — list of announcement dicts
        Each dict has keys like: NEWS_DT, HEADLINE, CATEGORYNAME, NEWSSUB, ...

    Returns:
        List of dicts with keys: date, headline, category, summary, ticker, source
    """
    rows = []
    if not data:
        return rows

    # The API returns data under "Table" key
    announcements = data.get("Table", [])
    if not announcements:
        # Some responses use a top-level list
        if isinstance(data, list):
            announcements = data
        else:
            return rows

    for ann in announcements:
        # Date — BSE format varies: "20240115T00:00:00" or "2024-01-15T00:00:00"
        raw_date = ann.get("NEWS_DT", "") or ann.get("DT_TM", "")
        try:
            # Strip time component and parse
            date_str = raw_date.split("T")[0].replace("-", "")
            parsed   = datetime.strptime(date_str, "%Y%m%d").date()
        except (ValueError, AttributeError):
            continue  # Skip rows with unparseable dates

        headline = (ann.get("HEADLINE", "") or "").strip()
        category = (ann.get("CATEGORYNAME", "") or ann.get("SUBCATNAME", "")).strip()
        summary  = (ann.get("NEWSSUB", "") or ann.get("ATTACHMENTNAME", "")).strip()

        if not headline:
            continue  # Skip empty rows

        rows.append({
            "date":     parsed.strftime("%Y-%m-%d"),
            "headline": headline,
            "category": category,
            "summary":  summary[:500],   # Truncate very long summaries
            "ticker":   ticker,
            "source":   "BSE",
        })

    return rows


# ── Per-stock scrape ──────────────────────────────────────────────────────────

def scrape_stock(ticker: str, start: str = START_DATE, end: str = END_DATE,
                 use_cache: bool = True) -> pd.DataFrame:
    """
    Scrape all BSE announcements for one stock across the full date range.

    Args:
        ticker:    NSE ticker with .NS suffix, e.g. "TCS.NS"
        start:     Start date string "YYYY-MM-DD"
        end:       End date string   "YYYY-MM-DD"
        use_cache: If True, skip months where raw JSON already exists on disk.

    Returns:
        DataFrame with columns: date, headline, category, summary, ticker, source
    """
    bse_code = BSE_CODES.get(ticker)
    if not bse_code:
        print(f"  ✗ No BSE code found for {ticker}. Skipping.")
        return pd.DataFrame()

    name = STOCK_NAMES.get(ticker, ticker)
    print(f"\n{'─'*60}")
    print(f"  Scraping {name} ({ticker} | BSE: {bse_code})")
    print(f"  Range: {start} → {end}")

    all_rows = []
    months   = list(month_ranges(start, end))

    for i, (from_d, to_d) in enumerate(months, 1):
        label = f"{from_d[:4]}-{from_d[4:6]}"
        print(f"    [{i:>2}/{len(months)}] {label} ... ", end="", flush=True)

        # Cache check
        cached = load_raw(ticker, from_d) if use_cache else None
        if cached is not None:
            print("(cached)", end="")
            data = cached
        else:
            data = fetch_bse_month(bse_code, from_d, to_d)
            if data is not None:
                save_raw(ticker, from_d, data)
            time.sleep(BSE_REQUEST_DELAY)   # Respect BSE rate limits

        rows = parse_bse_response(data, ticker) if data else []
        all_rows.extend(rows)
        print(f" → {len(rows)} announcements")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "headline"])
        df = df.sort_values("date").reset_index(drop=True)

    print(f"  ✓ Total: {len(df)} announcements for {name}")
    return df


# ── Save clean CSV ────────────────────────────────────────────────────────────

def save_bse_csv(ticker: str, df: pd.DataFrame):
    safe  = ticker.replace(".", "_").replace("&", "AND")
    path  = os.path.join(NEWS_DIR, f"{safe}_bse.csv")
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    return path


# ── Full run ──────────────────────────────────────────────────────────────────

def run_all(start: str = START_DATE, end: str = END_DATE):
    """Scrape all 20 stocks for the full date range."""
    ensure_dirs()
    print(f"\n{'='*60}")
    print(f"  BSE Filings Scraper — Full Run")
    print(f"  Stocks : {len(STOCKS)}")
    print(f"  Range  : {start} → {end}")
    print(f"  Delay  : {BSE_REQUEST_DELAY}s between requests")
    print(f"{'='*60}")

    summary = []
    for ticker in STOCKS:
        df = scrape_stock(ticker, start, end)
        if not df.empty:
            path = save_bse_csv(ticker, df)
            summary.append({"ticker": ticker, "count": len(df), "path": path})
        else:
            summary.append({"ticker": ticker, "count": 0, "path": "—"})

    # Print summary table
    print(f"\n{'='*60}")
    print("  Scraping Summary")
    print(f"{'─'*60}")
    print(f"  {'Ticker':<20} {'Count':>6}  Path")
    print(f"{'─'*60}")
    for row in summary:
        print(f"  {row['ticker']:<20} {row['count']:>6}  {row['path']}")

    total = sum(r["count"] for r in summary)
    print(f"{'─'*60}")
    print(f"  {'TOTAL':<20} {total:>6}")
    print(f"{'='*60}\n")


# ── Test run ────────────────────────────────────────────────────

def run_test():
    """
    Day 15 task: Test scraper on TCS for one month (January 2024).
    Print raw JSON structure so you understand the API response format.
    """
    ensure_dirs()
    print("\n────────────────────── Test: TCS, January 2024 ──────────────────────")
    ticker   = "TCS.NS"
    bse_code = BSE_CODES[ticker]

    print(f"\nFetching BSE announcements for {ticker} (code: {bse_code})...")
    data = fetch_bse_month(bse_code, "20240101", "20240131")

    if data is None:
        print("✗ API call failed. Check your internet connection and headers.")
        return

    # Save raw for inspection
    raw_file = os.path.join(RAW_BSE_DIR, "TCS_NS_2024_01_test_raw.json")
    os.makedirs(RAW_BSE_DIR, exist_ok=True)
    with open(raw_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Raw JSON saved to {raw_file}")

    # Show structure
    print("\nTop-level keys in response:", list(data.keys()) if isinstance(data, dict) else "list")
    records = data.get("Table", data) if isinstance(data, dict) else data
    print(f"Number of announcements:   {len(records)}")

    if records:
        print("\nFirst record keys:", list(records[0].keys()))
        print("\nFirst 3 announcements:")
        for r in records[:3]:
            print(f"  Date    : {r.get('NEWS_DT', r.get('DT_TM', 'N/A'))}")
            print(f"  Headline: {r.get('HEADLINE', 'N/A')[:80]}")
            print(f"  Category: {r.get('CATEGORYNAME', 'N/A')}")
            print()

    # Parse and show clean output
    df = parse_bse_response(data, ticker)
    print(f"Parsed {len(df)} clean rows.")
    if df:
        print("\nSample parsed output:")
        for row in df[:3]:
            print(f"  {row['date']} | {row['category']:<25} | {row['headline'][:60]}")

    print("\n✓ Test complete.")
    print("  If you see announcements above, the scraper is working correctly.")
    print("  If you got 0 announcements, the BSE API may have changed its response")
    print("  format — inspect the raw JSON file and update parse_bse_response().")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BSE Corporate Filings Scraper for Indian Multi-Agent Trading System"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test",  action="store_true",
                       help="Day 15: Test on TCS for January 2024 only")
    group.add_argument("--full",  action="store_true",
                       help="Day 16: Full run on all 20 stocks, 2024+2025")
    group.add_argument("--stock", type=str, metavar="TICKER",
                       help="Scrape a single stock, e.g. TCS.NS")

    parser.add_argument("--start", default=START_DATE,
                        help=f"Start date YYYY-MM-DD (default: {START_DATE})")
    parser.add_argument("--end",   default=END_DATE,
                        help=f"End date YYYY-MM-DD (default: {END_DATE})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-fetch even if raw JSON already cached on disk")

    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.full:
        run_all(args.start, args.end)
    elif args.stock:
        ensure_dirs()
        df = scrape_stock(args.stock, args.start, args.end,
                          use_cache=not args.no_cache)
        if not df.empty:
            save_bse_csv(args.stock, df)


if __name__ == "__main__":
    main()
