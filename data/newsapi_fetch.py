"""
data/newsapi_fetch.py — Fetch historical news from NewsAPI for all 20 Indian stocks.

NewsAPI /v2/everything constraints:
  - 100 requests/day on free tier
  - Historical range depends on your plan (free tier: some history available via from/to)
  - Max 100 articles per page; use pagination to collect all results
  - Endpoint: /v2/everything

Usage:
    python data/newsapi_fetch.py

    To fetch a different month, change FETCH_MONTH at the top of this file.
    To fetch a full year, set FETCH_START and FETCH_END directly instead.

Output:
    data/raw_news/newsapi/<TICKER>_newsapi_<LABEL>_p<N>.json  ← raw per-page responses
    data/news/<TICKER>_newsapi_<LABEL>.csv                    ← parsed, deduplicated CSV

Budget tracking:
    Each paginated request costs 1 API call.
    For a stock with 250 articles, that is 3 requests (pages 1, 2, 3).
    Worst case: 20 stocks × ~5 pages = 100 requests — exactly the daily limit.
    The script tracks and prints a running request count so you never go blind.

Re-run note:
    This script is designed to be run TWICE across your project:
      Run 1 — now, with FETCH_MONTH = "2024-12" for Dec 2024 coverage.
      Run 2 — during Dec 2025 / Jan 2026, with FETCH_MONTH = "2025-12".
    Change FETCH_MONTH at the top and re-run. Raw files are cached per page,
    so re-runs skip already-fetched pages.
"""

import os
import json
import time
import calendar
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change FETCH_MONTH here when re-running for 2025
# ─────────────────────────────────────────────────────────────────────────────

FETCH_MONTH = "2026-05"   # !! Change to "2025-12" for the second run !!

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Derive date range from FETCH_MONTH
_year, _month  = map(int, FETCH_MONTH.split("-"))
FROM_DATE      = f"{FETCH_MONTH}-01"
TO_DATE        = f"{FETCH_MONTH}-{calendar.monthrange(_year, _month)[1]:02d}"
FETCH_LABEL    = FETCH_MONTH   # used in filenames

# Pagination
PAGE_SIZE      = 100           # NewsAPI max per page
MAX_PAGES      = 10            # safety cap — 10 pages × 100 = 1 000 articles per stock

# Delays
REQUEST_DELAY  = 2.0           # seconds between every API call (pagination included)
BACKOFF_DELAY  = 60.0          # seconds to wait after a 429 rate-limit response

# Output directories
RAW_DIR  = Path("data/raw_news/newsapi")
NEWS_DIR = Path("data/news")
RAW_DIR.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# Indian financial news domains — restricting to these cuts noise dramatically.
# These are major Indian business/financial publishers available on NewsAPI.
INDIAN_FINANCE_DOMAINS = ",".join([
    "economictimes.indiatimes.com",
    "livemint.com",
    "moneycontrol.com",
    "businessstandard.com",
    "financialexpress.com",
    "ndtv.com",
    "business-standard.com",
    "thehindu.com",
    "hindustantimes.com",
    "reuters.com",
    "bloomberg.com",
])

# Stock → search query
# Quoted company name ensures exact match; stock/NSE/India narrow to financial context.
STOCK_QUERIES = {
    "TCS.NS":        '"Tata Consultancy Services" OR "TCS" stock NSE India',
    "INFY.NS":       '"Infosys" stock NSE India',
    "WIPRO.NS":      '"Wipro" stock NSE India',
    "HCLTECH.NS":    '"HCL Technologies" stock NSE India',
    "HDFCBANK.NS":   '"HDFC Bank" stock NSE India',
    "ICICIBANK.NS":  '"ICICI Bank" stock NSE India',
    "KOTAKBANK.NS":  '"Kotak Mahindra Bank" stock NSE India',
    "SBIN.NS":       '"State Bank of India" OR "SBI" stock NSE',
    "RELIANCE.NS":   '"Reliance Industries" stock NSE India',
    "ONGC.NS":       '"ONGC" OR "Oil and Natural Gas" stock NSE India',
    "POWERGRID.NS":  '"Power Grid Corporation" stock NSE India',
    "NTPC.NS":       '"NTPC" stock NSE India',
    "HINDUNILVR.NS": '"Hindustan Unilever" OR "HUL" stock NSE India',
    "ITC.NS":        '"ITC Limited" stock NSE India',
    "NESTLEIND.NS":  '"Nestle India" stock NSE',
    "BRITANNIA.NS":  '"Britannia Industries" stock NSE India',
    "MARUTI.NS":     '"Maruti Suzuki" stock NSE India',
    "EICHERMOT.NS":  '"Eicher Motors" OR "Royal Enfield" stock NSE India',
    "BAJAJ-AUTO.NS": '"Bajaj Auto" stock NSE India',
    "M&M.NS":        '"Mahindra" stock NSE India',
}


# ─────────────────────────────────────────────────────────────────────────────
# Request counter (module-level so all functions share it)
# ─────────────────────────────────────────────────────────────────────────────
_requests_made = 0


def _make_request(params: dict) -> dict | None:
    """
    Single HTTP GET to NewsAPI with retry on 429.
    Increments the global request counter.
    Returns parsed JSON dict or None on unrecoverable error.
    """
    global _requests_made

    for attempt in range(1, 4):   # up to 3 attempts per page
        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
            _requests_made += 1

            if resp.status_code == 200:
                return resp.json()

            elif resp.status_code == 401:
                raise EnvironmentError(
                    "NewsAPI 401 Unauthorized — check NEWSAPI_KEY in your .env file."
                )

            elif resp.status_code == 429:
                print(
                    f"    ⚠  Rate limited (429). Sleeping {BACKOFF_DELAY}s "
                    f"then retrying (attempt {attempt}/3)..."
                )
                time.sleep(BACKOFF_DELAY)
                continue

            elif resp.status_code == 426:
                # Plan upgrade required for this date range
                try:
                    msg = resp.json().get("message", "")
                except Exception:
                    msg = resp.text[:200]
                print(
                    f"    ⚠  NewsAPI 426: your plan does not support this date range.\n"
                    f"       API message: {msg}\n"
                    f"       Upgrade at https://newsapi.org/pricing or use BSE filings instead."
                )
                return None

            else:
                print(f"    ✗  HTTP {resp.status_code}: {resp.text[:200]}")
                return None

        except EnvironmentError:
            raise
        except requests.exceptions.RequestException as e:
            print(f"    ✗  Request error (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(REQUEST_DELAY * 2)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core paginated fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_pages(ticker: str, query: str) -> list[dict]:
    """
    Fetch ALL pages of results for one stock.

    Saves each page as a separate raw JSON file so the cache works at page
    granularity — if the run is interrupted, completed pages are not re-fetched.

    Returns a flat list of all article dicts across all pages.
    """
    safe_ticker = ticker.replace(".", "_").replace("&", "AND")
    all_articles: list[dict] = []
    page = 1

    while page <= MAX_PAGES:
        raw_path = RAW_DIR / f"{safe_ticker}_{FETCH_LABEL}_p{page}.json"

        # ── Cache hit ─────────────────────────────────────────────────────────
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            page_articles = data.get("articles", [])
            total_results = data.get("totalResults", 0)
            print(
                f"    [cache] page {page} — {len(page_articles)} articles "
                f"(total reported: {total_results})"
            )
            all_articles.extend(page_articles)

            # If this was the last page, stop
            if len(all_articles) >= total_results or len(page_articles) < PAGE_SIZE:
                break

            page += 1
            continue

        # ── Live API call ─────────────────────────────────────────────────────
        params = {
            "q":        query,
            "from":     FROM_DATE,
            "to":       TO_DATE,
            "language": "en",
            "sortBy":   "publishedAt",
            "domains":  INDIAN_FINANCE_DOMAINS,
            "pageSize": PAGE_SIZE,
            "page":     page,
            "apiKey":   NEWSAPI_KEY,
        }

        print(
            f"    → page {page}  "
            f"[API req #{_requests_made + 1}]  "
            f"{FROM_DATE} → {TO_DATE}"
        )

        data = _make_request(params)

        if data is None:
            print(f"    ✗  Stopping pagination for {ticker} at page {page}.")
            break

        # Save raw page
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        page_articles  = data.get("articles", [])
        total_results  = data.get("totalResults", 0)
        status         = data.get("status", "")

        if status != "ok":
            print(f"    ✗  API returned status='{status}': {data.get('message', '')}")
            break

        print(
            f"    ✓  page {page}: {len(page_articles)} articles  "
            f"(total available: {total_results})"
        )
        all_articles.extend(page_articles)

        # Stop conditions
        fetched_so_far = (page - 1) * PAGE_SIZE + len(page_articles)
        if len(page_articles) < PAGE_SIZE or fetched_so_far >= total_results:
            break

        page += 1
        time.sleep(REQUEST_DELAY)

    return all_articles


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_articles(articles: list[dict]) -> pd.DataFrame:
    """
    Convert raw NewsAPI article list → clean DataFrame.

    Columns: date (datetime), headline, source, url, summary, content_snippet
    Deduplicates by URL. Drops articles with empty headlines.
    """
    rows = []
    for art in articles:
        pub = art.get("publishedAt", "")
        if not pub:
            continue
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            date_str = pub[:10]

        rows.append({
            "date":            date_str,
            "headline":        (art.get("title")       or "").strip(),
            "source":          art.get("source", {}).get("name", "NewsAPI"),
            "url":             (art.get("url")          or "").strip(),
            "summary":         (art.get("description")  or "").strip(),
            "content_snippet": (art.get("content")      or "").strip(),
        })

    if not rows:
        return pd.DataFrame(
            columns=["date", "headline", "source", "url", "summary", "content_snippet"]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # Remove empty headlines and deduplicate by URL
    df = df[df["headline"].str.len() > 0]
    df = df.drop_duplicates(subset=["url"])

    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(ticker: str, df: pd.DataFrame) -> Path:
    """Save to data/news/<TICKER>_newsapi_<LABEL>.csv"""
    safe_ticker = ticker.replace(".", "_").replace("&", "AND")
    out_path    = NEWS_DIR / f"{safe_ticker}_newsapi_{FETCH_LABEL}.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not NEWSAPI_KEY:
        raise EnvironmentError(
            "NEWSAPI_KEY not found in environment.\n"
            "  1. Register free at https://newsapi.org/register\n"
            "  2. Add  NEWSAPI_KEY=your_key_here  to your .env file\n"
            "  3. Re-run this script."
        )

    print("=" * 65)
    print(f"  NewsAPI Historical Fetch")
    print(f"  Period  : {FROM_DATE}  →  {TO_DATE}  ({FETCH_LABEL})")
    print(f"  Stocks  : {len(STOCK_QUERIES)}")
    print(f"  Domains : Indian financial publishers only")
    print(f"  Key     : {NEWSAPI_KEY[:6]}{'*' * max(0, len(NEWSAPI_KEY) - 6)}")
    print(f"  Daily budget: 100 requests  |  Max pages/stock: {MAX_PAGES}")
    print("=" * 65)

    summary = []

    for i, (ticker, query) in enumerate(STOCK_QUERIES.items(), start=1):
        print(f"\n[{i:02d}/{len(STOCK_QUERIES)}] {ticker}")
        print(f"  Query  : {query}")

        articles = fetch_all_pages(ticker, query)
        df       = parse_articles(articles)
        out_path = save_csv(ticker, df)

        date_from = str(df["date"].min().date()) if len(df) else "—"
        date_to   = str(df["date"].max().date()) if len(df) else "—"

        summary.append({
            "ticker":    ticker,
            "articles":  len(df),
            "date_from": date_from,
            "date_to":   date_to,
            "requests":  _requests_made,
        })
        print(f"  Saved  : {out_path}  ({len(df)} articles)")
        print(f"  Budget : {_requests_made} / 100 requests used so far today")

        # Stop early if approaching the daily limit
        if _requests_made >= 95:
            print(
                "\n  ⚠  Approaching 100-request daily limit. "
                "Stopping to preserve budget.\n"
                "  Re-run tomorrow to fetch remaining stocks "
                "(cached pages will be skipped)."
            )
            break

        if i < len(STOCK_QUERIES):
            time.sleep(REQUEST_DELAY)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  FETCH SUMMARY  —  {FETCH_LABEL}")
    print("=" * 65)
    total_articles = 0
    for row in summary:
        icon = "✓" if row["articles"] > 0 else "—"
        print(
            f"  {icon}  {row['ticker'].ljust(15)}"
            f"  {str(row['articles']).rjust(4)} articles"
            f"  [{row['date_from']} → {row['date_to']}]"
        )
        total_articles += row["articles"]

    print("-" * 65)
    print(f"  Total articles : {total_articles}")
    print(f"  Total requests : {_requests_made} / 100 daily limit")
    print(f"  Remaining      : {100 - _requests_made} requests today")
    print("=" * 65)

    print(
        "\n  📌 NEXT STEP: Run data/merge_news.py to combine these NewsAPI CSVs\n"
        "     with BSE filing CSVs into unified data/news/<TICKER>_news.csv files.\n"
        "\n  📌 REMINDER: Re-run with FETCH_MONTH = '2025-12' in Dec 2025 / Jan 2026\n"
        "     to collect December 2025 coverage.\n"
    )


if __name__ == "__main__":
    main()
