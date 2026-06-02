"""
data/rss_news.py — Google News RSS fetcher for the Indian Multi-Agent Trading System.

PURPOSE (read before using):
    RSS feeds return CURRENT headlines only — no historical archive.
    This module is ONLY for live/interactive testing, e.g. running the agent
    today to see how it reasons about a stock right now.

    DO NOT feed RSS output into the historical backtest loop.
    For backtesting (Jan 2024 – Dec 2025), use:
        - BSE filings  → data/bse_scraper.py   (Day 16)
        - NewsAPI      → data/newsapi_fetch.py  (Day 18)

USAGE:
    from data.rss_news import fetch_rss_news, fetch_all_rss_news, get_live_context

    # Single stock
    items = fetch_rss_news("TCS.NS")
    for item in items:
        print(item["title"], item["published"])

    # All 20 stocks
    all_news = fetch_all_rss_news(delay=1.5)

    # Ready-to-paste LLM context string
    context = get_live_context("RELIANCE.NS", max_items=3)
    print(context)

REQUIREMENTS:
    pip install feedparser
"""

import time
import re
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

import feedparser

# ── Project imports ───────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import STOCKS, STOCK_NAMES

# ── Constants ─────────────────────────────────────────────────────────────────
GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"
DEFAULT_MAX_ITEMS    = 10    # items to fetch per stock
DEFAULT_DELAY        = 1.5   # seconds between requests (be polite to Google)

# Fixed locale params for Indian English results
LOCALE_PARAMS = {
    "hl":   "en-IN",
    "gl":   "IN",
    "ceid": "IN:en",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_query(ticker: str) -> str:
    """
    Build a search query string for a given NSE ticker.

    Strategy: use the friendly company name + "stock NSE" for broader coverage
    than the raw ticker symbol alone.

    Examples:
        "TCS.NS"      → "Tata Consultancy Services stock NSE"
        "M&M.NS"      → "Mahindra & Mahindra stock NSE"
        "HDFCBANK.NS" → "HDFC Bank stock NSE"
    """
    company_name = STOCK_NAMES.get(ticker, ticker.replace(".NS", ""))
    return f"{company_name} stock NSE"


def _build_url(ticker: str) -> str:
    """Return the full RSS URL for a ticker."""
    query = _build_query(ticker)
    params = {"q": query, **LOCALE_PARAMS}
    return f"{GOOGLE_NEWS_RSS_BASE}?{urllib.parse.urlencode(params)}"


def _clean_html(text: str) -> str:
    """Strip basic HTML tags from summary text returned by feedparser."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_entry(entry: dict, ticker: str) -> dict:
    """
    Convert a single feedparser entry into a clean dict.

    Returns:
        {
            "ticker":    str,
            "title":     str,
            "published": ISO-8601 datetime string (UTC),
            "summary":   str   (HTML stripped),
            "link":      str,
            "source":    str   (publisher name if available),
        }
    """
    # Title
    title = entry.get("title", "").strip()

    # Published date — feedparser populates published_parsed as a time.struct_time in UTC
    published_parsed = entry.get("published_parsed")
    if published_parsed:
        dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
        published_iso = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        published_iso = entry.get("published", "unknown")

    # Summary / description
    summary_raw = entry.get("summary", entry.get("description", ""))
    summary = _clean_html(summary_raw)

    # Link
    link = entry.get("link", "")

    # Source publisher (Google News wraps this in the <source> tag)
    source = ""
    if hasattr(entry, "source") and isinstance(entry.get("source"), dict):
        source = entry["source"].get("title", "")

    return {
        "ticker":    ticker,
        "title":     title,
        "published": published_iso,
        "summary":   summary,
        "link":      link,
        "source":    source,
    }


# ── Core fetch function ───────────────────────────────────────────────────────

def fetch_rss_news(
    ticker: str,
    max_items: int = DEFAULT_MAX_ITEMS,
    verbose: bool = False,
) -> list[dict]:
    """
    Fetch current Google News RSS headlines for a single NSE stock.

    Args:
        ticker:    NSE ticker symbol, e.g. "TCS.NS"
        max_items: Maximum number of items to return (default 10).
        verbose:   Print progress messages if True.

    Returns:
        List of dicts, each with keys: ticker, title, published, summary, link, source.
        Returns an empty list on any fetch/parse error (logs the error).
    """
    url = _build_url(ticker)
    company = STOCK_NAMES.get(ticker, ticker)

    if verbose:
        print(f"  Fetching RSS for {company} ({ticker})...")
        print(f"  URL: {url}")

    try:
        feed = feedparser.parse(url)

        # feedparser doesn't raise on HTTP errors — check bozo flag and status
        if feed.bozo and not feed.entries:
            raise ValueError(f"feedparser bozo error: {feed.bozo_exception}")

        status = getattr(feed, "status", 200)
        if status not in (200, 301, 302):
            raise ConnectionError(f"HTTP {status} from Google News RSS")

        entries = feed.entries[:max_items]
        items = [_parse_entry(e, ticker) for e in entries]

        if verbose:
            print(f"  ✓ {len(items)} items fetched")

        return items

    except Exception as e:
        print(f"  ✗ fetch_rss_news failed for {ticker}: {e}")
        return []


# ── Batch fetch for all 20 stocks ─────────────────────────────────────────────

def fetch_all_rss_news(
    tickers: list[str] = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    delay: float = DEFAULT_DELAY,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """
    Fetch RSS news for all stocks (or a custom list) with rate-limiting.

    Args:
        tickers:   List of NSE tickers. Defaults to all 20 from config.
        max_items: Max items per stock.
        delay:     Seconds to sleep between requests.
        verbose:   Print per-stock progress.

    Returns:
        Dict mapping ticker → list of news items.
        e.g. {"TCS.NS": [...], "INFY.NS": [...], ...}
    """
    if tickers is None:
        tickers = STOCKS

    print(f"\n── Google News RSS Fetch ({'all' if tickers is STOCKS else len(tickers)} stocks) ──")
    print("⚠  RSS = CURRENT news only. Not suitable for historical backtesting.\n")

    results = {}
    total   = len(tickers)

    for i, ticker in enumerate(tickers, start=1):
        company = STOCK_NAMES.get(ticker, ticker)
        print(f"[{i:2d}/{total}] {company} ({ticker})", end="")

        items = fetch_rss_news(ticker, max_items=max_items, verbose=False)
        results[ticker] = items
        print(f" → {len(items)} items")

        if i < total:
            time.sleep(delay)

    fetched = sum(len(v) for v in results.values())
    empty   = [t for t, v in results.items() if not v]
    print(f"\n✓ Done. {fetched} total items across {total} stocks.")
    if empty:
        print(f"⚠  No items returned for: {', '.join(empty)}")

    return results


# ── Live context string for LLM prompts ──────────────────────────────────────

def get_live_context(
    ticker: str,
    max_items: int = 3,
    verbose: bool = False,
) -> str:
    """
    Fetch RSS headlines and format them as a ready-to-paste LLM context string.

    This is the function you call inside the agent pipeline during live testing.
    Replace this with get_news_context() from data/get_news_context.py when
    running historical backtests.

    Args:
        ticker:    NSE ticker, e.g. "RELIANCE.NS"
        max_items: How many headlines to include (default 3, matching NEWS_MAX_ITEMS).
        verbose:   Print debug info.

    Returns:
        Multi-line string like:
            Recent News (live RSS, as of 2025-05-29):
            1. [2025-05-28] Reliance Q4 results beat estimates (Economic Times)
               Reliance Industries reported a 12% jump in net profit...
            2. ...
    """
    items = fetch_rss_news(ticker, max_items=max_items, verbose=verbose)
    company = STOCK_NAMES.get(ticker, ticker)
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if not items:
        return (
            f"Recent News for {company} ({ticker}):\n"
            f"  [No live RSS items available — check BSE filings for historical context]\n"
        )

    lines = [f"Recent News for {company} ({ticker}) — live RSS as of {today}:"]
    lines.append("  ⚠ Source: Google News RSS (current headlines only, not historical)")
    lines.append("")

    for idx, item in enumerate(items, start=1):
        # Show just the date portion for readability
        pub_date = item["published"].split(" ")[0] if item["published"] != "unknown" else "?"
        source_tag = f" ({item['source']})" if item["source"] else ""
        lines.append(f"{idx}. [{pub_date}] {item['title']}{source_tag}")
        if item["summary"]:
            # Truncate long summaries for prompt economy
            summary = item["summary"][:200] + "..." if len(item["summary"]) > 200 else item["summary"]
            lines.append(f"   {summary}")
        lines.append("")

    return "\n".join(lines)


# ── Pretty-print helper ────────────────────────────────────────────────────────

def print_news(items: list[dict]) -> None:
    """Print a list of news items in a readable format."""
    if not items:
        print("  (no items)")
        return
    for i, item in enumerate(items, start=1):
        print(f"\n  [{i}] {item['title']}")
        print(f"       Published : {item['published']}")
        if item["source"]:
            print(f"       Source    : {item['source']}")
        if item["summary"]:
            print(f"       Summary   : {item['summary'][:150]}...")
        print(f"       Link      : {item['link']}")


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    """
    Quick sanity check — fetches RSS for 3 stocks and prints results.
    Run directly: python data/rss_news.py
    """
    test_tickers = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]

    print("=" * 60)
    print("  RSS News Fetcher — Smoke Test")
    print("  ⚠  Results are LIVE/CURRENT — not historical")
    print("=" * 60)

    for ticker in test_tickers:
        company = STOCK_NAMES.get(ticker, ticker)
        print(f"\n── {company} ({ticker}) ──────────────────────")
        items = fetch_rss_news(ticker, max_items=3, verbose=True)
        print_news(items)

    print("\n── Live Context String Example (RELIANCE.NS) ──────────")
    print(get_live_context("RELIANCE.NS", max_items=3))

    print("\n✓ Smoke test complete.")
    print("  If you saw 0 items for all stocks, Google may be rate-limiting.")
    print("  Wait 60 seconds and retry, or check the URL manually in a browser.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    smoke_test()
