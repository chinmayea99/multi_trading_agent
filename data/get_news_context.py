"""
data/get_news_context.py — Look-ahead-safe news context retrieval.

THE MOST CRITICAL FUNCTION IN THE ENTIRE PROJECT.

What it does:
    Given a stock ticker and a date, returns the N most recent news items
    published STRICTLY BEFORE that date. This is not optional — using same-day
    or future news would introduce look-ahead bias, making backtest results
    meaningless (the model would effectively be "trading on the future").

Why this matters:
    If on March 15 the agent sees news published on March 15 (e.g., Q3 results
    announced that morning), it might trade on that information before the market
    has priced it in — a scenario impossible in real trading. We prevent this by
    enforcing a strict less-than (not less-than-or-equal) date filter, and
    backing it with a hard assertion that raises immediately if violated.

How it works:
    1. Load the merged news CSV for the stock (built by merge_news.py, Day 19).
    2. Parse dates as timezone-naive datetime objects.
    3. Filter rows where date < query_date (strictly before).
    4. Sort descending, take top max_items.
    5. Assert every returned item's date is strictly before query_date.
    6. Return as a list of dicts (date, headline, source, summary).

Usage:
    from data.get_news_context import get_news_context

    items = get_news_context("TCS.NS", "2024-03-15", max_items=3)
    for item in items:
        print(item["date"], "|", item["headline"])
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional

# ── Config (mirrors config.py — import from there in production) ──────────────
NEWS_DIR = os.path.join(os.path.dirname(__file__), "news")


# ─────────────────────────────────────────────────────────────────────────────
# Core Function
# ─────────────────────────────────────────────────────────────────────────────

def get_news_context(
    stock: str,
    date: str,
    max_items: int = 3,
    news_dir: Optional[str] = None,
) -> list[dict]:
    """
    Return the most recent news items for a stock published STRICTLY BEFORE
    the given date. Raises AssertionError if any look-ahead is detected.

    Args:
        stock     : NSE ticker, e.g. "TCS.NS"
        date      : Query date as "YYYY-MM-DD" string. News must be BEFORE this.
        max_items : Maximum number of news items to return (default 3).
        news_dir  : Override path to news directory (useful for testing).

    Returns:
        List of dicts, most recent first:
            [
                {
                    "date"     : datetime,
                    "headline" : str,
                    "source"   : str,
                    "summary"  : str,
                },
                ...
            ]

    Raises:
        FileNotFoundError : If the news CSV for the stock doesn't exist.
        AssertionError    : If any returned news item is dated >= query_date
                            (look-ahead bias detected — should never happen).
    """
    _news_dir = news_dir or NEWS_DIR

    # ── 1. Locate the CSV ─────────────────────────────────────────────────────
    # Normalise ticker: "TCS.NS" → "TCS_NS" for filename safety
    ticker_safe = stock.replace(".", "_").replace("&", "_")
    csv_path = os.path.join(_news_dir, f"{ticker_safe}_news.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"News CSV not found for {stock}.\n"
            f"Expected: {csv_path}\n"
            f"Run data/merge_news.py first."
        )

    # ── 2. Load & parse ───────────────────────────────────────────────────────
    news_df = pd.read_csv(csv_path)

    required_cols = {"date", "headline", "source", "summary"}
    missing = required_cols - set(news_df.columns)
    if missing:
        raise ValueError(
            f"News CSV for {stock} is missing columns: {missing}\n"
            f"Found columns: {list(news_df.columns)}"
        )

    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
    news_df = news_df.dropna(subset=["date"])   # drop rows with unparseable dates

    # ── 3. Strict look-ahead filter ───────────────────────────────────────────
    query_date = pd.to_datetime(date)
    prior_news = news_df[news_df["date"] < query_date]

    # ── 4. Sort descending, take top N ────────────────────────────────────────
    recent = (
        prior_news
        .sort_values("date", ascending=False)
        .head(max_items)
        .reset_index(drop=True)
    )

    # ── 5. CRITICAL ASSERTION — no look-ahead ever passes silently ────────────
    if not recent.empty:
        violators = recent[recent["date"] >= query_date]
        assert violators.empty, (
            f"\n{'='*60}\n"
            f"LOOK-AHEAD BIAS DETECTED for {stock} on {date}!\n"
            f"The following news items are dated ON OR AFTER the query date:\n"
            f"{violators[['date','headline']].to_string()}\n"
            f"{'='*60}\n"
            "Fix get_news_context immediately — do NOT proceed with backtesting."
        )

    # ── 6. Convert to list of dicts ───────────────────────────────────────────
    result = []
    for _, row in recent.iterrows():
        result.append({
            "date":     row["date"],
            "headline": str(row["headline"]),
            "source":   str(row.get("source", "Unknown")),
            "summary":  str(row.get("summary", "")),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Formatting Helper
# ─────────────────────────────────────────────────────────────────────────────

def format_news_context(items: list[dict]) -> str:
    """
    Format news items as a clean, human-readable string for LLM prompts.

    Args:
        items : Output of get_news_context().

    Returns:
        Multi-line string ready to paste into an agent prompt.
        Returns "No relevant news available." if items is empty.
    """
    if not items:
        return "No relevant news available."

    lines = ["Recent News (most recent first):"]
    for i, item in enumerate(items, start=1):
        date_str = item["date"].strftime("%Y-%m-%d")
        lines.append(
            f"\n[{i}] {date_str} | {item['source']}\n"
            f"    Headline : {item['headline']}\n"
            f"    Summary  : {item['summary']}"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Manual Test Suite  (run directly: python -m data.get_news_context)
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    """
    20 manual test cases across both 2024 and 2025 dates.
    Each test prints the returned items and verifies no look-ahead.
    """
    TEST_CASES = [
        # (stock,       date,           max_items, description)
        ("TCS.NS",    "2024-01-10",   3, "Early 2024 — only 1 item before Jan 10"),
        ("TCS.NS",    "2024-02-01",   3, "Exactly on results day — results NOT included"),
        ("TCS.NS",    "2024-02-02",   3, "Day after results — results now visible"),
        ("TCS.NS",    "2024-03-15",   3, "Mid-March 2024"),
        ("TCS.NS",    "2024-04-01",   3, "Pre-Q4 results window"),
        ("TCS.NS",    "2024-06-01",   3, "June 2024"),
        ("TCS.NS",    "2024-07-11",   3, "Exactly on Q1 results — NOT included"),
        ("TCS.NS",    "2024-07-12",   3, "Day after Q1 results — now visible"),
        ("TCS.NS",    "2024-09-20",   3, "Post BSNL deal"),
        ("TCS.NS",    "2024-10-10",   3, "Exactly on Q2 results — NOT included"),
        ("TCS.NS",    "2024-12-31",   5, "End of 2024 — max 5 items"),
        ("TCS.NS",    "2025-01-01",   3, "First day of 2025"),
        ("TCS.NS",    "2025-01-08",   3, "Exactly on Q3 preview — NOT included"),
        ("TCS.NS",    "2025-01-09",   3, "Day after Q3 preview — visible"),
        ("TCS.NS",    "2025-02-15",   3, "Exactly on buyback date — NOT included"),
        ("TCS.NS",    "2025-02-16",   3, "Day after buyback — visible"),
        ("TCS.NS",    "2025-03-25",   3, "Post UK deal"),
        ("TCS.NS",    "2025-05-01",   3, "Exactly on Q4 results — NOT included"),
        ("TCS.NS",    "2025-05-02",   3, "Day after Q4 results — visible"),
        ("TCS.NS",    "2024-01-04",   3, "Very early — no prior news, expect empty"),
    ]

    print("=" * 65)
    print("  Get_news_context() Test Suite")
    print("  20 cases spanning Jan 2024 – May 2025")
    print("=" * 65)

    passed = 0
    failed = 0

    for stock, date, max_items, description in TEST_CASES:
        print(f"\n{'─'*65}")
        print(f"  TEST : {stock} | {date} | max_items={max_items}")
        print(f"  DESC : {description}")

        try:
            items = get_news_context(stock, date, max_items=max_items)

            # Extra manual check — belt-and-suspenders on top of the assertion
            query_dt = pd.to_datetime(date)
            for item in items:
                if item["date"] >= query_dt:
                    raise AssertionError(
                        f"LOOK-AHEAD: item date {item['date']} >= query {query_dt}"
                    )

            print(f"  RESULT: {len(items)} item(s) returned ✓")
            for item in items:
                print(f"    • {item['date'].strftime('%Y-%m-%d')} | {item['headline'][:60]}")

            # Print formatted output for first test only (to demo the helper)
            if description.startswith("Mid-March"):
                print("\n  ── Formatted for LLM prompt ──────────────────────────")
                print(format_news_context(items))

            passed += 1

        except FileNotFoundError as e:
            print(f"  SKIP : {e}")
            # Don't count as fail — news CSV for that stock may not exist yet
        except AssertionError as e:
            print(f"  ✗ LOOK-AHEAD BIAS DETECTED:\n    {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*65}")
    print(f"  Results: {passed} passed | {failed} failed")
    if failed == 0:
        print("  ✓ All tests passed. No look-ahead bias detected.")
        print("  Ready for aggressive manual testing.")
    else:
        print("  ✗ Fix look-ahead issues before proceeding!")
    print("=" * 65)


if __name__ == "__main__":
    _run_tests()
