"""
tests/test_context_builder.py — Day 27 Test Script

Standalone test for data/context_builder.py.
Tests 5 stocks x 3 dates = 15 combinations across 2024 and 2025.

Run from project root (C:\\Users\\Chinmaye A\\multi_trading_agent):
    python tests\\test_context_builder.py

What this validates:
    1. build_context() returns a non-empty string for every test case.
    2. All required sections are present in the output.
    3. No look-ahead bias (assertion would raise inside build_context).
    4. Output token estimate is within LLM budget (<= 800 tokens).
    5. The query date itself does NOT appear in the price rows section.
    6. Metadata footer is present and reports PASSED.

Note on tickers:
    This test uses tickers that exist in your actual data/prices/ and
    data/news/ folders per File_structure.txt: TCS, INFY, HDFCBANK, M&M,
    BAJAJ-AUTO. If you have data for a different subset of the 20 stocks,
    edit TEST_CASES below to match what you've actually downloaded.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

# ── Make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.context_builder import build_context, get_context_metadata

# ── Test matrix (tickers chosen to match File_structure.txt) ──────────────────
TEST_CASES = [
    # stock,           date,          description
    ("TCS.NS",         "2024-03-15",  "TCS Q4 2024 mid-quarter"),
    ("TCS.NS",         "2024-09-10",  "TCS Q2 2025 pre-results"),
    ("TCS.NS",         "2025-02-20",  "TCS Q3 2025 post-results"),

    ("INFY.NS",        "2024-04-12",  "Infosys Q4 2024 results week"),
    ("INFY.NS",        "2024-10-15",  "Infosys Q2 2025 results week"),
    ("INFY.NS",        "2025-01-20",  "Infosys Q3 2025"),

    ("HDFCBANK.NS",    "2024-06-03",  "HDFC Bank June 2024"),
    ("HDFCBANK.NS",    "2024-11-05",  "HDFC Bank Q2 2025 post"),
    ("HDFCBANK.NS",    "2025-03-10",  "HDFC Bank Q4 preview"),

    ("M&M.NS",         "2024-04-22",  "M&M Q4 2024"),
    ("M&M.NS",         "2024-10-14",  "M&M Q2 2025"),
    ("M&M.NS",         "2025-01-27",  "M&M Jan 2025"),

    ("BAJAJ-AUTO.NS",  "2024-05-20",  "Bajaj Auto pre-election result"),
    ("BAJAJ-AUTO.NS",  "2024-08-12",  "Bajaj Auto Q1 2025"),
    ("BAJAJ-AUTO.NS",  "2025-02-10",  "Bajaj Auto Q3 2025 post"),
]

# Required strings that must appear in every briefing
REQUIRED_STRINGS = [
    "ANALYST BRIEFING",
    "SECTION 1",
    "SECTION 2",
    "Look-ahead check: PASSED",
    "all data is PRIOR to this date",
]

MAX_TOKEN_BUDGET = 800   # rough 4 chars/token


def validate_briefing(briefing: str, stock: str, date: str) -> list[str]:
    """Return list of validation error messages (empty = all good)."""
    errors = []

    if len(briefing) < 100:
        errors.append(f"Briefing too short ({len(briefing)} chars)")

    for req in REQUIRED_STRINGS:
        if req not in briefing:
            errors.append(f"Missing required string: '{req}'")

    token_est = len(briefing) // 4
    if token_est > MAX_TOKEN_BUDGET:
        errors.append(f"Token estimate {token_est} exceeds budget {MAX_TOKEN_BUDGET}")

    # The query date must NOT appear in price rows (it would mean same-day data)
    # We allow it in the header ("Analysis Date: March 15, 2024") but not in
    # the raw YYYY-MM-DD format inside the price table.
    price_section_start = briefing.find("SECTION 1")
    price_section_end   = briefing.find("SECTION 2")
    if price_section_start >= 0 and price_section_end >= 0:
        price_section = briefing[price_section_start:price_section_end]
        if date in price_section:
            errors.append(
                f"Query date {date} appears in PRICE SECTION — possible look-ahead!"
            )

    return errors


def run_tests() -> None:
    print()
    print("=" * 68)
    print("  DAY 27 - Context Builder Test Suite")
    print("  5 stocks x 3 dates = 15 combinations")
    print("=" * 68)

    results = {
        "passed":  [],
        "failed":  [],
        "skipped": [],
    }

    for i, (stock, date, desc) in enumerate(TEST_CASES, start=1):
        print(f"\n[{i:02d}/15] {desc}")
        print(f"         Stock={stock}  Date={date}")

        t0 = time.time()
        try:
            briefing  = build_context(stock, date)
            elapsed   = time.time() - t0
            errors    = validate_briefing(briefing, stock, date)
            meta      = get_context_metadata(stock, date)

            if errors:
                print(f"  FAIL ({elapsed:.2f}s)")
                for err in errors:
                    print(f"    - {err}")
                results["failed"].append((stock, date, desc, errors))
            else:
                token_est = len(briefing) // 4
                print(f"  PASS ({elapsed:.2f}s) | ~{token_est} tokens | "
                      f"price_rows={meta['price_row_count']} news={meta['news_item_count']}")
                results["passed"].append((stock, date, desc))

        except FileNotFoundError as exc:
            elapsed = time.time() - t0
            print(f"  SKIP ({elapsed:.2f}s) - data not found")
            print(f"    {exc}")
            results["skipped"].append((stock, date, desc))

        except AssertionError as exc:
            elapsed = time.time() - t0
            print(f"  LOOK-AHEAD BIAS DETECTED ({elapsed:.2f}s)")
            print(f"    {exc}")
            results["failed"].append((stock, date, desc, [str(exc)]))

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR ({elapsed:.2f}s) - {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results["failed"].append((stock, date, desc, [str(exc)]))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass  = len(results["passed"])
    n_fail  = len(results["failed"])
    n_skip  = len(results["skipped"])

    print()
    print("=" * 68)
    print(f"  RESULTS: {n_pass} passed | {n_fail} failed | {n_skip} skipped")
    print("=" * 68)

    if n_fail > 0:
        print("\n  FAILURES:")
        for stock, date, desc, errs in results["failed"]:
            print(f"  - {desc} ({stock} {date})")
            for e in errs:
                print(f"      * {e}")
        sys.exit(1)   # non-zero exit for CI
    elif n_pass == 0:
        print("\n  All tests were skipped - run data pipeline (Days 1-26) first.")
        sys.exit(2)
    else:
        print("\n  All executed tests passed. No look-ahead bias. Ready for Day 28.")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
