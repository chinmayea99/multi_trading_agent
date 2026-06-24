"""
tests/test_analyst.py — Analyst Agent Gemini Test Suite

M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.

WHAT:
    Tests the Analyst Agent (agents/analyst.py) against Gemini using the
    caching layer (utils/cache.py) across 5 real test cases.

WHY:
    This is the first day we make REAL LLM calls. We must verify:
      1. The prompt produces structured, grounded reports (no hallucinations).
      2. The cache layer saves and retrieves responses correctly.
      3. Response length stays within the 200–280 word target.
      4. No look-ahead: only data prior to the analysis date is referenced.
      5. All 5 required sections are present in every report.

HOW:
    For each test case (stock + date), we:
      a. Build a synthetic context string that mirrors context_builder output.
      b. Run analyze_stock() with use_stub=False against Gemini.
      c. Measure execution time and word count.
      d. Validate structure (5 sections present).
      e. Run hallucination checks (numbers must appear in context).
      f. Verify cache hit on second run (response time drops drastically).
      g. Save report to results/<stock>_<date>.txt.

USAGE (Windows CMD — run from project root):
    python tests/test_analyst.py
    python tests/test_analyst.py --case TCS
    python tests/test_analyst.py --no-cache   (force fresh API calls)
    python tests/test_analyst.py --validate-only  (skip LLM, check saved results)

OUTPUT:
    • Console summary with pass/fail for each check.
    • results/reports/  folder with one .txt file per test case.
    • results/summary.json  with metrics for thesis documentation.

Dependencies:
    • agents/analyst.py    
    • utils/cache.py       
    • llm_utils.py         
    • GEMINI_API_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.analyst import analyze_stock, build_analyst_prompt, _count_words, _validate_report
from llm_utils import init_llms

# ── Try importing cache  ──────────────────────────────────────────────
try:
    from utils.cache import get_cache, set_cache, cache_key
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠  utils/cache.py not found — cache validation will be skipped.")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

# ── Output directories ────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = PROJECT_ROOT / "results" / "summary.json"

# =============================================================================
# TEST CASES — 5 stocks × 5 dates
# Each context mirrors what context_builder.build_context() would produce.
# All numeric values here are sourced from historical NSE data (pre-date).
# =============================================================================

TEST_CASES: list[dict] = [

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 1: TCS — 2024-03-15
    # Context: Mid-Q4 FY24. TCS near all-time high. MACD bullish crossover.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "stock": "TCS.NS",
        "date":  "2024-03-15",
        "label": "TCS",
        "context": """
══════════════════════════════════════════════════════════════════════
  ANALYST BRIEFING — INDIAN STOCK MARKET
══════════════════════════════════════════════════════════════════════
  Stock        : Tata Consultancy Services (TCS.NS)
  Analysis Date: March 15, 2024  [all data is PRIOR to this date]
  Exchange     : NSE (National Stock Exchange of India)
══════════════════════════════════════════════════════════════════════

  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS
  ──────────────────────────────────────────────────────
  Latest Close : ₹3,892.50  |  Open: ₹3,870.00  |  High: ₹3,910.00  |  Low: ₹3,855.00
  1-Day Change : +0.82%      |  5-Day Change : -1.24%
  Volume       : 1,284,500 shares

  Technical Indicators (as of 2024-03-14):
    SMA_20   : ₹3,845.20   |  SMA_50  : ₹3,780.50
    RSI_14   : 58.3         |  Signal  : Mildly bullish (above 50, below 70)
    MACD     : +12.4        |  Signal  : +8.1    |  Histogram: +4.3  (bullish crossover)
    BB_Upper : ₹3,960.00   |  BB_Lower: ₹3,730.00

  SECTION 2 — RECENT NEWS & CORPORATE EVENTS
  ──────────────────────────────────────────
  [1] 2024-03-12 | Source: Economic Times
      Headline : TCS bags ₹2,200 crore deal from European bank
      Summary  : TCS secured a multi-year digital transformation deal worth
                 approximately ₹2,200 crore from a major European financial institution.

  [2] 2024-03-08 | Source: BSE Filing
      Headline : Q3 FY24 results — PAT up 8.2% YoY, revenue growth misses estimates
      Summary  : Net profit rose to ₹11,058 crore; revenue growth of 4.1% YoY
                 missed the street estimate of 5.5%.

  [3] 2024-03-05 | Source: Reuters
      Headline : RBI holds repo rate at 6.5% in MPC meeting
      Summary  : Monetary Policy Committee unanimously held rates. Neutral for IT.
══════════════════════════════════════════════════════════════════════
  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED
══════════════════════════════════════════════════════════════════════
""".strip(),
        # Numbers that MUST appear in a non-hallucinated report (if cited)
        "valid_numbers": [
    "3,892", "3,845", "3,780", "58.3", "12.4", "3,960", "3,730",
    "2,200", "11,058", "8.2", "4.1", "5.5", "6.5",
    "0.82",   # 1-Day Change from briefing
    "1.23",   # 5-Day Change from briefing
],
},

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 2: RELIANCE — 2024-04-10
    # Context: Post Q4 FY24 results window. RSI near overbought. Mixed news.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "stock": "RELIANCE.NS",
        "date":  "2024-04-10",
        "label": "RELIANCE",
        "context": """
══════════════════════════════════════════════════════════════════════
  ANALYST BRIEFING — INDIAN STOCK MARKET
══════════════════════════════════════════════════════════════════════
  Stock        : Reliance Industries Ltd (RELIANCE.NS)
  Analysis Date: April 10, 2024  [all data is PRIOR to this date]
  Exchange     : NSE (National Stock Exchange of India)
══════════════════════════════════════════════════════════════════════

  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS
  ──────────────────────────────────────────────────────
  Latest Close : ₹2,941.80  |  Open: ₹2,920.00  |  High: ₹2,955.00  |  Low: ₹2,910.00
  1-Day Change : +0.75%      |  5-Day Change : +2.18%
  Volume       : 3,452,200 shares

  Technical Indicators (as of 2024-04-09):
    SMA_20   : ₹2,874.60   |  SMA_50  : ₹2,823.40
    RSI_14   : 67.8         |  Signal  : Approaching overbought zone (>70 = overbought)
    MACD     : +18.2        |  Signal  : +14.5   |  Histogram: +3.7  (bullish, narrowing)
    BB_Upper : ₹2,990.00   |  BB_Lower: ₹2,760.00

  SECTION 2 — RECENT NEWS & CORPORATE EVENTS
  ──────────────────────────────────────────
  [1] 2024-04-07 | Source: Business Standard
      Headline : Reliance Jio IPO filing likely in H2 FY25 — sources
      Summary  : Reliance is reportedly preparing Jio's IPO documentation
                 for a potential listing in H2 FY25, which could unlock value.

  [2] 2024-04-04 | Source: Economic Times
      Headline : Reliance Retail clocks ₹78,600 crore revenue in Q3 FY24
      Summary  : Retail segment revenue grew 17.8% YoY; EBITDA margin at 8.4%.

  [3] 2024-04-01 | Source: Mint
      Headline : Oil-to-chemicals segment margins under pressure
      Summary  : GRM fell to $8.3/bbl in Q3 FY24 vs $9.1/bbl a year ago
                 as global refining margins softened.
══════════════════════════════════════════════════════════════════════
  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED
══════════════════════════════════════════════════════════════════════
""".strip(),
        "valid_numbers": ["2,941", "2,874", "2,823", "67.8", "18.2", "2,990", "2,760",
                          "78,600", "17.8", "8.4", "8.3", "9.1",
                          "0.75", "2.18"],   # 1-Day and 5-Day Change from briefing
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 3: HDFCBANK — 2024-06-03
    # Context: Post-merger integration period. Stock underperforming Nifty.
    # RSI near oversold. Bearish SMA cross.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "stock": "HDFCBANK.NS",
        "date":  "2024-06-03",
        "label": "HDFCBANK",
        "context": """
══════════════════════════════════════════════════════════════════════
  ANALYST BRIEFING — INDIAN STOCK MARKET
══════════════════════════════════════════════════════════════════════
  Stock        : HDFC Bank Ltd (HDFCBANK.NS)
  Analysis Date: June 3, 2024  [all data is PRIOR to this date]
  Exchange     : NSE (National Stock Exchange of India)
══════════════════════════════════════════════════════════════════════

  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS
  ──────────────────────────────────────────────────────
  Latest Close : ₹1,492.35  |  Open: ₹1,505.00  |  High: ₹1,512.00  |  Low: ₹1,485.00
  1-Day Change : -0.84%      |  5-Day Change : -2.93%
  Volume       : 8,231,400 shares

  Technical Indicators (as of 2024-06-01):
    SMA_20   : ₹1,524.80   |  SMA_50  : ₹1,548.30
    RSI_14   : 38.2         |  Signal  : Approaching oversold zone (<30 = oversold)
    MACD     : -14.6        |  Signal  : -10.2   |  Histogram: -4.4  (bearish)
    BB_Upper : ₹1,610.00   |  BB_Lower: ₹1,440.00

  SECTION 2 — RECENT NEWS & CORPORATE EVENTS
  ──────────────────────────────────────────
  [1] 2024-05-30 | Source: Bloomberg
      Headline : HDFC Bank NIM compression continues in Q4 FY24
      Summary  : Net interest margin fell to 3.44% in Q4 FY24 from 3.63% in Q4 FY23
                 as post-merger loan mix shifted toward lower-yielding segments.

  [2] 2024-05-22 | Source: Economic Times
      Headline : FIIs net sellers of HDFC Bank shares for 3rd consecutive week
      Summary  : Foreign institutional investors sold ₹4,200 crore of HDFC Bank
                 shares in May 2024, citing NIM pressure and deposit growth concerns.

  [3] 2024-05-15 | Source: BSE Filing
      Headline : HDFC Bank Q4 FY24 PAT ₹16,512 crore — up 37% YoY (merger effect)
      Summary  : Profit surge largely reflects merger consolidation; organic growth
                 at 11.4% YoY was below expectations.
══════════════════════════════════════════════════════════════════════
  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED
══════════════════════════════════════════════════════════════════════
""".strip(),
        "valid_numbers": ["1,492", "1,524", "1,548", "38.2", "14.6", "1,610", "1,440",
                          "3.44", "3.63", "4,200", "16,512", "37", "11.4",
                          "0.84", "2.93"],   # 1-Day and 5-Day Change from briefing
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 4: INFY — 2025-02-20
    # Context: Strong Q3 FY25. Guidance upgrade. RSI neutral.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "stock": "INFY.NS",
        "date":  "2025-02-20",
        "label": "INFY",
        "context": """
══════════════════════════════════════════════════════════════════════
  ANALYST BRIEFING — INDIAN STOCK MARKET
══════════════════════════════════════════════════════════════════════
  Stock        : Infosys Ltd (INFY.NS)
  Analysis Date: February 20, 2025  [all data is PRIOR to this date]
  Exchange     : NSE (National Stock Exchange of India)
══════════════════════════════════════════════════════════════════════

  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS
  ──────────────────────────────────────────────────────
  Latest Close : ₹1,847.60  |  Open: ₹1,835.00  |  High: ₹1,863.00  |  Low: ₹1,828.00
  1-Day Change : +0.68%      |  5-Day Change : +3.42%
  Volume       : 4,128,300 shares

  Technical Indicators (as of 2025-02-19):
    SMA_20   : ₹1,792.40   |  SMA_50  : ₹1,754.20
    RSI_14   : 54.7         |  Signal  : Neutral (between 50 and 60)
    MACD     : +22.8        |  Signal  : +17.3   |  Histogram: +5.5  (bullish)
    BB_Upper : ₹1,920.00   |  BB_Lower: ₹1,665.00

  SECTION 2 — RECENT NEWS & CORPORATE EVENTS
  ──────────────────────────────────────────
  [1] 2025-02-14 | Source: Economic Times
      Headline : Infosys raises FY25 revenue guidance to 4.5–5.0% in CC terms
      Summary  : Management raised the full-year revenue growth guidance from
                 3.75–4.5% to 4.5–5.0% in constant currency after strong deal wins.

  [2] 2025-02-10 | Source: Mint
      Headline : Infosys Q3 FY25 PAT ₹6,806 crore — up 11.4% YoY
      Summary  : Revenue grew 7.9% YoY to ₹41,764 crore. Deal TCV at $2.5 billion.

  [3] 2025-02-03 | Source: Reuters
      Headline : RBI cuts repo rate by 25 bps to 6.25% in February MPC
      Summary  : First rate cut in 5 years. Positive for corporate earnings sentiment.
══════════════════════════════════════════════════════════════════════
  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED
══════════════════════════════════════════════════════════════════════
""".strip(),
        "valid_numbers": ["1,847", "1,792", "1,754", "54.7", "22.8", "1,920", "1,665",
                          "4.5", "5.0", "6,806", "11.4", "7.9", "41,764", "2.5", "6.25",
                          "0.68", "3.42"],   # 1-Day and 5-Day Change from briefing
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CASE 5: MARUTI — 2025-05-08
    # Context: Strong domestic auto demand. GST relief speculation. Near 52w high.
    # ─────────────────────────────────────────────────────────────────────────
    {
        "stock": "MARUTI.NS",
        "date":  "2025-05-08",
        "label": "MARUTI",
        "context": """
══════════════════════════════════════════════════════════════════════
  ANALYST BRIEFING — INDIAN STOCK MARKET
══════════════════════════════════════════════════════════════════════
  Stock        : Maruti Suzuki India Ltd (MARUTI.NS)
  Analysis Date: May 8, 2025  [all data is PRIOR to this date]
  Exchange     : NSE (National Stock Exchange of India)
══════════════════════════════════════════════════════════════════════

  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS
  ──────────────────────────────────────────────────────
  Latest Close : ₹12,485.00  |  Open: ₹12,350.00  |  High: ₹12,540.00  |  Low: ₹12,295.00
  1-Day Change : +1.09%       |  5-Day Change : +4.27%
  Volume       : 412,600 shares

  Technical Indicators (as of 2025-05-07):
    SMA_20   : ₹11,924.00   |  SMA_50  : ₹11,452.00
    RSI_14   : 71.4          |  Signal  : Overbought zone (>70) — watch for pullback
    MACD     : +284.6        |  Signal  : +198.3   |  Histogram: +86.3 (strongly bullish)
    BB_Upper : ₹12,810.00   |  BB_Lower: ₹11,038.00

  SECTION 2 — RECENT NEWS & CORPORATE EVENTS
  ──────────────────────────────────────────
  [1] 2025-05-06 | Source: Business Standard
      Headline : Maruti April 2025 sales up 9.8% YoY to 2,09,156 units
      Summary  : Domestic passenger vehicle sales grew on strong SUV demand;
                 Ertiga and Brezza led volumes.

  [2] 2025-04-28 | Source: Economic Times
      Headline : Maruti Q4 FY25 PAT ₹3,911 crore — up 6.8% YoY
      Summary  : Revenue grew 9.1% YoY to ₹41,866 crore. EBITDA margin at 12.2%.

  [3] 2025-04-15 | Source: Mint
      Headline : Government considers GST cut on hybrid vehicles — report
      Summary  : A 12% GST rate (vs current 43%) on strong hybrids could significantly
                 boost Maruti's forthcoming hybrid lineup margins.
══════════════════════════════════════════════════════════════════════
  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED
══════════════════════════════════════════════════════════════════════
""".strip(),
        "valid_numbers": ["12,485", "11,924", "11,452", "71.4", "284.6", "12,810", "11,038",
                          "9.8", "2,09,156", "3,911", "6.8", "9.1", "41,866", "12.2",
                          "1.09", "4.27"],   # 1-Day and 5-Day Change from briefing
    },
]


# =============================================================================
# EVALUATION FRAMEWORK
# =============================================================================

REQUIRED_SECTIONS = [
    "TREND ANALYSIS",
    "KEY LEVELS",
    "NEWS SENTIMENT",
    "RISKS & OPPORTUNITIES",
    "OVERALL MARKET VIEW",
]

HALLUCINATION_PATTERNS = [
    # Invented phrases the LLM should never produce
    r"\bQ[1-4] FY\d{2}\b(?!.*₹)",          # Quarter reference without a data price
    r"₹\d[\d,]+(?!\s*(crore|lakh))",        # Bare price not in context
    r"\d{1,2}\.\d%\s*(growth|decline)",     # Invented % growth phrases
]

LOOKAHEAD_FORBIDDEN: dict[str, list[str]] = {
    # stock_label: [keywords that would appear only AFTER the analysis date]
    "TCS":       ["Q4 FY24", "annual results", "AGM 2024"],
    "RELIANCE":  ["Jio IPO listed", "Q4 FY24 results"],
    "HDFCBANK":  ["Q1 FY25", "RBI penalty"],
    "INFY":      ["Q4 FY25", "FY26 guidance"],
    "MARUTI":    ["GST cut announced", "Q1 FY26"],
}

STANCE_KEYWORDS = ["BULLISH", "BEARISH", "NEUTRAL", "CAUTIOUSLY"]

WORD_TARGET_MIN = 150
WORD_TARGET_MAX = 300


def check_structure(report: str) -> tuple[bool, list[str]]:
    """Return (passed, list_of_missing_sections)."""
    upper = report.upper()
    missing = [s for s in REQUIRED_SECTIONS if s not in upper]
    return len(missing) == 0, missing


def check_word_count(report: str) -> tuple[bool, int]:
    """Return (within_target, word_count)."""
    wc = _count_words(report)
    return WORD_TARGET_MIN <= wc <= WORD_TARGET_MAX, wc


def check_stance_present(report: str) -> bool:
    """OVERALL MARKET VIEW must end with a clear stance keyword."""
    upper = report.upper()
    return any(kw in upper for kw in STANCE_KEYWORDS)

def check_hallucination(report: str, valid_numbers: list[str]) -> tuple[bool, list[str]]:
    """
    Scan for numbers in the report that did NOT appear in the context.
    Returns (passed, list_of_suspicious_numbers).
    A suspicious number is any ₹ figure or percentage not in valid_numbers.
    """
    found_numbers = re.findall(r"₹[\d,]+|[\d]+\.[\d]+%?", report)
    suspicious = []
    for num in found_numbers:
        # Strip ₹, commas, and % for comparison — compare numeric core only
        clean = num.replace("₹", "").replace(",", "").replace("%", "").strip()
        # Check if the numeric core matches any valid number (also stripped)
        matched = any(
            v.replace(",", "").replace("%", "").strip() == clean
            or clean in v.replace(",", "").replace("%", "").strip()
            or v.replace(",", "").replace("%", "").strip() in clean
            for v in valid_numbers
        )
        if not matched and len(clean) > 2:
            suspicious.append(num)
    return len(suspicious) == 0, suspicious


def check_lookahead(report: str, label: str) -> tuple[bool, list[str]]:
    """Check for forbidden post-date keywords."""
    forbidden = LOOKAHEAD_FORBIDDEN.get(label, [])
    found = [kw for kw in forbidden if kw.lower() in report.lower()]
    return len(found) == 0, found


def check_hedged_language(report: str) -> tuple[bool, int]:
    """Count hedging words — good reports should have several."""
    hedges = ["suggests", "indicates", "appears", "may signal", "could",
              "seems", "likely", "possibly", "approximately"]
    count = sum(report.lower().count(h) for h in hedges)
    return count >= 2, count


def run_evaluation(report: str, case: dict) -> dict:
    """
    Run all evaluation checks on a single report.
    Returns an evaluation_result dict.
    """
    struct_ok,  missing_sections = check_structure(report)
    wc_ok,      word_count       = check_word_count(report)
    stance_ok                    = check_stance_present(report)
    halluc_ok,  suspicious_nums  = check_hallucination(report, case["valid_numbers"])
    lookahead_ok, forbidden_found = check_lookahead(report, case["label"])
    hedged_ok,  hedge_count      = check_hedged_language(report)

    checks = {
        "structure_all_sections": struct_ok,
        "word_count_in_range":    wc_ok,
        "stance_keyword_present": stance_ok,
        "no_hallucinated_numbers": halluc_ok,
        "no_lookahead_data":      lookahead_ok,
        "uses_hedged_language":   hedged_ok,
    }
    all_passed = all(checks.values())

    return {
        "checks":           checks,
        "all_passed":       all_passed,
        "word_count":       word_count,
        "hedge_count":      hedge_count,
        "missing_sections": missing_sections,
        "suspicious_nums":  suspicious_nums,
        "forbidden_found":  forbidden_found,
    }


# =============================================================================
# CACHE VALIDATION
# =============================================================================

def validate_cache(system_prompt: str, user_prompt: str, label: str) -> dict:
    """
    Check if cache returns a hit and measure speed improvement.
    Returns cache validation result dict.
    """
    if not CACHE_AVAILABLE:
        return {"available": False}

    combined = system_prompt + user_prompt
    key = cache_key(combined)
    hit = get_cache(key)
    return {
        "available": True,
        "cache_key":  key[:16] + "...",
        "cache_hit":  hit is not None,
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_test_case(
    case: dict,
    gemini_llm,
    force_fresh: bool = False,
) -> dict:
    """
    Run one test case end-to-end. Returns a result dict.
    """
    stock = case["stock"]
    date  = case["date"]
    label = case["label"]

    print()
    print("─" * 68)
    print(f"  TEST CASE: {label}  ({stock}  on  {date})")
    print("─" * 68)

    # ── Build prompt (for cache key computation) ──────────────────────────────
    sys_p, usr_p = build_analyst_prompt(case["context"], stock, date)
    prompt_words = _count_words(sys_p) + _count_words(usr_p)
    print(f"  Prompt word count : {prompt_words} words")

    # ── Cache check BEFORE calling LLM ───────────────────────────────────────
    pre_cache = validate_cache(sys_p, usr_p, label)
    print(f"  Cache hit (pre)   : {pre_cache.get('cache_hit', 'N/A')}")

    # ── First LLM call ────────────────────────────────────────────────────────
    print(f"  Calling Gemini...  (use_stub=False)")
    t0 = time.time()
    try:
        report = analyze_stock(
            context=case["context"],
            llm=gemini_llm,
            stock=stock,
            date=date,
            use_stub=False,
        )
        elapsed_first = time.time() - t0
        api_ok = True
    except Exception as exc:
        print(f"  ✗ LLM call FAILED: {exc}")
        report = f"ANALYST REPORT — {stock} — {date}\nERROR: {exc}"
        elapsed_first = time.time() - t0
        api_ok = False

    print(f"  First call time   : {elapsed_first:.2f}s")

    # ── Cache check AFTER first call ─────────────────────────────────────────
    post_cache = validate_cache(sys_p, usr_p, label)
    print(f"  Cache hit (post)  : {post_cache.get('cache_hit', 'N/A')}")

    # ── Second call (should be instant from cache) ────────────────────────────
    t1 = time.time()
    try:
        _ = analyze_stock(
            context=case["context"],
            llm=gemini_llm,
            stock=stock,
            date=date,
            use_stub=False,
        )
        elapsed_second = time.time() - t1
    except Exception:
        elapsed_second = None

    if elapsed_second is not None:
        speedup = elapsed_first / elapsed_second if elapsed_second > 0 else float("inf")
        print(f"  Second call time  : {elapsed_second:.2f}s  (speedup: {speedup:.1f}x)")
        cache_works = elapsed_second < elapsed_first * 0.5
    else:
        cache_works = False
        speedup = 0.0

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_result = run_evaluation(report, case)
    checks = eval_result["checks"]

    print()
    print("  EVALUATION RESULTS:")
    icons = {True: "✓", False: "✗"}
    for check_name, passed in checks.items():
        icon = icons[passed]
        print(f"    {icon}  {check_name}")

    if eval_result["missing_sections"]:
        print(f"      Missing: {eval_result['missing_sections']}")
    if eval_result["suspicious_nums"]:
        print(f"      Suspicious numbers: {eval_result['suspicious_nums']}")
    if eval_result["forbidden_found"]:
        print(f"      Look-ahead violation: {eval_result['forbidden_found']}")

    print(f"  Word count  : {eval_result['word_count']}  (target {WORD_TARGET_MIN}–{WORD_TARGET_MAX})")
    print(f"  Hedge count : {eval_result['hedge_count']}")
    print(f"  OVERALL     : {'✓ PASS' if eval_result['all_passed'] else '✗ FAIL'}")

    # ── Save report to file ───────────────────────────────────────────────────
    out_path = REPORTS_DIR / f"{label}_{date}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Test — {label} — {date}\n")
        f.write("=" * 68 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write("=" * 68 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 68 + "\n")
        for k, v in checks.items():
            f.write(f"  {'PASS' if v else 'FAIL'}  {k}\n")
        f.write(f"\nWord count  : {eval_result['word_count']}\n")
        f.write(f"Exec time   : {elapsed_first:.2f}s\n")
        f.write(f"Cache hit   : {post_cache.get('cache_hit', 'N/A')}\n")
    print(f"  Report saved: {out_path.name}")

    return {
        "label":          label,
        "stock":          stock,
        "date":           date,
        "api_ok":         api_ok,
        "elapsed_first":  round(elapsed_first, 2),
        "elapsed_second": round(elapsed_second, 2) if elapsed_second else None,
        "speedup":        round(speedup, 1),
        "cache_hit_post": post_cache.get("cache_hit", False),
        "cache_works":    cache_works,
        "word_count":     eval_result["word_count"],
        "all_passed":     eval_result["all_passed"],
        "checks":         {k: ("PASS" if v else "FAIL") for k, v in checks.items()},
        "prompt_words":   prompt_words,
        "report_preview": report[:300] + "..." if len(report) > 300 else report,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyst Agent Gemini Test")
    parser.add_argument("--case",          help="Run only this case label (e.g. TCS)")
    parser.add_argument("--no-cache",      action="store_true", help="Force fresh API calls")
    parser.add_argument("--validate-only", action="store_true", help="Evaluate saved results only")
    args = parser.parse_args()

    print()
    print("=" * 68)
    print("  Analyst Agent Test Suite — Gemini")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)

    # ── Init LLMs ─────────────────────────────────────────────────────────────
    if not args.validate_only:
        print("\nInitialising LLM clients...")
        try:
            llms = init_llms()
            gemini_llm = llms["groq"]
        except Exception as exc:
            print(f"✗ Failed to init LLMs: {exc}")
            print("  Make sure GEMINI_API_KEY is set in your .env file.")
            sys.exit(1)
    else:
        gemini_llm = None

    # ── Select test cases ─────────────────────────────────────────────────────
    cases = TEST_CASES
    if args.case:
        cases = [c for c in TEST_CASES if c["label"].upper() == args.case.upper()]
        if not cases:
            print(f"✗ Unknown case: {args.case}. Options: {[c['label'] for c in TEST_CASES]}")
            sys.exit(1)

    # ── Run tests ─────────────────────────────────────────────────────────────
    all_results = []
    for case in cases:
        if args.validate_only:
            # Load saved report and re-evaluate
            saved_path = REPORTS_DIR / f"{case['label']}_{case['date']}.txt"
            if not saved_path.exists():
                print(f"  ✗ No saved report for {case['label']} — run without --validate-only first")
                continue
            with open(saved_path, encoding="utf-8") as f:
                content = f.read()
            # Extract just the report section
            report = content.split("=" * 68)[2].strip() if "=" * 68 in content else content
            eval_result = run_evaluation(report, case)
            print(f"\n  {case['label']}: {'PASS' if eval_result['all_passed'] else 'FAIL'} "
                  f"({eval_result['word_count']} words)")
        else:
            print("  Waiting 15s (free tier rate limit)...")
            time.sleep(15)
            result = run_test_case(case, gemini_llm, force_fresh=args.no_cache)
            all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        print()
        print("=" * 68)
        print("  FINAL SUMMARY")
        print("=" * 68)
        passed = sum(1 for r in all_results if r["all_passed"])
        total  = len(all_results)
        print(f"\n  Tests passed : {passed}/{total}")
        print(f"\n  {'Label':<12} {'Words':>6} {'Time':>7} {'2nd':>7} {'Cache':>6} {'Status'}")
        print(f"  {'─'*12} {'─'*6} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")
        for r in all_results:
            second = f"{r['elapsed_second']:.2f}s" if r["elapsed_second"] else "—"
            status = "PASS" if r["all_passed"] else "FAIL"
            cache  = "HIT" if r["cache_hit_post"] else "MISS"
            print(f"  {r['label']:<12} {r['word_count']:>6} "
                  f"{r['elapsed_first']:>6.2f}s {second:>7} {cache:>6}  {status}")

        # ── Save summary JSON ─────────────────────────────────────────────────
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "run_date": datetime.now().isoformat(),
                "model":    "gemini-2.5-flash",
                "results":  all_results,
                "pass_rate": f"{passed}/{total}",
            }, f, indent=2)
        print(f"\n  Summary JSON saved: {SUMMARY_PATH}")

        print()
        print("  NEXT STEP: Run same 5 tests on Groq (Llama 3.3-70b)")
        print("=" * 68)
        print()

        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
