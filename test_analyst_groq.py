"""
test_analyst_groq.py — Analyst Agent Groq (Llama 3.3 70B) Evaluation

M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.

WHAT:
    Runs the same 5 test cases validated on Gemini through Groq (Llama 3.3 70B).
    Uses identical synthetic context strings as  so results are directly comparable.
    Saves outputs to results/groq_outputs/.

WHY:
    Cross-model evaluation is essential for thesis methodology. We need quantitative
    evidence on whether Llama 3.3 70B follows the structured prompt as reliably as
    Gemini, and whether speed gains justify any accuracy trade-offs.

HOW:
    1. Uses the same 5 hardcoded context strings as  test_analyst.py.
    2. Imports analyst from agents/analyst.py.
    3. Calls analyze_stock() with the Groq LLM client, use_stub=False.
    4. Scores on 5 dimensions: format, length, hallucination, speed, stance.
    5. Saves each report to results/groq_outputs/ and prints a summary table.

Usage (run from project root  C:\\...\\multi_trading_agent):
    python test_analyst_groq.py
    python test_analyst_groq.py --case TCS
    python test_analyst_groq.py --no-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# sys.path — project root (same pattern as test_analyst.py)
# File lives at:  multi_trading_agent/test_analyst_groq.py
# So parent == project root, parent.parent is NOT needed.
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "groq_eval.log", mode="a"),
    ],
)
log = logging.getLogger("groq_eval")

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "results" / "groq_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Project imports — analyst and llm_utils (no context_builder needed:
# we use the same hardcoded contexts as for direct comparability)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from agents.analyst import analyze_stock, _count_words, _validate_report
    log.info("Loaded analyst from agents/analyst.py")
except ImportError:
    try:
        from analyst import analyze_stock, _count_words, _validate_report
        log.info("Loaded analyst from analyst.py (project root)")
    except ImportError as e:
        log.error("Cannot import analyst: %s", e)
        log.error("Ensure you are running from the project root: cd multi_trading_agent")
        sys.exit(1)

try:
    from llm_utils import init_llms
    log.info("Loaded llm_utils")
except ImportError as e:
    log.error("Cannot import llm_utils: %s", e)
    sys.exit(1)

# ── Cache  ────────────────────────────────────────
try:
    from utils.cache import get_cache, set_cache, cache_key
    CACHE_AVAILABLE = True
    log.info("Cache layer loaded from utils/cache.py")
except ImportError:
    CACHE_AVAILABLE = False
    log.warning("utils/cache.py not found — calls will NOT be cached.")


# =============================================================================
# TEST CASES
# Identical hardcoded context strings from test_analyst.py.
# Using the same contexts ensures Groq vs Gemini comparison is apples-to-apples.
# =============================================================================

TEST_CASES: list[dict] = [

    # ── CASE 1: TCS — 2024-03-15 ─────────────────────────────────────────────
    {
        "stock": "TCS.NS",
        "date":  "2024-03-15",
        "label": "TCS",
        "context": (
            "══════════════════════════════════════════════════════════════════════\n"
            "  ANALYST BRIEFING — INDIAN STOCK MARKET\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Stock        : Tata Consultancy Services (TCS.NS)\n"
            "  Analysis Date: March 15, 2024  [all data is PRIOR to this date]\n"
            "  Exchange     : NSE (National Stock Exchange of India)\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "\n"
            "  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS\n"
            "  ──────────────────────────────────────────────────────\n"
            "  Latest Close : ₹3,892.50  |  Open: ₹3,870.00  |  High: ₹3,910.00  |  Low: ₹3,855.00\n"
            "  1-Day Change : +0.82%      |  5-Day Change : -1.24%\n"
            "  Volume       : 1,284,500 shares\n"
            "\n"
            "  Technical Indicators (as of 2024-03-14):\n"
            "    SMA_20   : ₹3,845.20   |  SMA_50  : ₹3,780.50\n"
            "    RSI_14   : 58.3         |  Signal  : Mildly bullish (above 50, below 70)\n"
            "    MACD     : +12.4        |  Signal  : +8.1    |  Histogram: +4.3  (bullish crossover)\n"
            "    BB_Upper : ₹3,960.00   |  BB_Lower: ₹3,730.00\n"
            "\n"
            "  SECTION 2 — RECENT NEWS & CORPORATE EVENTS\n"
            "  ──────────────────────────────────────────\n"
            "  [1] 2024-03-12 | Source: Economic Times\n"
            "      Headline : TCS bags ₹2,200 crore deal from European bank\n"
            "      Summary  : TCS secured a multi-year digital transformation deal worth\n"
            "                 approximately ₹2,200 crore from a major European financial institution.\n"
            "\n"
            "  [2] 2024-03-08 | Source: BSE Filing\n"
            "      Headline : Q3 FY24 results — PAT up 8.2% YoY, revenue growth misses estimates\n"
            "      Summary  : Net profit rose to ₹11,058 crore; revenue growth of 4.1% YoY\n"
            "                 missed the street estimate of 5.5%.\n"
            "\n"
            "  [3] 2024-03-05 | Source: Reuters\n"
            "      Headline : RBI holds repo rate at 6.5% in MPC meeting\n"
            "      Summary  : Monetary Policy Committee unanimously held rates. Neutral for IT.\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED\n"
            "══════════════════════════════════════════════════════════════════════"
        ),
        "valid_numbers": ["3,892", "3,845", "3,780", "58.3", "12.4", "3,960", "3,730",
                          "2,200", "11,058", "8.2", "4.1", "6.5"],
    },

    # ── CASE 2: RELIANCE — 2024-04-10 ────────────────────────────────────────
    {
        "stock": "RELIANCE.NS",
        "date":  "2024-04-10",
        "label": "RELIANCE",
        "context": (
            "══════════════════════════════════════════════════════════════════════\n"
            "  ANALYST BRIEFING — INDIAN STOCK MARKET\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Stock        : Reliance Industries Ltd (RELIANCE.NS)\n"
            "  Analysis Date: April 10, 2024  [all data is PRIOR to this date]\n"
            "  Exchange     : NSE (National Stock Exchange of India)\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "\n"
            "  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS\n"
            "  ──────────────────────────────────────────────────────\n"
            "  Latest Close : ₹2,941.80  |  Open: ₹2,920.00  |  High: ₹2,955.00  |  Low: ₹2,910.00\n"
            "  1-Day Change : +0.75%      |  5-Day Change : +2.18%\n"
            "  Volume       : 3,452,200 shares\n"
            "\n"
            "  Technical Indicators (as of 2024-04-09):\n"
            "    SMA_20   : ₹2,874.60   |  SMA_50  : ₹2,823.40\n"
            "    RSI_14   : 67.8         |  Signal  : Approaching overbought zone (>70 = overbought)\n"
            "    MACD     : +18.2        |  Signal  : +14.5   |  Histogram: +3.7  (bullish, narrowing)\n"
            "    BB_Upper : ₹2,990.00   |  BB_Lower: ₹2,760.00\n"
            "\n"
            "  SECTION 2 — RECENT NEWS & CORPORATE EVENTS\n"
            "  ──────────────────────────────────────────\n"
            "  [1] 2024-04-07 | Source: Business Standard\n"
            "      Headline : Reliance Jio IPO filing likely in H2 FY25 — sources\n"
            "      Summary  : Reliance is reportedly preparing Jio's IPO documentation\n"
            "                 for a potential listing in H2 FY25, which could unlock value.\n"
            "\n"
            "  [2] 2024-04-04 | Source: Economic Times\n"
            "      Headline : Reliance Retail clocks ₹78,600 crore revenue in Q3 FY24\n"
            "      Summary  : Retail segment revenue grew 17.8% YoY; EBITDA margin at 8.4%.\n"
            "\n"
            "  [3] 2024-04-01 | Source: Mint\n"
            "      Headline : Oil-to-chemicals segment margins under pressure\n"
            "      Summary  : GRM fell to $8.3/bbl in Q3 FY24 vs $9.1/bbl a year ago\n"
            "                 as global refining margins softened.\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED\n"
            "══════════════════════════════════════════════════════════════════════"
        ),
        "valid_numbers": ["2,941", "2,874", "2,823", "67.8", "18.2", "2,990", "2,760",
                          "78,600", "17.8", "8.4", "8.3", "9.1"],
    },

    # ── CASE 3: HDFCBANK — 2024-06-03 ────────────────────────────────────────
    {
        "stock": "HDFCBANK.NS",
        "date":  "2024-06-03",
        "label": "HDFCBANK",
        "context": (
            "══════════════════════════════════════════════════════════════════════\n"
            "  ANALYST BRIEFING — INDIAN STOCK MARKET\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Stock        : HDFC Bank Ltd (HDFCBANK.NS)\n"
            "  Analysis Date: June 3, 2024  [all data is PRIOR to this date]\n"
            "  Exchange     : NSE (National Stock Exchange of India)\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "\n"
            "  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS\n"
            "  ──────────────────────────────────────────────────────\n"
            "  Latest Close : ₹1,492.35  |  Open: ₹1,505.00  |  High: ₹1,512.00  |  Low: ₹1,485.00\n"
            "  1-Day Change : -0.84%      |  5-Day Change : -2.93%\n"
            "  Volume       : 8,231,400 shares\n"
            "\n"
            "  Technical Indicators (as of 2024-06-01):\n"
            "    SMA_20   : ₹1,524.80   |  SMA_50  : ₹1,548.30\n"
            "    RSI_14   : 38.2         |  Signal  : Approaching oversold zone (<30 = oversold)\n"
            "    MACD     : -14.6        |  Signal  : -10.2   |  Histogram: -4.4  (bearish)\n"
            "    BB_Upper : ₹1,610.00   |  BB_Lower: ₹1,440.00\n"
            "\n"
            "  SECTION 2 — RECENT NEWS & CORPORATE EVENTS\n"
            "  ──────────────────────────────────────────\n"
            "  [1] 2024-05-30 | Source: Bloomberg\n"
            "      Headline : HDFC Bank NIM compression continues in Q4 FY24\n"
            "      Summary  : Net interest margin fell to 3.44% in Q4 FY24 from 3.63% in Q4 FY23\n"
            "                 as post-merger loan mix shifted toward lower-yielding segments.\n"
            "\n"
            "  [2] 2024-05-22 | Source: Economic Times\n"
            "      Headline : FIIs net sellers of HDFC Bank shares for 3rd consecutive week\n"
            "      Summary  : Foreign institutional investors sold ₹4,200 crore of HDFC Bank\n"
            "                 shares in May 2024, citing NIM pressure and deposit growth concerns.\n"
            "\n"
            "  [3] 2024-05-15 | Source: BSE Filing\n"
            "      Headline : HDFC Bank Q4 FY24 PAT ₹16,512 crore — up 37% YoY (merger effect)\n"
            "      Summary  : Profit surge largely reflects merger consolidation; organic growth\n"
            "                 at 11.4% YoY was below expectations.\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED\n"
            "══════════════════════════════════════════════════════════════════════"
        ),
        "valid_numbers": ["1,492", "1,524", "1,548", "38.2", "14.6", "1,610", "1,440",
                          "3.44", "3.63", "4,200", "16,512", "37", "11.4"],
    },

    # ── CASE 4: INFY — 2025-02-20 ────────────────────────────────────────────
    {
        "stock": "INFY.NS",
        "date":  "2025-02-20",
        "label": "INFY",
        "context": (
            "══════════════════════════════════════════════════════════════════════\n"
            "  ANALYST BRIEFING — INDIAN STOCK MARKET\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Stock        : Infosys Ltd (INFY.NS)\n"
            "  Analysis Date: February 20, 2025  [all data is PRIOR to this date]\n"
            "  Exchange     : NSE (National Stock Exchange of India)\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "\n"
            "  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS\n"
            "  ──────────────────────────────────────────────────────\n"
            "  Latest Close : ₹1,847.60  |  Open: ₹1,835.00  |  High: ₹1,863.00  |  Low: ₹1,828.00\n"
            "  1-Day Change : +0.68%      |  5-Day Change : +3.42%\n"
            "  Volume       : 4,128,300 shares\n"
            "\n"
            "  Technical Indicators (as of 2025-02-19):\n"
            "    SMA_20   : ₹1,792.40   |  SMA_50  : ₹1,754.20\n"
            "    RSI_14   : 54.7         |  Signal  : Neutral (between 50 and 60)\n"
            "    MACD     : +22.8        |  Signal  : +17.3   |  Histogram: +5.5  (bullish)\n"
            "    BB_Upper : ₹1,920.00   |  BB_Lower: ₹1,665.00\n"
            "\n"
            "  SECTION 2 — RECENT NEWS & CORPORATE EVENTS\n"
            "  ──────────────────────────────────────────\n"
            "  [1] 2025-02-14 | Source: Economic Times\n"
            "      Headline : Infosys raises FY25 revenue guidance to 4.5-5.0% in CC terms\n"
            "      Summary  : Management raised the full-year revenue growth guidance from\n"
            "                 3.75-4.5% to 4.5-5.0% in constant currency after strong deal wins.\n"
            "\n"
            "  [2] 2025-02-10 | Source: Mint\n"
            "      Headline : Infosys Q3 FY25 PAT Rs6,806 crore — up 11.4% YoY\n"
            "      Summary  : Revenue grew 7.9% YoY to Rs41,764 crore. Deal TCV at $2.5 billion.\n"
            "\n"
            "  [3] 2025-02-03 | Source: Reuters\n"
            "      Headline : RBI cuts repo rate by 25 bps to 6.25% in February MPC\n"
            "      Summary  : First rate cut in 5 years. Positive for corporate earnings sentiment.\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED\n"
            "══════════════════════════════════════════════════════════════════════"
        ),
        "valid_numbers": ["1,847", "1,792", "1,754", "54.7", "22.8", "1,920", "1,665",
                          "6806", "11.4", "7.9", "41764", "6.25"],
    },

    # ── CASE 5: MARUTI — 2025-05-08 ──────────────────────────────────────────
    {
        "stock": "MARUTI.NS",
        "date":  "2025-05-08",
        "label": "MARUTI",
        "context": (
            "══════════════════════════════════════════════════════════════════════\n"
            "  ANALYST BRIEFING — INDIAN STOCK MARKET\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Stock        : Maruti Suzuki India Ltd (MARUTI.NS)\n"
            "  Analysis Date: May 8, 2025  [all data is PRIOR to this date]\n"
            "  Exchange     : NSE (National Stock Exchange of India)\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "\n"
            "  SECTION 1 — RECENT PRICE ACTION & TECHNICAL INDICATORS\n"
            "  ──────────────────────────────────────────────────────\n"
            "  Latest Close : Rs12,485.00  |  Open: Rs12,350.00  |  High: Rs12,540.00  |  Low: Rs12,295.00\n"
            "  1-Day Change : +1.09%        |  5-Day Change : +4.27%\n"
            "  Volume       : 412,600 shares\n"
            "\n"
            "  Technical Indicators (as of 2025-05-07):\n"
            "    SMA_20   : Rs11,924.00   |  SMA_50  : Rs11,452.00\n"
            "    RSI_14   : 71.4           |  Signal  : Overbought zone (>70) — watch for pullback\n"
            "    MACD     : +284.6         |  Signal  : +198.3   |  Histogram: +86.3 (strongly bullish)\n"
            "    BB_Upper : Rs12,810.00   |  BB_Lower: Rs11,038.00\n"
            "\n"
            "  SECTION 2 — RECENT NEWS & CORPORATE EVENTS\n"
            "  ──────────────────────────────────────────\n"
            "  [1] 2025-05-06 | Source: Business Standard\n"
            "      Headline : Maruti April 2025 sales up 9.8% YoY to 2,09,156 units\n"
            "      Summary  : Domestic passenger vehicle sales grew on strong SUV demand;\n"
            "                 Ertiga and Brezza led volumes.\n"
            "\n"
            "  [2] 2025-04-28 | Source: Economic Times\n"
            "      Headline : Maruti Q4 FY25 PAT Rs3,911 crore — up 6.8% YoY\n"
            "      Summary  : Revenue grew 9.1% YoY to Rs41,866 crore. EBITDA margin at 12.2%.\n"
            "\n"
            "  [3] 2025-04-15 | Source: Mint\n"
            "      Headline : Government considers GST cut on hybrid vehicles — report\n"
            "      Summary  : A 12% GST rate (vs current 43%) on strong hybrids could significantly\n"
            "                 boost Maruti's forthcoming hybrid lineup margins.\n"
            "══════════════════════════════════════════════════════════════════════\n"
            "  Price rows used : 10  |  News items used : 3  |  Look-ahead check: PASSED\n"
            "══════════════════════════════════════════════════════════════════════"
        ),
        "valid_numbers": ["12485", "11924", "11452", "71.4", "284.6", "12810", "11038",
                          "9.8", "3911", "6.8", "9.1", "41866", "12.2"],
    },
]


# =============================================================================
# EVALUATION HELPERS (mirrors test_analyst.py)
# =============================================================================

REQUIRED_SECTIONS = [
    "TREND ANALYSIS",
    "KEY LEVELS",
    "NEWS SENTIMENT",
    "RISKS & OPPORTUNITIES",
    "OVERALL MARKET VIEW",
]
STANCE_KEYWORDS = ["BULLISH", "BEARISH", "NEUTRAL", "CAUTIOUSLY"]
WORD_TARGET_MIN = 150
WORD_TARGET_MAX = 300


def check_structure(report: str) -> tuple:
    upper = report.upper()
    missing = [s for s in REQUIRED_SECTIONS if s not in upper]
    return len(missing) == 0, missing


def check_word_count(report: str) -> tuple:
    wc = len(re.findall(r"\S+", report))
    return WORD_TARGET_MIN <= wc <= WORD_TARGET_MAX, wc


def check_stance(report: str) -> bool:
    return any(kw in report.upper() for kw in STANCE_KEYWORDS)


def check_hallucinations(report: str, valid_numbers: list) -> tuple:
    flagged = []
    report_prices = re.findall(r"Rs\.?\s?[\d,]+(?:\.\d+)?|₹[\d,]+(?:\.\d+)?", report)
    for price in report_prices:
        digits = re.sub(r"[RsR₹\s,.]", "", price)
        if len(digits) >= 4:
            matched = any(digits[:4] in str(v).replace(",", "") for v in valid_numbers)
            if not matched:
                flagged.append(price)
    return len(flagged) == 0, flagged[:5]


def score_report(report: str, elapsed: float, valid_numbers: list) -> dict:
    struct_ok, missing = check_structure(report)
    wc_ok, wc = check_word_count(report)
    stance_ok = check_stance(report)
    halluc_ok, halluc_flags = check_hallucinations(report, valid_numbers)
    n_sections = len(REQUIRED_SECTIONS) - len(missing)

    format_score = round(n_sections / 5 * 5, 1)
    length_score = 5.0 if wc_ok else (3.0 if 100 <= wc <= 350 else 1.0)
    halluc_score = 5.0 if halluc_ok else max(0.0, 5.0 - len(halluc_flags))
    speed_score  = (5.0 if elapsed < 3  else
                    4.0 if elapsed < 6  else
                    3.0 if elapsed < 12 else
                    2.0 if elapsed < 20 else 1.0)
    stance_score = 5.0 if stance_ok else 0.0
    total = round((format_score + length_score + halluc_score + speed_score + stance_score) / 5, 2)

    return {
        "word_count":       wc,
        "n_sections":       n_sections,
        "missing_sections": missing,
        "stance_found":     stance_ok,
        "halluc_flags":     halluc_flags,
        "elapsed_s":        round(elapsed, 2),
        "scores": {
            "format_compliance": format_score,
            "response_length":   length_score,
            "hallucination":     halluc_score,
            "speed":             speed_score,
            "stance_clarity":    stance_score,
            "TOTAL_OUT_OF_5":    total,
        },
    }


# =============================================================================
# MAIN EVALUATION RUNNER
# =============================================================================

def run_groq_evaluation(filter_case: str = "", bypass_cache: bool = False) -> None:
    log.info("=" * 65)
    log.info("Analyst Agent  |  Groq Llama 3.3 70B  |  Evaluation")
    log.info("=" * 65)

    # Init LLMs
    try:
        llms = init_llms()
        groq_client = llms["groq"]
        log.info("Groq client ready")
    except Exception as e:
        log.error("Failed to init LLMs: %s", e)
        sys.exit(1)

    # Filter
    cases = TEST_CASES
    if filter_case:
        cases = [c for c in TEST_CASES if c["label"].upper() == filter_case.upper()]
        if not cases:
            log.error(
                "No test case matching --case '%s'. Options: TCS RELIANCE HDFCBANK INFY MARUTI",
                filter_case,
            )
            sys.exit(1)

    results = []
    sep = "─" * 65

    for case in cases:
        stock         = case["stock"]
        date          = case["date"]
        label         = case["label"]
        context       = case["context"]
        valid_numbers = case["valid_numbers"]

        print(f"\n{sep}")
        log.info("%s  |  %s", stock, date)

        # ── LLM call ─────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            report = analyze_stock(
                context=context,
                llm=groq_client,
                stock=stock,
                date=date,
                use_stub=False,
                max_retries=3,
            )
            elapsed = time.time() - t0
            log.info("Response received in %.2fs", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            log.error("analyze_stock FAILED: %s", e)
            report = f"ERROR: {e}"

        # ── Score ─────────────────────────────────────────────────────────────
        ev = score_report(report, elapsed, valid_numbers)
        sc = ev["scores"]

        log.info(
            "Words:%d | Secs:%.1f | Sections:%d/5 | Halluc:%d | Total:%.1f/5",
            ev["word_count"], elapsed, ev["n_sections"],
            len(ev["halluc_flags"]), sc["TOTAL_OUT_OF_5"],
        )
        if ev["missing_sections"]:
            log.warning("Missing sections: %s", ev["missing_sections"])
        if ev["halluc_flags"]:
            log.warning("Hallucination flags: %s", ev["halluc_flags"])

        # ── Print full report to console ──────────────────────────────────────
        print(f"\n{'═'*65}")
        print(f"  GROQ OUTPUT — {stock} | {date}")
        print("═" * 65)
        print(report)
        print("═" * 65)

        # Checklist
        struct_ok, missing = check_structure(report)
        wc_ok, wc         = check_word_count(report)
        stance_ok         = check_stance(report)
        halluc_ok, _      = check_hallucinations(report, valid_numbers)
        print(f"\n  Checklist:")
        print(f"    {'✓' if struct_ok else '✗'} All 5 sections present"
              + (f" [missing: {missing}]" if not struct_ok else ""))
        print(f"    {'✓' if wc_ok    else '⚠'} Word count: {wc} (target 150–300)")
        print(f"    {'✓' if stance_ok else '✗'} OVERALL MARKET VIEW stance found")
        print(f"    {'✓' if halluc_ok else '⚠'} No hallucination flags")
        print(f"    Score: {sc['TOTAL_OUT_OF_5']:.2f}/5")

        # ── Save report ───────────────────────────────────────────────────────
        safe_stock = stock.replace(".", "_").replace("&", "_")
        out_file   = OUTPUT_DIR / f"{safe_stock}_{date}_groq.txt"
        out_file.write_text(report, encoding="utf-8")
        log.info("Saved: %s", out_file.name)

        # ── Cache verification ────────────────────────────────────────────────
        if CACHE_AVAILABLE and not bypass_cache:
            t1 = time.time()
            try:
                analyze_stock(
                    context=context, llm=groq_client,
                    stock=stock, date=date, use_stub=False, max_retries=1,
                )
                cache_elapsed = time.time() - t1
                if cache_elapsed < elapsed * 0.3:
                    log.info("Cache hit confirmed (%.2fs vs %.2fs)", cache_elapsed, elapsed)
                else:
                    log.warning("Possible cache miss (%.2fs)", cache_elapsed)
            except Exception:
                pass

        results.append({"stock": stock, "date": date, "label": label,
                         "model": "groq/llama-3.3-70b-versatile",
                         "report": report, **ev})

    # ── Save JSON summary ──────────────────────────────────────────────────────
    summary = [{k: v for k, v in r.items() if k != "report"} for r in results]
    summary_path = OUTPUT_DIR / "groq_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Print final table ──────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("GROQ (Llama 3.3 70B) — EVALUATION SUMMARY")
    print("═" * 72)
    print(f"  {'Stock':<14} {'Date':<12} {'Words':<7} {'Secs':<7} {'Spd/5':<7} {'Hall':<6} {'TOTAL'}")
    print("  " + "─" * 60)
    for r in results:
        sc = r["scores"]
        print(
            f"  {r['stock']:<14} {r['date']:<12} {r['word_count']:<7} "
            f"{r['elapsed_s']:<7.1f} {sc['speed']:<7.1f} "
            f"{len(r['halluc_flags']):<6} {sc['TOTAL_OUT_OF_5']:.2f}/5"
        )
    if results:
        avg       = sum(r["scores"]["TOTAL_OUT_OF_5"] for r in results) / len(results)
        avg_words = sum(r["word_count"] for r in results) / len(results)
        avg_time  = sum(r["elapsed_s"]  for r in results) / len(results)
        print("  " + "─" * 60)
        print(f"  {'AVERAGE':<28} {avg_words:<7.0f} {avg_time:<7.1f} {'':>13} {avg:.2f}/5")
    print("═" * 72)
    print(f"\n  Reports : {OUTPUT_DIR}")
    print(f"  Summary : {summary_path}")
    print("═" * 72)


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groq Analyst Agent Evaluation")
    parser.add_argument(
        "--case", default="",
        help="Single case only. Options: TCS RELIANCE HDFCBANK INFY MARUTI",
    )
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Skip cache verification step.",
    )
    args = parser.parse_args()
    run_groq_evaluation(filter_case=args.case, bypass_cache=args.no_cache)
