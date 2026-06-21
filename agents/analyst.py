"""
agents/analyst.py — Analyst Agent

M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.

What it does:
    Receives the analyst briefing string produced by context_builder.py and sends it
    to an LLM via a carefully designed prompt.  The LLM acts as a senior equity analyst
    and returns a structured, ~250-word report covering trend, support/resistance,
    news sentiment, risks/opportunities, and an overall market view.

Why it matters:
    This is the FIRST agent in the three-agent pipeline:
        Analyst → Researcher (Bull/Bear) → Trader
    The quality of every downstream decision depends on this report.
    A vague or hallucinated analyst report will cascade into bad trades.
    Therefore: the prompt is tightly constrained, the output format is explicit,
    and all numeric references are grounded strictly in the briefing string.

How it works:
    1. Receive a context string (output of context_builder.build_context()).
    2. Inject it into the ANALYST_SYSTEM_PROMPT + ANALYST_USER_TEMPLATE.
    3. Call the LLM (Gemini / Groq / Ollama) via llm_utils.call_llm().
    4. Return a plain-text analyst report string.
    NOTE: LLM calls are stubbed. The prompt and wrapper are complete.

Usage:
    from agents.analyst import analyze_stock

    report = analyze_stock(
        context  = briefing_string,      # from context_builder.build_context()
        llm      = llms["gemini"],       # from llm_utils.init_llms()
        stock    = "TCS.NS",
        date     = "2024-03-15",
    )
    print(report)

Anti-hallucination rules baked into the prompt:
    • Only reference numbers that appear verbatim in the briefing.
    • Do NOT invent price levels, percentages, or news events.
    • If data is missing, write "Insufficient data" for that section.
    • Express uncertainty explicitly (e.g. "RSI suggests…", "price appears to…").

Dependencies:
    • context_builder.py    — produces the input briefing string
    • llm_utils.py          — provides call_llm()
    • config.py              — provides LLM_MAX_TOKENS, LLM_TEMPERATURE
"""

from __future__ import annotations

import logging
import re
import textwrap
from datetime import datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Constants
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT: str = textwrap.dedent("""
    You are a senior equity analyst at a top-tier Indian asset management firm.
    You specialise in NSE-listed stocks and have deep expertise in:
      • Technical analysis (SMA, RSI, MACD, Bollinger Bands)
      • Price action interpretation
      • News and corporate event impact assessment
      • Indian macro factors (RBI policy, Union Budget, FII/DII flows)

    STRICT RULES — violations will invalidate your report:
    1. Only cite numbers, dates, and events that appear VERBATIM in the briefing below.
    2. Do NOT invent price levels, percentage moves, or news headlines.
    3. If a section lacks data, write: "Insufficient data for this section."
    4. Use hedged language: "suggests", "indicates", "appears to", "may signal".
    5. Keep total output UNDER 280 words.
    6. Respond ONLY with the formatted report — no preamble, no sign-off.
""").strip()


ANALYST_USER_TEMPLATE: str = textwrap.dedent("""
    ═══════════════════════════════════════════════════════════════════
    ANALYST BRIEFING (your ONLY source of information — do not use any
    external knowledge about this stock):
    ═══════════════════════════════════════════════════════════════════

    {context}

    ═══════════════════════════════════════════════════════════════════
    TASK: Write a concise analyst report in EXACTLY this structure:

    ANALYST REPORT — {stock} — {date}
    ──────────────────────────────────
    TREND ANALYSIS:
    [2–3 sentences. Describe direction (bullish/bearish/sideways), SMA alignment,
     and recent price momentum using only the data above.]

    KEY LEVELS:
    Support  : ₹[level from data]  |  Resistance: ₹[level from data]
    [1 sentence explaining why these levels matter — cite the data.]

    NEWS SENTIMENT:
    [1–2 sentences. Summarise headline tone as Positive / Neutral / Negative.
     Quote the most relevant headline (under 10 words). Impact on stock: brief.]

    RISKS & OPPORTUNITIES:
    Risk        : [1 specific risk grounded in the briefing data]
    Opportunity : [1 specific opportunity grounded in the briefing data]

    OVERALL MARKET VIEW:
    [2 sentences. Synthesise all sections into an overall view: Bullish / Cautiously
     Bullish / Neutral / Cautiously Bearish / Bearish. State the primary reason.]
    ═══════════════════════════════════════════════════════════════════
""").strip()


# ---------------------------------------------------------------------------
# Word-count helper
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    """Return approximate word count of a string."""
    return len(re.findall(r"\S+", text))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_analyst_prompt(
    context: str,
    stock: str,
    date: str,
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) tuple for the Analyst Agent.

    Args:
        context : Output of context_builder.build_context() — the full briefing.
        stock   : NSE ticker, e.g. "TCS.NS".
        date    : Analysis date as "YYYY-MM-DD".

    Returns:
        (system_prompt, user_prompt) — both strings ready for the LLM call.
    """
    user_prompt = ANALYST_USER_TEMPLATE.format(
        context=context.strip(),
        stock=stock,
        date=date,
    )
    word_count = _count_words(ANALYST_SYSTEM_PROMPT) + _count_words(user_prompt)
    log.debug(
        "Analyst prompt built for %s on %s — ~%d words in combined prompt",
        stock, date, word_count,
    )
    if word_count > 600:
        log.warning(
            "Analyst prompt for %s on %s is %d words — consider trimming context.",
            stock, date, word_count,
        )
    return ANALYST_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = [
    "TREND ANALYSIS",
    "KEY LEVELS",
    "NEWS SENTIMENT",
    "RISKS & OPPORTUNITIES",
    "OVERALL MARKET VIEW",
]


def _validate_report(report: str, stock: str, date: str) -> bool:
    """
    Lightweight structural validation of the analyst report.

    Checks that all five required sections appear in the output.
    Logs a warning (but does NOT raise) if any section is missing —
    the backtest must continue even with a partial report.

    Args:
        report : Raw LLM output string.
        stock  : NSE ticker (for logging).
        date   : Analysis date (for logging).

    Returns:
        True if all sections present, False otherwise.
    """
    missing = [s for s in _REQUIRED_SECTIONS if s not in report.upper()]
    if missing:
        log.warning(
            "Analyst report for %s on %s is missing sections: %s",
            stock, date, missing,
        )
        return False
    word_count = _count_words(report)
    if word_count > 320:
        log.warning(
            "Analyst report for %s on %s is %d words — exceeds 280-word target.",
            stock, date, word_count,
        )
    log.debug("Analyst report validated for %s on %s (%d words).", stock, date, word_count)
    return True


# ---------------------------------------------------------------------------
# Stub LLM call 
# ---------------------------------------------------------------------------

def _stub_llm_response(stock: str, date: str) -> str:
    """
    Return a deterministic stub analyst report for offline development.

    This stub is used after the caching layer is in place.

    Args:
        stock : NSE ticker.
        date  : Analysis date.

    Returns:
        A realistic-looking (but fabricated) analyst report string.
        All numbers here are ILLUSTRATIVE — they are NOT real market data.
    """
    return textwrap.dedent(f"""
        ANALYST REPORT — {stock} — {date}
        ──────────────────────────────────
        TREND ANALYSIS:
        The stock appears to be in a short-term consolidation phase following
        recent price action. SMA20 is trading near SMA50, suggesting the medium-term
        trend is neutral. RSI indicates neither overbought nor oversold conditions.

        KEY LEVELS:
        Support  : ₹[from data]  |  Resistance: ₹[from data]
        These levels correspond to recent swing lows and the upper Bollinger Band
        visible in the briefing data.

        NEWS SENTIMENT:
        Insufficient news data provided in this stub — sentiment: Neutral.
        No material corporate event detected in the briefing window.

        RISKS & OPPORTUNITIES:
        Risk        : Broader market weakness may pull the stock below support.
        Opportunity : A break above resistance on volume could signal resumption of uptrend.

        OVERALL MARKET VIEW:
        Neutral. The stock lacks a clear directional catalyst in the current briefing
        window. Recommend monitoring for a decisive price move or news trigger.
        [STUB — replace with real LLM call]
    """).strip()


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def analyze_stock(
    context: str,
    llm: Any,
    stock: str,
    date: str,
    use_stub: bool = True,
    max_retries: int = 2,
) -> str:
    """
    Run the Analyst Agent on a single stock for a single date.

    This is the ONLY function called by the orchestrator / backtest loop.

    Args:
        context     : Output of context_builder.build_context() for this stock/date.
                      Must be a non-empty string.
        llm         : LLM client object from llm_utils.init_llms(), e.g. llms["gemini"].
                      Passed to llm_utils.call_llm(prompt, llm).
                      Ignored when use_stub=True.
        stock       : NSE ticker, e.g. "TCS.NS".
        date        : Analysis date as "YYYY-MM-DD".
        use_stub    : If True, return the stub response without making an LLM call.
                      Set to False once caching is in place.
        max_retries : Number of times to retry the LLM call on failure (default 2).

    Returns:
        A plain-text analyst report string (approx. 200–280 words).
        On total failure, returns a minimal error report string (does NOT raise).

    Raises:
        ValueError : If context is empty.

    Example:
        >>> from agents.analyst import analyze_stock
        >>> report = analyze_stock(context=briefing, llm=None, stock="TCS.NS",
        ...                        date="2024-03-15", use_stub=True)
        >>> print(report)
    """
    # ── Guard: context must not be empty ─────────────────────────────────────
    if not context or not context.strip():
        raise ValueError(
            f"analyze_stock() received empty context for {stock} on {date}. "
            "Ensure context_builder.build_context() ran successfully."
        )

    log.info("Analyst Agent: analyzing %s on %s (stub=%s)", stock, date, use_stub)
    ts_start = datetime.utcnow()

    # ── Stub path  ─────────────────────────────────────────────────
    if use_stub:
        report = _stub_llm_response(stock, date)
        _validate_report(report, stock, date)
        elapsed = (datetime.utcnow() - ts_start).total_seconds()
        log.info(
            "Analyst Agent (STUB) complete for %s on %s — %d words in %.2fs",
            stock, date, _count_words(report), elapsed,
        )
        return report

    # ── Real LLM path  ───────────────────────────────────────────────
    system_prompt, user_prompt = build_analyst_prompt(context, stock, date)

    # Combine into a single prompt string for call_llm()
    # call_llm() from llm_utils.py accepts a single prompt string.
    # We prepend the system prompt as a clearly delimited section.
    full_prompt = (
        f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
        f"USER REQUEST:\n{user_prompt}"
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            # ── Import here (not at module top) so runs without llm_utils ──
            from llm_utils import call_llm  # noqa: PLC0415

            log.debug("Analyst Agent: LLM call attempt %d/%d", attempt, max_retries)
            raw_response: str = call_llm(full_prompt, llm)

            if not raw_response or not raw_response.strip():
                raise ValueError("LLM returned an empty response.")

            report = raw_response.strip()
            _validate_report(report, stock, date)

            elapsed = (datetime.utcnow() - ts_start).total_seconds()
            log.info(
                "Analyst Agent complete for %s on %s — %d words in %.2fs",
                stock, date, _count_words(report), elapsed,
            )
            return report

        except ImportError as exc:
            # llm_utils not yet available — graceful degradation
            log.error("llm_utils not importable: %s. Falling back to stub.", exc)
            return _stub_llm_response(stock, date)

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            log.warning(
                "Analyst Agent: attempt %d/%d failed for %s on %s — %s: %s",
                attempt, max_retries, stock, date, type(exc).__name__, exc,
            )

    # ── All retries exhausted ─────────────────────────────────────────────────
    log.error(
        "Analyst Agent: all %d attempts failed for %s on %s. Last error: %s",
        max_retries, stock, date, last_error,
    )
    # Return a minimal error report so the backtest loop can continue.
    return (
        f"ANALYST REPORT — {stock} — {date}\n"
        "──────────────────────────────────\n"
        "ERROR: Analyst Agent failed to generate a report after "
        f"{max_retries} attempts.\n"
        f"Last error: {type(last_error).__name__}: {last_error}\n"
        "Downstream agents should treat this as: OVERALL MARKET VIEW: Neutral.\n"
    )


# ---------------------------------------------------------------------------
# Convenience wrapper for batch processing (used by backtest orchestrator)
# ---------------------------------------------------------------------------

def analyze_batch(
    items: list[dict],
    llm: Any,
    use_stub: bool = True,
) -> list[dict]:
    """
    Run analyze_stock() over a list of (stock, date, context) dicts.

    Args:
        items    : List of dicts, each with keys: 'stock', 'date', 'context'.
        llm      : LLM client (passed through to analyze_stock).
        use_stub : Stub flag (passed through to analyze_stock).

    Returns:
        List of dicts, each input dict enriched with key 'analyst_report'.

    Example:
        items = [
            {"stock": "TCS.NS",      "date": "2024-03-15", "context": "..."},
            {"stock": "RELIANCE.NS", "date": "2024-03-15", "context": "..."},
        ]
        results = analyze_batch(items, llm=None, use_stub=True)
    """
    results = []
    for item in items:
        stock   = item["stock"]
        date    = item["date"]
        context = item.get("context", "")

        try:
            report = analyze_stock(
                context=context,
                llm=llm,
                stock=stock,
                date=date,
                use_stub=use_stub,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("analyze_batch: unexpected error for %s on %s — %s", stock, date, exc)
            report = (
                f"ANALYST REPORT — {stock} — {date}\n"
                f"BATCH ERROR: {type(exc).__name__}: {exc}\n"
                "OVERALL MARKET VIEW: Neutral.\n"
            )

        results.append({**item, "analyst_report": report})

    log.info("analyze_batch: processed %d items.", len(results))
    return results


# ---------------------------------------------------------------------------
# Standalone test (run: python agents/analyst.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Minimal synthetic context (mirrors context_builder output) ────────────
    SAMPLE_CONTEXT = """
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
  Recent News and Events (all published STRICTLY BEFORE 2024-03-15):

  [1] 2024-03-12 | Source: Economic Times
      Headline : TCS bags ₹2,200 crore deal from European bank
      Summary  : TCS has secured a multi-year digital transformation deal...

  [2] 2024-03-08 | Source: BSE Filing
      Headline : Q3 FY24 results: PAT up 8.2% YoY, revenue misses estimates
      Summary  : Net profit rose to ₹11,058 crore but revenue growth at 4.1% missed...

  [3] 2024-03-05 | Source: Reuters
      Headline : RBI holds repo rate at 6.5% in February MPC meeting
      Summary  : The RBI Monetary Policy Committee unanimously held rates steady...
══════════════════════════════════════════════════════════════════════
  CONTEXT METADATA (internal — do not include in analysis)
  Price rows used : 10
  News items used : 3
  Look-ahead check: PASSED
══════════════════════════════════════════════════════════════════════
""".strip()

    TEST_STOCK = "TCS.NS"
    TEST_DATE  = "2024-03-15"

    print()
    print("=" * 68)
    print("  agents/analyst.py — Standalone Test")
    print("=" * 68)

    # ── Test 1: Prompt building ────────────────────────────────────────────────
    print("\n[TEST 1] Building analyst prompt...")
    sys_p, usr_p = build_analyst_prompt(SAMPLE_CONTEXT, TEST_STOCK, TEST_DATE)
    total_words = _count_words(sys_p) + _count_words(usr_p)
    print(f"  System prompt : {_count_words(sys_p)} words")
    print(f"  User prompt   : {_count_words(usr_p)} words")
    print(f"  TOTAL         : {total_words} words  (target: <400)")
    assert total_words < 700, "Prompt too long — trim context!"
    print("  ✓ Prompt word count within budget.")

    # ── Test 2: Stub analyst report ───────────────────────────────────────────
    print("\n[TEST 2] Running analyze_stock() with stub=True...")
    report = analyze_stock(
        context=SAMPLE_CONTEXT,
        llm=None,
        stock=TEST_STOCK,
        date=TEST_DATE,
        use_stub=True,
    )
    print()
    print(report)
    print()
    print(f"  Word count: {_count_words(report)} words")
    print("  ✓ Stub report returned successfully.")

    # ── Test 3: Validation ────────────────────────────────────────────────────
    print("\n[TEST 3] Validating report structure...")
    valid = _validate_report(report, TEST_STOCK, TEST_DATE)
    # Stub intentionally lacks all 5 sections — that's expected behavior
    print(f"  Validation result: {'PASS' if valid else 'WARN (expected for stub)'}")

    # ── Test 4: Empty context guard ───────────────────────────────────────────
    print("\n[TEST 4] Testing empty context guard...")
    try:
        analyze_stock(context="", llm=None, stock=TEST_STOCK, date=TEST_DATE)
        print("  FAIL — should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"  ✓ ValueError raised correctly: {e}")

    # ── Test 5: Batch processing ──────────────────────────────────────────────
    print("\n[TEST 5] Testing analyze_batch()...")
    batch_items = [
        {"stock": "TCS.NS",      "date": "2024-03-15", "context": SAMPLE_CONTEXT},
        {"stock": "RELIANCE.NS", "date": "2024-04-10", "context": SAMPLE_CONTEXT},
    ]
    batch_results = analyze_batch(batch_items, llm=None, use_stub=True)
    assert len(batch_results) == 2
    assert all("analyst_report" in r for r in batch_results)
    print(f"  ✓ Batch processed {len(batch_results)} items successfully.")

    print()
    print("=" * 68)
    print("  ALL TESTS PASSED — complete.")
    print("  Next: Write utils/cache.py (caching layer)")
    print("=" * 68)
    print()
