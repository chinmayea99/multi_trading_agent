"""
data/test_pipeline_day26.py — End-to-End Data Pipeline Test

What we are doing:
    Testing every component of the data pipeline together on TCS for two dates:
      - 2024-03-15  (historical date within year 1)
      - 2025-09-10  (historical date within year 2)
    For each date we call: price_context(), get_news_context(), event calendar
    lookup, and combine all outputs into a single analyst-ready briefing.

Why we are doing it:
    Before building any agent logic, we must prove the data pipeline is solid.
    A bug here (wrong prices, look-ahead news, missing indicators) invalidates
    every experiment downstream. This test acts as a pre-flight check.

How we are doing it:
    1. price_context()   — loads price CSV, slices last N rows, reads indicators
    2. get_news_context()— loads news CSV, filters strictly before query_date
    3. event_lookup()    — checks indian_events.csv for same-day / upcoming events
    4. combine_context() — merges all three into one formatted briefing string
    5. Data quality checks run after every section; failures are printed clearly.

Run:
    cd path\\to\\your\\project
    python data/test_pipeline_day26.py

Expected output:
    Two full analyst briefings printed to screen (one per date), each containing
    price table, technical indicators, news items, and event flags.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Path setup: allow running from project root OR from data/ folder ──────────
ROOT = Path(__file__).resolve().parent.parent   # adjust if script is in data/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── If script lives in project root, tweak ROOT accordingly ──────────────────
# If your project root already IS the parent of config.py, ROOT is correct.
# If config.py is at the same level as this script, use:
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Coloured output (degrades gracefully if colorama not installed) ───────────
try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (mirrors config.py — we import it if available)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import config
    PRICES_DIR   = config.PRICES_DIR        # e.g. "data/prices"
    NEWS_DIR     = config.NEWS_DIR          # e.g. "data/news"
    EVENTS_CSV   = config.EVENTS_CSV        # e.g. "data/events/indian_events.csv"
    HOLIDAYS_CSV = config.HOLIDAYS_CSV      # e.g. "data/events/market_holidays_2024.csv"
    EARNINGS_CSV = config.EARNINGS_CSV      # e.g. "data/events/earnings_calendar_2024.csv"
    PRICE_LOOKBACK_DAYS = config.PRICE_LOOKBACK_DAYS   # e.g. 10
    NEWS_MAX_ITEMS      = config.NEWS_MAX_ITEMS         # e.g. 3
    print(f"{GREEN}✓ config.py loaded successfully{RESET}")
except ImportError:
    # Fallback defaults — edit these to match your actual folder structure
    PRICES_DIR          = "data/prices"
    NEWS_DIR            = "data/news"
    EVENTS_CSV          = "data/events/indian_events.csv"
    HOLIDAYS_CSV        = "data/events/market_holidays_2024.csv"
    EARNINGS_CSV        = "data/events/earnings_calendar_2024.csv"
    PRICE_LOOKBACK_DAYS = 10
    NEWS_MAX_ITEMS      = 3
    print(f"{YELLOW}⚠ config.py not found — using fallback paths{RESET}")

# ── Test parameters ───────────────────────────────────────────────────────────
STOCK     = "TCS.NS"
COMPANY   = "Tata Consultancy Services"
TEST_DATES = ["2024-03-15", "2025-09-10"]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PRICE CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PRICE_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
REQUIRED_INDICATOR_COLS = ["SMA_20", "SMA_50", "RSI_14"]   # adjust to your columns


def price_context(stock: str, date: str, lookback: int = PRICE_LOOKBACK_DAYS) -> dict:
    """
    Load price CSV for `stock`, return the last `lookback` rows up to and
    including `date`, plus technical indicator values on `date`.

    Returns a dict:
        {
            "price_table"  : pd.DataFrame  — last N rows (Date, OHLCV, indicators),
            "latest_close" : float,
            "latest_date"  : str,
            "indicators"   : dict,          — {col: value} on query_date row
            "errors"       : list[str],     — any data quality issues found
        }
    """
    result = {
        "price_table": pd.DataFrame(),
        "latest_close": None,
        "latest_date": None,
        "indicators": {},
        "errors": [],
    }

    # ── 1. Locate CSV ─────────────────────────────────────────────────────────
    # Try multiple filename conventions in order:
    #   TCS.csv          — bare ticker (your actual file)
    #   TCS_NS.csv       — ticker with exchange suffix replaced
    #   BAJAJ-AUTO_NS.csv — handles hyphens too
    ticker_base = stock.split(".")[0]                          # "TCS"  from "TCS.NS"
    ticker_safe = stock.replace(".", "_").replace("&", "_")    # "TCS_NS"

    candidates = [
        os.path.join(PRICES_DIR, f"{ticker_base}.csv"),       # TCS.csv  ← your format
        os.path.join(PRICES_DIR, f"{ticker_safe}.csv"),        # TCS_NS.csv
        os.path.join(PRICES_DIR, f"{ticker_base.replace('-','_')}.csv"),  # BAJAJ_AUTO.csv
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)

    if csv_path is None:
        result["errors"].append(
            f"Price CSV not found. Tried: {[os.path.basename(p) for p in candidates]} "
            f"in '{PRICES_DIR}'  (run data/download_prices.py — Day 3–5 first)"
        )
        return result

    # ── 2. Load & parse ───────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]   # strip accidental whitespace

    # Normalise Date column name
    if "Date" not in df.columns and "date" in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)

    missing_base = [c for c in REQUIRED_PRICE_COLS if c not in df.columns]
    if missing_base:
        result["errors"].append(f"Missing OHLCV columns: {missing_base}")
        return result

    # ── Detect date format: DD-MM-YYYY (Indian files) vs YYYY-MM-DD ─────────
    sample = str(df["Date"].dropna().iloc[0]) if not df["Date"].dropna().empty else ""
    parts  = sample.replace("/", "-").split("-")
    if len(parts) == 3 and len(parts[0]) == 2:
        # Two-digit leading component → DD-MM-YYYY (Indian convention)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date")

    # ── 3. Check indicator columns (warn but don't abort) ─────────────────────
    missing_indicators = [c for c in REQUIRED_INDICATOR_COLS if c not in df.columns]
    if missing_indicators:
        result["errors"].append(
            f"Indicator columns missing: {missing_indicators}  "
            f"(run data/add_indicators.py — Day 13 first)"
        )

    # ── 4. Slice up to query_date (no look-ahead) ─────────────────────────────
    query_dt = pd.to_datetime(date)
    past = df[df["Date"] <= query_dt]   # include query_date itself for price

    if past.empty:
        result["errors"].append(
            f"No price rows found on or before {date}. "
            f"Earliest data: {df['Date'].min().date()}"
        )
        return result

    window = past.tail(lookback)

    # ── 5. Data quality checks ────────────────────────────────────────────────
    nan_counts = window[REQUIRED_PRICE_COLS].isna().sum()
    for col, cnt in nan_counts.items():
        if cnt > 0:
            result["errors"].append(f"NaN in {col}: {cnt} row(s)")

    zero_vol = (window["Volume"] <= 0).sum()
    if zero_vol > 0:
        result["errors"].append(f"Zero/negative Volume: {zero_vol} row(s)")

    neg_close = (window["Close"] <= 0).sum()
    if neg_close > 0:
        result["errors"].append(f"Zero/negative Close: {neg_close} row(s)")

    # Check OHLC consistency
    bad_hl = (window["High"] < window["Low"]).sum()
    if bad_hl > 0:
        result["errors"].append(f"High < Low: {bad_hl} row(s) — data corruption")

    # Large single-day moves (>20%)
    pct_chg = window["Close"].pct_change().abs()
    big_moves = (pct_chg > 0.20).sum()
    if big_moves > 0:
        result["errors"].append(
            f"Single-day move >20% detected: {big_moves} row(s) — verify if real"
        )

    # ── 6. Extract latest row's indicator values ──────────────────────────────
    latest_row = window.iloc[-1]
    result["latest_close"] = float(latest_row["Close"])
    result["latest_date"]  = latest_row["Date"].strftime("%Y-%m-%d")

    for col in REQUIRED_INDICATOR_COLS:
        if col in window.columns:
            val = latest_row.get(col, None)
            result["indicators"][col] = float(val) if pd.notna(val) else None

    result["price_table"] = window.reset_index(drop=True)
    return result


def format_price_context(pctx: dict, stock: str, date: str) -> str:
    """Format price context dict as a human/LLM-readable string."""
    lines = []
    lines.append(f"━━━ PRICE CONTEXT : {stock} | Query date: {date} ━━━")

    if pctx["errors"]:
        lines.append(f"{YELLOW}⚠ Data quality issues:")
        for e in pctx["errors"]:
            lines.append(f"    • {e}")
        lines.append(RESET)

    if pctx["price_table"].empty:
        lines.append("  [No price data available]")
        return "\n".join(lines)

    # Price table
    df = pctx["price_table"]
    cols_to_show = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if c in df.columns]
    lines.append(f"\nLast {len(df)} trading days of price data:")
    lines.append(df[cols_to_show].to_string(index=False))

    # Indicators
    if pctx["indicators"]:
        lines.append("\nTechnical indicators on latest date:")
        for k, v in pctx["indicators"].items():
            val_str = f"{v:.2f}" if v is not None else "N/A"
            lines.append(f"  {k:12s}: {val_str}")

    lines.append(f"\nLatest close  : ₹{pctx['latest_close']:.2f}  (on {pctx['latest_date']})")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — NEWS CONTEXT  (imports from existing get_news_context.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_news_context(stock: str, date: str, max_items: int = NEWS_MAX_ITEMS) -> dict:
    """
    Wrapper around get_news_context.get_news_context().
    Falls back to a direct CSV read if the module import fails.

    Returns:
        {
            "items"  : list[dict],   — [{date, headline, source, summary}, ...]
            "errors" : list[str],
            "method" : str,          — "module" or "direct"
        }
    """
    result = {"items": [], "errors": [], "method": "unknown"}

    # ── Try to import the existing module first ───────────────────────────────
    try:
        from get_news_context import get_news_context, format_news_context
        items = get_news_context(stock, date, max_items=max_items)
        result["items"]  = items
        result["method"] = "module"

        # ── Look-ahead assertion (belt-and-suspenders) ────────────────────────
        query_dt = pd.to_datetime(date)
        for item in items:
            if item["date"] >= query_dt:
                result["errors"].append(
                    f"LOOK-AHEAD BIAS: item date {item['date'].date()} "
                    f">= query date {date}"
                )
        return result

    except ImportError:
        result["errors"].append(
            "get_news_context module not importable — falling back to direct CSV read"
        )

    # ── Fallback: read CSV directly ────────────────────────────────────────────
    ticker_safe = stock.replace(".", "_").replace("&", "_")
    csv_path = os.path.join(NEWS_DIR, f"{ticker_safe}_news.csv")

    if not os.path.exists(csv_path):
        result["errors"].append(
            f"News CSV not found: {csv_path}  "
            f"(run data/merge_news.py — Day 19 first)"
        )
        return result

    try:
        news_df = pd.read_csv(csv_path)
        required = {"date", "headline"}
        missing  = required - set(news_df.columns)
        if missing:
            result["errors"].append(f"News CSV missing columns: {missing}")
            return result

        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
        news_df = news_df.dropna(subset=["date"])

        query_dt = pd.to_datetime(date)
        prior    = news_df[news_df["date"] < query_dt]   # STRICT less-than
        recent   = prior.sort_values("date", ascending=False).head(max_items)

        # Confirm no look-ahead
        violators = recent[recent["date"] >= query_dt]
        if not violators.empty:
            result["errors"].append(
                f"LOOK-AHEAD BIAS in fallback path! Rows: "
                f"{violators[['date','headline']].to_string()}"
            )

        items = []
        for _, row in recent.iterrows():
            items.append({
                "date":     row["date"],
                "headline": str(row.get("headline", "")),
                "source":   str(row.get("source", "Unknown")),
                "summary":  str(row.get("summary", "")),
            })

        result["items"]  = items
        result["method"] = "direct_csv"

    except Exception as exc:
        result["errors"].append(f"Direct CSV read failed: {exc}")

    return result


def format_news_section(news_result: dict, date: str) -> str:
    """Format news context dict as a string."""
    lines = []
    lines.append(f"━━━ NEWS CONTEXT | Strictly before {date} ━━━")

    if news_result["errors"]:
        for e in news_result["errors"]:
            lines.append(f"{RED}  ✗ {e}{RESET}")

    items = news_result["items"]
    if not items:
        lines.append("  [No news items found before this date]")
        lines.append("  (This is valid if the stock had no coverage before this date)")
        return "\n".join(lines)

    lines.append(f"  Source method: {news_result['method']} | {len(items)} item(s)\n")
    for i, item in enumerate(items, 1):
        date_str = item["date"].strftime("%Y-%m-%d")
        lines.append(f"  [{i}] {date_str} | {item['source']}")
        lines.append(f"      Headline : {item['headline']}")
        summary = str(item.get("summary", ""))
        if summary and summary.lower() not in ("nan", "none", ""):
            short = summary[:200] + "..." if len(summary) > 200 else summary
            lines.append(f"      Summary  : {short}")
        lines.append("")

    # Look-ahead check summary
    query_dt = pd.to_datetime(date)
    bad = [item for item in items if item["date"] >= query_dt]
    if bad:
        lines.append(f"{RED}  ✗ LOOK-AHEAD DETECTED: {len(bad)} item(s) ON or AFTER {date}{RESET}")
    else:
        lines.append(f"{GREEN}  ✓ No look-ahead bias detected (all items strictly before {date}){RESET}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EVENT CALENDAR LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def event_lookup(stock: str, date: str, lookahead_days: int = 7) -> dict:
    """
    Check indian_events.csv, earnings_calendar and market_holidays CSV files
    for events on `date` or within the next `lookahead_days` days.

    Returns:
        {
            "same_day"   : list[str],   — event descriptions on query_date
            "upcoming"   : list[str],   — events in next lookahead_days days
            "is_holiday" : bool,
            "errors"     : list[str],
        }
    """
    result = {
        "same_day":   [],
        "upcoming":   [],
        "is_holiday": False,
        "errors":     [],
    }

    query_dt  = pd.to_datetime(date)
    end_dt    = query_dt + timedelta(days=lookahead_days)

    # ── Helper: load any CSV safely ───────────────────────────────────────────
    def _load(path):
        if not os.path.exists(path):
            return None, f"Not found: {path}"
        try:
            return pd.read_csv(path), None
        except Exception as exc:
            return None, str(exc)

    # ── 1. Market holidays ────────────────────────────────────────────────────
    hol_df, hol_err = _load(HOLIDAYS_CSV)
    if hol_err:
        result["errors"].append(f"Holidays CSV: {hol_err}")
    elif hol_df is not None:
        date_col = next(
            (c for c in hol_df.columns if "date" in c.lower()), None
        )
        if date_col:
            hol_df[date_col] = pd.to_datetime(hol_df[date_col], dayfirst=True, errors="coerce")
            if (hol_df[date_col] == query_dt).any():
                result["is_holiday"] = True
                result["same_day"].append("⚠ NSE MARKET HOLIDAY — no trading today")

    # ── 2. Indian market events (RBI, Budget, etc.) ───────────────────────────
    ev_df, ev_err = _load(EVENTS_CSV)
    if ev_err:
        result["errors"].append(f"Events CSV: {ev_err}")
    elif ev_df is not None:
        # Normalise columns
        ev_df.columns = [c.strip().lower() for c in ev_df.columns]
        date_col = next((c for c in ev_df.columns if "date" in c), None)
        desc_col = next(
            (c for c in ev_df.columns if any(k in c for k in ["event","desc","name"])), None
        )

        if date_col and desc_col:
            ev_df[date_col] = pd.to_datetime(ev_df[date_col], dayfirst=True, errors="coerce")

            # Filter for stock relevance if 'stock' or 'ticker' column exists
            stk_col = next((c for c in ev_df.columns if "stock" in c or "ticker" in c), None)
            if stk_col:
                ev_df = ev_df[
                    ev_df[stk_col].isna() |
                    ev_df[stk_col].str.strip().str.upper().eq(stock.upper()) |
                    ev_df[stk_col].str.strip().str.upper().eq("ALL")
                ]

            same  = ev_df[ev_df[date_col] == query_dt]
            ahead = ev_df[(ev_df[date_col] > query_dt) & (ev_df[date_col] <= end_dt)]

            for _, row in same.iterrows():
                result["same_day"].append(str(row[desc_col]))

            for _, row in ahead.iterrows():
                days_away = (row[date_col] - query_dt).days
                result["upcoming"].append(
                    f"In {days_away}d: {row[date_col].strftime('%Y-%m-%d')} — {row[desc_col]}"
                )
        else:
            result["errors"].append(
                f"Events CSV columns not recognised: {list(ev_df.columns)}"
            )

    # ── 3. Earnings calendar ──────────────────────────────────────────────────
    earn_df, earn_err = _load(EARNINGS_CSV)
    if earn_err:
        result["errors"].append(f"Earnings CSV: {earn_err}")
    elif earn_df is not None:
        earn_df.columns = [c.strip().lower() for c in earn_df.columns]
        date_col = next((c for c in earn_df.columns if "date" in c), None)
        stk_col  = next(
            (c for c in earn_df.columns if "stock" in c or "ticker" in c), None
        )
        desc_col = next(
            (c for c in earn_df.columns
             if any(k in c for k in ["event","quarter","desc","q"])), None
        )

        if date_col:
            earn_df[date_col] = pd.to_datetime(earn_df[date_col], dayfirst=False, errors="coerce")

            # Filter for this stock
            if stk_col:
                earn_df = earn_df[
                    earn_df[stk_col].str.strip().str.upper().eq(stock.upper())
                ]

            same  = earn_df[earn_df[date_col] == query_dt]
            ahead = earn_df[(earn_df[date_col] > query_dt) & (earn_df[date_col] <= end_dt)]

            for _, row in same.iterrows():
                desc = str(row[desc_col]) if desc_col else "Earnings results"
                result["same_day"].append(f"📊 EARNINGS DAY: {desc}")

            for _, row in ahead.iterrows():
                days_away = (row[date_col] - query_dt).days
                desc = str(row[desc_col]) if desc_col else "Earnings results"
                result["upcoming"].append(
                    f"In {days_away}d: {row[date_col].strftime('%Y-%m-%d')} — 📊 Earnings: {desc}"
                )

    return result


def format_event_section(ev: dict, date: str) -> str:
    """Format event lookup dict as a string."""
    lines = []
    lines.append(f"━━━ EVENT CALENDAR | {date} ━━━")

    if ev["errors"]:
        for e in ev["errors"]:
            lines.append(f"  {YELLOW}⚠ {e}{RESET}")

    if ev["is_holiday"]:
        lines.append(f"  {RED}⚠ THIS IS AN NSE MARKET HOLIDAY — agent should not trade{RESET}")

    if ev["same_day"]:
        lines.append("  Events TODAY:")
        for e in ev["same_day"]:
            lines.append(f"    🔴 {e}")
    else:
        lines.append("  Events today: None")

    if ev["upcoming"]:
        lines.append("  Upcoming events (next 7 days):")
        for e in ev["upcoming"]:
            lines.append(f"    🟡 {e}")
    else:
        lines.append("  Upcoming events: None in next 7 days")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COMBINE INTO ANALYST BRIEFING
# ─────────────────────────────────────────────────────────────────────────────

def build_analyst_briefing(
    stock:    str,
    company:  str,
    date:     str,
    pctx:     dict,
    news_ctx: dict,
    ev:       dict,
) -> str:
    """
    Combine price, news, and event context into a single analyst-ready briefing.
    This string is the exact input format that will be fed to the Analyst Agent.
    """
    sep = "=" * 70

    header = (
        f"\n{sep}\n"
        f"  ANALYST BRIEFING\n"
        f"  Stock   : {company} ({stock})\n"
        f"  Date    : {date}\n"
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{sep}\n"
    )

    # ── Data quality summary ──────────────────────────────────────────────────
    all_errors = pctx["errors"] + news_ctx["errors"] + ev["errors"]
    if all_errors:
        dq = f"\n{YELLOW}DATA QUALITY WARNINGS:\n"
        for e in all_errors:
            dq += f"  • {e}\n"
        dq += RESET
    else:
        dq = f"\n{GREEN}✓ All data quality checks passed{RESET}\n"

    price_section = "\n" + format_price_context(pctx, stock, date)
    news_section  = "\n" + format_news_section(news_ctx, date)
    event_section = "\n" + format_event_section(ev, date)

    # ── Brief interpretive summary (template — agent fills in real analysis) ──
    close = pctx.get("latest_close")
    sma20 = pctx["indicators"].get("SMA_20")
    sma50 = pctx["indicators"].get("SMA_50")
    rsi   = pctx["indicators"].get("RSI_14")

    summary_lines = ["\n━━━ QUICK CONTEXT SUMMARY ━━━"]

    if close and sma20 and sma50:
        trend = "above" if close > sma20 else "below"
        bias  = "BULLISH bias" if close > sma20 > sma50 else (
                "BEARISH bias" if close < sma20 < sma50 else "MIXED trend")
        summary_lines.append(f"  Price ₹{close:.2f} is {trend} SMA20 (₹{sma20:.2f}) — {bias}")
    elif close:
        summary_lines.append(f"  Latest close: ₹{close:.2f}")

    if rsi is not None:
        zone = "OVERBOUGHT (>70)" if rsi > 70 else ("OVERSOLD (<30)" if rsi < 30 else "NEUTRAL (30-70)")
        summary_lines.append(f"  RSI {rsi:.1f} — {zone}")

    n_news = len(news_ctx["items"])
    summary_lines.append(
        f"  News: {n_news} item(s) available from before {date}"
        if n_news else f"  News: No items found before {date}"
    )

    if ev["same_day"]:
        summary_lines.append(f"  ⚠ {len(ev['same_day'])} event(s) on this exact date")
    if ev["upcoming"]:
        summary_lines.append(f"  Upcoming: {len(ev['upcoming'])} event(s) in next 7 days")

    summary = "\n".join(summary_lines)

    return header + dq + price_section + news_section + event_section + summary


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — VALIDATION CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_checklist(
    stock: str,
    date: str,
    pctx: dict,
    news_ctx: dict,
    ev: dict,
) -> dict:
    """
    Run all validation checks. Returns pass/fail for each item.
    This is logged to screen and used to decide if the date is safe to use
    in backtesting.
    """
    checks = {}

    # Price checks
    checks["price_csv_exists"] = not any(
        "not found" in e.lower() for e in pctx["errors"]
    ) and not pctx["price_table"].empty

    checks["price_has_n_rows"] = len(pctx["price_table"]) >= 3

    checks["no_nan_ohlcv"] = not any(
        "nan" in e.lower() for e in pctx["errors"]
    )

    checks["no_negative_close"] = not any(
        "negative close" in e.lower() for e in pctx["errors"]
    )

    checks["ohlc_consistent"] = not any(
        "high < low" in e.lower() for e in pctx["errors"]
    )

    checks["indicators_present"] = bool(pctx["indicators"])

    # News checks
    checks["news_csv_exists"] = not any(
        "not found" in e.lower() for e in news_ctx["errors"]
    )

    checks["no_lookahead_bias"] = not any(
        "look-ahead" in e.lower() for e in news_ctx["errors"]
    )

    # Verify manually that all news dates < query_date
    query_dt = pd.to_datetime(date)
    items = news_ctx["items"]
    all_before = all(item["date"] < query_dt for item in items)
    checks["news_dates_strictly_before"] = all_before

    # Event checks
    checks["event_calendar_loaded"] = not any(
        "not found" in e.lower() for e in ev["errors"]
    )

    checks["not_a_holiday"] = not ev["is_holiday"]

    return checks


def print_checklist(checks: dict, stock: str, date: str):
    """Print the validation checklist in a readable table."""
    print(f"\n{'─'*60}")
    print(f"  VALIDATION CHECKLIST : {stock} | {date}")
    print(f"{'─'*60}")
    passed = 0
    failed = 0
    for key, val in checks.items():
        icon   = f"{GREEN}✓{RESET}" if val else f"{RED}✗{RESET}"
        label  = key.replace("_", " ").title()
        print(f"  {icon}  {label}")
        if val:
            passed += 1
        else:
            failed += 1
    print(f"{'─'*60}")
    print(f"  {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}")
    if failed == 0:
        print(f"  {GREEN}✓ This date is SAFE for backtesting.{RESET}")
    else:
        print(f"  {RED}✗ Fix the above issues before using this date in backtests!{RESET}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run tests for both dates
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'#'*70}")
    print(f"   END-TO-END DATA PIPELINE TEST")
    print(f"  Stock : {COMPANY} ({STOCK})")
    print(f"  Dates : {', '.join(TEST_DATES)}")
    print(f"{'#'*70}\n")

    briefings = {}

    for date in TEST_DATES:
        print(f"\n{'*'*70}")
        print(f"  TESTING DATE: {date}")
        print(f"{'*'*70}")

        # ── 1. Price context ──────────────────────────────────────────────────
        print(f"\n{CYAN}[1/3] Loading price context...{RESET}")
        pctx = price_context(STOCK, date)
        if pctx["price_table"].empty:
            print(f"{RED}  ✗ Price data unavailable — check PRICES_DIR = '{PRICES_DIR}'{RESET}")
        else:
            print(f"{GREEN}  ✓ Price data loaded: {len(pctx['price_table'])} rows{RESET}")
            if pctx["indicators"]:
                print(f"{GREEN}  ✓ Indicators: {list(pctx['indicators'].keys())}{RESET}")
            else:
                print(f"{YELLOW}  ⚠ No indicator columns found{RESET}")
        for e in pctx["errors"]:
            print(f"  {YELLOW}⚠ {e}{RESET}")

        # ── 2. News context ───────────────────────────────────────────────────
        print(f"\n{CYAN}[2/3] Loading news context (look-ahead-safe)...{RESET}")
        news_ctx = _load_news_context(STOCK, date)
        n = len(news_ctx["items"])
        if news_ctx["errors"] and "look-ahead" in " ".join(news_ctx["errors"]).lower():
            print(f"{RED}  ✗ LOOK-AHEAD BIAS DETECTED — CRITICAL ERROR{RESET}")
        elif n > 0:
            print(f"{GREEN}  ✓ {n} news item(s) found before {date}{RESET}")
        else:
            print(f"{YELLOW}  ⚠ 0 news items found before {date} (may be valid){RESET}")
        for e in news_ctx["errors"]:
            print(f"  {YELLOW}⚠ {e}{RESET}")

        # ── 3. Event lookup ───────────────────────────────────────────────────
        print(f"\n{CYAN}[3/3] Looking up event calendar...{RESET}")
        ev = event_lookup(STOCK, date)
        if ev["is_holiday"]:
            print(f"{RED}  ⚠ NSE HOLIDAY on {date}{RESET}")
        if ev["same_day"]:
            print(f"{YELLOW}  ⚠ {len(ev['same_day'])} event(s) on this date{RESET}")
        if ev["upcoming"]:
            print(f"{YELLOW}  ℹ {len(ev['upcoming'])} upcoming event(s) in next 7 days{RESET}")
        for e in ev["errors"]:
            print(f"  {YELLOW}⚠ {e}{RESET}")
        if not ev["errors"] and not ev["same_day"] and not ev["is_holiday"]:
            print(f"{GREEN}  ✓ Event calendar checked — no conflicts{RESET}")

        # ── 4. Combine into briefing ──────────────────────────────────────────
        briefing = build_analyst_briefing(STOCK, COMPANY, date, pctx, news_ctx, ev)
        briefings[date] = briefing

        # ── 5. Validation checklist ───────────────────────────────────────────
        checks = run_validation_checklist(STOCK, date, pctx, news_ctx, ev)
        print_checklist(checks, STOCK, date)

    # ── Print final combined briefings ────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  FINAL ANALYST BRIEFINGS")
    print(f"{'#'*70}")
    for date, briefing in briefings.items():
        print(briefing)
        print("\n")

    print(f"\n{'#'*70}")
    print(f"  COMPLETE")
    print(f"  Next: Write data/context_builder.py")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
