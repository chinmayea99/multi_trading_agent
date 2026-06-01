"""
data/merge_news.py — Day 19: Merge all news sources into one unified CSV per stock.

ACTUAL FILE LAYOUT (confirmed from your local machine):
  BSE filings (clean parsed CSV):
      data/news/<TICKER_SAFE>_bse.csv
      e.g.  data/news/BAJAJ-AUTO_NS_bse.csv

  NewsAPI CSVs (one file per fetch month, may not exist if free-tier window missed):
      data/news/<TICKER_SAFE>_newsapi_<YYYY-MM>.csv
      e.g.  data/news/BAJAJ-AUTO_NS_newsapi_2024-12.csv

  Raw JSON files (bse_scraper / newsapi_fetch internal cache — NOT read here):
      data/raw_news/bse/...
      data/raw_news/newsapi/...

OUTPUT:
  data/news/<TICKER_SAFE>_news.csv
  Columns: date, headline, source, summary
  Sorted ascending by date, deduplicated on (date + headline).

USAGE:
  cd C:\\Users\\Chinmaye A\\multi_trading_agent
  python data/merge_news.py

DESIGN:
  - BSE file is mandatory.  Missing BSE → stock is skipped with an error.
  - NewsAPI files are optional.  Missing → a warning is logged and BSE-only
    output is still produced.  This is expected when the free-tier 30-day
    window didn't overlap your needed historical range.
  - Multiple NewsAPI month files (e.g. 2024-12 AND 2026-05) are all loaded
    and combined automatically.
"""

import os
import sys
import glob
import logging
import pandas as pd

# ── resolve project root so this script can be run from anywhere ─────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   # one level up from data/
sys.path.insert(0, PROJECT_ROOT)

from config import STOCKS, NEWS_DIR, RAW_NEWS_DIR

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── canonical output schema ───────────────────────────────────────────────────
OUTPUT_COLUMNS = ["date", "headline", "source", "summary"]


# ── filename helper ───────────────────────────────────────────────────────────

def ticker_to_safe(ticker: str) -> str:
    """
    Convert NSE ticker to the safe filename prefix used by bse_scraper.py
    and newsapi_fetch.py.

    Examples:
      "TCS.NS"       → "TCS_NS"
      "BAJAJ-AUTO.NS"→ "BAJAJ-AUTO_NS"
      "M&M.NS"       → "MAND&M_NS"  ← matches bse_scraper's replace("&","AND")
    """
    return ticker.replace(".", "_").replace("&", "AND")


# ── loaders ───────────────────────────────────────────────────────────────────

def _load_bse(ticker: str) -> pd.DataFrame | None:
    """
    Load BSE filings CSV from data/news/.
    Returns None if the file does not exist (caller handles error).
    """
    safe     = ticker_to_safe(ticker)
    bse_path = os.path.join(NEWS_DIR, f"{safe}_bse.csv")

    if not os.path.exists(bse_path):
        log.error(
            "[%s] BSE file NOT FOUND: %s\n"
            "       → Run data/bse_scraper.py --full first.",
            ticker, bse_path,
        )
        return None

    df = pd.read_csv(bse_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # BSE scraper saves: date, headline, category, summary, ticker, source
    # Normalise to the four output columns we need
    if "headline" not in df.columns:
        log.error("[%s] BSE file missing 'headline' column. Columns: %s", ticker, list(df.columns))
        return None

    # Fill missing columns with sensible defaults
    if "source"  not in df.columns: df["source"]  = "BSE"
    if "summary" not in df.columns: df["summary"] = ""

    df["source"] = "BSE"   # force canonical label regardless of stored value
    df = _normalise(df, "BSE")
    log.info("[%s] BSE: %d rows loaded from %s", ticker, len(df), bse_path)
    return df


def _load_newsapi(ticker: str) -> pd.DataFrame | None:
    """
    Load all NewsAPI month CSVs for this ticker from data/news/.
    File glob: data/news/<SAFE>_newsapi_*.csv
    Returns combined DataFrame, or None if no files found.
    """
    safe    = ticker_to_safe(ticker)
    pattern = os.path.join(NEWS_DIR, f"{safe}_newsapi_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        log.info(
            "[%s] No NewsAPI files found (pattern: %s). "
            "This is expected if the free-tier 30-day window didn't cover 2024–2025. "
            "BSE-only output will be produced.",
            ticker, pattern,
        )
        return None

    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            df.columns = [c.strip().lower() for c in df.columns]
            df = _normalise(df, "NewsAPI")
            log.info("[%s] NewsAPI: %d rows from %s", ticker, len(df), os.path.basename(fpath))
            frames.append(df)
        except Exception as exc:
            log.warning("[%s] Could not read %s: %s", ticker, fpath, exc)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    log.info("[%s] NewsAPI total: %d rows across %d file(s)", ticker, len(combined), len(frames))
    return combined


def _normalise(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Coerce a raw DataFrame to the canonical four-column schema.
    Drops rows with unparseable dates or empty headlines.
    """
    # Ensure required columns exist
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "date" else pd.NaT

    df = df[OUTPUT_COLUMNS].copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad = df["date"].isna().sum()
    if n_bad:
        log.warning("  %s: dropped %d rows with bad dates", source_label, n_bad)
    df = df.dropna(subset=["date"])

    # Clean strings
    df["headline"] = df["headline"].fillna("").astype(str).str.strip()
    df["summary"]  = df["summary"].fillna("").astype(str).str.strip()
    df["source"]   = source_label

    # Drop rows with empty headlines
    df = df[df["headline"].str.len() > 0]
    return df.reset_index(drop=True)


# ── deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where (date, lowercase headline) is identical across sources.
    BSE rows are added first in the concat, so BSE wins on ties.
    """
    df["_key"] = (
        df["date"].dt.normalize().astype(str)
        + "|"
        + df["headline"].str.lower().str.strip()
    )
    before = len(df)
    df = df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    removed = before - len(df)
    if removed:
        log.info("  Deduplication removed %d duplicate rows", removed)
    return df.reset_index(drop=True)


# ── per-stock merge ───────────────────────────────────────────────────────────

def merge_stock(ticker: str) -> bool:
    """Merge all sources for one ticker. Returns True on success."""
    log.info("─── %s ─────────────────────────────────", ticker)

    frames = []

    # 1. BSE (mandatory)
    bse_df = _load_bse(ticker)
    if bse_df is None:
        return False
    frames.append(bse_df)

    # 2. NewsAPI (optional)
    newsapi_df = _load_newsapi(ticker)
    if newsapi_df is not None:
        frames.append(newsapi_df)

    # 3. Merge, dedup, sort
    merged = pd.concat(frames, ignore_index=True)
    merged = _deduplicate(merged)
    merged = merged.sort_values("date", ascending=True).reset_index(drop=True)

    # 4. Save
    os.makedirs(NEWS_DIR, exist_ok=True)
    safe     = ticker_to_safe(ticker)
    out_path = os.path.join(NEWS_DIR, f"{safe}_news.csv")
    merged.to_csv(out_path, index=False)

    log.info("[%s] ✓ Saved %d rows → %s", ticker, len(merged), out_path)
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Day 19 — merge_news.py")
    log.info("News dir   : %s", NEWS_DIR)
    log.info("Stocks     : %d", len(STOCKS))
    log.info("=" * 60)

    ok, failed = [], []
    for ticker in STOCKS:
        success = merge_stock(ticker)
        (ok if success else failed).append(ticker)
        print()

    print("=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    for t in ok:
        print(f"  ✓  {t}")
    for t in failed:
        print(f"  ✗  {t}  ← BSE file missing — run bse_scraper.py first")
    print()
    print(f"  Succeeded : {len(ok)}")
    print(f"  Failed    : {len(failed)}")
    print("=" * 60)

    if failed:
        print("\n⚠  Fix failed stocks then re-run before moving to Day 20.")
        sys.exit(1)
    else:
        print("\n✓ All done. Proceed to Day 20 — get_news_context.py")


if __name__ == "__main__":
    main()
