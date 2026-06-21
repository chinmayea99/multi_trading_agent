"""
utils/cache.py — LLM Response Caching Layer

Day 32 of M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.

What it does:
    Intercepts every LLM API call and checks whether the exact same prompt has been
    sent before. If yes, returns the previously saved response from disk (cache HIT).
    If no, calls the real LLM, saves the response to disk, then returns it (cache MISS).

Why it matters:
    1. COST: Gemini and Groq have free-tier rate limits. Running 20 stocks × 500 trading
       days × 3 agents = 30,000 LLM calls. Without caching this is impossible on a free tier.
    2. SPEED: A disk read (~1 ms) vs an API call (~1–3 s). Second run is ~1000× faster.
    3. REPRODUCIBILITY: Thesis experiments must be exactly reproducible. With caching,
       the LLM "answers" are frozen on first run — re-running the backtest always gives
       identical results, which is a scientific requirement.
    4. DEBUGGING: You can inspect any cached prompt/response as a plain JSON file.

How it works:
    1. Compute MD5 hash of the full prompt string → 32-char hex string (cache key).
    2. Check if  cache/<key>.json  exists on disk.
    3. HIT  → load JSON, return the 'response' field, log cache hit.
    4. MISS → call the real LLM via llm_utils.call_llm(), save {prompt, response,
              model, timestamp} as JSON, log cache miss, return response.

Cache file format (one JSON file per unique prompt):
    {
        "cache_key":  "a3f8c1d2...",          # MD5 of prompt
        "model":      "gemini-2.5-flash",     # which LLM answered
        "timestamp":  "2025-05-21T14:23:11",  # when first called
        "prompt":     "SYSTEM INSTRUCTIONS...",
        "response":   "ANALYST REPORT — TCS..."
    }

Usage:
    from utils.cache import cached_llm_call
    response = cached_llm_call(prompt=full_prompt, llm=llms["gemini"], model_name="gemini")

Dependencies:
    • llm_utils.py  (Day 10) — call_llm()
    • config.py     (Day 1)  — CACHE_DIR
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import CACHE_DIR
except ImportError:
    CACHE_DIR = "cache"   # fallback if running standalone

CACHE_PATH = Path(CACHE_DIR)

# ── Logger ────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CACHE KEY
# ─────────────────────────────────────────────────────────────────────────────

def get_cache_key(prompt: str) -> str:
    """
    Compute a deterministic MD5 hash of the prompt string.

    MD5 produces a 32-character hexadecimal string. It is not
    cryptographically secure but is fast and collision-resistant enough
    for caching purposes where prompts are long (200–500 words).

    Args:
        prompt: The complete prompt string sent to the LLM.

    Returns:
        32-character lowercase hexadecimal string, e.g. "a3f8c1d2e5b7..."
    """
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD FROM CACHE
# ─────────────────────────────────────────────────────────────────────────────

def load_from_cache(cache_key: str) -> Optional[str]:
    """
    Attempt to load a previously cached LLM response from disk.

    Args:
        cache_key: MD5 hex string returned by get_cache_key().

    Returns:
        The cached response string if the file exists and is valid JSON,
        otherwise None (which signals a cache miss to the caller).
    """
    cache_file = CACHE_PATH / f"{cache_key}.json"

    if not cache_file.exists():
        log.debug("[cache] MISS  key=%s  (file not found)", cache_key)
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as fh:
            data: dict = json.load(fh)

        response = data.get("response", "")
        if not response:
            log.warning("[cache] CORRUPTED  key=%s  (empty response field)", cache_key)
            return None

        model     = data.get("model", "unknown")
        timestamp = data.get("timestamp", "unknown")
        log.info(
            "[cache] HIT   key=%s  model=%s  cached_at=%s",
            cache_key, model, timestamp,
        )
        return response

    except (json.JSONDecodeError, OSError) as exc:
        log.warning("[cache] READ ERROR  key=%s  error=%s", cache_key, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAVE TO CACHE
# ─────────────────────────────────────────────────────────────────────────────

def save_to_cache(
    cache_key: str,
    prompt: str,
    response: str,
    model_name: str = "unknown",
) -> bool:
    """
    Save an LLM response to disk as a JSON file.

    The file is written atomically (write to temp, then rename) to avoid
    corrupted files if the process is interrupted mid-write.

    Args:
        cache_key:  MD5 hex string (file will be  cache/<cache_key>.json).
        prompt:     The original full prompt — stored for human inspection.
        response:   The LLM's response string.
        model_name: Label for the model, e.g. "gemini", "groq", "ollama".

    Returns:
        True on success, False on any write error.
    """
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    cache_file  = CACHE_PATH / f"{cache_key}.json"
    temp_file   = CACHE_PATH / f"{cache_key}.tmp"

    payload = {
        "cache_key": cache_key,
        "model":      model_name,
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt":     prompt,
        "response":   response,
    }

    try:
        with open(temp_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        temp_file.replace(cache_file)   # atomic rename on all OSes
        log.info("[cache] SAVED  key=%s  model=%s", cache_key, model_name)
        return True

    except OSError as exc:
        log.error("[cache] WRITE ERROR  key=%s  error=%s", cache_key, exc)
        # Clean up partial temp file if it exists
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PUBLIC FUNCTION — cached_llm_call
# ─────────────────────────────────────────────────────────────────────────────

def cached_llm_call(
    prompt: str,
    llm: Any,
    model_name: str = "unknown",
) -> str:
    """
    Drop-in replacement for llm_utils.call_llm() with transparent caching.

    This is the ONLY function agents should call. It handles the full
    cache-check → LLM-call → cache-save lifecycle automatically.

    Workflow:
        1. Compute MD5 cache key from prompt.
        2. Check cache/ directory for a matching JSON file.
        3a. HIT  → return cached response immediately (no API call).
        3b. MISS → call the real LLM, cache the result, return response.

    Args:
        prompt:     Complete prompt string (system + user combined).
        llm:        LLM client object from llm_utils.init_llms() — one of:
                      • Groq client instance
                      • google_genai.Client instance
                      • str  (Ollama model name)
                    OR a callable with signature (prompt: str) -> str
                    (useful for testing / mocking).
        model_name: Human-readable model label for cache metadata.
                    Examples: "gemini", "groq", "ollama".

    Returns:
        LLM response as a plain string.

    Raises:
        RuntimeError: If the LLM call fails and no cached response exists.
    """
    cache_key = get_cache_key(prompt)

    # ── Step 1: Check cache ───────────────────────────────────────────────────
    cached = load_from_cache(cache_key)
    if cached is not None:
        return cached

    # ── Step 2: Cache miss — call the real LLM ────────────────────────────────
    log.info("[cache] MISS  key=%s  model=%s — calling LLM...", cache_key, model_name)

    try:
        # Support both: llm_utils.call_llm-style objects AND plain callables (for tests)
        if callable(llm) and not _is_llm_client(llm):
            response: str = llm(prompt)
        else:
            from llm_utils import call_llm  # imported lazily so cache.py works standalone
            response = call_llm(prompt, llm)

    except Exception as exc:
        raise RuntimeError(
            f"[cache] LLM call failed for key={cache_key}: {exc}"
        ) from exc

    # ── Step 3: Save to cache ─────────────────────────────────────────────────
    save_to_cache(cache_key, prompt, response, model_name)

    return response


def _is_llm_client(obj: Any) -> bool:
    """
    Return True if obj looks like a Groq/Gemini/Ollama client object
    (i.e. not a plain Python callable we should call directly).

    This heuristic allows test code to pass a plain lambda as the LLM.
    """
    type_name = type(obj).__name__
    return type_name in {"Groq", "Client"} or isinstance(obj, str)


# ─────────────────────────────────────────────────────────────────────────────
# 5. UTILITY — cache statistics
# ─────────────────────────────────────────────────────────────────────────────

def cache_stats() -> dict:
    """
    Return a summary dict of the current cache state.

    Useful for logging at the end of a backtest run.

    Returns:
        {
            "total_files":    int,
            "total_size_kb":  float,
            "oldest":         str | None,   # ISO timestamp of oldest entry
            "newest":         str | None,   # ISO timestamp of newest entry
        }
    """
    if not CACHE_PATH.exists():
        return {"total_files": 0, "total_size_kb": 0.0, "oldest": None, "newest": None}

    files    = list(CACHE_PATH.glob("*.json"))
    total_kb = sum(f.stat().st_size for f in files) / 1024
    timestamps: list[str] = []

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                ts = json.load(fh).get("timestamp")
            if ts:
                timestamps.append(ts)
        except Exception:
            pass

    timestamps.sort()
    return {
        "total_files":   len(files),
        "total_size_kb": round(total_kb, 2),
        "oldest":        timestamps[0]  if timestamps else None,
        "newest":        timestamps[-1] if timestamps else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. UTILITY — clear cache (use with care!)
# ─────────────────────────────────────────────────────────────────────────────

def clear_cache(confirm: bool = False) -> int:
    """
    Delete all cached JSON files.

    Args:
        confirm: Must be True to actually delete files (safety guard).

    Returns:
        Number of files deleted.
    """
    if not confirm:
        log.warning("[cache] clear_cache() called without confirm=True. Nothing deleted.")
        return 0

    deleted = 0
    for f in CACHE_PATH.glob("*.json"):
        f.unlink()
        deleted += 1
    log.info("[cache] Cleared %d cache files.", deleted)
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# 7. SMOKE TEST  (run: python utils/cache.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("utils/cache.py — Smoke Test")
    print("=" * 60)

    TEST_PROMPT = (
        "SYSTEM INSTRUCTIONS:\nYou are a senior equity analyst.\n\n"
        "USER REQUEST:\nAnalyse TCS.NS for 2024-03-15."
    )

    # Use a plain lambda as a mock LLM so the smoke test needs no API key
    mock_llm = lambda p: "ANALYST REPORT — TCS — 2024-03-15\nTrend: Bullish."

    print("\n[1] First call (expect MISS + save):")
    r1 = cached_llm_call(TEST_PROMPT, mock_llm, model_name="mock")
    print(f"    Response: {r1[:60]}...")

    print("\n[2] Second call with identical prompt (expect HIT):")
    r2 = cached_llm_call(TEST_PROMPT, mock_llm, model_name="mock")
    assert r1 == r2, "Cache hit should return identical response"
    print(f"    Response: {r2[:60]}...")

    print("\n[3] Different prompt (expect MISS):")
    r3 = cached_llm_call("Different prompt entirely.", mock_llm, model_name="mock")
    print(f"    Response: {r3[:60]}...")

    print("\n[4] Cache stats:")
    stats = cache_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n✓ Smoke test passed. Check the cache/ directory for JSON files.")
    print("=" * 60)
