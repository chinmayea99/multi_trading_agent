"""
tests/test_cache.py — Unit Tests for utils/cache.py

Day 32 of M.Tech Project: Multi-Agent LLM-Based Trading System for the Indian Stock Market.

Run:
    python -m pytest tests/test_cache.py -v

All tests use a temporary directory so they never touch the real cache/ folder.
No API keys required — a plain Python lambda acts as the mock LLM.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import pytest

# ── Make sure project root is on sys.path ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Override CACHE_DIR before importing cache module ─────────────────────────
import importlib

# We monkey-patch the module after import using the tmp_path fixture
# (see conftest / fixture below).


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def cache_module(tmp_path, monkeypatch):
    """
    Import utils.cache with CACHE_PATH redirected to a temporary directory.
    This ensures tests are fully isolated and never touch the real cache/.
    """
    import utils.cache as cache
    monkeypatch.setattr(cache, "CACHE_PATH", tmp_path)
    # Also reset any module-level state if needed
    return cache


@pytest.fixture()
def mock_llm():
    """A plain callable that mimics an LLM without any API call."""
    call_count = {"n": 0}

    def _llm(prompt: str) -> str:
        call_count["n"] += 1
        return f"MOCK RESPONSE for prompt hash: {len(prompt)}"

    _llm.call_count = call_count
    return _llm


# ─────────────────────────────────────────────────────────────────────────────
# TEST: get_cache_key
# ─────────────────────────────────────────────────────────────────────────────

class TestGetCacheKey:
    def test_returns_32_char_hex(self, cache_module):
        key = cache_module.get_cache_key("hello world")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic(self, cache_module):
        """Same prompt must always produce same key."""
        p = "Analyse TCS.NS for 2024-03-15."
        assert cache_module.get_cache_key(p) == cache_module.get_cache_key(p)

    def test_different_prompts_different_keys(self, cache_module):
        k1 = cache_module.get_cache_key("Analyse TCS")
        k2 = cache_module.get_cache_key("Analyse INFY")
        assert k1 != k2

    def test_empty_prompt(self, cache_module):
        key = cache_module.get_cache_key("")
        assert len(key) == 32


# ─────────────────────────────────────────────────────────────────────────────
# TEST: load_from_cache (miss scenarios)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFromCacheMiss:
    def test_returns_none_when_file_absent(self, cache_module):
        result = cache_module.load_from_cache("nonexistentkey1234567890abcdef12")
        assert result is None

    def test_returns_none_for_corrupted_json(self, cache_module, tmp_path):
        bad_file = tmp_path / "deadbeefdeadbeefdeadbeefdeadbeef.json"
        bad_file.write_text("NOT VALID JSON {{{{", encoding="utf-8")
        result = cache_module.load_from_cache("deadbeefdeadbeefdeadbeefdeadbeef")
        assert result is None

    def test_returns_none_for_empty_response_field(self, cache_module, tmp_path):
        key = "a" * 32
        f = tmp_path / f"{key}.json"
        f.write_text(json.dumps({"cache_key": key, "response": ""}), encoding="utf-8")
        result = cache_module.load_from_cache(key)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TEST: save_to_cache
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveToCache:
    def test_creates_json_file(self, cache_module, tmp_path):
        key = cache_module.get_cache_key("test prompt")
        ok  = cache_module.save_to_cache(key, "test prompt", "test response", "gemini")
        assert ok is True
        assert (tmp_path / f"{key}.json").exists()

    def test_saved_file_is_valid_json(self, cache_module, tmp_path):
        key = cache_module.get_cache_key("another prompt")
        cache_module.save_to_cache(key, "another prompt", "hello response", "groq")
        data = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
        assert data["response"] == "hello response"
        assert data["model"]    == "groq"
        assert "timestamp"      in data

    def test_overwrites_existing_file(self, cache_module, tmp_path):
        key = cache_module.get_cache_key("overwrite me")
        cache_module.save_to_cache(key, "overwrite me", "first",  "gemini")
        cache_module.save_to_cache(key, "overwrite me", "second", "gemini")
        data = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
        assert data["response"] == "second"

    def test_creates_cache_dir_if_missing(self, cache_module, tmp_path, monkeypatch):
        deep_path = tmp_path / "a" / "b" / "c"
        monkeypatch.setattr(cache_module, "CACHE_PATH", deep_path)
        key = cache_module.get_cache_key("nested dir test")
        ok  = cache_module.save_to_cache(key, "nested dir test", "resp", "ollama")
        assert ok is True
        assert (deep_path / f"{key}.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# TEST: cached_llm_call — CACHE HIT
# ─────────────────────────────────────────────────────────────────────────────

class TestCachedLlmCallHit:
    def test_second_call_does_not_invoke_llm(self, cache_module, mock_llm):
        prompt = "Hit test prompt — TCS 2024-03-15"
        # First call — cache MISS, LLM is called
        r1 = cache_module.cached_llm_call(prompt, mock_llm, "mock")
        assert mock_llm.call_count["n"] == 1

        # Second call — cache HIT, LLM should NOT be called again
        r2 = cache_module.cached_llm_call(prompt, mock_llm, "mock")
        assert mock_llm.call_count["n"] == 1   # still 1, not 2

    def test_hit_returns_same_response(self, cache_module, mock_llm):
        prompt = "Reproducibility test — INFY 2024-06-03"
        r1 = cache_module.cached_llm_call(prompt, mock_llm, "mock")
        r2 = cache_module.cached_llm_call(prompt, mock_llm, "mock")
        assert r1 == r2

    def test_hit_works_across_different_model_names(self, cache_module, tmp_path):
        """Cache is keyed on prompt only, not on model_name."""
        prompt = "Cross-model cache test"
        llm_a  = lambda p: "Response from model A"
        llm_b  = lambda p: "Response from model B"

        r1 = cache_module.cached_llm_call(prompt, llm_a, "gemini")
        r2 = cache_module.cached_llm_call(prompt, llm_b, "groq")   # should hit cache
        assert r1 == r2   # same response because same prompt


# ─────────────────────────────────────────────────────────────────────────────
# TEST: cached_llm_call — CACHE MISS
# ─────────────────────────────────────────────────────────────────────────────

class TestCachedLlmCallMiss:
    def test_first_call_invokes_llm(self, cache_module, mock_llm):
        prompt = "Miss test — RELIANCE 2024-04-10"
        cache_module.cached_llm_call(prompt, mock_llm, "mock")
        assert mock_llm.call_count["n"] == 1

    def test_different_prompts_each_invoke_llm(self, cache_module, mock_llm):
        cache_module.cached_llm_call("Prompt Alpha", mock_llm, "mock")
        cache_module.cached_llm_call("Prompt Beta",  mock_llm, "mock")
        assert mock_llm.call_count["n"] == 2

    def test_response_is_saved_after_miss(self, cache_module, mock_llm, tmp_path):
        prompt = "Save after miss — HDFCBANK 2024-06-03"
        key    = cache_module.get_cache_key(prompt)
        cache_module.cached_llm_call(prompt, mock_llm, "mock")
        assert (tmp_path / f"{key}.json").exists()

    def test_minor_prompt_difference_causes_miss(self, cache_module, mock_llm):
        """Even a single extra space makes it a different prompt → separate cache entry."""
        cache_module.cached_llm_call("Analyse TCS",  mock_llm, "mock")
        cache_module.cached_llm_call("Analyse TCS ", mock_llm, "mock")  # trailing space
        assert mock_llm.call_count["n"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# TEST: cache_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheStats:
    def test_empty_cache_returns_zeros(self, cache_module, tmp_path):
        stats = cache_module.cache_stats()
        assert stats["total_files"] == 0

    def test_stats_count_increases(self, cache_module, mock_llm):
        cache_module.cached_llm_call("Stats test 1", mock_llm, "mock")
        cache_module.cached_llm_call("Stats test 2", mock_llm, "mock")
        stats = cache_module.cache_stats()
        assert stats["total_files"] == 2
        assert stats["total_size_kb"] > 0

    def test_stats_has_timestamps(self, cache_module, mock_llm):
        cache_module.cached_llm_call("Timestamp test", mock_llm, "mock")
        stats = cache_module.cache_stats()
        assert stats["oldest"] is not None
        assert stats["newest"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# TEST: clear_cache
# ─────────────────────────────────────────────────────────────────────────────

class TestClearCache:
    def test_no_delete_without_confirm(self, cache_module, mock_llm):
        cache_module.cached_llm_call("Clear test", mock_llm, "mock")
        deleted = cache_module.clear_cache(confirm=False)
        assert deleted == 0
        assert cache_module.cache_stats()["total_files"] == 1

    def test_clears_all_files_with_confirm(self, cache_module, mock_llm):
        cache_module.cached_llm_call("Clear A", mock_llm, "mock")
        cache_module.cached_llm_call("Clear B", mock_llm, "mock")
        deleted = cache_module.clear_cache(confirm=True)
        assert deleted == 2
        assert cache_module.cache_stats()["total_files"] == 0
