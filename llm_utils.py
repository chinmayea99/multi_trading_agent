"""
llm_utils.py — LLM connection wrappers for the Indian Multi-Agent Trading System.

SDKs used:
  - Groq      : groq (pip install groq)
  - Gemini    : google-genai (pip install google-genai)   ← new SDK, NOT google-generativeai
  - Ollama    : ollama (pip install ollama)

Models:
  - Groq   → llama-3.3-70b-versatile   (replaced decommissioned 3.1)
  - Gemini → gemini-2.5-flash           (replaced deprecated 1.5-flash)
  - Ollama → phi3:mini                  (fits in ~2.4 GB RAM; pull with: ollama pull phi3:mini)
             OR mistral:7b-instruct-q4_0 if you have 5+ GB free RAM

Usage:
    from llm_utils import init_llms, call_llm
    llms = init_llms()
    response = call_llm("Analyse this stock.", llms["gemini"])
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── SDK imports ───────────────────────────────────────────────────────────────
from groq import Groq
from google import genai as google_genai          # NEW SDK: pip install google-genai
import ollama as ollama_client

# ── Config (read from .env / environment) ────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_MODEL   = "llama-3.3-70b-versatile"         # updated from 3.1 (decommissioned)
GEMINI_MODEL = "gemini-2.5-flash"                 # updated from 1.5-flash (deprecated)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")  # override in .env if needed

LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS  = 1024


# ── Initialisation ────────────────────────────────────────────────────────────

def init_llms() -> dict:
    """
    Initialise all three LLM clients and return them in a dict.

    Returns:
        {
            "groq":   Groq client instance,
            "gemini": google_genai.Client instance,
            "ollama": model name string (ollama is stateless),
        }
    """
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY missing from .env")
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY missing from .env")

    groq_client   = Groq(api_key=GROQ_API_KEY)
    gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
    ollama_name   = OLLAMA_MODEL

    print("✓ All LLM clients initialised.")
    print(f"  Groq   → {GROQ_MODEL}")
    print(f"  Gemini → {GEMINI_MODEL}")
    print(f"  Ollama → {ollama_name}")

    return {
        "groq":   groq_client,
        "gemini": gemini_client,
        "ollama": ollama_name,
    }


# ── Unified call interface ────────────────────────────────────────────────────

def call_llm(prompt: str, llm, retries: int = 3, retry_delay: float = 5.0) -> str:
    """
    Send a prompt to any of the three LLMs and return the response as a string.

    Args:
        prompt:      The full prompt string.
        llm:         Groq client, google_genai.Client, or str (ollama model name).
        retries:     Retry attempts on transient errors.
        retry_delay: Seconds between retries.

    Returns:
        Response text as a plain string.
    """
    for attempt in range(1, retries + 1):
        try:
            # ── Groq (Llama 3.3 70B) ─────────────────────────────────────────
            if isinstance(llm, Groq):
                response = llm.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                )
                return response.choices[0].message.content.strip()

            # ── Gemini 2.5 Flash (new google-genai SDK) ───────────────────────
            elif isinstance(llm, google_genai.Client):
                response = llm.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=google_genai.types.GenerateContentConfig(
                        temperature=LLM_TEMPERATURE,
                        max_output_tokens=LLM_MAX_TOKENS,
                    ),
                )
                return response.text.strip()

            # ── Ollama (local model) ──────────────────────────────────────────
            elif isinstance(llm, str):
                response = ollama_client.chat(
                    model=llm,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS,
                    },
                )
                return response["message"]["content"].strip()

            else:
                raise ValueError(
                    f"Unrecognised llm type: {type(llm)}. "
                    "Pass a Groq client, google_genai.Client, or ollama model name string."
                )

        except Exception as e:
            if attempt == retries:
                raise RuntimeError(
                    f"call_llm failed after {retries} attempts. Last error: {e}"
                ) from e
            print(f"  [llm_utils] Attempt {attempt} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test(llms: dict) -> None:
    """Send a hello-world prompt to each LLM and print results."""
    TEST_PROMPT = (
        "You are a financial analyst assistant. "
        "Reply with exactly one sentence confirming you are ready to analyse Indian stocks."
    )

    results = {}
    for name, llm in llms.items():
        print(f"\nTesting {name.upper()}...")
        try:
            start = time.time()
            reply = call_llm(TEST_PROMPT, llm)
            elapsed = time.time() - start
            print(f"  ✓ Response ({elapsed:.1f}s): {reply}")
            results[name] = "OK"
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results[name] = f"FAILED: {e}"

    print("\n── Smoke Test Summary ──────────────────────")
    for name, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {name.ljust(8)} → {status}")

    failed = [k for k, v in results.items() if v != "OK"]
    if failed:
        print(f"\n⚠  Fix these before proceeding: {', '.join(failed)}")
    else:
        print("\n✓ All LLMs responding. Ready for Day 10.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llms = init_llms()
    smoke_test(llms)
