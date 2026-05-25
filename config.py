"""
config.py — Central configuration for the Indian Multi-Agent Trading System.
Every other module imports from here. Do not hardcode values elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# STOCKS  (NSE tickers — suffix .NS for yfinance)
# ─────────────────────────────────────────────
STOCKS = [
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "SBIN.NS",
    "RELIANCE.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "BAJAJ-AUTO.NS",
    "M&M.NS",
]

# Friendly name lookup (used in prompts / logs)
STOCK_NAMES = {
    "TCS.NS":         "Tata Consultancy Services",
    "INFY.NS":        "Infosys",
    "WIPRO.NS":       "Wipro",
    "HCLTECH.NS":     "HCL Technologies",
    "HDFCBANK.NS":    "HDFC Bank",
    "ICICIBANK.NS":   "ICICI Bank",
    "KOTAKBANK.NS":   "Kotak Mahindra Bank",
    "SBIN.NS":        "State Bank of India",
    "RELIANCE.NS":    "Reliance Industries",
    "ONGC.NS":        "ONGC",
    "POWERGRID.NS":   "Power Grid Corporation",
    "NTPC.NS":        "NTPC",
    "HINDUNILVR.NS":  "Hindustan Unilever",
    "ITC.NS":         "ITC",
    "NESTLEIND.NS":   "Nestle India",
    "BRITANNIA.NS":   "Britannia Industries",
    "MARUTI.NS":      "Maruti Suzuki",
    "TATAMOTORS.NS":  "Tata Motors",
    "BAJAJ-AUTO.NS":  "Bajaj Auto",
    "M&M.NS":         "Mahindra & Mahindra",
}

# BSE security codes (for BSE filings scraper)
BSE_CODES = {
    "TCS.NS":         "532540",
    "INFY.NS":        "500209",
    "WIPRO.NS":       "507685",
    "HCLTECH.NS":     "532281",
    "HDFCBANK.NS":    "500180",
    "ICICIBANK.NS":   "532174",
    "KOTAKBANK.NS":   "500247",
    "SBIN.NS":        "500112",
    "RELIANCE.NS":    "500325",
    "ONGC.NS":        "500312",
    "POWERGRID.NS":   "532898",
    "NTPC.NS":        "532555",
    "HINDUNILVR.NS":  "500696",
    "ITC.NS":         "500875",
    "NESTLEIND.NS":   "500790",
    "BRITANNIA.NS":   "500825",
    "MARUTI.NS":      "532500",
    "TATAMOTORS.NS":  "500570",
    "BAJAJ-AUTO.NS":  "532977",
    "M&M.NS":         "500520",
}

# ─────────────────────────────────────────────
# DATE RANGE
# ─────────────────────────────────────────────
START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

# ─────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────
INITIAL_CAPITAL     = 1_000_000   # INR 10 lakh
RISK_FREE_RATE      = 0.065       # 6.5% — RBI repo rate proxy for Sharpe ratio
MAX_POSITION_PCT    = 0.20        # Max 20% of portfolio in any single stock
TRANSACTION_COST    = 0.001       # 0.1% per trade (brokerage + STT approximation)

# ─────────────────────────────────────────────
# API KEYS  (loaded from .env — never hardcode)
# ─────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY")       # Optional — newsapi.org free tier

# Fail fast if critical keys are missing
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Add it to your .env file.")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found. Add it to your .env file.")

# ─────────────────────────────────────────────
# LLM MODELS
# ─────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_MODEL = "mistral"

# LLM generation settings
LLM_TEMPERATURE = 0.3    # Low temperature for consistent, analytical outputs
LLM_MAX_TOKENS  = 1024

# ─────────────────────────────────────────────
# RATE LIMITS  (seconds between API calls)
# ─────────────────────────────────────────────
GROQ_DELAY   = 1.0   # Groq free tier: ~30 req/min
GEMINI_DELAY = 0.5   # Gemini free tier: ~60 req/min
OLLAMA_DELAY = 0.0   # Local — no limit

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
DATA_DIR          = "data"
PRICES_DIR        = f"{DATA_DIR}/prices"
NEWS_DIR          = f"{DATA_DIR}/news"
EVENTS_DIR        = f"{DATA_DIR}/events"
RAW_NEWS_DIR      = f"{DATA_DIR}/raw_news"

STOCKS_CSV        = f"{DATA_DIR}/stocks.csv"
EVENTS_CSV        = f"{EVENTS_DIR}/indian_events.csv"
HOLIDAYS_CSV      = f"{EVENTS_DIR}/market_holidays_2024.csv"
EARNINGS_CSV      = f"{EVENTS_DIR}/earnings_calendar_2024.csv"

RESULTS_DIR       = "results"
LOGS_DIR          = "logs"
CACHE_DIR         = ".cache"
EXPERIMENTS_DIR   = "experiments"

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
SMA_SHORT   = 20    # Short-term simple moving average window
SMA_LONG    = 50    # Long-term simple moving average window
RSI_PERIOD  = 14
PRICE_LOOKBACK_DAYS = 10   # Days of price history fed to Analyst Agent prompt

# ─────────────────────────────────────────────
# NEWS CONTEXT
# ─────────────────────────────────────────────
NEWS_MAX_ITEMS = 3   # Max news items per stock per day in agent prompt
BSE_REQUEST_DELAY = 2.0   # Seconds between BSE API calls (respect rate limits)

# ─────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────
EXPERIMENTS = {
    "buy_and_hold": {
        "pipeline":  "buy_and_hold",
        "model":      None,
        "description": "Buy equal weight all 20 stocks on Jan 1, hold until Dec 31",
    },
    "single_agent_gemini": {
        "pipeline":  "single_agent",
        "model":      GEMINI_MODEL,
        "llm":        "gemini",
        "description": "Trader Agent only — no Analyst or Researcher — using Gemini",
    },
    "three_agent_gemini": {
        "pipeline":  "three_agent",
        "model":      GEMINI_MODEL,
        "llm":        "gemini",
        "description": "Full Analyst → Researcher (bull/bear) → Trader using Gemini",
    },
    "three_agent_llama": {
        "pipeline":  "three_agent",
        "model":      GROQ_MODEL,
        "llm":        "groq",
        "description": "Full three-agent pipeline using Llama 3.1 70B via Groq",
    },
    "three_agent_mistral": {
        "pipeline":  "three_agent",
        "model":      OLLAMA_MODEL,
        "llm":        "ollama",
        "description": "Full three-agent pipeline using Mistral 7B via Ollama (local)",
    },
}
