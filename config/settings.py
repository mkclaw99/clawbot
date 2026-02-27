"""Non-safety runtime settings — these CAN be changed without resealing."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Capital ---
STARTING_CAPITAL          = float(os.getenv("STARTING_CAPITAL", "100000.0"))
HUMAN_APPROVAL_THRESHOLD  = float(os.getenv("HUMAN_APPROVAL_THRESHOLD", "5000.0"))

# --- Live trading gate ---
LIVE_TRADING_ENABLED  = os.getenv("LIVE_TRADING_ENABLED", "false").lower() == "true"
APPROVAL_TOKEN_HASH   = os.getenv("APPROVAL_TOKEN_HASH", "")

# --- Database ---
DATABASE_URL  = os.getenv("DATABASE_URL",  f"sqlite:///{BASE_DIR}/clawbot.db")
MARKET_DB_URL = os.getenv("MARKET_DB_URL", f"sqlite:///{BASE_DIR}/market.db")

# --- Reddit ---
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "clawbot/1.0")

# --- NewsAPI ---
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# --- Tavily (web search for optimizer + research) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Local LLM (self-improvement analysis) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "qwen3.5-397b-a17b")

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Strategy scheduler intervals (seconds) ---
STRATEGY_RUN_INTERVAL   = 60       # run strategies every 60s during market hours
CRAWLER_RUN_INTERVAL    = 300      # refresh web signals every 5 minutes
PORTFOLIO_SYNC_INTERVAL = 30       # sync portfolio state every 30s

# --- Self-improvement optimizer ---
OPTIMIZER_RUN_INTERVAL  = 21600    # run optimizer every 6 hours (after-hours only)
MIN_TRADES_FOR_TUNING   = 10       # minimum trades before a strategy can be tuned
OPTIMIZER_COOLING_HOURS = 24       # minimum hours between changes to the same param
OPTIMIZER_MAX_STEP_PCT  = 0.10     # max parameter change per cycle (10%)
WEB_SEARCH_MAX_REQUESTS = 10       # max DuckDuckGo requests per optimizer run
