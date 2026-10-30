"""
Configuration management for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Loads environment variables, defines default paths, actor IDs, and hyperparameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    # Asset Settings
    TARGET_ASSET: str = os.getenv("TARGET_ASSET", "TSLA")
    ASSET_NAME: str = os.getenv("ASSET_NAME", "Tesla, Inc.")
    LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "180"))
    INTERVAL: str = os.getenv("INTERVAL", "1d")  # 1d or 1h

    # Apify Settings
    APIFY_API_TOKEN: str = os.getenv("APIFY_API_TOKEN", "")
    APIFY_ACTOR_ID: str = os.getenv("APIFY_ACTOR_ID", "apidojo/tweet-scraper")
    APIFY_SEARCH_TERMS: List[str] = field(
        default_factory=lambda: ["$TSLA", "TSLA", "Tesla", "Elon Musk"]
    )
    APIFY_MAX_ITEMS: int = int(os.getenv("APIFY_MAX_ITEMS", "3000"))

    # Tavily / News Settings
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
    TAVILY_SEARCH_DAYS: int = int(os.getenv("TAVILY_SEARCH_DAYS", "30"))

    # Database
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = BASE_DIR / "data" / "crowdwisdom_trading.db"

    # Outputs
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    CHARTS_DIR: Path = BASE_DIR / "outputs" / "charts"
    REPORTS_DIR: Path = BASE_DIR / "outputs" / "reports"

    # Forecasting Horizons (as required by specification: n+0, n+1, n+2, n+3, n+4, n+5)
    FORECAST_HORIZONS: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    # Walk-Forward Validation Parameters
    WFV_TRAIN_WINDOW: int = int(os.getenv("WFV_TRAIN_WINDOW", "60"))  # initial train bars
    WFV_TEST_WINDOW: int = int(os.getenv("WFV_TEST_WINDOW", "15"))   # out-of-sample step
    PURGE_WINDOW: int = int(os.getenv("PURGE_WINDOW", "5"))          # purge overlapping bars (n+5)

    # Trading Parameters
    INITIAL_CAPITAL: float = 100_000.0
    TRANSACTION_COST_BPS: float = 10.0  # 10 bps (0.10%) per trade
    RISK_FREE_RATE: float = 0.045       # 4.5% annual risk-free rate (SOFR/Treasury proxy)

    def __post_init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


config = Settings()
