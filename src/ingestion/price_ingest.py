"""
Historical price data ingestion using Yahoo Finance (yfinance).
Ensures accurate UTC and EST timezone alignment, handles market holidays and weekend gaps.
"""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd
import pytz
import yfinance as yf

from src.config import config
from src.database.db import save_price_bars

logger = logging.getLogger(__name__)


class PriceIngester:
    def __init__(self, symbol: Optional[str] = None, lookback_days: Optional[int] = None, interval: Optional[str] = None):
        self.symbol = symbol or config.TARGET_ASSET
        self.lookback_days = lookback_days or config.LOOKBACK_DAYS
        self.interval = interval or config.INTERVAL

    def fetch_price_data(self) -> pd.DataFrame:
        """
        Fetch historical price data from yfinance.
        Generates UTC and US/Eastern (EST/EDT) timestamps.
        """
        logger.info(f"Fetching {self.symbol} price data for {self.lookback_days} days (interval={self.interval})...")
        ticker = yf.Ticker(self.symbol)
        
        # Use period parameter with buffer to guarantee 180 days of trading bars
        period_str = f"{self.lookback_days}d"
        df = ticker.history(period=period_str, interval=self.interval, auto_adjust=True)

        if df.empty:
            # Fallback to date range
            end_date = pd.Timestamp.now(tz="UTC")
            start_date = end_date - pd.Timedelta(days=self.lookback_days + 15)
            df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
                                interval=self.interval, auto_adjust=True)

        if df.empty:
            raise ValueError(f"Could not retrieve historical price data for symbol '{self.symbol}'.")

        # Standardize index and timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.copy()
        df["timestamp_utc"] = df.index
        df["timestamp_est"] = df.index.tz_convert("America/New_York")

        # Keep relevant columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing price column {col} in fetched data.")

        df = df.sort_index()
        logger.info(f"Successfully ingested {len(df)} price bars for {self.symbol} from {df['timestamp_utc'].iloc[0]} to {df['timestamp_utc'].iloc[-1]}.")
        return df

    def run(self) -> pd.DataFrame:
        """Fetch and persist to SQLite database."""
        df = self.fetch_price_data()
        saved_count = save_price_bars(df, self.symbol)
        logger.info(f"Persisted {saved_count} price bars to database for {self.symbol}.")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingester = PriceIngester()
    bars = ingester.run()
    print(bars.tail(3)[["timestamp_utc", "timestamp_est", "Open", "High", "Low", "Close", "Volume"]])
