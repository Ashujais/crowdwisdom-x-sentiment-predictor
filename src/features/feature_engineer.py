"""
Quantitative Feature Engineering Engine.
Strictly creates time-based features from sentiment and price only.
Engineers technical momentum, volatility, engagement-weighted sentiment metrics,
and multi-horizon target labels (n+0 through n+5).
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.config import config
from src.database.db import load_price_bars, load_tweets, load_news

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, symbol: Optional[str] = None):
        self.symbol = symbol or config.TARGET_ASSET
        self.horizons = config.FORECAST_HORIZONS  # [0, 1, 2, 3, 4, 5]

    def build_dataset(self) -> pd.DataFrame:
        """
        Loads price, tweet, and news tables, aligns them strictly chronologically
        without lookahead, engineers all features, and constructs target horizons.
        """
        logger.info(f"Loading data for {self.symbol} to construct feature dataset...")
        price_df = load_price_bars(self.symbol)
        tweets_df = load_tweets(self.symbol)
        news_df = load_news(self.symbol)

        if price_df.empty:
            raise ValueError(f"No price bars found in database for symbol {self.symbol}.")
        if tweets_df.empty:
            raise ValueError(f"No tweets found in database for symbol {self.symbol}.")

        # Ensure datetime indexing
        price_df = price_df.sort_values("timestamp_utc").reset_index(drop=True)
        tweets_df = tweets_df.sort_values("timestamp_utc").reset_index(drop=True)

        # ---------------------------------------------------------
        # 1. Price Feature Engineering (Price Only)
        # ---------------------------------------------------------
        df = price_df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        vol = df["volume"]

        # Log returns
        df["ret_1d"] = np.log(close / close.shift(1))
        df["ret_lag1"] = df["ret_1d"].shift(1)
        df["ret_lag2"] = df["ret_1d"].shift(2)
        df["ret_lag3"] = df["ret_1d"].shift(3)
        df["ret_lag5"] = df["ret_1d"].shift(5)

        # Price ratios
        df["hl_ratio"] = (high - low) / (close + 1e-6)
        df["co_ratio"] = (close - open_) / (open_ + 1e-6)

        # Realized Volatility
        df["volatility_5d"] = df["ret_1d"].rolling(5).std() * np.sqrt(252)
        df["volatility_10d"] = df["ret_1d"].rolling(10).std() * np.sqrt(252)
        df["volatility_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)

        # Moving Averages
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        df["close_to_sma5"] = (close / sma_5) - 1.0
        df["close_to_sma20"] = (close / sma_20) - 1.0

        # Technical Indicators: RSI(14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Technical Indicators: MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Volume Z-Score
        vol_mean_20 = vol.rolling(20).mean()
        vol_std_20 = vol.rolling(20).std() + 1e-6
        df["volume_zscore"] = (vol - vol_mean_20) / vol_std_20

        # ---------------------------------------------------------
        # 2. Sentiment Aggregation & Feature Engineering (Sentiment Only)
        # ---------------------------------------------------------
        # For each price bar, aggregate tweets that occurred prior to or during that bar
        # To prevent lookahead: bar t spans (timestamp_{t-1}, timestamp_t]
        sentiment_records = []
        news_sentiment_list = []

        timestamps = df["timestamp_utc"].tolist()
        for i in range(len(timestamps)):
            curr_time = timestamps[i]
            prev_time = timestamps[i - 1] if i > 0 else (curr_time - pd.Timedelta(days=1))

            # Filter tweets strictly within [prev_time, curr_time]
            mask_tweets = (tweets_df["timestamp_utc"] > prev_time) & (tweets_df["timestamp_utc"] <= curr_time)
            sub_tweets = tweets_df[mask_tweets]

            if not sub_tweets.empty:
                s_scores = sub_tweets["sentiment_score"].values
                weights = sub_tweets["engagement_weight"].values
                total_w = float(np.sum(weights))
                if total_w > 0:
                    weighted_sent = float(np.sum(s_scores * weights) / total_w)
                else:
                    weighted_sent = float(np.mean(s_scores))

                mean_sent = float(np.mean(s_scores))
                std_sent = float(np.std(s_scores)) if len(s_scores) > 1 else 0.0
                post_vol = len(sub_tweets)
                bulls = int(np.sum(s_scores > 0.05))
                bears = int(np.sum(s_scores < -0.05))
                bb_ratio = float((bulls + 1.0) / (bears + 1.0))
                sub_subj = float(np.mean(sub_tweets["sentiment_subjectivity"]))
            else:
                weighted_sent = 0.0
                mean_sent = 0.0
                std_sent = 0.0
                post_vol = 0
                bulls = 0
                bears = 0
                bb_ratio = 1.0
                sub_subj = 0.0

            sentiment_records.append({
                "sentiment_mean": mean_sent,
                "sentiment_weighted": weighted_sent,
                "sentiment_std": std_sent,
                "sentiment_subjectivity": sub_subj,
                "post_volume": post_vol,
                "bull_count": bulls,
                "bear_count": bears,
                "bull_bear_ratio": bb_ratio
            })

            # News sentiment up to curr_time
            if not news_df.empty:
                mask_news = (news_df["published_at_utc"] <= curr_time) & (news_df["published_at_utc"] > (curr_time - pd.Timedelta(days=7)))
                sub_news = news_df[mask_news]
                if not sub_news.empty:
                    news_sent = float(sub_news["sentiment_score"].mean())
                else:
                    news_sent = 0.0
            else:
                news_sent = 0.0
            news_sentiment_list.append(news_sent)

        sent_df = pd.DataFrame(sentiment_records)
        for col in sent_df.columns:
            df[col] = sent_df[col]
        df["news_sentiment"] = news_sentiment_list

        # Rolling sentiment features & momentum
        df["post_volume_zscore"] = (df["post_volume"] - df["post_volume"].rolling(14).mean()) / (df["post_volume"].rolling(14).std() + 1e-6)
        df["sentiment_momentum_1d"] = df["sentiment_weighted"] - df["sentiment_weighted"].shift(1)
        df["sentiment_momentum_3d"] = df["sentiment_weighted"] - df["sentiment_weighted"].shift(3)
        df["sentiment_price_divergence"] = np.sign(df["sentiment_momentum_1d"]) * np.sign(df["ret_1d"])

        # ---------------------------------------------------------
        # 3. Time Horizon Features (Derived from timestamps only)
        # ---------------------------------------------------------
        dates = pd.to_datetime(df["timestamp_utc"])
        day_of_week = dates.dt.dayofweek
        month = dates.dt.month

        df["sin_day_of_week"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df["cos_day_of_week"] = np.cos(2 * np.pi * day_of_week / 7.0)
        df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
        df["is_monday"] = (day_of_week == 0).astype(int)
        df["is_friday"] = (day_of_week == 4).astype(int)

        # ---------------------------------------------------------
        # 4. Multi-Horizon Targets: n+0, n+1, n+2, n+3, n+4, n+5
        # ---------------------------------------------------------
        # n is the current period.
        # Target Price at n+k:
        # n+0: close price of current period n
        # n+k (k >= 1): close price k periods into the future
        for h in self.horizons:
            df[f"target_price_n{h}"] = df["close"].shift(-h)
            if h == 0:
                # n+0 return is contemporaneous period return
                df[f"target_ret_n{h}"] = df["ret_1d"]
            else:
                # n+k return is forward return from period n to period n+k
                df[f"target_ret_n{h}"] = np.log(df[f"target_price_n{h}"] / df["close"])

        # Drop initial NaN rows created by rolling windows (e.g. 20-day MA)
        # Keep rows that have valid features
        feature_cols = self.get_feature_columns()
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        logger.info(f"Engineered feature matrix with {len(df)} valid periods and {len(feature_cols)} features.")
        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Returns the list of features strictly derived from sentiment and price only."""
        return [
            # Price Only
            "ret_1d", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
            "hl_ratio", "co_ratio",
            "volatility_5d", "volatility_10d", "volatility_20d",
            "close_to_sma5", "close_to_sma20",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "volume_zscore",
            # Sentiment Only
            "sentiment_mean", "sentiment_weighted", "sentiment_std", "sentiment_subjectivity",
            "post_volume", "post_volume_zscore",
            "bull_count", "bear_count", "bull_bear_ratio",
            "sentiment_momentum_1d", "sentiment_momentum_3d",
            "sentiment_price_divergence", "news_sentiment",
            # Time Horizon / Calendar
            "sin_day_of_week", "cos_day_of_week",
            "sin_month", "cos_month",
            "is_monday", "is_friday"
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fe = FeatureEngineer()
    df_feat = fe.build_dataset()
    print("Dataset Shape:", df_feat.shape)
    print("Features Preview:")
    print(df_feat[fe.get_feature_columns()].head(2))
    print("Targets Preview:")
    print(df_feat[[f"target_price_n{h}" for h in [0, 1, 2, 3, 4, 5]]].head(2))
