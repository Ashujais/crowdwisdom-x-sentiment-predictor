"""
Domain-specific Financial Sentiment Analysis Engine.
Combines VADER with custom financial market lexicon (Loughran-McDonald, WallStreetBets, FinTwit),
computes raw sentiment polarity, subjectivity, and engagement-weighted impact scores.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Any, Tuple
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure required NLTK assets
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


# Financial & FinTwit Domain Specific Lexicon
FINANCIAL_LEXICON_UPDATES = {
    # Strong Bullish
    "bullish": 2.5,
    "breakout": 2.2,
    "surge": 2.0,
    "rally": 2.0,
    "moon": 2.4,
    "soaring": 2.1,
    "upgrade": 2.2,
    "upgraded": 2.2,
    "outperform": 2.3,
    "buy": 1.8,
    "long": 1.5,
    "accumulate": 1.7,
    "calls": 1.6,
    "ath": 2.5,
    "all-time-high": 2.5,
    "record": 1.8,
    "beat": 2.0,
    "beats": 2.0,
    "exceeded": 1.9,
    "profit": 1.8,
    "profitable": 1.8,
    "growth": 1.6,
    "undervalued": 1.8,
    "delivery beat": 2.6,
    "crushed": 2.2,
    "crushing": 2.0,
    "strong buy": 2.8,
    "oversold": 1.4,
    "green": 1.2,
    "favorable": 1.5,
    "innovative": 1.6,
    "favorable": 1.5,
    "guidance raise": 2.5,
    "robotaxi": 1.5,
    "fsd": 1.4,

    # Strong Bearish
    "bearish": -2.5,
    "dump": -2.3,
    "crash": -2.6,
    "plunge": -2.4,
    "slump": -2.0,
    "collapse": -2.8,
    "short": -1.6,
    "sell": -1.8,
    "downgrade": -2.2,
    "downgraded": -2.2,
    "underperform": -2.3,
    "miss": -2.1,
    "missed": -2.1,
    "puts": -1.6,
    "loss": -1.8,
    "unprofitable": -2.0,
    "rekt": -2.5,
    "overvalued": -2.0,
    "recall": -2.2,
    "recalls": -2.2,
    "fraud": -3.2,
    "lawsuit": -2.4,
    "investigation": -2.3,
    "sec": -1.5,
    "subpoena": -2.6,
    "dilution": -2.2,
    "offering": -1.7,
    "cut": -1.8,
    "slashed": -2.2,
    "warning": -1.9,
    "bankrupt": -3.5,
    "bankruptcy": -3.5,
    "red": -1.2,
    "margin compression": -2.2,
    "tariff": -1.4,
    "delays": -1.8
}


class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Enforce financial lexicon onto VADER
        self.vader.lexicon.update(FINANCIAL_LEXICON_UPDATES)

    def clean_text(self, text: str) -> str:
        """Clean tweet text while preserving financial cashtags and ticker mentions."""
        if not isinstance(text, str):
            return ""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Compute compound, positive, negative, and neutral scores.
        Compound ranges from -1.0 (extremely bearish) to +1.0 (extremely bullish).
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0, "subjectivity": 0.0}

        scores = self.vader.polarity_scores(cleaned)
        # Subjectivity estimate: proportion of emotive vs neutral words
        subjectivity = float(scores["pos"] + scores["neg"])
        scores["subjectivity"] = subjectivity
        return scores

    @staticmethod
    def calculate_engagement_weight(
        likes: int = 0,
        reposts: int = 0,
        replies: int = 0,
        views: int = 0,
        followers: int = 0
    ) -> float:
        """
        Calculate engagement weight representing social reach and market-moving influence.
        Uses logarithmic scaling to prevent massive accounts from dominating exponentially.
        """
        likes = max(0, int(likes))
        reposts = max(0, int(reposts))
        replies = max(0, int(replies))
        views = max(0, int(views))
        followers = max(0, int(followers))

        interaction_power = likes + (2.5 * reposts) + (1.5 * replies) + (0.02 * views)
        weight = math.log1p(interaction_power) * math.log1p(followers + 10)
        # Standardize baseline weight to be around 1.0 for median tweets
        return max(0.1, weight / 10.0)


# Module singleton
sentiment_engine = SentimentAnalyzer()


if __name__ == "__main__":
    t1 = "$TSLA delivery numbers crushed Wall St consensus! Robotaxi rollout confirmed. Strong buy!"
    t2 = "Tesla margins under serious pressure due to price cuts and EV competition. Slashed price target."
    print("Bullish test:", sentiment_engine.score_text(t1))
    print("Bearish test:", sentiment_engine.score_text(t2))
    w = sentiment_engine.calculate_engagement_weight(likes=1500, reposts=300, views=80000, followers=50000)
    print("Calculated engagement weight:", w)
