"""
Unit tests for Financial Sentiment Analysis Engine.
"""

import pytest
from src.features.sentiment_analyzer import sentiment_engine


def test_sentiment_scoring_bullish():
    text = "$TSLA Q3 earnings crushed consensus! Record gross margins and robotaxi rollout confirmed."
    scores = sentiment_engine.score_text(text)
    assert scores["compound"] > 0.4
    assert scores["pos"] > scores["neg"]


def test_sentiment_scoring_bearish():
    text = "Tesla facing catastrophic recall and SEC investigation. Slashed price target, bankrupt risk."
    scores = sentiment_engine.score_text(text)
    assert scores["compound"] < -0.4
    assert scores["neg"] > scores["pos"]


def test_clean_text():
    raw = "Check out $TSLA at https://t.co/abc123xyz !   Moon soon!  "
    cleaned = sentiment_engine.clean_text(raw)
    assert "https://" not in cleaned
    assert "$TSLA" in cleaned
    assert "  " not in cleaned


def test_engagement_weight_monotonicity():
    w_low = sentiment_engine.calculate_engagement_weight(likes=1, reposts=0, views=10, followers=50)
    w_high = sentiment_engine.calculate_engagement_weight(likes=5000, reposts=1000, views=500000, followers=200000)
    assert w_high > w_low
    assert w_low >= 0.1
