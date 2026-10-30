"""
Unit tests for Quantitative Feature Engineer and Data Alignment.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineer import FeatureEngineer


def test_feature_columns_constraint():
    """Verify that features are strictly derived from sentiment and price only."""
    feat_cols = FeatureEngineer.get_feature_columns()
    allowed_keywords = [
        "ret", "hl", "co", "volatility", "sma", "rsi", "macd", "volume",
        "sentiment", "post", "bull", "bear", "news",
        "day_of_week", "month", "monday", "friday"
    ]
    for col in feat_cols:
        assert any(k in col for k in allowed_keywords), f"Unallowed feature detected: {col}"


def test_feature_dataset_integrity():
    """Verify dataset builds cleanly from database with no null feature rows."""
    fe = FeatureEngineer()
    df = fe.build_dataset()
    assert not df.empty
    assert len(df) > 50

    feat_cols = fe.get_feature_columns()
    # Check no NaNs in features
    assert df[feat_cols].isna().sum().sum() == 0

    # Check all targets n+0 to n+5 exist
    for h in [0, 1, 2, 3, 4, 5]:
        assert f"target_price_n{h}" in df.columns
        assert f"target_ret_n{h}" in df.columns
