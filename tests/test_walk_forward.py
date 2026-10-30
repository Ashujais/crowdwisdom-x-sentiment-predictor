"""
Unit tests for Walk-Forward Validator and Target Alignment.
"""

import pytest
import pandas as pd
from src.features.feature_engineer import FeatureEngineer
from src.models.walk_forward import WalkForwardValidator


def test_walk_forward_splits():
    fe = FeatureEngineer()
    df = fe.build_dataset()

    wfv = WalkForwardValidator(train_window=50, test_window=10, purge_window=5)
    oos_df, h_metrics, imp_df = wfv.run_walk_forward(df)

    assert not oos_df.empty
    assert len(oos_df) >= 30
    assert "pred_price_n0" in oos_df.columns
    assert "pred_price_n1" in oos_df.columns
    assert "pred_price_n5" in oos_df.columns

    for h in [0, 1, 2, 3, 4, 5]:
        assert f"n+{h}" in h_metrics
        assert "rmse_price" in h_metrics[f"n+{h}"]
        assert "hit_ratio_pct" in h_metrics[f"n+{h}"]
