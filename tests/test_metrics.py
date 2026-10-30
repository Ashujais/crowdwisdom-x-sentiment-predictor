"""
Unit tests for Performance Metrics (Sharpe, Sortino, Max Drawdown).
"""

import pytest
import numpy as np
import pandas as pd
from src.backtest.metrics import PerformanceMetrics


def test_metrics_calculation_synthetic():
    # Perfectly positive daily returns (0.1% daily)
    r_strat = pd.Series([0.001] * 252)
    r_bench = pd.Series([0.0005] * 252)

    m = PerformanceMetrics.calculate_metrics(r_strat, r_bench, risk_free_rate=0.045)

    assert "sharpe_ratio" in m
    assert "sortino_ratio" in m
    assert "max_drawdown_pct" in m
    assert "cagr_pct" in m
    # With zero negative returns, drawdown is 0%
    assert m["max_drawdown_pct"] == 0.0
    assert m["total_return_pct"] > 25.0


def test_drawdown_calculation():
    # Synthetic series with a known 20% drop
    prices = [100.0, 110.0, 120.0, 96.0, 96.0, 105.0]
    returns = pd.Series(pd.Series(prices).pct_change().dropna())
    bench = pd.Series([0.0] * len(returns))

    m = PerformanceMetrics.calculate_metrics(returns, bench, risk_free_rate=0.0)
    # Peak is 120, trough is 96 -> (96 - 120) / 120 = -20%
    assert abs(m["max_drawdown_pct"] - (-20.0)) < 0.1
