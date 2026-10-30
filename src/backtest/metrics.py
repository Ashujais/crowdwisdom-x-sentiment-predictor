"""
Quantitative Portfolio & Performance Metrics Engine.
Computes institutional-grade metrics: Sharpe Ratio, Sortino Ratio, Maximum Drawdown,
Calmar Ratio, Profit Factor, Win Rate, and Drawdown Duration.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.config import config


class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = config.RISK_FREE_RATE,
        annualization_factor: int = 252
    ) -> Dict[str, Any]:
        """
        Compute comprehensive quantitative performance metrics.
        """
        r_strat = strategy_returns.dropna().values
        r_bench = benchmark_returns.dropna().values
        N = len(r_strat)

        if N == 0:
            return {}

        rf_daily = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0

        # Equity Curves
        strat_cum = np.cumprod(1.0 + r_strat)
        bench_cum = np.cumprod(1.0 + r_bench)

        # Total & Annualized Return
        total_return = strat_cum[-1] - 1.0
        bench_total_return = bench_cum[-1] - 1.0

        years = max(N / float(annualization_factor), 1.0 / annualization_factor)
        cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if (1.0 + total_return) > 0 else -1.0
        bench_cagr = (1.0 + bench_total_return) ** (1.0 / years) - 1.0 if (1.0 + bench_total_return) > 0 else -1.0

        # Volatility
        vol_daily = np.std(r_strat, ddof=1) if len(r_strat) > 1 else 0.0
        vol_ann = vol_daily * np.sqrt(annualization_factor)
        bench_vol = (np.std(r_bench, ddof=1) if len(r_bench) > 1 else 0.0) * np.sqrt(annualization_factor)

        # Sharpe Ratio
        excess_returns = r_strat - rf_daily
        sharpe = (np.mean(excess_returns) / (vol_daily + 1e-8)) * np.sqrt(annualization_factor) if vol_daily > 0 else 0.0

        # Sortino Ratio (downside deviation only)
        downside_returns = np.minimum(0.0, excess_returns)
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        sortino = (np.mean(excess_returns) / (downside_deviation + 1e-8)) * np.sqrt(annualization_factor) if downside_deviation > 0 else 0.0

        # Maximum Drawdown & Underwater Series
        running_max = np.maximum.accumulate(strat_cum)
        drawdowns = (strat_cum - running_max) / running_max
        max_drawdown = float(np.min(drawdowns))  # negative number e.g. -0.15 for -15%

        # Benchmark Drawdown
        bench_running_max = np.maximum.accumulate(bench_cum)
        bench_drawdowns = (bench_cum - bench_running_max) / bench_running_max
        bench_mdd = float(np.min(bench_drawdowns))

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else 0.0

        # Win Rate & Profit Factor
        winning_trades = r_strat[r_strat > 0]
        losing_trades = r_strat[r_strat < 0]
        win_rate = (len(winning_trades) / len(r_strat[r_strat != 0])) * 100.0 if len(r_strat[r_strat != 0]) > 0 else 0.0
        gross_profits = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_losses = np.abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 999.0

        return {
            "total_return_pct": float(total_return * 100.0),
            "cagr_pct": float(cagr * 100.0),
            "benchmark_total_return_pct": float(bench_total_return * 100.0),
            "benchmark_cagr_pct": float(bench_cagr * 100.0),
            "annualized_volatility_pct": float(vol_ann * 100.0),
            "benchmark_volatility_pct": float(bench_vol * 100.0),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown_pct": float(max_drawdown * 100.0),
            "benchmark_max_drawdown_pct": float(bench_mdd * 100.0),
            "calmar_ratio": float(calmar),
            "win_rate_pct": float(win_rate),
            "profit_factor": float(profit_factor),
            "n_periods": int(N)
        }
