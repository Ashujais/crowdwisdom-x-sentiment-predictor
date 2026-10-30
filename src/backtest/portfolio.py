"""
Portfolio Execution and Vectorized Backtesting Engine.
Simulates trading based on multi-horizon predicted trajectories, incorporates
transaction cost frictions, dynamic position sizing, and tracks portfolio value over time.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from src.config import config
from src.backtest.metrics import PerformanceMetrics
from src.models.predictor import ReinforcementLearningPolicy

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    def __init__(
        self,
        initial_capital: float = config.INITIAL_CAPITAL,
        transaction_cost_bps: float = config.TRANSACTION_COST_BPS,
        long_only: bool = True
    ):
        self.initial_capital = initial_capital
        self.cost_rate = transaction_cost_bps / 10_000.0  # 10 bps = 0.0010
        self.long_only = long_only
        self.rl_policy = ReinforcementLearningPolicy(confidence_threshold=0.001, max_leverage=1.0)

    def backtest(self, oos_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes backtest over out-of-sample prediction results.
        oos_df contains actual close, target returns, and predicted returns n+0..n+5.
        """
        df = oos_df.sort_values("timestamp_utc").reset_index(drop=True).copy()

        # Extract predicted multi-horizon returns matrix [N, 6]
        pred_cols = [f"pred_ret_n{h}" for h in config.FORECAST_HORIZONS]
        preds_matrix = df[pred_cols].values

        # Generate target positions using RL policy
        raw_positions = self.rl_policy.compute_action(preds_matrix)

        if self.long_only:
            raw_positions = np.maximum(0.0, raw_positions)

        # Lag position by 1 period so position decided at close of bar t is held during bar t+1
        # This completely guarantees no execution lookahead!
        df["target_position"] = raw_positions
        df["held_position"] = df["target_position"].shift(1).fillna(0.0)

        # Asset next-period return: (close_t - close_{t-1}) / close_{t-1}
        df["asset_ret"] = df["close"].pct_change().fillna(0.0)

        # Cash yield on unallocated balances (e.g. Treasury / SOFR yield)
        rf_daily = (1.0 + config.RISK_FREE_RATE) ** (1.0 / 252.0) - 1.0
        df["cash_yield"] = np.maximum(0.0, 1.0 - np.abs(df["held_position"])) * rf_daily

        # Turnover and transaction friction
        pos_delta = (df["held_position"] - df["held_position"].shift(1).fillna(0.0)).abs()
        df["trade_cost"] = pos_delta * self.cost_rate

        # Gross and Net Strategy Returns
        df["gross_strat_ret"] = (df["held_position"] * df["asset_ret"]) + df["cash_yield"]
        df["strat_ret"] = df["gross_strat_ret"] - df["trade_cost"]
        df["bench_ret"] = df["asset_ret"]

        # Portfolio Values
        df["strategy_equity"] = self.initial_capital * np.cumprod(1.0 + df["strat_ret"])
        df["benchmark_equity"] = self.initial_capital * np.cumprod(1.0 + df["bench_ret"])

        # Compute drawdown series
        running_max = df["strategy_equity"].cummax()
        df["drawdown"] = (df["strategy_equity"] - running_max) / running_max

        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(
            strategy_returns=df["strat_ret"],
            benchmark_returns=df["bench_ret"],
            risk_free_rate=config.RISK_FREE_RATE
        )

        logger.info(
            f"Backtest Finished: Strategy CAGR: {metrics.get('cagr_pct', 0.0):.2f}% vs Bench: {metrics.get('benchmark_cagr_pct', 0.0):.2f}% | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0.0):.2f} | Sortino: {metrics.get('sortino_ratio', 0.0):.2f} | "
            f"MDD: {metrics.get('max_drawdown_pct', 0.0):.2f}%"
        )

        return df, metrics
