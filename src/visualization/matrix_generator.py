"""
Visual Matrix Output Generator.
Generates institutional-grade, publication-quality visualizations:
1. X Sentiment change against Asset Price over time (dual-axis with volume bars)
2. Multi-Horizon Forecast Matrix (actual vs. n+0..n+5 trajectories)
3. Cumulative Strategy Performance & Underwater Drawdown vs. Benchmark
4. Sentiment-Price Correlation Heatmap
5. Master 4-Panel Visual Matrix Dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.config import config

logger = logging.getLogger(__name__)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


class VisualMatrixGenerator:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sentiment_vs_price(self, df: pd.DataFrame, symbol: str = config.TARGET_ASSET) -> Path:
        """
        Dual-axis time series chart:
        Asset Price against X Sentiment Change and Post Volume over time.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

        dates = pd.to_datetime(df["timestamp_utc"])
        price = df["close"]
        sentiment = df["sentiment_weighted"]
        volume = df["post_volume"]

        # Panel 1: Price Action with Sentiment Shading
        ax1.plot(dates, price, color="#1f77b4", linewidth=2.0, label=f"{symbol} Close Price")
        if "close_to_sma20" in df.columns:
            sma20 = price.rolling(20).mean()
            ax1.plot(dates, sma20, color="#ff7f0e", linestyle="--", linewidth=1.5, label="20-Day SMA")

        ax1.set_title(f"CrowdWisdomTrading: {symbol} Price Action vs. X (Twitter) Crowd Sentiment Dynamics", fontsize=14, weight="bold", pad=12)
        ax1.set_ylabel("Price ($ USD)", fontsize=11, weight="semibold")
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Sentiment Score Line + Tweet Volume
        ax2_vol = ax2.twinx()
        ax2_vol.bar(dates, volume, color="#9467bd", alpha=0.3, width=0.8, label="X Post Volume")
        ax2_vol.set_ylabel("X Post Volume", color="#9467bd", fontsize=10)
        ax2_vol.grid(False)

        # Color sentiment: positive green, negative red
        ax2.plot(dates, sentiment, color="#2ca02c", linewidth=1.8, label="Weighted Sentiment Score")
        ax2.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
        ax2.fill_between(dates, sentiment, 0, where=(sentiment >= 0), color="#2ca02c", alpha=0.25, interpolate=True)
        ax2.fill_between(dates, sentiment, 0, where=(sentiment < 0), color="#d62728", alpha=0.25, interpolate=True)

        ax2.set_ylabel("Sentiment Score [-1, +1]", fontsize=10, weight="semibold")
        ax2.set_xlabel("Date (UTC / EST Aligned)", fontsize=11, weight="semibold")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = self.output_dir / "sentiment_vs_price_timeseries.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
        return save_path

    def plot_multi_horizon_forecast(self, oos_df: pd.DataFrame, symbol: str = config.TARGET_ASSET) -> Path:
        """
        Multi-horizon prediction matrix comparing actual prices vs. forecasts at n+0, n+1, n+3, and n+5.
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        horizons_to_plot = [(0, axs[0, 0]), (1, axs[0, 1]), (3, axs[1, 0]), (5, axs[1, 1])]

        dates = pd.to_datetime(oos_df["timestamp_utc"])

        for h, ax in horizons_to_plot:
            actual = oos_df[f"target_price_n{h}"]
            pred = oos_df[f"pred_price_n{h}"]

            ax.plot(dates, actual, color="#333333", linewidth=1.8, label="Actual Price")
            ax.plot(dates, pred, color="#007acc", linestyle="--", linewidth=1.8, label=f"Predicted (n+{h})")

            # Error metric
            mae = np.mean(np.abs(actual - pred))
            hit_ratio = np.mean(np.sign(oos_df[f"target_ret_n{h}"]) == np.sign(oos_df[f"pred_ret_n{h}"])) * 100.0

            ax.set_title(f"Horizon n+{h} Forecast | MAE: ${mae:.2f} | Hit Ratio: {hit_ratio:.1f}%", fontsize=11, weight="bold")
            ax.set_ylabel("Price ($)", fontsize=10)
            ax.legend(loc="upper left")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        fig.suptitle(f"{symbol} Multi-Horizon Price Forecast Matrix (Out-of-Sample Walk-Forward)", fontsize=14, weight="bold", y=0.99)
        plt.tight_layout()
        save_path = self.output_dir / "multi_horizon_forecast_matrix.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
        return save_path

    def plot_equity_and_drawdown(self, backtest_df: pd.DataFrame, metrics: Dict[str, Any], symbol: str = config.TARGET_ASSET) -> Path:
        """
        Equity curve and underwater drawdown comparing Sentiment Strategy vs. Buy & Hold Benchmark.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})

        dates = pd.to_datetime(backtest_df["timestamp_utc"])
        strat_eq = backtest_df["strategy_equity"]
        bench_eq = backtest_df["benchmark_equity"]
        drawdown = backtest_df["drawdown"] * 100.0

        # Equity Curve
        ax1.plot(dates, strat_eq, color="#008080", linewidth=2.2, label=f"X Sentiment ML Strategy (Sharpe: {metrics.get('sharpe_ratio', 0.0):.2f}, Sortino: {metrics.get('sortino_ratio', 0.0):.2f})")
        ax1.plot(dates, bench_eq, color="#7f7f7f", linestyle="--", linewidth=1.8, label=f"Buy & Hold {symbol} Benchmark (Return: {metrics.get('benchmark_total_return_pct', 0.0):.1f}%)")

        ax1.set_title(f"Walk-Forward Cumulative Equity & Risk Profile: X Sentiment Strategy vs. {symbol}", fontsize=14, weight="bold")
        ax1.set_ylabel("Portfolio Value ($ USD)", fontsize=11, weight="semibold")
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Underwater Drawdown Plot
        ax2.fill_between(dates, drawdown, 0, color="#d9534f", alpha=0.4, label="Strategy Drawdown")
        ax2.plot(dates, drawdown, color="#c9302c", linewidth=1.2)
        ax2.set_ylabel("Drawdown (%)", fontsize=10, weight="semibold")
        ax2.set_xlabel("Date", fontsize=11, weight="semibold")
        ax2.axhline(metrics.get("max_drawdown_pct", 0.0), color="#900", linestyle=":", label=f"Max DD: {metrics.get('max_drawdown_pct', 0.0):.1f}%")
        ax2.legend(loc="lower left")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        plt.tight_layout()
        save_path = self.output_dir / "equity_curve_and_drawdown.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
        return save_path

    def plot_feature_importance(self, imp_df: pd.DataFrame, top_n: int = 15) -> Path:
        """Horizontal bar chart of top predictive features."""
        fig, ax = plt.subplots(figsize=(12, 7))
        top_df = imp_df.head(top_n).iloc[::-1]

        colors = ["#2ca02c" if any(s in feat for s in ["sentiment", "post", "bull", "bear", "news"]) else "#1f77b4" for feat in top_df["feature"]]

        ax.barh(top_df["feature"], top_df["mean_importance"], color=colors, alpha=0.85, edgecolor="#333333")
        ax.set_title("Walk-Forward Model Feature Importance (Green = Sentiment Features, Blue = Price Features)", fontsize=13, weight="bold")
        ax.set_xlabel("Mean Feature Importance (across horizons n+0 to n+5)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        save_path = self.output_dir / "feature_importance_matrix.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
        return save_path

    def plot_master_visual_matrix(
        self,
        df: pd.DataFrame,
        oos_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        metrics: Dict[str, Any],
        imp_df: pd.DataFrame,
        symbol: str = config.TARGET_ASSET
    ) -> Path:
        """
        Master 4-Panel Visual Matrix Output Dashboard.
        Combines Sentiment-Price dynamics, forecast accuracy, equity curve, and feature weights into a single executive artifact.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

        # Panel 1: Sentiment vs Price Time-series
        ax1 = fig.add_subplot(gs[0, 0])
        dates1 = pd.to_datetime(df["timestamp_utc"])
        ax1.plot(dates1, df["close"], color="#1f77b4", linewidth=1.8, label="Close Price")
        ax1.set_title("Panel A: Asset Price vs. X Sentiment Dynamics", fontsize=12, weight="bold")
        ax1.set_ylabel("Price ($)")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(dates1, df["sentiment_weighted"], color="#2ca02c", alpha=0.7, label="Sentiment")
        ax1_twin.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax1_twin.set_ylabel("Sentiment Score", color="#2ca02c")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

        # Panel 2: Multi-Horizon Forecast (n+1)
        ax2 = fig.add_subplot(gs[0, 1])
        dates2 = pd.to_datetime(oos_df["timestamp_utc"])
        ax2.plot(dates2, oos_df["target_price_n1"], color="#333333", linewidth=1.5, label="Actual (n+1)")
        ax2.plot(dates2, oos_df["pred_price_n1"], color="#d62728", linestyle="--", linewidth=1.5, label="Predicted (n+1)")
        ax2.set_title("Panel B: Multi-Horizon Walk-Forward Forecast (Horizon n+1)", fontsize=12, weight="bold")
        ax2.set_ylabel("Price ($)")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        # Panel 3: Cumulative Performance & Drawdown
        ax3 = fig.add_subplot(gs[1, 0])
        dates3 = pd.to_datetime(backtest_df["timestamp_utc"])
        ax3.plot(dates3, backtest_df["strategy_equity"], color="#008080", linewidth=2.0, label="Sentiment Strategy")
        ax3.plot(dates3, backtest_df["benchmark_equity"], color="#7f7f7f", linestyle="--", linewidth=1.5, label="Benchmark")
        ax3.set_title(f"Panel C: Cumulative Wealth (Sharpe: {metrics.get('sharpe_ratio', 0.0):.2f} | Sortino: {metrics.get('sortino_ratio', 0.0):.2f})", fontsize=12, weight="bold")
        ax3.set_ylabel("Portfolio Value ($)")
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        # Panel 4: Top Features
        ax4 = fig.add_subplot(gs[1, 1])
        top_10 = imp_df.head(8).iloc[::-1]
        colors = ["#2ca02c" if any(s in f for s in ["sentiment", "post", "bull", "bear", "news"]) else "#1f77b4" for f in top_10["feature"]]
        ax4.barh(top_10["feature"], top_10["mean_importance"], color=colors, alpha=0.85)
        ax4.set_title("Panel D: Feature Contribution (Green=Sentiment, Blue=Price)", fontsize=12, weight="bold")
        ax4.set_xlabel("Mean Importance")
        ax4.grid(True, linestyle=":", alpha=0.5)

        fig.suptitle(f"CrowdWisdomTrading: {symbol} X Sentiment Price Prediction Visual Matrix", fontsize=16, weight="bold", y=0.98)
        save_path = self.output_dir / "visual_matrix_summary_dashboard.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {save_path}")
        return save_path
