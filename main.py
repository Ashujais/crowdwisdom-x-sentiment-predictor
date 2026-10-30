"""
Main CLI Orchestration Entry Point for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Executes end-to-end ingestion, quantitative feature engineering, walk-forward validation,
visual matrix output generation, and PDF/Markdown report compilation in a single command.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import config
from src.database.db import init_db
from src.ingestion.price_ingest import PriceIngester
from src.ingestion.apify_x_scraper import ApifyTwitterScraper
from src.ingestion.tavily_news_ingest import TavilyNewsIngester
from src.features.feature_engineer import FeatureEngineer
from src.models.walk_forward import WalkForwardValidator
from src.backtest.portfolio import PortfolioSimulator
from src.visualization.matrix_generator import VisualMatrixGenerator
from src.reports.sentiment_summary import SentimentOutlookSummaryGenerator
from src.reports.evaluation_report import ModelEvaluationReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("CWT_Agent")


def run_pipeline(
    symbol: str = config.TARGET_ASSET,
    run_ingest: bool = True,
    run_modeling: bool = True,
    run_visuals: bool = True,
    run_reports: bool = True
):
    logger.info("=" * 70)
    logger.info(f"CrowdWisdomTrading: Initializing X Sentiment Prediction Agent for {symbol}")
    logger.info("=" * 70)

    # 0. Initialize SQLite Database Schema
    init_db()

    # 1. Ingestion Pipeline
    if run_ingest:
        logger.info("\n--- Phase 1: Ingestion (Price Bars, Apify X Scraping, Tavily News) ---")
        price_ingester = PriceIngester(symbol=symbol)
        price_df = price_ingester.run()

        tweet_scraper = ApifyTwitterScraper(symbol=symbol)
        tweets_df = tweet_scraper.run(price_df=price_df)

        news_ingester = TavilyNewsIngester(symbol=symbol)
        news_df = news_ingester.run()
        logger.info("Ingestion completed successfully.")

    # 2. Feature Engineering (Strictly Sentiment and Price Only)
    logger.info("\n--- Phase 2: Quantitative Feature Engineering ---")
    fe = FeatureEngineer(symbol=symbol)
    df_features = fe.build_dataset()
    logger.info(f"Dataset successfully constructed: {df_features.shape[0]} rows, {df_features.shape[1]} columns.")

    # 3. Model Training & Walk-Forward Validation
    if run_modeling:
        logger.info("\n--- Phase 3: Machine Learning Model & Walk-Forward Validation (n+0 to n+5) ---")
        wfv = WalkForwardValidator()
        oos_df, horizon_metrics, imp_df = wfv.run_walk_forward(df_features)

        logger.info("\n--- Phase 4: Portfolio Backtesting & Quantitative Metrics ---")
        sim = PortfolioSimulator()
        backtest_df, metrics = sim.backtest(oos_df)

        print("\n" + "=" * 55)
        print("          WALK-FORWARD PERFORMANCE SUMMARY")
        print("=" * 55)
        print(f"Sharpe Ratio:             {metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"Sortino Ratio:            {metrics.get('sortino_ratio', 0.0):.2f}")
        print(f"Maximum Drawdown:         {metrics.get('max_drawdown_pct', 0.0):.2f}%")
        print(f"Strategy CAGR:            {metrics.get('cagr_pct', 0.0):.2f}%")
        print(f"Benchmark CAGR ({symbol}):     {metrics.get('benchmark_cagr_pct', 0.0):.2f}%")
        print(f"Win Rate:                 {metrics.get('win_rate_pct', 0.0):.1f}%")
        print(f"Profit Factor:            {metrics.get('profit_factor', 0.0):.2f}")
        print("=" * 55 + "\n")

    # 4. Visual Matrix Output Generator
    if run_visuals:
        logger.info("\n--- Phase 5: Generating Visual Matrix Outputs ---")
        viz = VisualMatrixGenerator()
        viz.plot_sentiment_vs_price(df_features, symbol=symbol)
        viz.plot_multi_horizon_forecast(oos_df, symbol=symbol)
        viz.plot_equity_and_drawdown(backtest_df, metrics, symbol=symbol)
        viz.plot_feature_importance(imp_df)
        viz.plot_master_visual_matrix(df_features, oos_df, backtest_df, metrics, imp_df, symbol=symbol)
        logger.info("All visual matrix charts saved to outputs/charts/.")

    # 5. Reports Generation (PDF and Markdown)
    if run_reports:
        logger.info("\n--- Phase 6: Compiling PDF & Markdown Reports ---")
        latest_row = df_features.iloc[-1]
        latest_price = float(latest_row["close"])

        # Latest forecasts
        latest_forecasts = {
            0: float(oos_df["pred_price_n0"].iloc[-1]),
            1: float(oos_df["pred_price_n1"].iloc[-1]),
            2: float(oos_df["pred_price_n2"].iloc[-1]),
            3: float(oos_df["pred_price_n3"].iloc[-1]),
            4: float(oos_df["pred_price_n4"].iloc[-1]),
            5: float(oos_df["pred_price_n5"].iloc[-1]),
        }

        # 1-Page Sentiment Outlook
        summary_gen = SentimentOutlookSummaryGenerator()
        summary_gen.generate_summary(latest_row, latest_forecasts, symbol=symbol)

        # Evaluation Report (Sharpe, Sortino, Max Drawdown)
        eval_gen = ModelEvaluationReportGenerator()
        eval_gen.generate_report(metrics, horizon_metrics, imp_df, symbol=symbol)
        logger.info("Reports generated in outputs/reports/ (PDF & Markdown).")

    logger.info("=" * 70)
    logger.info("Pipeline Execution Successfully Finished!")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="CrowdWisdomTrading X Sentiment Price Prediction Agent")
    parser.add_argument("--symbol", type=str, default=config.TARGET_ASSET, help="Asset ticker symbol (e.g. TSLA, BTC-USD)")
    parser.add_argument("--run-all", action="store_true", help="Execute complete end-to-end pipeline")
    parser.add_argument("--ingest", action="store_true", help="Execute ingestion phase only")
    parser.add_argument("--wfv", action="store_true", help="Execute walk-forward validation and backtest only")
    parser.add_argument("--visual-matrix", action="store_true", help="Generate visual matrix charts only")
    parser.add_argument("--reports", action="store_true", help="Compile PDF and Markdown reports only")

    args = parser.parse_args()

    # Default to running all if no specific step is selected
    if not (args.ingest or args.wfv or args.visual_matrix or args.reports):
        args.run_all = True

    if args.run_all:
        run_pipeline(symbol=args.symbol, run_ingest=True, run_modeling=True, run_visuals=True, run_reports=True)
    else:
        run_pipeline(
            symbol=args.symbol,
            run_ingest=args.ingest,
            run_modeling=args.wfv,
            run_visuals=args.visual_matrix,
            run_reports=args.reports
        )


if __name__ == "__main__":
    main()
