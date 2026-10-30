"""
Streamlit Web Dashboard for CrowdWisdomTrading X Sentiment Price Prediction Agent.
Interactive exploration of X sentiment dynamics, multi-horizon forecasts,
and walk-forward risk metrics.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from src.config import config
from src.database.db import load_price_bars, load_tweets, load_news
from src.features.feature_engineer import FeatureEngineer
from src.models.walk_forward import WalkForwardValidator
from src.backtest.portfolio import PortfolioSimulator
from src.visualization.matrix_generator import VisualMatrixGenerator

st.set_page_config(
    page_title="CrowdWisdomTrading | X Sentiment Agent",
    page_icon="📈",
    layout="wide"
)

st.title("CrowdWisdomTrading: X Sentiment Price Prediction Agent")
st.caption("Quantitative Multi-Horizon Forecasting Engine ($n+0$ to $n+5$) | Powered by X Sentiment & Price Only")

# Sidebar
st.sidebar.header("Agent Parameters")
symbol = st.sidebar.selectbox("Select Asset Ticker", ["TSLA", "BTC-USD", "NVDA", "AAPL"], index=0)
model_choice = st.sidebar.selectbox("Forecasting Architecture", ["xgboost", "ridge"], index=0)
train_window = st.sidebar.slider("WFV Initial Train Window (Bars)", 30, 90, 60, step=5)
test_window = st.sidebar.slider("WFV Step Window (Bars)", 5, 30, 15, step=5)

tab_overview, tab_matrix, tab_forecast, tab_backtest, tab_news = st.tabs([
    "Executive Overview", "Visual Matrix", "Multi-Horizon Forecasts", "Walk-Forward Performance", "Tavily News Feed"
])

# Load database data
price_df = load_price_bars(symbol)
tweets_df = load_tweets(symbol)
news_df = load_news(symbol)

with tab_overview:
    if price_df.empty or tweets_df.empty:
        st.warning(f"No cached data found for {symbol}. Run `python main.py --run-all` to ingest and process data.")
    else:
        latest_price = price_df["close"].iloc[-1]
        prev_price = price_df["close"].iloc[-2]
        price_change_pct = ((latest_price - prev_price) / prev_price) * 100.0

        latest_sent = tweets_df["sentiment_score"].mean()
        post_count = len(tweets_df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Spot Price", f"${latest_price:.2f}", f"{price_change_pct:+.2f}%")
        col2.metric("Mean X Sentiment Score", f"{latest_sent:+.3f}", "Scale: [-1.0, +1.0]")
        col3.metric("Total Ingested Tweets", f"{post_count:,}", "180-Day Scope")
        col4.metric("Complementary News Count", f"{len(news_df)}", "Tavily Verified")

        st.subheader("Price History & Aligned UTC / EST Timestamps")
        st.dataframe(price_df.tail(10)[["timestamp_utc", "timestamp_est", "open", "high", "low", "close", "volume"]], use_container_width=True)

with tab_matrix:
    st.subheader("Visual Matrix Output Generator")
    chart_files = [
        ("Executive 4-Panel Master Visual Matrix", config.CHARTS_DIR / "visual_matrix_summary_dashboard.png"),
        ("X Sentiment Dynamics vs. Asset Price Action", config.CHARTS_DIR / "sentiment_vs_price_timeseries.png"),
        ("Multi-Horizon Price Forecasts ($n+0$ to $n+5$)", config.CHARTS_DIR / "multi_horizon_forecast_matrix.png"),
        ("Cumulative Equity Curve & Underwater Drawdown", config.CHARTS_DIR / "equity_curve_and_drawdown.png"),
        ("Feature Importance Matrix (Sentiment vs. Price)", config.CHARTS_DIR / "feature_importance_matrix.png"),
    ]

    for title, path in chart_files:
        if path.exists():
            st.markdown(f"### {title}")
            image = Image.open(path)
            st.image(image, use_container_width=True)
            st.divider()
        else:
            st.info(f"Chart `{path.name}` not found. Run `python main.py --visual-matrix` to generate it.")

with tab_forecast:
    st.subheader("Multi-Horizon Trajectory Engine ($n+0$ through $n+5$)")
    st.markdown("""
    Each horizon represents a distinct temporal target:
    - **$n+0$**: Contemporaneous bar close / nowcast.
    - **$n+1$ through $n+5$**: Forward multi-step forecast trajectory.
    """)
    if not price_df.empty and not tweets_df.empty:
        fe = FeatureEngineer(symbol)
        try:
            feat_df = fe.build_dataset()
            st.success(f"Successfully engineered {feat_df.shape[1]} features across {feat_df.shape[0]} valid bars.")
            st.dataframe(feat_df[[f"target_price_n{h}" for h in [0, 1, 2, 3, 4, 5]]].tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error computing features: {e}")

with tab_backtest:
    st.subheader("Walk-Forward Validation & Hedge Fund Metrics")
    st.markdown("""
    Evaluates out-of-sample Sharpe Ratio, Sortino Ratio, and Maximum Drawdown using **Purged & Embargoed WFV**.
    """)
    rep_path = config.REPORTS_DIR / "model_evaluation_report.md"
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Evaluation report not yet generated. Run `python main.py --run-all`.")

with tab_news:
    st.subheader("Complementary Tavily News Articles")
    if not news_df.empty:
        for _, row in news_df.tail(8).iterrows():
            st.markdown(f"**{row['title']}** (Sentiment: `{row['sentiment_score']:+.2f}`)")
            st.caption(f"Published: {row['published_at_utc']} | Source: {row['source']}")
            st.write(row['snippet'])
            st.markdown("---")
    else:
        st.info("No news articles currently stored in database.")
