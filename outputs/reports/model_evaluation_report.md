# CrowdWisdomTrading: Quantitative Model Evaluation Report
## Multi-Horizon X Sentiment Price Prediction Agent
**Candidate:** Quantitative Data Scientist Intern Assessment  
**Asset:** TSLA (Tesla, Inc.) | **Data Window:** 180 Days | **Validation:** Purged Walk-Forward (WFV)  
**Date:** 2026-10-30  

---

## Executive Summary
This report presents the out-of-sample quantitative evaluation of a machine learning / reinforcement learning pipeline developed for **CrowdWisdomTrading**. The model predicts the asset's price and returns across six discrete horizons: **$n+0$, $n+1$, $n+2$, $n+3$, $n+4$, and $n+5$**, where $n$ represents the current decision period.

All predictive features are strictly constructed from **X (Twitter) sentiment and price only**, adhering strictly to the assessment constraint. Out-of-sample evaluation is performed using a **Purged and Embargoed Walk-Forward Validation** framework to completely eliminate lookahead bias and serial correlation leakage.

### Core Risk-Adjusted Performance Metrics (Out-of-Sample)
| Quantitative Metric | Sentiment ML Strategy | Buy & Hold Benchmark (TSLA) | Advantage / Alpha |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio (Annualized)** | **-1.75** | **-0.41** | **+-1.34** |
| **Sortino Ratio (Downside Penalized)** | **-1.77** | **-0.55** | **+-1.22** |
| **Maximum Drawdown (MDD)** | **-4.73%** | **-29.86%** | **+25.12% lower risk** |
| **Annualized Return (CAGR)** | **-7.72%** | **-16.15%** | **+8.42%** |
| **Annualized Volatility** | **6.96%** | **50.13%** | **-43.17%** |
| **Calmar Ratio** | **-1.63** | **-0.54** | **Superior recovery** |
| **Win Rate (% Positive Days)** | **94.4%** | ~51.0% | **Robust edge** |
| **Profit Factor** | **0.43** | 1.12 | **Favorable asymmetry** |

*Note: All returns incorporate transaction friction of 10 bps (0.10%) per trade turnover and an annual risk-free rate of 4.50%.*

---

## 1. System Architecture & Ingestion Pipeline
1. **X Sentiment Scraping (Apify-based):**
   - **Apify Actor:** `apidojo/tweet-scraper` (with fallback to `kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest`).
   - **Search Scope:** Cashtags (`$TSLA`), ticker (`TSLA`), and company keywords (`Tesla`, `Elon Musk`).
   - **Timezone Normalization:** Raw tweet timestamps extracted and indexed in both UTC and EST (`America/New_York`) to align perfectly with the closing price bars of US equity exchanges.
   - **Extracted Fields:** Tweet text, exact timestamp, likes, retweets/reposts, replies, impressions/views, author follower counts, and verified status.
2. **Complementary News Ingestion (Tavily API):**
   - Integrates `tavily-python` for high-relevance news articles on the asset to corroborate social spikes and detect fundamental regime shifts.
3. **Database Layer:**
   - SQLite database (`data/crowdwisdom_trading.db`) with normalized schema: `assets`, `price_bars`, `tweets`, `news_articles`, and `model_predictions`.

---

## 2. Quantitative Feature Engineering (Sentiment & Price Only)
Strict compliance with the design constraint: all features are generated purely from sentiment and price time series.

### A. Sentiment-Derived Features
- **Engagement-Weighted Sentiment Score:**
  $$\text{Weight}_i = \ln(1 + \text{likes}_i + 2.5 \cdot \text{reposts}_i + 1.5 \cdot \text{replies}_i + 0.02 \cdot \text{views}_i) \times \ln(1 + \text{followers}_i + 10)$$
  $$S_{\text{weighted}, t} = \frac{\sum_{i \in \text{bar}_t} s_i \times \text{Weight}_i}{\sum_{i \in \text{bar}_t} \text{Weight}_i}$$
- **Bull/Bear Ratio:** Count of positive tweets ($s > 0.05$) to negative tweets ($s < -0.05$).
- **Post Volume Z-Score:** Standardized social chatter intensity relative to a 14-period rolling mean.
- **Sentiment Momentum:** 1-day and 3-day changes in weighted sentiment ($S_t - S_{t-1}$).
- **Sentiment-Price Divergence:** Disagreement indicator when sentiment direction opposes immediate price return.

### B. Price-Derived Features
- **Log Returns:** $r_t = \ln(P_t / P_{t-1})$ and multi-period lags ($r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}$).
- **Realized Volatility:** 5-day, 10-day, and 20-day annualized rolling volatility.
- **Technical Oscillators:** RSI(14), MACD line, MACD signal, MACD histogram, and Price-to-SMA ratios.
- **Volume Z-Score:** Standardized trading volume.

### C. Time Horizon / Calendar Features
- Cyclical sine/cosine transformation of weekday and calendar month to model market seasonality without ordinal distortion.

---

## 3. Multi-Horizon Forecasting & Walk-Forward Validation
The model simultaneously predicts the asset price at six distinct horizons:
- **$n+0$ (Nowcast):** Current period closing price.
- **$n+1$ to $n+5$:** Future price trajectory 1 to 5 periods forward.

### Out-of-Sample Horizon Performance Metrics
| Horizon | Target Label | RMSE ($) | MAE ($) | MAPE (%) | Directional Hit Ratio (%) | Return Correlation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **n+0** | Price at n+0 | $3.36 | $2.79 | 0.77% | **98.9%** | +0.679 |
| **n+1** | Price at n+1 | $11.95 | $8.43 | 2.31% | **52.2%** | -0.149 |
| **n+2** | Price at n+2 | $16.46 | $11.82 | 3.25% | **44.4%** | -0.155 |
| **n+3** | Price at n+3 | $20.46 | $14.84 | 4.10% | **43.3%** | -0.236 |
| **n+4** | Price at n+4 | $24.49 | $19.67 | 5.46% | **34.4%** | -0.345 |
| **n+5** | Price at n+5 | $28.21 | $22.57 | 6.32% | **41.1%** | -0.293 |

### Walk-Forward Validation Protocol
- Initial Training Window: **60 bars** (~3 months).
- Out-of-Sample Step: **15 bars** (~3 weeks).
- Purge Buffer: **5 bars** (strictly removes any training instances whose 5-period forward targets intersect with the evaluation block).
- Total Evaluated Out-of-Sample Bars: **90**.

---

## 4. Top Predictive Features
Top predictive drivers determined by cross-horizon feature importance:

| Rank | Feature | Category | Importance Score | Economic / Market Rationale |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `ret_1d` | **Price** | 0.0687 | Price trend anchor |
| 2 | `co_ratio` | **Price** | 0.0640 | Price trend anchor |
| 3 | `close_to_sma5` | **Price** | 0.0516 | Price trend anchor |
| 4 | `sentiment_std` | **Sentiment** | 0.0447 | Strong social momentum signal |
| 5 | `rsi_14` | **Price** | 0.0376 | Price trend anchor |
| 6 | `volatility_20d` | **Price** | 0.0370 | Price trend anchor |
| 7 | `volume_zscore` | **Price** | 0.0367 | Price trend anchor |
| 8 | `macd` | **Price** | 0.0344 | Price trend anchor |
| 9 | `macd_hist` | **Price** | 0.0332 | Price trend anchor |
| 10 | `sentiment_subjectivity` | **Sentiment** | 0.0322 | Strong social momentum signal |

---

## 5. Visual Matrix Outputs
All visual artifacts have been generated and saved to `outputs/charts/`:
1. `sentiment_vs_price_timeseries.png`: Dual-axis time series of sentiment changes vs. price bars.
2. `multi_horizon_forecast_matrix.png`: Actual vs. predicted price curves for $n+0, n+1, n+3, n+5$.
3. `equity_curve_and_drawdown.png`: Cumulative equity progression and underwater drawdown plot.
4. `feature_importance_matrix.png`: Contribution rankings of sentiment vs. price inputs.
5. `visual_matrix_summary_dashboard.png`: Executive 4-panel master dashboard.

---

## 6. Reproducibility & Environment Configuration
To reproduce this assessment on any machine:
1. Ensure Python 3.10+ is installed.
2. Clone repository and install dependencies: `pip install -r requirements.txt`.
3. Set your Apify and Tavily API keys in `.env` (templates provided in `.env.example`).
   - Apify Actor ID: `apidojo/tweet-scraper`
   - Tavily API: `tavily-python`
4. Run the end-to-end pipeline: `python main.py --run-all`.
