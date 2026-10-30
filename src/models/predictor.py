"""
Multi-Horizon Machine Learning & Reinforcement Learning Forecasting Agent.
Forecasts price and return at n+0, n+1, n+2, n+3, n+4, and n+5 using sentiment and price features only.
Employs XGBoost with Bayesian-style regularization and an RL-inspired adaptive policy.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

from src.config import config

logger = logging.getLogger(__name__)


class MultiHorizonPredictor:
    """
    Predictive engine that forecasts price and return at horizons n+0 through n+5.
    Uses dedicated horizon-specific models with feature importance tracking.
    """
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.horizons = config.FORECAST_HORIZONS  # [0, 1, 2, 3, 4, 5]
        self.models: Dict[int, Any] = {}
        self.scalers: Dict[int, RobustScaler] = {}
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y_df: pd.DataFrame) -> "MultiHorizonPredictor":
        """
        Train individual forecasting models for each horizon h in [0..5].
        X: Feature matrix (sentiment and price features only)
        y_df: DataFrame containing target_ret_n0 ... target_ret_n5
        """
        self.feature_names = list(X.columns)
        for h in self.horizons:
            target_col = f"target_ret_n{h}"
            y_h = y_df[target_col].values

            # Scale features robustly against outliers
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X.values)
            self.scalers[h] = scaler

            if self.model_type == "xgboost":
                # Regularized XGBoost Regressor to prevent overfitting on financial time series
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.5,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_scaled, y_h)
            elif self.model_type == "ridge":
                model = Ridge(alpha=10.0, random_state=42)
                model.fit(X_scaled, y_h)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            self.models[h] = model

        return self

    def predict(self, X: pd.DataFrame, current_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predicted returns and price levels for all horizons n+0 .. n+5.
        current_prices: array of current period n close prices (P_n).
        Returns dictionary with predicted returns and reconstructed predicted prices.
        """
        preds = {}
        for h in self.horizons:
            model = self.models[h]
            scaler = self.scalers[h]
            X_scaled = scaler.transform(X.values)
            pred_return = model.predict(X_scaled)
            preds[f"pred_ret_n{h}"] = pred_return

            if h == 0:
                # n+0 price is nowcasted close: P_n
                preds[f"pred_price_n{h}"] = current_prices * np.exp(pred_return)
            else:
                # n+k price is P_n * exp(pred_ret_nh)
                preds[f"pred_price_n{h}"] = current_prices * np.exp(pred_return)

        return preds

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract and aggregate feature importances across all horizon models."""
        importance_dict = {"feature": self.feature_names}
        for h in self.horizons:
            model = self.models[h]
            if hasattr(model, "feature_importances_"):
                importance_dict[f"horizon_n{h}"] = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance_dict[f"horizon_n{h}"] = np.abs(model.coef_)
            else:
                importance_dict[f"horizon_n{h}"] = np.zeros(len(self.feature_names))

        df_imp = pd.DataFrame(importance_dict)
        # Average importance across all horizons
        horizon_cols = [f"horizon_n{h}" for h in self.horizons]
        df_imp["mean_importance"] = df_imp[horizon_cols].mean(axis=1)
        return df_imp.sort_values("mean_importance", ascending=False).reset_index(drop=True)


class ReinforcementLearningPolicy:
    """
    Q-learning / Policy Gradient inspired Execution & Allocation Agent.
    Dynamically maps multi-horizon predicted return vectors [r_n0, r_n1, ..., r_n5]
    into continuous risk-budgeted portfolio allocations [-1.0, 1.0].
    """
    def __init__(self, confidence_threshold: float = 0.0005, max_leverage: float = 1.0):
        self.threshold = confidence_threshold
        self.max_leverage = max_leverage

    def compute_action(self, pred_returns_matrix: np.ndarray) -> np.ndarray:
        """
        Computes dynamic allocation signal based on horizon consensus and velocity.
        pred_returns_matrix: shape (N, 6) for horizons [0, 1, 2, 3, 4, 5]
        """
        # Focus on actionable forward horizons n+1 and n+2 with dead-band
        horizon_weights = np.array([0.0, 0.75, 0.25, 0.0, 0.0, 0.0])
        composite_signal = np.dot(pred_returns_matrix, horizon_weights)

        # Non-linear position sizing
        signals = np.zeros_like(composite_signal)
        for i, sig in enumerate(composite_signal):
            if sig > self.threshold:
                # Long allocation
                signals[i] = min(self.max_leverage, 1.0)
            elif sig < -self.threshold:
                # Short allocation
                signals[i] = max(-self.max_leverage, -1.0)
            else:
                signals[i] = 0.0  # Flat / Cash preservation

        return signals
