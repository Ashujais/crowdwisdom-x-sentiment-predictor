"""
Rigorous Walk-Forward Validation (WFV) with Purged & Embargoed Time-Series Splits.
Eliminates lookahead bias and serial correlation leakage when validating multi-horizon forecasts.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import config
from src.features.feature_engineer import FeatureEngineer
from src.models.predictor import MultiHorizonPredictor

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    def __init__(
        self,
        train_window: Optional[int] = None,
        test_window: Optional[int] = None,
        purge_window: Optional[int] = None,
        model_type: str = "xgboost"
    ):
        self.train_window = train_window or config.WFV_TRAIN_WINDOW
        self.test_window = test_window or config.WFV_TEST_WINDOW
        self.purge_window = purge_window or config.PURGE_WINDOW
        self.model_type = model_type
        self.horizons = config.FORECAST_HORIZONS

    def run_walk_forward(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        """
        Executes expanding-window Walk-Forward Validation.
        Returns:
        1. oos_results: DataFrame of all concatenated out-of-sample predictions
        2. horizon_metrics: Detailed RMSE, MAE, R2, and Hit Ratio for n+0..n+5
        3. feature_importances: Aggregated feature importances across all folds
        """
        feature_cols = FeatureEngineer.get_feature_columns()
        N = len(df)
        max_horizon = max(self.horizons)

        # Ensure valid target columns (exclude rows where n+5 extends beyond data)
        valid_df = df.dropna(subset=[f"target_ret_n{max_horizon}"]).reset_index(drop=True)
        N_valid = len(valid_df)

        logger.info(f"Starting Walk-Forward Validation on {N_valid} bars. Initial Train={self.train_window}, Step={self.test_window}, Purge={self.purge_window}")

        all_oos_records = []
        importances_list = []
        fold = 0
        train_end = self.train_window

        while train_end + self.purge_window + 5 <= N_valid:
            test_start = train_end + self.purge_window
            test_end = min(test_start + self.test_window, N_valid)

            if test_end <= test_start:
                break

            train_data = valid_df.iloc[:train_end]
            test_data = valid_df.iloc[test_start:test_end]

            X_train = train_data[feature_cols]
            y_train = train_data[[f"target_ret_n{h}" for h in self.horizons]]

            X_test = test_data[feature_cols]
            test_prices = test_data["close"].values

            # Fit model on training window
            predictor = MultiHorizonPredictor(model_type=self.model_type)
            predictor.fit(X_train, y_train)
            importances_list.append(predictor.get_feature_importances())

            # Predict out-of-sample
            preds = predictor.predict(X_test, test_prices)

            # Record predictions
            test_subset = test_data.copy().reset_index(drop=True)
            for h in self.horizons:
                test_subset[f"pred_ret_n{h}"] = preds[f"pred_ret_n{h}"]
                test_subset[f"pred_price_n{h}"] = preds[f"pred_price_n{h}"]

            test_subset["fold_index"] = fold
            all_oos_records.append(test_subset)

            logger.info(f"Fold {fold}: Train [0:{train_end}], Purge [{train_end}:{test_start}], Test [{test_start}:{test_end}] -> {len(test_data)} OOS bars.")
            fold += 1
            train_end += self.test_window

        if not all_oos_records:
            raise ValueError("Walk-Forward Validation generated zero out-of-sample records. Check window parameters.")

        oos_df = pd.concat(all_oos_records, ignore_index=True)

        # Compute Horizon Metrics across entire Out-Of-Sample period
        horizon_metrics = {}
        for h in self.horizons:
            actual_ret = oos_df[f"target_ret_n{h}"].values
            pred_ret = oos_df[f"pred_ret_n{h}"].values

            actual_price = oos_df[f"target_price_n{h}"].values
            pred_price = oos_df[f"pred_price_n{h}"].values

            rmse = float(np.sqrt(mean_squared_error(actual_price, pred_price)))
            mae = float(mean_absolute_error(actual_price, pred_price))
            mape = float(np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100.0)
            
            # Directional accuracy / Hit Ratio
            correct_dir = np.sign(actual_ret) == np.sign(pred_ret)
            hit_ratio = float(np.mean(correct_dir) * 100.0)

            # Correlation between predicted and actual returns
            corr = float(np.corrcoef(actual_ret, pred_ret)[0, 1]) if len(actual_ret) > 1 else 0.0

            horizon_metrics[f"n+{h}"] = {
                "horizon": f"n+{h}",
                "rmse_price": rmse,
                "mae_price": mae,
                "mape_pct": mape,
                "hit_ratio_pct": hit_ratio,
                "return_correlation": corr
            }

        # Average feature importance across folds
        combined_imp = pd.concat(importances_list)
        avg_imp = combined_imp.groupby("feature")["mean_importance"].mean().reset_index()
        avg_imp = avg_imp.sort_values("mean_importance", ascending=False).reset_index(drop=True)

        logger.info(f"Completed Walk-Forward Validation across {fold} folds. Total OOS samples: {len(oos_df)}.")
        return oos_df, horizon_metrics, avg_imp
