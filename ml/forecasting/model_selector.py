"""
EnerVision AI - Model Selector
Trains all enabled models, compares metrics, selects the best automatically.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ml.forecasting.base_model import BaseModel
from ml.forecasting.linear_model import LinearRegressionModel
from ml.forecasting.random_forest_model import RandomForestModel
from ml.forecasting.xgboost_model import XGBoostModel
from ml.forecasting.statistical_models import ARIMAModel, SARIMAModel, SARIMAXModel
from ml.utils.config_loader import config
from ml.utils.helpers import metrics_table, time_split
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


class ModelSelector:
    """
    Orchestrates training and evaluation of all forecasting models,
    then selects the best based on RMSE (primary metric).

    Usage::

        selector = ModelSelector()
        best_model, results = selector.run(X_train, y_train, X_test, y_test)
    """

    # ML models that accept feature DataFrames
    _ML_MODELS = {
        "linear_regression": LinearRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
    }

    # Statistical models that take the raw target series
    _STAT_MODELS = {
        "arima": ARIMAModel,
        "sarima": SARIMAModel,
        "sarimax": SARIMAXModel,
    }

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        self._fc_cfg = self._cfg.forecasting
        self._target_col: str = self._cfg.data.target_column
        self._primary_metric: str = "rmse"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[BaseModel, Dict[str, Dict]]:
        """
        Train all enabled models on (X_train, y_train) and evaluate on
        (X_test, y_test).

        Returns:
            best_model: The fitted model with the lowest RMSE.
            results:    Dict[model_name → metrics_dict].
        """
        with PipelineLogger(logger, "ModelSelector.run"):
            results: Dict[str, dict] = {}
            fitted_models: Dict[str, BaseModel] = {}

            # --- ML models ---
            for key, ModelClass in self._ML_MODELS.items():
                if not self._is_enabled(key):
                    continue
                try:
                    model = ModelClass()
                    model.fit(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)
                    results[model.name] = metrics
                    fitted_models[model.name] = model
                except Exception as exc:
                    logger.error("Model '%s' failed: %s", key, exc, exc_info=True)

            # --- Statistical models (use target series + exog columns) ---
            # For stat models we pass X (which may include exog cols) and y separately
            stat_X_train = X_train  # includes exog cols for SARIMAX
            stat_X_test = X_test

            for key, ModelClass in self._STAT_MODELS.items():
                if not self._is_enabled(key):
                    continue
                try:
                    # Limit stat model training and test size for performance
                    max_stat_rows = 5000
                    max_test_rows = 2000
                    y_tr = y_train.iloc[-max_stat_rows:] if len(y_train) > max_stat_rows else y_train
                    X_tr = stat_X_train.iloc[-max_stat_rows:] if len(stat_X_train) > max_stat_rows else stat_X_train
                    y_te = y_test.iloc[:max_test_rows] if len(y_test) > max_test_rows else y_test
                    X_te = stat_X_test.iloc[:max_test_rows] if len(stat_X_test) > max_test_rows else stat_X_test

                    model = ModelClass()
                    model.fit(X_tr, y_tr)
                    metrics = model.evaluate(X_te, y_te)
                    results[model.name] = metrics
                    fitted_models[model.name] = model
                except Exception as exc:
                    logger.error("Model '%s' failed: %s", key, exc, exc_info=True)

            if not results:
                raise RuntimeError("No models trained successfully.")

            # Print comparison table
            logger.info("\n%s", metrics_table(results))

            # Select best by RMSE
            best_name = min(results, key=lambda k: results[k][self._primary_metric])
            best_model = fitted_models[best_name]
            logger.info(
                "🏆 Best model: %s (RMSE=%.2f)", best_name, results[best_name]["rmse"]
            )

            return best_model, results

    def _is_enabled(self, key: str) -> bool:
        try:
            return bool(self._fc_cfg.models[key]["enabled"])
        except (KeyError, AttributeError):
            return True  # default to enabled
