"""
EnerVision AI - Base Model Interface
Abstract base class for all forecasting models.
"""

import abc
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ml.utils.helpers import compute_metrics
from ml.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(abc.ABC):
    """
    Abstract interface every forecasting model must implement.

    Subclasses must implement:
        fit(X_train, y_train)
        predict(X)
        name  (property)
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._train_metrics: Dict = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModel":
        """Train the model; return self for chaining."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions as a 1-D numpy array."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Generate predictions on test set and compute RMSE/MAE/MAPE."""
        if not self._is_fitted:
            raise RuntimeError(f"Model '{self.name}' is not fitted yet.")
        preds = self.predict(X_test)
        metrics = compute_metrics(y_test.values, preds)
        logger.info(
            "[%s] Eval → RMSE=%.2f | MAE=%.2f | MAPE=%.2f%%",
            self.name, metrics["rmse"], metrics["mae"], metrics["mape"],
        )
        return metrics

    def fit_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple["BaseModel", Dict[str, float]]:
        """Convenience: fit then evaluate, return (self, metrics)."""
        self.fit(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        return self, metrics

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"<{self.name} [{status}]>"
