"""
EnerVision AI - Linear Regression Model
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ml.forecasting.base_model import BaseModel
from ml.utils.logger import get_logger

logger = get_logger(__name__)


class LinearRegressionModel(BaseModel):
    """
    Ordinary Least Squares Linear Regression with Z-score feature scaling.
    Useful as a fast baseline and for detecting linear trends.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._model = LinearRegression()

    @property
    def name(self) -> str:
        return "LinearRegression"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LinearRegressionModel":
        logger.info("[%s] Fitting on %d samples, %d features.", self.name, *X_train.shape)
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        self._is_fitted = True
        logger.info("[%s] Fitting complete.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return absolute coefficient values as proxy for importance."""
        return np.abs(self._model.coef_)
