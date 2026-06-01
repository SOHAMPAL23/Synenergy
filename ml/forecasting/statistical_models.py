"""
EnerVision AI - ARIMA, SARIMA, SARIMAX Models
Statistical time-series models via statsmodels.
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ml.forecasting.base_model import BaseModel
from ml.utils.config_loader import config
from ml.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

class ARIMAModel(BaseModel):
    """
    ARIMA(p,d,q) – no seasonal component, no exogenous regressors.
    Fitted on the univariate target series.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self._cfg = cfg or config
        arima_cfg = self._cfg.forecasting.models.arima
        self._order: tuple = tuple(arima_cfg.get("order", [2, 1, 2]))
        self._max_iter: int = int(arima_cfg.get("max_iter", 50))
        self._result = None
        self._train_index = None

    @property
    def name(self) -> str:
        return "ARIMA"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ARIMAModel":
        logger.info("[%s] Fitting ARIMA%s on %d samples.", self.name, self._order, len(y_train))
        model = SARIMAX(
            y_train,
            order=self._order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, maxiter=self._max_iter)
        self._train_index = y_train.index
        self._is_fitted = True
        logger.info("[%s] AIC=%.2f", self.name, self._result.aic)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        forecast = self._result.forecast(steps=n)
        return np.asarray(forecast)


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

class SARIMAModel(BaseModel):
    """
    SARIMA(p,d,q)(P,D,Q,s) – seasonal ARIMA, no exogenous regressors.
    Default seasonal period = 24 (hourly data, daily seasonality).
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self._cfg = cfg or config
        sarima_cfg = self._cfg.forecasting.models.sarima
        self._order: tuple = tuple(sarima_cfg.get("order", [1, 1, 1]))
        self._seasonal_order: tuple = tuple(sarima_cfg.get("seasonal_order", [1, 1, 1, 24]))
        self._max_iter: int = int(sarima_cfg.get("max_iter", 50))
        self._result = None

    @property
    def name(self) -> str:
        return "SARIMA"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SARIMAModel":
        logger.info(
            "[%s] Fitting SARIMA%s x %s on %d samples.",
            self.name, self._order, self._seasonal_order, len(y_train),
        )
        model = SARIMAX(
            y_train,
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, maxiter=self._max_iter)
        self._is_fitted = True
        logger.info("[%s] AIC=%.2f", self.name, self._result.aic)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        forecast = self._result.forecast(steps=n)
        return np.asarray(forecast)


# ---------------------------------------------------------------------------
# SARIMAX
# ---------------------------------------------------------------------------

class SARIMAXModel(BaseModel):
    """
    SARIMAX – SARIMA with exogenous regressors (solar & wind generation).
    Leverages external drivers to sharpen seasonal load forecasts.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self._cfg = cfg or config
        sarimax_cfg = self._cfg.forecasting.models.sarimax
        self._order: tuple = tuple(sarimax_cfg.get("order", [1, 1, 1]))
        self._seasonal_order: tuple = tuple(sarimax_cfg.get("seasonal_order", [1, 1, 1, 24]))
        self._max_iter: int = int(sarimax_cfg.get("max_iter", 50))
        self._exog_cols: List[str] = list(self._cfg.data.exog_columns or [])
        self._result = None

    @property
    def name(self) -> str:
        return "SARIMAX"

    def _get_exog(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        available = [c for c in self._exog_cols if c in df.columns]
        if not available:
            return None
        return df[available]

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SARIMAXModel":
        exog = self._get_exog(X_train)
        logger.info(
            "[%s] Fitting SARIMAX%s x %s | exog=%s | samples=%d.",
            self.name, self._order, self._seasonal_order,
            list(exog.columns) if exog is not None else None,
            len(y_train),
        )
        model = SARIMAX(
            y_train,
            exog=exog,
            order=self._order,
            seasonal_order=self._seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, maxiter=self._max_iter)
        self._is_fitted = True
        logger.info("[%s] AIC=%.2f", self.name, self._result.aic)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        exog = self._get_exog(X)
        n = len(X)
        forecast = self._result.forecast(steps=n, exog=exog)
        return np.asarray(forecast)
