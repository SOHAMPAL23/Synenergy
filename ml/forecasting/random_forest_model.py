"""
EnerVision AI - Random Forest Regressor
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.forecasting.base_model import BaseModel
from ml.utils.config_loader import config
from ml.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """
    Ensemble of decision trees with bootstrap aggregation.
    Robust to outliers and naturally handles non-linear interactions.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self._cfg = cfg or config
        rf_cfg = self._cfg.forecasting.models.random_forest
        self._model = RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 10),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=self._cfg.forecasting.random_state,
        )

    @property
    def name(self) -> str:
        return "RandomForest"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RandomForestModel":
        logger.info("[%s] Fitting on %d samples, %d features.", self.name, *X_train.shape)
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("[%s] Fitting complete.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
