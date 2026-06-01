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
        
        from sklearn.model_selection import RandomizedSearchCV
        rf_cfg = self._cfg.forecasting.models.random_forest
        if rf_cfg.get("tuning", True):
            logger.info("[%s] Hyperparameter fine-tuning enabled. Running RandomizedSearchCV...", self.name)
            default_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
            raw_grid = rf_cfg.get("param_grid", default_grid)
            param_grid = {k: list(v) for k, v in raw_grid.items()}
            
            search = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=self._cfg.forecasting.random_state),
                param_distributions=param_grid,
                n_iter=10,
                scoring="neg_root_mean_squared_error",
                cv=3,
                random_state=self._cfg.forecasting.random_state,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            logger.info("[%s] Best parameters found: %s", self.name, search.best_params_)
            self._model = search.best_estimator_
        else:
            self._model.fit(X_train, y_train)
            
        self._is_fitted = True
        logger.info("[%s] Fitting complete.", self.name)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
