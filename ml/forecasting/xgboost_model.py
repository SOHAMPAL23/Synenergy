"""
EnerVision AI - XGBoost Regressor
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from ml.forecasting.base_model import BaseModel
from ml.utils.config_loader import config
from ml.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    Gradient-boosted tree ensemble via XGBoost.
    Typically the highest-accuracy model in the suite for tabular data.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self._cfg = cfg or config
        self.xgb_cfg = self._cfg.forecasting.models.xgboost
        self._model = xgb.XGBRegressor(
            n_estimators=self.xgb_cfg.get("n_estimators", 200),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            max_depth=xgb_cfg.get("max_depth", 6),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            n_jobs=xgb_cfg.get("n_jobs", -1),
            verbosity=xgb_cfg.get("verbosity", 0),
            random_state=self._cfg.forecasting.random_state,
            eval_metric="rmse",
        )
        self._feature_names = None

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "XGBoostModel":
        logger.info("[%s] Fitting on %d samples, %d features.", self.name, *X_train.shape)
        self._feature_names = list(X_train.columns)

        if self.xgb_cfg.get("tuning", False) and "param_grid" in self.xgb_cfg:
            logger.info("[%s] Hyperparameter fine-tuning enabled. Running RandomizedSearchCV...", self.name)
            param_grid = {k: list(v) for k, v in self.xgb_cfg["param_grid"].items()}
            
            search = RandomizedSearchCV(
                estimator=xgb.XGBRegressor(
                    n_jobs=self.xgb_cfg.get("n_jobs", -1),
                    verbosity=self.xgb_cfg.get("verbosity", 0),
                    random_state=self._cfg.forecasting.random_state,
                    eval_metric="rmse"
                ),
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

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )
        self._is_fitted = True
        try:
            best_iter = self._model.best_iteration
        except AttributeError:
            best_iter = self._model.n_estimators
        logger.info("[%s] Fitting complete. Iterations: %s", self.name, best_iter)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    @property
    def booster(self):
        return self._model.get_booster()
