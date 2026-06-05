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
            learning_rate=self.xgb_cfg.get("learning_rate", 0.05),
            max_depth=self.xgb_cfg.get("max_depth", 6),
            subsample=self.xgb_cfg.get("subsample", 0.8),
            colsample_bytree=self.xgb_cfg.get("colsample_bytree", 0.8),
            n_jobs=self.xgb_cfg.get("n_jobs", -1),
            verbosity=self.xgb_cfg.get("verbosity", 0),
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

        if self.xgb_cfg.get("tuning", True):
            logger.info("[%s] Hyperparameter fine-tuning enabled. Running RandomizedSearchCV...", self.name)
            
            default_grid = {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9]
            }
            raw_grid = self.xgb_cfg.get("param_grid", default_grid)
            param_grid = {k: list(v) for k, v in raw_grid.items()}
            
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

        # Split validation set for early stopping (last 10% of training data, min 5 rows, max 100 rows)
        if len(X_train) >= 20:
            val_size = min(100, max(5, int(len(X_train) * 0.1)))
            X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
            y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
            
            # Setup early stopping
            self._model.set_params(early_stopping_rounds=15)
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            # Fallback for very small datasets (e.g. unit tests)
            self._model.set_params(early_stopping_rounds=None)
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
