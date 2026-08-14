"""
EnerVision AI - LightGBM Regressor with Optuna Tuning
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error

from ml.forecasting.base_model import BaseModel
from ml.utils.config_loader import config
from ml.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBM model, highly optimized for speed and accuracy.
    Includes built-in Optuna hyperparameter optimization.
    """

    def __init__(self, cfg=None) -> None:
        super().__init__()
        self._cfg = cfg or config
        
        # Default empty dict if lightgbm config is missing
        try:
            self.lgb_cfg = self._cfg.forecasting.models.lightgbm
        except AttributeError:
            self.lgb_cfg = {}

        self._model = lgb.LGBMRegressor(
            n_estimators=self.lgb_cfg.get("n_estimators", 200) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "n_estimators", 200),
            learning_rate=self.lgb_cfg.get("learning_rate", 0.05) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "learning_rate", 0.05),
            max_depth=self.lgb_cfg.get("max_depth", 6) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "max_depth", 6),
            num_leaves=self.lgb_cfg.get("num_leaves", 31) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "num_leaves", 31),
            n_jobs=self.lgb_cfg.get("n_jobs", -1) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "n_jobs", -1),
            random_state=self._cfg.forecasting.random_state,
            verbose=-1,
        )
        self._feature_names = None

    @property
    def name(self) -> str:
        return "LightGBM"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LightGBMModel":
        logger.info("[%s] Fitting on %d samples, %d features.", self.name, *X_train.shape)
        self._feature_names = list(X_train.columns)
        
        tuning = self.lgb_cfg.get("tuning", True) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "tuning", True)

        if tuning:
            logger.info("[%s] Optuna hyperparameter tuning enabled.", self.name)
            
            def objective(trial):
                param = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "random_state": self._cfg.forecasting.random_state,
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "n_jobs": self.lgb_cfg.get("n_jobs", -1) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "n_jobs", -1),
                }
                
                # Simple train/val split for optuna
                val_size = min(100, max(5, int(len(X_train) * 0.1)))
                X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
                y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
                
                model = lgb.LGBMRegressor(**param)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )
                preds = model.predict(X_val)
                rmse = mean_squared_error(y_val, preds, squared=False)
                return rmse
                
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self._cfg.forecasting.random_state))
            
            n_trials = self.lgb_cfg.get("optuna_trials", 15) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "optuna_trials", 15)
            study.optimize(objective, n_trials=n_trials)
            
            logger.info("[%s] Best Optuna params: %s", self.name, study.best_params)
            
            best_params = study.best_params
            best_params["random_state"] = self._cfg.forecasting.random_state
            best_params["n_jobs"] = self.lgb_cfg.get("n_jobs", -1) if isinstance(self.lgb_cfg, dict) else getattr(self.lgb_cfg, "n_jobs", -1)
            best_params["verbose"] = -1
            self._model = lgb.LGBMRegressor(**best_params)

        # Final fit on all data or with early stopping if large enough
        if len(X_train) >= 20:
            val_size = min(100, max(5, int(len(X_train) * 0.1)))
            X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
            y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
            
            self._model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
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
