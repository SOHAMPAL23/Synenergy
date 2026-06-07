"""
EnerVision AI - Feature Engineering Pipeline
Creates time features, lag features, rolling statistics, and holiday flags.
"""

from typing import List

import numpy as np
import pandas as pd

from ml.utils.config_loader import config
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)

# Optional: holiday support
try:
    import holidays as holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False
    logger.warning("'holidays' package not installed; holiday flag will be zeros.")


class FeatureEngineer:
    """
    Transforms a clean energy DataFrame into a feature-rich ML-ready DataFrame.

    Features created
    ----------------
    Time-based:
        hour, day, month, week, quarter, season, is_weekend, is_holiday

    Lag features (on target):
        load_t_1, load_t_24, load_t_168

    Rolling statistics (on target):
        rolling_mean_7, rolling_mean_30, rolling_std_7

    Usage::

        fe = FeatureEngineer()
        df_features = fe.transform(df_clean)
    """

    # Map month → meteorological season (Northern Hemisphere)
    _SEASON_MAP = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    _SEASON_ENCODE = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        fe_cfg = self._cfg.feature_engineering
        
        # Enforce highly predictive lags and windows, merging with configured ones
        config_lags = list(fe_cfg.lag_hours)
        config_windows = list(fe_cfg.rolling_windows)
        
        # We need lag+1 of these to compute diffs. Lags: 1, 2, 3, 24, 25, 48, 49, 168, 169
        predictive_lags = {1, 2, 3, 24, 25, 48, 49, 168, 169}
        self._lag_hours: List[int] = sorted(list(predictive_lags.union(config_lags)))
        
        # Rolling windows for short and long term baselines
        predictive_windows = {7, 24, 30, 168}
        self._rolling_windows: List[int] = sorted(list(predictive_windows.union(config_windows)))
        
        self._include_holidays: bool = bool(fe_cfg.include_holidays)
        self._holiday_country: str = str(fe_cfg.holiday_country)
        self._target_col: str = self._cfg.data.target_column
        self._exog_cols: List[str] = list(self._cfg.data.get("exog_columns", []))
        
        # Cache country holidays to speed up recursive forecasting
        self._country_holidays = None
        if self._include_holidays and _HOLIDAYS_AVAILABLE:
            try:
                self._country_holidays = holidays_lib.country_holidays(self._holiday_country)
                logger.info("Holiday flag initialized for country '%s'.", self._holiday_country)
            except Exception as exc:
                logger.warning("Holiday initialization failed: %s. Using zeros.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all feature columns in-place and return the enriched DataFrame.
        NaN rows introduced by lagging are dropped at the end.
        """
        with PipelineLogger(logger, "FeatureEngineer.transform"):
            df = df.copy()
            df = self._add_time_features(df)
            df = self._add_lag_features(df)
            df = self._add_rolling_features(df)

            before = len(df)
            df = df.dropna()
            logger.info(
                "Dropped %d NaN rows after feature creation (%d → %d).",
                before - len(df), before, len(df),
            )
            return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return list of engineered feature column names (excludes target)."""
        return [c for c in df.columns if c != self._target_col]

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index

        # Basic calendar
        df["hour"] = idx.hour
        df["day"] = idx.day
        df["month"] = idx.month
        df["week"] = idx.isocalendar().week.astype(int)
        df["quarter"] = idx.quarter
        df["day_of_week"] = idx.dayofweek          # 0=Mon … 6=Sun
        df["day_of_year"] = idx.dayofyear

        # Season (encoded numerically)
        df["season"] = df["month"].map(self._SEASON_MAP).map(self._SEASON_ENCODE)

        # Weekend flag
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

        # Cyclical encoding for hour and month to preserve periodicity
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Holiday flag
        df["is_holiday"] = self._compute_holiday_flag(idx)

        # Time interaction features
        df["hour_is_weekend"] = df["hour"] * df["is_weekend"]

        logger.info("Time features added: hour, day, month, week, quarter, season, is_weekend, is_holiday + cyclicals")
        return df

    def _compute_holiday_flag(self, idx: pd.DatetimeIndex) -> pd.Series:
        if self._country_holidays is None:
            return pd.Series(0, index=idx)
        try:
            flag = pd.Series(
                [1 if d.date() in self._country_holidays else 0 for d in idx],
                index=idx,
            )
            return flag
        except Exception as exc:
            logger.warning("Holiday computation failed: %s. Using zeros.", exc)
            return pd.Series(0, index=idx)

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        target = df[self._target_col]
        for lag in self._lag_hours:
            col_name = f"load_t_{lag}"
            df[col_name] = target.shift(lag)
            logger.debug("Lag feature created: %s", col_name)

        # Diff features (hourly daily, and weekly acceleration/deceleration)
        if "load_t_1" in df.columns and "load_t_2" in df.columns:
            df["load_diff_1"] = df["load_t_1"] - df["load_t_2"]
        if "load_t_24" in df.columns and "load_t_25" in df.columns:
            df["load_diff_24"] = df["load_t_24"] - df["load_t_25"]
        if "load_t_168" in df.columns and "load_t_169" in df.columns:
            df["load_diff_168"] = df["load_t_168"] - df["load_t_169"]

        # Add lag features for exogenous columns to make models sharper
        for exog_col in self._exog_cols:
            if exog_col in df.columns:
                for lag in self._lag_hours:
                    col_name = f"{exog_col}_t_{lag}"
                    df[col_name] = df[exog_col].shift(lag)
                    logger.debug("Exog lag feature created: %s", col_name)

        logger.info("Lag features and difference momentum features added.")
        return df

    # ------------------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        target = df[self._target_col]
        for window in self._rolling_windows:
            df[f"rolling_mean_{window}"] = (
                target.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"rolling_std_{window}"] = (
                target.shift(1).rolling(window=window, min_periods=1).std()
            )

        # Add rolling features for exogenous columns to make models sharper
        for exog_col in self._exog_cols:
            if exog_col in df.columns:
                for window in self._rolling_windows:
                    df[f"{exog_col}_rolling_mean_{window}"] = (
                        df[exog_col].shift(1).rolling(window=window, min_periods=1).mean()
                    )
                    df[f"{exog_col}_rolling_std_{window}"] = (
                        df[exog_col].shift(1).rolling(window=window, min_periods=1).std()
                    )

        logger.info(
            "Rolling features added: windows=%s", self._rolling_windows
        )
        return df
