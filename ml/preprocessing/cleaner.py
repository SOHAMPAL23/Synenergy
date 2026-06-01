"""
EnerVision AI - Data Cleaner
Handles missing values, duplicates, and basic outlier removal
before feature engineering.
"""

from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from ml.utils.config_loader import config
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans a raw energy DataFrame by:

    1. Removing duplicate index entries.
    2. Filling missing values via interpolation / forward-fill / back-fill.
    3. Detecting and capping statistical outliers in the target column.

    Usage::

        cleaner = DataCleaner()
        df_clean = cleaner.clean(df)
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        pp_cfg = self._cfg.preprocessing
        self._missing_strategy: str = pp_cfg.missing_value_strategy
        self._outlier_method: str = pp_cfg.outlier_method
        self._outlier_threshold: float = float(pp_cfg.outlier_threshold)
        self._iqr_multiplier: float = float(pp_cfg.iqr_multiplier)
        self._target_col: str = self._cfg.data.target_column

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a clean DataFrame."""
        with PipelineLogger(logger, "DataCleaner.clean"):
            df = df.copy()
            initial_rows = len(df)

            df = self._remove_duplicates(df)
            df = self._handle_missing(df)
            df = self._handle_outliers(df)

            logger.info(
                "Cleaning complete: %d → %d rows (removed %d)",
                initial_rows, len(df), initial_rows - len(df),
            )
            return df

    # ------------------------------------------------------------------
    # Step 1 – Duplicates
    # ------------------------------------------------------------------

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_dupes = df.index.duplicated().sum()
        if n_dupes:
            logger.warning("Removing %d duplicate index entries.", n_dupes)
            df = df[~df.index.duplicated(keep="first")]
        else:
            logger.info("No duplicate indices found.")
        return df

    # ------------------------------------------------------------------
    # Step 2 – Missing values
    # ------------------------------------------------------------------

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        total_missing = df.isnull().sum().sum()
        logger.info("Missing values before imputation: %d", total_missing)

        strategy = self._missing_strategy.lower()

        if strategy == "interpolate":
            # Time-aware linear interpolation; cap gaps > 6 h then ffill/bfill
            df = df.interpolate(method="time", limit=6)
            df = df.ffill().bfill()

        elif strategy == "ffill":
            df = df.ffill().bfill()

        elif strategy == "bfill":
            df = df.bfill().ffill()

        elif strategy == "drop":
            before = len(df)
            df = df.dropna(subset=[self._target_col])
            logger.info("Dropped %d rows with null target.", before - len(df))

        else:
            logger.warning(
                "Unknown missing strategy '%s'; falling back to interpolate.", strategy
            )
            df = df.interpolate(method="time", limit=6).ffill().bfill()

        remaining = df.isnull().sum().sum()
        logger.info("Missing values after imputation: %d", remaining)
        return df

    # ------------------------------------------------------------------
    # Step 3 – Outliers
    # ------------------------------------------------------------------

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._target_col not in df.columns:
            return df

        method = self._outlier_method.lower()
        col = df[self._target_col]
        original_null = col.isnull().sum()

        if method == "zscore":
            mask = self._zscore_mask(col)
        elif method == "iqr":
            mask = self._iqr_mask(col)
        else:
            logger.warning("Unknown outlier method '%s'; skipping.", method)
            return df

        n_outliers = mask.sum()
        logger.info(
            "Outlier detection (%s): %d outliers detected in '%s'.",
            method, n_outliers, self._target_col,
        )

        # Cap outliers at boundary values instead of dropping rows
        lower, upper = self._get_bounds(col, method)
        df[self._target_col] = col.clip(lower=lower, upper=upper)
        return df

    def _zscore_mask(self, col: pd.Series) -> pd.Series:
        z = np.abs(stats.zscore(col.dropna()))
        idx = col.dropna().index
        mask = pd.Series(False, index=col.index)
        mask[idx] = z > self._outlier_threshold
        return mask

    def _iqr_mask(self, col: pd.Series) -> pd.Series:
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - self._iqr_multiplier * IQR
        upper = Q3 + self._iqr_multiplier * IQR
        return (col < lower) | (col > upper)

    def _get_bounds(self, col: pd.Series, method: str):
        if method == "zscore":
            mu, sigma = col.mean(), col.std()
            return (
                mu - self._outlier_threshold * sigma,
                mu + self._outlier_threshold * sigma,
            )
        else:  # iqr
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            return (
                Q1 - self._iqr_multiplier * IQR,
                Q3 + self._iqr_multiplier * IQR,
            )
