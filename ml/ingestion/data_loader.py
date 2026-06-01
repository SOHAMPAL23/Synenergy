"""
EnerVision AI - Data Loader
Loads and performs initial validation on energy time-series CSV datasets.
"""

import os
from typing import List, Optional

import pandas as pd

from ml.utils.config_loader import config
from ml.utils.logger import get_logger, PipelineLogger

logger = get_logger(__name__)


class DataLoader:
    """
    Responsible for loading raw CSV data from disk, selecting
    relevant columns, and performing initial type coercion.

    Usage::

        loader = DataLoader()
        df = loader.load()
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        data_cfg = self._cfg.data
        self._raw_dir: str = data_cfg.raw_dir
        self._primary_file: str = data_cfg.primary_file
        self._timestamp_col: str = data_cfg.timestamp_column
        self._target_col: str = data_cfg.target_column
        self._exog_cols: List[str] = list(data_cfg.exog_columns or [])
        self._max_rows: Optional[int] = data_cfg.get("max_rows")
        self._start_date: Optional[str] = data_cfg.get("start_date")
        self._end_date: Optional[str] = data_cfg.get("end_date")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Load the primary dataset, parse timestamps, select key columns,
        and apply optional date range filtering.

        Returns:
            pd.DataFrame with a DatetimeIndex and selected feature columns.
        """
        with PipelineLogger(logger, "DataLoader.load"):
            path = os.path.join(self._raw_dir, self._primary_file)
            logger.info("Loading dataset: %s", path)

            if not os.path.isfile(path):
                raise FileNotFoundError(f"Dataset not found: {path}")

            usecols = self._build_usecols()
            df = pd.read_csv(
                path,
                usecols=usecols,
                nrows=self._max_rows if self._max_rows else None,
                low_memory=False,
            )
            logger.info("Raw shape: %s", df.shape)

            df = self._parse_timestamps(df)
            df = self._filter_date_range(df)
            df = self._coerce_numeric(df)

            logger.info("Loaded shape after filtering: %s", df.shape)
            return df

    def load_supplementary(self, filename: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load any supplementary CSV from the data directory."""
        path = os.path.join(self._raw_dir, filename)
        logger.info("Loading supplementary file: %s", path)
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        logger.info("Supplementary shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_usecols(self) -> List[str]:
        """Determine which columns to load from the CSV."""
        cols = [self._timestamp_col, self._target_col]
        for col in self._exog_cols:
            if col not in cols:
                cols.append(col)
        return cols

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse UTC timestamp string → DatetimeIndex."""
        df[self._timestamp_col] = pd.to_datetime(
            df[self._timestamp_col], utc=True, errors="coerce"
        )
        df = df.dropna(subset=[self._timestamp_col])
        df = df.set_index(self._timestamp_col)
        df.index.name = "timestamp"
        df = df.sort_index()
        logger.info(
            "Date range: %s → %s",
            df.index.min().isoformat(),
            df.index.max().isoformat(),
        )
        return df

    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slice to configured start / end dates."""
        if self._start_date:
            df = df[df.index >= pd.Timestamp(self._start_date, tz="UTC")]
        if self._end_date:
            df = df[df.index <= pd.Timestamp(self._end_date, tz="UTC")]
        return df

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Force all non-index columns to numeric dtype."""
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
