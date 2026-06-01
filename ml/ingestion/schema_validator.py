"""
EnerVision AI - Schema Validator
Validates DataFrame schema against configured requirements.
"""

from typing import List

import pandas as pd

from ml.utils.config_loader import config
from ml.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaValidationError(Exception):
    """Raised when the DataFrame schema does not match expectations."""


class SchemaValidator:
    """
    Validates a loaded DataFrame against required columns and basic
    data-quality rules before it enters the preprocessing stage.

    Usage::

        validator = SchemaValidator()
        validator.validate(df)   # raises SchemaValidationError on failure
    """

    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config
        data_cfg = self._cfg.data
        self._required_cols: List[str] = list(data_cfg.required_columns or [])
        self._target_col: str = data_cfg.target_column
        self._min_rows: int = 100

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """
        Run all validation checks.  Raises :class:`SchemaValidationError`
        if any check fails.

        Args:
            df: DataFrame with DatetimeIndex (output of DataLoader).
        """
        logger.info("Running schema validation on DataFrame (%d rows, %d cols)…", *df.shape)
        errors: List[str] = []

        errors += self._check_index(df)
        errors += self._check_required_columns(df)
        errors += self._check_min_rows(df)
        errors += self._check_target_not_all_null(df)

        if errors:
            msg = "Schema validation failed:\n  - " + "\n  - ".join(errors)
            logger.error(msg)
            raise SchemaValidationError(msg)

        logger.info("Schema validation passed ✓")
        self._log_summary(df)

    def report(self, df: pd.DataFrame) -> dict:
        """Return a dict summary of data quality stats (non-raising)."""
        total = len(df)
        report = {
            "total_rows": total,
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            },
            "missing_per_column": df.isnull().sum().to_dict(),
            "missing_pct_per_column": (df.isnull().mean() * 100).round(2).to_dict(),
            "duplicated_rows": int(df.index.duplicated().sum()),
        }
        return report

    # ------------------------------------------------------------------
    # Private checks
    # ------------------------------------------------------------------

    def _check_index(self, df: pd.DataFrame) -> List[str]:
        errors = []
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("DataFrame index must be a DatetimeIndex.")
        return errors

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """
        required_columns in config refer to the original CSV column names.
        After DataLoader the timestamp becomes the index, so we only check
        non-timestamp required columns against df.columns.
        """
        errors = []
        non_ts_required = [
            c for c in self._required_cols if c != "utc_timestamp"
        ]
        missing = [c for c in non_ts_required if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        return errors

    def _check_min_rows(self, df: pd.DataFrame) -> List[str]:
        if len(df) < self._min_rows:
            return [f"Dataset too small: {len(df)} rows (minimum {self._min_rows})."]
        return []

    def _check_target_not_all_null(self, df: pd.DataFrame) -> List[str]:
        if self._target_col not in df.columns:
            return []  # already caught by required columns check
        null_pct = df[self._target_col].isnull().mean() * 100
        if null_pct == 100:
            return [f"Target column '{self._target_col}' is entirely null."]
        if null_pct > 50:
            logger.warning(
                "Target column '%s' is %.1f%% null — results may be unreliable.",
                self._target_col, null_pct,
            )
        return []

    def _log_summary(self, df: pd.DataFrame) -> None:
        rpt = self.report(df)
        logger.info(
            "Data summary → rows=%d | cols=%d | range=[%s → %s] | duplicates=%d",
            rpt["total_rows"],
            rpt["total_columns"],
            rpt["date_range"]["start"],
            rpt["date_range"]["end"],
            rpt["duplicated_rows"],
        )
        for col, pct in rpt["missing_pct_per_column"].items():
            if pct > 0:
                logger.warning("  Missing: %-50s %.1f%%", col, pct)
