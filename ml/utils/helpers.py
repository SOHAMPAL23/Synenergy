"""
EnerVision AI - Shared Helper Utilities
Functions used across multiple modules.
"""

import os
from typing import Union

import numpy as np
import pandas as pd

from ml.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it does not exist; return path."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (returns value in %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with RMSE, MAE, MAPE."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def safe_concat(frames: list, **kwargs) -> pd.DataFrame:
    """pd.concat with empty-list guard."""
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, **kwargs)


def to_numpy(series: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Coerce Series or ndarray to 1-D numpy array."""
    if isinstance(series, pd.Series):
        return series.values
    return np.asarray(series).ravel()


def time_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.0,
) -> tuple:
    """
    Chronological train / (val) / test split — no shuffling.

    Returns:
        (train, test) or (train, val, test) if val_size > 0.
    """
    n = len(df)
    test_n = int(n * test_size)
    val_n = int(n * val_size)
    train_end = n - test_n - val_n

    train = df.iloc[:train_end]
    if val_size > 0:
        val = df.iloc[train_end: train_end + val_n]
        test = df.iloc[train_end + val_n:]
        logger.info(
            "Time split → train=%d | val=%d | test=%d", len(train), len(val), len(test)
        )
        return train, val, test

    test = df.iloc[train_end:]
    logger.info("Time split → train=%d | test=%d", len(train), len(test))
    return train, test


def clip_predictions(
    preds: np.ndarray, lower: float = 0.0, upper: float = None
) -> np.ndarray:
    """Clip predictions to physical bounds."""
    preds = np.clip(preds, lower, upper)
    return preds


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def metrics_table(results: dict) -> str:
    """Pretty-print a metrics results dict as a table string."""
    header = f"{'Model':<25} {'RMSE':>12} {'MAE':>12} {'MAPE (%)':>12}"
    sep = "-" * len(header)
    rows = [header, sep]
    for model, m in results.items():
        rows.append(
            f"{model:<25} {m['rmse']:>12.2f} {m['mae']:>12.2f} {m['mape']:>12.2f}"
        )
    return "\n".join(rows)
