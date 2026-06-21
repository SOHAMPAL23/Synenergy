"""EnerVision AI - Utility helpers for backend."""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def paginate(items: list, skip: int = 0, limit: int = 100) -> list:
    """Simple list pagination."""
    return items[skip: skip + limit]
