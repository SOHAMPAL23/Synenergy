"""
EnerVision AI - Configuration Loader
Loads config.yaml and exposes values via dot-notation or dict access.
"""

import os
from typing import Any

import yaml

from ml.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "config.yaml"
)


class ConfigLoader:
    """
    Loads a YAML configuration file and provides attribute-style access.

    Example::

        cfg = ConfigLoader()
        print(cfg.data.target_column)
        print(cfg["forecasting"]["test_size"])
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        self._path = os.path.abspath(config_path)
        self._data: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        self._data = raw or {}
        logger.info("Config loaded from: %s", self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return top-level key value with optional default."""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return _DotDict(value)
            return value
        raise AttributeError(f"Config has no key '{key}'")

    def as_dict(self) -> dict:
        """Return the full configuration as a plain dict."""
        return dict(self._data)

    def reload(self) -> None:
        """Re-read the config file from disk."""
        self._load()
        logger.info("Config reloaded.")


class _DotDict:
    """Lightweight wrapper for nested dict access via attributes."""

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key in data:
            value = data[key]
            if isinstance(value, dict):
                return _DotDict(value)
            return value
        raise AttributeError(f"No config key '{key}'")

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_data")[key]

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_data").get(key, default)

    def as_dict(self) -> dict:
        return object.__getattribute__(self, "_data")

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_data"))


# Module-level singleton ─ import and use directly
config = ConfigLoader()
