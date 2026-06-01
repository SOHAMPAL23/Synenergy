"""
EnerVision AI - Centralized Logger
Provides structured, leveled logging for every module in the pipeline.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "ml/outputs/logs",
    log_file: str = "enevision_pipeline.log",
    fmt: str = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
) -> logging.Logger:
    """
    Build and return a named logger with both console and rotating-file handlers.

    Args:
        name:     Logger name (typically __name__ of the calling module).
        level:    Logging level string (DEBUG / INFO / WARNING / ERROR).
        log_dir:  Directory where log files are stored.
        log_file: Log file name.
        fmt:      Log record format string.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating file handler ---
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


class PipelineLogger:
    """Context-manager wrapper that logs section start/end with timing."""

    def __init__(self, logger: logging.Logger, section: str) -> None:
        self._logger = logger
        self._section = section
        self._start: Optional[float] = None

    def __enter__(self) -> "PipelineLogger":
        import time
        self._start = time.perf_counter()
        self._logger.info(">>> START  [%s]", self._section)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import time
        elapsed = time.perf_counter() - self._start  # type: ignore[operator]
        if exc_type is None:
            self._logger.info("<<< DONE   [%s] in %.2fs", self._section, elapsed)
        else:
            self._logger.error(
                "<<< FAILED [%s] after %.2fs — %s: %s",
                self._section, elapsed, exc_type.__name__, exc_val,
            )
        return False  # Re-raise exceptions
