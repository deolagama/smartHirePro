"""
SmartHirePro - Structured Logging Utility
==========================================
Configures a consistent, production-ready logger for every module
that imports it.  Logs go to both the console and a rotating file.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Import lazily to avoid circular dependency at package init time
def get_logger(name: str, log_file: str | None = None, level: str | None = None) -> logging.Logger:
    """
    Return a configured logger with the given name.

    Args:
        name:     Typically __name__ of the calling module.
        log_file: Override the default log file path.
        level:    Override the default log level (e.g. "DEBUG").

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    from config import settings  # deferred import

    _level = getattr(logging, (level or settings.LOG_LEVEL).upper(), logging.INFO)
    _log_file = log_file or settings.LOG_FILE

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(_level)

    # ------------------------------------------------------------------
    # Formatter
    # ------------------------------------------------------------------
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Console handler (stdout)
    # ------------------------------------------------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(_level)
    logger.addHandler(ch)

    # ------------------------------------------------------------------
    # Rotating file handler (5 MB × 5 backups)
    # ------------------------------------------------------------------
    try:
        Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            _log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        fh.setLevel(_level)
        logger.addHandler(fh)
    except Exception as exc:
        logger.warning("Could not set up file logging: %s", exc)

    # Prevent log records from propagating to the root logger
    logger.propagate = False

    return logger
