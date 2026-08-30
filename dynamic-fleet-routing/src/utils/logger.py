"""Structured logging setup for the fleet routing system.

Provides consistent logging configuration across all modules
with support for file and console output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def setup_logger(
    name: str = "fleet_routing",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Create and configure a logger with consistent formatting.

    Args:
        name: Logger name (used for module identification).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        console: Whether to output logs to console.

    Returns:
        Configured logger instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "fleet_routing") -> logging.Logger:
    """Get an existing logger or create a default one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
