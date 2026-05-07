"""
Structured logging configuration with file rotation.
Provides separate loggers for each system component.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.config import settings


def _create_formatter() -> logging.Formatter:
    """Create a consistent log formatter."""
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _create_file_handler(log_file: str) -> RotatingFileHandler:
    """Create a rotating file handler."""
    log_path = Path(settings.log_dir) / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(_create_formatter())
    return handler


def _create_console_handler() -> logging.StreamHandler:
    """Create a console (stdout) handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_create_formatter())
    return handler


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """
    Get or create a logger with both file and console handlers.

    Args:
        name: Logger name (e.g., 'nlp', 'cv', 'fusion', 'safety', 'api')
        log_file: Log file name within the log directory

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"cdss.{name}")

    # Avoid adding duplicate handlers
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.addHandler(_create_file_handler(log_file))
        logger.addHandler(_create_console_handler())
        logger.propagate = False

    return logger


# Pre-configured loggers for each component
api_logger = get_logger("api", "api.log")
nlp_logger = get_logger("nlp", "nlp.log")
cv_logger = get_logger("cv", "cv.log")
fusion_logger = get_logger("fusion", "fusion.log")
safety_logger = get_logger("safety", "safety.log")
db_logger = get_logger("database", "database.log")
