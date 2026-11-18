"""
Centralized logging utility.

Every module should consume the logger
from this file rather than creating
individual loggers.

Benefits:
- Consistent formatting
- Easier troubleshooting
- Production readiness
"""

import logging

from pathlib import Path

from config import LOG_DIR


def get_logger(
    logger_name: str
) -> logging.Logger:
    """
    Create and return application logger.

    Args:
        logger_name:
            Name of logger.

    Returns:
        Configured logger.
    """

    logger = logging.getLogger(
        logger_name
    )

    logger.setLevel(
        logging.INFO
    )

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        Path(LOG_DIR) / "application.log"
    )

    file_handler.setFormatter(
        formatter
    )

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    return logger