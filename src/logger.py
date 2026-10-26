"""Logging setup: console + file."""
import logging

from src.config import LOG_DIR, LOG_FILE

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT)

    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger
