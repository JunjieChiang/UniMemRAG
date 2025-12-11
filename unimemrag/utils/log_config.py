# logger_config.py
from loguru import logger
import sys
import os

LOG_PATH = "log/api.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def setup_logger(log_path=LOG_PATH, level="INFO"):
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
        "| <level>{level: <8}</level> "
        "| <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )

    logger.add(
        log_path,
        level=level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=log_format
    )

    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=log_format
    )

    return logger