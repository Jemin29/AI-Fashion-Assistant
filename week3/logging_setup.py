"""
week3/logging_setup.py
======================
Loguru logging configurations for Week 3 ControlNet Assistant.
Sets up console logs and rotated multi-sink file logs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from loguru import logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


# =============================================================================
# ── Standard Library Intercept
# =============================================================================

class InterceptHandler(logging.Handler):
    """
    Intercepts standard logging calls and redirects them to Loguru.
    Ensures third-party library warnings/logs are caught in the same pipeline.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# =============================================================================
# ── Setup Function
# =============================================================================

def setup_logging(log_dir: Optional[Path] = None, level: str = "INFO") -> None:
    """
    Initializes and configures the Loguru logger.
    
    Creates:
      1. Console logging sink (stderr, color-coded, level-aware).
      2. Rotary general log sink (week3_general.log, rotated daily).
      3. Rotary error log sink (errors.log, rotated daily, ERROR level).
      4. Rotary controlnet log sink (controlnet.log, filters for preprocessor/model).
    """
    if not _HAS_LOGURU:
        # Fallback to standard logging if Loguru is not installed
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
        logging.warning("Loguru not installed. Falling back to standard library logging configuration.")
        return

    # Determine logging directory
    dir_path = Path(log_dir).resolve() if log_dir else _DEFAULT_LOG_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    # Remove all default handlers to prevent duplication
    logger.remove()

    # 1. Console Log Sink (Stdout/Stderr)
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=console_format, colorize=True)

    # 2. General Log File (rotated daily, debug level)
    logger.add(
        dir_path / "week3_general.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="10 days",
        compression="zip",
        encoding="utf-8"
    )

    # 3. Dedicated Error Log File (rotated daily, error level)
    logger.add(
        dir_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8"
    )

    # 4. Dedicated ControlNet & Preprocessing Log File
    # Filters logging messages to capture preprocessor and pipeline operations
    def cnet_filter(record: Dict[str, Any]) -> bool:
        name = record["name"]
        return "controlnet" in name or "preprocessor" in name or "pipeline" in name

    logger.add(
        dir_path / "controlnet.log",
        level="DEBUG",
        filter=cnet_filter,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="10 days",
        encoding="utf-8"
    )

    # Redirect standard library logging to Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Silence verbose logs from heavy ML libraries (diffusers, transformers)
    for lib in ("urllib3", "h5py", "matplotlib", "PIL", "filelock"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger.success("Logging framework initialized successfully. Logs directed to {}", dir_path)
