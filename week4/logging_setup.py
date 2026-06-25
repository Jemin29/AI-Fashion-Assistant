"""
week4/logging_setup.py
======================
Loguru logging configuration system for Week 4 Brand Style Personalization.
Establishes console sink and daily rotated file sinks for general, error, and training traces.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
    Intercepts standard library logging calls and redirects them to Loguru.
    Ensures warnings/info from diffusers, transformers, and accelerate are captured.
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
    Initialize and configure Loguru logger.
    
    Creates:
      1. Console logging sink (stderr, color-coded, level-aware).
      2. Rotary general log sink (week4_general.log, rotated daily).
      3. Rotary error log sink (errors.log, rotated daily, ERROR level).
      4. Rotary trainer log sink (lora_trainer.log, filters for training progress).
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
        dir_path / "week4_general.log",
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
        retention="15 days",
        compression="zip",
        encoding="utf-8"
    )

    # 4. Custom Trainer Log File (filters for trainer traces)
    def trainer_log_filter(record: Dict[str, Any]) -> bool:
        name = record["name"]
        msg = record["message"].lower()
        # Log trainer module, lora module, or general logs containing training/loss/epoch keywords
        return (
            name.startswith("week4.trainers") or
            name.startswith("week4.lora") or
            "epoch" in msg or
            "loss" in msg or
            "train" in msg or
            "adapter" in msg
        )

    logger.add(
        dir_path / "lora_trainer.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        filter=trainer_log_filter,
        rotation="00:00",
        retention="10 days",
        compression="zip",
        encoding="utf-8"
    )

    # 5. Redirect standard library logs (e.g. from diffusers/transformers) to Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Force diffusers, transformers, and accelerate to propagate their logs
    for lib_name in ["diffusers", "transformers", "accelerate"]:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.handlers = [InterceptHandler()]
        lib_logger.propagate = False

    logger.success("Week 4 Loguru logging system initialized. general, error, and trainer sinks active.")
