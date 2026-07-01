from __future__ import annotations

import logging
import sys
from pathlib import Path
from loguru import logger

# Base Directory of the Backend
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class InterceptHandler(logging.Handler):
    """Custom standard logging handler that forwards messages to Loguru sink."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame = sys._getframe(6)
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back

        cf = frame.f_code if frame else None
        filename = cf.co_filename if cf else record.pathname
        lineno = frame.f_lineno if frame else record.lineno
        func_name = cf.co_name if cf else record.funcName

        logger.opt(depth=6, exception=record.exc_info).log(
            level, 
            record.getMessage()
        )


def configure_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """Redirect standard logger outputs and configure Loguru sinks."""
    # Remove default handler
    logger.remove()

    # Console Handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=True,
        diagnose=True,
    )

    # File Handler
    if log_to_file:
        log_dir = BACKEND_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "app.log"
        
        logger.add(
            str(log_path),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="50 MB",
            retention="10 days",
            compression="gz",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    # Intercept standard logging messages from libraries
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Specific libraries configuration overrides
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "sqlalchemy.engine"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    logger.info("Structured logging environment initialized successfully.")
