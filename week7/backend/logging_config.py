from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

# Base directory of the backend
BACKEND_ROOT = Path(__file__).resolve().parent


class LoguruInterceptHandler(logging.Handler):
    """Custom logging handler to intercept standard library logging messages and route them to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = sys._getframe(6)
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back

        cf = frame.f_code if frame else None
        filename = cf.co_filename if cf else record.pathname
        lineno = frame.f_lineno if frame else record.lineno

        logger.opt(depth=6, exception=record.exc_info).log(
            level,
            record.getMessage()
        )


def configure_logging(log_level: str = "INFO", log_to_file: bool = True) -> None:
    """Initialize central Loguru handlers for clean, rotated structured logging."""
    logger.remove()

    # Standard console logging with vibrant, readable aesthetics
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=True,
        diagnose=True,
    )

    if log_to_file:
        log_dir = BACKEND_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "app.log"

        logger.add(
            str(log_path),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message} | {extra}",
            rotation="50 MB",
            retention="10 days",
            compression="gz",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    # Intercept standard python logging
    logging.basicConfig(handlers=[LoguruInterceptHandler()], level=0, force=True)

    # Route uvicorn and database engine logs to loguru
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "sqlalchemy.engine"):
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [LoguruInterceptHandler()]
        logging_logger.propagate = False

    logger.info("Centralized logging system initialized successfully.")


# ── Centralized Structured Logging Helpers ────────────────────────────────────

def log_error(message: str, exc: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log system exceptions and critical errors with tracebacks."""
    bind_logger = logger.bind(**(extra or {}))
    if exc:
        bind_logger.exception(f"{message} | Exception: {str(exc)}")
    else:
        bind_logger.error(message)


def log_warning(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log system warnings indicating unexpected behaviors or soft issues."""
    logger.bind(**(extra or {})).warning(message)


def log_api_request(method: str, path: str, status_code: int, latency_ms: float, client_ip: str) -> None:
    """Log incoming HTTP API request details and latency metrics."""
    # Use different colors/levels for different status code classes
    log_func = logger.info
    if status_code >= 500:
        log_func = logger.error
    elif status_code >= 400:
        log_func = logger.warning

    log_func(
        f"API Request: {method} {path} | Status: {status_code} | Latency: {latency_ms:.2f}ms | IP: {client_ip}",
        method=method,
        path=path,
        status_code=status_code,
        latency_ms=latency_ms,
        client_ip=client_ip
    )


def log_generation_failure(prompt: str, error_msg: str, session_id: Optional[str] = None) -> None:
    """Log Stable Diffusion / ControlNet / LoRA generation failures."""
    logger.error(
        f"Generation Failure | Prompt: '{prompt}' | Session: {session_id or 'none'} | Error: {error_msg}",
        prompt=prompt,
        session_id=session_id,
        error=error_msg
    )


def log_redis_failure(operation: str, error_msg: str) -> None:
    """Log caching, task tracking, or pub/sub failures when talking to Redis."""
    logger.error(
        f"Redis Connection Failure | Operation: {operation} | Error: {error_msg}",
        redis_operation=operation,
        error=error_msg
    )


def log_celery_failure(task_name: str, task_id: str, error_msg: str) -> None:
    """Log worker-level Celery background task errors."""
    logger.error(
        f"Celery Task Failure | Task: {task_name} | ID: {task_id} | Error: {error_msg}",
        task_name=task_name,
        task_id=task_id,
        error=error_msg
    )
