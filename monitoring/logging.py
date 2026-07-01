from __future__ import annotations

import logging
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

# Import metrics registry to update states on event triggers
from monitoring.metrics import registry

# Root logs folder
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = WORKSPACE_ROOT / "logs"


class LoguruInterceptHandler(logging.Handler):
    """Custom logging handler to intercept standard library logging messages and route them to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the frame where the log message was generated
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
    """Initialize structured centralized Loguru logger with rotation and stderr routing."""
    logger.remove()

    # Console handler: rich aesthetic output format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=True,
        diagnose=True,
    )

    # File handler: structured formats compressed daily
    if log_to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "monitoring.log"

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

    # Redirect standard library logging handlers
    logging.basicConfig(handlers=[LoguruInterceptHandler()], level=0, force=True)

    # Intercept common server modules
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "sqlalchemy.engine", "celery"):
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [LoguruInterceptHandler()]
        logging_logger.propagate = False

    logger.info("Structured logging system initialized successfully.")


# ── Structured Event Loggers ──────────────────────────────────────────────────

def log_error(message: str, exc: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log system exceptions with tracebacks and metrics integration."""
    bind_logger = logger.bind(**(extra or {}))
    if exc:
        bind_logger.exception(f"{message} | Exception: {str(exc)}")
    else:
        bind_logger.error(message)


def log_warning(message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Log system warnings indicating unexpected behaviors or degraded states."""
    logger.bind(**(extra or {})).warning(message)


def log_api_request(method: str, path: str, status_code: int, latency_seconds: float, client_ip: str) -> None:
    """Log HTTP API request details and record operational latency."""
    # Increment request counts and observe request duration
    registry.inc_counter("api_requests_total")
    registry.observe_histogram("api_latency", latency_seconds)

    # Set appropriate logging level based on response code
    log_func = logger.info
    if status_code >= 500:
        log_func = logger.error
    elif status_code >= 400:
        log_func = logger.warning

    latency_ms = latency_seconds * 1000.0
    log_func(
        f"API Request: {method} {path} | Status: {status_code} | Latency: {latency_ms:.2f}ms | IP: {client_ip}",
        method=method,
        path=path,
        status_code=status_code,
        latency_ms=latency_ms,
        client_ip=client_ip
    )


def log_generation(prompt: str, duration_seconds: float, success: bool, error_msg: Optional[str] = None) -> None:
    """Log image/design generation outcome and duration telemetry."""
    registry.inc_counter("generation_requests_total")
    registry.observe_histogram("generation_duration", duration_seconds)

    if success:
        logger.info(
            f"Generation Success | Prompt: '{prompt}' | Duration: {duration_seconds:.2f}s",
            prompt=prompt,
            duration=duration_seconds,
            status="success"
        )
    else:
        logger.error(
            f"Generation Failure | Prompt: '{prompt}' | Error: {error_msg}",
            prompt=prompt,
            error=error_msg,
            status="failure"
        )


def log_redis_failure(operation: str, error_msg: str) -> None:
    """Log Redis connection/caching failures and flag telemetry status."""
    registry.set_gauge("redis_status", 0)  # Flag redis down
    logger.error(
        f"Redis Connection Failure | Operation: {operation} | Error: {error_msg}",
        redis_operation=operation,
        error=error_msg
    )


def log_celery_failure(task_name: str, task_id: str, error_msg: str) -> None:
    """Log worker-level background task exceptions."""
    logger.error(
        f"Celery Task Failure | Task: {task_name} | ID: {task_id} | Error: {error_msg}",
        task_name=task_name,
        task_id=task_id,
        error=error_msg
    )
