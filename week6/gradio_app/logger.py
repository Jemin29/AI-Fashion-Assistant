"""
Week 6 — AI Fashion Creative Studio
Logging Framework

Provides a structured, Loguru-based logging system with:
- Console output (colorized, INFO+)
- Rotating file sink (DEBUG+)
- Error-only sink (ERROR+)
- Request access log
- Context-aware logger binding

Usage:
    from week6.gradio_app.logger import get_logger, setup_logging

    setup_logging()                          # Call once at app startup
    logger = get_logger(__name__)            # Get module-specific logger
    logger.info("Studio initialized")
    logger.bind(user_action="generate").info("Image generation started")
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
_WEEK6_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _WEEK6_ROOT / "logging_config.yaml"
_LOGS_DIR = _WEEK6_ROOT / "logs"

# ── Module-level flag to prevent double initialization ────────────────────────
_initialized: bool = False


def _load_config() -> dict:
    """Load logging YAML config or return sensible defaults."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Initialize the Loguru logging system from YAML config.

    Args:
        log_level: Override log level (DEBUG/INFO/WARNING/ERROR).
                   Defaults to env var FASHION_STUDIO_LOG_LEVEL or INFO.
        log_dir:   Override log directory. Defaults to week6/logs/.
        force:     Re-initialize even if already set up.
    """
    global _initialized
    if _initialized and not force:
        return

    # Remove all existing handlers
    logger.remove()

    config = _load_config()
    effective_level = (
        log_level
        or os.getenv("FASHION_STUDIO_LOG_LEVEL", "INFO")
    ).upper()

    resolved_log_dir = log_dir or _LOGS_DIR
    resolved_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Console Sink ─────────────────────────────────────────────────────────
    console_cfg = config.get("console", {})
    if console_cfg.get("enabled", True):
        console_fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            level=effective_level,
            format=console_fmt,
            colorize=True,
            backtrace=console_cfg.get("backtrace", True),
            diagnose=console_cfg.get("diagnose", False),
        )

    # ── Rotating File Sink ───────────────────────────────────────────────────
    file_cfg = config.get("file", {})
    if file_cfg.get("enabled", True):
        logger.add(
            str(resolved_log_dir / "studio.log"),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=file_cfg.get("rotation", "10 MB"),
            retention=file_cfg.get("retention", "7 days"),
            compression=file_cfg.get("compression", "zip"),
            backtrace=True,
            diagnose=False,
            enqueue=file_cfg.get("enqueue", True),
            encoding="utf-8",
        )

    # ── Error-Only Sink ──────────────────────────────────────────────────────
    error_cfg = config.get("error_file", {})
    if error_cfg.get("enabled", True):
        logger.add(
            str(resolved_log_dir / "errors.log"),
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
            rotation=error_cfg.get("rotation", "5 MB"),
            retention=error_cfg.get("retention", "30 days"),
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
            encoding="utf-8",
        )

    # ── Access Log Sink ───────────────────────────────────────────────────────
    access_cfg = config.get("access", {})
    if access_cfg.get("enabled", True):
        logger.add(
            str(resolved_log_dir / "access.log"),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | ACCESS | {message}",
            rotation=access_cfg.get("rotation", "50 MB"),
            retention=access_cfg.get("retention", "3 days"),
            filter=lambda record: record["extra"].get("access", False),
            enqueue=True,
            encoding="utf-8",
        )

    _initialized = True
    logger.info(
        f"Logging initialized | level={effective_level} | log_dir={resolved_log_dir}"
    )


def get_logger(name: str):
    """
    Return a module-scoped Loguru logger with the given name bound to it.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Loguru logger with 'module' context bound.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting service")
    """
    return logger.bind(module=name)


def log_access(action: str, details: str = "") -> None:
    """Log a user action to the access log."""
    logger.bind(access=True).info(f"action={action} | {details}")


def log_generation(
    prompt: str,
    model: str,
    latency_ms: float,
    success: bool = True,
) -> None:
    """Structured log entry for image generation events."""
    status = "SUCCESS" if success else "FAILURE"
    logger.bind(module="generation").info(
        f"[GENERATION] {status} | model={model} | latency={latency_ms:.0f}ms | "
        f"prompt={prompt[:80]}{'...' if len(prompt) > 80 else ''}"
    )


def log_rag_query(
    query: str,
    n_results: int,
    latency_ms: float,
    collection: str = "fashion_knowledge",
) -> None:
    """Structured log entry for RAG retrieval events."""
    logger.bind(module="rag").info(
        f"[RAG QUERY] collection={collection} | n_results={n_results} | "
        f"latency={latency_ms:.0f}ms | query={query[:80]}{'...' if len(query) > 80 else ''}"
    )
