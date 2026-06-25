"""
src/utils/logging_setup.py
==========================
Unified Loguru logging configuration system for Weeks 2, 3, 4, and 5.
Establishes console sink and daily rotated file sinks for general, error, retrieval, training, and generation traces.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

try:
    from loguru import logger
    _HAS_LOGURU = True
except ImportError:
    _HAS_LOGURU = False

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_INITIALIZED = False


# =============================================================================
# ── Standard Library Intercept
# =============================================================================

class InterceptHandler(logging.Handler):
    """
    Intercepts standard library logging calls and redirects them to Loguru.
    Ensures warnings/info from transformers, diffusers, and vector DB libraries are captured.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# =============================================================================
# ── Filters
# =============================================================================

def _is_generation_record(record: dict) -> bool:
    """Filter: only pass records tagged with 'generation' in extra."""
    return record.get("extra", {}).get("generation", False) is True


def _retrieval_log_filter(record: Dict[str, Any]) -> bool:
    """Filter: only pass RAG, similarity search, and database records."""
    name = record["name"]
    msg = record["message"].lower()
    return (
        name.startswith("src.rag") or
        name.startswith("week5.rag") or
        name.startswith("week5.retrieval") or
        name.startswith("week5.vector_db") or
        name.startswith("week5.embeddings") or
        name.startswith("week5.knowledge_base") or
        "query" in msg or
        "retrieval" in msg or
        "match" in msg or
        "similarity" in msg or
        "index" in msg or
        "embed" in msg or
        "vector" in msg or
        "search" in msg
    )


def _trainer_log_filter(record: Dict[str, Any]) -> bool:
    """Filter: only pass training-specific logs (LoRA trainer, adapters)."""
    name = record["name"]
    msg = record["message"].lower()
    return (
        "loss" in msg or
        "epoch" in msg or
        "adapter" in msg or
        "lora" in msg or
        "trainer" in msg or
        name.startswith("src.lora") or
        name.startswith("week4.lora")
    )


# =============================================================================
# ── Setup Function
# =============================================================================

def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    colorize: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
    enable_generation_sink: bool = True,
    intercept_stdlib: bool = True,
) -> None:
    """
    Initialize and configure Loguru logger. Safe to call multiple times.
    
    Creates:
      1. Console logging sink (stderr, color-coded, level-aware).
      2. Rotary general log sinks (week5_general.log, week4_general.log, week2_YYYY-MM-DD.log).
      3. Rotary error log sinks (errors.log, errors_YYYY-MM-DD.log).
      4. Rotary retrieval log sink (rag_retrieval.log).
      5. Rotary training log sink (lora_trainer.log).
      6. Rotary generation log sink (generation_YYYY-MM-DD.log).
    """
    if not _HAS_LOGURU:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
        logging.warning("Loguru not installed. Falling back to standard library logging.")
        return

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
    logger.add(sys.stderr, level=level, format=console_format, colorize=colorize)

    # 2. General Log Files
    logger.add(
        dir_path / "week5_general.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="10 days",
        compression="zip",
        encoding="utf-8"
    )
    logger.add(
        dir_path / "week4_general.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation="00:00",
        retention="10 days",
        compression="zip",
        encoding="utf-8"
    )
    logger.add(
        str(dir_path / "week2_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
        encoding="utf-8"
    )

    # 3. Error Log Files
    logger.add(
        dir_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation="00:00",
        retention="15 days",
        compression="zip",
        encoding="utf-8"
    )
    logger.add(
        str(dir_path / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation="5 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8"
    )

    # 4. Custom Retrieval Log File
    logger.add(
        dir_path / "rag_retrieval.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        filter=_retrieval_log_filter,
        rotation="00:00",
        retention="10 days",
        compression="zip",
        encoding="utf-8"
    )

    # 5. Custom Trainer Log File
    logger.add(
        dir_path / "lora_trainer.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        filter=_trainer_log_filter,
        rotation="00:00",
        retention="10 days",
        compression="zip",
        encoding="utf-8"
    )

    # 6. Custom Generation Log File
    if enable_generation_sink:
        logger.add(
            str(dir_path / "generation_{time:YYYY-MM-DD}.log"),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
            filter=_is_generation_record,
            rotation="20 MB",
            retention="14 days",
            compression="zip",
            encoding="utf-8"
        )

    # Redirect standard library logs if requested
    if intercept_stdlib:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        
        # Suppress verbose logging from third-party ML libraries
        noisy_libs = ["faiss", "sentence_transformers", "torch", "diffusers", "transformers", "PIL", "urllib3"]
        for lib_name in noisy_libs:
            lib_logger = logging.getLogger(lib_name)
            lib_logger.handlers = [InterceptHandler()]
            lib_logger.propagate = False

    _INITIALIZED = True
    logger.success("Unified Loguru logging system initialized.")


def teardown_logging() -> None:
    """Remove all loguru sinks (useful for test isolation)."""
    global _INITIALIZED
    logger.remove()
    _INITIALIZED = False


# =============================================================================
# ── Public Helpers
# =============================================================================

def get_logger(name: str):
    """Return a bound Loguru logger with the given module name."""
    return logger.bind(name=name)


@contextmanager
def generation_logger(
    prompt: str = "",
    image_id: str = "",
    **extra,
) -> Generator[None, None, None]:
    """
    Context manager that tags all log records emitted inside the block
    as generation events, routing them to the generation sink.
    """
    ctx = logger.contextualize(
        generation=True,
        prompt=prompt,
        image_id=image_id,
        **extra,
    )
    with ctx:
        yield

