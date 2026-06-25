"""
week2/logging_setup.py
======================
Production-grade Loguru logging system for Week 2.

Features
--------
- Multi-sink architecture (console + file + error-file + generation-file)
- Auto-creates log directory
- Silences verbose third-party libraries
- `generation_logger` context manager for tagged generation events
- Thread-safe singleton initialisation

Usage
-----
    from week2.logging_setup import setup_logging, get_logger

    setup_logging()                         # Call once at startup
    logger = get_logger(__name__)
    logger.info("Ready to generate!")

    # Tag a log record for the generation sink:
    from week2.logging_setup import generation_logger
    with generation_logger(prompt="a red dress"):
        logger.info("Starting generation")
"""

from __future__ import annotations

import sys
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
_WEEK2_DIR = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _WEEK2_DIR / "logs"

# ── State ─────────────────────────────────────────────────────────────────────
_INITIALIZED: bool = False


# =============================================================================
# ── Third-party library silencing
# =============================================================================

_NOISY_LIBS = [
    "diffusers",
    "transformers",
    "torch",
    "PIL",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "tokenizers",
    "accelerate",
    "safetensors",
]


def _silence_libraries(level: str = "WARNING") -> None:
    """Suppress verbose logging from third-party ML libraries."""
    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(getattr(logging, level, logging.WARNING))


# =============================================================================
# ── Intercept Handler (route stdlib logging → loguru)
# =============================================================================

class _InterceptHandler(logging.Handler):
    """
    Redirect stdlib `logging` calls into Loguru so everything
    flows through a single pipeline.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Find the Loguru level name for this record
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Walk the call stack to find the real caller
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# =============================================================================
# ── Log Format Strings
# =============================================================================

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)


# =============================================================================
# ── Generation Filter
# =============================================================================

def _is_generation_record(record: dict) -> bool:
    """Filter: only pass records tagged with 'generation' in extra."""
    return record.get("extra", {}).get("generation", False) is True


# =============================================================================
# ── Setup
# =============================================================================

def setup_logging(
    level: str                  = "INFO",
    log_dir: Optional[Path]     = None,
    colorize: bool              = True,
    rotation: str               = "10 MB",
    retention: str              = "7 days",
    enable_generation_sink: bool= True,
    intercept_stdlib: bool      = True,
) -> None:
    """
    Initialise the logging system. Safe to call multiple times.

    Parameters
    ----------
    level : str
        Root log level (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    log_dir : Path, optional
        Directory for log files. Defaults to ``week2/logs/``.
    colorize : bool
        Colourize console output.
    rotation : str
        Loguru rotation policy (e.g. "10 MB", "1 day").
    retention : str
        Loguru retention policy (e.g. "7 days").
    enable_generation_sink : bool
        Add a separate sink for generation-tagged events.
    intercept_stdlib : bool
        Route stdlib logging → loguru.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Remove default loguru handler ─────────────────────────────────────
    logger.remove()

    # ── Console sink ──────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level       = level,
        format      = _CONSOLE_FORMAT,
        colorize    = colorize,
        backtrace   = True,
        diagnose    = True,
        enqueue     = False,
    )

    # ── General file sink (all levels) ────────────────────────────────────
    logger.add(
        str(log_dir / "week2_{time:YYYY-MM-DD}.log"),
        level       = "DEBUG",
        format      = _FILE_FORMAT,
        rotation    = rotation,
        retention   = retention,
        compression = "gz",
        encoding    = "utf-8",
        backtrace   = True,
        diagnose    = True,
        enqueue     = True,         # Thread-safe async write
    )

    # ── Error-only file sink ───────────────────────────────────────────────
    logger.add(
        str(log_dir / "errors_{time:YYYY-MM-DD}.log"),
        level       = "ERROR",
        format      = _FILE_FORMAT,
        rotation    = "5 MB",
        retention   = "30 days",
        compression = "gz",
        encoding    = "utf-8",
        backtrace   = True,
        diagnose    = True,
        enqueue     = True,
    )

    # ── Generation events sink (tagged records only) ───────────────────────
    if enable_generation_sink:
        logger.add(
            str(log_dir / "generation_{time:YYYY-MM-DD}.log"),
            level       = "INFO",
            format      = _FILE_FORMAT,
            rotation    = "20 MB",
            retention   = "14 days",
            compression = "gz",
            encoding    = "utf-8",
            filter      = _is_generation_record,
            enqueue     = True,
        )

    # ── Intercept stdlib logging ───────────────────────────────────────────
    if intercept_stdlib:
        _silence_libraries()
        logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    _INITIALIZED = True
    logger.info(
        "Logging initialised | level={} | log_dir={}",
        level,
        log_dir,
    )


def teardown_logging() -> None:
    """Remove all loguru sinks (useful for test isolation)."""
    global _INITIALIZED
    logger.remove()
    _INITIALIZED = False


# =============================================================================
# ── Public helpers
# =============================================================================

def get_logger(name: str):
    """
    Return a bound Loguru logger with the given module name.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    BoundLogger
        Loguru logger bound with ``name`` in its extra context.
    """
    return logger.bind(name=name)


@contextmanager
def generation_logger(
    prompt: str    = "",
    image_id: str  = "",
    **extra,
) -> Generator[None, None, None]:
    """
    Context manager that tags all log records emitted inside the block
    as generation events, routing them to the generation sink.

    Usage
    -----
        with generation_logger(prompt="red dress", image_id="IMG_001"):
            logger.info("Generating…")
    """
    ctx = logger.contextualize(
        generation=True,
        prompt=prompt,
        image_id=image_id,
        **extra,
    )
    with ctx:
        yield


# =============================================================================
# ── Auto-initialise with defaults if imported directly
# =============================================================================

def _auto_init_from_config() -> None:
    """
    Attempt to initialise logging from config_manager if available.
    Falls back to safe defaults if config loading fails.
    """
    try:
        from week2.config_manager import get_config  # type: ignore[import]
        cfg = get_config()
        setup_logging(
            level       = "INFO",
            log_dir     = cfg.log_dir,
            colorize    = True,
            rotation    = "10 MB",
            retention   = "7 days",
        )
    except Exception:
        # Config not yet available — use safe defaults
        setup_logging()
