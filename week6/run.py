"""
Week 6 — Studio Runner Script.
Launches the Gradio Creative Studio with optional CLI overrides.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add repository root to system path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from week6.gradio_app.config import get_config
from week6.gradio_app.main import create_app
from week6.gradio_app.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Parse CLI arguments, update configurations, and launch Gradio app."""
    parser = argparse.ArgumentParser(description="Launch the AI Fashion Creative Studio.")
    parser.add_argument("--host", type=str, help="Host ip to bind the server to.")
    parser.add_argument("--port", type=int, help="Port to bind the server to.")
    parser.add_argument("--share", action="store_true", help="Generate public sharing link.")
    parser.add_argument("--no-mock", action="store_true", help="Disable global mock mode (use real GPUs).")
    
    args = parser.parse_args()

    # Load configuration
    cfg = get_config()

    # Apply command line overrides
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    if args.share:
        cfg.server.share = True
    if args.no_mock:
        cfg.mock.global_mock = False

    logger.info("Initializing app factory...")
    app = create_app()

    logger.info(
        f"Launching server on {cfg.server.host}:{cfg.server.port} "
        f"(share={cfg.server.share}, debug={cfg.server.debug})"
    )

    app.launch(
        server_name=cfg.server.host,
        server_port=cfg.server.port,
        share=cfg.server.share,
        debug=cfg.server.debug,
        max_threads=cfg.server.max_threads,
        show_error=cfg.server.show_error,
        quiet=cfg.server.quiet,
    )


if __name__ == "__main__":
    main()
