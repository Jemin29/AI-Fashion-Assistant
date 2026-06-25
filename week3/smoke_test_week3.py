"""
week3/smoke_test_week3.py
==========================
Smoke test script to verify folder structure, configurations, and logging setups
for the Week 3 ControlNet milestone.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from week3.config_manager import get_config
from week3.logging_setup import setup_logging

def verify_structure(base_dir: Path) -> List[str]:
    """Check if all required directories exist."""
    required = [
        "configs", "controlnet", "preprocessors", "datasets",
        "evaluation", "pipelines", "outputs", "logs", "tests", "docs"
    ]
    missing = []
    for folder in required:
        if not (base_dir / folder).is_dir():
            missing.append(folder)
    return missing

def main() -> int:
    week3_path = Path(__file__).resolve().parent
    print(f"Auditing Week 3 Workspace at: {week3_path}")
    
    # ── 1. Check directories ──────────────────────────────────────────────────
    missing = verify_structure(week3_path)
    if missing:
        print(f"Error: Missing directories in workspace: {missing}", file=sys.stderr)
        return 1
    print("[OK] All 10 subdirectories are present.")
    
    # ── 2. Check Pydantic Config manager ──────────────────────────────────────
    try:
        cfg = get_config()
        print("[OK] CentralizedConfig loaded and validated successfully.")
        print(f"  - Device: {cfg.model.runtime.device}")
        print(f"  - Default ControlNet: {cfg.controlnet.model_id}")
        print(f"  - Canny resolution: {cfg.preprocessors.canny.resolution}")
    except Exception as err:
        print(f"Error loading configuration manager: {err}", file=sys.stderr)
        return 2
        
    # ── 3. Check logging setup ────────────────────────────────────────────────
    try:
        setup_logging(log_dir=cfg.log_dir)
        # Import loguru to test log emit
        from loguru import logger
        logger.success("[OK] Smoke test logging verified.")
    except Exception as err:
        print(f"Error initializing logging setup: {err}", file=sys.stderr)
        return 3
        
    print("\n=======================================================")
    print("   WEEK 3 SMOKE TEST COMPLETED SUCCESSFULLY (0 GAPS)")
    print("=======================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())
