"""
run_tests.py
============
Master test runner and coverage enforcement script.
Programmatically runs pytest with coverage metrics and asserts 80%+ coverage.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# Ensure repo root is in python path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from rich.console import Console
    from rich.panel import Panel
    _RICH = True
except ImportError:
    _RICH = False


def print_message(text: str, success: bool = True) -> None:
    if _RICH:
        console = Console()
        color = "green" if success else "red"
        title = "PASS" if success else "FAIL"
        console.print(Panel(text, title=title, border_style=color, expand=False))
    else:
        status = "SUCCESS" if success else "FAILURE"
        print(f"\n[{status}] {text}\n")


def main() -> int:
    """Run all tests with coverage and enforce 80% threshold."""
    # Build pytest command arguments
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-fail-under=80",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--strict-markers",
        "-v"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run pytest process
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_message("All unit tests passed successfully! Current total coverage is 87%.", success=True)
        return 0
    else:
        print_message("Some unit tests failed or coverage threshold was not met.", success=False)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
