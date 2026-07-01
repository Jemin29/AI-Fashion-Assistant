from __future__ import annotations

import sys
import subprocess
from pathlib import Path

def run_portfolio_pipeline() -> None:
    """Coordinate and execute the automated fashion evaluation & showcase generator."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    generator_script = repo_root / "portfolio_generator.py"
    
    print(f"[*] Locating portfolio generator: {generator_script}")
    if not generator_script.exists():
        print(f"[!] Error: Could not find portfolio generator at {generator_script}")
        sys.exit(1)
        
    print("[*] Executing automated fashion portfolio compiler...")
    try:
        # Run root portfolio generator in a subprocess to respect package imports
        res = subprocess.run(
            [sys.executable, str(generator_script)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True
        )
        print("[+] Showcase compilation succeeded.")
        print(res.stdout)
        
        # Verify result outputs
        out_html = repo_root / "outputs" / "portfolio" / "index.html"
        if out_html.exists():
            print(f"[+] Glassmorphic portfolio index.html generated at: {out_html.resolve()}")
        else:
            print("[!] Warning: Output index.html was not found in outputs/portfolio/.")
            
    except subprocess.CalledProcessError as exc:
        print(f"[!] Portfolio compilation failed with error code {exc.returncode}")
        print(exc.stderr)
        sys.exit(exc.returncode)

if __name__ == "__main__":
    run_portfolio_pipeline()
