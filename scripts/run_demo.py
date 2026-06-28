"""Launch the Build Console in DEMO mode — serves the curated demo set (build/ui_verify) with the teleprompter
auto-demo available. Fixes the foot-gun where the default build root (build/ui_workbench) has none of the demo
packages the auto-demo references, so a reviewer running the app the obvious way saw a dead demo.

Usage:  python scripts/run_demo.py [--web] [--port 8765]
Then open http://127.0.0.1:8765/?demo=1 (browser, with --web) or the native window, and drive the walkthrough
with the teleprompter remote or the Left/Right arrow keys. If build/ui_verify is missing/stale, first run
`python scripts/rebake_demo_set.py`.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    cmd = [sys.executable, "-m", "virturoid.ui_server", "--build-root", "build/ui_verify", *sys.argv[1:]]
    print("Demo mode: serving build/ui_verify. Open the teleprompter walkthrough at  /?demo=1")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
