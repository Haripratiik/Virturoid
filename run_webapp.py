#!/usr/bin/env python
"""Zero-config launcher for the Virturoid web app.

Run from the project folder with NO environment setup:

    python run_webapp.py

(optionally: python run_webapp.py --port 8000 --host 127.0.0.1)

It wires up the source path itself and gives a clear message if a dependency
is missing, so you never hit a silent "connection refused".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _check_deps() -> None:
    missing = []
    for mod, pip_name in [("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"), ("mujoco", "mujoco")]:
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(pip_name)
    if missing:
        print("\n[Virturoid] Missing Python packages: " + ", ".join(missing))
        print("Install them with:\n")
        print("    pip install " + " ".join(f'"{m}"' for m in missing) + "\n")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Virturoid web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workspace", default=str(ROOT / "build" / "webapp"))
    args = parser.parse_args()

    _check_deps()

    import uvicorn

    from virturoid.webapp import create_app

    app = create_app(Path(args.workspace))
    url = f"http://{args.host}:{args.port}"
    print("\n" + "=" * 52)
    print("  Virturoid is running.")
    print(f"  Open this in your browser:  {url}")
    print("  (Press Ctrl+C here to stop.)")
    print("=" * 52 + "\n")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except OSError as exc:
        print(f"\n[Virturoid] Could not bind {url}: {exc}")
        print("The port is probably in use. Try a different one, e.g.:  python run_webapp.py --port 8010\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
