"""Convenience launcher for the Virturoid build console / agent-sessions viewer.

Adds ``src`` to ``sys.path`` (so no editable install is required) and runs the
ui_server CLI. Open http://127.0.0.1:<port>/sessions for the 'watch the agent
build' viewer (C1-C3), or / for the build console.

    python scripts/run_ui.py --web --port 8790
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from virturoid.ui_server import main  # noqa: E402

if __name__ == "__main__":
    main()
