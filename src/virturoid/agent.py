"""Conversational agent entrypoint.

`python -m virturoid.agent` opens a REPL where you ask Virturoid to build and
iterate on robots in natural language. `--message` runs one turn and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from virturoid.services.agent import VirturoidAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to Virturoid to build and iterate on robots.")
    parser.add_argument("--workspace", default=str(Path("build") / "agent_session"))
    parser.add_argument("--message", default=None, help="Run a single message and exit.")
    parser.add_argument("--target-success", type=float, default=0.8)
    args = parser.parse_args()

    agent = VirturoidAgent(Path(args.workspace), target_success=args.target_success)

    if args.message is not None:
        print(agent.handle(args.message).message)
        return

    print("Virturoid agent. Ask me to build, evaluate, iterate, perceive, train, export. Ctrl-D to exit.")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(agent.handle(line).message)


if __name__ == "__main__":
    main()
