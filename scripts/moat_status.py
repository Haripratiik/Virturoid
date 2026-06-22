"""Print the cumulative flywheel / moat status for a workspace's persistent memory.

Reads the MemoryDB + flywheel_manifest.jsonl and reports how the platform has compounded across ALL builds:
designs/skills/lessons/species banked, and how often warm-start reused prior learning. This is the
"prove the moat" artifact (startup plan §13 memory).

    PYTHONPATH=src python scripts/moat_status.py [--memory-dir build/memory] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory-dir", default="build/memory", help="Workspace memory dir.")
    ap.add_argument("--json", action="store_true", help="Emit the full JSON instead of the dashboard.")
    args = ap.parse_args()

    from virturoid.services.flywheel_status import moat_status

    status = moat_status(Path(args.memory_dir))
    if args.json:
        print(json.dumps(status, indent=2))
        return 0

    c = status["counts"]
    w = status["warm_start"]
    print("\n=== Virturoid moat status ===")
    print(f"  builds banked     : {c['runs']}")
    print(f"  designs banked    : {c['designs']}")
    print(f"  skills/policies   : {len(status['skills'])} distinct")
    print(f"  lessons banked    : {c['lessons']}")
    print(f"  species-tree nodes: {status['species_tree_nodes']}")
    print(f"  warm-start used   : {w['warm_started_builds']}/{w['trainable_builds']} ({w['utilization']:.0%})")
    print(f"  COMPOUNDING       : {status['compounding']}")
    if status["skills"]:
        print("  top banked skills :")
        for sk in status["skills"][:5]:
            print(f"     - {sk['skill_id']}: best {sk['best_success']:.0%}, {sk['builds']} builds, "
                  f"warm {sk['warm_started']}, trainers {sk['trainers']}")
    print(f"\n  {status['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
