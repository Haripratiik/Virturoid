"""CLI: import an existing robot (URDF/MJCF) into Virturoid (roadmap N5.1, startup plan §5/§28.2).

    python -m virturoid.import_robot path/to/robot.urdf
    python -m virturoid.import_robot path/to/robot.xml --save-gene out/gene.json
    python -m virturoid.import_robot robot.urdf --workspace build/desktop   # also store as a species node

Prints the recovered RobotGene summary, the honest import warnings, and the backend support matrix.
With --workspace, records the imported gene in the species tree so it joins the flywheel (a later
build can reuse/amend it). Needs MuJoCo (it parses both URDF and MJCF).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import a URDF/MJCF robot into Virturoid.")
    ap.add_argument("source", help="path to a .urdf or .mjcf/.xml file")
    ap.add_argument("--id", default=None, help="robot/gene id (default: file stem)")
    ap.add_argument("--species", default=None, help="species pattern (default: <class>.imported)")
    ap.add_argument("--save-gene", default=None, help="write the recovered RobotGene JSON to this path")
    ap.add_argument("--workspace", default=None, help="if set, record the gene as a species node (flywheel)")
    args = ap.parse_args(argv)

    from virturoid.services.robot_import import import_robot as run_import

    src = Path(args.source)
    robot_id = args.id or (src.stem if src.suffix else "imported_robot")
    try:
        out = run_import(str(src), robot_id=robot_id, species=args.species)
    except Exception as exc:  # noqa: BLE001 - a parse failure should be a clear message, not a traceback
        print(f"ERROR: could not import {args.source!r}: {exc}", file=sys.stderr)
        return 1

    gene = out["gene"]
    actuated = gene.actuated_joints()
    ee = gene.end_effector()
    print(f"Imported: {gene.id}  (class={out['robot_class']}, species={out['species']})")
    print(f"  segments: {len(gene.segments)} | actuated joints: {len(actuated)} "
          f"| end-effector: {ee.name if ee else '—'}")
    print(f"  links: {' -> '.join(s.name for s in gene.segments)}")
    print(f"  valid RobotGene: {'yes' if out['valid'] else 'NO (see warnings)'}")

    print("\nBackend support:")
    for backend, info in out["backend_support"].items():
        warns = ("; ".join(info["warnings"])) if info["warnings"] else "—"
        print(f"  {backend:7} {info['status']:24} {warns}")

    if out["warnings"]:
        print(f"\nWarnings ({len(out['warnings'])}) — review, don't ignore:")
        for w in out["warnings"]:
            print(f"  - {w}")
    else:
        print("\nNo warnings — clean import.")

    if args.save_gene:
        p = Path(args.save_gene); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(gene.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote gene -> {p}")

    if args.workspace:
        try:
            from virturoid.services.memory_db import MemoryDB
            from virturoid.services.species_discovery import auto_place_species
            db_path = Path(args.workspace) / "memory" / "virturoid_memory.db"
            with MemoryDB(db_path) as db:
                placement = auto_place_species(gene, db, source="imported")
            if placement["action"] == "classified":
                print(f"\nAuto-categorized: matches existing species {placement['species_pattern']!r} "
                      f"(morphology distance {placement['distance']:.3f}) — recorded as a variant.")
            else:
                print(f"\nAuto-categorized: NEW species {placement['species_pattern']!r} "
                      f"(parent {placement['parent']!r}, nearest distance "
                      f"{placement['distance'] if placement['distance'] is None else round(placement['distance'],3)}) "
                      "— added to the species tree by morphology.")
        except Exception as exc:  # noqa: BLE001 - recording is best-effort
            print(f"(could not record species node: {exc})", file=sys.stderr)

    return 0 if out["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
