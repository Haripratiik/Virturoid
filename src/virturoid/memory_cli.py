"""Inspect and export the Virturoid Memory database.

The memory DB is what the AI agents learn from across builds (startup plan §13). This
CLI lets you see what's accumulated and export the supervised training set used to
fine-tune the local model (the data flywheel).

    python -m virturoid.memory_cli stats
    python -m virturoid.memory_cli recent [-n 10]
    python -m virturoid.memory_cli similar "sort blocks into bins"
    python -m virturoid.memory_cli failures --class manipulator --task pick_place_sort
    python -m virturoid.memory_cli export --min-success 0.8 --out build/memory/train.jsonl

By default it reads build/memory/virturoid_memory.db (override with --db).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect/export Virturoid Memory.")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="path to the memory .db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="row counts per memory table")

    p_recent = sub.add_parser("recent", help="most recent runs")
    p_recent.add_argument("-n", type=int, default=10)

    p_sim = sub.add_parser("similar", help="prior runs similar to a prompt")
    p_sim.add_argument("prompt")
    p_sim.add_argument("-n", type=int, default=5)

    p_fail = sub.add_parser("failures", help="known failures for a class+task")
    p_fail.add_argument("--class", dest="robot_class", required=True)
    p_fail.add_argument("--task", dest="task_type", required=True)

    p_exp = sub.add_parser("export", help="export the LoRA training dataset (JSONL)")
    p_exp.add_argument("--min-success", type=float, default=0.8)
    p_exp.add_argument("--out", default="build/memory/train.jsonl")

    sub.add_parser("tree", help="print the robot species tree")

    p_vault = sub.add_parser("vault", help="export the species tree as an Obsidian vault")
    p_vault.add_argument("--out", default="build/memory/species_vault")

    p_prune = sub.add_parser("prune", help="merge near-duplicate + trim redundant species")
    p_prune.add_argument("--apply", action="store_true", help="apply changes (otherwise dry-run)")

    args = ap.parse_args(argv)
    db_path = Path(args.db)
    if not db_path.exists() and args.cmd != "export":
        print(f"No memory DB at {db_path}. Run a build first (it's created on the first run).")
        return 1

    with MemoryDB(db_path) as db:
        if args.cmd == "stats":
            print(json.dumps(db.stats(), indent=2))
        elif args.cmd == "recent":
            rows = db.conn.execute(
                "SELECT prompt, robot_class, task_type, success_rate, succeeded, backend, created_at "
                "FROM runs ORDER BY id DESC LIMIT ?", (args.n,)
            ).fetchall()
            for r in rows:
                print(f"[{r['created_at']}] {r['robot_class']}/{r['task_type']} "
                      f"success={r['success_rate']} backend={r['backend']} :: {r['prompt'][:70]}")
            if not rows:
                print("(no runs yet)")
        elif args.cmd == "similar":
            for r in db.similar_runs(args.prompt, limit=args.n):
                print(f"sim={r['similarity']:.2f} {r['robot_class']}/{r['task_type']} "
                      f"success={r['success_rate']} :: {r['prompt'][:70]}")
        elif args.cmd == "failures":
            fails = db.known_failures(args.robot_class, args.task_type)
            for f in fails:
                print(f"{f['label']} x{f['total']}  cause={f['root_cause']}  fix={f['recommended_action']}")
            if not fails:
                print("(no failures recorded)")
        elif args.cmd == "export":
            n = db.export_training_jsonl(Path(args.out), min_success=args.min_success)
            print(f"exported {n} training rows (success >= {args.min_success}) -> {args.out}")
        elif args.cmd == "tree":
            nodes = db.species_tree_nodes()
            by_parent = {}
            for n in nodes:
                by_parent.setdefault(n.get("parent_species") or "", []).append(n)
            roots = sorted({(n.get("parent_species") or "") for n in nodes} - {n["species_pattern"] for n in nodes})

            def walk(p, depth):
                for c in sorted(by_parent.get(p, []), key=lambda x: x["species_pattern"]):
                    mark = "" if c.get("buildable") else "  (not buildable yet)"
                    print(f"{'  ' * depth}- {c['species_pattern']} [{c.get('robot_class')}]{mark}")
                    walk(c["species_pattern"], depth + 1)

            for root in roots:
                print(f"* {root or '(root)'}")
                walk(root, 1)
            if not nodes:
                print("(species tree empty — run a build first)")
        elif args.cmd == "vault":
            from virturoid.services.species_vault import export_species_vault

            n = export_species_vault(db, Path(args.out))
            print(f"exported {n} species notes -> {args.out} (open in Obsidian)")
        elif args.cmd == "prune":
            from virturoid.services.species_maintenance import prune_tree

            result = prune_tree(db, dry_run=not args.apply)
            verb = "applied" if args.apply else "would (dry-run; use --apply)"
            print(f"{verb}: {result.note}")
            print(f"  merge: {result.merged}")
            print(f"  trim:  {result.trimmed}")
            if result.skipped:
                print(f"  skipped (gated): {result.skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
