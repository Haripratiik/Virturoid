"""CLI: compose a robot from building blocks (prompt -> recipe -> validated gene) + auto-categorize it.

Unlike ``virturoid.build`` (which selects a hand-coded species and co-designs it), this composes a robot
from primitives for ANY task and lets the species tree self-organize around it — the LEGO + vector-space
premise. With a workspace, the composed gene auto-places into the species tree by morphology.

    python -m virturoid.compose --prompt "warehouse arm to move 2 kg boxes with 0.9 m reach"
    python -m virturoid.compose --prompt "mobile base to deliver parts" --workspace build/desktop
    python -m virturoid.compose --prompt "sort blocks" --save-gene out/gene.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compose a robot from building blocks.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--reach", type=float, default=None, help="target reach (m)")
    ap.add_argument("--payload", type=float, default=None, help="payload (kg)")
    ap.add_argument("--save-gene", default=None)
    ap.add_argument("--workspace", default=None, help="if set, auto-place the gene in that workspace's species tree")
    ap.add_argument("--offline-template", action="store_true",
                    help="explicitly allow the legacy heuristic/template fallback when no LLM is available")
    ap.add_argument("--build", default=None, metavar="OUTPUT_DIR",
                    help="also compile the composed robot into a package and evaluate it in real physics")
    ap.add_argument("--co-design", action="store_true",
                    help="co-design the composed robot (body+controller) into a WORKING one before building")
    ap.add_argument("--gated-export", action="store_true",
                    help="with --build (manipulator): run the §15.3 tiered robustness gate on the built robot and "
                         "persist reports/evaluation_run.json so the readiness ledger ENFORCES export-readiness "
                         "(heavier; for an export verdict, not every quick build)")
    ap.add_argument("--evaluate", action="store_true",
                    help="evaluate the composed robot on its morphology-implied task (pick-place / navigation / coverage / locomotion)")
    ap.add_argument("--benchmark", action="store_true",
                    help="score the robot across a difficulty suite (capability + robustness + grade)")
    ap.add_argument("--run-task", action="store_true",
                    help="general task layer: plan a task from the prompt, verify it's feasible for this "
                         "robot, and run it (reports feasibility + gaps + score)")
    args = ap.parse_args(argv)

    from virturoid.services.morphology_composer import compose_robot, compose_working_robot

    controller_params = None
    try:
        if args.co_design and args.workspace:
            # DESIGN FLYWHEEL: warm-start from banked designs + bank the result, so runs self-improve.
            from virturoid.services.design_flywheel import co_design_with_memory
            from virturoid.services.memory_db import MemoryDB
            gene = compose_robot(args.prompt, reach_m=args.reach, payload_kg=args.payload,
                                 strict_llm=not args.offline_template)
            with MemoryDB(Path(args.workspace) / "memory" / "virturoid_memory.db") as db:
                res = co_design_with_memory(gene, args.prompt, db)
            gene = res["gene"]
            warm = " · warm-started" if res["warm_started"] else ""
            print(f"Co-designed (flywheel{warm}): {res['baseline_value']} -> {res['best_value']} "
                  f"(prior banked best {res['prior_best']}).")
        elif args.co_design:
            res = compose_working_robot(args.prompt, llm="auto", reach_m=args.reach, payload_kg=args.payload,
                                        strict_llm=not args.offline_template)
            gene = res["gene"]
            controller_params = res["controller_params"]   # build with the SAME brain co-design tuned (§22)
            print(f"Co-designed into a working robot: task success {res['baseline_success']} -> "
                  f"{res['success_rate']} (history {res['history']}).")
        else:
            gene = compose_robot(args.prompt, reach_m=args.reach, payload_kg=args.payload,
                                 strict_llm=not args.offline_template)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not compose a robot: {exc}", file=sys.stderr)
        return 1

    ee = gene.end_effector()
    src = getattr(gene, "design_source", "heuristic")
    print(f"Composed: {gene.id}  (class={gene.robot_class}, ee={gene.end_effector_type})")
    print(f"  body design: {'LLM-designed' if src == 'llm' else 'offline template (no LLM backend / fell back)'}")
    print(f"  blocks: {' -> '.join(s.name for s in gene.segments)}")
    print(f"  dof: {len(gene.actuated_joints())} | end-effector: {ee.name if ee else '—'} | valid: "
          f"{'yes' if not gene.validate() else 'NO'}")

    try:  # confirm it is buildable in physics
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        print(f"  compiles to MuJoCo: nbody={mj.nbody} nu={mj.nu}")
    except Exception as exc:  # noqa: BLE001 - MuJoCo optional
        print(f"  (MuJoCo compile check skipped: {exc})")

    if args.save_gene:
        p = Path(args.save_gene); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(gene.to_dict(), indent=2), encoding="utf-8")
        print(f"  wrote gene -> {p}")

    if args.build:
        try:
            from virturoid.services.gene_build import build_gene_package
            summary = build_gene_package(gene, args.prompt, Path(args.build), controller_params=controller_params,
                                         gated_export=args.gated_export)
            if summary.get("export_gate"):
                eg = summary["export_gate"]
                print(f"  export gate (§15.3 tiered): export_ready={eg['export_ready']} "
                      f"tiers={eg['tier_success']}" + (f" blockers={eg['blockers']}" if eg["blockers"] else ""))
            n_scenes = len(summary.get("episodes", [])) if isinstance(summary.get("episodes"), list) else summary.get("episodes", "?")
            tail = "" if controller_params else " (run with --co-design to tune it into a working robot)"
            print(f"  built + evaluated in real physics: task success {summary.get('success_rate')} "
                  f"over {n_scenes} scenes -> {args.build}{tail}")
        except Exception as exc:  # noqa: BLE001 - build needs MuJoCo; never crash the compose
            print(f"  (build skipped: {exc})", file=sys.stderr)

    if args.evaluate:
        try:
            from virturoid.services.task_matched_eval import evaluate_robot
            ev = evaluate_robot(gene, prompt=args.prompt, controller_params=controller_params)
            print(f"  task-matched eval: {ev['task']} -> {ev['metric']}={ev['value']}")
        except Exception as exc:  # noqa: BLE001 - eval needs MuJoCo
            print(f"  (evaluation skipped: {exc})", file=sys.stderr)

    if args.benchmark:
        try:
            from virturoid.services.benchmark import run_benchmark
            sc = run_benchmark(gene, prompt=args.prompt, controller_params=controller_params)
            print(f"  benchmark: grade {sc['grade']}  capability={sc['capability_score']}  "
                  f"robustness={sc['robustness']}  levels={[(l['level'], l['value']) for l in sc['levels']]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (benchmark skipped: {exc})", file=sys.stderr)

    if args.run_task:
        try:
            from virturoid.services.task_executor import evaluate_task
            r = evaluate_task(args.prompt, gene)
            if not r["feasible"]:
                print(f"  task: NOT feasible for this robot — {('; '.join(r['issues'][:2]))}")
            else:
                verdict = "succeeded" if r["success"] else "partial"
                print(f"  task ({r.get('task_source')}): {verdict} — skills {r.get('steps_planned')} · "
                      f"score {r['score']} ({r['goal_met']}/{r['goal_total']} goals)")
        except Exception as exc:  # noqa: BLE001
            print(f"  (run-task skipped: {exc})", file=sys.stderr)

    if args.workspace:
        try:
            from virturoid.services.memory_db import MemoryDB
            from virturoid.services.species_discovery import auto_place_species
            with MemoryDB(Path(args.workspace) / "memory" / "virturoid_memory.db") as db:
                pl = auto_place_species(gene, db, source="composed")
            if pl["action"] == "classified":
                print(f"  species tree: matches {pl['species_pattern']!r} (distance {pl['distance']:.3f}) — variant.")
            else:
                print(f"  species tree: NEW species {pl['species_pattern']!r} under {pl['parent']!r} (by morphology).")
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not auto-place: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
