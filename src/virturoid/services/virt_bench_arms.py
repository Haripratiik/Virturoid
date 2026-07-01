"""VIRT-Bench arms (breakthrough plan WS3, §7) — run competing DESIGN methods against the frozen tasks and score
BOTH with the independent verifier, so neither arm scores itself and the A->B delta is the *measured* value of
the search harness (the whole "beats Claude+MCP" claim in one number).

  Arm A  the fixed-pipeline baseline: compose the body, drive it with the DEFAULT controller, no search. This is
         the "Claude + MCP: one move" shape -- a single build with a stock gait, no closed loop.
  Arm B  our design-search harness: search the CPG gait direction on the cheap CPU rung with honesty-gate
         selection (services/design_search + the LLM-free cpg_grid_proposer), then submit the SELECTED best.

Both submit to virt_bench.verify_submission, which RE-RUNS the artifact in physics and applies the H1 gate --
the arm's self-claim is never consulted. This is the proof AND the development compass: a task BOTH arms fail is
the next build priority (e.g. forward hexapod locomotion, which needs the learned GPU residual, not just a CPG).
"""

from __future__ import annotations

from virturoid.services.virt_bench import get_task, list_tasks, verify_submission


def _task_body(task: dict):
    """Deterministically compose the body a locomotion task calls for (returns None for unsupported families)."""
    if task["family"] != "locomotion":
        return None
    from virturoid.services.steerable_body import steerable_quadruped
    if task["id"] == "L1_quad_walk":
        return steerable_quadruped(n_legs=4)
    if task["id"] == "L2_hex_walk":
        return steerable_quadruped(n_legs=6, bilateral=True)
    return steerable_quadruped(n_legs=4)                       # sensible default for any other legged task


def _zero_policy_with_cpg(gene, cpg: dict | None):
    """A zero-residual MorphPolicy carrying ``cpg`` as its gait prior -> recipe_rollout_morph drives the pure CPG
    (the verifier re-runs THIS). Isolates the searched gait from any learned residual."""
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import MorphPolicy, compiled_model, robot_mjcf
    graph = encode_robot(compiled_model(robot_mjcf(gene)))
    pol = MorphPolicy(graph.feature_dim)
    pol.set_params(np.zeros(pol.n_params))
    pol.cpg = dict(cpg) if cpg else None
    return pol


def run_arm_a(task_id: str, *, steps: int = 600) -> dict:
    """Baseline arm: compose the body + the DEFAULT CPG gait, NO search -> independent verify."""
    from virturoid.services.morph_policy import CPG_DEFAULT
    task = get_task(task_id)
    gene = _task_body(task)
    if gene is None:
        return {"task": task_id, "arm": "A", "verified_pass": False, "failure_mode": "unsupported_task",
                "metrics": {}, "method": "fixed-pipeline (no search)"}
    res = verify_submission(task_id, gene, _zero_policy_with_cpg(gene, CPG_DEFAULT), steps=steps)
    res["arm"] = "A"; res["method"] = "fixed-pipeline: default CPG, no search"
    return res


def run_arm_b(task_id: str, *, steps: int = 600, max_evals: int = 12, on_node=None) -> dict:
    """Harness arm: search the CPG gait direction on the CPU rung (honesty-gate selection) -> verify the SELECTED
    best. Returns the verifier's verdict plus ``searched`` (the winning CPG) and ``n_evals`` (search cost)."""
    from virturoid.services.design_search import run_design_search
    from virturoid.services.search_adapters import cpg_grid_proposer, make_locomotion_evaluate
    task = get_task(task_id)
    gene = _task_body(task)
    if gene is None:
        return {"task": task_id, "arm": "B", "verified_pass": False, "failure_mode": "unsupported_task",
                "metrics": {}, "method": "CPG-search harness"}
    evaluate = make_locomotion_evaluate(gene, steps=steps)
    report = run_design_search(propose=cpg_grid_proposer(), evaluate=evaluate, task_type="locomotion",
                               gates=task["gates"], max_evals=max_evals, on_node=on_node)
    best_cpg = (report.best.result.get("cpg") if report.best else None)
    res = verify_submission(task_id, gene, _zero_policy_with_cpg(gene, best_cpg), steps=steps)
    res["arm"] = "B"; res["method"] = f"CPG-search harness ({report.n_evals} evals, {report.stopped_reason})"
    res["searched"] = best_cpg; res["n_evals"] = report.n_evals
    return res


def run_dev_scoreboard(*, steps: int = 600, max_evals: int = 12) -> dict:
    """Run both arms over the dev-split LOCOMOTION tasks; return an honest, verifier-scored A-vs-B scoreboard.
    ``B_solved - A_solved`` is the measured value of the search harness on this slice."""
    rows = []
    for task in list_tasks("dev"):
        if task["family"] != "locomotion":
            continue
        a = run_arm_a(task["id"], steps=steps)
        b = run_arm_b(task["id"], steps=steps, max_evals=max_evals)
        rows.append({"task": task["id"], "A_pass": bool(a["verified_pass"]), "B_pass": bool(b["verified_pass"]),
                     "A_fwd": a["metrics"].get("forward_m"), "B_fwd": b["metrics"].get("forward_m"),
                     "B_searched": b.get("searched"), "B_n_evals": b.get("n_evals")})
    return {"rows": rows, "A_solved": sum(r["A_pass"] for r in rows),
            "B_solved": sum(r["B_pass"] for r in rows), "n_tasks": len(rows),
            "harness_delta": sum(r["B_pass"] for r in rows) - sum(r["A_pass"] for r in rows)}
