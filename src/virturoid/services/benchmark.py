"""Robot benchmark + scorecard (startup plan §46 / §11.3 / §4 transferability wedge).

A single task-matched score (`task_matched_eval`) tells you a robot works on one scene; a benchmark tells
you how WELL and how ROBUSTLY across a standardized difficulty suite. `run_benchmark(gene)` runs the
robot's morphology-matched task at several difficulty levels and returns a scorecard: per-level metrics,
an aggregate **capability score** (mean normalized metric), and a **robustness** figure (worst/mean —
does it degrade gracefully or fall off a cliff?). This is the comparable, defensible score the product
wedge needs, built on the per-class evaluators. Needs MuJoCo.
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene


def run_benchmark(gene: RobotGene, *, prompt: str = "", controller_params: dict | None = None) -> dict:
    """Score ``gene`` across a difficulty suite for its class. Returns a scorecard dict."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    from virturoid.services.task_matched_eval import robot_kind
    cls, ee = gene.robot_class, gene.end_effector_type
    kind = robot_kind(gene)                               # dispatch by STRUCTURE, not class string
    levels: list[dict] = []

    if kind == "legged":
        from virturoid.services.locomotion_controller import run_locomotion_episode
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        for name, horizon in (("short", 1500), ("long", 3000)):
            r = run_locomotion_episode(mj, horizon=horizon)
            # normalize: walking >= ~0.5 m in the long run is full marks
            levels.append({"level": name, "metric": "distance_m", "value": r["distance_m"],
                           "score": min(1.0, r["distance_m"] / (0.3 if name == "short" else 0.6))})

    elif kind == "mobile":
        from virturoid.services.navigation_controller import run_navigation_episode
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        suites = {"near": [(0.8, 0.0), (0.0, 0.8), (-0.7, 0.3)],
                  "far": [(1.6, 0.0), (1.2, 1.0), (-1.3, 0.8)]}
        for name, goals in suites.items():
            reached = sum(run_navigation_episode(mj, g, [], horizon=4000)["status"] == "reached" for g in goals)
            levels.append({"level": name, "metric": "goal_reach_rate", "value": round(reached / len(goals), 3),
                           "score": reached / len(goals)})

    elif kind == "spray":
        from virturoid.services.spray_coverage import evaluate_spray_coverage
        for name, panel in (("small", {"x": [0.34, 0.42], "y": [-0.08, 0.08], "z": 0.18}),
                            ("wide", {"x": [0.30, 0.46], "y": [-0.12, 0.12], "z": 0.18})):
            r = evaluate_spray_coverage(gene, panel=panel, grid=4)
            levels.append({"level": name, "metric": "coverage", "value": r["coverage"], "score": r["coverage"]})

    else:  # manipulator pick-place across scene samples (robustness to object placement)
        from virturoid.services.task_runtime import select_task_spec, generate_task_scenes, evaluate_gene_on_task
        spec = select_task_spec(prompt or "sort blocks into bins")
        params = controller_params or {"kp": 9.0, "kd": 1.4, "phase_steps": 300}
        for name, seed in (("set_a", 0), ("set_b", 7)):
            scenes = generate_task_scenes(gene, spec, count=4, seed=seed)
            r = evaluate_gene_on_task(gene, spec, scenes, params=params)
            levels.append({"level": name, "metric": "success_rate", "value": r["success_rate"],
                           "score": r["success_rate"]})

    scores = [lv["score"] for lv in levels] or [0.0]
    capability = round(sum(scores) / len(scores), 3)
    robustness = round((min(scores) / (sum(scores) / len(scores))) if sum(scores) > 0 else 0.0, 3)
    return {"robot_class": cls, "task": levels[0]["metric"] if levels else "none",
            "levels": levels, "capability_score": capability, "robustness": robustness,
            "grade": _grade(capability)}


def _grade(score: float) -> str:
    return ("A" if score >= 0.85 else "B" if score >= 0.65 else "C" if score >= 0.4
            else "D" if score >= 0.2 else "F")
