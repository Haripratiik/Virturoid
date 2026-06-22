"""Hardware co-optimization against real task performance.

This is the hardware arm of the physical-AI co-design loop: instead of only
training a policy, the optimizer changes the robot's *body* (forearm and wrist
link lengths, actuator torque), rebuilds the real Robot Genome and MJCF for each
candidate, evaluates pick-and-place in real MuJoCo physics, and keeps the design
that maximizes task success. Scene layouts are held fixed across candidates so
the comparison isolates the effect of the hardware change.

The same pattern extends to scene/curriculum and policy parameters; this module
demonstrates the loop on the hardware dimension where the payoff (reach envelope
vs. block positions) is the clearest.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

from virturoid.fixtures.components import curated_component_library
from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.schemas.codesign import CodesignReport, DesignCandidate
from virturoid.services.mujoco_runner import mujoco_available
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.robot_arm_builder import build_reference_robot_arm
from virturoid.services.scene_generator import generate_scene_set
from virturoid.services.task_builder import build_task_graph

_SEARCH_SPACE = {
    "forearm_mm": (280.0, 430.0),
    "wrist_mm": (120.0, 240.0),
    "actuator_torque_nm": (8.0, 22.0),
}


def co_optimize_hardware(
    prompt: str = "Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
    eval_scene_count: int = 3,
    iterations: int = 4,
    candidates: int = 6,
    seed: int = 3,
    seed_design: dict | None = None,
) -> CodesignReport:
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to co-optimize hardware.")

    import numpy as np

    requirements = build_requirements_from_prompt(prompt)
    task = build_task_graph(requirements)
    components = curated_component_library()
    # Fixed evaluation scenes spanning the real near+far task distribution so the
    # chosen body balances both, not just a lucky subset.
    scenes = (
        generate_scene_set(task, count=eval_scene_count, purpose="variation").scenes
        + generate_scene_set(task, count=3, purpose="holdout").scenes
    )
    scene_objects = [
        {"id": s.id, "objects": {o.name: tuple(o.pose_xyz_rpy[:3]) for o in s.objects}} for s in scenes
    ]
    # A converged body must physically cover the whole task workspace, not just
    # the lucky eval subset, so the chosen design also passes the reach gate.
    required_reach = _required_reach(task)

    def evaluate(forearm_mm, wrist_mm, torque_nm) -> DesignCandidate:
        genome, cad = _build_variant(requirements, components, forearm_mm, wrist_mm, torque_nm)
        placed = total = success = 0
        for scene in scene_objects:
            placed_n, total_n, ok = _evaluate_scene(genome, cad, scene)
            placed += placed_n
            total += total_n
            success += 1 if ok else 0
        rate = round(success / len(scene_objects), 3) if scene_objects else 0.0
        return DesignCandidate(
            forearm_mm=round(forearm_mm, 1),
            wrist_mm=round(wrist_mm, 1),
            actuator_torque_nm=round(torque_nm, 1),
            success_rate=rate,
            blocks_placed=placed,
            blocks_total=total,
        )

    # Baseline = a naive first-guess body (short forearm, weak servo). The loop
    # discovers a better body from real task performance, not human tuning.
    baseline = evaluate(290.0, 130.0, 9.0)
    history = [baseline]

    def feasible(cand) -> bool:
        envelope = (cand.forearm_mm + cand.wrist_mm) / 1000.0 + 0.07
        return envelope >= required_reach + 0.02

    def better(cand, current) -> bool:
        # Prefer feasible bodies; among equal feasibility, prefer task success.
        if feasible(cand) != feasible(current):
            return feasible(cand)
        if cand.success_rate != current.success_rate:
            return cand.success_rate > current.success_rate
        return cand.blocks_placed > current.blocks_placed

    rng = np.random.default_rng(seed)
    # Start the search at the naive body, or warm-start from an LLM-proposed design
    # (the Robot Designer agent) when one is supplied — physics still arbitrates.
    if seed_design:
        mean = np.array([
            seed_design.get("forearm_mm", 290.0),
            seed_design.get("wrist_mm", 130.0),
            seed_design.get("actuator_torque_nm", 9.0),
        ])
    else:
        mean = np.array([290.0, 130.0, 9.0])
    std = np.array([55.0, 35.0, 4.0])
    best = baseline
    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(candidates, 3))
        evaluated = []
        for forearm_mm, wrist_mm, torque_nm in samples:
            cand = evaluate(*_clip_design(forearm_mm, wrist_mm, torque_nm))
            evaluated.append(cand)
            history.append(cand)
            if better(cand, best):
                best = cand
        # Re-center the search on the top designs (CEM update), favouring bodies
        # that cover the workspace so the search converges to a feasible design.
        evaluated.sort(key=lambda c: (feasible(c), c.success_rate, c.blocks_placed), reverse=True)
        elites = evaluated[: max(2, candidates // 3)]
        mean = np.array(
            [
                np.mean([c.forearm_mm for c in elites]),
                np.mean([c.wrist_mm for c in elites]),
                np.mean([c.actuator_torque_nm for c in elites]),
            ]
        )
        std = np.maximum(np.array([15.0, 12.0, 1.5]), std * 0.8)

    changed = {
        "forearm_mm": {"from": baseline.forearm_mm, "to": best.forearm_mm},
        "wrist_mm": {"from": baseline.wrist_mm, "to": best.wrist_mm},
        "actuator_torque_nm": {"from": baseline.actuator_torque_nm, "to": best.actuator_torque_nm},
    }
    report = CodesignReport(
        id="codesign_arm_hardware",
        task_prompt=prompt,
        eval_scene_count=len(scene_objects),
        iterations=iterations,
        candidates_evaluated=len(history),
        baseline=baseline,
        best=best,
        success_improvement=round(best.success_rate - baseline.success_rate, 3),
        search_space=_SEARCH_SPACE,
        history=history,
        changed_parameters=changed,
        notes=[
            "AI changed the robot's body (link lengths, actuator torque) to raise real task success.",
            "Each candidate was rebuilt and evaluated in real MuJoCo pick-and-place.",
            "Scene layouts were held fixed so the gain is attributable to the hardware change.",
        ],
    )
    validation = report.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{report.id} failed validation: {issues}")
    return report


def write_codesign_report(report: CodesignReport, package_dir: Path) -> Path:
    path = Path(package_dir) / "reports" / "codesign_optimization_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


_CONTROLLER_SPACE = {
    "engage_radius": (0.08, 0.16),
    "hover": (0.14, 0.26),
    "grasp_z": (0.045, 0.08),
    "kp": (6.0, 14.0),
}


def tune_controller(
    prompt: str,
    design: dict,
    eval_scene_count: int = 5,
    iterations: int = 4,
    candidates: int = 6,
    seed: int = 5,
) -> dict:
    """Search controller ("brain") parameters on a fixed body against real task success.

    The hardware is held at ``design`` while the grasp engage radius, hover and
    grasp heights, and PD gain are co-optimized over real pick-and-place rollouts.
    Returns the best controller params and the baseline/best success rates.
    """
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to tune the controller.")

    import numpy as np

    requirements = build_requirements_from_prompt(prompt)
    task = build_task_graph(requirements)
    components = curated_component_library()
    genome, cad = _build_variant(
        requirements, components, design["forearm_mm"], design["wrist_mm"], design["actuator_torque_nm"]
    )
    scenes = (
        generate_scene_set(task, count=eval_scene_count, purpose="variation").scenes
        + generate_scene_set(task, count=3, purpose="holdout").scenes
    )
    scene_objects = [{"id": s.id, "objects": {o.name: tuple(o.pose_xyz_rpy[:3]) for o in s.objects}} for s in scenes]

    keys = list(_CONTROLLER_SPACE)
    defaults = {"engage_radius": 0.12, "hover": 0.20, "grasp_z": 0.06, "kp": 9.0}

    def clip(vec):
        return {k: float(min(max(vec[i], _CONTROLLER_SPACE[k][0]), _CONTROLLER_SPACE[k][1])) for i, k in enumerate(keys)}

    def score(params):
        success = 0
        for s in scene_objects:
            _, _, ok = _evaluate_scene(genome, cad, s, controller_params=params)
            success += 1 if ok else 0
        return success / len(scene_objects)

    baseline_params = dict(defaults)
    baseline_success = score(baseline_params)
    rng = np.random.default_rng(seed)
    mean = np.array([defaults[k] for k in keys])
    std = np.array([(_CONTROLLER_SPACE[k][1] - _CONTROLLER_SPACE[k][0]) / 4 for k in keys])
    best_params, best_success = baseline_params, baseline_success
    for _ in range(iterations):
        samples = [clip(rng.normal(mean, std)) for _ in range(candidates)]
        scored = sorted(((score(p), p) for p in samples), key=lambda it: it[0], reverse=True)
        elites = scored[: max(2, candidates // 3)]
        mean = np.array([np.mean([e[1][k] for e in elites]) for k in keys])
        std = np.maximum(std * 0.8, np.array([(_CONTROLLER_SPACE[k][1] - _CONTROLLER_SPACE[k][0]) / 12 for k in keys]))
        if scored[0][0] > best_success:
            best_success, best_params = scored[0][0], scored[0][1]
    return {
        "baseline_params": baseline_params,
        "baseline_success": round(baseline_success, 3),
        "best_params": best_params,
        "best_success": round(best_success, 3),
        "eval_scene_count": len(scene_objects),
    }


def _required_reach(task) -> float:
    """Farthest block distance from the shoulder across the task workspace."""
    import math

    shoulder_z = 0.165  # table_top + base + upper riser (independent of forearm)
    farthest = 0.0
    for purpose, count in (("variation", 8), ("holdout", 5)):
        for scene in generate_scene_set(task, count=count, purpose=purpose).scenes:
            for obj in scene.objects:
                if obj.object_type in {"cube", "box"}:
                    x, y, z = obj.pose_xyz_rpy[0], obj.pose_xyz_rpy[1], obj.pose_xyz_rpy[2]
                    farthest = max(farthest, math.hypot(math.hypot(x, y), z - shoulder_z))
    return round(farthest, 4)


def _build_variant(requirements, components, forearm_mm, wrist_mm, torque_nm, morphology_template=None):
    build = build_reference_robot_arm(requirements, components, morphology_template=morphology_template)
    cad = copy.deepcopy(build.cad_models)
    for model in cad:
        if model.id == "cad_forearm_link":
            model.editable_parameters = {**model.editable_parameters, "link_length_mm": forearm_mm}
        elif model.id == "cad_wrist_camera_mount":
            model.editable_parameters = {**model.editable_parameters, "link_length_mm": wrist_mm}
    genome = build.robot_genome
    # Push the chosen actuator torque into joint efforts (flows into MJCF forcerange).
    for joint in genome.joints:
        if joint.limit:
            joint.limit = replace(joint.limit, effort=torque_nm)
    return genome, cad


def _evaluate_scene(genome, cad, scene, controller_params: dict | None = None):
    import mujoco

    from virturoid.services.mujoco_exporter import robot_and_single_scene_to_mjcf
    from virturoid.services.pick_place_controller import run_pick_place_episode
    from virturoid.schemas.scenes import SceneGraph, SceneObject

    scene_graph = SceneGraph(
        id=scene["id"],
        name=scene["id"],
        backend_targets=["mujoco"],
        robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        objects=[
            SceneObject(name=name, object_type=_object_type(name), pose_xyz_rpy=(*xyz, 0.0, 0.0, 0.0))
            for name, xyz in scene["objects"].items()
        ],
    )
    xml = robot_and_single_scene_to_mjcf(genome, scene_graph, "codesign", cad)
    model = mujoco.MjModel.from_xml_string(xml)
    outcome = run_pick_place_episode(model, {"objects": scene["objects"]}, params=controller_params)
    return outcome["placed_count"], outcome["block_count"], outcome["status"] == "success"


def _object_type(name: str) -> str:
    if "bin" in name:
        return "container"
    return "cube"


def _clip_design(forearm_mm, wrist_mm, torque_nm):
    f = min(max(float(forearm_mm), _SEARCH_SPACE["forearm_mm"][0]), _SEARCH_SPACE["forearm_mm"][1])
    w = min(max(float(wrist_mm), _SEARCH_SPACE["wrist_mm"][0]), _SEARCH_SPACE["wrist_mm"][1])
    t = min(max(float(torque_nm), _SEARCH_SPACE["actuator_torque_nm"][0]), _SEARCH_SPACE["actuator_torque_nm"][1])
    return f, w, t
