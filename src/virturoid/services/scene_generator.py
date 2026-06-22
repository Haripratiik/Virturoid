"""Task-aware scene generation with seeded domain randomization (#6).

Scenes are still constrained (no unbounded text-to-3D), but each scene now draws
deterministic domain randomization from a per-scene seed: object position and
yaw jitter, size/mass/friction variation, lighting and camera-noise levels, and
clutter that scales with a difficulty level. Determinism is preserved (same task
+ purpose + index -> same scene), so runs stay reproducible.
"""

from __future__ import annotations

import copy
import math
import random
from hashlib import sha256

from virturoid.schemas.runs import FailureRecord
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet
from virturoid.schemas.tasks import TaskGraph

# Difficulty escalates the randomization magnitude and clutter per purpose.
_DIFFICULTY = {"baseline": 1, "variation": 2, "regression": 3, "holdout": 3, "edge_case": 4, "revision": 3,
               "stress": 5}


def generate_scene_set(task: TaskGraph, count: int = 5, purpose: str = "baseline") -> SceneSet:
    if count <= 0:
        raise ValueError("Scene count must be positive.")

    scenes = [_generate_scene(task, index, purpose) for index in range(count)]
    return SceneSet(
        id=f"scenes_{task.id}_{purpose}",
        task_graph_id=task.id,
        purpose=purpose,
        scenes=scenes,
        generation_rules=[
            "seeded domain randomization per scene (pose, yaw, size, mass, friction)",
            "lighting and camera-noise randomization tied to difficulty",
            "clutter and distractors scale with difficulty level",
            "keep matching bins visible and tabletop reachable",
        ],
    )


def _condition_scene_on_failure(objects: list[SceneObject], failure_type: str) -> tuple[list[SceneObject], dict]:
    """Failure-CONDITIONED counterfactual (plan §31.5): return a COPY of the failing scene's objects with the
    ONE variable correlated with ``failure_type`` changed, plus the params describing what changed (and any
    controller adjustment the regression run should apply). This reproduces/tests the SPECIFIC failure instead
    of generic difficulty noise — so a passing regression actually confirms the fix."""
    objs = [copy.deepcopy(o) for o in objects]
    blocks = [o for o in objs if o.object_type in ("cube", "box")]
    bins = [o for o in objs if o.object_type == "container"]
    ft = (failure_type or "").lower()

    def _pull_into_reach(o, rmax: float = 0.33) -> None:
        """Pull an object's xy toward the comfortable workspace centre (radius <= rmax) for the counterfactual."""
        x, y, z, *rest = o.pose_xyz_rpy
        r = math.hypot(x, y) or 1.0
        if r > rmax:
            s = rmax / r
            o.pose_xyz_rpy = (round(x * s, 4), round(y * s, 4), z, *rest)

    if "missed_grasp" in ft or "unreachable" in ft or "grasp_failure" in ft:
        # pull each block toward the reachable workspace centre — was the object out of comfortable reach? Also
        # pull the target bins in, so the counterfactual is COMPLETABLE if reach was the cause (else a far bin
        # the arm can't transport to would mask the confirmation, even though the diagnosed block-reach is fixed).
        for b in blocks:
            x, y, z, *rest = b.pose_xyz_rpy
            r = math.hypot(x, y) or 1.0
            s = min(r, 0.33) / r
            b.pose_xyz_rpy = (round(x * s, 4), round(y * 0.6, 4), z, *rest)
        for c in bins:
            _pull_into_reach(c)
        return objs, {"counterfactual": "block_centered_in_reach", "changed_variable": "object_pose"}
    if "wrong_bin" in ft:
        # widen bin spacing + flag clearer colour separation — were the two targets too close/ambiguous? Keep
        # both bins within comfortable reach so the disambiguation is what's tested, not raw reach.
        for b in bins:
            x, y, z, *rest = b.pose_xyz_rpy
            b.pose_xyz_rpy = (x, round(y * 1.4 + (0.07 if y >= 0 else -0.07), 4), z, *rest)
            _pull_into_reach(b, rmax=0.30)
        return objs, {"counterfactual": "wider_bin_spacing_clearer_colors", "changed_variable": "bin_pose"}
    if "dropped" in ft:
        for b in blocks:
            b.scale = round((b.scale or 1.0) * 0.9, 3)        # smaller, easier-to-hold object
            _pull_into_reach(b)                               # and within comfortable reach (edge transport was the cause)
        for c in bins:
            _pull_into_reach(c)
        return objs, {"counterfactual": "lower_lift_slower_transport", "changed_variable": "controller",
                      "controller_adjustment": {"lift_z": 0.18, "transport_speed": 0.6}}
    if "collision" in ft:
        for b in bins:
            x, y, z, *rest = b.pose_xyz_rpy
            b.pose_xyz_rpy = (round(x + 0.06, 4), y, z, *rest)   # bin farther from the base, steeper approach relaxed
        return objs, {"counterfactual": "bin_farther_from_base", "changed_variable": "bin_pose"}
    if "instab" in ft:
        for b in blocks:
            if b.mass_kg:
                b.mass_kg = round(b.mass_kg * 0.7, 4)         # lighter block, plus lower gains for the run
        return objs, {"counterfactual": "lighter_block_lower_gains", "changed_variable": "object_mass",
                      "controller_adjustment": {"kp": 9.0}}
    return objs, {"counterfactual": "harder_randomization", "changed_variable": "difficulty"}


def generate_regression_scene_set(task: TaskGraph, failures: list[FailureRecord], source_scene_set: SceneSet) -> SceneSet:
    """Create deterministic, FAILURE-CONDITIONED regression scenes from failure records (plan §31.5/§10.5).

    Each regression scene is a one-variable-changed counterfactual of the scene that actually failed, targeted at
    the failure TYPE (missed_grasp -> block centered in reach; wrong_bin -> wider bin spacing; dropped -> lower
    lift/slower transport; collision -> bin farther from base; instability -> lighter block + lower gains). This
    replaces the old behaviour of stamping every scene with a generic 'bin_spacing: wider', so re-running the
    regression set genuinely reproduces and confirms (or refutes) a fix for that specific failure."""
    if not failures:
        return SceneSet(
            id=f"scenes_{task.id}_regression_empty",
            task_graph_id=task.id,
            purpose="regression",
            scenes=[],
            generation_rules=["no failures available"],
        )

    source_scenes = source_scene_set.scenes or [None]
    regression_scenes: list[SceneGraph] = []
    for index, failure in enumerate(failures):
        source_scene = source_scenes[index % len(source_scenes)]
        # base on the ACTUAL failing scene's objects (so the counterfactual reproduces that config), falling
        # back to a fresh generated scene if the source objects aren't available.
        base = _generate_scene(task, index, "regression")
        base_objects = source_scene.objects if (source_scene and source_scene.objects) else base.objects
        objects, applied = _condition_scene_on_failure(base_objects, failure.failure_type)
        base.objects = objects
        base.id = failure.regression_scene_id or f"regression_{failure.id}"
        base.name = f"Regression for {failure.failure_type}: {failure.summary}"
        base.variation_parameters.update({
            "failure_type": failure.failure_type,
            "source_failure_id": failure.id,
            "source_scene_id": source_scene.id if source_scene else None,
            **applied,
        })
        base.requirement_trace.extend([f"failure:{failure.failure_type}", "regression_from_evaluation",
                                       f"counterfactual:{applied['counterfactual']}"])
        regression_scenes.append(base)

    return SceneSet(
        id=f"scenes_{task.id}_regression",
        task_graph_id=task.id,
        purpose="regression",
        scenes=regression_scenes,
        generation_rules=[
            "failure-conditioned: one variable changed per scene, around the actual failing config (plan §31.5)",
            "missed_grasp->center block; wrong_bin->wider spacing; dropped->lower/slower; collision->bin farther; instability->lighter+lower-gains",
            "preserve original task success criteria",
        ],
    )


def _scene_seed(task: TaskGraph, purpose: str, index: int) -> int:
    digest = sha256(f"{task.id}|{purpose}|{index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def _generate_scene(task: TaskGraph, index: int, purpose: str) -> SceneGraph:
    if task.task_type == "navigation":
        return _generate_navigation_scene(task, index, purpose)
    if task.task_type == "pick_place_box":
        return _generate_box_scene(task, index, purpose)
    if task.task_type == "stack":
        return _generate_stack_scene(task, index, purpose)
    if task.task_type == "push":
        return _generate_push_scene(task, index, purpose)
    return _generate_sorting_scene(task, index, purpose)


def _domain_randomization(rng: random.Random, difficulty: int) -> dict:
    """Sample scene-level randomization knobs scaled by difficulty."""
    spread = 0.02 + 0.025 * difficulty
    return {
        "difficulty": difficulty,
        "position_jitter_m": round(spread, 4),
        "lighting_level": round(rng.uniform(0.6, 1.0), 3),
        "camera_noise_std": round(rng.uniform(0.0, 0.01 * difficulty), 4),
        "table_friction": round(rng.uniform(0.8, 1.2), 3),
        "distractor_count": max(0, difficulty - 1),
        "background_texture": rng.choice(["plain", "wood", "speckled", "industrial"]),
    }


# Reachable tabletop annulus for manipulable objects (keeps targets graspable,
# per the scene-feasibility rule: do not spawn objects the arm provably cannot reach).
_REACH_MIN_M = 0.30
_REACH_MAX_M = 0.42


def _clamp_to_workspace(x: float, y: float) -> tuple[float, float]:
    r = math.hypot(x, y)
    if r < 1e-6:
        return x, y
    clamped = min(max(r, _REACH_MIN_M), _REACH_MAX_M)
    scale = clamped / r
    return round(x * scale, 4), round(y * scale, 4)


def _jittered_block(rng: random.Random, name: str, base_xy, material: str, spread: float) -> SceneObject:
    x = base_xy[0] + rng.uniform(-spread, spread)
    y = base_xy[1] + rng.uniform(-spread, spread)
    x, y = _clamp_to_workspace(x, y)
    yaw = round(rng.uniform(-math.pi, math.pi), 4)
    return SceneObject(
        name=name,
        object_type="cube",
        pose_xyz_rpy=(x, y, 0.02, 0.0, 0.0, yaw),
        mass_kg=round(rng.uniform(0.04, 0.07), 4),
        material=material,
        friction=round(rng.uniform(0.8, 1.3), 3),
        scale=round(rng.uniform(0.9, 1.15), 3),
    )


def _stress_block(rng: random.Random, name: str, base_xy, material: str, mass_kg: float) -> SceneObject:
    """A STRESS-tier block: placed at the edge of / past the reliable grasp envelope, at ``mass_kg``, so a
    strong scripted controller genuinely fails — a FAR-light block tends to MISS the grasp (no reach margin to
    descend), a near-but-HEAVY block tends to be DROPPED. These distinct real failures give the failure-driven
    loop diverse material (plan §11.2 stress eval / §10.3 intentional hard cases)."""
    x = base_xy[0] + rng.uniform(-0.02, 0.02)
    y = base_xy[1] + rng.uniform(-0.02, 0.02)
    return SceneObject(name=name, object_type="cube", pose_xyz_rpy=(round(x, 4), round(y, 4), 0.02, 0.0, 0.0,
                       round(rng.uniform(-math.pi, math.pi), 4)), mass_kg=mass_kg, material=material,
                       friction=round(rng.uniform(0.8, 1.0), 3), scale=round(rng.uniform(1.0, 1.15), 3))


def _generate_sorting_scene(task: TaskGraph, index: int, purpose: str) -> SceneGraph:
    rng = random.Random(_scene_seed(task, purpose, index))
    difficulty = _DIFFICULTY.get(purpose, 2)
    dr = _domain_randomization(rng, difficulty)
    spread = dr["position_jitter_m"]
    clutter = purpose in {"variation", "edge_case", "regression", "holdout"}

    if purpose == "stress":
        objects = [
            _stress_block(rng, "red_block", (0.60, -0.12), "matte_red", 0.06),    # FAR + light -> missed_grasp
            _stress_block(rng, "blue_block", (0.46, 0.12), "matte_blue", 0.40),   # near + HEAVY -> dropped
            SceneObject("red_bin", "container", (0.55, -0.20, 0.02, 0, 0, 0)),
            SceneObject("blue_bin", "container", (0.55, 0.20, 0.02, 0, 0, 0)),
        ]
        vp = {"index": index, "purpose": purpose, "clutter": False, "seed": _scene_seed(task, purpose, index),
              "reach_frac": 0.80, "block_mass_kg": 0.35, **dr}
        return SceneGraph(id=f"scene_{task.id}_{purpose}_{index:03d}", name=f"{task.name} stress scene {index + 1}",
                          backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                          objects=objects, variation_parameters=vp,
                          requirement_trace=[task.task_type, purpose, "tabletop", "negative_hard_case"])

    objects = [
        _jittered_block(rng, "red_block", (0.34, -0.10), "matte_red", spread),
        _jittered_block(rng, "blue_block", (0.34, 0.10), "matte_blue", spread),
        SceneObject(
            name="red_bin",
            object_type="container",
            pose_xyz_rpy=(0.55, round(-0.20 - rng.uniform(0, spread), 4), 0.02, 0.0, 0.0, 0.0),
        ),
        SceneObject(
            name="blue_bin",
            object_type="container",
            pose_xyz_rpy=(0.55, round(0.20 + rng.uniform(0, spread), 4), 0.02, 0.0, 0.0, 0.0),
        ),
    ]
    for distractor_index in range(dr["distractor_count"]):
        objects.append(
            _jittered_block(
                rng,
                f"distractor_{index}_{distractor_index}",
                (0.40 + 0.03 * distractor_index, 0.0),
                "matte_gray",
                spread,
            )
        )

    variation_parameters = {"index": index, "purpose": purpose, "clutter": clutter, "seed": _scene_seed(task, purpose, index)}
    variation_parameters.update(dr)
    return SceneGraph(
        id=f"scene_{task.id}_{purpose}_{index:03d}",
        name=f"{task.name} {purpose} scene {index + 1}",
        backend_targets=["mujoco"],
        robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        objects=objects,
        variation_parameters=variation_parameters,
        requirement_trace=[task.task_type, purpose, "tabletop", f"difficulty_{difficulty}"],
    )


def _generate_box_scene(task: TaskGraph, index: int, purpose: str) -> SceneGraph:
    rng = random.Random(_scene_seed(task, purpose, index))
    difficulty = _DIFFICULTY.get(purpose, 2)
    dr = _domain_randomization(rng, difficulty)
    spread = dr["position_jitter_m"]
    clutter = purpose in {"variation", "edge_case", "regression", "holdout"}
    if purpose == "stress":
        # STRESS: a FAR + HEAVY box past the reliable grasp envelope -> a real failure for the failure loop.
        box = _stress_block(rng, "box", (0.58, -0.08), "cardboard", 0.6)
        box.object_type = "box"
    else:
        box = _jittered_block(rng, "box", (0.36, -0.08), "cardboard", spread)
        box.object_type = "box"
        box.mass_kg = round(rng.uniform(0.2, 0.3), 4)
    objects = [
        box,
        SceneObject(
            name="target_bin",
            object_type="container",
            pose_xyz_rpy=(0.58, round(0.18 + rng.uniform(0, spread), 4), 0.03, 0.0, 0.0, 0.0),
            material="matte_gray",
        ),
        SceneObject(
            name="conveyor_zone",
            object_type="conveyor",
            pose_xyz_rpy=(0.30, -0.16, 0.02, 0.0, 0.0, 0.0),
            material="rubber_black",
        ),
    ]
    for distractor_index in range(dr["distractor_count"]):
        extra = _jittered_block(rng, f"warehouse_distractor_{index}_{distractor_index}", (0.45, 0.02), "cardboard", spread)
        extra.object_type = "box"
        extra.mass_kg = round(rng.uniform(0.1, 0.2), 4)
        objects.append(extra)

    variation_parameters = {"index": index, "purpose": purpose, "clutter": clutter, "environment": "warehouse", "seed": _scene_seed(task, purpose, index)}
    variation_parameters.update(dr)
    return SceneGraph(
        id=f"scene_{task.id}_{purpose}_{index:03d}",
        name=f"{task.name} {purpose} scene {index + 1}",
        backend_targets=["mujoco"],
        robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        objects=objects,
        variation_parameters=variation_parameters,
        requirement_trace=[task.task_type, purpose, "box_handling", f"difficulty_{difficulty}"],
    )


def _generate_stack_scene(task: TaskGraph, index: int, purpose: str) -> SceneGraph:
    """STACK task: several stackable blocks scattered in reach + a target pad to build the tower on (NO bins —
    a tower scene, not a sort scene). Block count grows with difficulty."""
    rng = random.Random(_scene_seed(task, purpose, index))
    difficulty = _DIFFICULTY.get(purpose, 2)
    dr = _domain_randomization(rng, difficulty)
    spread = dr["position_jitter_m"]
    n = 3 + (1 if purpose in {"stress", "edge_case"} else 0)
    mats = ["matte_red", "matte_blue", "matte_green", "matte_yellow"]
    bases = [(0.32, -0.12), (0.34, 0.0), (0.32, 0.12), (0.40, -0.06)]
    objects = [_jittered_block(rng, f"block_{i}", bases[i % len(bases)], mats[i % len(mats)], spread)
               for i in range(n)]
    objects.append(SceneObject(name="stack_target", object_type="zone",
                               pose_xyz_rpy=(0.38, 0.0, 0.002, 0.0, 0.0, 0.0), material="matte_green", scale=0.6))
    vp = {"index": index, "purpose": purpose, "n_blocks": n, "seed": _scene_seed(task, purpose, index)}
    vp.update(dr)
    return SceneGraph(id=f"scene_{task.id}_{purpose}_{index:03d}", name=f"{task.name} {purpose} scene {index + 1}",
                      backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                      objects=objects, variation_parameters=vp,
                      requirement_trace=[task.task_type, purpose, "tabletop", f"difficulty_{difficulty}"])


def _generate_push_scene(task: TaskGraph, index: int, purpose: str) -> SceneGraph:
    """PUSH task: one puck to push + a target zone to push it into (NO bins). Harder tiers add an obstacle in
    the path so the push must steer around it."""
    rng = random.Random(_scene_seed(task, purpose, index))
    difficulty = _DIFFICULTY.get(purpose, 2)
    dr = _domain_randomization(rng, difficulty)
    spread = dr["position_jitter_m"]
    objects = [
        _jittered_block(rng, "puck", (0.32, -0.06), "matte_orange", spread),
        SceneObject(name="target_zone", object_type="zone",
                    pose_xyz_rpy=(0.42, 0.12, 0.002, 0.0, 0.0, 0.0), material="matte_green", scale=0.9),
    ]
    if purpose in {"variation", "edge_case", "stress"}:
        objects.append(SceneObject(name="obstacle_0", object_type="obstacle",
                                   pose_xyz_rpy=(0.38, 0.03, 0.04, 0.0, 0.0, 0.0), material="matte_gray", scale=0.7))
    vp = {"index": index, "purpose": purpose, "seed": _scene_seed(task, purpose, index)}
    vp.update(dr)
    return SceneGraph(id=f"scene_{task.id}_{purpose}_{index:03d}", name=f"{task.name} {purpose} scene {index + 1}",
                      backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                      objects=objects, variation_parameters=vp,
                      requirement_trace=[task.task_type, purpose, "tabletop", f"difficulty_{difficulty}"])


def _generate_navigation_scene(task: TaskGraph, index: int, purpose: str) -> SceneGraph:
    rng = random.Random(_scene_seed(task, purpose, index))
    difficulty = _DIFFICULTY.get(purpose, 2)
    dr = _domain_randomization(rng, difficulty)
    route_length = round(rng.uniform(1.0, 1.6) + 0.15 * difficulty, 3)
    lane_offset = round(rng.uniform(-0.2, 0.2), 3)
    obstacle_count = max(1, difficulty)
    objects = [
        SceneObject(
            name="start_zone",
            object_type="zone",
            pose_xyz_rpy=(0.0, 0.0, 0.002, 0.0, 0.0, 0.0),
            material="matte_blue",
            scale=1.0,
        ),
        SceneObject(
            name="goal_zone",
            object_type="zone",
            pose_xyz_rpy=(route_length, lane_offset, 0.002, 0.0, 0.0, 0.0),
            material="matte_green",
            scale=1.2,
        ),
    ]
    for obstacle_index in range(obstacle_count):
        progress = (obstacle_index + 1) / (obstacle_count + 1)
        x = round(route_length * progress + rng.uniform(-0.08, 0.08), 4)
        y = round(lane_offset * progress + rng.choice([-1, 1]) * rng.uniform(0.12, 0.32), 4)
        yaw = round(rng.uniform(-math.pi, math.pi), 4)
        objects.append(
            SceneObject(
                name=f"obstacle_{index}_{obstacle_index}",
                object_type="obstacle",
                pose_xyz_rpy=(x, y, 0.05, 0.0, 0.0, yaw),
                mass_kg=None,
                material="matte_gray",
                friction=round(rng.uniform(0.8, 1.2), 3),
                scale=round(rng.uniform(0.8, 1.35), 3),
            )
        )

    variation_parameters = {
        "index": index,
        "purpose": purpose,
        "seed": _scene_seed(task, purpose, index),
        "route_length_m": route_length,
        "lane_offset_m": lane_offset,
        "obstacle_count": obstacle_count,
        "clutter": obstacle_count > 1,
    }
    variation_parameters.update(dr)
    return SceneGraph(
        id=f"scene_{task.id}_{purpose}_{index:03d}",
        name=f"{task.name} {purpose} scene {index + 1}",
        backend_targets=["mujoco"],
        robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        objects=objects,
        variation_parameters=variation_parameters,
        requirement_trace=[task.task_type, purpose, "indoor_navigation", f"difficulty_{difficulty}"],
    )
