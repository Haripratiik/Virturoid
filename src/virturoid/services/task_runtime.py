"""Task runtime (plan Phase 2): run DIVERSE pick-place-family tasks on one robot.

Phase 1 turns a prompt into a structured objective; this turns a task into something the
*existing scripted controller* actually executes — generalizing the hard-coded color-sort into
a spec-driven ``object -> target`` family. So a gene can now be evaluated on **sort**,
**place-to-target**, or **transport-to-zone** (the Customer-1/3 scenarios) with the controller
we already have, on CPU, no RL needed. Contact-heavy tasks (spray-coverage, insertion) wait
for learned skills (RL on MJX); this proves task diversity end to end now.

Objects + targets are placed in the compiled robot's reachable workspace (robot-matched), and
the controller is handed the spec's ``assignments`` so success = each object reaching its
assigned target.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PickPlaceTaskSpec:
    task_type: str
    manipulables: list[str]              # cube objects to move
    targets: list[str]                   # target zones/containers
    assignments: dict[str, str]          # object -> target
    materials: dict[str, str] = field(default_factory=dict)  # name -> 'red'/'blue'/'gray'


def builtin_task_specs() -> dict[str, PickPlaceTaskSpec]:
    return {
        "pick_place_sort": PickPlaceTaskSpec(
            "pick_place_sort", ["red_block", "blue_block"], ["red_bin", "blue_bin"],
            {"red_block": "red_bin", "blue_block": "blue_bin"},
            {"red_block": "red", "blue_block": "blue", "red_bin": "red", "blue_bin": "blue"}),
        "place_to_target": PickPlaceTaskSpec(
            "place_to_target", ["box"], ["target_zone"], {"box": "target_zone"},
            {"box": "gray", "target_zone": "gray"}),
        "transport_to_zone": PickPlaceTaskSpec(
            "transport_to_zone", ["box"], ["drop_zone"], {"box": "drop_zone"},
            {"box": "gray", "drop_zone": "gray"}),
    }


def select_task_spec(prompt: str) -> PickPlaceTaskSpec:
    """Pick the pick-place-family task a prompt implies (deterministic, offline-safe)."""
    p = (prompt or "").lower()
    specs = builtin_task_specs()
    if "sort" in p or ("block" in p and "bin" in p):
        return specs["pick_place_sort"]
    if any(w in p for w in ("transport", "carry", "deliver", "bring", "move it", "across")):
        return specs["transport_to_zone"]
    if any(w in p for w in ("lift", "place", "shelf", "onto", "stack", "load", "pick up", "pick and place")):
        return specs["place_to_target"]
    return specs["place_to_target"]


def generate_task_scenes(gene, spec: PickPlaceTaskSpec, count: int = 6, *, seed: int = 0):
    """Robot-matched scenes for an arbitrary object->target task (objects in the reachable set)."""
    import random

    from virturoid.schemas.scenes import SceneGraph, SceneObject
    from virturoid.services.gene_build import generate_reachable_scenes, robot_reachable_points

    pts = robot_reachable_points(gene)
    need = len(spec.manipulables) + len(spec.targets)
    if len(pts) < need:
        return generate_reachable_scenes(gene, count=count, seed=seed)  # fallback to default sort scene

    def far_enough(p, chosen, d=0.16):
        return all((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 >= d * d for c in chosen)

    def mat(name: str, kind: str) -> str:
        m = spec.materials.get(name, "gray")
        return f"{m}_{kind}"

    scenes = []
    for i in range(count):
        rng = random.Random(seed + i)
        pool = pts[:]
        rng.shuffle(pool)
        chosen = []
        for pt in pool:
            if far_enough(pt, chosen):
                chosen.append(pt)
            if len(chosen) == need:
                break
        if len(chosen) < need:
            chosen = pool[:need]
        objs = []
        k = 0
        for name in spec.manipulables:
            x, y = chosen[k]; k += 1
            objs.append(SceneObject(name, "cube", (x, y, 0.02, 0, 0, round(rng.uniform(-3.14, 3.14), 3)),
                                    mass_kg=round(rng.uniform(0.04, 0.07), 4), material=mat(name, "block"),
                                    friction=1.0, scale=1.0))
        for name in spec.targets:
            x, y = chosen[k]; k += 1
            objs.append(SceneObject(name, "container", (x, y, 0.0, 0, 0, 0), material=mat(name, "bin")))
        scenes.append(SceneGraph(
            id=f"{spec.task_type}_{gene.id}_{i}", version="0.1.0",
            name=f"{spec.task_type} scene {i + 1}", backend_targets=["mujoco"],
            robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0), objects=objs,
            variation_parameters={"task_type": spec.task_type}, requirement_trace=[spec.task_type]))
    return scenes


_EV: dict = {}


def _init_eval_worker(gene_dict, assignments, params):
    import warnings
    warnings.filterwarnings("ignore")
    from virturoid.schemas.gene import RobotGene
    _EV["gene"] = RobotGene.from_dict(gene_dict)
    _EV["assign"] = assignments
    _EV["params"] = params


def _eval_one_scene(payload):
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import run_pick_place_episode
    scene_id, objects_list = payload
    model = mujoco.MjModel.from_xml_string(compile_gene_with_scene(_EV["gene"], objects_list))
    objs = {o.name: tuple(o.pose_xyz_rpy[:3]) for o in objects_list}
    out = run_pick_place_episode(model, {"objects": objs, "assignments": _EV["assign"]}, params=_EV["params"])
    return {"scene_id": scene_id, **{k: out[k] for k in ("status", "failure_label", "placed_count", "block_count")},
            "metrics": out.get("metrics", {})}


def evaluate_gene_on_task(gene, spec: PickPlaceTaskSpec, scenes, params: dict | None = None,
                          *, workers: int | None = None) -> dict:
    """Run the scripted controller on a gene for an arbitrary object->target task. Needs MuJoCo. Each scene is
    an INDEPENDENT episode, so they run in PARALLEL across CPU cores in production (``workers``: default all
    but one core; serial under pytest)."""
    import mujoco

    from virturoid.services.gene_build import default_controller_params
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import run_pick_place_episode

    if params is None:
        params = default_controller_params(gene)
    if workers is None:
        import multiprocessing as mp
        import sys
        workers = 1 if "pytest" in sys.modules else max(1, mp.cpu_count() - 1)

    episodes = []
    pool = None
    if workers > 1 and len(scenes) > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=min(workers, len(scenes)), initializer=_init_eval_worker,
                            initargs=(gene.to_dict(), spec.assignments, params))
        except Exception:  # noqa: BLE001 - fall back to serial
            pool = None
    if pool is not None:
        try:
            episodes = list(pool.map(_eval_one_scene, [(s.id, list(s.objects)) for s in scenes]))
        except Exception:  # noqa: BLE001 - a worker died -> redo serially
            episodes = []
        finally:
            pool.close()
            pool.join()
    if not episodes:
        for scene in scenes:
            model = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, scene.objects))
            objects = {o.name: tuple(o.pose_xyz_rpy[:3]) for o in scene.objects}
            out = run_pick_place_episode(model, {"objects": objects, "assignments": spec.assignments}, params=params)
            episodes.append({"scene_id": scene.id,
                             **{k: out[k] for k in ("status", "failure_label", "placed_count", "block_count")},
                             "metrics": out.get("metrics", {})})
    n = len(episodes)
    return {
        "task_type": spec.task_type, "species": gene.species, "robot_class": gene.robot_class,
        # DISCLOSE the grasp model so the readiness gate can't read this as a full-contact physics pass: the
        # scripted pick-place controller PINS the block to the gripper on contact (idealized engagement), it is
        # NOT a learned closed-finger contact grasp. Honest export-readiness must reflect that.
        "grasp_model": "idealized_pin",
        "episodes": episodes,
        "success_rate": round(sum(1 for e in episodes if e["status"] == "success") / n, 3) if n else 0.0,
        "blocks_placed": sum(e["placed_count"] for e in episodes),
        "blocks_total": sum(e["block_count"] for e in episodes),
    }
