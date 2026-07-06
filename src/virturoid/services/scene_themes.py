"""AI-native SCENE THEMES (docs/ai_native_plan.md P4) — the vocabulary that makes "change the scene to a
house instead of a warehouse" a THEME SWAP, not a regeneration. A theme is the *look* axis (floor/wall
materials + a furniture set); it composes with the task-structure family (ProcTHOR family+seed pattern,
arXiv:2206.06994), so swapping the theme keeps the task, robot, spawn, and seed and changes only the setting.

Follows the repo's constraint-authored scene model (a ``SceneGraph`` of ``SceneObject``s with materials): a
theme swap re-materials the floor/walls and replaces the furniture set, re-running the validity gate. Themes
are DATA, not per-scene code — new themes are dict entries.
"""
from __future__ import annotations

# theme -> {floor, wall materials; a furniture set as (category, (sx,sy,sz)) footprints an agent must avoid}.
THEMES: dict[str, dict] = {
    "warehouse": {"floor": "concrete", "wall": "steel", "ambient": "industrial",
                  "furniture": [("pallet", (0.9, 0.9, 0.15)), ("shelf", (1.6, 0.5, 1.8)),
                                ("crate", (0.6, 0.6, 0.6)), ("forklift_zone", (1.2, 0.8, 0.05))]},
    "house": {"floor": "wood", "wall": "plaster", "ambient": "warm",
              "furniture": [("sofa", (1.8, 0.9, 0.8)), ("coffee_table", (1.0, 0.6, 0.45)),
                            ("bookshelf", (0.9, 0.35, 1.9)), ("rug", (2.0, 1.4, 0.02))]},
    "kitchen": {"floor": "tile", "wall": "tile", "ambient": "bright",
                "furniture": [("counter", (2.2, 0.65, 0.9)), ("island", (1.2, 0.9, 0.9)),
                              ("fridge", (0.8, 0.8, 1.9)), ("stool", (0.4, 0.4, 0.75))]},
    "lab": {"floor": "epoxy", "wall": "panel", "ambient": "clean",
            "furniture": [("workbench", (2.0, 0.8, 0.9)), ("rack", (1.0, 0.5, 1.9)),
                          ("cart", (0.7, 0.5, 0.9)), ("cabinet", (0.9, 0.6, 1.6))]},
    "yard": {"floor": "grass", "wall": "fence", "ambient": "outdoor",
             "furniture": [("planter", (0.8, 0.8, 0.6)), ("bench", (1.6, 0.5, 0.5)),
                           ("rock", (0.6, 0.6, 0.5)), ("shed", (1.4, 1.2, 2.0))]},
}


def theme_names() -> list[str]:
    return sorted(THEMES)


def _rng(seed: int):
    import numpy as np
    return np.random.default_rng(int(seed))


def build_scene(task: str = "navigation", theme: str = "warehouse", *, seed: int = 0, room_m: float = 5.0):
    """Build a valid themed ROOM ``SceneGraph`` for a task: a floor + 4 perimeter walls + a themed furniture
    set (obstacles) + a robot spawn + a goal zone. The task-structure axis (obstacle count/layout) is seeded;
    the theme axis is the look. Keeps them separable so ``swap_theme`` changes only materials + furniture."""
    from virturoid.schemas.scenes import SceneGraph, SceneObject
    theme = theme if theme in THEMES else "warehouse"
    spec = THEMES[theme]
    rng = _rng(seed)
    W = float(room_m); half = W / 2.0
    objs: list = [SceneObject(name="floor", object_type="floor", category="floor", material=spec["floor"],
                              size_xyz=(W, W, 0.05), pose_xyz_rpy=(0.0, 0.0, -0.025, 0, 0, 0))]
    th = 0.1; h = 2.44
    for nm, (x, y) in (("wall_n", (0.0, half)), ("wall_s", (0.0, -half)),
                       ("wall_e", (half, 0.0)), ("wall_w", (-half, 0.0))):
        sx = (W, th, h) if "n" in nm[-1] or "s" in nm[-1] else (th, W, h)
        objs.append(SceneObject(name=nm, object_type="wall", category="wall", material=spec["wall"],
                                size_xyz=sx, pose_xyz_rpy=(x, y, h / 2.0, 0, 0, 0)))
    # themed furniture as obstacles, spaced on a jittered grid inside the room (keep a clear central spawn lane)
    furn = spec["furniture"]
    n = 3 + int(rng.integers(0, 2))
    for i in range(n):
        cat, (sx, sy, sz) = furn[i % len(furn)]
        px = float(rng.uniform(-half + 0.8, half - 0.8)); py = float(rng.uniform(-half + 0.8, half - 0.8))
        if abs(px) < 0.6 and abs(py) < 0.6:
            px += 1.2
        objs.append(SceneObject(name=f"furn{i}_{cat}", object_type="obstacle", category=cat,
                                material=spec["wall"] if cat in ("bookshelf", "shelf", "fridge") else spec["floor"],
                                size_xyz=(sx, sy, sz), pose_xyz_rpy=(round(px, 2), round(py, 2), sz / 2.0, 0, 0, 0)))
    if task == "navigation":
        objs.append(SceneObject(name="goal", object_type="zone", category="goal", material="matte_green",
                                size_xyz=(0.4, 0.4, 0.02), pose_xyz_rpy=(half - 0.6, half - 0.6, 0.01, 0, 0, 0)))
    else:  # a table for a manipulation task
        objs.append(SceneObject(name="table", object_type="table", category="table", material=spec["floor"],
                                size_xyz=(1.0, 0.6, 0.4), pose_xyz_rpy=(0.6, 0.0, 0.2, 0, 0, 0)))
    return SceneGraph(id=f"scene_{theme}_{task}_{int(seed)}", name=f"{theme}_{task}", backend_targets=["mujoco"],
                      robot_spawn_xyz_rpy=(-half + 0.8, -half + 0.8, 0.1, 0, 0, 0), objects=objs,
                      variation_parameters={"theme": theme, "task": task, "seed": int(seed)},
                      bounds=((-half, -half, 0.0), (half, half, h)))


def apply_theme(scene_graph, theme: str):
    """RE-THEME an existing scene in place-ish (returns a new graph): re-material floor+walls and swap the
    furniture set to the new theme's, preserving the layout positions + task objects (goal/table) + spawn.
    This is the 'house instead of warehouse' op — everything but the look is kept."""
    from virturoid.schemas.scenes import SceneGraph, SceneObject
    if theme not in THEMES:
        raise ValueError(f"unknown theme '{theme}'; known: {theme_names()}")
    spec = THEMES[theme]
    new_objs: list = []
    fi = 0
    for o in scene_graph.objects:
        cat = (o.category or o.object_type or "")
        if cat == "floor":
            new_objs.append(SceneObject(name=o.name, object_type=o.object_type, category=o.category,
                                        material=spec["floor"], size_xyz=o.size_xyz, pose_xyz_rpy=o.pose_xyz_rpy))
        elif cat == "wall":
            new_objs.append(SceneObject(name=o.name, object_type=o.object_type, category=o.category,
                                        material=spec["wall"], size_xyz=o.size_xyz, pose_xyz_rpy=o.pose_xyz_rpy))
        elif o.object_type == "obstacle":                    # swap the furniture identity but KEEP its position
            cat2, (sx, sy, sz) = spec["furniture"][fi % len(spec["furniture"])]; fi += 1
            x, y = o.pose_xyz_rpy[0], o.pose_xyz_rpy[1]
            new_objs.append(SceneObject(name=f"furn{fi-1}_{cat2}", object_type="obstacle", category=cat2,
                                        material=spec["wall"] if cat2 in ("bookshelf", "shelf", "fridge") else spec["floor"],
                                        size_xyz=(sx, sy, sz), pose_xyz_rpy=(x, y, sz / 2.0, 0, 0, 0)))
        else:                                                # task objects (goal/table), spawn markers: keep as-is
            new_objs.append(o)
    vp = dict(scene_graph.variation_parameters or {}); vp["theme"] = theme
    return SceneGraph(id=f"scene_{theme}_{vp.get('task', 'scene')}_{vp.get('seed', 0)}",
                      name=f"{theme}_{vp.get('task', 'scene')}", backend_targets=list(scene_graph.backend_targets),
                      robot_spawn_xyz_rpy=scene_graph.robot_spawn_xyz_rpy, objects=new_objs,
                      variation_parameters=vp, bounds=scene_graph.bounds)
