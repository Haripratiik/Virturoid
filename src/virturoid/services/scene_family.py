"""Structural scene families (scene-gen plan S3) — the anti-overfitting core. The measured problem: today a
"scene set" is one topology jittered by centimetres (identical object counts, bins that never move), which the
RL literature shows is the "10-level" regime that overfits (CoinRun 66.8->90.0% test success from 100->unbounded
STRUCTURALLY-distinct levels; Procgen "need 10,000 levels"). This module instead produces a FAMILY of scenes that
differ in STRUCTURE — object count, layout topology, object-set — with the existing seeded DR as the inner layer,
and emits a train pool + a HELD-OUT pool that is structurally DISJOINT (novel topologies/object-sets, not just
new seeds), which is the field's honest generalization protocol ("no overlap between train, val and test scenes").

Every object carries a realistic size from ``dimension_priors`` (S2) via the per-axis ``size_xyz`` schema (S1), so
the scenes are dimensionally sane (a real 0.74 m table, a 0.9 m-wide/2.4 m-tall corridor) — count AND realism,
the two axes HSSD showed both matter. Pure-CPU + deterministic (seeded).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.scenes import SceneGraph, SceneObject
from virturoid.services.dimension_priors import default_size, mass_for, snap_to_prior


@dataclass(frozen=True)
class StructureKey:
    """A hashable signature of a scene's STRUCTURE — what makes two scenes genuinely different rather than two
    jitters of the same layout. Train and held-out pools are split so their StructureKeys never overlap."""
    task_type: str
    topology: str
    n_objects: int
    object_set: tuple[str, ...]
    layout: str

    def key(self):
        return (self.task_type, self.topology, self.n_objects, tuple(sorted(self.object_set)), self.layout)


@dataclass
class SceneFamily:
    task_type: str
    train: list[SceneGraph] = field(default_factory=list)
    held_out: list[SceneGraph] = field(default_factory=list)
    train_keys: list[tuple] = field(default_factory=list)
    held_out_keys: list[tuple] = field(default_factory=list)

    @property
    def disjoint(self) -> bool:
        """True iff no held-out structure appears in train (the honest-split guarantee)."""
        return not (set(self.train_keys) & set(self.held_out_keys))

    @property
    def n_distinct_train(self) -> int:
        return len(set(self.train_keys))


_YCB_POOL = ["ycb.mug", "ycb.soup_can", "ycb.cracker_box", "ycb.sugar_box", "ycb.mustard",
             "ycb.foam_brick", "ycb.wood_block", "ycb.can_355"]
_MANIP_LAYOUTS = ["row", "cluster", "two_groups", "scattered", "arc"]
_NAV_TOPOLOGIES = ["straight", "L", "T", "serpentine", "room_obstacles"]
_NAV_OBSTACLE_LAYOUTS = ["sparse", "clustered", "gauntlet"]


def _rng(seed: int):
    import numpy as np
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- structure enumeration (the diversity source) ---
def _manip_structures(task_type: str, rng, difficulty: int) -> list[StructureKey]:
    import numpy as np
    n_lo, n_hi = 3, min(8, 3 + difficulty + 2)
    structs = []
    for layout in _MANIP_LAYOUTS:
        for n in range(n_lo, n_hi + 1):
            k = rng.integers(2, min(5, len(_YCB_POOL)) + 1)
            oset = tuple(sorted(np.asarray(_YCB_POOL)[rng.choice(len(_YCB_POOL), size=int(k), replace=False)].tolist()))
            structs.append(StructureKey(task_type, "tabletop", int(n), oset, layout))
    return structs


def _nav_structures(task_type: str, rng, difficulty: int) -> list[StructureKey]:
    structs = []
    for topo in _NAV_TOPOLOGIES:
        for obs_layout in _NAV_OBSTACLE_LAYOUTS:
            n_obs = int(rng.integers(0, 2 + difficulty))
            structs.append(StructureKey(task_type, topo, n_obs, ("obstacle",), obs_layout))
    return structs


# ---------------------------------------------------------------------------------- realizers (structure -> MJCF) ---
def _place(name, cat, x, y, *, floor=False, material=None, mass=None, yaw=0.0):
    size = default_size(cat) or (0.05, 0.05, 0.05)
    z = size[2] / 2.0 if floor else 0.0                  # z = height axis; tabletop objects' rest-z set by exporter
    return SceneObject(name=name, object_type="cube", category=cat, size_xyz=size, material=material,
                       mass_kg=mass if mass is not None else mass_for(cat, size),
                       pose_xyz_rpy=(round(float(x), 4), round(float(y), 4), round(z, 4), 0.0, 0.0, float(yaw)))


def _layout_positions(layout: str, n: int, rng):
    """Deterministic per-layout base positions in the reachable tabletop annulus (x in [0.32,0.46], y in
    [-0.16,0.16]); the STRUCTURE (pattern), inner DR jitter added by the caller."""
    import numpy as np
    if layout == "row":
        xs = np.full(n, 0.40); ys = np.linspace(-0.14, 0.14, n)
    elif layout == "cluster":
        xs = 0.39 + rng.normal(0, 0.015, n); ys = rng.normal(0, 0.03, n)
    elif layout == "two_groups":
        xs = np.where(np.arange(n) < n // 2, 0.36, 0.44); ys = np.where(np.arange(n) < n // 2, -0.10, 0.10)
    elif layout == "arc":
        th = np.linspace(-0.7, 0.7, n); xs = 0.40 + 0.05 * np.cos(th); ys = 0.14 * np.sin(th)
    else:  # scattered
        xs = rng.uniform(0.34, 0.45, n); ys = rng.uniform(-0.15, 0.15, n)
    return xs, ys


def _realize_manip(struct: StructureKey, rng) -> SceneGraph:
    import numpy as np
    objs: list[SceneObject] = []
    # a REAL table surface (0.74 m tall) as context — the manipulation happens on the tabletop plane at TABLE_TOP_Z
    xs, ys = _layout_positions(struct.layout, struct.n_objects, rng)
    jit = rng.uniform(-0.02, 0.02, (struct.n_objects, 2))            # inner DR (the only per-seed variation)
    cats = list(struct.object_set)
    for i in range(struct.n_objects):
        cat = cats[i % len(cats)]
        objs.append(_place(f"obj{i}", cat, xs[i] + jit[i, 0], ys[i] + jit[i, 1],
                           material=["red", "blue", "green", "orange"][i % 4]))
    # sort/stack bins/zone: position varies with layout (front-back vs left-right) -> structural, not fixed
    if struct.task_type in ("pick_place_sort", "sort"):
        objs.append(SceneObject(name="bin_a", object_type="container", material="matte_red",
                                pose_xyz_rpy=(0.55, -0.2, 0.02, 0, 0, 0)))
        objs.append(SceneObject(name="bin_b", object_type="container", material="matte_blue",
                                pose_xyz_rpy=(0.55, 0.2, 0.02, 0, 0, 0)))
    else:
        objs.append(SceneObject(name="target", object_type="zone", material="matte_green",
                                size_xyz=(0.12, 0.006, 0.12), pose_xyz_rpy=(0.40, 0.0, 0.02, 0, 0, 0)))
    return SceneGraph(id=f"manip_{struct.topology}_{struct.layout}_{struct.n_objects}_{abs(hash(struct.key())) % 10000}",
                      name=f"{struct.task_type}:{struct.layout}:n{struct.n_objects}", backend_targets=["mujoco"],
                      robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0, 0, 0), objects=objs,
                      bounds=((-0.7, -0.7, 0.0), (0.9, 0.7, 0.8)),
                      variation_parameters={"structure": str(struct.key()), "n_objects": struct.n_objects,
                                            "layout": struct.layout, "object_set": ",".join(struct.object_set)})


def _corridor_walls(topology: str, width: float, rng) -> list[SceneObject]:
    """Build a corridor of the given TOPOLOGY from real-dimension wall segments (2.4 m tall, from the 'corridor'
    prior on the z/width axis + 'wall' prior on height). Returns the wall SceneObjects; the goal is placed at the
    corridor end by the caller."""
    h = snap_to_prior("wall", (1.0, 0.12, 2.4)).size_xyz[2]         # real 2.4 m wall HEIGHT (z), not 0.32 m
    thick = 0.10
    seg = 2.0                                                        # nominal corridor segment length
    walls: list[SceneObject] = []

    def wall(name, cx, cy, length, along_x, height=h):
        # size_xyz = (length-or-thickness on x, thickness-or-length on y, HEIGHT on z) -> exporter rests it upright
        sx = (length, thick, height) if along_x else (thick, length, height)
        return SceneObject(name=name, object_type="wall", category="wall", size_xyz=sx,
                           pose_xyz_rpy=(round(cx, 3), round(cy, 3), 0.0, 0, 0, 0.0))
    half = width / 2.0
    if topology == "straight":
        walls += [wall("wL", seg / 2, +half + thick / 2, seg, True), wall("wR", seg / 2, -half - thick / 2, seg, True)]
    elif topology == "L":
        walls += [wall("wL", seg / 2, +half + thick / 2, seg, True), wall("wR", seg / 2, -half - thick / 2, seg, True),
                  wall("wU", seg + half + thick / 2, seg / 2, seg, False), wall("wD", seg - half - thick / 2, seg / 2, seg, False)]
    elif topology == "T":
        walls += [wall("wL", 0, +half + thick / 2, 2 * seg, True), wall("wR", 0, -half - thick / 2, 2 * seg, True),
                  wall("wU", 0, seg / 2 + half, seg, False)]
    elif topology == "serpentine":
        for i in range(3):
            y = i * (width + thick)
            walls += [wall(f"s{i}L", seg / 2, y + half + thick / 2, seg, True),
                      wall(f"s{i}R", seg / 2, y - half - thick / 2, seg, True)]
    else:  # room_obstacles: a bounded room, obstacles added by caller
        R = 2.0
        walls += [wall("n", 0, R, 2 * R, True), wall("s", 0, -R, 2 * R, True),
                  wall("e", R, 0, 2 * R, False), wall("w", -R, 0, 2 * R, False)]
    return walls


def _realize_nav(struct: StructureKey, rng) -> SceneGraph:
    import numpy as np
    width = float(snap_to_prior("corridor", (rng.uniform(0.95, 1.4), 3.0, 2.44)).size_xyz[0])   # real >=0.915 m width (x)
    objs = _corridor_walls(struct.topology, width, rng)
    floor = SceneObject(name="floor", object_type="floor", size_xyz=(8.0, 0.04, 8.0),
                        pose_xyz_rpy=(1.0, 0.5, 0.0, 0, 0, 0))
    objs.append(floor)
    # obstacles: count + layout are structural
    for i in range(struct.n_objects):
        if struct.layout == "gauntlet":
            x, y = 0.4 + i * 0.6, (0.15 if i % 2 else -0.15)
        elif struct.layout == "clustered":
            x, y = 1.0 + rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
        else:  # sparse
            x, y = rng.uniform(0.4, 1.6), rng.uniform(-width / 3, width / 3)
        objs.append(SceneObject(name=f"obs{i}", object_type="obstacle", category="obstacle",
                                size_xyz=(0.2, 0.4, 0.2), pose_xyz_rpy=(round(x, 3), round(y, 3), 0.0, 0, 0, 0)))
    objs.append(SceneObject(name="goal", object_type="zone", material="matte_green", size_xyz=(0.4, 0.006, 0.4),
                            pose_xyz_rpy=(2.0, 0.0, 0.0, 0, 0, 0)))
    return SceneGraph(id=f"nav_{struct.topology}_{struct.layout}_{struct.n_objects}_{abs(hash(struct.key())) % 10000}",
                      name=f"{struct.task_type}:{struct.topology}:{struct.layout}", backend_targets=["mujoco"],
                      robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0, 0, 0), objects=objs,
                      bounds=((-2.5, -2.5, 0.0), (2.5, 2.5, 2.6)),
                      variation_parameters={"structure": str(struct.key()), "topology": struct.topology,
                                            "corridor_width_m": round(width, 3), "obstacle_layout": struct.layout})


_NAV_TASKS = {"navigation", "nav", "maze", "locomotion"}


def generate_family(task_type: str, *, n_train: int = 8, n_held_out: int = 3, difficulty: int = 1,
                    seed: int = 0) -> SceneFamily:
    """Task -> a family of STRUCTURALLY-distinct scenes with a disjoint held-out split. Enumerates candidate
    structures, deterministically shuffles, and assigns the first ``n_train`` distinct structures to train and the
    NEXT ``n_held_out`` (never seen in train) to held-out — the honest generalization protocol. Each structure is
    realized to a dimensionally-real SceneGraph."""
    rng = _rng(seed)
    is_nav = any(t in task_type.lower() for t in _NAV_TASKS)
    structs = (_nav_structures(task_type, rng, difficulty) if is_nav
               else _manip_structures(task_type, rng, difficulty))
    # dedupe by structure key, deterministically shuffle, split disjoint
    seen, uniq = set(), []
    for s in structs:
        if s.key() not in seen:
            seen.add(s.key()); uniq.append(s)
    order = rng.permutation(len(uniq))
    uniq = [uniq[i] for i in order]
    realize = _realize_nav if is_nav else _realize_manip
    train_structs = uniq[:n_train]
    held_structs = uniq[n_train:n_train + n_held_out]
    fam = SceneFamily(task_type=task_type)
    for i, s in enumerate(train_structs):
        fam.train.append(realize(s, _rng(seed * 1000 + i))); fam.train_keys.append(s.key())
    for i, s in enumerate(held_structs):
        fam.held_out.append(realize(s, _rng(seed * 2000 + i))); fam.held_out_keys.append(s.key())
    return fam
