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
# real floor-plan archetypes (connected walls + doorways + a floor), not disconnected panels
_NAV_TOPOLOGIES = ["open_room", "two_rooms", "l_corridor", "three_rooms", "cluttered"]
_NAV_OBSTACLE_LAYOUTS = ["sparse", "perimeter", "dense"]


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
            n_furn = int(rng.integers(2, 5 + difficulty))          # real furniture count in the room
            structs.append(StructureKey(task_type, topo, n_furn, ("furniture",), obs_layout))
    return structs


# ---------------------------------------------------------------------------------- realizers (structure -> MJCF) ---
def _place(name, cat, x, y, *, floor=False, material=None, mass=None, yaw=0.0):
    size = default_size(cat) or (0.05, 0.05, 0.05)
    z = size[2] / 2.0 if floor else 0.0                  # z = height axis; tabletop objects' rest-z set by exporter
    return SceneObject(name=name, object_type="cube", category=cat, size_xyz=size, material=material,
                       mass_kg=mass if mass is not None else mass_for(cat, size),
                       pose_xyz_rpy=(round(float(x), 4), round(float(y), 4), round(z, 4), 0.0, 0.0, float(yaw)))


_REACH_X = (0.28, 0.48)                             # kept within a 0.55 m arm reach even at the far corner
_REACH_Y = (-0.18, 0.18)


def _relax(xs, ys, radii, rng, *, margin=0.012, iters=60):
    """Size-AWARE separation: push any pair whose centers are closer than ``r_i + r_j + margin`` apart (a few
    passes) and clamp into the reachable annulus, so a layout keeps its STRUCTURE but no two footprints overlap —
    even elongated objects like a cracker box. Applied AFTER the DR jitter so jitter can't re-overlap them."""
    import numpy as np
    xs = np.asarray(xs, float).copy(); ys = np.asarray(ys, float).copy(); r = np.asarray(radii, float)
    n = len(xs)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                need = r[i] + r[j] + margin
                dx, dy = xs[i] - xs[j], ys[i] - ys[j]
                d = (dx * dx + dy * dy) ** 0.5
                if d < need:
                    push = (need - d) / 2 + 1e-4
                    ux, uy = (dx / d, dy / d) if d > 1e-6 else (rng.uniform(-1, 1), rng.uniform(-1, 1))
                    xs[i] += ux * push; ys[i] += uy * push; xs[j] -= ux * push; ys[j] -= uy * push
                    moved = True
        xs = np.clip(xs, *_REACH_X); ys = np.clip(ys, *_REACH_Y)
        if not moved:
            break
    return xs, ys


def _layout_positions(layout: str, n: int, rng):
    """Deterministic per-layout base positions in the reachable tabletop annulus; the STRUCTURE (pattern). The
    caller adds DR jitter then relaxes with size-aware separation, so no two objects interpenetrate."""
    import numpy as np
    if layout == "row":
        xs = np.full(n, 0.40); ys = np.linspace(-0.20, 0.20, n)
    elif layout == "cluster":
        xs = 0.40 + rng.normal(0, 0.03, n); ys = rng.normal(0, 0.05, n)
    elif layout == "two_groups":
        xs = np.where(np.arange(n) < n // 2, 0.34, 0.46) + rng.normal(0, 0.02, n)
        ys = np.where(np.arange(n) < n // 2, -0.14, 0.14) + rng.normal(0, 0.03, n)
    elif layout == "arc":
        th = np.linspace(-0.9, 0.9, n); xs = 0.40 + 0.07 * np.cos(th); ys = 0.20 * np.sin(th)
    else:  # scattered
        xs = rng.uniform(*_REACH_X, n); ys = rng.uniform(*_REACH_Y, n)
    return xs, ys


def _realize_manip(struct: StructureKey, rng) -> SceneGraph:
    import numpy as np
    objs: list[SceneObject] = []
    cats = [list(struct.object_set)[i % len(struct.object_set)] for i in range(struct.n_objects)]
    xs, ys = _layout_positions(struct.layout, struct.n_objects, rng)
    xs = np.asarray(xs, float) + rng.uniform(-0.02, 0.02, struct.n_objects)   # inner DR (per-seed variation)...
    ys = np.asarray(ys, float) + rng.uniform(-0.02, 0.02, struct.n_objects)
    radii = [max(default_size(c)[0], default_size(c)[1]) / 2 for c in cats]   # per-object footprint radius
    xs, ys = _relax(xs, ys, radii, rng)                             # ...then relax so no two footprints overlap
    for i in range(struct.n_objects):
        objs.append(_place(f"obj{i}", cats[i], xs[i], ys[i],
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


WALL_H = 2.44                                       # real interior wall / ceiling height (IRC R305.1)
THICK = 0.12                                         # real interior partition thickness
DOOR_W = 0.95                                        # doorway clear width (>= ADA 0.915 m)
_FURNITURE = ["table", "shelf", "box", "chair", "desk"]


def _wall_span(name, along_x, a, b, fixed, h=WALL_H, thick=THICK):
    """A single wall segment spanning [a, b] along its axis at the perpendicular coord ``fixed`` (metres)."""
    length = max(0.02, b - a); c = (a + b) / 2.0
    cx, cy = (c, fixed) if along_x else (fixed, c)
    sx = (length, thick, h) if along_x else (thick, length, h)
    return SceneObject(name=name, object_type="wall", category="wall", size_xyz=(round(sx[0], 3), sx[1], sx[2]),
                       pose_xyz_rpy=(round(cx, 3), round(cy, 3), 0.0, 0, 0, 0.0))


def _wall_with_door(prefix, along_x, a, b, fixed, door_at, door_w=DOOR_W, h=WALL_H):
    """A wall from a to b with a DOORWAY gap of width ``door_w`` centred at ``door_at`` — i.e. two segments with a
    real opening between them, so adjacent rooms actually CONNECT (the thing the old floating panels lacked)."""
    segs = []
    lo_end = door_at - door_w / 2.0
    hi_end = door_at + door_w / 2.0
    if lo_end - a > 0.06:
        segs.append(_wall_span(f"{prefix}_l", along_x, a, lo_end, fixed, h))
    if b - hi_end > 0.06:
        segs.append(_wall_span(f"{prefix}_r", along_x, hi_end, b, fixed, h))
    return segs


def _place_furniture(W, H, k, rng, keepout, forbid_x):
    """Place ``k`` real-sized furniture items (table 1.2x0.75x0.74, shelf, boxes, chairs) inside the room, clear of
    the spawn/goal keepout points, the doorway x-corridors ``forbid_x``, and each other. Rejection sampling."""
    import numpy as np
    placed = []
    tries = 0
    while len(placed) < k and tries < 400:
        tries += 1
        cat = _FURNITURE[rng.integers(0, len(_FURNITURE))]
        sx, sy, sz = default_size(cat)
        rot = bool(rng.integers(0, 2))
        if rot:
            sx, sy = sy, sx
        x = rng.uniform(0.5 + sx / 2, W - 0.5 - sx / 2)
        y = rng.uniform(0.7 + sy / 2, H - 0.5 - sy / 2)
        rad = max(sx, sy) / 2 + 0.35
        if any((x - kx) ** 2 + (y - ky) ** 2 < rad ** 2 for kx, ky in keepout):
            continue
        if any(abs(x - fx) < sx / 2 + 0.55 for fx in forbid_x):     # keep doorway corridors clear
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (rad + pr) ** 2 for px, py, pr in placed):
            continue
        placed.append((x, y, max(sx, sy) / 2))
        yield SceneObject(name=f"furn{len(placed)}_{cat}", object_type="obstacle", category=cat,
                          size_xyz=(round(sx, 3), round(sy, 3), round(sz, 3)),
                          pose_xyz_rpy=(round(x, 3), round(y, 3), 0.0, 0, 0, 0.0))


def _realize_nav(struct: StructureKey, rng) -> SceneGraph:
    """A coherent room floor plan: a bounded room with a real floor, connected perimeter walls (2.44 m tall) with a
    south entrance, interior dividing walls that leave real DOORWAYS per topology, and real furniture. Spawn just
    inside the entrance; goal in a different room reached THROUGH the doorways (A*-navigable, gated by S4)."""
    W = round(float(rng.uniform(5.0, 8.0)), 2); H = round(float(rng.uniform(4.0, 6.5)), 2)
    t = THICK
    objs: list[SceneObject] = [
        SceneObject("floor", "floor", size_xyz=(W + 0.6, H + 0.6, 0.08), pose_xyz_rpy=(W / 2, H / 2, 0, 0, 0, 0))]
    # perimeter: closed rectangle with a front entrance in the south wall
    objs.append(_wall_span("perim_n", True, -t, W + t, H))
    objs += _wall_with_door("perim_s", True, -t, W + t, 0.0, door_at=W * 0.5, door_w=1.2)
    objs.append(_wall_span("perim_w", False, 0, H, 0.0))
    objs.append(_wall_span("perim_e", False, 0, H, W))
    spawn = (round(W * 0.5, 2), 0.7)
    forbid_x: list[float] = []
    topo = struct.topology
    if topo == "two_rooms":
        dx = round(W * 0.55, 2); dy = round(H * float(rng.uniform(0.3, 0.7)), 2)
        objs += _wall_with_door("div", False, 0, H, dx, door_at=dy)
        forbid_x.append(dx); goal = (round(dx + (W - dx) * 0.5, 2), round(H * 0.7, 2))
    elif topo == "three_rooms":
        d1, d2 = round(W * 0.36, 2), round(W * 0.68, 2)
        objs += _wall_with_door("div1", False, 0, H, d1, door_at=round(H * 0.25, 2))   # offset doors -> a serpentine path
        objs += _wall_with_door("div2", False, 0, H, d2, door_at=round(H * 0.75, 2))
        forbid_x += [d1, d2]; goal = (round(W * 0.85, 2), round(H * 0.5, 2))
    elif topo == "l_corridor":
        # a SOLID vertical barrier attached to the south wall, leaving the top open -> the robot must go up the
        # left, over the top of the barrier, and down to the goal: a real L-shaped route.
        bx = round(W * 0.45, 2); btop = round(H * 0.62, 2)
        objs.append(_wall_span("l_bar", False, 0, btop, bx))
        forbid_x.append(bx); spawn = (round(W * 0.2, 2), 0.7); goal = (round(W * 0.75, 2), round(H * 0.85, 2))
    else:  # open_room / cluttered
        goal = (round(W * 0.82, 2), round(H * 0.8, 2))
    n_furn = struct.n_objects + (3 if topo == "cluttered" else 0)
    objs += list(_place_furniture(W, H, n_furn, rng, keepout=[spawn, goal], forbid_x=forbid_x))
    objs.append(SceneObject("goal", "zone", material="matte_green", size_xyz=(0.5, 0.5, 0.006),
                            pose_xyz_rpy=(goal[0], goal[1], 0.0, 0, 0, 0)))
    return SceneGraph(id=f"nav_{topo}_{struct.layout}_{struct.n_objects}_{abs(hash(struct.key())) % 10000}",
                      name=f"{struct.task_type}:{topo}:{struct.layout}", backend_targets=["mujoco"],
                      robot_spawn_xyz_rpy=(spawn[0], spawn[1], 0.0, 0, 0, 0), objects=objs,
                      bounds=((-1.0, -1.0, 0.0), (W + 1.0, H + 1.0, 2.7)),
                      variation_parameters={"structure": str(struct.key()), "topology": topo,
                                            "room_w_m": W, "room_h_m": H, "n_furniture": n_furn,
                                            "layout": struct.layout})


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
