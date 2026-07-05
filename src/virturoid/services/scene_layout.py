"""General procedural scene-layout engine (scene-gen). ONE solver realizes ANY room graph a proposer emits — there
is NO per-environment hardcoded geometry. The pipeline is the Holodeck/ProcTHOR pattern:

    prompt --propose_room_graph--> [room TYPES]  (LLM for any prompt; a keyword heuristic offline)
          --realize_scene_spec--> BSP floor plan (rooms split by walls, each split leaves a DOORWAY so the whole
                                   plan is connected) + each room furnished from a room-type -> furniture DATA table
          --> a dimensionally-real, navigable SceneGraph (verified by the S4 gates)

Adding an environment (clinic, library, restaurant, lab, ...) = adding a room type + its furniture list to the
DATA below, or letting the LLM emit new rooms — NOT writing new geometry code. Dimensions come from
``dimension_priors`` (code-owned), never hallucinated. Pure-CPU, deterministic, LLM-optional.
"""

from __future__ import annotations

from virturoid.schemas.scenes import SceneGraph, SceneObject
from virturoid.services.dimension_priors import default_size

WALL_H, THICK, DOOR_W = 2.44, 0.12, 0.95

# --- DATA: room type -> (furniture categories, fill strategy, target area m^2). A proposer/LLM picks room types;
# code owns the sizes (dimension_priors) and the placement. This table IS the "knowledge" — extend it, not code.
ROOM_TYPES: dict[str, tuple[list[str], str, float]] = {
    "living_room": (["sofa", "coffee_table", "tv_stand", "armchair"], "perimeter", 18.0),
    "kitchen":     (["counter", "fridge", "dining_table"], "perimeter", 12.0),
    "bedroom":     (["bed", "wardrobe", "nightstand"], "perimeter", 12.0),
    "bathroom":    (["toilet", "sink"], "perimeter", 5.0),
    "office":      (["desk", "desk", "shelf", "armchair"], "perimeter", 12.0),
    "lobby":       (["armchair", "armchair", "coffee_table"], "perimeter", 14.0),
    "storage":     (["shelf", "shelf", "crate", "crate"], "perimeter", 10.0),
    "warehouse":   (["rack"], "aisles", 60.0),
    "empty":       ([], "empty", 10.0),
}

# --- DATA: environment keyword -> a room list (the offline heuristic proposer). The LLM can emit ANY list; this is
# only the fallback so the generator works key-free.
ENV_TEMPLATES: dict[str, list[str]] = {
    "warehouse": ["warehouse"], "store": ["warehouse", "lobby"], "shop": ["warehouse", "lobby"],
    "fulfil": ["warehouse"], "logistics": ["warehouse"],
    "house": ["living_room", "kitchen", "bedroom", "bathroom"],
    "home": ["living_room", "kitchen", "bedroom", "bathroom"], "apartment": ["living_room", "kitchen", "bedroom"],
    "roomba": ["living_room", "kitchen", "bedroom", "bathroom"], "vacuum": ["living_room", "kitchen", "bedroom"],
    "office": ["office", "office", "lobby", "bathroom"], "clinic": ["lobby", "office", "bathroom", "storage"],
    "lab": ["office", "storage", "office"],
}

_MAT = {"sofa": "matte_blue", "bed": "matte_blue", "armchair": "matte_gray", "coffee_table": "wood",
        "dining_table": "wood", "desk": "wood", "tv_stand": "matte_gray", "counter": "matte_gray",
        "fridge": "matte_gray", "wardrobe": "wood", "nightstand": "wood", "shelf": "matte_gray",
        "rack": "matte_gray", "crate": "cardboard", "carton": "cardboard", "toilet": "matte_gray",
        "sink": "matte_gray", "pallet": "wood", "box": "cardboard"}


# --------------------------------------------------------------------------------------------- wall primitives ---
def _wall_span(name, along_x, a, b, fixed, h=WALL_H, thick=THICK):
    length = max(0.02, b - a); c = (a + b) / 2.0
    cx, cy = (c, fixed) if along_x else (fixed, c)
    sx = (length, thick, h) if along_x else (thick, length, h)
    return SceneObject(name=name, object_type="wall", category="wall", size_xyz=(round(sx[0], 3), sx[1], sx[2]),
                       pose_xyz_rpy=(round(cx, 3), round(cy, 3), 0.0, 0, 0, 0.0))


def _wall_with_door(name, along_x, a, b, fixed, door_at, door_w=DOOR_W, h=WALL_H):
    lo_end, hi_end, segs = door_at - door_w / 2, door_at + door_w / 2, []
    if lo_end - a > 0.06:
        segs.append(_wall_span(f"{name}_l", along_x, a, lo_end, fixed, h))
    if b - hi_end > 0.06:
        segs.append(_wall_span(f"{name}_r", along_x, hi_end, b, fixed, h))
    return segs


def _furn(name, cat, x, y, rot=False):
    sx, sy, sz = default_size(cat) or (0.4, 0.4, 0.4)
    if rot:
        sx, sy = sy, sx
    return SceneObject(name=name, object_type="obstacle", category=cat, material=_MAT.get(cat, "matte_gray"),
                       size_xyz=(round(sx, 3), round(sy, 3), round(sz, 3)),
                       pose_xyz_rpy=(round(float(x), 3), round(float(y), 3), 0.0, 0, 0, 0.0))


# --------------------------------------------------------------------------- proposer: prompt -> room graph ---
def propose_room_graph(prompt: str, *, llm=None, rng=None) -> list[str]:
    """Prompt -> a list of room TYPES. With ``llm`` it proposes rooms for ANY prompt (each validated against
    ROOM_TYPES, unknowns dropped, backfilled to >=1); offline it matches an environment keyword, else returns a
    generic multi-room space so the engine always has something to realize."""
    import numpy as np
    rng = rng if rng is not None else np.random.default_rng(0)
    if llm is not None:
        try:
            import json
            spec = json.loads(llm.complete(_prompt(prompt)))
            rooms = [r for r in spec.get("rooms", []) if r in ROOM_TYPES]
            if rooms:
                return rooms
        except Exception:  # noqa: BLE001 - malformed LLM output -> heuristic
            pass
    tl = prompt.lower()
    for kw, rooms in ENV_TEMPLATES.items():
        if kw in tl:
            return list(rooms)
    n = int(rng.integers(2, 4))                                 # generic: a few connected rooms
    return ["empty"] * n


def _prompt(prompt: str) -> str:
    types = ", ".join(ROOM_TYPES)
    return (f'Design the ROOMS for a scene for: "{prompt}". Return JSON {{"rooms": [types...]}} choosing only from '
            f'[{types}]. List each room (repeat types for multiples, e.g. two offices). No coordinates, no sizes — '
            f'the layout engine solves those.')


# --------------------------------------------------------------------------------- BSP floor-plan solver ---
def _split(rooms, x0, y0, x1, y1, rng, out, walls, ctr, doors):
    """Recursively BSP-split the footprint into len(rooms) rectangles. Each split emits an interior wall WITH a
    doorway (whose centre is recorded in ``doors`` so furniture stays clear of it), so every room connects to its
    neighbour and the whole plan is navigable (a tree of doorways)."""
    if len(rooms) <= 1:
        out.append((rooms[0] if rooms else "empty", (x0, y0, x1, y1))); return
    n_left = len(rooms) // 2
    frac = min(0.68, max(0.32, n_left / len(rooms) + float(rng.uniform(-0.06, 0.06))))
    ctr[0] += 1
    if (x1 - x0) >= (y1 - y0):                                  # vertical split
        xs = x0 + (x1 - x0) * frac
        if y1 - y0 > DOOR_W + 0.4:
            dy = float(rng.uniform(y0 + DOOR_W, y1 - DOOR_W)); doors.append((xs, dy))
            walls.extend(_wall_with_door(f"iw{ctr[0]}", False, y0, y1, xs, door_at=dy))
        _split(rooms[:n_left], x0, y0, xs, y1, rng, out, walls, ctr, doors)
        _split(rooms[n_left:], xs, y0, x1, y1, rng, out, walls, ctr, doors)
    else:                                                      # horizontal split
        ys = y0 + (y1 - y0) * frac
        if x1 - x0 > DOOR_W + 0.4:
            dx = float(rng.uniform(x0 + DOOR_W, x1 - DOOR_W)); doors.append((dx, ys))
            walls.extend(_wall_with_door(f"iw{ctr[0]}", True, x0, x1, ys, door_at=dx))
        _split(rooms[:n_left], x0, y0, x1, ys, rng, out, walls, ctr, doors)
        _split(rooms[n_left:], x0, ys, x1, y1, rng, out, walls, ctr, doors)


def _furnish_perimeter(rect, cats, rng, tag, doors):
    """Place a room's furniture flush against its walls, skipping items too big for the room, rejecting overlaps,
    and keeping clear of DOORWAYS (so furniture never blocks a passage). Generic for any furniture list."""
    x0, y0, x1, y1 = rect; placed, objs = [], []
    sides = ["n", "s", "e", "w"]
    for i, cat in enumerate(cats):
        sz = default_size(cat) or (0.4, 0.4, 0.4)
        if sz[0] > (x1 - x0) - 0.3 or sz[1] > (y1 - y0) - 0.3:
            continue                                            # too big for this room
        for _ in range(14):
            side = sides[int(rng.integers(0, 4))]; fr = float(rng.uniform(0.05, 0.95))
            x, y, rot = _against(rect, cat, side, fr)
            r = max(sz[0], sz[1]) / 2
            if any(abs(x - dx) < r + 0.7 and abs(y - dy) < r + 0.7 for dx, dy in doors):
                continue                                        # would block a doorway
            if any((x - px) ** 2 + (y - py) ** 2 < (r + pr + 0.15) ** 2 for px, py, pr in placed):
                continue
            placed.append((x, y, r)); objs.append(_furn(f"{tag}_{cat}{i}", cat, x, y, rot))
            break
    return objs


def _furnish_aisles(rect, rng, tag):
    """Fill a room with parallel storage RACKS separated by robot-width aisles + a few floor cartons — the generic
    'warehouse' fill strategy (data-selected by room type, not a hardcoded env)."""
    x0, y0, x1, y1 = rect; objs = []
    aisle = float(rng.uniform(1.4, 1.8)); rack_d = 1.0
    n = max(1, int((x1 - x0 - aisle) / (rack_d + aisle)))
    rlen = max(1.0, (y1 - y0) - 1.2)
    for i in range(n):
        cx = x0 + aisle + i * (rack_d + aisle) + rack_d / 2
        objs.append(SceneObject(f"{tag}_rack{i}", "obstacle", category="rack", material="matte_gray",
                                size_xyz=(round(rack_d, 3), round(rlen, 3), 2.5),
                                pose_xyz_rpy=(round(cx, 3), round((y0 + y1) / 2, 3), 0.0, 0, 0, 0)))
        for j in range(int(rng.integers(1, 3))):
            objs.append(_furn(f"{tag}_carton{i}_{j}", "carton", cx - rack_d / 2 - aisle * 0.4,
                              float(rng.uniform(y0 + 0.6, y1 - 0.6))))
    return objs


def _against(rect, cat, side, frac):
    x0, y0, x1, y1 = rect
    sx, sy, _ = default_size(cat) or (0.4, 0.4, 0.4)
    m = 0.06
    if side in ("n", "s"):
        rot = False
        x = x0 + m + frac * (x1 - x0 - 2 * m - sx) + sx / 2
        y = (y1 - m - sy / 2) if side == "n" else (y0 + m + sy / 2)
    else:
        rot = True
        y = y0 + m + frac * (y1 - y0 - 2 * m - sy) + sy / 2
        x = (x1 - m - sx / 2) if side == "e" else (x0 + m + sx / 2)
    return x, y, rot


def _open_goal(objs, W, H, spawn, robot_r, rng, n=400):
    blocks = [(o.pose_xyz_rpy[0], o.pose_xyz_rpy[1], o.size_xyz[0] / 2, o.size_xyz[1] / 2)
              for o in objs if o.object_type in ("wall", "obstacle") and o.size_xyz]
    best, bestd = None, -1.0
    for _ in range(n):
        x = float(rng.uniform(0.9, W - 0.9)); y = float(rng.uniform(0.9, H - 0.9))
        if any(abs(x - bx) < hx + robot_r + 0.12 and abs(y - by) < hy + robot_r + 0.12 for bx, by, hx, hy in blocks):
            continue
        d = (x - spawn[0]) ** 2 + (y - spawn[1]) ** 2
        if d > bestd:
            bestd, best = d, (round(x, 2), round(y, 2))
    return best or (round(W * 0.5, 2), round(H * 0.5, 2))


# ------------------------------------------------------------------------ the ONE general realizer ---
def realize_scene_spec(rooms: list[str], *, env: str = "scene", seed: int = 0, robot_r: float = 0.18) -> SceneGraph:
    """Realize ANY room list into a coherent, navigable floor plan: size a footprint from the rooms' target areas,
    add a floor + a perimeter with an entrance, BSP-split into the rooms (walls with doorways), furnish each room by
    its type's strategy, and place spawn (entrance) + goal (an open point far away). No per-environment code."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rooms = rooms or ["empty"]
    area = sum(ROOM_TYPES.get(r, ([], "empty", 10.0))[2] for r in rooms) * 1.25
    aspect = float(rng.uniform(0.8, 1.3))
    W = round(float(np.sqrt(area * aspect)), 2); H = round(area / W, 2)
    t = THICK
    objs: list[SceneObject] = [SceneObject("floor", "floor", size_xyz=(W + 0.6, H + 0.6, 0.08),
                                           pose_xyz_rpy=(W / 2, H / 2, 0, 0, 0, 0))]
    objs.append(_wall_span("perim_n", True, -t, W + t, H))
    objs += _wall_with_door("perim_s", True, -t, W + t, 0.0, door_at=W * 0.35, door_w=1.2)   # entrance
    objs.append(_wall_span("perim_w", False, 0, H, 0.0)); objs.append(_wall_span("perim_e", False, 0, H, W))
    rects, iwalls, doors = [], [], [(W * 0.35, 0.0)]          # doorway centres (entrance + interior), kept clear
    order = list(rng.permutation(len(rooms)))
    _split([rooms[i] for i in order], 0.0, 0.0, W, H, rng, rects, iwalls, [0], doors)
    objs += iwalls
    for ri, (rtype, rect) in enumerate(rects):
        cats, fill, _a = ROOM_TYPES.get(rtype, ([], "empty", 10.0))
        if fill == "aisles":
            objs += _furnish_aisles(rect, rng, f"r{ri}")
        elif fill == "perimeter":
            objs += _furnish_perimeter(rect, cats, rng, f"r{ri}", doors)
    spawn = (round(W * 0.35, 2), 0.7)
    goal = _open_goal(objs, W, H, spawn, robot_r, rng)
    objs.append(SceneObject("goal", "zone", material="matte_green", size_xyz=(0.5, 0.5, 0.006),
                            pose_xyz_rpy=(goal[0], goal[1], 0.0, 0, 0, 0)))
    return SceneGraph(id=f"{env}_{'_'.join(r[:3] for r in rooms)}_{seed}",
                      name=f"{env}:{'+'.join(rooms)}", backend_targets=["mujoco"],
                      robot_spawn_xyz_rpy=(spawn[0], spawn[1], 0.0, 0, 0, 0), objects=objs,
                      bounds=((-1.0, -1.0, 0.0), (W + 1.0, H + 1.0, 2.9)),
                      variation_parameters={"env": env, "rooms": ",".join(rooms), "w_m": W, "h_m": H,
                                            "n_rooms": len(rooms)})


def generate_scene(prompt: str, *, seed: int = 0, robot_r: float = 0.18, llm=None, max_tries: int = 12):
    """The general entry point: prompt -> a validated, navigable scene, ON ITS OWN. Propose a room graph, realize
    it, and keep it only if the S4 physical-validity gates pass — retrying with fresh seeds (rejection sampling,
    the field's keep-if-valid discipline) so the OUTPUT is always sane even when a raw layout occasionally isn't.
    Returns ``(SceneGraph, report)`` or the last attempt if none passed."""
    import numpy as np

    from virturoid.services.scene_validity import validate_scene_physical
    rng = np.random.default_rng(seed)
    rooms = propose_room_graph(prompt, llm=llm, rng=rng)
    env = next((w for w in prompt.lower().split() if w in ENV_TEMPLATES), rooms[0] if rooms else "scene")
    last = None
    for k in range(max_tries):
        s = realize_scene_spec(rooms, env=env, seed=seed * 100 + k, robot_r=robot_r)
        rep = validate_scene_physical(s, robot_radius=robot_r, run_settle=False)
        last = (s, rep, rooms)
        if rep["ok"]:
            return s, {"rooms": rooms, "tries": k + 1, "valid": True}
    return last[0], {"rooms": rooms, "tries": max_tries, "valid": False, "violations": last[1]["violations"][:2]}
