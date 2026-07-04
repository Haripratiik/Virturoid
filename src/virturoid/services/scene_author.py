"""Scene Author (breakthrough v5, B3): the scene half of our 'simulation AI'. The existing ``scene_generator``
places objects per TASK TEMPLATE (sort/stack/push/maze); ``scene_agent`` proposes DR envelopes. This module adds
the missing GENERAL layer the plan calls for: an LLM (or a heuristic) proposes a scene as ASSET IDs + RELATIONAL
CONSTRAINTS (on / near / clear / within-reach / in-bounds), and a TRUSTED deterministic solver realizes poses +
runs fail-closed validation GATES. This is the Holodeck discipline — the model says WHAT relationships must hold,
trusted code decides WHERE, and a gate proves the scene is physically sane before it ever reaches the trainer.

The model never places coordinates directly (hallucinated poses interpenetrate or float) and never grades its own
scene. Pure-CPU, seeded + deterministic, LLM-optional -> unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Asset:
    """A scene object. ``radius`` is a bounding radius (m) for clearance/interpenetration; ``height`` its own
    vertical extent (for stacking); ``kind`` gates behaviour (a 'support' can carry ``On`` objects)."""
    id: str
    radius: float = 0.04
    height: float = 0.05
    kind: str = "object"               # object | support | obstacle
    mass: float = 0.2


# Relational constraints (the vocabulary the proposer emits). Kept as plain tuples so an LLM can emit JSON.
def On(obj: str, support: str):            return ("on", obj, support)
def Near(a: str, b: str, dist: float):     return ("near", a, b, float(dist))
def Clear(obj: str, radius: float):        return ("clear", obj, float(radius))
def WithinReach(obj: str, r: float):       return ("reach", obj, float(r))
def InBounds(obj: str):                    return ("bounds", obj)


@dataclass
class Placement:
    pos: np.ndarray                    # (3,) world position of the object's center
    asset: Asset


@dataclass
class Scene:
    placements: dict                   # id -> Placement
    assets: dict                       # id -> Asset
    constraints: list
    base_pos: np.ndarray
    bounds: np.ndarray                 # (2,3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    solved: bool = False
    report: dict = field(default_factory=dict)


def realize_scene(assets: list[Asset], constraints: list, *, base_pos=(0.0, 0.0, 0.0),
                  bounds=((-0.6, -0.6, 0.0), (0.6, 0.6, 0.5)), seed: int = 0, tries: int = 400) -> Scene:
    """TRUSTED deterministic placement. Greedy seeded rejection-sampling: place each asset in turn so that its
    constraints hold against already-placed assets; supports are placed first. Returns a Scene whose ``solved``
    flag says every object found a legal pose within ``tries`` attempts."""
    rng = np.random.default_rng(seed)
    amap = {a.id: a for a in assets}
    base = np.asarray(base_pos, dtype=float)
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    by_obj: dict[str, list] = {a.id: [] for a in assets}
    for c in constraints:
        by_obj.setdefault(c[1], []).append(c)

    # place supports (and obstacles) first, then objects (which may rest 'on' supports)
    order = sorted(assets, key=lambda a: 0 if a.kind in ("support", "obstacle") else 1)
    placed: dict[str, Placement] = {}

    def _ok(aid: str, pos: np.ndarray) -> bool:
        a = amap[aid]
        if np.any(pos[:2] < lo[:2]) or np.any(pos[:2] > hi[:2]):
            return False
        for c in by_obj.get(aid, []):
            if c[0] == "near" and c[2] in placed:
                if np.linalg.norm(pos[:2] - placed[c[2]].pos[:2]) > c[3]:
                    return False
            elif c[0] == "reach":
                if np.linalg.norm(pos[:2] - base[:2]) > c[2]:
                    return False
            elif c[0] == "clear":
                for oid, pl in placed.items():
                    if oid != aid and np.linalg.norm(pos[:2] - pl.pos[:2]) < c[2]:
                        return False
        # never interpenetrate any already-placed object (unless intentionally stacked ON it)
        on_support = next((c[2] for c in by_obj.get(aid, []) if c[0] == "on"), None)
        for oid, pl in placed.items():
            if oid == on_support:
                continue
            if np.linalg.norm(pos[:2] - pl.pos[:2]) < (a.radius + pl.asset.radius) * 0.98:
                return False
        return True

    for a in order:
        on_support = next((c[2] for c in by_obj.get(a.id, []) if c[0] == "on"), None)
        got = None
        for _ in range(tries):
            if on_support and on_support in placed:
                sp = placed[on_support]
                jitter = (rng.random(2) - 0.5) * 2 * max(1e-3, sp.asset.radius - a.radius)
                pos = np.array([sp.pos[0] + jitter[0], sp.pos[1] + jitter[1],
                                sp.pos[2] + sp.asset.height / 2 + a.height / 2])
            else:
                p = lo + rng.random(3) * (hi - lo)
                pos = np.array([p[0], p[1], a.height / 2])   # rest on the ground plane
            if _ok(a.id, pos):
                got = pos; break
        if got is None:
            return Scene(placements=placed, assets=amap, constraints=constraints, base_pos=base,
                         bounds=np.array([lo, hi]), solved=False,
                         report={"failed_asset": a.id, "reason": "no legal pose within tries"})
        placed[a.id] = Placement(pos=got, asset=a)

    sc = Scene(placements=placed, assets=amap, constraints=constraints, base_pos=base,
               bounds=np.array([lo, hi]), solved=True)
    sc.report = validate_scene(sc)
    sc.solved = sc.report["ok"]
    return sc


def validate_scene(scene: Scene) -> dict:
    """Fail-closed GATES on a realized scene: (1) no interpenetration between free objects, (2) every ``on``
    object actually rests on its support's top surface, (3) every ``reach`` object is within reach of the base,
    (4) everything is in bounds. Returns ``{"ok": bool, "violations": [...]}``."""
    viol: list[str] = []
    pl = scene.placements
    lo, hi = scene.bounds[0], scene.bounds[1]
    on_pairs = {(c[1], c[2]) for c in scene.constraints if c[0] == "on"}

    ids = list(pl.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (a, b) in on_pairs or (b, a) in on_pairs:
                continue
            d = float(np.linalg.norm(pl[a].pos[:2] - pl[b].pos[:2]))
            if d < (pl[a].asset.radius + pl[b].asset.radius) * 0.98:
                viol.append(f"interpenetration: {a} & {b} (d={d:.3f})")
    for c in scene.constraints:
        if c[0] == "on" and c[1] in pl and c[2] in pl:
            obj, sup = pl[c[1]], pl[c[2]]
            expect_z = sup.pos[2] + sup.asset.height / 2 + obj.asset.height / 2
            if abs(obj.pos[2] - expect_z) > 1e-3:
                viol.append(f"'{c[1]}' not resting on '{c[2]}'")
        elif c[0] == "reach" and c[1] in pl:
            if float(np.linalg.norm(pl[c[1]].pos[:2] - scene.base_pos[:2])) > c[2] + 1e-6:
                viol.append(f"'{c[1]}' out of reach ({c[2]}m)")
    for aid, p in pl.items():
        if np.any(p.pos[:2] < lo[:2] - 1e-6) or np.any(p.pos[:2] > hi[:2] + 1e-6):
            viol.append(f"'{aid}' out of bounds")
    return {"ok": not viol, "violations": viol}


def scene_to_mjcf(scene: Scene) -> str:
    """Emit the realized scene's objects as MuJoCo geoms (free bodies for objects, static for supports/obstacles),
    ready to splice into a worldbody. Supports/obstacles are boxes; objects are the classic manipulable blocks."""
    out = []
    for aid, p in scene.placements.items():
        a = p.asset
        x, y, z = p.pos
        if a.kind in ("support", "obstacle"):
            out.append(f'<geom name="{aid}" type="box" pos="{x:.3f} {y:.3f} {z:.3f}" '
                       f'size="{a.radius:.3f} {a.radius:.3f} {a.height/2:.3f}" contype="1" conaffinity="1" '
                       f'material="mat_metal"/>')
        else:
            out.append(f'<body name="{aid}" pos="{x:.3f} {y:.3f} {z:.3f}"><freejoint/>'
                       f'<geom type="box" size="{a.radius:.3f} {a.radius:.3f} {a.height/2:.3f}" '
                       f'mass="{a.mass:.3f}" contype="1" conaffinity="1" material="mat_shell"/></body>')
    return "\n".join(out)


def propose_scene(task: str, catalog: list[Asset], *, llm=None) -> tuple[list[Asset], list]:
    """Propose (assets, constraints) for a task. With ``llm`` the model selects asset IDs from ``catalog`` and
    emits relational constraints as JSON; validated against the catalog + the known constraint vocabulary, with a
    heuristic fallback (a table support + 3 reachable, cleared blocks) so the loop never stalls."""
    if llm is not None:
        try:
            import json
            spec = json.loads(llm.complete(_scene_prompt(task, catalog)))
            ids = {a.id for a in catalog}
            assets = [a for a in catalog if a.id in set(spec.get("assets", []))]
            cons = [tuple(c) for c in spec.get("constraints", [])
                    if c and c[0] in ("on", "near", "clear", "reach", "bounds") and c[1] in ids]
            if assets:
                return assets, cons
        except Exception:  # noqa: BLE001 - malformed LLM output -> heuristic fallback
            pass
    # heuristic: a table + up to 3 blocks, each reachable + mutually cleared
    table = next((a for a in catalog if a.kind == "support"), Asset("table", radius=0.25, height=0.02, kind="support"))
    blocks = [a for a in catalog if a.kind == "object"][:3] or [Asset(f"block{i}", radius=0.03, height=0.05) for i in range(3)]
    assets = [table] + blocks
    cons = [InBounds(table.id)]
    for b in blocks:
        cons += [On(b.id, table.id), WithinReach(b.id, 0.5), Clear(b.id, 0.09)]
    return assets, cons


def _scene_prompt(task: str, catalog: list[Asset]) -> str:
    ids = ", ".join(a.id for a in catalog)
    return (f'Design a scene for the task: {task}. Choose asset IDs from: [{ids}]. '
            f'Return JSON {{"assets": [ids...], "constraints": [["on",obj,support], ["near",a,b,dist], '
            f'["clear",obj,r], ["reach",obj,r], ["bounds",obj]]}}. Only these constraint types. '
            f'Do NOT give coordinates — the placement engine solves those.')
