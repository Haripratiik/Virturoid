"""AI-native EDIT OPERATORS (docs/ai_native_plan.md P1) — typed, LOCALIZED, gated mutations over a robot gene.

The proven pattern (CAD-Assistant arXiv:2412.13810 / LLM4CAD-Editor / Zoo edit-endpoint): a small semantic
request ("make it taller", "give it longer legs", "make it carbon-fiber") becomes a TYPED operator over the
existing design — never a regeneration. Each op: deep-copy -> mutate only the matched segments -> re-derive
masses/BOM (``ground_gene``) -> run the validity gate -> return a DIFF of exactly what changed. Everything
else is preserved byte-for-byte, so the edit is reviewable and one-undo-able (session_state owns the ring).

MEASURED (docs/ai_native_plan.md §0 probe): scale legs length x1.2 -> standing height 0.329->0.391 (+19%),
segment identity preserved, gene valid, appendage discovery unchanged. Ops return ``(new_gene, diff)`` or
raise ``EditError`` with a TEACHING message (SWE-agent ACI: the error tells the agent how to fix it).
"""
from __future__ import annotations

import re


class EditError(Exception):
    """A localized edit could not be applied (bad group / factor / it broke a validity gate). Message teaches."""


def _num(value, name, *, cast=float):
    """Coerce a user-supplied edit arg to a number with a TEACHING error, never a raw ValueError. A tester who
    passes {factor:'big'} should read 'factor must be a number', not a Python stack-trace string."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise EditError(f"{name} must be a number, got {value!r}")


# group name -> substrings that identify its segments (robust across the product's builders; a bespoke LLM gene
# still names limbs recognizably). "all" matches everything.
_GROUP_WORDS = {
    "legs": ("leg", "thigh", "shank", "femur", "tibia", "shin", "calf", "coxa", "hip"),
    "arms": ("arm", "forearm", "upper_arm", "wrist", "elbow", "shoulder"),
    "torso": ("torso", "body", "trunk", "chest", "abdomen", "spine", "base_link"),
    "head": ("head", "skull", "snout", "cranium"),
    "neck": ("neck",),
    "tail": ("tail",),
    "feet": ("foot", "feet", "toe"),
}
_DIMS = ("length", "girth", "both")


def _clone(gene):
    """A genuinely INDEPENDENT copy. ``RobotGene.to_dict`` passes ``geometry`` through by REFERENCE
    (``"geometry": s.geometry``), so a plain to_dict/from_dict round-trip hands the clone the original's own
    geometry dicts -- and ``scale_group`` mutates geometry in place (``_scale_geo``/``_scale_geo_length``). The
    edit therefore leaked back into the caller's gene. Latent while every operator was called once and
    committed; it surfaced the moment ``set_height`` began probing several factors to solve for a target, where
    each probe compounded on the last (measured: factor 1.70 -> 1.39 m, then an identical 1.75 -> 2.44 m, then
    4.04, diverging instead of converging). Deep-copy the dict before rebuilding."""
    import copy

    from virturoid.schemas.gene import RobotGene
    return RobotGene.from_dict(copy.deepcopy(gene.to_dict()))


def _dominant_material(gene) -> str:
    from collections import Counter
    mats = Counter((s.material or "") for s in gene.segments if s.material)
    return (mats.most_common(1)[0][0] if mats else "") or "aluminum"


# short tokens that are substrings of unrelated words get a WORD boundary so "hip" matches FL_hip / hip_joint
# but never "battleship"/"microchip"; long distinctive tokens keep cheap substring matching. ("hip" was added to
# the legs group so `scale_group legs` reaches a Unitree-style FL_hip/FR_hip chain — #216-adjacent.)
_BOUNDED_WORDS = frozenset({"hip", "arm", "leg", "toe", "shin", "calf"})


def _name_in_group(name: str | None, words) -> bool:
    nm = (name or "").lower()
    for w in words:
        if w in _BOUNDED_WORDS:
            if re.search(rf"(?:^|[^a-z]){re.escape(w)}(?:$|[^a-z])", nm):
                return True
        elif w in nm:
            return True
    return False


def segments_for_group(gene, group: str) -> list:
    """The segments a group name refers to. 'all' -> every segment. Unknown -> EditError listing valid groups."""
    group = (group or "").lower().strip()
    if group == "all":
        return list(gene.segments)
    words = _GROUP_WORDS.get(group)
    if words is None:
        raise EditError(f"unknown group '{group}'; valid groups: {sorted(_GROUP_WORDS) + ['all']}")
    hits = [s for s in gene.segments if _name_in_group(s.name, words)]
    if group == "arms" and (gene.robot_class or "").lower() == "manipulator":
        # MEASURED live 2026-07-22: the composed manipulator names its REAL links j1/j2 (0.325 m each), so the
        # word list matched only the 0.05 m shoulder/wrist stubs — "make the arm longer" scaled two joints,
        # moved reach 0.87 -> 0.90 m for a 1.3x request, and the barely-changed geometry verdicted STUCK. On a
        # MANIPULATOR the arm IS the actuated chain: everything except the welded base and the gripper fingers.
        import re as _re
        chain = [s for s in gene.segments
                 if _re.fullmatch(r"(?:j|joint|link|seg)[\s_]?\d+", (s.name or "").lower())]
        merged = {id(s): s for s in hits + chain}
        return list(merged.values())
    return hits


def _standing_height(gene) -> float:
    try:
        from virturoid.services.gene_compiler import standing_spawn_z
        return round(float(standing_spawn_z(gene)), 4)
    except Exception:  # noqa: BLE001
        return 0.0


def _reground_and_gate(gene, *, material: str):
    """Re-derive masses/BOM for the mutated geometry, then GATE: the gene must still validate. Teaches on fail."""
    from virturoid.services.grounded_physics import ground_gene
    try:
        ground_gene(gene, material=material, fill=0.25)
    except Exception:  # noqa: BLE001 - grounding is value-add; a mutated gene can still be scored/rendered
        pass
    issues = gene.validate()
    if issues:
        raise EditError(f"edit would make the robot invalid ({'; '.join(issues[:2])}); "
                        "try a smaller factor or a different group")


def scale_group(gene, *, group: str = "legs", dims: str = "length", factor: float = 1.2, only=None):
    """LENGTHEN / THICKEN a group of segments by ``factor`` (dims: length | girth | both). The workhorse:
    'make it taller' -> scale_group(legs, length, ~1.2). Only the matched segments change; mass/BOM re-derive.

    ``only`` narrows the selection to an explicit set of segment names (used by :func:`set_height`, which picks
    the height-bearing chain structurally rather than by word list). Internal — not exposed as an op argument.
    """
    if dims not in _DIMS:
        raise EditError(f"dims must be one of {_DIMS}, got '{dims}'")
    f = _num(factor, "factor")
    if not (0.2 <= f <= 5.0):
        raise EditError(f"factor {factor} out of the safe range [0.2, 5.0]")
    g = _clone(gene)
    targets = segments_for_group(g, group)
    if only is not None:
        targets = [s for s in targets if s.name in only]
        if not targets:
            raise EditError(f"none of the {len(only)} requested segment(s) exist on this robot")
    if not targets:
        raise EditError(f"no '{group}' segments on this robot (it is a {g.robot_class}); "
                        f"available groups here: {[k for k in _GROUP_WORDS if segments_for_group(g, k)] + ['all']}")
    from virturoid.services.bom_builder import _scale_geo, _scale_geo_length
    children_of: dict = {}
    for c in g.segments:
        if c.parent is not None:
            children_of.setdefault(c.parent, []).append(c)
    target_names = {s.name for s in targets}
    changed = []
    for s in targets:
        before = (round(s.length_m, 4), round(s.radius_m, 4))
        if dims in ("length", "both"):
            s.length_m = round(s.length_m * f, 5)
            if s.geometry:
                _scale_geo_length(s.geometry, f)          # #216a: keep the VISUAL link as long as the collider
        if dims in ("girth", "both"):
            s.radius_m = round(s.radius_m * f, 5)
            if s.geometry:
                _scale_geo(s.geometry, f)                 # cross-section only
        # #216b: a child attaches at pos = (mount_x, mount_y, parent.length + mount_z). Those offsets bake in
        # the parent's OLD length (e.g. a thigh mounts at the torso base with mount_z = -torso_length), so when
        # the parent scales they must scale too or the child drifts off its anchor (detaches). Scale the z
        # offset with LENGTH and the lateral offsets with GIRTH; skip children already in the target set for
        # that axis (they'd be double-scaled through their own edit). Ambient/anatomy-graph bodies store
        # mount_bounds separately, so this only corrects the parametric offsets that actually place the child.
        for child in children_of.get(s.name, []):
            mo = list(getattr(child, "mount_offset", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
            if dims in ("length", "both"):
                mo[2] = round(mo[2] * f, 5)
            if dims in ("girth", "both"):
                mo[0] = round(mo[0] * f, 5); mo[1] = round(mo[1] * f, 5)
            child.mount_offset = tuple(mo)
        changed.append({"segment": s.name, "length_m": [before[0], round(s.length_m, 4)],
                        "radius_m": [before[1], round(s.radius_m, 4)]})
    h0 = _standing_height(gene)
    _reground_and_gate(g, material=_dominant_material(gene))
    h1 = _standing_height(g)
    diff = {"op": "scale_group", "group": group, "dims": dims, "factor": round(f, 3),
            "n_changed": len(changed), "changed": changed[:8], "n_segments_total": len(g.segments),
            "standing_height_m": [h0, h1]}
    return g, diff


#: how close ``set_height`` must land before it will call itself a success. 2% of the target, floored at 5 mm.
_HEIGHT_TOL_FRAC = 0.02
_HEIGHT_TOL_FLOOR_M = 0.005


def height_bearing_segments(gene) -> list[str]:
    """The links that ACTUALLY carry standing height: every ancestor-or-self of a ground-contacting leaf.

    Structural, not lexical, and that is the whole point. ``segments_for_group(gene, "legs")`` matches on a
    word list (``leg|thigh|shank|femur|tibia|shin|calf|coxa|hip``) which knows nothing about ``knee``,
    ``ankle`` or ``pelvis`` -- so on an imported Unitree G1, whose links are named ``left_hip_yaw_link`` /
    ``left_knee_link`` / ``left_ankle_roll_link``, it selected the SIX hip bodies and nothing else. Scaling
    those alone lengthened the hip blocks while the knee/ankle chain below them kept its own length and had
    its mount offsets scaled anyway: measured, that left the thighs floating off the pelvis and the shins and
    feet hanging below with a visible gap. Walking the tree instead reaches all twelve leg links on the G1,
    every leg on a hexapod, and the whole support chain on any customer robot regardless of its naming.

    Falls back to the ``legs`` word group, then to every segment, if the body cannot be compiled/measured.
    """
    try:
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False, spawn_z=0.1))
        d = mujoco.MjData(mj)
        if mj.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj, d, 0)
        mujoco.mj_forward(mj, d)
        # lowest world z reached by each BODY (its own geoms only), via each geom's tight local AABB
        aabb = mj.geom_aabb.reshape(mj.ngeom, 6)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        low: dict[int, float] = {}
        for g in range(mj.ngeom):
            b = int(mj.geom_bodyid[g])
            if b == 0:
                continue
            corners = aabb[g, :3] + signs * aabb[g, 3:]
            wz = float((d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
            low[b] = min(low.get(b, wz), wz)
        if not low:
            raise ValueError("no body geoms")
        floor = min(low.values())
        span = max(max(low.values()) - floor, 1e-6)
        # every body whose lowest point is within 15% of the body's own vertical span of the ground -- the feet,
        # including a second/third pair a quadruped or hexapod stands on.
        contacts = {b for b, z in low.items() if (z - floor) <= 0.15 * span}
        by_name = {s.name: s for s in gene.segments}
        chain: set[str] = set()
        for b in contacts:
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            seg = by_name.get(name)
            guard = 0
            while seg is not None and guard < 200:       # walk to the root; the root itself does not add height
                guard += 1
                if seg.parent is None:
                    break
                chain.add(seg.name)
                seg = by_name.get(seg.parent)
        if chain:
            return sorted(chain)
    except Exception:  # noqa: BLE001 - measurement aid; a compile failure falls back to the lexical group
        pass
    legs = segments_for_group(gene, "legs")
    return sorted(s.name for s in legs) if legs else sorted(s.name for s in gene.segments)


def set_height(gene, *, target_m: float):
    """Make the robot stand at ~``target_m`` by scaling the height-bearing chain, and FAIL if it misses.

    Two defects used to live here and they compounded. (1) The chain was chosen by name, so an imported
    humanoid had only its hip blocks scaled -- see :func:`height_bearing_segments`. (2) The factor was solved
    once, as ``target / current``, and whatever came out was reported as success. Standing height is AFFINE in
    the scale factor, not proportional (there is a fixed mount reference and a ground clearance in it), so one
    division cannot hit the target even when the right links are scaled. Measured on a Menagerie Unitree G1
    asked for 1.5 m: 6 of 30 links scaled 1.70x, the body landed at 1.117 m -- a 0.383 m miss -- and the tool
    returned ``ok: true`` with a tidy diff and no warning.

    Now: solve on the real chain, refine against the measured height, and raise :class:`EditError` (which the
    ``edit_robot`` tool turns into ``ok: false``) with the measured miss rather than shipping a body that is
    neither the old height nor the requested one.
    """
    tm = _num(target_m, "target_m")
    if not (0.05 <= tm <= 5.0):
        raise EditError(f"target height {target_m} m is implausible (expected 0.05-5.0 m)")
    cur = _standing_height(gene)
    if cur <= 1e-3:
        raise EditError("cannot measure the robot's current height to scale toward a target")
    names = height_bearing_segments(gene)
    tol = max(_HEIGHT_TOL_FRAC * tm, _HEIGHT_TOL_FLOOR_M)

    # Refine the cumulative factor against the MEASURED height. h(f) is affine in f, so a secant step converges
    # in one or two rounds; each attempt is applied to the ORIGINAL body so factors never compound silently.
    best = None
    probes: list[tuple[float, float]] = [(1.0, cur)]
    f = tm / cur
    for _ in range(6):
        f = min(5.0, max(0.2, f))
        if any(abs(f - pf) < 1e-6 for pf, _ in probes):
            break
        try:
            g, diff = _scale_named(gene, names, f)
        except EditError:
            # This factor makes the body invalid (a link out of range, a failed re-ground). That is a real
            # limit on how far this body can be scaled, not a reason to abandon the search -- fall back toward
            # the best factor found so far and report the honest miss if nothing lands.
            if best is None:
                raise
            f = (f + best[3]) / 2.0
            continue
        h = float(diff["standing_height_m"][1])
        probes.append((f, h))
        if best is None or abs(h - tm) < abs(best[2] - tm):
            best = (g, diff, h, f)
        if abs(h - tm) <= tol:
            break
        (f0, h0), (f1, h1) = probes[-2], probes[-1]
        if abs(h1 - h0) < 1e-9 or abs(f1 - f0) < 1e-12:
            break
        slope = (h1 - h0) / (f1 - f0)
        f = f1 + (tm - h1) / slope

    if best is None:
        raise EditError(f"could not scale this body toward {tm:g} m at all "
                        f"(no height-bearing links found among {len(gene.segments)} segments)")
    g, diff, got, used = best
    diff.update({"op": "set_height", "target_m": round(tm, 4), "achieved_m": round(got, 4),
                 "miss_m": round(got - tm, 4), "factor": round(used, 4),
                 "height_bearing_links": len(names)})
    if abs(got - tm) > tol:
        raise EditError(
            f"set_height MISSED: asked for {tm:g} m, the body reaches {got:.3f} m "
            f"({got - tm:+.3f} m, {100 * abs(got - tm) / tm:.1f}% off) after scaling its "
            f"{len(names)} height-bearing link(s) by {used:.3f}x. The edit was NOT applied — this body cannot "
            f"reach that height by scaling alone (its {'reach' if got < tm else 'range'} is limited by joint "
            "geometry or the safe scale range). Ask for a height nearer "
            f"{got:.2f} m, or change the topology (set_leg_count / add_limb).")
    return g, diff


def _scale_named(gene, names, factor: float):
    """``scale_group`` over an EXPLICIT segment-name set (the structural height chain), not a word group."""
    return scale_group(gene, group="all", dims="length", factor=factor, only=set(names))


def scale_robot(gene, *, factor: float = 1.2):
    """Uniformly scale the WHOLE robot (all segments, both dims) by ``factor`` — a bigger/smaller version."""
    return scale_group(gene, group="all", dims="both", factor=factor)


def set_material(gene, *, group: str = "all", material: str = "aluminum"):
    """Change the (render/BOM) material of a group of segments; re-derives mass at the new density."""
    known = ("steel", "aluminum", "carbon_fiber", "titanium", "abs_plastic", "shell", "metal", "skeleton", "frame")
    if material not in known:
        raise EditError(f"unknown material '{material}'; known: {known}")
    g = _clone(gene)
    targets = segments_for_group(g, group)
    if not targets:
        raise EditError(f"no '{group}' segments to re-material on this {g.robot_class}")
    for s in targets:
        s.material = material
    _reground_and_gate(g, material=material)
    return g, {"op": "set_material", "group": group, "material": material, "n_changed": len(targets)}


def set_leg_count(gene, *, n_pairs: int):
    """STRUCTURAL edit (confirm-gated by the caller): rebuild the body as an N-pair legged creature. This
    changes topology, so it is a bigger edit than a parameter tweak — flagged ``structural`` in the diff."""
    n = _num(n_pairs, "n_pairs", cast=int)
    if not (1 <= n <= 8):
        raise EditError(f"n_pairs {n_pairs} out of range [1, 8]")
    from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
    old_legs = len({s.name.rsplit("_", 1)[0] for s in segments_for_group(gene, "legs")}) or 0
    g = build_from_anatomy(_generic_legged_graph(n_pairs=n))
    if g is None:
        raise EditError("could not build a body at that leg count")
    _reground_and_gate(g, material=_dominant_material(gene))
    return g, {"op": "set_leg_count", "n_pairs": n, "structural": True,
               "note": "topology changed (legs rebuilt); torso/appendage customization not carried over"}


_ATTACH_SITES = {                      # where on the parent, as a fraction of its own extent (x_frac, y_frac, z_frac)
    "top": (0.0, 0.0, 1.0), "bottom": (0.0, 0.0, -1.0), "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0), "tip": (0.0, 0.0, 0.0),
}


def add_limb(gene, *, parent: str | None = None, segments: int = 3, length_m: float = 0.25,
             radius_m: float = 0.03, attach: str = "top", joint_axes: list | None = None,
             end_effector: str = "gripper", name: str = "limb", taper: float = 0.82,
             payload_kg: float = 0.5):
    """STRUCTURAL edit: GROW A NEW ARTICULATED CHAIN on an existing body. 'Give it a third arm', 'add a tail',
    'put a sensor mast on the back' are all this one op.

    Every other operator RESIZES or RESTYLES what is already there -- measured, the whole edit vocabulary was
    scale_group / set_height / scale_robot / set_material / set_leg_count / set_payload / adopt_walkable_template,
    so a customer could make their robot taller but could not add anything to it. set_leg_count looks like an
    exception but it REBUILDS the body from a template, discarding the customer's own robot.

    Deliberately NOT ``add_arm``: an arm, a tail, a neck, a mast and a leg are the same structure -- a serial
    chain of N tapering links with revolute joints, optionally ending in a tool. Encoding the ROLE in the op name
    would mean a new op per body part, which is the enum-per-species trap this codebase keeps paying off. The
    caller says how many links, how long, where they attach and what tops them; the role is just the name.

    ``attach`` names a face of the parent ('top' for a back-mounted arm, 'front' for a head, 'tip' for a serial
    extension) and is resolved against the PARENT'S OWN size, so the same call works on a Go2 and on a humanoid.
    """
    n = _num(segments, "segments", cast=int)
    if not (1 <= n <= 8):
        raise EditError(f"segments {segments} out of range [1, 8]")
    seg_len = _num(length_m, "length_m")
    seg_rad = _num(radius_m, "radius_m")
    if not (0.01 <= seg_len <= 2.0):
        raise EditError(f"length_m {length_m} out of range [0.01, 2.0] per link")
    if not (0.004 <= seg_rad <= 0.3):
        raise EditError(f"radius_m {radius_m} out of range [0.004, 0.3]")
    site = str(attach or "top").lower()
    if site not in _ATTACH_SITES:
        raise EditError(f"attach {attach!r} unknown; use one of {sorted(_ATTACH_SITES)}")

    from virturoid.schemas.gene import GeneSegment
    g = _clone(gene)
    by = {s.name: s for s in g.segments}
    if parent:
        host = by.get(str(parent))
        if host is None:
            raise EditError(f"no segment named {parent!r}; this robot has "
                            f"{sorted(by)[:12]}{' ...' if len(by) > 12 else ''}")
    else:                                          # default to the ROOT (the trunk/chassis), the usual mount
        host = next((s for s in g.segments if s.parent is None), None)
        if host is None:
            raise EditError("this robot has no root segment to attach a limb to")

    base = name if name not in by else f"{name}_{sum(1 for k in by if k.startswith(name)) + 1}"
    fx, fy, fz = _ATTACH_SITES[site]
    hr = float(host.radius_m or 0.03)
    hl = float(host.length_m or 0.1)
    # mount_offset is measured from the PARENT'S TIP (the compiler puts a child at (x, y, parent.length + z)),
    # so a 'top' mount sits at the tip and a 'back' mount steps back along the parent's own length.
    off = (round(fx * hl * 0.5, 5), round(fy * hr, 5), round((fz - 1.0) * hl * 0.5 if fz else -hl * 0.5, 5))
    axes = list(joint_axes or [])
    added = []
    prev = host.name
    for i in range(n):
        axis = tuple(axes[i]) if i < len(axes) else ((0.0, 0.0, 1.0) if i == 0 else (0.0, 1.0, 0.0))
        r_i = round(max(0.004, seg_rad * (taper ** i)), 5)
        seg = GeneSegment(
            name=f"{base}_{i}", parent=prev, shape="capsule", length_m=round(seg_len, 5), radius_m=r_i,
            mass_kg=round(max(0.02, 1.2 * seg_len * r_i * 30), 4), joint_type="revolute", joint_axis=axis,
            joint_lower=-2.6, joint_upper=2.6,
            mount_offset=off if i == 0 else (0.0, 0.0, 0.0),
            actuator_torque_nm=round(max(2.0, 40.0 * (taper ** i)), 2), is_end_effector=False)
        g.segments.append(seg)
        added.append(seg.name)
        prev = seg.name
    tool = str(end_effector or "").lower()
    if tool in ("gripper", "hand", "tool", "pad"):
        tip = GeneSegment(
            name=f"{base}_{tool}", parent=prev, shape="box",
            length_m=round(max(0.03, 0.28 * seg_len), 5), radius_m=round(max(0.008, seg_rad * 0.9), 5),
            mass_kg=round(max(0.02, 0.35 * seg_len * seg_rad * 30), 4), joint_type="fixed",
            is_end_effector=False)
        g.segments.append(tip)
        added.append(tip.name)

    # SIZE EACH MOTOR FROM THE LOAD IT ACTUALLY HOLDS, not a constant. Measured with a flat 40 N.m placeholder,
    # re-grounding drove every joint of a 0.22 m arm link to 520 N.m and the catalog -- correctly -- answered
    # with a 104 mm motor, giving a can/tube ratio of 2.21. That is the "long stick with two circles": the
    # actuator models are real, the SPEC handed to them was not. For reference a Unitree Z1 (the arm that
    # actually mounts on a Go2) lifts 3 kg at 0.7 m on ~30 N.m joints, and on every real arm the joint housing
    # and the link tube are near the SAME diameter (UR5e ~127 mm tapering to ~90 mm, housings matching) -- the
    # arm reads as a continuous limb, never as beads on a rod.
    #
    # Static worst case: everything distal to this joint held at full extension. Mass acts at the sub-chain's
    # centroid, so torque = m_downstream * g * r_centroid, x1.5 for dynamics/grip margin.
    _new = [s for s in g.segments if s.name in added]
    _pk = max(0.0, _num(payload_kg, "payload_kg"))
    for idx, s in enumerate(_new):
        if s.joint_type != "revolute":
            continue
        downstream = _new[idx:]
        m_down = sum(float(x.mass_kg or 0.0) for x in downstream) + _pk
        reach = sum(float(x.length_m or 0.0) for x in downstream)
        r_com = (0.5 * reach) if _pk <= 0 else (0.5 * reach * (1 - _pk / max(m_down, 1e-6))
                                                + reach * (_pk / max(m_down, 1e-6)))
        s.torque_req_nm = round(max(1.0, m_down * 9.81 * max(r_com, 0.02) * 1.5), 2)
        s.actuator_torque_nm = s.torque_req_nm
    _reground_and_gate(g, material=_dominant_material(gene))
    return g, {"op": "add_limb", "parent": host.name, "attach": site, "segments_added": added,
               "n_actuated_added": n, "structural": True,
               "note": f"grew a {n}-link chain on {host.name!r}; the robot's existing structure is untouched"}


def set_payload(gene, *, payload_kg: float = 2.0, girth_scale: bool = True):
    """CAPABILITY amend: make the robot CARRY / LIFT a target payload by upsizing its actuators (and, optionally,
    the load-path link girth) so its joints can hold the extra load. The actuated joints must support the robot's
    own weight PLUS the payload, so each joint's required torque is scaled by ``(total_mass + payload)/total_mass``;
    re-grounding then selects BIGGER real motors for that torque and the BOM/mass rise honestly (a stronger robot
    costs more + weighs more). This is what turns 'make it lift 10 kg' into a real amendment, not just a wish."""
    pk = _num(payload_kg, "payload_kg")
    if not (0.1 <= pk <= 50.0):
        raise EditError(f"payload {payload_kg} kg out of the safe range [0.1, 50.0]; "
                        "for heavier loads redesign with a larger actuator class")
    g = _clone(gene)
    actuated = [s for s in g.segments if s.joint_type in ("revolute", "prismatic")]
    if not actuated:
        raise EditError("this robot has no actuated joints, so it cannot be amended to carry a payload")
    total_mass = sum(float(s.mass_kg or 0.0) for s in g.segments) or 1.0
    load_factor = (total_mass + pk) / total_mass
    changed = []
    for s in actuated:                                          # scale the joint's torque REQUIREMENT (not the last
        req0 = s.torque_req_nm if s.torque_req_nm is not None else abs(s.actuator_torque_nm or 8.0)
        before = round(float(req0), 2)                          # motor's peak) so re-grounding upsizes the motor and
        s.torque_req_nm = round(before * load_factor, 2)        # stays idempotent under repeated grounding
        s.actuator_torque_nm = s.torque_req_nm                  # keep them consistent until ground re-selects the peak
        changed.append({"segment": s.name, "required_nm": [before, round(s.torque_req_nm, 2)]})
    if girth_scale and load_factor > 1.05:                      # thicken the load-path limbs (sub-linear) for rigidity
        girth_mult = min(1.4, float(load_factor) ** 0.3)
        for s in actuated:
            if s.radius_m and s.radius_m > 0:
                s.radius_m = round(max(0.005, s.radius_m * girth_mult), 5)
    required_by_name = {c["segment"]: c["required_nm"][1] for c in changed}
    # RECORD WHAT THE ROBOT IS NOW RATED FOR. Nothing downstream could previously tell a heavy robot from a robot
    # BUILT HEAVY ON PURPOSE, because the requested payload was applied to the joints and then forgotten. That is
    # not cosmetic: gene_validation's mass_budget screens total mass against the class band for an UNLOADED
    # machine, so amending a clean 12.8 kg quadruped to carry 15 kg produced a NEW "implausibly heavy" finding and
    # the amend gate auto-reverted the customer's own explicit request. (Boston Dynamics' Spot is 32.5 kg and
    # rated for 14 kg -- the band was calling real hardware implausible.) With the rating on the gene, the band
    # can widen by exactly what was asked for and by nothing else.
    md = dict(getattr(g, "metadata", None) or {})
    md["rated_payload_kg"] = round(float(pk), 3)
    g.metadata = md
    mass0 = round(total_mass, 3)
    _reground_and_gate(g, material=_dominant_material(gene))    # upsizes actuators for the new torque; re-derives mass
    mass1 = round(sum(float(s.mass_kg or 0.0) for s in g.segments), 3)
    # HONEST saturation check: if a joint's chosen real motor can't actually meet the scaled requirement, the
    # payload exceeds what the actuator catalog can drive -- say so (a gearbox / bigger class is needed) rather
    # than pretending the maxed-out motor is enough.
    undersized = []
    for s in g.segments:
        req = required_by_name.get(s.name)
        if req is not None and float(s.actuator_torque_nm or 0.0) + 1e-6 < float(req):
            undersized.append({"segment": s.name, "required_nm": round(float(req), 1),
                               "best_motor_nm": round(float(s.actuator_torque_nm or 0.0), 1)})
    out = {"op": "set_payload", "payload_kg": pk, "load_factor": round(load_factor, 3),
           "n_joints_upsized": len(changed), "changed": changed[:8], "total_mass_kg": [mass0, mass1],
           "note": "joint torque raised for the payload -> re-grounding upsized the real motors (mass/cost rose)"}
    if undersized:
        out["undersized_joints"] = undersized[:8]
        out["warning"] = (f"{len(undersized)} joint(s) exceed the strongest catalog motor for this payload -- add a "
                          "gearbox / reduction stage or split the load; the design is over the actuator envelope here")
    return g, out


# op name -> callable(gene, **args). The typed operator library the intent-classifier maps requests onto.
def adopt_walkable_template(gene, **_):
    """B2: EXPLICITLY replace an imported/composed quadruped's body with a size-matched walkable fanned template.
    Only ever invoked by the customer (never automatically) -- ingest KEEPS the original geometry and merely
    OFFERS this. Lands as one undo step, so 'undo' restores the customer's original body exactly."""
    from virturoid.services.anatomy_compiler import ensure_walkable_quad
    before = len(gene.segments)
    new = ensure_walkable_quad(gene, "adopt walkable template", force=True)
    applied = bool(dict(getattr(new, "metadata", None) or {}).get("walkability_fallback", {}).get("applied"))
    return new, {"op": "adopt_walkable_template", "applied": applied, "segments_before": before,
                 "segments_after": len(new.segments),
                 "note": ("adopted a size-matched walkable template (undo to restore the original body)" if applied
                          else "the original body already walks or no better template was found -- unchanged")}


OPERATORS = {
    "scale_group": scale_group,
    "set_height": set_height,
    "scale_robot": scale_robot,
    "set_material": set_material,
    "set_leg_count": set_leg_count,
    "set_payload": set_payload,
    "add_limb": add_limb,
    "adopt_walkable_template": adopt_walkable_template,
}
_STRUCTURAL = {"set_leg_count", "adopt_walkable_template", "add_limb"}


def op_specs() -> list[dict]:
    """JSON-schema-ish specs of the edit operators (for the intent classifier + tool docs)."""
    return [
        {"op": "scale_group", "args": {"group": list(_GROUP_WORDS) + ["all"], "dims": list(_DIMS), "factor": "0.2-5.0"},
         "for": "lengthen/thicken part of the robot, e.g. taller = scale_group legs length 1.2"},
        {"op": "set_height", "args": {"target_m": "0.05-5.0"}, "for": "stand at a specific height"},
        {"op": "scale_robot", "args": {"factor": "0.2-5.0"}, "for": "make the whole robot bigger/smaller"},
        {"op": "set_material", "args": {"group": list(_GROUP_WORDS) + ["all"], "material": "steel|aluminum|carbon_fiber|titanium|..."},
         "for": "change what a part is made of"},
        {"op": "set_leg_count", "args": {"n_pairs": "1-8"}, "for": "STRUCTURAL: change how many legs (rebuilds)"},
        {"op": "add_limb", "args": {"parent": "<segment name, default the root>", "segments": "1-8",
                                    "length_m": "0.01-2.0", "radius_m": "0.004-0.3",
                                    "attach": sorted(_ATTACH_SITES), "end_effector": "gripper|hand|tool|pad|none",
                                    "name": "<prefix, e.g. arm3 / tail / mast>"},
         "for": "STRUCTURAL: GROW a new articulated chain on the existing body — 'add a third arm', 'add a "
                "tail', 'put a sensor mast on the back'. Keeps the robot; only adds."},
        {"op": "set_payload", "args": {"payload_kg": "0.1-50.0", "girth_scale": "true|false"},
         "for": "make it CARRY/LIFT heavier: upsize actuators (+ load-path girth) for the payload; BOM/mass rise"},
        {"op": "adopt_walkable_template", "args": {},
         "for": "OPT-IN: replace an imported quadruped that can't walk with a size-matched walkable template (undoable)"},
    ]


def apply_op(gene, op: str, args: dict | None = None):
    """Apply one typed operator. Returns ``(new_gene, diff)`` or raises ``EditError``."""
    fn = OPERATORS.get(op)
    if fn is None:
        raise EditError(f"unknown edit op '{op}'; valid ops: {sorted(OPERATORS)}")
    new_gene, diff = fn(gene, **(args or {}))
    diff["structural"] = op in _STRUCTURAL or diff.get("structural", False)
    return new_gene, diff


def apply_ops(gene, ops: list[dict]):
    """Apply a sequence of ops, threading the gene. Returns ``(final_gene, [diffs])``; the FIRST failure raises
    (nothing is committed by this module — the caller commits the returned gene as one undo step)."""
    g = gene
    diffs = []
    for spec in (ops or []):
        g, d = apply_op(g, str(spec.get("op")), spec.get("args") or {})
        diffs.append(d)
    return g, diffs
