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


def _shape_key(gene) -> dict:
    """``{name: (shape, length_m, radius_m)}`` — what an op has to have CHANGED for a re-derived mass to be
    the customer's own request rather than a side effect."""
    return {s.name: (str(s.shape), round(float(s.length_m or 0.0), 6), round(float(s.radius_m or 0.0), 6))
            for s in gene.segments}


def _reground_and_gate(gene, *, material: str, original=None, respec=None, keep_mass=None) -> dict:
    """Re-derive masses/BOM for the mutated geometry, then GATE: the gene must still validate. Teaches on fail.

    AND DO NOT RE-MASS THE CUSTOMER'S ROBOT ON THE WAY THROUGH. This called ``ground_gene`` bare, which takes
    ``preserve_mass=False`` -- so every link's mass was re-derived from (primitive volume x density x fill) +
    one of OUR catalog motors, on a body whose manufacturer masses already include its own motors. Measured
    through ``call_tool`` on a real Menagerie Unitree Go2 ingested with ``mass_provenance.preserved: true,
    delta_kg: 0.0``, then ONE ``add_limb``:

        total 15.206 -> 32.683 kg, of which the arm is 2.733 -- so +14.7 kg is the customer's own Go2 being
        silently re-weighed: base 6.921 -> 10.306, every thigh 1.152 -> 2.050, every calf 0.241 -> 2.177 (9x)

    ...and ``metadata['mass_source']`` still read ``source_model``, so every door downstream would go on citing
    Unitree for our number. ``ingest_project`` already routes an authoritative body through
    ``gene_build.ground_and_repair`` for exactly this reason; the edit door was the one that did not, which is
    why the guarantee survived ingest and died on the first amend.

    The rule is per LINK, because provenance is per link:

      * links this op CREATED (in ``gene`` but not in ``original``) have no manufacturer number to keep;
      * links this op RESIZED (shape/length/radius moved) or RE-SPECIFIED (``respec`` -- e.g. ``set_payload``
        raising a joint's torque requirement so a bigger motor is selected) were changed BY THE CUSTOMER, so
        re-deriving them is honouring the request, not overriding it;
      * every other link keeps the mass it arrived with.

    ``keep_mass`` is the fourth case and it OVERRIDES the other three: links whose mass is a number the customer
    STATED rather than one to derive -- a payload. It has to beat both flags, because a payload link is new (so
    the ``added`` rule would derive it from box volume on an imported body) and every link is derived on a
    composed one. See ``grounded_physics.ground_gene``'s ``preserve_mass_links``.

    Returns a mass ledger the operator puts in its diff, so a mass that does move is stated with its number
    instead of being discovered later on a spec sheet.
    """
    from virturoid.services.gene_build import grounding_config
    from virturoid.services.grounded_physics import ground_gene
    before_mass = {s.name: float(s.mass_kg or 0.0) for s in gene.segments}
    cfg = grounding_config(gene)
    keep: set[str] = {str(n) for n in (keep_mass or ())}
    # A load ALREADY on the robot stays the customer's stated number through every LATER edit too, not only the
    # one that put it there -- otherwise a `scale_robot` after a `set_payload` re-derives 25 kg of cargo from
    # the scaled box's volume and the robot quietly stops carrying what was asked for.
    keep |= {str(n) for n in ((((getattr(gene, "metadata", None) or {}).get("embodied_mass") or {})
                               .get("payload_kg")) or {})}
    derive: set[str] = set(str(n) for n in (respec or ()))
    added: set[str] = set()
    if original is not None:
        was, now = _shape_key(original), _shape_key(gene)
        added = {n for n in now if n not in was}
        derive |= added | {n for n in now if n in was and now[n] != was[n]}
        before_mass = {s.name: float(s.mass_kg or 0.0) for s in original.segments}
    derive -= keep
    try:
        if cfg["preserve_mass"]:
            # The body's masses are the manufacturer's. Ground with the same config the EXPORT door uses, so
            # the robot the customer edits cannot differ from the one that leaves the building.
            ground_gene(gene, material=cfg["material"], fill=cfg["fill"], preserve_mass=True,
                        derive_mass_links=derive, preserve_mass_links=keep)
        else:
            ground_gene(gene, material=material, fill=0.25, preserve_mass_links=keep)
            # ...and put the parts list's own battery/compute/sensors back on. A re-ground re-derives every
            # link mass from geometry + motor, which DROPS what ``embody_component_masses`` had bolted on, so
            # without this an amend quietly took ~2.9 kg of real hardware off the robot and the mass ledger
            # below would report the loss as if the customer's edit had caused it. Skipped on a preserved body
            # for the same reason embodiment declines there: those masses are the manufacturer's.
            from virturoid.services.grounded_physics import embody_component_masses
            embody_component_masses(gene)
    except Exception:  # noqa: BLE001 - grounding is value-add; a mutated gene can still be scored/rendered
        pass
    issues = gene.validate()
    if issues:
        raise EditError(f"edit would make the robot invalid ({'; '.join(issues[:2])}); "
                        "try a smaller factor or a different group")
    return _mass_ledger(before_mass, gene, added=added, preserved=bool(cfg["preserve_mass"]))


def _mass_ledger(before_mass: dict, gene, *, added: set, preserved: bool) -> dict:
    """What this edit did to the robot's mass, per link, in numbers. The disclosure half of the promise.

    ``source_masses_preserved`` IS THE MEASUREMENT, not the intent. It used to be ``bool(preserved)`` -- a copy
    of ``grounding_config()['preserve_mass']``, i.e. "this body's masses CAME FROM the manufacturer", which is
    true of an imported robot no matter what the edit then did to it. Measured through ``call_tool`` on a real
    Menagerie Go2, one ``set_payload{payload_kg: 25}`` printed

        "source_masses_preserved": true   next to   "n_existing_links_remassed": 15

    with FL_calf at 0.241 -> 4.405 kg, and the same constant read ``true`` on ``scale_group``, ``set_height``,
    ``scale_robot`` and ``set_material`` while each of them re-massed 12-13 of the customer's links. Some of
    those re-massings are the request (a longer link weighs more; carbon-fibre legs weigh less) and stay
    exactly as they are -- what changes here is that the flag stops CONTRADICTING the count printed beside it.
    ``mass_authority`` keeps the other fact, separately: whether there were manufacturer masses at all.
    """
    after = {s.name: float(s.mass_kg or 0.0) for s in gene.segments}
    m0 = round(sum(before_mass.values()), 3)
    m1 = round(sum(after.values()), 3)
    new_mass = round(sum(v for n, v in after.items() if n in added), 3)
    moved = [{"segment": n, "mass_kg": [round(before_mass[n], 3), round(after[n], 3)]}
             for n in sorted(after)
             if n in before_mass and abs(after[n] - before_mass[n]) > 1e-3]
    # A link that is GONE took its mass with it, and no per-link comparison can see that -- which is how a
    # body-replacing op (``set_leg_count``, ``adopt_walkable_template``) could discard all 13 of a Go2's links
    # for a template's 20 and still satisfy "no existing link changed mass".
    dropped = sorted(n for n in before_mass if n not in after)
    led = {"total_mass_kg": [m0, m1], "added_mass_kg": new_mass,
           # `or 0.0` so an unchanged robot reads 0.0, never -0.0 -- a minus sign in front of a mass delta is
           # exactly the thing this ledger exists to make readable.
           "existing_mass_changed_kg": round((m1 - m0) - new_mass, 3) or 0.0,
           "n_existing_links_remassed": len(moved), "n_existing_links_dropped": len(dropped),
           "source_masses_preserved": bool(preserved) and not moved and not dropped,
           "mass_authority": "source_model" if preserved else "derived"}
    if moved:
        led["remassed"] = moved[:8]
    if dropped:
        led["dropped"] = dropped[:8]
    # WHOSE NUMBER WAS THROWN AWAY. On a body whose masses are the manufacturer's, a re-derived link does not
    # get "their figure, adjusted" -- it gets OURS, computed from (primitive volume x density x fill) + a
    # catalog motor. Measured through ``call_tool`` on a real Menagerie Go2 (15.206 kg, Unitree's own per-link
    # masses), one op each on a freshly ingested robot:
    #
    #   scale_group{legs, length, 1.2}  15.206 -> 23.621 kg, 12 links re-derived, FL_calf 0.241 -> 2.002 (8.3x)
    #   set_height{target_m: 0.45}      15.206 -> 24.397 kg, 12 links re-derived, FL_calf 0.241 -> 2.056
    #   scale_robot{factor: 1.2}        15.206 -> 30.424 kg, 13 links re-derived, FL_calf 0.241 -> 2.191
    #   set_material{all, carbon_fiber} 15.206 -> 21.935 kg, 13 links re-derived, FL_calf 0.241 -> 1.942
    #
    # The last one moves NO geometry at all, and carbon fibre is lighter than what a Go2 calf is made of, so an
    # 8x rise cannot be a material effect -- it is our model replacing their measurement. Some re-derivation is
    # the request (a longer link does weigh more) and that is not decided here. What is decided here is that a
    # count alone ("12 links changed mass") reads as "your edit moved these" when the truth is "your figures
    # were discarded for ours", so the pairs are named. ``mass_authority`` is deliberately left alone: it
    # answers the separate question of whether manufacturer masses existed at all (pinned by
    # ``test_set_payload.test_source_masses_preserved_is_the_measurement_not_a_constant``).
    if preserved and moved:
        led["n_source_masses_replaced"] = len(moved)
        led["source_masses_replaced"] = [
            {"segment": m["segment"], "your_kg": m["mass_kg"][0], "our_derived_kg": m["mass_kg"][1]}
            for m in moved[:8]]
        worst = max(moved, key=lambda m: abs(m["mass_kg"][1] - m["mass_kg"][0]))
        led["source_mass_note"] = (
            f"{len(moved)} link(s) carried masses from YOUR model and this edit re-derived them from our "
            f"geometry/density model rather than adjusting your figures (worst: {worst['segment']} "
            f"{worst['mass_kg'][0]:.3f} -> {worst['mass_kg'][1]:.3f} kg). edit_robot op:'undo' restores them.")
    return led


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
    mass = _reground_and_gate(g, material=_dominant_material(gene), original=gene)
    h1 = _standing_height(g)
    diff = {"op": "scale_group", "group": group, "dims": dims, "factor": round(f, 3),
            "n_changed": len(changed), "changed": changed[:8], "n_segments_total": len(g.segments),
            "standing_height_m": [h0, h1], "mass": mass}
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
    # A material change IS a mass re-spec for the parts it names (and only those): asking for carbon-fibre legs
    # is asking for lighter legs. `respec` therefore releases exactly those links from mass preservation.
    mass = _reground_and_gate(g, material=material, original=gene, respec={s.name for s in targets})
    return g, {"op": "set_material", "group": group, "material": material, "n_changed": len(targets),
               "mass": mass}


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
    # `original=gene` is honest here even though NOTHING carries over: every link of the rebuilt body is new,
    # so the ledger reports the whole mass as added and states the before/after the customer actually cares
    # about ("my 15.2 kg robot came back as a 13.5 kg template").
    mass = _reground_and_gate(g, material=_dominant_material(gene), original=gene)
    return g, {"op": "set_leg_count", "n_pairs": n, "structural": True, "mass": mass,
               "note": "topology changed (legs rebuilt); torso/appendage customization not carried over"}


#: Attach faces as a direction in the ROBOT'S OWN frame — +x forward, +y left, +z up. NOT in the parent link's
#: local frame, and that distinction is the whole bug this table used to carry. A link's local +z is its LENGTH
#: axis, and on an imported robot the trunk's length axis is horizontal: measured on a Menagerie Unitree Go2,
#: the reconstructed base carries ``mount_euler`` (-1.571, 1.335, 1.335), so its local +z points along world
#: (0.972, 0.234, 0.000) — forward and 13.5 deg to the left. Reading these numbers as parent-local therefore put
#: an ``attach:"top"`` arm 1.07 m FORWARD of the base, 0.26 m off centreline and 0.00 m up: a horizontal
#: broomstick out of the dog's nose. Resolved in the robot's frame, "top" is up on every body, whatever
#: arbitrary frame its own root happens to have been reconstructed in.
_ATTACH_SITES = {
    "top": (0.0, 0.0, 1.0), "bottom": (0.0, 0.0, -1.0), "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0), "tip": (0.0, 0.0, 0.0),
}
#: how far toward the parent's surface the limb's ROOT sits, as a fraction of the parent's half-extent in that
#: direction. <1 so the first link starts INSIDE the shell and the seam is covered (a limb rooted exactly on
#: the surface reads as a separate object stuck to the robot, and trips the no-new-detachment gate).
_MOUNT_INSET = 0.85


def _parent_frame(gene, host_name: str):
    """``(R_world_from_parent, aabb_lo, aabb_hi, parent_world_pos, root_world_pos)`` for ``host_name``, measured
    on the COMPILED body — the only place the parent's real orientation and real extent are both knowable.

    The extent is the parent's OWN COLLIDING geoms — children excluded (they are separate bodies), and the
    cosmetic ones excluded too (mass=0 contype=0 motor cans, collars and fairings are drawn at the joints and
    stick out well past the shell: on the Go2's base they stretch the measured envelope from 0.366 m to 0.624 m
    along the trunk axis, which would mount a 'front' limb 70 mm in front of the robot). What is left is the
    body the physics and the verdict are computed on, which is the right thing to bolt onto.

    ``None`` if the body cannot be compiled/measured.
    """
    try:
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False, spawn_z=0.5))
        d = mujoco.MjData(mj)
        mujoco.mj_forward(mj, d)
        bid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, host_name)
        root = gene.root()
        rid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, root.name) if root is not None else -1
        if bid < 0:
            return None
        aabb = mj.geom_aabb.reshape(mj.ngeom, 6)
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)

        def envelope(colliding_only: bool):
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for gi in range(mj.ngeom):
                if int(mj.geom_bodyid[gi]) != bid:
                    continue
                if colliding_only and not (int(mj.geom_contype[gi]) or int(mj.geom_conaffinity[gi])):
                    continue
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, mj.geom_quat[gi])
                pts = mj.geom_pos[gi] + (aabb[gi, :3] + corners * aabb[gi, 3:]) @ rot.reshape(3, 3).T
                lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
            return (lo, hi) if (np.isfinite(lo).all() and np.isfinite(hi).all()) else None

        box = envelope(True) or envelope(False)    # a link with no collider at all still has to be mountable
        if box is None:
            return None
        lo, hi = box
        return (d.xmat[bid].reshape(3, 3).copy(), lo, hi, d.xpos[bid].copy(),
                (d.xpos[rid].copy() if rid >= 0 else d.xpos[bid].copy()))
    except Exception:  # noqa: BLE001 - measurement aid; the primitive fallback below still places the limb
        return None


def _limb_mount(gene, host, site: str):
    """Where the new chain's first link mounts, and which way it grows: ``(mount_offset, mount_euler, placed)``.

    ``mount_offset`` is in the PARENT'S local frame (the compiler puts a child at ``(x, y, parent.length + z)``)
    and ``mount_euler`` turns the child's local +z — which is the direction the chain extends — to the world
    direction the attach face names. ``placed`` is the disclosure: the anchor in the ROBOT'S frame relative to
    its root, and the direction the limb grows, so the caller can state where the arm went instead of guessing.
    """
    d_world = _ATTACH_SITES[site]
    if site == "tip":                                  # serial extension: continue along the parent's own axis
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), {"attach": "tip", "grows_along": "the parent's own axis"}
    try:
        import numpy as np

        from virturoid.services.anatomy_compiler import _aim_R, _R_to_euler
    except Exception:  # noqa: BLE001
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), {"attach": site}
    d_world = np.array(d_world, float)
    d_world /= (np.linalg.norm(d_world) or 1.0)
    frame = _parent_frame(gene, host.name)
    if frame is None:
        # No compiled measurement: fall back to the parent's PRIMITIVE envelope and assume its local frame is
        # the robot frame (true of every body this composer draws — trunk +z = height, +x = fore-aft).
        R = np.eye(3)
        hl, hr = max(1e-4, float(host.length_m or 0.1)), max(1e-4, float(host.radius_m or 0.03))
        lo, hi = np.array([-hr, -hr, 0.0]), np.array([hr, hr, hl])
        p_world = r_world = np.zeros(3)
        measured = False
    else:
        R, lo, hi, p_world, r_world = frame
        measured = True
    d_local = R.T @ d_world
    centre, half = (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 1e-6)
    # distance from the parent's centre to its surface along d_local (the AABB slab hit)
    reach = min([half[i] / abs(d_local[i]) for i in range(3) if abs(d_local[i]) > 1e-9]
                or [float(np.max(half))])
    anchor = centre + d_local * (reach * _MOUNT_INSET)
    off = anchor - np.array([0.0, 0.0, float(host.length_m or 0.0)])
    euler = _R_to_euler(R.T @ _aim_R(d_world))
    at_robot = (R @ anchor) + p_world - r_world
    return (tuple(round(float(v), 5) for v in off), tuple(round(float(v), 6) for v in euler),
            {"attach": site, "grows_toward": [round(float(v), 3) for v in d_world],
             "anchor_m_from_root": [round(float(v), 4) for v in at_robot],
             "frame": ("measured on the compiled body" if measured
                       else "estimated from the parent's primitive envelope (could not compile the body)")})


def add_limb(gene, *, parent: str | None = None, segments: int = 3, length_m: float = 0.25,
             radius_m: float = 0.03, attach: str = "top", joint_axes: list | None = None,
             end_effector: str = "gripper", name: str = "limb", taper: float = 0.82,
             payload_kg: float = 0.5, rest_angles: list | None = None):
    """STRUCTURAL edit: GROW A NEW ARTICULATED CHAIN on an existing body. 'Give it a third arm', 'add a tail',
    'put a sensor mast on the back' are all this one op.

    Every other operator RESIZES or RESTYLES what is already there -- measured, the whole edit vocabulary was
    scale_group / set_height / scale_robot / set_material / set_leg_count / set_payload / adopt_walkable_template,
    so a customer could make their robot taller but could not add anything to it. set_leg_count looks like an
    exception but it REBUILDS the body from a template, discarding the customer's own robot. (``set_payload``
    now appends one link too -- the load itself -- but a chain of articulated links is still only this op.)

    Deliberately NOT ``add_arm``: an arm, a tail, a neck, a mast and a leg are the same structure -- a serial
    chain of N tapering links with revolute joints, optionally ending in a tool. Encoding the ROLE in the op name
    would mean a new op per body part, which is the enum-per-species trap this codebase keeps paying off. The
    caller says how many links, how long, where they attach and what tops them; the role is just the name.

    ``attach`` names a face of the parent ('top' for a back-mounted arm, 'front' for a head, 'tip' for a serial
    extension). It is resolved in the ROBOT'S frame against the parent's MEASURED extent, so 'top' is the top of
    the machine on a Go2, on a humanoid and on a body whose imported root frame is rotated arbitrarily -- see
    :data:`_ATTACH_SITES` for the 1.07 m broomstick that reading it parent-locally produced. The chain grows
    along the same direction, and the diff's ``placement`` states where it landed relative to the root.

    ``rest_angles`` (radians, one per actuated link, short lists padded with 0) is the REST POSTURE. A chain at
    all-zero joints is a straight line of collinear capsules, which reads as a mast -- correct for a sensor mast
    and not for an arm, whose shoulder and elbow are only visible when they are BENT. It stays an argument
    rather than a per-role default because the whole point of this operator is that a tail, a mast, a neck and
    an arm are one structure; the caller knows which one they asked for.
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
    # WHERE THE FACE ACTUALLY IS, on the body as compiled — see :func:`_limb_mount` / :data:`_ATTACH_SITES`.
    off, euler0, placed = _limb_mount(gene, host, site)
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
            mount_euler=euler0 if i == 0 else (0.0, 0.0, 0.0),
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
    if rest_angles:
        _rest = dict((getattr(g, "metadata", None) or {}).get("rest_pose") or {})
        for _i in range(n):
            try:
                _a = float(rest_angles[_i]) if _i < len(rest_angles) else 0.0
            except (TypeError, ValueError):
                raise EditError(f"rest_angles[{_i}] must be a number (radians), got {rest_angles[_i]!r}") from None
            _rest[f"{base}_{_i}_joint"] = round(max(-2.6, min(2.6, _a)), 5)
        g.metadata = {**(getattr(g, "metadata", None) or {}), "rest_pose": _rest}
    # ONLY THE NEW LINKS ARE OURS TO WEIGH. `original=gene` is what makes that true rather than aspirational --
    # see :func:`_reground_and_gate`. The `note` states the arithmetic, because "the robot's existing structure
    # is untouched" was already printed on the diff of the run that added 14.7 kg to the customer's Go2.
    mass = _reground_and_gate(g, material=_dominant_material(gene), original=gene)
    kept = "" if not mass["n_existing_links_remassed"] else (
        f"; {mass['n_existing_links_remassed']} EXISTING link(s) also changed mass "
        f"({mass['existing_mass_changed_kg']:+.3f} kg) -- see diff['mass']['remassed']")
    return g, {"op": "add_limb", "parent": host.name, "attach": site, "segments_added": added,
               "n_actuated_added": n, "structural": True, "placement": placed, "mass": mass,
               "note": f"grew a {n}-link chain on {host.name!r} at its {site!r} face; it adds "
                       f"{mass['added_mass_kg']:.3f} kg, taking the robot {mass['total_mass_kg'][0]:.3f} -> "
                       f"{mass['total_mass_kg'][1]:.3f} kg{kept}"}


#: Density used to size the PAYLOAD's box, kg/m3. ~1000 is mixed cargo / a sealed battery-or-tool case: dense
#: enough not to draw a room-sized crate for 25 kg, light enough not to draw a lead brick. It only sets what the
#: load LOOKS like and how much room it takes; the mass is the customer's number and is never re-derived from it.
_PAYLOAD_DENSITY_KG_M3 = 1000.0
#: The link name a payload lands on. Fixed and predictable so ``probe_robot``/exports/a second amend can find it.
PAYLOAD_LINK = "payload"
#: Classes that carry a payload at the TOOL rather than on the chassis. Same split ``gene_validation`` uses when
#: it decides whether to hang the payload off the chain tip for the torque-margin check, so the two agree about
#: where the load is.
_TIP_LOADED_CLASSES = ("manipulator", "humanoid", "biped")


def _payload_host(gene):
    """The link a payload rides on: the TOOL for a grasping body, the chassis for anything that carries."""
    if (gene.robot_class or "").lower() in _TIP_LOADED_CLASSES:
        tip = gene.end_effector()
        if tip is not None:
            return tip
        leaves = {s.name for s in gene.segments} - {s.parent for s in gene.segments if s.parent}
        if leaves:
            return next(s for s in gene.segments if s.name in leaves)
    return gene.root() or (gene.segments[0] if gene.segments else None)


def set_payload(gene, *, payload_kg: float = 2.0, girth_scale: bool = True,
                upsize_actuators: str | bool = "auto"):
    """CAPABILITY amend: make the robot CARRY / LIFT a target payload.

    THE PAYLOAD IS ADDED TO THE ROBOT. That is the operation, and for a long time it was the one thing this
    operator did not do: it scaled joint torques, stamped a rating in metadata, and returned
    ``added_mass_kg: 0`` for a 25 kg request. Measured through ``call_tool`` on a real Menagerie Unitree Go2,
    ``set_payload{payload_kg: 25}`` left the robot at exactly its own 15.206 kg of links -- while adding 30.654
    kg of OUR re-derived mass to the customer's own parts. The customer asked to carry 25 kg, carried nothing,
    and gained 30 kg. A stated payload now lands as a real link (:data:`PAYLOAD_LINK`) with exactly that mass on
    the link that would carry it -- the chassis on a legged/wheeled body, the tool on a manipulator -- so the
    sim, the render, the verdict and the export all see the loaded machine.

    Then the joints: the actuated chain must hold the robot's own weight PLUS the payload, so each joint's
    requirement scales by ``(total_mass + payload)/total_mass``.

    ``upsize_actuators`` decides what happens to joints whose limits are THE CUSTOMER'S -- read from their own
    file and marked AUTHORITATIVE at import (:func:`grounded_physics.source_declared_torques`):

      ``"auto"`` (default)  re-spec the joints whose limits are OURS; for the customer's, compute the
                            requirement and PROPOSE the part, changing nothing. Their Go2's 23.7 / 45.43 N.m
                            stay 23.7 / 45.43.
      ``True``              re-spec everything, and name every declared limit that was overwritten in
                            ``source_torque_rewritten``. The capability, on the record, on request.
      ``False``             propose only; no joint is touched.

    The default used to be an unconditional overwrite, and it was invisible: on the Go2 those two numbers became
    360 and 520 N.m -- 15.2x and 11.4x -- on the same call that reported ``source_masses_preserved: true``. A
    limit read off the customer's hardware is a measurement of a machine that exists; a limit WE want is a
    proposal about parts that do not. They must not be written by the same line.

    ``girth_scale`` thickens the load path for rigidity, and it is likewise skipped on a body whose geometry is
    the customer's (an import): re-cutting a Go2's calf from 29.0 to 38.9 mm is a change to their model, not to
    ours. It applies as before to bodies we composed.
    """
    from virturoid.services.gene_build import grounding_config
    from virturoid.services.grounded_physics import source_declared_torques
    pk = _num(payload_kg, "payload_kg")
    if not (0.1 <= pk <= 50.0):
        raise EditError(f"payload {payload_kg} kg out of the safe range [0.1, 50.0]; "
                        "for heavier loads redesign with a larger actuator class")
    mode = str(upsize_actuators).lower() if not isinstance(upsize_actuators, bool) else upsize_actuators
    if mode not in (True, False, "auto"):
        raise EditError(f"upsize_actuators must be 'auto', true or false, got {upsize_actuators!r}")
    g = _clone(gene)
    actuated = [s for s in g.segments if s.joint_type in ("revolute", "prismatic")]
    if not actuated:
        raise EditError("this robot has no actuated joints, so it cannot be amended to carry a payload")
    host = _payload_host(g)
    if host is None:
        raise EditError("this robot has no link to mount a payload on")

    # WHOSE NUMBER IS THIS JOINT'S LIMIT? Empty for every body we generate, so a composed robot behaves exactly
    # as before; populated for an import, where each entry is a figure out of the customer's own file.
    declared = source_declared_torques(gene)
    cfg = grounding_config(gene)
    total_mass = sum(float(s.mass_kg or 0.0) for s in g.segments) or 1.0
    load_factor = (total_mass + pk) / total_mass

    changed, proposal, overwritten = [], [], []
    respec: set[str] = set()
    for s in actuated:
        keep_declared = float(declared.get(s.name) or 0.0)
        req0 = s.torque_req_nm if s.torque_req_nm is not None else abs(s.actuator_torque_nm or 8.0)
        before = round(float(req0 or 0.0), 2)
        required = round(before * load_factor, 2)
        # ``auto`` re-specs a joint whose limit is OURS and proposes for one that is the customer's; ``True``
        # re-specs both; ``False`` proposes for both and changes nothing.
        if not ((mode is True) or (mode == "auto" and not keep_declared)):
            entry = {"segment": s.name,
                     "declared_nm": (round(keep_declared, 2) if keep_declared else None),
                     "current_nm": before, "required_nm": required,
                     "shortfall_x": round(required / max(keep_declared or before, 1e-6), 2),
                     "limit_source": ("your model" if keep_declared else "our catalog sizing")}
            # NAME THE PART. A shortfall figure alone is a complaint; the point of refusing to overwrite the
            # customer's limit is to hand them the actuator that WOULD carry the load, sized exactly the way
            # ``ground_gene`` would size it (same margin, same continuous-torque rule), so accepting the
            # proposal cannot produce a different motor than the one quoted here.
            if s.joint_type == "revolute":
                try:
                    from virturoid.services.component_catalog import select_actuator
                    act = select_actuator(required, margin=1.3, continuous_torque_nm=required * 1.3)
                    entry.update({"part": act.name, "part_stall_nm": round(float(act.peak_torque_nm), 1),
                                  "part_mass_kg": round(float(act.mass_kg), 3),
                                  "over_catalog": bool(float(act.peak_torque_nm) + 1e-6 < required)})
                except Exception:  # noqa: BLE001 - an unavailable catalog must not turn a proposal into a crash
                    pass
            proposal.append(entry)
            continue
        if keep_declared:
            overwritten.append({"segment": s.name, "declared_nm": round(keep_declared, 2),
                                "respecified_nm": required})
        # scale the joint's REQUIREMENT (not the last motor's peak) so re-grounding upsizes the motor and stays
        # idempotent under repeated grounding
        s.torque_req_nm = required
        s.actuator_torque_nm = required          # consistent until ground re-selects the peak
        respec.add(s.name)
        changed.append({"segment": s.name, "required_nm": [before, required]})
    # Thicken the load path (sub-linear) for rigidity -- but only where the geometry is OURS to cut, and only
    # on the joints this call actually re-specified.
    girth_applied = bool(girth_scale) and load_factor > 1.05 and not cfg["preserve_geometry"] and bool(respec)
    if girth_applied:
        girth_mult = min(1.4, float(load_factor) ** 0.3)
        for s in actuated:
            if s.name in respec and s.radius_m and s.radius_m > 0:
                s.radius_m = round(max(0.005, s.radius_m * girth_mult), 5)
    required_by_name = {c["segment"]: c["required_nm"][1] for c in changed}

    # THE LOAD ITSELF. A fixed link, mass exactly as asked, sized from a cargo density so it occupies real room
    # instead of being a point mass at the mount. ``_limb_mount`` puts it on the host's measured face in the
    # ROBOT's frame, the same call ``add_limb`` uses, so "on its back" is its back on any body.
    from virturoid.schemas.gene import GeneSegment
    side = max(0.02, (pk / _PAYLOAD_DENSITY_KG_M3) ** (1.0 / 3.0))
    existing = {s.name for s in g.segments}
    p_name = PAYLOAD_LINK if PAYLOAD_LINK not in existing else \
        f"{PAYLOAD_LINK}_{sum(1 for k in existing if k.startswith(PAYLOAD_LINK)) + 1}"
    site = "tip" if (gene.robot_class or "").lower() in _TIP_LOADED_CLASSES else "top"
    off, euler0, placed = _limb_mount(gene, host, site)
    g.segments.append(GeneSegment(
        name=p_name, parent=host.name, shape="box", length_m=round(side, 5), radius_m=round(side / 2.0, 5),
        mass_kg=round(float(pk), 4), joint_type="fixed", mount_offset=off, mount_euler=euler0,
        is_end_effector=False))

    # RECORD WHAT THE ROBOT IS NOW RATED FOR. Nothing downstream could previously tell a heavy robot from a robot
    # BUILT HEAVY ON PURPOSE, because the requested payload was applied to the joints and then forgotten. That is
    # not cosmetic: gene_validation's mass_budget screens total mass against the class band for an UNLOADED
    # machine, so amending a clean 12.8 kg quadruped to carry 15 kg produced a NEW "implausibly heavy" finding and
    # the amend gate auto-reverted the customer's own explicit request. (Boston Dynamics' Spot is 32.5 kg and
    # rated for 14 kg -- the band was calling real hardware implausible.) With the rating on the gene, the band
    # can widen by exactly what was asked for and by nothing else.
    md = dict(getattr(g, "metadata", None) or {})
    md["rated_payload_kg"] = round(float(pk), 3)
    md["payload"] = {"link": p_name, "mass_kg": round(float(pk), 4), "carried_by": host.name, "attach": site}
    g.metadata = md
    mass0 = round(total_mass, 3)
    # Joints this call RE-SPECIFIED have their mass re-derived (a bigger motor really is heavier) even on a body
    # whose masses are otherwise the manufacturer's. Every link the customer did not ask to change -- which now
    # includes every joint whose declared limit we refused to overwrite -- keeps its own number. The payload's
    # mass is the customer's figure and is never derived from the box we drew for it.
    ledger = _reground_and_gate(g, material=_dominant_material(gene), original=gene,
                                respec=respec, keep_mass={p_name})
    # ...and the parts list must not sell the cargo as raw stock. Recorded AFTER grounding, because a re-ground
    # rebuilds the actuator/component buckets around it.
    from virturoid.services.grounded_physics import record_payload_mass
    record_payload_mass(g, p_name, pk)
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
    note = (f"added a {pk:.3f} kg payload as '{p_name}' on {host.name!r} ({side * 1000:.0f} mm box, "
            f"{'at the tool' if site == 'tip' else 'on the chassis'}), taking the robot "
            f"{mass0:.3f} -> {mass1:.3f} kg")
    note += (f"; {len(changed)} joint(s) re-specified for the load" if changed
             else "; no joint was re-specified")
    n_source_kept = sum(1 for p in proposal if p["declared_nm"])
    if n_source_kept:
        note += (f"; {n_source_kept} joint limit(s) are YOUR declared figures and were left untouched -- see "
                 "actuator_proposal (re-send with upsize_actuators: true to re-spec them)")
    elif proposal:
        note += f"; {len(proposal)} joint(s) were left alone (upsize_actuators: false) -- see actuator_proposal"
    out = {"op": "set_payload", "payload_kg": pk, "load_factor": round(load_factor, 3),
           "payload_link": p_name, "payload_mass_kg": round(float(pk), 4), "carried_by": host.name,
           "payload_box_m": round(side, 4), "placement": placed,
           "n_joints_upsized": len(changed), "changed": changed[:8], "total_mass_kg": [mass0, mass1],
           "mass": ledger,
           "source_torque_preserved": bool(n_source_kept), "n_source_torque_preserved": n_source_kept,
           "girth_scaled": girth_applied, "structural": True, "note": note}
    if not girth_applied and girth_scale and cfg["preserve_geometry"]:
        out["girth_scale_skipped"] = ("this body's link geometry is your own model's; thickening the load path "
                                      "would re-cut it. Nothing was resized.")
    if proposal:
        out["actuator_proposal"] = proposal[:12]
        out["proposal_note"] = (
            (f"{n_source_kept} joint(s) carry limits read from YOUR model and marked authoritative, so they were "
             "not rewritten. " if n_source_kept else "No joint was re-specified (upsize_actuators: false). ")
            + "The figures above are what this payload would require and are a PROPOSAL about new parts, not a "
              "change to your robot. Apply them with upsize_actuators: true.")
    if overwritten:
        out["source_torque_rewritten"] = overwritten[:12]
        out["warning"] = (f"upsize_actuators: true re-specified {len(overwritten)} joint limit(s) that came from "
                          "YOUR model — they are no longer your measured figures (edit_robot op:'undo' restores them)")
    if undersized:
        out["undersized_joints"] = undersized[:8]
        out["warning"] = (f"{len(undersized)} joint(s) exceed the strongest catalog motor for this payload -- add a "
                          "gearbox / reduction stage or split the load; the design is over the actuator envelope here")
    return g, out


def _geometry_signature(gene) -> tuple:
    """Everything about a body an operator could move without changing its part COUNT or its NAMES.

    Used to check the claim "nothing was changed" instead of inferring it from one metadata flag. Names alone
    are not enough: ``anatomy_compiler._splay_before_substituting`` rotates each leg's proximal MOUNT and
    "preserves every authored segment, its geometry, its proportions and its part count" -- a real change to the
    robot the customer gets back, invisible to a name-list comparison.
    """
    def _vec(v):                                             # a vector field may be list/tuple/ndarray/None --
        try:                                                 # `v or ()` would raise on an ndarray, so never do that
            return tuple(float(x) for x in v) if v is not None else ()
        except TypeError:                                    # a scalar (or anything unindexable) is its own value
            return (v,)

    return tuple((getattr(s, "name", None), getattr(s, "parent", None), getattr(s, "shape", None),
                  getattr(s, "length_m", None), getattr(s, "radius_m", None),
                  _vec(getattr(s, "mount_offset", None)), _vec(getattr(s, "mount_euler", None)),
                  getattr(s, "joint_type", None), _vec(getattr(s, "joint_axis", None)),
                  getattr(s, "joint_lower", None), getattr(s, "joint_upper", None),
                  getattr(s, "mass_kg", None))
                 for s in getattr(gene, "segments", ()) or ())


# op name -> callable(gene, **args). The typed operator library the intent-classifier maps requests onto.
def adopt_walkable_template(gene, **_):
    """B2: EXPLICITLY replace an imported/composed quadruped's body with a size-matched walkable fanned template.
    Only ever invoked by the customer (never automatically) -- ingest KEEPS the original geometry and merely
    OFFERS this. Lands as one undo step, so 'undo' restores the customer's original body exactly.

    It is the ONLY op that can hand back a robot with none of the customer's own links on it, and until the
    sweep behind this line it was also the only one whose diff carried no ``mass`` block at all -- measured on a
    real Go2, it returned 20 template segments weighing 9.793 kg in place of 13 links weighing 15.207, and said
    so nowhere in numbers. The ledger is the same one every other operator reports, so "what did I get back"
    has one answer everywhere; ``n_existing_links_dropped`` is what makes a wholesale swap visible in it.

    WHAT IT ACTUALLY DOES WHEN IT FIRES, measured 2026-08-12 through this operator (nothing anywhere proved
    this before -- see tests/test_import_verify_honesty.py): on a real Menagerie ``unitree_go2`` it takes the
    body from 0.000 m / CROUCH to 1.998 m / CREDIBLE WALK, and on the fixed-base fixture quad in that test file
    from 0.000 m / SLIDE to 1.356 m / CREDIBLE WALK. It works -- at the price of every one of the customer's
    links, which is what the mass ledger is for.

    A REFUSAL IS AN ANSWER AND MUST NAME ITSELF (2026-08-12). All of the above is the ADOPT case; the DECLINE
    case said ONE sentence for eight different situations -- "the original body already walks or no better
    template was found -- unchanged". MEASURED through this operator on three real bodies: a Menagerie
    ``unitree_g1`` (30 links, 33.341 kg, 0.000 m, CROUCH), the already-walking ingest fixture quad (1.604 m,
    CREDIBLE WALK), and a composed 6-axis arm all got that identical string. For the humanoid its first
    disjunct is FALSE, and its second implies we measured a template against their body when we never left the
    robot-class check; for the arm it is meaningless. Meanwhile ``input_training_tools._ingest_project`` tells
    the SAME humanoid the true reason, in the same session -- two surfaces answering one question in two
    framings is the #215/#218 shape. ``ensure_walkable_quad`` now reports WHICH unchanged-return fired, so the
    sentence a customer reads is the branch that actually ran, and ``declined.reason`` carries it
    machine-readably the way every other refusal in the amend surface does."""
    from virturoid.services.anatomy_compiler import ensure_walkable_quad
    from virturoid.services.gene_build import grounding_config
    before = len(gene.segments)
    before_mass = {s.name: float(s.mass_kg or 0.0) for s in gene.segments}
    before_geom = _geometry_signature(gene)                  # captured BEFORE the call: some paths return a
    why: dict = {}                                           # modified body, some mutate this one's metadata
    new = ensure_walkable_quad(gene, "adopt walkable template", force=True, decline=why)
    applied = bool(dict(getattr(new, "metadata", None) or {}).get("walkability_fallback", {}).get("applied"))
    mass = _mass_ledger(before_mass, new,
                        added={s.name for s in new.segments} - set(before_mass),
                        preserved=bool(grounding_config(gene)["preserve_mass"]))
    out = {"op": "adopt_walkable_template", "applied": applied, "segments_before": before,
           "segments_after": len(new.segments), "mass": mass,
           "note": (f"adopted a size-matched walkable template: {mass['n_existing_links_dropped']} of your "
                    f"link(s) were replaced and the robot went {mass['total_mass_kg'][0]:.3f} -> "
                    f"{mass['total_mass_kg'][1]:.3f} kg (undo restores your original body)" if applied
                    else "")}
    if not applied:
        # "NOTHING WAS CHANGED" IS A CLAIM, SO COMPUTE IT. ``applied`` reads ONE metadata key
        # (``walkability_fallback``), and ``ensure_walkable_quad`` has a path -- ``_splay_before_substituting``
        # -- that keeps every authored part but WIDENS THE STANCE and returns that, setting a different key. A
        # note that said "unchanged" off ``applied`` alone would be false there. (Measured: an ingested body
        # cannot reach that path at all -- imports carry no ``metadata['grounding']``, which the splay is gated
        # on -- but a grounded composed body can, so this is computed rather than assumed.)
        untouched = before_geom == _geometry_signature(new)
        out["geometry_unchanged"] = untouched
        out["note"] = ("nothing was changed -- your body was kept as-is" if untouched else
                       "no template was substituted, but your body's own geometry WAS adjusted in place "
                       "(see the mass ledger and design_delta; edit_robot op:'undo' restores it)")
        # NEVER INVENT A REASON. If the compiler named none, say exactly that instead of reaching for a
        # plausible-sounding catch-all -- reaching for one is how the wrong claim shipped in the first place.
        out["declined"] = dict(why) if why.get("reason") else {
            "reason": "unreported",
            "detail": "the walkability check returned your body without naming a reason -- no template was "
                      "substituted"}
        out["note"] += f" -- {out['declined'].get('detail') or out['declined'].get('reason')}"
    return new, out


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

#: Finding checks an op is EXPECTED to move, because moving them IS the request. A gate that reverts these is
#: not protecting the design, it is refusing the instruction.
#:
#: ``add_limb``/``symmetry`` is the one that cost a customer their afternoon. ``anatomy_critic`` flags a legged
#: body whose y-centroid is more than 40 mm off centre, which is exactly and unavoidably what mounting ONE arm
#: on a quadruped does -- so "put an arm on my Go2", the single most-requested amend on the product, was
#: auto-reverted by a med-severity note describing the thing that had been asked for. The finding is still
#: MEASURED and still REPORTED (see :func:`explain_findings`); it just no longer votes to revert.
#: ``set_payload``/``part_balance`` is the same shape and cost the front door its headline request. The payload
#: lands as a real link (that IS the operator), and ``anatomy_critic`` flags any non-root part over 55% of the
#: body volume as "one part dominates the silhouette; shrink it relative to the body" -- advice that is
#: meaningless for cargo, since the load's size is the customer's number and not a proportion to tune. Measured
#: through ``call_tool`` on a grounded 3.356 kg tabletop arm, ``set_payload`` applied at 0.5/1/2/3 kg and was
#: REVERTED at 5, 10 and 25 kg, held mass unchanged at 3.356 -- so on a small body the operator silently did
#: nothing for exactly the requests that motivated it. (The real Menagerie Go2 at 15.206 kg clears the
#: threshold and was unaffected, which is why the sweep on the imported robot never saw it.) The finding is
#: still MEASURED and still REPORTED as EXPECTED in ``explain_findings``; it no longer votes to revert. A
#: payload the actuators genuinely cannot deliver is refused elsewhere, by name and margin, in
#: ``set_payload``'s own ``actuator_proposal`` / ``undersized_joints``.
_EXPECTED_FINDINGS = {"add_limb": {"symmetry"}, "set_payload": {"part_balance"}}


def expected_findings(ops) -> set[str]:
    """Finding checks the given ``[{op, args}]`` sequence is allowed to introduce without being reverted.

    A BATCH INTERSECTS; IT DOES NOT UNION, and the difference is a real hole rather than a nicety. Until
    2026-08-13 this OR-ed the sets, so ``[set_payload, scale_group]`` exempted ``part_balance`` outright -- and
    a ``part_balance`` that ``scale_group`` introduced (a leg scaled until it dominates the silhouette, which is
    a genuine design finding) rode through on the payload's exemption. An exemption says "THIS op is allowed to
    cause THIS finding"; nothing here can attribute a finding to one op inside a batch, so the only honest
    reading of a mixed batch is the one every op agrees on.

    The direction is chosen deliberately. Over-exempting lets a real defect apply SILENTLY; under-exempting
    reverts a legitimate edit while naming the finding, the threshold and ``gate_non_regression: false``. The
    second is recoverable in one call and the first is not, which is the same trade the rest of this module
    makes. A single-op call is unaffected: the intersection over one set is that set.
    """
    specs = [s for s in (ops or []) if s]
    if not specs:
        return set()
    sets = [_EXPECTED_FINDINGS.get(str(s.get("op") or ""), set()) for s in specs]
    return set.intersection(*sets) if len(sets) > 1 else set(sets[0])


def design_findings(gene) -> list[dict]:
    """Every deterministic design finding on this body, NAMED: ``[{check, severity, detail, part}]``.

    The non-regression gate used to compare two INTEGERS -- ``(high_or_fatal, weighted_findings)`` -- and print
    them. Measured, that produced ``before {high_or_fatal: 0, weighted_findings: 0} / after {0, 2}`` on a
    perfectly reasonable ``add_limb``: zero fatal findings on either side, the #1 use case blocked, and not one
    word about WHICH check moved, on WHICH part, or what to do instead. The finding it would not name was a
    single ``med``. This returns the findings themselves so a refusal can quote them.

    Sources are the same three the gate already scored, so the explanation cannot drift from the decision.
    """
    out: list[dict] = []
    try:
        from virturoid.services.gene_validation import validate_gene_design
        for f in validate_gene_design(gene)["risk_flags"]:
            out.append({"check": str(f.get("check")), "severity": str(f.get("severity")),
                        "detail": str(f.get("detail") or ""), "source": "gene_validation"})
    except Exception:  # noqa: BLE001 - an unavailable verifier contributes no finding, never invented evidence
        pass
    try:
        from virturoid.services.anatomy_critic import critique_gene
        for f in critique_gene(gene)["issues"]:
            out.append({"check": str(f.get("check")), "severity": str(f.get("severity")),
                        "detail": str(f.get("detail") or ""), "source": "anatomy_critic"})
    except Exception:  # noqa: BLE001
        pass
    try:
        from virturoid.services.visual_physics_gate import audit_gene
        for i in audit_gene(gene).issues:
            out.append({"check": str(i.code), "severity": "high", "detail": str(i.detail or ""),
                        "part": str(i.geom or "") or None, "source": "visual_physics_gate"})
    except Exception:  # noqa: BLE001
        pass
    return out


_FINDING_WEIGHT = {"fatal": 8, "high": 5, "med": 2, "low": 1}


def findings_score(findings, *, ignore=()) -> tuple[int, int]:
    """``(high_or_fatal, weighted_total)`` over findings, skipping checks in ``ignore``. The gate's ordering."""
    keep = [f for f in findings if f.get("check") not in set(ignore)]
    return (sum(f.get("severity") in ("fatal", "high") for f in keep),
            sum(_FINDING_WEIGHT.get(f.get("severity"), 0) for f in keep))


def explain_findings(before, after, *, ops=None) -> dict:
    """Which findings this edit INTRODUCED, and what a caller can actually do about it.

    Returns ``{new, expected, worse, score_before, score_after, message}``. ``message`` is prose an engineer can
    act on: it names each new finding, its severity, its part, whether it was the point of the request, and the
    three ways forward (accept it with ``gate_non_regression: false``, change the op, or fix the finding).
    """
    ignore = expected_findings(ops)
    fb, fa = design_findings(before), design_findings(after)
    seen = list(fb)
    new: list[dict] = []
    for f in fa:                                   # multiset difference: two identical findings are two findings
        match = next((x for x in seen if x.get("check") == f.get("check")
                      and x.get("detail") == f.get("detail")), None)
        if match is None:
            new.append(f)
        else:
            seen.remove(match)
    blocking = [f for f in new if f.get("check") not in ignore]
    intended = [f for f in new if f.get("check") in ignore]
    lines = []
    for f in blocking:
        part = f" on '{f['part']}'" if f.get("part") else ""
        lines.append(f"  - [{f.get('severity')}] {f.get('check')}{part}: {f.get('detail')}")
    for f in intended:
        lines.append(f"  - [{f.get('severity')}] {f.get('check')}: {f.get('detail')} "
                     f"(EXPECTED for this edit — not counted against it)")
    return {"new": new, "expected_checks": sorted(ignore), "blocking": blocking,
            "score_before": list(findings_score(fb, ignore=ignore)),
            "score_after": list(findings_score(fa, ignore=ignore)),
            "message": "\n".join(lines)}


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
                                    "name": "<prefix, e.g. arm3 / tail / mast>",
                                    "rest_angles": "[rad per actuated link] — the resting posture; omit for a "
                                                   "straight chain (a mast), give e.g. [0, -0.6, 1.2] for an arm "
                                                   "with a visible shoulder and elbow"},
         "for": "STRUCTURAL: GROW a new articulated chain on the existing body — 'add a third arm', 'add a "
                "tail', 'put a sensor mast on the back'. Keeps the robot; only adds. `attach` is a face of the "
                "robot (top = its back, front = its nose), not of the parent link's own axis; the chain grows "
                "that way and the diff reports where it landed plus what it added to the robot's mass."},
        {"op": "set_payload", "args": {"payload_kg": "0.1-50.0", "girth_scale": "true|false",
                                       "upsize_actuators": "auto|true|false"},
         "for": "make it CARRY/LIFT a load: the payload is ADDED to the robot as a real link (on the chassis, or "
                "at the tool on an arm) and every joint's torque requirement rises with it. `upsize_actuators` "
                "'auto' (default) re-specs the joints whose limits are OURS and PROPOSES parts for the ones read "
                "from your own model; true re-specs those too (and says which); false proposes only. The diff "
                "reports payload_mass_kg, what it changed, and what it deliberately did not."},
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
