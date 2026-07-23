"""Robot import (startup plan §28.2 / roadmap N5.1): URDF or MJCF -> RobotGene + honest warnings.

This is the MVP entry point the platform skipped by *generating* robots from genes: bring an
EXISTING robot in. We use MuJoCo as the parser (it loads both URDF and MJCF), then walk the body
tree into the platform's canonical ``RobotGene`` so an imported robot flows into the whole
pipeline — compile, evaluate, co-design, amend, and the species-tree flywheel.

Principle (§28.2): produce WARNINGS, not silent fixes. Users must be able to trust the model, so we
surface missing mass/inertia, missing joint limits, multiple roots, unsupported joint types, and
MJX-unsafe geoms rather than quietly papering over them. The returned gene is best-effort; its own
``validate()`` issues are folded into the warnings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from virturoid.schemas.gene import RobotGene, GeneSegment

# MuJoCo joint-type ints -> our gene joint types (mjtJoint: FREE=0, BALL=1, SLIDE=2, HINGE=3).
_JNT = {3: "revolute", 2: "prismatic"}
# MuJoCo geom-type ints -> shapes (mjtGeom: PLANE=0, SPHERE=2, CAPSULE=3, ELLIPSOID=4, CYLINDER=5, BOX=6).
_GEOM = {2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}
# Body-name hints are unambiguous substrings (NOT bare "ee" — that false-matches wheel/knee/feet/steel).
_EE_NAME_HINTS = ("gripper", "hand", "finger", "effector", "tcp", "grasp")
# Site names are deliberate (ee_site, tcp, tool0), so a short "ee" hint is safe there.
_EE_SITE_HINTS = ("ee", "tcp", "tool", "grasp", "effector", "tip")


def import_robot(source: str, *, robot_id: str | None = None, species: str | None = None) -> dict:
    """Import a URDF/MJCF (file path or XML string) into a RobotGene.

    Returns ``{"gene", "warnings", "backend_support", "species", "robot_class", "valid"}``.
    Needs MuJoCo. Raises only if the source can't be parsed at all (with a clear message).
    """
    import mujoco

    warnings: list[str] = []

    # M4 (2026-07-24 audit): a xacro TEMPLATE is not a loadable URDF. MuJoCo can't expand ${...}/<xacro:...>, so
    # it used to crash into the generic fallback and hold a PHANTOM "mobile_base" that has nothing to do with the
    # file. Detect it up front and REFUSE with the same expand instructions import_model gives (now also matching
    # the .urdf.xacro suffix), instead of inventing a body.
    try:
        _p = Path(source)
        _text = _p.read_text(encoding="utf-8", errors="replace") if (len(source) < 512 and _p.exists()) else source
    except OSError:
        _text = source
    _is_xacro_suffix = len(source) < 512 and (source.endswith(".xacro") or source.endswith(".urdf.xacro"))
    if "<xacro:" in _text or "${" in _text or _is_xacro_suffix:
        return {
            "gene": None, "backend_support": {}, "species": None, "robot_class": None, "valid": False,
            "warnings": ["this is a xacro template with unexpanded macros (${...} / <xacro:...>), not a loadable "
                         "URDF. Expand it first: `ros2 run xacro xacro robot.urdf.xacro > robot.urdf` (or "
                         "`rosrun xacro xacro`), then import the generated .urdf."],
        }

    mj = _load_model(source, mujoco, warnings)

    name_of = lambda obj, i: (mujoco.mj_id2name(mj, obj, i) or "")  # noqa: E731
    BODY, JNT, GEOM, SITE = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                             mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE)

    # Map each non-world body to a segment. Body 0 is the world; geoms on it (floor/table) are scenery.
    body_name = {i: (name_of(BODY, i) or f"body{i}") for i in range(mj.nbody)}
    children: dict[int, list[int]] = {}
    for i in range(1, mj.nbody):
        children.setdefault(int(mj.body_parentid[i]), []).append(i)
    leaves = {i for i in range(1, mj.nbody) if i not in children}

    # Joints per body (a tree robot has one joint to its parent; flag anything else).
    jnts_of: dict[int, list[int]] = {}
    for j in range(mj.njnt):
        jnts_of.setdefault(int(mj.jnt_bodyid[j]), []).append(j)
    # Actuator force range per actuated joint (for actuator_torque_nm).
    jnt_force: dict[int, float] = {}
    for u in range(mj.nu):
        if int(mj.actuator_trntype[u]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            jid = int(mj.actuator_trnid[u, 0])
            fr = mj.actuator_forcerange[u]
            jnt_force[jid] = max(jnt_force.get(jid, 0.0), abs(float(fr[1])) or abs(float(fr[0])))

    roots = [i for i in range(1, mj.nbody) if int(mj.body_parentid[i]) == 0]
    if len(roots) > 1:
        warnings.append(f"{len(roots)} root bodies attach to the world ({', '.join(body_name[r] for r in roots)}); "
                        "a RobotGene needs a single root. Imported as a multi-root tree — pick a base or split.")

    segments: list[GeneSegment] = []
    ee_candidates: list[str] = []
    for i in range(1, mj.nbody):
        bname = body_name[i]
        parent = None if int(mj.body_parentid[i]) == 0 else body_name[int(mj.body_parentid[i])]
        shape, length_m, radius_m = _primary_geom(mj, i, GEOM, name_of, warnings)
        mass = float(mj.body_mass[i])
        if mass <= 0:
            warnings.append(f"body {bname!r} has zero/negative mass — inertia missing or massless link; using 0.01 kg.")
            mass = 0.01

        # Recover a link's LENGTH from the kinematic span to its child joint. A URDF's joint origins encode link
        # lengths even when the visual/collision MESH is missing (it imported as a tiny placeholder box) — so a
        # limb whose geom underrepresents the span to the next joint is rebuilt as a capsule of that real length.
        child_ids = children.get(i, [])
        if child_ids:
            span = max(float((mj.body_pos[c][0] ** 2 + mj.body_pos[c][1] ** 2 + mj.body_pos[c][2] ** 2) ** 0.5)
                       for c in child_ids)
            placeholder = shape == "box" and length_m <= 0.09 and abs(length_m - radius_m) < 0.03
            if span >= 0.04 and (placeholder or span > length_m * 1.8):
                length_m = float(min(1.5, span))
                shape = "capsule"
                radius_m = float(min(max(radius_m, 0.02), 0.12, max(0.02, span * 0.18)))

        joint_type, axis, lo, hi, torque = _joint_for_body(mj, i, jnts_of, jnt_force, bname, warnings)
        is_leaf = i in leaves
        name_hit = any(h in bname.lower() for h in _EE_NAME_HINTS)
        if is_leaf or name_hit:
            ee_candidates.append(bname)
        segments.append(GeneSegment(
            name=bname, parent=parent, shape=shape, length_m=length_m, radius_m=radius_m, mass_kg=mass,
            joint_type=joint_type, joint_axis=axis, joint_lower=lo, joint_upper=hi,
            actuator_torque_nm=torque, is_end_effector=False))

    # Normalize the base: a RobotGene needs exactly ONE WELDED root. MuJoCo merges a URDF's fixed base
    # link into the world, so the first real body usually becomes a root that still carries its
    # (actuated) joint — invalid as a gene root — and multi-arm robots have several roots. In either
    # case synthesize one welded base segment and reparent the current roots onto it (preserving their
    # joints, e.g. the shoulder pan). Round-trips our own compiled genes too (whose welded root is
    # likewise merged into the world on reload).
    root_segs = [s for s in segments if s.parent is None]
    if len(root_segs) != 1 or any(r.joint_type not in (None, "fixed") for r in root_segs):
        base_name = next((n for n in ("base_link", "imported_base", "base_mount")
                          if not any(s.name == n for s in segments)), "imported_base")
        r0 = root_segs[0] if root_segs else None
        segments.insert(0, GeneSegment(
            name=base_name, parent=None, shape="box", length_m=0.05, radius_m=0.04,
            mass_kg=max((r0.mass_kg if r0 else 0.1), 0.05), joint_type=None, is_end_effector=False))
        for r in root_segs:
            r.parent = base_name
        warnings.append(f"synthesized a welded base segment {base_name!r} as the gene root — the URDF's "
                        "fixed base was merged into the world by the parser (or the robot had multiple "
                        "roots), leaving an actuated/ambiguous root that a RobotGene cannot use directly.")

    # Exactly one end-effector. Priority: a body carrying an ee/tcp/tool SITE (robust, and round-trips
    # our own compiler's ee_site), then a name-hinted body, then any leaf.
    ee_site_body = None
    for si in range(mj.nsite):
        if any(h in name_of(SITE, si).lower() for h in _EE_SITE_HINTS):
            ee_site_body = body_name[int(mj.site_bodyid[si])]
            break
    ee_name = (ee_site_body
               or next((c for c in ee_candidates if any(h in c.lower() for h in _EE_NAME_HINTS)), None)
               or (body_name[max(leaves)] if leaves else (segments[-1].name if segments else None)))
    if ee_name is None:
        warnings.append("no body found to mark as the end-effector.")
    else:
        for s in segments:
            s.is_end_effector = (s.name == ee_name)
        if len(ee_candidates) > 1:
            warnings.append(f"multiple end-effector candidates ({', '.join(sorted(set(ee_candidates)))}); "
                            f"chose {ee_name!r} — set the correct one if wrong.")

    robot_class = _infer_class(segments, roots, mj)
    gene = RobotGene(
        id=robot_id or "imported_robot",
        species=species or f"{robot_class}.imported",
        robot_class=robot_class,
        segments=segments,
        # a robot that MOVES needs a FREE (floating 6-DOF) base, not a welded one -- "floor"/"table" weld the base
        # to the world so the body cannot translate at all (the compiled model has no base joint, and every gait
        # rolls out to 0 forward). A manipulator stays table-mounted.
        base_mount="free" if robot_class in ("mobile_base", "quadruped", "hexapod", "humanoid") else "table",
        end_effector_type="gripper" if any("grip" in (s.name.lower()) for s in segments) else "none",
        metadata={"imported_from": "mjcf_or_urdf", "n_bodies": mj.nbody, "n_joints": mj.njnt},
    )
    gene_issues = gene.validate()
    warnings.extend(f"gene validation: {iss}" for iss in gene_issues)

    backend_support = _backend_support(segments)
    return {
        "gene": gene,
        "warnings": warnings,
        "backend_support": backend_support,
        "species": gene.species,
        "robot_class": robot_class,
        "valid": not gene_issues,
    }


# --------------------------------------------------------------------------- helpers
def _sanitize_urdf_meshes(text: str, base_dir: str | None) -> tuple[str, int]:
    """Neutralize ``<mesh filename=...>`` references whose file can't be found (the #1 enterprise-URDF problem:
    a robot description ships without its meshes, or with them at a different relative path). A missing mesh geom
    is replaced with a modest primitive BOX so MuJoCo still compiles and the KINEMATIC TREE (links + joints +
    inertials) — the part the editable RobotGene needs — is preserved. Returns (new_text, n_replaced)."""
    import os
    import re

    def _present(fn: str) -> bool:
        if os.path.isabs(fn) and os.path.exists(fn):
            return True
        for root in filter(None, (base_dir,)):
            if os.path.exists(os.path.normpath(os.path.join(root, fn))):
                return True
        return False

    n = 0

    def repl(mo):
        nonlocal n
        fn_mo = re.search(r'filename\s*=\s*"([^"]+)"', mo.group(0))
        if fn_mo and not _present(fn_mo.group(1)):
            n += 1
            return '<box size="0.06 0.06 0.06"/>'               # keep the link collidable + sized, not a mesh
        return mo.group(0)

    # both the self-closing <mesh .../> and the paired <mesh ...>...</mesh> forms
    text = re.sub(r'<mesh\b[^>]*?/>', repl, text)
    text = re.sub(r'<mesh\b[^>]*?>.*?</mesh>', repl, text, flags=re.DOTALL)
    return text, n


def _load_model(source: str, mujoco, warnings: list[str]):
    """Load an MjModel from a path or XML string. URDFs load from a file (MuJoCo runs its URDF compiler); a URDF
    that references MISSING meshes is retried with those meshes swapped for primitive geoms so the import still
    succeeds (with a warning) instead of hard-failing on a moved/mesh-less folder."""
    p = Path(source)
    is_file = len(source) < 512 and p.exists()
    base_dir = str(p.parent) if is_file else None
    try:
        text = p.read_text(encoding="utf-8", errors="replace") if is_file else source
    except OSError:
        text = source
    is_urdf = "<robot" in text[:400].lower()

    def _compile(xml_text: str):
        if not is_urdf:
            return mujoco.MjModel.from_xml_string(xml_text)
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8", dir=base_dir) as f:
            f.write(xml_text); tmp = f.name                     # write NEXT TO the real meshes so relative paths resolve
        try:
            return mujoco.MjModel.from_xml_path(tmp)
        finally:
            try:
                Path(tmp).unlink()
            except OSError:
                pass

    # 1) faithful attempt (a real file loads straight from its path so meshes/compiler resolve)
    try:
        return mujoco.MjModel.from_xml_path(str(p)) if is_file else _compile(text)
    except (OSError, ValueError) as exc:
        first_err = str(exc)
    # 2) DETERMINISTIC REPAIR PASS + retry — a real exporter's URDF (the as-published Go2) hits MuJoCo's
    # strict-XML rules (a material re-defined per-visual, many <material> in one <visual>) BEFORE any mesh
    # issue. Share the same repair the faithful lane uses so the GENE twin is built from the customer's REAL
    # structure (FL_hip/FL_thigh names, real topology) instead of crashing here and falling back to a generic
    # description-composed body. Every repair is surfaced as an import warning.
    if is_urdf:
        try:
            from virturoid.services.model_import import repair_urdf_text
            repaired, repairs = repair_urdf_text(text, mesh_root=Path(base_dir) if base_dir else None)
            if repairs:
                for rep in repairs:
                    warnings.append(f"URDF repair ({rep['kind']}): {rep['detail']}")
                try:
                    return _compile(repaired)
                except (OSError, ValueError) as exc_r:
                    first_err = str(exc_r)
                    text = repaired          # carry repairs into the mesh-sanitize retry below
        except Exception:  # noqa: BLE001 - repair is additive; fall through to mesh handling
            pass
    # 3) URDF with missing meshes -> swap them for primitives and retry
    if is_urdf:
        sanitized, n = _sanitize_urdf_meshes(text, base_dir)
        if n:
            warnings.append(f"{n} mesh file(s) referenced by the URDF were missing; imported with primitive geoms "
                            "in their place (kinematic tree + inertials preserved).")
            try:
                return _compile(sanitized)
            except (OSError, ValueError) as exc2:
                first_err = str(exc2)
    raise ValueError(first_err)


def _primary_geom(mj, body_id: int, GEOM, name_of, warnings: list[str]):
    """Pick the body's representative geom -> (shape, length_m, radius_m). Approximate, with warnings."""
    gids = [g for g in range(mj.ngeom) if int(mj.geom_bodyid[g]) == body_id]
    if not gids:
        return "box", 0.05, 0.02   # massless/visual-only link; tiny default (gene needs positive dims)
    # choose the largest geom by bounding size as the body's representative shape
    g = max(gids, key=lambda gi: float(mj.geom_size[gi].sum()))
    gtype = int(mj.geom_type[g]); size = mj.geom_size[g]
    shape = _GEOM.get(gtype, "box")
    if gtype == 6:        # box: size = half-extents -> length along z, radius = x half-extent
        return "box", max(2 * float(size[2]), 1e-3), max(float(size[0]), 1e-3)
    if gtype in (3, 5):   # capsule/cylinder: size = (radius, half-length)
        return shape, max(2 * float(size[1]), 1e-3), max(float(size[0]), 1e-3)
    if gtype == 2:        # sphere: size = (radius,)
        r = max(float(size[0]), 1e-3)
        return "capsule", 2 * r, r
    return "box", max(2 * float(size[0]), 1e-3), max(float(size[0]), 1e-3)


def _joint_for_body(mj, body_id, jnts_of, jnt_force, bname, warnings):
    """Return (joint_type, axis, lower, upper, torque) for the joint attaching this body to its parent."""
    js = jnts_of.get(body_id, [])
    if not js:
        return None, (0.0, 0.0, 1.0), None, None, None   # welded (fixed) link
    if len(js) > 1:
        warnings.append(f"body {bname!r} has {len(js)} joints; a gene segment models one. Using the first.")
    j = js[0]
    jt_int = int(mj.jnt_type[j])
    jt = _JNT.get(jt_int)
    if jt is None:
        warnings.append(f"body {bname!r} uses an unsupported joint type ({'free' if jt_int == 0 else 'ball'}); "
                        "modeled as a fixed weld. Free/ball joints aren't representable in a RobotGene chain.")
        return None, (0.0, 0.0, 1.0), None, None, None
    axis = tuple(round(float(v), 4) for v in mj.jnt_axis[j])
    lo = hi = None
    if int(mj.jnt_limited[j]):
        lo, hi = float(mj.jnt_range[j][0]), float(mj.jnt_range[j][1])
    else:
        warnings.append(f"joint on {bname!r} has no limits (continuous); set joint_lower/upper for a bounded design.")
    torque = round(jnt_force.get(j, 0.0), 3) or None
    return jt, axis, lo, hi, torque


def _infer_class(segments, roots, mj) -> str:
    """Species/class guess from the morphology. Counts LIMB-CHAINS hanging off the root (a child that begins a
    chain of >=2 actuated segments = a leg/arm) so a symmetric multi-legged body is recognized as legged instead
    of defaulting to 'manipulator' — which is what let an imported quadruped URDF read as an arm."""
    from collections import defaultdict

    n_rev = sum(1 for s in segments if s.joint_type == "revolute")
    has_grip = any("grip" in s.name.lower() or "finger" in s.name.lower() for s in segments)
    free_root = any(int(mj.jnt_type[j]) == 0 for j in range(mj.njnt))

    by_name = {s.name: s for s in segments}
    children = defaultdict(list)
    for s in segments:
        if s.parent:
            children[s.parent].append(s.name)

    def actuated_depth(name: str, seen=None) -> int:
        seen = seen or set()
        if name in seen or name not in by_name:
            return 0
        seen.add(name)
        s = by_name[name]
        deepest = max((actuated_depth(c, seen) for c in children.get(name, [])), default=0)
        return (1 if s.joint_type in ("revolute", "prismatic") else 0) + deepest

    root_names = [s.name for s in segments if not s.parent] or [segments[0].name] if segments else []
    limbs = sum(1 for rn in root_names for c in children.get(rn, []) if actuated_depth(c) >= 2)

    if limbs >= 6:
        return "hexapod"
    if limbs >= 4:
        return "quadruped"
    if free_root:
        return "mobile_base"
    if limbs == 2 and n_rev >= 6:
        return "humanoid"                                       # a bipedal / two-limb body with many joints
    if n_rev <= 1 and not has_grip:
        return "mobile_base"
    return "manipulator"


def _backend_support(segments) -> dict:
    """Per-backend compatibility (startup plan §28.3 support matrix)."""
    mjx_warns = []
    if any(s.shape in ("cylinder", "ellipsoid") for s in segments):
        mjx_warns.append("cylinder/ellipsoid colliders are unsupported against box/mesh in MJX-JAX; "
                         "emit them visual-only with a capsule/box collider, or use the MJX-Warp backend.")
    return {
        "mujoco": {"status": "supported", "warnings": []},
        "mjx": {"status": "supported_with_warnings" if mjx_warns else "supported", "warnings": mjx_warns},
        "isaac": {"status": "not_yet_supported", "warnings": ["isaac adapter not implemented"]},
        "gazebo": {"status": "not_yet_supported", "warnings": ["sdf export not implemented"]},
    }
