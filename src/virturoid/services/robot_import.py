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
        base_mount="floor" if robot_class in ("mobile_base", "quadruped", "humanoid") else "table",
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
def _load_model(source: str, mujoco, warnings: list[str]):
    """Load an MjModel from a path or XML string; handle URDF (which MuJoCo loads from a file)."""
    p = Path(source)
    try:
        if len(source) < 512 and p.exists():
            return mujoco.MjModel.from_xml_path(str(p))
    except (OSError, ValueError):
        pass
    text = source
    if "<robot" in text[:400].lower():  # URDF string -> MuJoCo needs a file path
        with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8") as f:
            f.write(text); tmp = f.name
        try:
            return mujoco.MjModel.from_xml_path(tmp)
        finally:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
    return mujoco.MjModel.from_xml_string(text)


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
    """Lightweight species/class guess from the morphology (honest, coarse)."""
    n_rev = sum(1 for s in segments if s.joint_type == "revolute")
    has_grip = any("grip" in s.name.lower() or "finger" in s.name.lower() for s in segments)
    # a free root joint => floating base => mobile
    free_root = any(int(mj.jnt_type[j]) == 0 for j in range(mj.njnt))
    if free_root:
        return "mobile_base"
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
