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

import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from virturoid.schemas.gene import RobotGene, GeneSegment


def _slug_name(value: str) -> str:
    """Filesystem-safe stem for a link/robot name (customer link names are arbitrary)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "part")).strip("_") or "part"

# MuJoCo joint-type ints -> our gene joint types (mjtJoint: FREE=0, BALL=1, SLIDE=2, HINGE=3).
_JNT = {3: "revolute", 2: "prismatic"}
# MuJoCo geom-type ints -> shapes (mjtGeom: PLANE=0, SPHERE=2, CAPSULE=3, ELLIPSOID=4, CYLINDER=5, BOX=6).
_GEOM = {2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}
# Body-name hints are unambiguous substrings (NOT bare "ee" — that false-matches wheel/knee/feet/steel).
_EE_NAME_HINTS = ("gripper", "hand", "finger", "effector", "tcp", "grasp")
# Site names are deliberate (ee_site, tcp, tool0), so a short "ee" hint is safe there.
_EE_SITE_HINTS = ("ee", "tcp", "tool", "grasp", "effector", "tip")


def _quat_to_euler_xyz(quat) -> tuple[float, float, float]:
    """MuJoCo body quaternion (w, x, y, z) -> the intrinsic XYZ euler MuJoCo's ``euler=`` attribute expects
    (its default ``eulerseq`` is "xyz"), i.e. R = Rx(a) @ Ry(b) @ Rz(c)."""
    import math

    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    r02 = 2.0 * (x * z + w * y)
    r12 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    b = math.asin(max(-1.0, min(1.0, r02)))
    if abs(r02) > 0.999999:                        # gimbal lock: fold the free rotation into a
        return (math.atan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)), b, 0.0)
    return (math.atan2(-r12, r22), b, math.atan2(-r01, r00))


def _quat_mat(quat):
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    import numpy as np

    w, x, y, z = (float(v) for v in quat)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _mat_to_euler_xyz(R) -> tuple[float, float, float]:
    """3x3 -> the intrinsic XYZ euler MuJoCo's ``euler=`` attribute expects (R = Rx(a) @ Ry(b) @ Rz(c))."""
    import math

    b = math.asin(max(-1.0, min(1.0, float(R[0, 2]))))
    if abs(float(R[0, 2])) > 0.999999:                       # gimbal lock
        return (math.atan2(float(R[2, 1]), float(R[1, 1])), b, 0.0)
    return (math.atan2(-float(R[1, 2]), float(R[2, 2])), b, math.atan2(-float(R[0, 1]), float(R[0, 0])))


def _rot_z_to(v):
    """Smallest rotation taking +z onto ``v`` (Rodrigues). Identity when v is already +z."""
    import numpy as np

    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.eye(3)
    u = v / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, u))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:                                        # antiparallel: flip 180 deg about x
        return np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    a = np.cross(z, u)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def _link_vector(mj, body_id: int, children: dict):
    """The link's own AXIS in its body frame: origin -> next joint (or, for a leaf, -> its farthest geom).

    A RobotGene segment spans [0, length] along its local +z, but a real robot's links point wherever the design
    puts them -- a Go2's thigh and calf both run DOWN (-z) from their joints. Importing without this, every link
    was extruded along +z and came out MIRRORED ABOUT ITS OWN JOINT: joints in the right places, link bodies on
    the wrong side of them, which rendered an imported Go2 as a bundle of pipes standing through its own chassis.
    """
    import numpy as np

    kids = children.get(body_id, [])
    if kids:
        best = max(kids, key=lambda c: float(np.linalg.norm(mj.body_pos[c])))
        return np.asarray(mj.body_pos[best], dtype=float)
    far, best_n = np.zeros(3), 0.0
    for g in range(mj.ngeom):
        if int(mj.geom_bodyid[g]) != body_id:
            continue
        sz = mj.geom_size[g]
        ext = float(sz[2] if int(mj.geom_type[g]) in (6, 7) else sz[max(0, len(sz) - 2)])
        p = np.asarray(mj.geom_pos[g], dtype=float)
        nrm = float(np.linalg.norm(p))
        reach = nrm + abs(ext)
        if reach > best_n:
            best_n, far = reach, (p * (reach / nrm) if nrm > 1e-9 else np.array([0.0, 0.0, reach]))
    return far


def _bake_source_mesh(mj, body_id: int, S, out_path) -> bool:
    """Write THIS body's own mesh geoms, transformed into our LINK frame, as one binary STL (millimetres).

    The importer builds two lanes: a FAITHFUL one that keeps the customer's MJCF + meshes, and an EDITABLE
    RobotGene the amend operators work on. Measured, the editable lane carried the customer's geometry on
    0 of 13 segments -- every link was rebuilt as a plain capsule -- so the robot you could edit and the robot
    you saw rendered was a stick figure of the one you imported, no matter how exact the kinematics became.
    Their real meshes were sitting in the faithful lane, unused. This baker is the bridge: each editable
    segment gets the source geometry of the link it stands for, expressed in that segment's own frame.
    """
    import struct

    import numpy as np

    tris = []
    for g in range(mj.ngeom):
        if int(mj.geom_bodyid[g]) != body_id or int(mj.geom_type[g]) != 7:      # 7 = mjGEOM_MESH
            continue
        mid = int(mj.geom_dataid[g])
        if mid < 0:
            continue
        v0, vn = int(mj.mesh_vertadr[mid]), int(mj.mesh_vertnum[mid])
        f0, fn = int(mj.mesh_faceadr[mid]), int(mj.mesh_facenum[mid])
        verts = np.asarray(mj.mesh_vert, dtype=float).reshape(-1, 3)[v0:v0 + vn]
        faces = np.asarray(mj.mesh_face, dtype=int).reshape(-1, 3)[f0:f0 + fn]
        R = _quat_mat(mj.geom_quat[g])
        p = np.asarray(mj.geom_pos[g], dtype=float)
        world = (verts @ R.T + p) @ S                       # geom -> body -> LINK frame (S maps link->body)
        tris.append(world[faces] * 1000.0)                  # STL is millimetres, matching the compiler
    if not tris:
        return False
    T = np.concatenate(tris, axis=0)
    try:
        with open(out_path, "wb") as fh:
            fh.write(b"\0" * 80)
            fh.write(struct.pack("<I", len(T)))
            for t in T:
                n = np.cross(t[1] - t[0], t[2] - t[0])
                ln = float(np.linalg.norm(n))
                n = n / ln if ln > 1e-12 else np.zeros(3)
                fh.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))
    except OSError:
        return False
    return True


_REST_KEY_PREFERENCE = ("home", "stand", "standing", "default", "retract", "init")


def _source_rest_pose(mj, name_of, BODY) -> tuple[dict, str | None]:
    """The source model's intended stance as ``{our_joint_name: angle}``, plus which keyframe it came from.

    Prefers a conventionally-named key (`home`, `stand`, ...) and otherwise takes the first one. Only hinge and
    slide joints are carried -- a free/ball base is not representable in a RobotGene chain and its height is
    re-derived by ``standing_spawn_z`` anyway. Our joint name is ``{segment}_joint`` and a segment IS the source
    body, so the source joint's own body gives the mapping.
    """
    import mujoco

    kid, kname = _preferred_rest_key(mj, name_of)
    if kid is None:
        return {}, None
    qpos = mj.key_qpos[kid]
    pose: dict[str, float] = {}
    for j in range(mj.njnt):
        if int(mj.jnt_type[j]) not in (2, 3):                    # 2 = slide, 3 = hinge
            continue
        adr = int(mj.jnt_qposadr[j])
        if not (0 <= adr < len(qpos)):
            continue
        bid = int(mj.jnt_bodyid[j])
        # Match the segment builder's own naming, including its fallback for an unnamed body, or the pose keys
        # silently miss those joints.
        body = name_of(BODY, bid) or f"body{bid}"
        val = float(qpos[adr])
        if abs(val) > 1e-6:                                      # a zero angle is already the compiler's default
            pose[f"{body}_joint"] = round(val, 5)
    return pose, (kname or f"key{kid}")


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
    seg_len_by_name: dict[str, float] = {}      # a child's mount z is measured from its parent's TIP
    _rot_by_id: dict[int, object] = {}          # body id -> S_i, the body-frame -> LINK-frame rotation
    try:                                        # per-link STLs of the customer's own meshes (visual only)
        from pathlib import Path as _P
        _mesh_dir = _P("build/_importmesh") / _slug_name(robot_id or os.path.basename(str(source)) or "import")
        _mesh_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 - baking source meshes is a fidelity aid, never an import blocker
        _mesh_dir = None
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
        # JOINT-TO-JOINT DISTANCE IS THE GROUND TRUTH, so prefer it whenever a child exists rather than only
        # when the geom looks like a placeholder. A geom bbox is the visual HULL -- it includes housings,
        # brackets and overhang -- so it systematically over-reads: the Go2's thigh measured 0.373 m from its
        # mesh against a true 0.213 m joint span, and the resulting oversized links drove grounded mass to
        # 39.9 kg for a 15.2 kg robot. The old condition (`span > length_m * 1.8`) meant the exact span was
        # DISCARDED exactly when the mesh was big, i.e. on every real mesh-based robot.
        import numpy as _np
        _v = _link_vector(mj, i, children)
        _span = float(_np.linalg.norm(_v))
        if _span >= 0.02:
            length_m = float(min(1.5, _span))
            shape = "capsule" if shape != "box" or length_m > 2.5 * radius_m else shape
            radius_m = float(min(max(radius_m, 0.015), 0.12, max(0.015, _span * 0.28)))
        # S_i rotates the segment so its local +z runs along the REAL link axis (see _link_vector).
        _S = _rot_z_to(_v) if _span >= 1e-6 else _np.eye(3)
        _rot_by_id[i] = _S

        joint_type, axis, lo, hi, torque = _joint_for_body(mj, i, jnts_of, jnt_force, bname, warnings)
        axis = tuple(float(a) for a in (_S.T @ _np.asarray(axis, dtype=float)))   # axis into the segment frame
        is_leaf = i in leaves
        name_hit = any(h in bname.lower() for h in _EE_NAME_HINTS)
        if is_leaf or name_hit:
            ee_candidates.append(bname)
        # CARRY THE CUSTOMER'S ACTUAL KINEMATICS. body_pos/body_quat are where this link really sits on its
        # parent; without them every child defaults to mount_offset (0,0,0) and the whole robot COLLAPSES ONTO
        # ONE POINT. Measured on the real Unitree Go2: our editable twin had all 13 links at (0,0,0) while the
        # source places FL_hip at (0.193, 0.047, 0), FL_thigh at (0, 0.096, 0) and FL_calf at (0, 0, -0.213).
        # The twin kept their link NAMES and masses but not their geometry, so "amend the customer's robot" was
        # amending a pile of links -- and it verified CROUCH for a robot that walks. Note the span heuristic
        # above already reads body_pos to infer a LENGTH; it just never kept the position itself.
        #
        # Two frames differ, and BOTH have to be converted or the twin is wrong in a different way each time:
        #   * ORIGIN vs TIP  - MuJoCo's body_pos is relative to the parent's ORIGIN; our mount_offset is
        #     relative to the parent's TIP (the compiler places a child at (mo.x, mo.y, parent.length + mo.z)).
        #   * BODY vs LINK   - our segment's local +z IS the link axis, while a MuJoCo body's frame is whatever
        #     the author chose. So everything expressed in the parent's body frame must be rotated into the
        #     parent's LINK frame by S_p, and this segment's own rotation composes with its own S_i.
        # Skipping the second conversion left joints correctly placed but every link body mirrored onto the
        # wrong side of its joint (a Go2 rendered as pipes standing up through its own chassis).
        if parent:
            _Sp = _rot_by_id.get(int(mj.body_parentid[i]), _np.eye(3))
            _p_len = float(seg_len_by_name.get(parent, 0.0))
            _local = _Sp.T @ _np.asarray(mj.body_pos[i], dtype=float)      # parent body frame -> parent link frame
            mount_offset = (float(_local[0]), float(_local[1]), float(_local[2]) - _p_len)
            mount_euler = _mat_to_euler_xyz(_Sp.T @ _quat_mat(mj.body_quat[i]) @ _S)
        else:
            mount_offset, mount_euler = (0.0, 0.0, 0.0), _mat_to_euler_xyz(_S)
        seg_len_by_name[bname] = length_m
        # Keep the customer's OWN geometry on the editable segment (see _bake_source_mesh). Visual only: the
        # collider stays the primitive derived above, so physics/MJX behaviour is unchanged and every existing
        # amend operator keeps working on numbers it understands.
        _geo = None
        if _mesh_dir is not None:
            _fp = _mesh_dir / f"{_slug_name(bname)}.stl"
            if _bake_source_mesh(mj, i, _S, _fp):
                _geo = {"family": "source_mesh", "path": str(_fp.resolve()).replace("\\", "/"),
                        "provenance": "customer_import"}
        segments.append(GeneSegment(
            name=bname, parent=parent, shape=shape, length_m=length_m, radius_m=radius_m, mass_kg=mass,
            joint_type=joint_type, joint_axis=axis, joint_lower=lo, joint_upper=hi,
            mount_offset=mount_offset, mount_euler=mount_euler, geometry=_geo,
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

    robot_class = _infer_class(segments, roots, mj, name_of)
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
    # CARRY THE SOURCE'S OWN STANDING POSE. A robot description ships the stance its designers intended as a
    # named keyframe -- 45 of the 74 MuJoCo Menagerie models do (`home`, `stand`, `standing`, `retract`) -- and
    # for a legged robot that pose is the whole point: a Go2's home key is base z=0.27 with every leg at
    # (0, 0.9, -1.8), i.e. HIP FORWARD AND KNEE FOLDED BACK. We ignored it and spawned every imported robot at
    # qpos 0, which for a quadruped means STRAIGHT LEGS: measured, that read 0.601 m tall against a real 0.394,
    # and it is why an imported Go2 verified CROUCH/FELL. Our own composer has always baked a bent-knee rest
    # pose (morphology_composer: thigh -0.55, calf +1.10) and gene_compiler emits it as a keyframe -- only the
    # IMPORT path never populated it, so customer robots were held to a stance no real robot uses.
    #
    # Angles transfer 1:1 despite the link-frame rotation: S is a rotation, and the joint axis was carried into
    # the segment frame by S^T, so a rotation of theta about (S^T a) equals a rotation of theta about a.
    try:
        _pose, _key = _source_rest_pose(mj, name_of, BODY)
        if _pose:
            gene.metadata = {**(gene.metadata or {}), "rest_pose": _pose,
                             "rest_pose_source": f"source keyframe {_key!r}"}
    except Exception as _exc:  # noqa: BLE001 - a pose is a fidelity aid, never an import blocker
        # But it must not be a SILENT aid: this clause once swallowed a plain NameError introduced by a refactor,
        # and every model quietly lost its stance. The breadth sweep spotted it first (Cassie's `posed` count went
        # 15 -> 0) simply because it runs every model, and a dropped pose is indistinguishable from a model that
        # ships no keyframe unless you are watching a robot known to have one. A warning makes it visible in the
        # place the customer actually looks.
        warnings.append(f"could not read the source's rest keyframe ({type(_exc).__name__}: {_exc}); the body "
                        f"will spawn at its zero pose, which for a legged robot means straight legs.")
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
    if gtype == 7:        # MESH: geom_size carries the bbox half-extents, same convention as a box.
        # This used to fall through to the generic branch below, which reads size[0] -- the X half-extent -- as
        # the LENGTH. Real limb meshes are long in Z and thin in X, so every mesh link imported as its own
        # thickness: a Unitree Go2 calf (half-extents 0.019 x 0.029 x 0.158) came in at 2*0.0188 = 37.6 mm
        # against a true 213 mm, 5.7x too short. That is what left visible gaps between hip/thigh/calf in the
        # render and made an imported Go2 verify CROUCH. Mesh links are the norm in any real robot description.
        return "capsule", max(2 * float(size[2]), 1e-3), max(float(max(size[0], size[1])), 1e-3)
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


def _standing_limbs(mj, name_of) -> tuple[int, float]:
    """``(how many limb-chains reach the ground, support-polygon area / height^2)`` in the source's rest stance.

    Counting limb-chains alone cannot tell a leg from an arm or a finger, and that is the whole difficulty: a
    humanoid has FOUR limbs (two legs + two arms) and a four-fingered hand has four too. What separates them is
    that only legs carry the body -- so pose the model in the stance its designers shipped and ask which chains
    touch down. Wheels are excluded for free by the >=2-actuated-joint test a limb already has to pass (a wheel is
    one continuous hinge), which is also why a wheeled base with an arm counts ZERO standing limbs.

    The second return value is the real physics of standing: the horizontal area spanned by the limb TIPS,
    normalised by the body's height so it is scale-free. Measured across the corpus it separates cleanly --
    quadrupeds 1.04-1.79, Tiago's four casters 0.078, and every hand, gripper and arm exactly 0.0 because
    fingertips are collinear. That is what lets a FIXED-BASE body still be recognised as legged, which matters
    because URDF has no floating-base concept: MuJoCo's URDF loader adds no freejoint, so an ordinary URDF
    quadruped presents exactly like a hand does.
    """
    import mujoco
    import numpy as np

    kids = defaultdict(list)
    for b in range(1, mj.nbody):
        kids[int(mj.body_parentid[b])].append(b)
    if not kids.get(0):
        return 0, 0.0

    d = mujoco.MjData(mj)
    kid, _ = _preferred_rest_key(mj, name_of)
    if kid is not None:
        mujoco.mj_resetDataKeyframe(mj, d, kid)
    mujoco.mj_forward(mj, d)

    zlo, zhi, xy = {}, {}, {}
    for g in range(mj.ngeom):
        if int(mj.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c, h = mj.geom_aabb[g][:3], mj.geom_aabb[g][3:]
        p = d.geom_xpos[g] + R @ c
        ctr, ext = float(p[2]), float((np.abs(R) @ h)[2])
        b = int(mj.geom_bodyid[g])
        if b not in zlo or ctr - ext < zlo[b]:
            xy[b] = (float(p[0]), float(p[1]))                   # where this body's LOWEST point sits, in plan
        zlo[b] = min(zlo.get(b, ctr - ext), ctr - ext)
        zhi[b] = max(zhi.get(b, ctr + ext), ctr + ext)
    if not zlo:
        return 0, 0.0
    lo, hi = min(zlo.values()), max(zhi.values())
    height = max(hi - lo, 1e-6)
    band = lo + 0.18 * height                                    # "touching down", tolerant of a bent-knee stance

    def joints_deep(b: int, seen=None) -> int:                   # deepest run of actuated joints below b
        seen = seen or set()
        if b in seen:
            return 0
        seen.add(b)
        n = sum(1 for j in range(mj.njnt) if int(mj.jnt_bodyid[j]) == b and int(mj.jnt_type[j]) in (2, 3))
        return n + max((joints_deep(c, seen) for c in kids.get(b, [])), default=0)

    def carries(b: int) -> bool:
        """Does the subtree at ``b`` hold the robot up? Ground contact, articulation and SIZE together.

        A LEG IS A SUBSTANTIAL FRACTION OF THE MACHINE'S HEIGHT; a caster is not, and that is what separates them
        once ground contact and joint count both pass. Measured: Tiago's four casters are 2-joint chains that DO
        touch the floor, spanning 0.08 of its height, against 0.54-0.68 for the legs of a G1, Go2 or OP3. Joint
        LIMITS looked like the cleaner signal (a wheel turns without end) and separate Tiago perfectly -- but
        ROBOTIS OP3 ships 20 actuated joints and zero ranges, so "limited" is a model-authoring choice, not a
        physical property, and testing it dropped a real humanoid to mobile_base.
        """
        chain = _descendants(kids, b) | {b}
        if joints_deep(b) < 2:                                   # a single-hinge wheel is not a limb
            return False
        if not any(zlo.get(x, hi) <= band for x in chain):
            return False
        solid = [x for x in chain if x in zhi]
        return bool(solid) and (max(zhi[x] for x in solid) - min(zlo[x] for x in solid)) >= 0.25 * height

    def tip_of(b: int) -> tuple[float, float]:
        """Where this limb touches down, in plan view — its lowest ground-band body."""
        chain = _descendants(kids, b) | {b}
        low = [x for x in chain if x in zlo and zlo[x] <= band]
        best = min(low or [x for x in chain if x in zlo] or [b], key=lambda x: zlo.get(x, hi))
        return xy.get(best, (0.0, 0.0))

    def walk(b: int, seen=None) -> list:
        """Descend to where the load-bearing chains actually DIVERGE, then collect one tip per limb.

        Counting the root's direct children is not enough: a humanoid's legs often hang off a pelvis or waist
        rather than off the root, so the root sees ONE ground-reaching branch. Measured -- Booster T1's legs are
        under `Waist` and both ToddlerBots' under `waist_gears`, and all three read as mobile_base until the walk
        followed the single branch down to the hip split.
        """
        seen = seen or set()
        if b in seen:
            return []
        seen.add(b)
        limbs = [c for c in kids.get(b, []) if c not in seen and carries(c)]
        if not limbs:
            return []
        if len(limbs) == 1:
            return walk(limbs[0], seen) or [tip_of(limbs[0])]     # still one chain; keep descending
        out = []
        for c in limbs:                                           # the split: each side is its own limb
            out += walk(c, seen) or [tip_of(c)]
        return out

    # Start at the WORLD, not at a chosen root body. For a normal model that descends through the single robot
    # root and is identical to starting there; but MuJoCo FUSES a static root into the world, so a URDF whose
    # torso has no joint to the world leaves four separate leg roots and no trunk at all. Picking the "biggest"
    # root body then picked ONE LEG and counted one limb. Measured on the 4-leg fixture: 1 limb -> manipulator.
    tips = walk(0)
    if not tips:
        return 0, 0.0
    T = np.asarray(tips, dtype=float)
    area = 0.0
    if len(T) >= 3:                                               # shoelace over the angularly-sorted hull
        ctr = T.mean(axis=0)
        order = np.argsort(np.arctan2(T[:, 1] - ctr[1], T[:, 0] - ctr[0]))
        P = T[order]
        area = 0.5 * abs(float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])))
    return len(T), area / (height ** 2)


def _descendants(kids: dict, b: int) -> set:
    out, stack = set(), list(kids.get(b, []))
    while stack:
        x = stack.pop()
        if x in out:
            continue
        out.add(x)
        stack += kids.get(x, [])
    return out


def _preferred_rest_key(mj, name_of) -> tuple[int | None, str]:
    """``(index, name)`` of the source's intended rest keyframe (`home`, `stand`, ...); ``(None, "")`` if none."""
    import mujoco

    if int(getattr(mj, "nkey", 0)) <= 0:
        return None, ""
    names = [(name_of(mujoco.mjtObj.mjOBJ_KEY, k) or "").lower() for k in range(mj.nkey)]
    kid = next((i for pref in _REST_KEY_PREFERENCE for i, n in enumerate(names) if n == pref), None)
    if kid is None:
        kid = next((i for pref in _REST_KEY_PREFERENCE for i, n in enumerate(names) if pref in n), 0)
    return kid, names[kid]


def _infer_class(segments, roots, mj, name_of=None) -> str:
    """Species/class guess from the morphology, decided by what CARRIES THE BODY rather than by branch count.

    The previous rule counted limb-chains off the root and dispatched on the count, which made three whole
    families unreachable (measured across 63 MuJoCo Menagerie models):

      * ``limbs >= 4 -> quadruped`` caught every HUMANOID, because two legs plus two arms is four limbs, and
        every four-fingered HAND (Allegro, LEAP) for the same reason;
      * ``free_root -> mobile_base`` sat BEFORE the humanoid rule, and every legged robot has a free root, so
        the ``limbs == 2`` humanoid branch could never fire for a floating-base biped.

    Result: 12 of 12 bipeds misclassified (G1, H1, Cassie, Talos, Apollo, Fourier N1, Adam Lite, Berkeley ->
    mobile_base; OP3, Booster T1, ToddlerBot x2 -> quadruped), while the four bodies labelled ``humanoid`` were a
    bimanual station, a bimanual arm and two grippers -- precision 0/4 AND recall 0/12. Not cosmetic: the verify
    rubric is chosen by kind, so an imported humanoid was judged by a wheeled-driving rubric.

    Two corrections. ``free_root`` becomes a PRECONDITION for a locomotor instead of a disqualifier -- every
    hand, gripper and arm in the corpus is fixed-base while every legged and wheeled robot is floating-base. And
    among floating-base bodies the count that decides the class is ``_standing_limbs``: chains that actually
    reach the ground in the source's own stance, so arms and fingers stop voting.
    """
    n_rev = sum(1 for s in segments if s.joint_type == "revolute")
    has_grip = any("grip" in s.name.lower() or "finger" in s.name.lower() for s in segments)
    free_root = any(int(mj.jnt_type[j]) == 0 for j in range(mj.njnt))

    try:
        legs, support = _standing_limbs(mj, name_of) if name_of is not None else (0, 0.0)
    except Exception:  # noqa: BLE001 - a classification guess must never break the import
        legs, support = 0, 0.0
    if free_root:
        if legs >= 6:
            return "hexapod"
        if legs >= 4:
            return "quadruped"
        if legs >= 2:
            return "humanoid"                                   # a biped: the two chains holding it up
        return "mobile_base"                                    # rolls, flies or is carried; nothing walks
    # FIXED BASE, but that is weak evidence on its own: URDF has no floating-base concept, so MuJoCo's URDF
    # loader adds no freejoint and an ordinary URDF quadruped -- or one whose static torso got fused into the
    # world, leaving four separate leg roots -- looks exactly as "fixed" as a bench-mounted hand. Only a real
    # SUPPORT POLYGON promotes it: >=3 limbs whose tips span meaningful ground area. Measured, that threshold has
    # room on both sides (quadrupeds 1.04-1.79, casters 0.078, every hand and gripper exactly 0.0 because
    # fingertips are collinear). Deliberately NOT extended to 2 limbs: a two-fingered gripper would become a
    # biped, and a fixed-base URDF biped is far rarer than that mistake would be common.
    if legs >= 3 and support >= 0.5:
        return "hexapod" if legs >= 6 else "quadruped"
    if n_rev <= 1 and not has_grip:
        return "mobile_base"                                    # a fixed sensor/prop, not an articulated machine
    return "manipulator"                                        # fixed base + articulation = an arm or a hand


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
