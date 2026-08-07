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
from virturoid.services.body_kind import FLOATING_BASE_CLASSES, family_from_legs


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


def _root_own_span(mj, body_id: int):
    """The root body's own longest COLLISION extent — how long the trunk actually is.

    Not from its children: a torso's children are a symmetric pair of shoulders or four hips, and reading a shape
    off where things are attached to it is what rebuilt a humanoid chest sideways. Not from visual geometry
    either: the Go2's base carries five non-colliding shrouds, and measuring those with the collision box returns
    0.530 m for a 0.376 m structure. The MAIN structural geom, unioned with the other collision geoms only where
    they extend it along the same axis, keeps a nose or an antenna from becoming length.
    """
    import numpy as np

    ids = [g for g in range(mj.ngeom)
           if int(mj.geom_bodyid[g]) == body_id and int(mj.geom_type[g]) != 0]
    hard = [g for g in ids if int(mj.geom_contype[g]) or int(mj.geom_conaffinity[g])] or ids
    if not hard:
        return None

    def _half(g):
        t, s = int(mj.geom_type[g]), np.asarray(mj.geom_size[g], dtype=float)
        if t == 2:
            return np.array([s[0]] * 3)
        if t in (3, 5):                                    # capsule / cylinder: (radius, half-length)
            return np.array([s[0], s[0], s[1] + (s[0] if t == 3 else 0.0)])
        return np.abs(s[:3])

    main = max(hard, key=lambda g: float(np.prod(_half(g)) + 1e-12))
    h = _half(main)
    span = float(2.0 * np.max(h))
    return span if span >= 0.02 else None


def _import_mesh_key(source: str, robot_id: str | None) -> str:
    """Directory name under ``build/_importmesh`` for THIS import's baked link meshes.

    The key used to be the caller's ``robot_id`` or, when that was absent, the model's BASENAME — and the
    product's own ingest path passes no robot_id, so every customer who ever drops a ``go2.xml``, a
    ``robot.urdf`` or a ``scene.xml`` wrote into one shared directory. Two imports of different robots then
    overwrite each other's STLs link by link, and an import running while another reads produces a
    partially-written file: measured, that surfaced as ``decoder failed for mesh file ... has wrong size;
    perhaps this is an ASCII file?`` and MuJoCo refuses to compile the model AT ALL, so a name collision does
    not degrade one link's appearance, it takes the whole robot down.

    The basename is kept as a readable prefix (these directories get inspected by hand) and disambiguated with a
    digest of the model's ABSOLUTE path, so the same file re-imported reuses its own directory — the cache still
    works — while two different files never share one.
    """
    import hashlib

    if robot_id:
        return _slug_name(robot_id)
    src = str(source or "")
    stem = os.path.basename(src) or "import"
    try:
        ident = str(Path(src).resolve()) if (len(src) < 512 and Path(src).exists()) else src[:4096]
    except OSError:
        ident = src[:4096]
    return f"{_slug_name(stem)}_{hashlib.md5(ident.encode('utf-8', 'replace')).hexdigest()[:10]}"


def _link_mesh_stem(body_name: str, claimed: dict[str, str]) -> str:
    """A filename stem for THIS link's baked STL that no OTHER link in the same import can take.

    ``_slug_name`` is many-to-one: it rewrites every run of characters a filesystem dislikes into a single
    ``_``, so a customer whose model names two links ``arm/1`` and ``arm:1`` -- or ``left hip`` and
    ``left_hip``, both of which occur in CAD- and xacro-exported URDFs -- collapses them onto ONE file. The
    second bake then overwrites the first and both segments' ``geometry.path`` point at it, so one link
    silently renders as another link's geometry. Nothing errors; the model still compiles; the robot is just
    wrong in a way that only shows up by eye. (Swept: zero of the 63 MuJoCo Menagerie packages collide, which
    is exactly why this cannot be left to be caught by a real model in CI.)

    MuJoCo body names are unique, so the body name itself is the key: the readable slug is kept when it is
    free, and disambiguated with a digest OF THE ORIGINAL NAME when it is not. Deterministic, so re-importing
    the same model reuses the same files, and stable regardless of the order links are baked in.
    """
    import hashlib

    stem = _slug_name(body_name)
    if claimed.setdefault(stem, body_name) != body_name:
        stem = f"{stem}_{hashlib.md5(body_name.encode('utf-8', 'replace')).hexdigest()[:8]}"
        claimed.setdefault(stem, body_name)
    return stem


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
    # ATOMIC: build the whole STL beside the target and rename it into place. A binary STL declares its triangle
    # count in the header, so a reader that opens one mid-write sees a length that disagrees with the file size
    # and MuJoCo rejects it outright -- taking down the entire model, not just this link. Writing in place made
    # that reachable whenever two imports overlapped; os.replace is atomic on both POSIX and Windows, so a reader
    # now sees either the previous complete file or the new complete file and never a half of either.
    tmp = f"{out_path}.{os.getpid()}.part"
    try:
        with open(tmp, "wb") as fh:
            fh.write(b"\0" * 80)
            fh.write(struct.pack("<I", len(T)))
            for t in T:
                n = np.cross(t[1] - t[0], t[2] - t[0])
                ln = float(np.linalg.norm(n))
                n = n / ln if ln > 1e-12 else np.zeros(3)
                fh.write(struct.pack("<12fH", *n, *t[0], *t[1], *t[2], 0))
        os.replace(tmp, out_path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
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


# --------------------------------------------------------------------------------- import memoization
#
# IMPORTING A REAL ROBOT IS NOT CHEAP, and the test suite imports the same handful over and over. MEASURED on
# this checkout (2026-08-06, one CPU), a single ``import_robot`` call: Unitree G1 13.88 s, Go2 11.29 s, Talos
# 7.45 s, Booster T1 4.93 s, Panda 4.07 s, Spot 3.84 s, UR5e 3.49 s, Cassie 2.79 s. The suite names ~115
# Menagerie model references across nine test files -- go2.xml 26 times, g1.xml 13, ur5e.xml 11 -- and every one
# re-parses the same unchanged file, re-walks the same body tree and re-bakes the same per-link STLs. A memoized
# repeat costs 2-4 ms, so on those eight files alone the suite stops paying ~563 s (9.4 min).
#
#   VIRTUROID_IMPORT_CACHE=1   memoize the import on the SOURCE's identity for the life of the process.
#
# DEFAULT-OFF, exactly like ``VIRTUROID_GAIT_FIT_CACHE`` (gait_flywheel), so a product run is byte-identical and
# a customer re-importing a file they just edited can never be handed yesterday's robot. Two things this has to
# get right, and both were failure modes of the gait cache before it got them right:
#
#   * CALLERS MUTATE WHAT THEY ARE GIVEN. ``ingest_project`` re-grounds masses onto the gene, amend operators
#     rewrite segments in place, and several tests set ``gene.loop_closures = []`` or poke ``metadata``. Handing
#     the same object to the next caller would let one import silently rewrite another's robot. So the cache
#     stores a SERIALIZED snapshot (``RobotGene.to_dict``, which round-trips) and every hit is rebuilt from a
#     deep copy of it -- including the first store, so nothing the caller does afterwards can reach the cache.
#   * THE KEY MUST COVER EVERYTHING THAT CHANGES THE ANSWER. Not just the path: a CONTENT DIGEST of the model
#     file (mtime alone is not enough -- see ``_import_cache_key``), the two options that alter the result
#     (``robot_id`` picks the baked-mesh directory and the gene id; ``species`` names it), AND a bounded
#     fingerprint of the model's own directory -- because an MJCF pulls in ``<include>`` files, meshes and
#     keyframes that live beside it, and a robot whose mesh changed is a different robot while its own .xml is
#     byte-for-byte unchanged. An XML STRING has no path, so it is keyed by digest of the text itself.
#
# A FAILED import is never cached (same rule as gait_flywheel._remember): a parse error is not an answer about
# the model, and the next caller must get a real attempt.
_IMPORT_CACHE: dict[tuple, dict] = {}
# The directory walk is bounded so a customer who drops a model into a 100k-file monorepo pays milliseconds, not
# seconds. Truncation is recorded IN the fingerprint, so a truncated key never compares equal to a complete one.
_DIR_FINGERPRINT_CAP = 5000


def _dir_fingerprint(root: Path) -> tuple:
    """``(n_files, total_bytes, newest_mtime_ns, truncated)`` for everything under ``root``.

    Deterministic (sorted walk) and cheap: 30-200 files for a Menagerie package, a few milliseconds. It is a
    fingerprint, not a hash of contents -- an edit that preserves a file's size AND its mtime is invisible to
    it, which is the documented limit of a process-scoped, opt-in test cache.
    """
    n = total = newest = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for fn in sorted(filenames):
                try:
                    st = os.stat(os.path.join(dirpath, fn))
                except OSError:
                    continue
                n += 1
                total += int(st.st_size)
                newest = max(newest, int(st.st_mtime_ns))
                if n >= _DIR_FINGERPRINT_CAP:
                    return (n, total, newest, True)
    except OSError:
        return (-1, 0, 0, False)
    return (n, total, newest, False)


_HASH_MODEL_MAX_BYTES = 4 * 1024 * 1024


def _import_cache_key(source: str, robot_id: str | None, species: str | None):
    """The identity of THIS import, or ``None`` when it must not be cached.

    The MODEL FILE ITSELF is keyed by CONTENT DIGEST, not by mtime. Mtime alone is not safe here: a test that
    writes a fixture URDF, imports it, rewrites it and imports again is an ordinary thing to do, and Windows
    file timestamps are not guaranteed to resolve two writes milliseconds apart -- so an mtime key can hand back
    the previous robot for the new file. A digest cannot. It costs microseconds on the tens-of-KB XML a robot
    description actually is, and files past ``_HASH_MODEL_MAX_BYTES`` fall back to size+mtime.
    """
    import hashlib

    if os.environ.get("VIRTUROID_IMPORT_CACHE") != "1":
        return None
    opts = (robot_id, species)
    try:
        p = Path(source)
        if len(source) < 512 and p.is_file():
            st = p.stat()
            if int(st.st_size) <= _HASH_MODEL_MAX_BYTES:
                with open(p, "rb") as fh:
                    body = ("sha256", hashlib.sha256(fh.read()).hexdigest())
            else:
                body = ("stat", int(st.st_size), int(st.st_mtime_ns))
            return ("file", str(p.resolve()), body, _dir_fingerprint(p.parent), opts)
    except OSError:
        return None
    return ("text", hashlib.sha256(source.encode("utf-8", "replace")).hexdigest(), opts)


def clear_import_cache() -> None:
    """Drop the memoized imports. For a test that deliberately wants the parser to run again."""
    _IMPORT_CACHE.clear()


def _snapshot_import(out: dict) -> dict:
    """A detached, serializable copy of an import result — safe to hand out again, whatever the caller does."""
    import copy

    gene = out.get("gene")
    return {
        "gene": copy.deepcopy(gene.to_dict()) if gene is not None else None,
        "warnings": list(out.get("warnings") or []),
        "backend_support": copy.deepcopy(out.get("backend_support") or {}),
        "species": out.get("species"), "robot_class": out.get("robot_class"),
        "valid": bool(out.get("valid")),
    }


def _restore_import(snap: dict) -> dict:
    import copy

    d = snap.get("gene")
    return {
        "gene": RobotGene.from_dict(copy.deepcopy(d)) if d is not None else None,
        "warnings": list(snap.get("warnings") or []),
        "backend_support": copy.deepcopy(snap.get("backend_support") or {}),
        "species": snap.get("species"), "robot_class": snap.get("robot_class"),
        "valid": bool(snap.get("valid")),
    }


def import_robot(source: str, *, robot_id: str | None = None, species: str | None = None) -> dict:
    """Import a URDF/MJCF (file path or XML string) into a RobotGene.

    Returns ``{"gene", "warnings", "backend_support", "species", "robot_class", "valid"}``.
    Needs MuJoCo. Raises only if the source can't be parsed at all (with a clear message).

    Memoized ONLY under ``VIRTUROID_IMPORT_CACHE=1`` (see the block above); otherwise every call does the full
    parse, exactly as before.
    """
    key = _import_cache_key(source, robot_id, species)
    snap = _IMPORT_CACHE.get(key) if key is not None else None
    if snap is not None:
        return _restore_import(snap)
    out = _import_robot_uncached(source, robot_id=robot_id, species=species)
    if key is not None:
        try:                                     # a cache may make an import faster; it may never make it fail
            _IMPORT_CACHE[key] = _snapshot_import(out)
        except Exception:  # noqa: BLE001 - unsnapshotable result -> simply not memoized
            pass
    return out


def _import_robot_uncached(source: str, *, robot_id: str | None = None, species: str | None = None) -> dict:
    """The real import. ``import_robot`` is the (opt-in) memoizing front door onto this."""
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

    # AN INCLUDE FRAGMENT IS NOT A MODEL. Real model directories ship files meant to be pulled INTO a parent --
    # keyframes.xml, assets.xml, a shared defaults block -- and a customer dropping a folder will hand us one.
    # Measured on MuJoCo Menagerie: shadow_hand/keyframes.xml is the single import failure in the whole 63-model
    # corpus, and it fails as a raw `ValueError: keyframe 'scissors': invalid qpos size, expected 0, got 24`,
    # which tells the customer nothing. A <mujoco> root carrying no BODY is the signature: it declares things for
    # someone else's worldbody. Say so, and name the model file sitting next to it.
    if "<mujoco" in _text and "<body" not in _text:
        _sibling = ""
        try:
            _p = Path(source)
            if len(source) < 512 and _p.is_file():
                _peers = sorted(q.name for q in _p.parent.glob("*.xml")
                                if q != _p and "<body" in q.read_text(encoding="utf-8", errors="replace"))
                if _peers:
                    _sibling = f" The model in this directory is {_peers[0]!r}" + (
                        f" (also: {', '.join(_peers[1:4])})." if len(_peers) > 1 else ".")
        except OSError:
            pass
        return {
            "gene": None, "backend_support": {}, "species": None, "robot_class": None, "valid": False,
            "warnings": ["this is an MJCF INCLUDE FRAGMENT, not a standalone model: it has a <mujoco> root but "
                         "declares no <body>, so it only contributes keyframes/assets/defaults to a parent model "
                         "that <include>s it. Import the model file instead." + _sibling],
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
    # THE CUSTOMER'S OWN TORQUE LIMITS, from all three places a real model puts them (see the helper).
    jnt_force = _declared_joint_torque(mj, mujoco)

    roots = [i for i in range(1, mj.nbody) if int(mj.body_parentid[i]) == 0]
    if len(roots) > 1:
        warnings.append(f"{len(roots)} root bodies attach to the world ({', '.join(body_name[r] for r in roots)}); "
                        "a RobotGene needs a single root. Imported as a multi-root tree — pick a base or split.")

    segments: list[GeneSegment] = []
    seg_len_by_name: dict[str, float] = {}      # a child's mount z is measured from its parent's TIP
    _rot_by_id: dict[int, object] = {}          # body id -> S_i, the body-frame -> LINK-frame rotation
    try:                                        # per-link STLs of the customer's own meshes (visual only)
        from pathlib import Path as _P
        _mesh_dir = _P("build/_importmesh") / _import_mesh_key(source, robot_id)
        _mesh_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001 - baking source meshes is a fidelity aid, never an import blocker
        _mesh_dir = None
    _mesh_stems: dict[str, str] = {}            # baked STL stem -> the ONE link allowed to write it
    ee_candidates: list[str] = []
    # The customer's declared ACTUATOR CAPACITY and JOINT DYNAMICS, kept as a record on the gene so grounding
    # cannot overwrite what the file already told us (see the metadata block after the loop).
    src_torque: dict[str, float] = {}
    src_torque_where: dict[str, str] = {}
    src_dynamics: dict[str, dict] = {}
    undeclared_torque: list[str] = []
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
        # THE ROOT measures its own body, not the reach to one child. `_link_vector` picks the farthest child,
        # which is right for a serial link (the Go2's thigh comes out at 0.2130 m against a real 0.213) and close
        # to meaningless for a torso: it imported the Go2's trunk at 0.199 m against a real ~0.387.
        #
        # Deliberately ROOT-ONLY, and deliberately from the body's OWN geometry. The first attempt at this
        # (reverted, 3279c57) deformed 14 of 27 Menagerie models by up to 0.45 m, in three ways this avoids:
        #   * it read the axis off the PRINCIPAL AXIS OF THE CHILD SPREAD, and a humanoid torso carries a
        #     symmetric PAIR of shoulders — so that axis runs side to side and H1's chest was rebuilt 90 degrees
        #     wrong. Here the axis comes from the body's own longest collision extent, which a symmetric pair
        #     cannot mislead;
        #   * it shifted a hub's frame origin without correcting the hub's own mount to ITS parent, so every
        #     subtree below a NESTED hub slid. The root HAS no parent, so root-only removes that case entirely;
        #   * it changed the datum under `_bake_source_mesh`, which rotates but does not translate. No datum
        #     moves here — only the drawn LENGTH — so the customer's meshes stay where they were.
        #
        # Children are unaffected either way: a child's mount is computed as `local.z - parent_len`, so it
        # self-compensates for whatever length the parent has. The length only decides how the segment is DRAWN.
        # Gate: tests/test_imported_kinematics_are_preserved.py, which fails on the reverted version.
        if parent is None:
            _own = _root_own_span(mj, i)
            if _own is not None and _own > _span:
                _span = _own
                _v = _v * (_own / max(float(_np.linalg.norm(_v)), 1e-9)) if float(_np.linalg.norm(_v)) > 1e-9 \
                    else _np.array([_own, 0.0, 0.0])
        if _span >= 0.02:
            length_m = float(min(1.5, _span))
            shape = "capsule" if shape != "box" or length_m > 2.5 * radius_m else shape
            radius_m = float(min(max(radius_m, 0.015), 0.12, max(0.015, _span * 0.28)))
        # S_i rotates the segment so its local +z runs along the REAL link axis (see _link_vector).
        _S = _rot_z_to(_v) if _span >= 1e-6 else _np.eye(3)
        _rot_by_id[i] = _S

        joint_type, axis, lo, hi, torque, jid = _joint_for_body(mj, i, jnts_of, jnt_force, bname, warnings)
        axis = tuple(float(a) for a in (_S.T @ _np.asarray(axis, dtype=float)))   # axis into the segment frame
        if joint_type in ("revolute", "prismatic") and jid is not None:
            # CARRY THE DRIVETRAIN, NOT JUST THE SKELETON. armature (reflected rotor inertia), damping and
            # frictionloss are the three parameters `sysid.fit_parameters` identifies -- and the Go2 declares
            # all three (0.01 / 2.0 / 0.2) and the Panda two (0.1 / 1.0) right there in the file. Discarding
            # them and substituting the compiler's structural prior means a calibration run "identifies"
            # numbers the customer handed us, and a customer comparing their sim to ours sees a gap we
            # manufactured. The values recorded are the COMPILED ones, i.e. exactly what the customer's own
            # MuJoCo integrates with (MuJoCo's own default for all three is 0, so an undeclared parameter
            # records as the 0 their simulator uses -- not as a guess of ours).
            _dyn = _source_joint_dynamics(mj, jid)
            if _dyn is not None:
                src_dynamics[bname] = _dyn
            _decl = jnt_force.get(jid)
            if _decl:
                src_torque[bname] = float(_decl["nm"])
                src_torque_where[bname] = str(_decl["where"])
            else:
                undeclared_torque.append(bname)
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
            _fp = _mesh_dir / f"{_link_mesh_stem(bname, _mesh_stems)}.stl"
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
    # THE LOOPS ARE PART OF THE MACHINE (see _source_loop_closures). Read before the gene is built so they are
    # on it from the start and `validate()` below rules on them like any other field.
    loop_closures = _source_loop_closures(mj, mujoco, body_name, _rot_by_id, warnings)
    gene = RobotGene(
        id=robot_id or "imported_robot",
        species=species or f"{robot_class}.imported",
        robot_class=robot_class,
        segments=segments,
        loop_closures=loop_closures,
        # a robot that MOVES needs a FREE (floating 6-DOF) base, not a welded one -- "floor"/"table" weld the base
        # to the world so the body cannot translate at all (the compiled model has no base joint, and every gait
        # rolls out to 0 forward). A manipulator stays table-mounted. The set is shared with anatomy_compiler:
        # the private copy that used to sit here was missing "legged"/"biped"/"aerial"/"aquatic", so a body in
        # one of those families was bolted to a table on the way in and then judged for not walking.
        base_mount="free" if robot_class in FLOATING_BASE_CLASSES else "table",
        end_effector_type="gripper" if any("grip" in (s.name.lower()) for s in segments) else "none",
        # ``mass_source`` marks these per-link masses as AUTHORITATIVE: they are the manufacturer's own
        # ``body_mass`` read straight off the customer's model, not our estimate. Grounding must size actuators
        # around them, never replace them -- re-deriving from primitive volume x density turned a Menagerie
        # Go2's 15.206 kg into 13.235 kg of carbon-fibre guesswork (base 6.921 -> 6.107) at export time, so the
        # package shipped a robot the customer never verified and never owned.
        metadata={"imported_from": "mjcf_or_urdf", "n_bodies": mj.nbody, "n_joints": mj.njnt,
                  "mass_source": "source_model"},
    )
    # ``torque_source`` is to ACTUATOR CAPACITY what ``mass_source`` above is to mass: a marker that these
    # numbers are the manufacturer's, not ours, so ``grounded_physics.ground_gene`` sizes a BOM part around them
    # instead of replacing them with a catalog pick. There was no such marker, and the loss was total: a
    # Menagerie Go2 that declares 23.7/23.7/45.43 N.m shipped as 10.6/10.6/1.5 -- with the ORDERING INVERTED,
    # the knee being the strongest joint on the real machine and the weakest by 30x on ours -- and a UR5e's
    # 150/150/150/28/28/28 became 360/360/48/4.1/1.5/0.5. The safety consequence is not hypothetical:
    # ``sysid.excitation`` bounds a plan meant to be RUN ON PHYSICAL HARDWARE at a fraction of this number, and
    # on the Panda that put 180 N.m of commanded torque on an 87 N.m joint.
    if src_torque:
        gene.metadata["torque_source"] = "source_model"
        gene.metadata["source_actuator_torque_nm"] = dict(src_torque)
        gene.metadata["source_actuator_torque_where"] = dict(src_torque_where)
    if src_dynamics:
        gene.metadata["source_joint_dynamics"] = dict(src_dynamics)
        gene.metadata["source_joint_dynamics_provenance"] = (
            "read from the customer's compiled model (dof_armature / dof_damping / dof_frictionloss) at "
            "import; these are the values their own MuJoCo integrates with, not an estimate of ours")
    if src_torque:
        _sites = sorted(set(src_torque_where.values()))
        warnings.append(
            f"actuator torque limits for {len(src_torque)} joint(s) were read from the source model "
            f"({'; '.join(_sites)}) and are treated as AUTHORITATIVE: grounding will size a BOM part around "
            f"them instead of replacing them with a catalog pick. Range "
            f"{min(src_torque.values()):.2f}-{max(src_torque.values()):.2f} N.m.")
    if undeclared_torque:
        warnings.append(
            f"{len(undeclared_torque)} actuated joint(s) declare NO torque limit anywhere in the source "
            f"(no actuator forcerange, no force-mode ctrlrange, no joint actuatorfrcrange): "
            f"{', '.join(sorted(undeclared_torque)[:6])}"
            f"{'...' if len(undeclared_torque) > 6 else ''}. Grounding will size a catalog actuator for these "
            f"from the structural load — an ESTIMATE, not your motor. Add actuatorfrcrange (or a motor "
            f"forcerange) to the joints you care about.")
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


def _declared_joint_torque(mj, mujoco) -> dict[int, dict]:
    """``{joint_id: {"nm", "where"}}`` — the torque limit the CUSTOMER'S OWN FILE declares, from all three
    places a real robot description puts it. We used to read exactly one of them and silently return zero.

      * ``<general|motor forcerange>``     — the clamp MuJoCo applies to the actuator's scalar force.
        Franka Panda 87/87/87/87/12/12/12, UR5e 150/150/150/28/28/28. This was the only site we read.
      * ``<motor ctrlrange>`` in FORCE MODE — where ``ctrl`` IS the commanded force, so its range is a torque
        range. The Unitree Go2 declares 23.7/23.7/45.43 N.m here with ``forcelimited=false``, so reading
        forcerange returned ``[0, 0]`` and the importer recorded NOTHING for a robot whose real numbers were
        sitting in the file.
      * ``<joint actuatorfrcrange>``       — a cap on the TOTAL actuator force at that joint, which is where
        the Unitree G1 puts its entire torque spec (88/139/50/25/5 N.m) and where nothing on the actuator
        says anything at all.

    ``ctrlrange`` IS ONLY A TORQUE FOR A FORCE-MODE ACTUATOR, and that has to be checked rather than assumed.
    The G1's actuators are ``<position>`` servos whose ctrlrange is the joint's POSITION range in radians
    (``inheritrange="1"``): reading it as a torque would give the 88 N.m hip a limit of 2.88 N.m — 30x too
    weak — and stamp it as the manufacturer's own number, which is worse than reading nothing. Force mode is
    ``gaintype=fixed`` (ctrl scaled by a constant), ``biastype=none`` (no position/velocity feedback term) and
    ``dyntype=none`` (ctrl is not filtered or integrated into a state ctrlrange does not bound) — precisely
    MJCF's ``<motor>``.

    Units: MuJoCo clamps the scalar force to ``forcerange`` and then applies ``moment * force``, and for a
    joint transmission the moment is ``gear[0]``. So the JOINT torque limit is ``|gear| * force``, and the
    force-mode ctrl path is ``|gear| * |gain| * ctrl_max``.
    """
    JT = {int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)}
    out: dict[int, dict] = {}
    for u in range(mj.nu):
        if int(mj.actuator_trntype[u]) not in JT:
            continue
        jid = int(mj.actuator_trnid[u, 0])
        gear = abs(float(mj.actuator_gear[u, 0])) or 1.0
        cands: list[tuple[float, str]] = []
        if int(mj.actuator_forcelimited[u]):
            f = max(abs(float(mj.actuator_forcerange[u][0])), abs(float(mj.actuator_forcerange[u][1])))
            if f > 0.0:
                cands.append((gear * f, "actuator/forcerange"))
        force_mode = (int(mj.actuator_gaintype[u]) == int(mujoco.mjtGain.mjGAIN_FIXED)
                      and int(mj.actuator_biastype[u]) == int(mujoco.mjtBias.mjBIAS_NONE)
                      and int(mj.actuator_dyntype[u]) == int(mujoco.mjtDyn.mjDYN_NONE))
        if force_mode and int(mj.actuator_ctrllimited[u]):
            k = abs(float(mj.actuator_gainprm[u, 0]))
            c = max(abs(float(mj.actuator_ctrlrange[u][0])), abs(float(mj.actuator_ctrlrange[u][1])))
            if k > 0.0 and c > 0.0:
                cands.append((gear * k * c, "actuator/ctrlrange (force-mode motor) x gear"))
        if not cands:
            continue
        nm, where = min(cands)                      # both clamps apply in series -> the smaller one binds
        prev = out.get(jid)
        if prev is None or nm > float(prev["nm"]):  # several actuators on one joint: the strongest sets reach
            out[jid] = {"nm": nm, "where": where}
    # The joint-level cap applies whether or not any actuator declared anything — and where nothing else did,
    # it IS the declaration.
    for j in range(mj.njnt):
        if not int(getattr(mj, "jnt_actfrclimited", [0] * mj.njnt)[j]):
            continue
        cap = max(abs(float(mj.jnt_actfrcrange[j][0])), abs(float(mj.jnt_actfrcrange[j][1])))
        if cap <= 0.0:
            continue
        prev = out.get(j)
        if prev is None:
            out[j] = {"nm": cap, "where": "joint/actuatorfrcrange"}
        elif cap < float(prev["nm"]):
            out[j] = {"nm": cap, "where": f"{prev['where']} capped by joint/actuatorfrcrange"}
    return {k: {"nm": round(float(v["nm"]), 3), "where": v["where"]} for k, v in out.items()
            if round(float(v["nm"]), 3) > 0.0}


def _source_joint_dynamics(mj, j: int) -> dict | None:
    """``{armature, damping, frictionloss}`` for joint ``j`` as the customer's model compiles them.

    These three are exactly what ``sysid.fit_parameters`` identifies. The Go2 declares all three
    (``armature=0.01 damping=2.0 frictionloss=0.2``) and the Panda two (``armature=0.1 damping=1.0``); before
    this the importer had no reference to any of them anywhere in the file and the compiler's structural
    prior stood in — so a bench calibration "identified" values the customer had already handed us, and the
    sim-to-sim gap against their own model was one we created at import.
    """
    adr = int(mj.jnt_dofadr[j])
    if adr < 0:
        return None
    return {"armature": round(float(mj.dof_armature[adr]), 6),
            "damping": round(float(mj.dof_damping[adr]), 6),
            "frictionloss": round(float(mj.dof_frictionloss[adr]), 6)}


def _source_loop_closures(mj, mujoco, body_name: dict, rot_by_id: dict, warnings: list[str]) -> list[dict]:
    """The source model's ``<equality>`` block, as far as a RobotGene can carry it.

    A CLOSED-LOOP ROBOT WITHOUT ITS LOOPS IS A DIFFERENT MACHINE. Agility's Cassie declares four
    ``<connect>`` constraints -- the plantar rod to the foot and the achilles rod to the heel spring, per leg --
    and those four are what make each leg a four-bar linkage instead of a leg with two rods dangling off it.
    The importer read the body tree and nothing else, so the twin compiled with ``neq = 0``: the rods swung
    free, the linkage did not exist, and we verified, certified and exported that machine as the customer's.

    Nothing here is new machinery. ``RobotGene.loop_closures`` already models exactly this, ``gene_compiler``
    already emits ``<connect>`` from it, and ``gene_validation`` already checks ``loop_closures_compiled``
    against ``m.neq``. This is the READ that was never wired.

    Two frames have to be reconciled, the same pair the rest of this importer reconciles:
      * MJCF's ``anchor`` is in **body1's own frame**; a segment's frame is the LINK frame, whose +z runs
        along the link axis (see ``_link_vector``). So the anchor is rotated by ``S^T`` like every other
        vector this importer carries across.
      * A ``<connect>`` may name SITES rather than bodies (both ToddlerBots do). Then the anchor is the
        site's position on its body and ``eq_data`` is zero, so the site has to be resolved to its body.

    Everything the gene CANNOT model is reported instead of dropped in silence. ``<equality><joint>`` -- a
    coupled/mimic DOF, which is what a Robotiq or Panda gripper uses to slave one finger to the other -- has no
    representation in a RobotGene at all, and 24 of the 37 Menagerie models that declare equalities declare
    only that kind. Saying so is the difference between a lossy import and a wrong one.
    """
    import numpy as np

    CONNECT = int(mujoco.mjtEq.mjEQ_CONNECT)
    SITE = int(mujoco.mjtObj.mjOBJ_SITE)
    _TYPE = {int(getattr(mujoco.mjtEq, n)): n.replace("mjEQ_", "").lower()
             for n in dir(mujoco.mjtEq) if n.startswith("mjEQ_")}
    active0 = getattr(mj, "eq_active0", None)
    if active0 is None:
        active0 = getattr(mj, "eq_active", None)

    loops: list[dict] = []
    unmodelled: dict[str, int] = {}
    skipped: list[str] = []
    for e in range(int(getattr(mj, "neq", 0))):
        et = int(mj.eq_type[e])
        if et != CONNECT:
            k = _TYPE.get(et, str(et))
            unmodelled[k] = unmodelled.get(k, 0) + 1
            continue
        if active0 is not None and not int(active0[e]):
            skipped.append("one <connect> is shipped DISABLED (active=\"false\") and was not carried")
            continue
        objtype = int(mj.eq_objtype[e]) if hasattr(mj, "eq_objtype") else -1
        o1, o2 = int(mj.eq_obj1id[e]), int(mj.eq_obj2id[e])
        if objtype == SITE:
            b1, b2 = int(mj.site_bodyid[o1]), int(mj.site_bodyid[o2])
            anchor = np.asarray(mj.site_pos[o1], dtype=float)
        else:
            b1, b2 = o1, o2
            anchor = np.asarray(mj.eq_data[e][:3], dtype=float)
        a, b = body_name.get(b1), body_name.get(b2)
        if b1 <= 0 or b2 <= 0 or not a or not b:
            skipped.append("a <connect> anchors a body to the WORLD; a RobotGene loop joins two segments")
            continue
        if a == b:
            skipped.append(f"a <connect> joins {a!r} to itself")
            continue
        if int(mj.body_parentid[b2]) == b1 or int(mj.body_parentid[b1]) == b2:
            # `validate()` rejects this on purpose: a parent and child are already rigidly related through
            # their joint, so restating the edge as a loop is a contradiction rather than a second path.
            skipped.append(f"a <connect> joins {a!r} and {b!r}, which are already parent and child")
            continue
        S = rot_by_id.get(b1)
        local = (np.asarray(S, dtype=float).T @ anchor) if S is not None else anchor
        loops.append({"a": a, "b": b, "anchor": [round(float(v), 6) for v in local]})

    if loops:
        _pairs = ", ".join(sorted({f"{lc['a']}<->{lc['b']}" for lc in loops})[:4])
        warnings.append(
            f"{len(loops)} closed kinematic loop(s) were read from the source's <equality><connect> and are "
            f"carried on the gene ({_pairs}{'...' if len(loops) > 4 else ''}). Without them the twin is a "
            f"DIFFERENT MACHINE — the loop members swing free and the linkage does not exist.")
    for msg in dict.fromkeys(skipped):
        warnings.append(f"equality constraint not carried: {msg}.")
    if unmodelled:
        _kinds = ", ".join(f"{n}x <{k}>" for k, n in sorted(unmodelled.items()))
        warnings.append(
            f"the source declares {sum(unmodelled.values())} equality constraint(s) a RobotGene cannot model "
            f"({_kinds}) and they were NOT carried. A <joint> equality couples two DOF (a mimic/slaved gripper "
            f"finger); the twin's corresponding joints move independently.")
    return loops


def _joint_for_body(mj, body_id, jnts_of, jnt_force, bname, warnings):
    """Return (joint_type, axis, lower, upper, torque, joint_id) for the joint attaching this body to its
    parent. ``joint_id`` is the SOURCE joint index (None for a weld) so the caller can carry the source's own
    per-joint drivetrain record — armature/damping/frictionloss — off the same joint."""
    js = jnts_of.get(body_id, [])
    if not js:
        return None, (0.0, 0.0, 1.0), None, None, None, None   # welded (fixed) link
    if len(js) > 1:
        warnings.append(f"body {bname!r} has {len(js)} joints; a gene segment models one. Using the first.")
    j = js[0]
    jt_int = int(mj.jnt_type[j])
    jt = _JNT.get(jt_int)
    if jt is None:
        warnings.append(f"body {bname!r} uses an unsupported joint type ({'free' if jt_int == 0 else 'ball'}); "
                        "modeled as a fixed weld. Free/ball joints aren't representable in a RobotGene chain.")
        return None, (0.0, 0.0, 1.0), None, None, None, None
    axis = tuple(round(float(v), 4) for v in mj.jnt_axis[j])
    lo = hi = None
    if int(mj.jnt_limited[j]):
        lo, hi = float(mj.jnt_range[j][0]), float(mj.jnt_range[j][1])
    else:
        warnings.append(f"joint on {bname!r} has no limits (continuous); set joint_lower/upper for a bounded design.")
    torque = float((jnt_force.get(j) or {}).get("nm") or 0.0) or None
    return jt, axis, lo, hi, torque, j


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
        # ONE ladder, shared with every other site that turns a leg count into a family (body_kind).
        return family_from_legs(legs) or "mobile_base"          # nothing carries it: rolls, flies or is carried
    # FIXED BASE, but that is weak evidence on its own: URDF has no floating-base concept, so MuJoCo's URDF
    # loader adds no freejoint and an ordinary URDF quadruped -- or one whose static torso got fused into the
    # world, leaving four separate leg roots -- looks exactly as "fixed" as a bench-mounted hand. Only a real
    # SUPPORT POLYGON promotes it: >=3 limbs whose tips span meaningful ground area. Measured, that threshold has
    # room on both sides (quadrupeds 1.04-1.79, casters 0.078, every hand and gripper exactly 0.0 because
    # fingertips are collinear). Deliberately NOT extended to 2 limbs: a two-fingered gripper would become a
    # biped, and a fixed-base URDF biped is far rarer than that mistake would be common.
    if legs >= 3 and support >= 0.5:
        return family_from_legs(legs)
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
