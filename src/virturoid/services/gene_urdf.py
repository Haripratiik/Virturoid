"""Emit a geometrically-correct URDF for ANY RobotGene by transcribing the COMPILED MuJoCo model.

The legacy path (``urdf_exporter`` + ``robot_kinematics.compute_arm_layout``) assumes a SERIAL ARM: every child
link is placed at ``(0, 0, parent_length)`` — straight up its parent. For a legged/multi-limb body that stacks
all N legs vertically at the torso tip (the Studio viewport showed a hexapod as a pile of boxes). MuJoCo already
lays every body out correctly (legs spread to their real mount offsets + orientations), so we read each body's
pose relative to its parent (``body_pos``/``body_quat`` = exactly a URDF joint origin) and each joint's axis/limits
straight from the model. Result: the URDF matches what we simulate, for a quad/hexapod/humanoid/arm alike.

Visuals + collisions are the model's primitive geoms (box/cylinder/sphere) — self-contained, no external meshes.

CONSTRAINTS. A RobotGene carries two kinds URDF treats very differently, and this file emits one and discloses
the other. ``coupled_joints`` (mimic/slaved DOF: a Panda's two fingers, a Stretch's 10x gripper, ToddlerBot's
negative gear ratios) map exactly onto URDF's native ``<mimic>`` — a degree-1 relation, which is what all 97
corpus couplings turned out to be — so they are EMITTED. ``loop_closures`` cannot be expressed at all: URDF is a
strict tree, one parent per link, and a closed loop needs a second path. Those are stated in an XML comment in
the file itself rather than dropped in silence.
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

_MJ_FREE, _MJ_SLIDE, _MJ_HINGE = 0, 2, 3
_MJ_SPHERE, _MJ_CAPSULE, _MJ_ELLIPSOID, _MJ_CYLINDER, _MJ_BOX = 2, 3, 4, 5, 6

_MESH_CACHE_DIR = None


def _mesh_cache_dir() -> str:
    """A process-shared cache dir for baked shape-program STLs (content-hash filenames), so an
    identically-shaped body/segment bakes ONCE per process — a big win for the test suite and a long-running
    server that rebuilds the same species. Copied into each package so the exported URDF stays self-contained."""
    global _MESH_CACHE_DIR
    if _MESH_CACHE_DIR is None:
        import tempfile
        _MESH_CACHE_DIR = tempfile.mkdtemp(prefix="virturoid_mesh_cache_")
    return _MESH_CACHE_DIR


def _quat_to_rpy(q) -> tuple[float, float, float]:
    """MuJoCo quat (w, x, y, z) -> URDF roll/pitch/yaw (rad), the ZYX/fixed-axis convention URDF uses."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _geom_visual_xml(model, g) -> tuple[str, str]:
    """(geometry_xml, origin_xyz_rpy_attr) for one MuJoCo geom as a URDF primitive."""
    import numpy as np
    gt = int(model.geom_type[g])
    s = [float(v) for v in model.geom_size[g]]
    pos = [float(v) for v in model.geom_pos[g]]
    r, p, y = _quat_to_rpy(model.geom_quat[g])
    origin = f'<origin xyz="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" rpy="{r:.5f} {p:.5f} {y:.5f}" />'
    if gt == _MJ_BOX:
        geom = f'<box size="{2 * s[0]:.5f} {2 * s[1]:.5f} {2 * s[2]:.5f}" />'
    elif gt in (_MJ_CAPSULE, _MJ_CYLINDER):
        geom = f'<cylinder radius="{max(1e-4, s[0]):.5f}" length="{max(1e-4, 2 * s[1]):.5f}" />'
    elif gt == _MJ_SPHERE:
        geom = f'<sphere radius="{max(1e-4, s[0]):.5f}" />'
    elif gt == _MJ_ELLIPSOID:
        geom = f'<sphere radius="{max(1e-4, max(s[:3])):.5f}" />'
    else:
        return "", ""                                   # plane / hfield / mesh: skip
    del np
    return geom, origin


def _comment(text: str) -> str:
    """One well-formed XML comment carrying ``text``.

    Disclosure is only worth anything if the file still parses. An XML comment may not contain ``--`` or end in
    ``-``, and the text below interpolates SEGMENT NAMES that come from a customer's own model, so neither rule
    can be satisfied by writing the literal carefully. Double hyphens are separated and a trailing hyphen padded;
    nothing is dropped.
    """
    body = " ".join(str(text).split()).replace("--", "- -")
    while body.endswith("-"):
        body += " "
    return f"  <!-- {body} -->"


def _num(v) -> str:
    """A coupling ratio printed at full carried precision. ToddlerBot's neck runs through -0.90909091 and its
    hip through -0.85714286; ``%.4f`` would ship -0.9091 and -0.8571, i.e. a different gearbox."""
    return f"{float(v):.10g}"


def _flatten_couplings(gene):
    """Resolve every carried coupling onto an INDEPENDENT driver joint. ``{driven: (driver, multiplier, offset)}``.

    A gene coupling is ``q_a = offset + ratio * q_b`` and URDF's ``<mimic>`` is ``q_a = multiplier * q_b + offset``
    -- the same degree-1 relation, so the carried numbers transfer 1:1 with no rescaling.

    They are COMPOSED first, because two corpus bodies declare CHAINS: a Stretch's four telescoping arm stages
    are ``l0<-l1<-l2<-l3`` and a Talos gripper's six-bar is ``motor_single<-motor_double<-inner_double``. MuJoCo
    solves such a chain jointly, but a URDF consumer does not: ``robot_state_publisher`` evaluates each mimic
    from the joint_states message, and a mimic whose reference is ITSELF a mimic never appears there -- so the
    dependent stage reads 0 and the arm telescopes wrong. Composition is exact for degree 1
    (``a = r1*b + o1``, ``b = r2*c + o2`` => ``a = r1*r2*c + (r1*o2 + o1)``), so flattening loses nothing and
    every emitted ``<mimic>`` points at a joint the controller actually commands.

    Returns ``(resolved, notes)``; ``notes`` names anything NOT emitted, so the URDF can say so in the file.
    """
    direct: dict[str, tuple[str, float, float]] = {}
    notes: list[str] = []
    for cj in (getattr(gene, "coupled_joints", None) or []):
        a, b = (cj or {}).get("a"), (cj or {}).get("b")
        try:
            ratio = float((cj or {}).get("ratio", 1.0))
            offset = float((cj or {}).get("offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            notes.append(f"a coupling on {a!r} has a non-numeric ratio/offset")
            continue
        if not a or not b or a == b or ratio == 0.0 or ratio != ratio or abs(ratio) == float("inf"):
            notes.append(f"a coupling naming {a!r}/{b!r} is malformed (self-join or zero/non-finite ratio)")
            continue
        if a in direct:
            # URDF allows exactly ONE <mimic> per joint. Two couplings driving one joint is a constraint pair
            # MuJoCo can solve and URDF cannot state; emitting the second would silently overwrite the first.
            notes.append(f"{a} is driven by more than one coupling; URDF permits one <mimic> per joint, so only "
                         f"the relation to {direct[a][0]} is emitted")
            continue
        direct[a] = (str(b), ratio, offset)

    resolved: dict[str, tuple[str, float, float]] = {}
    for a, (b0, r0, o0) in direct.items():
        drv, mul, off = b0, r0, o0
        seen = {a}
        while drv in direct and drv not in seen:
            seen.add(drv)
            d2, r2, o2 = direct[drv]
            drv, mul, off = d2, mul * r2, mul * o2 + off
        if drv in direct:                               # closed cycle of couplings: no independent driver exists
            notes.append(f"the coupling on {a} closes a cycle ({' -> '.join(sorted(seen))}); URDF <mimic> needs "
                         "an independently-commanded reference joint and a cycle has none")
            continue
        resolved[a] = (drv, mul, off)
    return resolved, notes


def gene_to_urdf(gene, *, name: str | None = None, mesh_dir: str | None = None) -> str:
    """Compile ``gene`` to MuJoCo and transcribe it into a geometrically-correct URDF string.

    VISUAL-MESH BRIDGE (the differentiation fix, ISSUES A2): when ``mesh_dir`` is given AND build123d is
    available, each segment's SHAPE PROGRAM (``geometry``) is baked to an STL in ``mesh_dir`` and emitted as
    that link's VISUAL ``<mesh>`` — so the Studio viewport (which loads this URDF) renders the real octopus
    mantle / tapering tentacle instead of the collider capsule. COLLISION stays the primitive (physics
    untouched). Falls back to self-contained primitive visuals when ``mesh_dir`` is None or any bake fails —
    byte-identical to the prior behavior, so a missing build123d never breaks the export."""
    import os

    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    model = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=False, spawn_z=standing_spawn_z(gene)))
    robot_name = escape(name or getattr(gene, "species", None) or getattr(gene, "id", None) or "virturoid_robot")

    # Bake each segment's shape program to an STL (normalized to the link's [0,length] +z frame — the SAME
    # convention gene_to_meshed_mjcf uses, so a mesh visual with an IDENTITY origin aligns with the link frame
    # the URDF already transcribes). Keyed by segment name == MJCF body name. Best-effort.
    seg_mesh: dict[str, str] = {}
    if mesh_dir:
        try:
            from virturoid.services.cad_geometry import build_visual_meshes
            # Bake into the process-shared cache (bakes each unique segment shape ONCE), then copy the referenced
            # STLs next to robot.urdf so the exported package remains self-contained (no external dependency).
            from virturoid.services.gene_compiler import stage_mesh
            baked = build_visual_meshes(gene, _mesh_cache_dir())
            os.makedirs(str(mesh_dir), exist_ok=True)
            # ``stage_mesh``, not a copy keyed on the basename: the old rule skipped the copy whenever a file of
            # that name already sat in ``mesh_dir``, so two links whose baked STLs share a basename shipped ONE
            # geometry under both names -- and a leftover from an earlier export of a DIFFERENT robot was
            # adopted as this one's. Shared claim map => one destination file per link, verified byte-identical.
            claimed: dict[str, str] = {}
            for seg, src in baked.items():
                seg_mesh[seg] = str(stage_mesh(src, mesh_dir, seg, claimed))
        except Exception:  # noqa: BLE001 - no build123d / awkward solid -> primitive visuals below
            seg_mesh = {}

    def bname(b):
        return escape(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"link{b}")

    # one joint per body (our genes are 1-DOF-per-segment); free joint = the floating base (not a URDF joint)
    joint_of = {}
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt in (_MJ_HINGE, _MJ_SLIDE):
            joint_of[int(model.jnt_bodyid[j])] = j

    def jname_of(j):
        return escape(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                      or f"{bname(int(model.jnt_bodyid[j]))}_joint")

    def joint_id_for_segment(seg) -> int | None:
        """The compiled model's joint for a gene segment. Read through the MODEL rather than by string-formatting
        ``<segment>_joint``, so the reference in a <mimic> is by construction the same name the joint below is
        WRITTEN under -- including whatever escaping or fallback that name went through."""
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(seg))
        return joint_of.get(int(bid)) if bid >= 0 else None

    # COUPLED (mimic/slaved) DOF -> URDF's own <mimic>. This is a NATIVE URDF element for exactly this relation,
    # so a carried coupling is EMITTED here rather than merely disclosed: a Panda whose two fingers open
    # separately is not a Panda, and an engineer taking that URDF to RViz/MoveIt/Gazebo gets a machine that does
    # not exist. Ratios are carried as the source states them -- ToddlerBot's -0.90909091 and -0.85714286 are
    # NEGATIVE and a Stretch's gripper runs at 10x its slider; normalising any of those to +/-1 is a different
    # transmission.
    _couplings, _cnotes = _flatten_couplings(gene)
    _mimic_of: dict[int, str] = {}
    _emitted: list[tuple[str, str, float, float]] = []
    for _a, (_b, _mul, _off) in _couplings.items():
        _ja, _jb = joint_id_for_segment(_a), joint_id_for_segment(_b)
        if _ja is None or _jb is None or _ja == _jb:
            _cnotes.append(f"the coupling {_a}<-{_b} names a segment with no joint in the compiled model")
            continue
        _mimic_of[_ja] = (f'    <mimic joint="{jname_of(_jb)}" multiplier="{_num(_mul)}" '
                          f'offset="{_num(_off)}" />')
        _emitted.append((jname_of(_ja), jname_of(_jb), _mul, _off))

    lines = [f'<robot name="{robot_name}">']
    if _emitted:
        lines.append(_comment(
            f"{len(_emitted)} coupled (mimic/slaved) joint pair(s) from this design are emitted below as URDF "
            "<mimic>: "
            + "; ".join(f"{a} = {_num(m)}*{b}" + (f" + {_num(o)}" if o else "") for a, b, m, o in _emitted)
            + ". These DOF are NOT independently commandable: a controller that drives a mimic joint on its "
            "own is fighting the transmission. Chained couplings are composed onto an independently-driven "
            "reference joint (exact for a degree-1 relation), because a <mimic> whose reference is itself a "
            "<mimic> is not resolved by robot_state_publisher."))
    if _cnotes:
        lines.append(_comment(
            "WARNING: coupling(s) NOT represented below: "
            + "; ".join(str(n) for n in dict.fromkeys(_cnotes))
            + ". The MJCF export carries them as <equality><joint>; in this file those DOF move independently."))
    # URDF IS A TREE, so a declared closed loop CANNOT be represented here — a gantry's bridge can only hang off
    # one column, a delta's arms cannot share a platform. Dropping that silently is the worst option available:
    # the file would look complete and ship the exact cantilever the loop was added to fix, under a green export.
    # So say it, in the file itself, where anyone opening the URDF will see it.
    _loops = getattr(gene, "loop_closures", None) or []
    if _loops:
        lines.append(_comment(
            f"WARNING: {len(_loops)} closed kinematic loop(s) in this design are NOT represented below. "
            "URDF is a strict tree (one parent per link) and cannot express them. Affected: "
            + "; ".join(f"{str((lc or {}).get('a'))}<->{str((lc or {}).get('b'))}" for lc in _loops)
            + ". The MJCF export carries them as <equality><connect>; this file describes the same robot with "
            "those joins OPEN, so any load path through them is missing."))
    for b in range(1, model.nbody):
        nm = bname(b)
        mass = max(1e-4, float(model.body_mass[b]))
        vis, col = [], []
        body_rgba = None
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != b:
                continue
            geom, origin = _geom_visual_xml(model, g)
            if not geom:
                continue
            # `geom_rgba` is frequently MuJoCo's neutral default when colour
            # comes from a named material.  Resolve the material first so the
            # Studio sees the same palette as the simulated model.
            matid = int(model.geom_matid[g])
            rgba = ([float(v) for v in model.mat_rgba[matid]]
                    if matid >= 0 else [float(v) for v in model.geom_rgba[g]])
            if body_rgba is None and len(rgba) >= 3:
                body_rgba = rgba                        # the mesh-visual color = this link's primary geom color
            mat = (f'<material name="{nm}_m{g}"><color rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1" /></material>'
                   if len(rgba) >= 3 else "")
            vis.append(f'    <visual>{origin}<geometry>{geom}</geometry>{mat}</visual>')
            # COLLISION == WHAT ACTUALLY COLLIDES. Every geom used to be transcribed into both blocks, so the
            # shipped URDF declared the decoration -- motor housings, panel/vent detail, the visual-only shells
            # that carry mass=0 contype=0 conaffinity=0 -- as real collision geometry. Measured: a hexapod
            # exported 62 <collision> elements where the model we simulate, verify and train on collides with
            # 26; a Go2 exported 46 against 13. That is a customer loading our URDF into Gazebo or RViz and
            # getting a robot that self-collides on parts our own simulator passes straight through.
            #
            # It also has to be read from ``contype``/``conaffinity`` rather than from the geom's provenance,
            # because otherwise the collision set MOVES whenever the visual layer changes: attaching an
            # imported link's own mesh suppresses that link's housing + detail geoms, which silently dropped 33
            # of the Go2's 46 declared collisions. Reading what collides makes the URDF's collision set equal
            # to the compiled model's -- for a generated body and an imported one alike -- and invariant to
            # every visual decision above.
            if int(model.geom_contype[g]) or int(model.geom_conaffinity[g]):
                col.append(f'    <collision>{origin}<geometry>{geom}</geometry></collision>')
        if not vis:                                     # a geom-less body still needs a renderable stub
            vis.append('    <visual><geometry><box size="0.02 0.02 0.02" /></geometry></visual>')
        if not col:
            # A body whose every geom is visual-only has no collider to declare. Every link carried a collision
            # element before this, and some importers assume one, so it keeps a 1 mm placeholder at the link
            # origin rather than none: structurally the same file, and honest about there being nothing there.
            col.append('    <collision><geometry><box size="0.001 0.001 0.001" /></geometry></collision>')
        # VISUAL-MESH BRIDGE: if this link's segment baked a shape-program STL, the link's VISUAL becomes that
        # single mesh (identity origin — the STL is already in the link's [0,length] +z frame), while COLLISION
        # keeps the primitive geoms above (physics untouched). This is what makes an octopus render as an octopus.
        if nm in seg_mesh:
            rel = "meshes/" + escape(os.path.basename(seg_mesh[nm]))
            c = body_rgba or [0.5, 0.55, 0.6]
            mmat = (f'<material name="{nm}_mesh"><color rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1" /></material>')
            vis = [f'    <visual><origin xyz="0 0 0" rpy="0 0 0" /><geometry>'
                   f'<mesh filename="{rel}" scale="0.001 0.001 0.001" /></geometry>{mmat}</visual>']
        # diagonal inertia about the body frame (from the model's body inertia, a safe positive-definite set)
        ine = [float(v) for v in model.body_inertia[b]]
        lines.append(f'  <link name="{nm}">')
        lines.append(f'    <inertial><origin xyz="0 0 0" rpy="0 0 0" /><mass value="{round(mass, 5)}" />'
                     f'<inertia ixx="{max(1e-6, ine[0]):.6f}" ixy="0" ixz="0" iyy="{max(1e-6, ine[1]):.6f}" '
                     f'iyz="0" izz="{max(1e-6, ine[2]):.6f}" /></inertial>')
        lines.extend(vis)
        lines.extend(col)
        lines.append("  </link>")

    for b in range(1, model.nbody):
        parent = int(model.body_parentid[b])
        if parent == 0:
            continue                                    # a top-level body = the root link (URDF root, no joint)
        pos = [float(v) for v in model.body_pos[b]]
        r, p, y = _quat_to_rpy(model.body_quat[b])
        origin = f'    <origin xyz="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" rpy="{r:.5f} {p:.5f} {y:.5f}" />'
        j = joint_of.get(b)
        if j is None:                                   # welded body -> a fixed joint
            lines += [f'  <joint name="{bname(b)}_fixed" type="fixed">',
                      f'    <parent link="{bname(parent)}" /><child link="{bname(b)}" />', origin, "  </joint>"]
            continue
        jtype = "revolute" if int(model.jnt_type[j]) == _MJ_HINGE else "prismatic"
        axis = np.asarray(model.jnt_axis[j], float)
        lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
        if not bool(model.jnt_limited[j]):
            lo, hi = -3.14159, 3.14159
        jname = jname_of(j)
        lines += [f'  <joint name="{jname}" type="{jtype}">',
                  f'    <parent link="{bname(parent)}" /><child link="{bname(b)}" />', origin,
                  f'    <axis xyz="{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}" />',
                  f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="30" velocity="10" />']
        if j in _mimic_of:
            lines.append(_mimic_of[j])
        lines.append("  </joint>")

    lines.append("</robot>")
    return "\n".join(lines) + "\n"
