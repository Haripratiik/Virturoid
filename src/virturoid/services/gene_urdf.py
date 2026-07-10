"""Emit a geometrically-correct URDF for ANY RobotGene by transcribing the COMPILED MuJoCo model.

The legacy path (``urdf_exporter`` + ``robot_kinematics.compute_arm_layout``) assumes a SERIAL ARM: every child
link is placed at ``(0, 0, parent_length)`` — straight up its parent. For a legged/multi-limb body that stacks
all N legs vertically at the torso tip (the Studio viewport showed a hexapod as a pile of boxes). MuJoCo already
lays every body out correctly (legs spread to their real mount offsets + orientations), so we read each body's
pose relative to its parent (``body_pos``/``body_quat`` = exactly a URDF joint origin) and each joint's axis/limits
straight from the model. Result: the URDF matches what we simulate, for a quad/hexapod/humanoid/arm alike.

Visuals + collisions are the model's primitive geoms (box/cylinder/sphere) — self-contained, no external meshes.
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
            import shutil

            from virturoid.services.cad_geometry import build_visual_meshes
            # Bake into the process-shared cache (bakes each unique segment shape ONCE), then copy the referenced
            # STLs next to robot.urdf so the exported package remains self-contained (no external dependency).
            baked = build_visual_meshes(gene, _mesh_cache_dir())
            os.makedirs(str(mesh_dir), exist_ok=True)
            for seg, src in baked.items():
                dst = os.path.join(str(mesh_dir), os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copyfile(src, dst)
                seg_mesh[seg] = dst
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

    lines = [f'<robot name="{robot_name}">']
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
            col.append(f'    <collision>{origin}<geometry>{geom}</geometry></collision>')
        if not vis:                                     # a geom-less body still needs a renderable stub
            vis.append('    <visual><geometry><box size="0.02 0.02 0.02" /></geometry></visual>')
            col.append('    <collision><geometry><box size="0.02 0.02 0.02" /></geometry></collision>')
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
        jname = escape(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"{bname(b)}_joint")
        lines += [f'  <joint name="{jname}" type="{jtype}">',
                  f'    <parent link="{bname(parent)}" /><child link="{bname(b)}" />', origin,
                  f'    <axis xyz="{axis[0]:.4f} {axis[1]:.4f} {axis[2]:.4f}" />',
                  f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="30" velocity="10" />', "  </joint>"]

    lines.append("</robot>")
    return "\n".join(lines) + "\n"
