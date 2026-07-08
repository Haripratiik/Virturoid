"""Render the robot's SENSORS where they actually sit — visual-only geoms placed ACCURATELY on the right body
part, using each segment's real WORLD orientation (forward-kinematics), not an assumed local frame.

The BOM says a humanoid/quadruped carries camera "eyes", a LiDAR, an IMU, etc.; this turns that parts list
into geometry the viewport shows in the correct POSITION + DIRECTION: cameras (housing + lens) on the FORWARD
face of the sensor head pointing forward, a LiDAR puck on TOP of the chassis pointing up, a wrist F/T collar on
the end-effector. Because placement is solved from the segment's actual world rotation, it is correct whether
the head's local +z points forward (the anatomy compiler) or up (the anthropometric humanoid). Everything is
``mass=0 contype=0 conaffinity=0`` (visual-only) so dynamics/contacts stay byte-identical.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

_VIS = 'mass="0" contype="0" conaffinity="0"'


def _head(gene):
    h = next((s for s in gene.segments if "head" in s.name.lower()), None)
    if h is not None:
        return h
    shells = [s for s in gene.segments if getattr(s, "material", None) == "shell" and s.parent is not None]
    return shells[-1] if shells else gene.root()


def _host(cat: str, mount: str, gene):
    m = (mount or "").lower()
    if cat == "camera" and ("wrist" in m or "hand" in m or "eye-in-hand" in m):
        return gene.end_effector() or _head(gene)
    if cat == "force_torque" or "wrist" in m:
        return gene.end_effector() or gene.root()
    if cat in ("camera", "thermal") or "head" in m:
        return _head(gene) or gene.root()
    return gene.root()                                  # torso / chassis / deck / mast -> the body


def _world_frames(gene) -> dict:
    """{segment_name: 3x3 world rotation} at the robot's rest stance, via a quick FK pass on the primitive
    model. Lets us place a sensor along a WORLD direction (forward/up) regardless of the link's local frame."""
    try:
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))
        d = mujoco.MjData(m)
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        out = {}
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm:
                out[nm] = np.array(d.xmat[b]).reshape(3, 3)
        return out
    except Exception:  # noqa: BLE001
        return {}


def _box(name, p, hx, hy, hz, mat):
    return (f'<geom name="{name}" type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="{mat}" {_VIS}/>')


def _cyl(name, a, b, rad, mat):
    return (f'<geom name="{name}" type="cylinder" '
            f'fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" '
            f'size="{rad:.4f}" material="{mat}" {_VIS}/>')


def _pinned_camera(gene):
    """The camera the user PINNED for this robot (its datasheet FOV wins over the auto-selected suite camera)."""
    name = (getattr(gene, "metadata", None) or {}).get("pinned_parts", {}).get("camera")
    if name:
        from virturoid.services.component_catalog import resolve_part
        p = resolve_part(name)
        if p is not None and getattr(p, "category", "") == "camera":
            return p
    return None


def _camera_fovy_deg(comp) -> float:
    """The camera part's VERTICAL field of view (MuJoCo fovy) from its real datasheet specs, or a sensible default.
    fov_deg is stored [horizontal, vertical] (or [h]); MuJoCo's fovy is the vertical angle."""
    specs = getattr(comp, "specs", None) or {}
    fov = specs.get("fov_deg")
    if isinstance(fov, (list, tuple)) and fov:
        return float(fov[1] if len(fov) > 1 else fov[0])
    if isinstance(fov, (int, float)):
        return float(fov)
    return 60.0


def plan_sensor_geoms(gene, sensors, frames=None) -> dict:
    """``sensors`` = [(part_name, qty, mount), ...] from the BOM. Returns {segment_name: "<geom .../>\\n..."} —
    visual sensor geoms nested in each host body, positioned + oriented from the host's WORLD frame."""
    import numpy as np

    from virturoid.services.component_catalog import component
    if frames is None:
        frames = _world_frames(gene)

    out: dict = {}
    robot_cam_done = False
    for idx, (name, qty, mount) in enumerate(sensors):
        c = component(name)
        cat = c.category if c else "imu"
        host = _host(cat, mount, gene)
        if host is None:
            continue
        R = frames.get(host.name)
        if R is None:
            R = np.eye(3)
        L = float(host.length_m)
        r = max(0.025, float(host.radius_m))

        def world_in_local(wd):
            v = R.T @ np.array(wd, float)
            n = np.linalg.norm(v)
            return v / n if n else v

        fwd = world_in_local((1.0, 0.0, 0.0))           # robot forward / up / left, expressed in the link frame
        up = world_in_local((0.0, 0.0, 1.0))
        lat = world_in_local((0.0, 1.0, 0.0))
        center = np.array([0.0, 0.0, L / 2.0])          # link geom spans 0..L along its local +z
        reach = lambda v: abs(v[0]) * r + abs(v[1]) * r + abs(v[2]) * (L / 2.0)  # half-extent along v
        nm = escape(host.name)
        geoms: list[str] = []
        if cat == "camera":                             # housing + lens on the FORWARD face, eyes ±lateral
            n = min(max(1, int(qty)), 2)
            offs = [0.0] if n == 1 else [-0.42 * r, 0.42 * r]
            base = center + fwd * reach(fwd)
            for j, o in enumerate(offs):
                p = base + lat * o
                geoms.append(_box(f"{nm}_cam{idx}_{j}", p, max(0.012, 0.32 * r), max(0.011, 0.27 * r), 0.014, "mat_joint"))
                geoms.append(_cyl(f"{nm}_lens{idx}_{j}", p, p + fwd * 0.024, max(0.009, 0.12 * r), "mat_ee"))
            # ...and a FUNCTIONAL MuJoCo <camera name="robot_cam"> at the eye, looking FORWARD, with the vertical FOV
            # taken from the REAL camera part's datasheet (so a wide-FOV RealSense and a narrow lens differ in sim).
            # A MuJoCo cam looks down its own -z; xyaxes gives (right, up), and view = -cross(right, up) = fwd when
            # right = cross(fwd, up). Emitted once (the perception camera), keyed so the render path can find it.
            if not robot_cam_done:
                fov = _camera_fovy_deg(_pinned_camera(gene) or c)  # a pinned camera's real FOV wins
                right = np.cross(fwd, up)
                eye = center + fwd * reach(fwd)
                geoms.append(
                    f'<camera name="robot_cam" pos="{eye[0]:.4f} {eye[1]:.4f} {eye[2]:.4f}" '
                    f'xyaxes="{right[0]:.4f} {right[1]:.4f} {right[2]:.4f} {up[0]:.4f} {up[1]:.4f} {up[2]:.4f}" '
                    f'fovy="{fov:.1f}"/>')
                robot_cam_done = True
        elif cat == "lidar":                            # vertical scanning puck on TOP, pointing up
            base = center + up * reach(up)
            geoms.append(_cyl(f"{nm}_lidar{idx}", base, base + up * 0.05, min(0.05, 0.5 * r), "mat_joint"))
        elif cat == "force_torque":                     # a collar at the wrist face
            base = center + fwd * reach(fwd)
            geoms.append(_cyl(f"{nm}_ft{idx}", base - fwd * 0.012, base, 1.1 * r, "mat_ee"))
        else:                                           # imu / gps / thermal / microphone: small housing on top
            mat = "mat_accent" if cat in ("thermal", "gps") else "mat_ee"
            base = center + up * (reach(up) * 0.95)
            geoms.append(_box(f"{nm}_{escape(cat)}{idx}", base, max(0.012, 0.2 * r), max(0.012, 0.2 * r), 0.009, mat))
        if geoms:
            out.setdefault(host.name, []).extend(geoms)
    return {k: "\n".join(v) for k, v in out.items()}


def sensor_geoms_for_gene(gene, *, task: str = "") -> dict:
    """Derive the class+task sensor suite (the same one the BOM uses) and plan its geometry from the robot's
    rest-pose world frames. Used by the render path so the viewport shows sensors in the right place."""
    try:
        from virturoid.services.bom_builder import _sensor_suite
        scale_kg = sum(s.mass_kg for s in gene.segments)
        return plan_sensor_geoms(gene, _sensor_suite(gene.robot_class, None, task, scale_kg))
    except Exception:  # noqa: BLE001 - sensors are a visual nicety; never block a compile
        return {}
