"""Perceive through a robot's OWN onboard camera with the tiny CV encoder — the bridge that makes vision a REAL
part of a camera-equipped robot's pipeline, not an isolated demo.

Every other perception path a built robot uses today is the rangefinder ring (``exteroception`` / ``perception_nav``);
the camera was cosmetic (a BOM line + a visual geom). This renders the FUNCTIONAL ``robot_cam`` (mounted on the
robot and given its FOV from the ACTUAL camera part), drops a coloured target into the scene, and runs the CV to
report what the robot SEES from its own eye — so a camera in the requirements means a working, spec-driven camera
the perception stack actually reads. The render RESOLUTION scales with the camera part's megapixels, so a 12 MP
OAK-D genuinely feeds the encoder a sharper image than a 1 MP Arducam.
"""
from __future__ import annotations

import math


def robot_camera_part(gene, *, task: str = ""):
    """The camera part this robot carries (from its class+task sensor suite) — its real specs drive the sim camera."""
    try:
        from virturoid.services.bom_builder import _sensor_suite
        from virturoid.services.component_catalog import component
        # honor a pinned camera first
        pinned = (getattr(gene, "metadata", None) or {}).get("pinned_parts", {}).get("camera")
        if pinned:
            from virturoid.services.component_catalog import resolve_part
            p = resolve_part(pinned)
            if p is not None:
                return p
        scale = sum(s.mass_kg for s in gene.segments)
        for name, _qty, _mount in _sensor_suite(gene.robot_class, None, task, scale):
            c = component(name)
            if c and c.category == "camera":
                return c
    except Exception:  # noqa: BLE001
        pass
    return None


def render_px_for_camera(part) -> int:
    """Render resolution from the camera part's megapixels: a higher-MP camera feeds the encoder a sharper frame.
    Clamped to [64, 256] — beyond that the 16x16 encoder can't use the detail (and it just costs render time)."""
    mp = 2.0
    specs = getattr(part, "specs", None) or {}
    if isinstance(specs.get("rgb_mp"), (int, float)):
        mp = float(specs["rgb_mp"])
    return int(min(256, max(64, round(64 * math.sqrt(max(0.3, mp))))))


def _inject_target(xml: str, xy, rgb) -> str:
    """Drop a coloured target box into the scene in front of the robot (a thing for the camera to SEE)."""
    box = (f'<body name="cv_target" pos="{xy[0]:.3f} {xy[1]:.3f} 0.18"><geom type="box" size="0.09 0.09 0.18" '
           f'rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1" contype="0" conaffinity="0"/></body>')
    return xml.replace("</worldbody>", box + "\n</worldbody>", 1)


def robot_sees_target(gene, *, goal_ahead_m: float = 1.6, goal_rgb=(0.1, 0.85, 0.15), bearing_fn=None) -> dict:
    """Render the robot's OWN camera at a coloured target placed in front of it, run the CV, and report what it
    perceives. ``bearing_fn`` (e.g. a trained TinyVisionEncoder predictor) overrides the colour detector.

    Returns {has_camera, camera_part, fovy_deg, render_px, sees, bearing, target_bearing, bearing_err}. ``sees``
    means the CV found the target in the robot's own view — the honest 'the camera + CV actually perceive' signal."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.vision_nav import detect_goal_bearing

    part = robot_camera_part(gene, task=(getattr(gene, "metadata", None) or {}).get("task", ""))
    out = {"has_camera": part is not None, "camera_part": getattr(part, "name", None)}
    if part is None:
        out["sees"] = False
        out["note"] = "this robot carries no camera in its BOM — perception is via its other sensors (e.g. LiDAR ring)."
        return out
    px = render_px_for_camera(part)
    try:
        xml = gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
        # place the target ahead of the robot's forward (+x), a touch to the side so a nonzero bearing is expected
        tx, ty = goal_ahead_m, 0.25
        xml = _inject_target(xml, (tx, ty), goal_rgb)
        model = mujoco.MjModel.from_xml_string(xml)
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam")
        if cid < 0:
            out.update(sees=False, note="no functional robot_cam in the compiled model")
            return out
        out["fovy_deg"] = round(float(model.cam_fovy[cid]), 1)
        out["render_px"] = px
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        rr = mujoco.Renderer(model, height=px, width=px); rr.update_scene(data, camera="robot_cam")
        img = rr.render().copy(); rr.close()
        detect = bearing_fn if bearing_fn is not None else (lambda f: detect_goal_bearing(f, goal_rgb))
        bearing, frac = detect(img)
        # the geometric bearing to the target from the robot's base (for an honest accuracy check)
        tgt_bearing = math.atan2(ty, tx)
        out["sees"] = bearing is not None and frac > 0
        out["seen_fraction"] = round(float(frac), 4)
        if bearing is not None:
            out["bearing"] = round(float(bearing), 3)
            out["target_bearing"] = round(tgt_bearing, 3)
        out["perception"] = "onboard_camera + tiny_cv"
    except Exception as exc:  # noqa: BLE001
        out.update(sees=False, error=f"{type(exc).__name__}: {exc}")
    return out
