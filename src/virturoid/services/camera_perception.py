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


def robot_camera_context(gene, *, task: str = ""):
    """``(camera_part_or_None, provenance)`` — the camera this robot carries AND on whose authority.

    ``provenance`` is one of:

    * ``"pinned"``     — the customer named this exact part for this robot.
    * ``"our_design"`` — we composed this robot, so its class+task sensor suite is the design we are proposing.
    * ``None``         — no camera we are entitled to claim; ``robot_camera_part`` returns None.

    THE CASE THAT MADE THIS A TUPLE. ``_sensor_suite`` maps ``robot_class`` to parts, and for a quadruped that
    is an Intel RealSense D435i. Applied to a body we composed that is a proposal. Applied to a MENAGERIE GO2 --
    ``ncam 0, nsensor 0`` -- it fitted a camera to a machine the customer already owns, and everything
    downstream then measured it: ``sensor_geometry`` emitted a functional ``<camera name="robot_cam">`` from the
    suite, ``robot_sees_target`` rendered through it, and ``verify_robot.vision`` published
    ``camera_part: Intel RealSense D435i, sees: true`` about a camera that does not exist. So the question
    "which camera" is now inseparable from "says who". See ``sensor_provenance``.
    """
    from virturoid.services.sensor_provenance import camera_is_ours_to_add
    try:
        # A PIN IS THE CUSTOMER'S OWN INSTRUCTION and outranks both branches below: on their imported machine it
        # says "this camera is / will be on it", which is theirs to assert; on ours it overrides our pick.
        pinned = (getattr(gene, "metadata", None) or {}).get("pinned_parts", {}).get("camera")
        if pinned:
            from virturoid.services.component_catalog import resolve_part
            p = resolve_part(pinned)
            if p is not None:
                return p, "pinned"
        if not camera_is_ours_to_add(gene)[0]:
            return None, None
        from virturoid.services.bom_builder import _sensor_suite
        from virturoid.services.component_catalog import component
        scale = sum(s.mass_kg for s in gene.segments)
        for name, _qty, _mount in _sensor_suite(gene.robot_class, None, task, scale):
            c = component(name)
            if c and c.category == "camera":
                return c, "our_design"
    except Exception:  # noqa: BLE001
        pass
    return None, None


def robot_camera_part(gene, *, task: str = ""):
    """The camera part this robot carries (from its class+task sensor suite) — its real specs drive the sim camera.

    ``None`` for an imported machine whose own model declares no camera: we do not fit hardware to a robot the
    customer already owns. ``robot_camera_context`` returns the reason alongside.
    """
    return robot_camera_context(gene, task=task)[0]


def render_px_for_camera(part) -> int:
    """Render resolution from the camera part's megapixels: a higher-MP camera feeds the encoder a sharper frame.
    Clamped to [64, 256] — beyond that the 16x16 encoder can't use the detail (and it just costs render time)."""
    mp = 2.0
    specs = getattr(part, "specs", None) or {}
    if isinstance(specs.get("rgb_mp"), (int, float)):
        mp = float(specs["rgb_mp"])
    return int(min(256, max(64, round(64 * math.sqrt(max(0.3, mp))))))


def _inject_target(xml: str, xy, rgb, *, size=(0.09, 0.09, 0.18), z: float = 0.18) -> str:
    """Drop a coloured target into the scene in front of the robot (a thing for the camera to SEE). Default is a
    small floor box (a detection check); training uses a taller LANDMARK (bigger size + height) so a distant goal is
    still a meaningful signal once the view is downsampled to the tiny encoder's 16x16 input."""
    sx, sy, sz = size
    box = (f'<body name="cv_target" pos="{xy[0]:.3f} {xy[1]:.3f} {z:.3f}"><geom type="box" '
           f'size="{sx:.3f} {sy:.3f} {sz:.3f}" rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1" '
           f'contype="0" conaffinity="0"/></body>')
    return xml.replace("</worldbody>", box + "\n</worldbody>", 1)


def robot_sees_target(gene, *, goal_ahead_m: float = 1.6, goal_rgb=(0.1, 0.85, 0.15), bearing_fn=None) -> dict:
    """Render the robot's OWN camera at a coloured target placed in front of it, run the CV, and report what it
    perceives. ``bearing_fn`` (e.g. a trained TinyVisionEncoder predictor) overrides the colour detector.

    Returns {has_camera, camera_part, fovy_deg, render_px, sees, bearing, target_bearing, bearing_err}. ``sees``
    means the CV found the target in the robot's own view — the honest 'the camera + CV actually perceive' signal."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.sensor_provenance import camera_is_ours_to_add, declared, is_imported
    from virturoid.services.vision_nav import detect_goal_bearing

    part, cam_src = robot_camera_context(gene, task=(getattr(gene, "metadata", None) or {}).get("task", ""))
    out = {"has_camera": part is not None, "camera_part": getattr(part, "name", None)}
    if part is None:
        out["sees"] = False
        # WHY there is no camera matters, because the two reasons are different claims. On a body we composed,
        # "no camera" is a design fact. On the CUSTOMER'S machine it is a REFUSAL to invent one, and it has to
        # be legible as such -- the same shape as `controller_provenance.what_we_took_from_your_model`, which
        # states what was read verbatim instead of asserting past the evidence.
        if is_imported(gene):
            inv = declared(gene) or {}
            out["your_model_declares"] = {"cameras": inv.get("ncam"), "sensors": inv.get("nsensor")}
            out["we_did_not_add_one"] = camera_is_ours_to_add(gene)[1]
            out["note"] = ("no vision claim about your machine: we report what your model declares, and we do "
                           "not fit a camera to your robot and then measure it")
        else:
            out["note"] = ("this robot carries no camera in its BOM — perception is via its other sensors "
                           "(e.g. LiDAR ring).")
        return out
    out["camera_is"] = ("a part YOU pinned for this robot" if cam_src == "pinned" else
                        "part of the design we are proposing for this robot")
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
        # if the caller didn't supply a detector, PREFER the robot's own banked learned vision (trained on its own
        # camera by train_camera_policy) over the colour detector — the trained CV actually gets consumed here.
        trained = False
        if bearing_fn is None:
            from virturoid.services.robot_vision import learned_bearing_fn
            learned = learned_bearing_fn(gene, goal_rgb=goal_rgb)
            if learned is not None:
                bearing_fn, trained = learned, True
        detect = bearing_fn if bearing_fn is not None else (lambda f: detect_goal_bearing(f, goal_rgb))
        bearing, frac = detect(img)
        # the geometric bearing to the target from the robot's base (for an honest accuracy check)
        tgt_bearing = math.atan2(ty, tx)
        out["sees"] = bearing is not None and frac > 0
        out["seen_fraction"] = round(float(frac), 4)
        if bearing is not None:
            out["bearing"] = round(float(bearing), 3)
            out["target_bearing"] = round(tgt_bearing, 3)
        out["vision_trained"] = bool(trained)                    # the robot's OWN learned readout drove this perception
        out["perception"] = ("learned_onboard_camera + tiny_cv" if trained else "onboard_camera + tiny_cv")
    except Exception as exc:  # noqa: BLE001
        out.update(sees=False, error=f"{type(exc).__name__}: {exc}")
    return out
