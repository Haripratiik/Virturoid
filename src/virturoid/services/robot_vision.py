"""Train the tiny CV encoder on a BUILT robot's OWN onboard camera, then have that robot NAVIGATE by its learned
vision — the in-house CV training that ties the perception to the actual robot, on CPU (no Madrona needed).

vision_train proves the encoder learns on a generic bearing scene; this trains it on the SPECIFIC robot's
functional ``robot_cam`` (mounted + FOV'd from the robot's real camera part), so the learned perception is
calibrated to the camera the robot actually carries. Then ``robot_vision_nav`` glides the robot to a coloured goal
using ONLY the learned encoder's read of its own camera — the honest "the robot is driven by trained vision".

Scope: supervised perception + a steer-to-bearing controller (CPU-real, deterministic, testable). Large-scale
end-to-end vision-RL (SAC over pixels, 1024 envs) is the Madrona-MJX / native-Linux extension; this owns the
per-robot camera dataset, the training objective, and the closed-loop metric that extension would optimize.
"""
from __future__ import annotations

import math

GOAL_RGB = (0.1, 0.85, 0.15)
_FOV_HALF = 0.62   # normalize the bearing label to ~[-1, 1] over the camera's half-FOV (matches vision_train)
# a TALL coloured landmark (not a small floor box) so a distant goal is still a real signal after the render is
# downsampled to the encoder's 16x16 input — a 0.4x0.4 m, ~1.4 m-tall pillar.
_GOAL_SIZE = (0.20, 0.20, 0.70)
_GOAL_Z = 0.70


def _cam_half_fov(gene) -> float:
    """The robot camera's half horizontal FOV in radians (with a margin so the goal stays fully in frame). The
    render is square, so horizontal FOV == the part's vertical fovy. Sampling/steering INSIDE this keeps the goal
    visible — the fix for the 40%-off-screen noise that stopped the encoder converging."""
    from virturoid.services.camera_perception import robot_camera_part
    from virturoid.services.sensor_geometry import _camera_fovy_deg
    fovy = _camera_fovy_deg(robot_camera_part(gene)) if robot_camera_part(gene) else 60.0
    return math.radians(fovy / 2.0) * 0.82


def _base_qadr(model):
    """qpos address of the robot's free (floating) base — the 7 numbers (xyz + quat) we glide it by."""
    import mujoco
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[j])
    return -1


def _set_base(model, data, adr, x, y, z, yaw):
    import mujoco
    if adr < 0:
        return
    data.qpos[adr:adr + 3] = [x, y, z]
    data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def gen_robot_camera_dataset(gene, *, n: int = 300, seed: int = 0):
    """Render the robot's OWN robot_cam over n goal placements; label = the goal's bearing in the robot frame
    (normalized). Returns (images[N,16,16,3], bearings[N,1]). The dataset the encoder learns THIS robot's camera on."""
    import mujoco
    import numpy as np

    from virturoid.services.camera_perception import _inject_target, render_px_for_camera, robot_camera_part
    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.vision_encoder import to_encoder_input

    part = robot_camera_part(gene)
    px = render_px_for_camera(part)
    base_xml = gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
    rng = np.random.default_rng(seed)
    half = _cam_half_fov(gene)
    imgs, ys = [], []
    for _ in range(n):
        gx = float(rng.uniform(0.9, 2.6))
        bearing = float(rng.uniform(-half, half))               # sample a bearing INSIDE the real FOV -> always in view
        gy = gx * math.tan(bearing)
        model = mujoco.MjModel.from_xml_string(
            _inject_target(base_xml, (gx, gy), GOAL_RGB, size=_GOAL_SIZE, z=_GOAL_Z))
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam") < 0:
            raise ValueError("this robot has no functional robot_cam — it carries no camera to train vision on")
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        rr = mujoco.Renderer(model, height=px, width=px); rr.update_scene(data, camera="robot_cam")
        img = rr.render().copy(); rr.close()
        imgs.append(np.asarray(to_encoder_input(img), np.float32))
        ys.append([bearing / half])                             # normalize by the real half-FOV -> label in ~[-1, 1]
    return np.asarray(imgs, np.float32), np.asarray(ys, np.float32)


_ENC_SEED = 0    # the frozen tiny-CNN's init; deterministic so the same robot trains the same vision


def _encode_batch(images):
    """Frozen tiny 2-conv CNN features for a batch of (16,16,3) images -> (N, 64). The convs are random+fixed
    (the 'random conv features' regime — see percept_pretrain); we FIT a linear readout over them, which is the
    robust, deterministic in-house training that converges where end-to-end conv backprop stalls on these frames."""
    import numpy as np

    from virturoid.services.vision_encoder import TinyVisionEncoder
    enc = TinyVisionEncoder(seed=_ENC_SEED)
    return np.stack([np.asarray(enc.encode(np.asarray(im, np.float32))) for im in images])


def train_robot_vision(gene, *, n: int = 300, epochs: int = 0, seed: int = 0, ridge: float = 1e-2) -> dict:
    """Train the CV to read the goal's bearing from THIS robot's OWN camera: encode each frame with the tiny CNN,
    then FIT a closed-form ridge readout (feature -> bearing). Deterministic + in-house. Returns the learned model
    + a held-out error vs a predict-the-mean baseline. (``epochs`` kept for signature compatibility; the ridge
    fit is closed-form.)"""
    import numpy as np

    x, y = gen_robot_camera_dataset(gene, n=n, seed=seed)
    feats = _encode_batch(x)
    y = y[:, 0]
    ntr = max(8, int(n * 0.8))
    F, b = feats[:ntr], y[:ntr]
    Fa = np.hstack([F, np.ones((len(F), 1))])                   # bias column
    w = np.linalg.solve(Fa.T @ Fa + ridge * np.eye(Fa.shape[1]), Fa.T @ b)
    Fte = np.hstack([feats[ntr:], np.ones((len(feats) - ntr, 1))])
    pred = Fte @ w
    test_mae = float(np.mean(np.abs(pred - y[ntr:])))
    baseline_mae = float(np.mean(np.abs(y[ntr:] - y[:ntr].mean())))   # predict-the-mean
    half = _cam_half_fov(gene)
    return {"readout": w, "enc_seed": _ENC_SEED, "n": n,
            "test_mae": round(test_mae, 4), "baseline_mae": round(baseline_mae, 4),
            "improvement_x": round(baseline_mae / test_mae, 2) if test_mae > 0 else None,
            "test_mae_deg": round(test_mae * half * 180 / math.pi, 2),
            "learned": bool(test_mae < 0.5 * baseline_mae)}


def banked_vision(gene):
    """The learned vision policy BANKED on this robot by ``train_camera_policy`` (or None). Deploys with the robot."""
    return (getattr(gene, "metadata", None) or {}).get("vision_policy")


def learned_bearing_fn(gene, model=None, *, goal_rgb=GOAL_RGB):
    """A ``(bearing, frac)`` detector driven by the robot's BANKED learned readout over its OWN camera — so the
    trained CV is actually CONSUMED downstream (verify_robot's vision), not just measured once. ``bearing`` is the
    predicted normalized frame offset in [-1, 1]; ``frac`` is a goal-colour visibility gate (the readout was only
    fitted on in-view goals). Returns None if the robot has no banked vision — the caller falls back to the detector."""
    import numpy as np

    from virturoid.services.vision_encoder import TinyVisionEncoder, to_encoder_input
    m = model or banked_vision(gene)
    if not m or "readout" not in m:
        return None
    enc = TinyVisionEncoder(seed=int(m.get("enc_seed", _ENC_SEED)))
    w = np.asarray(m["readout"], np.float32)
    tgt = np.asarray(goal_rgb, np.float32)

    def detect(img):
        enc_in = np.asarray(to_encoder_input(img), np.float32)   # (16,16,3) in [0,1]
        frac = float((np.abs(enc_in - tgt).max(axis=2) < 0.28).mean())
        if frac < 0.0015:
            return None, frac
        f = np.asarray(enc.encode(enc_in), np.float32)
        bearing = float(np.append(f, 1.0) @ w)                   # learned readout -> normalized bearing
        return float(np.clip(bearing, -1.0, 1.0)), frac

    return detect


def robot_vision_nav(gene, vision_model, *, goal=(3.0, 0.0), goal_rgb=GOAL_RGB, start=(0.0, 0.0), start_yaw=0.0,
                     horizon: int = 400, v: float = 0.03, turn_gain: float = 1.5, search: float = 0.08,
                     reach_m: float = 0.7) -> dict:
    """Glide the BUILT robot to a goal it can SEE using ONLY the CV read of its OWN robot_cam (a trained tiny-CNN
    feature readout). The steering is the learned vision; the base is moved kinematically (a perception test, not a
    gait test). Returns reached / final_distance_m / steps / saw_goal."""
    import mujoco
    import numpy as np

    from virturoid.services.camera_perception import _inject_target, render_px_for_camera, robot_camera_part
    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.vision_encoder import TinyVisionEncoder, to_encoder_input

    part = robot_camera_part(gene)
    px = render_px_for_camera(part)
    base_xml = gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
    xml = _inject_target(base_xml, (float(goal[0]), float(goal[1])), goal_rgb, size=_GOAL_SIZE, z=_GOAL_Z)
    model = mujoco.MjModel.from_xml_string(xml)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_cam") < 0:
        return {"reached": False, "error": "no robot_cam"}
    data = mujoco.MjData(model)
    adr = _base_qadr(model)
    z0 = float(standing_spawn_z(gene))
    enc = TinyVisionEncoder(seed=int(vision_model["enc_seed"]))
    w = np.asarray(vision_model["readout"], np.float32)         # [feature weights..., bias]

    def perceive(img16):                                        # the learned CV read: tiny-CNN features -> bearing
        f = np.asarray(enc.encode(np.asarray(img16, np.float32)), np.float32)
        return float(np.append(f, 1.0) @ w)

    rr = mujoco.Renderer(model, height=px, width=px)
    half = _cam_half_fov(gene)

    x, y, yaw = float(start[0]), float(start[1]), float(start_yaw)
    gx, gy = float(goal[0]), float(goal[1])
    reached, saw = False, False
    step = 0
    try:
        for step in range(horizon):
            _set_base(model, data, adr, x, y, z0, yaw)
            rr.update_scene(data, camera="robot_cam")
            img = np.asarray(to_encoder_input(rr.render().copy()), np.float32)
            # visible? a trivial goal-colour gate (the regressor only saw in-view goals)
            r, g, b = img[..., 0], img[..., 1], img[..., 2]
            frac = float(np.mean((g > 0.45) & (r < 0.55) & (b < 0.55)))
            if frac > 0.0015:
                saw = True
                bearing = perceive(img) * half                  # CV read -> the goal's bearing in radians
                yaw += float(np.clip(turn_gain * bearing, -0.25, 0.25))   # steer toward the goal's bearing
                x += v * math.cos(yaw); y += v * math.sin(yaw)
            else:
                yaw += search                                   # rotate to search
            if math.hypot(gx - x, gy - y) < reach_m:
                reached = True
                break
    finally:
        rr.close()
    return {"reached": bool(reached), "saw_goal": bool(saw), "steps": step + 1,
            "final_distance_m": round(math.hypot(gx - x, gy - y), 3), "perception": "learned_onboard_camera"}


def evaluate_robot_vision(gene, *, n_train: int = 280, seed: int = 0) -> dict:
    """In-house CV training on the BUILT robot: train the tiny-CNN readout on the robot's OWN camera to read a
    target's bearing, and report the HELD-OUT accuracy vs a predict-the-mean baseline (an untrained readout). This
    is the un-gameable proof that the CV is trained, on THIS robot's camera, to actually perceive — the honest
    'CV is part of building a camera robot'. (Closed-loop nav is a separate utility; a steer-to-bearing glide is a
    weak discriminator because the search behaviour can reach a goal without good vision.)"""
    from virturoid.services.camera_perception import robot_camera_part, render_px_for_camera
    from virturoid.services.sensor_provenance import camera_is_ours_to_add
    part = robot_camera_part(gene)
    if part is None:
        allowed, why = camera_is_ours_to_add(gene)
        return {"has_camera": False, "trained_in_house": False,
                "note": (why if not allowed else "this robot carries no camera — nothing to train vision on")}
    tr = train_robot_vision(gene, n=n_train, seed=seed)
    return {"has_camera": True, "camera_part": part.name, "render_px": render_px_for_camera(part),
            "bearing_mae_deg": tr["test_mae_deg"], "baseline_norm": tr["baseline_mae"],
            "improvement_x": tr["improvement_x"], "learned_perception": tr["learned"],
            "trained_in_house": True, "method": "tiny-CNN features + fitted readout on the robot's own camera"}
