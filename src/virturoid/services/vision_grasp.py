"""Vision-guided grasp: the tiny CV encoder is TRAINED to locate a box from an overhead camera, then the arm
grasps the box at the LEARNED position -- never handed the true coordinates.

This is the honest 'is it actually training?' demo. The box sits at a random (x,y); an overhead camera renders
it; the tiny 2-conv encoder (the 'small CV model', ~24K params) predicts (x,y) from raw pixels; and the arm
AIMS at that prediction (grasp_eval.aim_xy) while the box really is at the true spot. Proof of training:
- UNTRAINED encoder -> predicts the wrong place -> the arm misses (no contact).
- TRAINED encoder   -> predicts within ~1cm    -> the arm grasps and lifts.
Nothing is premade: the grasp target comes from the network's pixels-in, (x,y)-out prediction, not the sim
state. See [[vision-cv-decision]], [[demo-arm-real-contact-grasp]].
"""
from __future__ import annotations

from functools import lru_cache

# the workspace the composed arm reliably grasps (proven 6/6 by grasp_eval): gx forward of the base, gy lateral
GX0, GX1 = 0.40, 0.48
GY0, GY1 = -0.06, 0.06
_GXC, _GXR = (GX0 + GX1) / 2, (GX1 - GX0) / 2
_GYC, _GYR = (GY0 + GY1) / 2, (GY1 - GY0) / 2
_BOX_Z = 0.05
_ENC_KEYS = ("W1", "b1", "W2", "b2", "Wf", "bf")


def _norm(gx, gy):
    return (gx - _GXC) / _GXR, (gy - _GYC) / _GYR


def _denorm(nx, ny):
    return _GXC + nx * _GXR, _GYC + ny * _GYR


@lru_cache(maxsize=2)
def build_arm(prompt: str = "grasp and lift a box on a table"):
    """The composed grasp arm (offline composer -> a real 6-DOF manipulator that grasps 6/6 at true positions)."""
    from virturoid.services.morphology_composer import compose_robot

    return compose_robot(prompt)


def _render_model(gene, render_px: int):
    """Compile a render model: arm + a red box (free body) + an overhead camera. The box recolors red so the
    camera (and the encoder) have an unambiguous target against the bright table."""
    import mujoco

    from virturoid.schemas.scenes import SceneObject
    from virturoid.services.gene_compiler import compile_gene_with_scene

    cube = SceneObject("box", "cube", (_GXC, 0.0, _BOX_Z, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=1.0, scale=1.0)
    cam = f'<camera name="overhead" pos="{_GXC} 0 0.40" xyaxes="1 0 0 0 1 0" fovy="42"/>'
    xml = compile_gene_with_scene(gene, [cube]).replace("</worldbody>", cam + "</worldbody>")
    mj = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(mj)
    mujoco.mj_resetData(mj, data)
    bg = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box")
    mj.geom_rgba[bg] = [0.95, 0.1, 0.1, 1]
    mj.geom_matid[bg] = -1
    bj = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
    return mj, data, int(mj.jnt_qposadr[bj]), mujoco.Renderer(mj, height=render_px, width=render_px)


def _render_at(mj, data, box_qadr, renderer, gx, gy):
    """Place the box at (gx, gy) (arm at rest), render the overhead camera, return the 16x16 encoder input."""
    import mujoco

    from virturoid.services.vision_encoder import to_encoder_input

    mujoco.mj_resetData(mj, data)
    data.qpos[box_qadr], data.qpos[box_qadr + 1], data.qpos[box_qadr + 2] = gx, gy, _BOX_Z
    mujoco.mj_forward(mj, data)
    renderer.update_scene(data, camera="overhead")
    return to_encoder_input(renderer.render())


def gen_box_dataset(gene, n: int = 400, *, seed: int = 0, render_px: int = 64):
    """Render n overhead views of the box at random workspace (x,y) -> (X [n,16,16,3], Y [n,2] normalized xy)."""
    import numpy as np

    mj, data, bq, renderer = _render_model(gene, render_px)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    try:
        for _ in range(n):
            gx, gy = float(rng.uniform(GX0, GX1)), float(rng.uniform(GY0, GY1))
            xs.append(_render_at(mj, data, bq, renderer, gx, gy))
            ys.append(_norm(gx, gy))
    finally:
        renderer.close()
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def train_box_locator(x, y, *, epochs: int = 120, lr: float = 3e-3, seed: int = 0):
    """Fit encoder + a 2-output head to predict (normalized x, y) from the overhead image. Returns
    (params, test_mae, baseline_mae) -- baseline is the untrained encoder, so test_mae << baseline = learned."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from virturoid.services.vision_encoder import TinyVisionEncoder, jax_encode, to_jax_params

    n = len(x)
    ntr = int(n * 0.8)
    xtr, ytr, xte, yte = jnp.asarray(x[:ntr]), jnp.asarray(y[:ntr]), jnp.asarray(x[ntr:]), jnp.asarray(y[ntr:])
    params = {**to_jax_params(TinyVisionEncoder(seed=seed)),
              "Wh": jnp.zeros((64, 2), jnp.float32), "bh": jnp.zeros((2,), jnp.float32)}

    def predict(p, batch):
        feats = jax.vmap(lambda im: jax_encode({k: p[k] for k in _ENC_KEYS}, im))(batch)
        return feats @ p["Wh"] + p["bh"]

    def mse(p, batch, target):
        return jnp.mean((predict(p, batch) - target) ** 2)

    def mae(p, batch, target):
        return float(jnp.mean(jnp.abs(predict(p, batch) - target)))

    baseline_mae = mae(params, xte, yte)
    grad_fn = jax.jit(jax.value_and_grad(mse))
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(val) for k, val in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        _, grads = grad_fn(params, xtr, ytr)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            params[k] = params[k] - lr * (m[k] / (1 - b1 ** t)) / (jnp.sqrt(v[k] / (1 - b2 ** t)) + eps)
    return params, mae(params, xte, yte), baseline_mae


def make_box_locator(params):
    """params -> locate(image16) -> (gx, gy) in WORLD coords. This is the perception the arm grasps from."""
    import jax
    import numpy as np

    from virturoid.services.vision_encoder import jax_encode

    enc = {k: params[k] for k in _ENC_KEYS}
    wh, bh = params["Wh"], params["bh"]
    fwd = jax.jit(lambda im: jax_encode(enc, im) @ wh + bh)

    def locate(img16):
        nx, ny = np.asarray(fwd(img16))
        return _denorm(float(nx), float(ny))

    return locate


def evaluate_vision_grasp(gene, locator, *, n: int = 8, seed: int = 1, render_px: int = 64):
    """For n random box placements: SEE (render -> locate) then GRASP at the predicted (x,y) while the box is at
    the true (x,y). Returns grasp_success_rate + mean localisation error -- the end-to-end vision->grasp result."""
    import numpy as np

    from virturoid.services.grasp_eval import grasp_lift_attempt

    mj, data, bq, renderer = _render_model(gene, render_px)
    rng = np.random.default_rng(seed)
    results = []
    try:
        for _ in range(n):
            gx, gy = float(rng.uniform(GX0, GX1)), float(rng.uniform(GY0, GY1))
            img = _render_at(mj, data, bq, renderer, gx, gy)
            gx_hat, gy_hat = locator(img)
            grasp = grasp_lift_attempt(gene, gx, gy, aim_xy=(gx_hat, gy_hat))     # box true; arm aims at prediction
            results.append({"true": (round(gx, 3), round(gy, 3)), "pred": (round(gx_hat, 3), round(gy_hat, 3)),
                            "loc_err_m": round(float(np.hypot(gx - gx_hat, gy - gy_hat)), 4),
                            "grasped": grasp["success"], "lifted": grasp["lifted"]})
    finally:
        renderer.close()
    n = max(1, len(results))
    return {"grasp_success_rate": round(sum(r["grasped"] for r in results) / n, 3),
            "mean_loc_err_m": round(sum(r["loc_err_m"] for r in results) / n, 4),
            "n": len(results), "results": results}


def vision_grasp_once(gene, locator, gx, gy, *, render_px: int = 64, record_frames: bool = False,
                      frame_every: int = 16):
    """SEE the box at (gx, gy) via the overhead camera, LOCATE it with ``locator``, GRASP at the prediction.

    The single-box version behind the demo: returns the grasp result dict (plus ``true``/``pred`` and, if
    ``record_frames``, the side-view ``frames`` of the arm grasping)."""
    import numpy as np

    from virturoid.services.grasp_eval import grasp_lift_attempt

    mj, data, bq, renderer = _render_model(gene, render_px)
    try:
        img = _render_at(mj, data, bq, renderer, gx, gy)
    finally:
        renderer.close()
    gx_hat, gy_hat = locator(img)
    grasp = grasp_lift_attempt(gene, gx, gy, aim_xy=(gx_hat, gy_hat), record_frames=record_frames,
                               frame_every=frame_every)
    grasp["true"] = (round(float(gx), 4), round(float(gy), 4))
    grasp["pred"] = (round(float(gx_hat), 4), round(float(gy_hat), 4))
    grasp["loc_err_m"] = round(float(np.hypot(gx - gx_hat, gy - gy_hat)), 4)
    return grasp
