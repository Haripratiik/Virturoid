"""Train the tiny vision encoder to SEE -- supervised proof that the small CV model learns.

The GPU vision-RL leg (the Squint recipe: SAC over pixels, 1024 envs) is gated on a from-source Madrona-MJX
build. But the core claim -- a tiny 2-conv encoder LEARNS useful perception from raw pixels -- is provable on
CPU right now, and certain to converge: a supervised regression where the encoder maps a 16x16 camera image to
the goal's bearing (its horizontal angle). The trained encoder then drops into vision_nav in place of the
hand-written color detector. This is the differentiable inner step a reward-driven RL loop would otherwise do
end-to-end. See [[vision-cv-decision]], [[make-it-learn-dont-teach-it]].
"""
from __future__ import annotations

import math

_FOV_HALF = 0.62   # ~half of the camera's 72deg fovy, in radians -- goals are sampled within view


def _bearing_scene(render_px: int) -> str:
    return f"""<mujoco>
  <visual><global offwidth="{render_px}" offheight="{render_px}"/>
    <headlight ambient="0.6 0.6 0.6" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/></visual>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="30 30 0.1" rgba="0.45 0.45 0.5 1"/>
    <camera name="cam" pos="0 0 0.4" xyaxes="0 -1 0 0 0 1" fovy="72"/>
    <body name="goal" mocap="true" pos="3 0 0.35">
      <geom type="box" size="0.25 0.25 0.35" rgba="0.1 0.85 0.15 1"/>
    </body>
  </worldbody>
</mujoco>"""


def gen_bearing_dataset(n: int = 400, *, seed: int = 0, render_px: int = 32):
    """Render n images of the goal at a random bearing/distance -> (X images [n,16,16,3], Y bearings [n] in [-1,1])."""
    import mujoco
    import numpy as np

    from virturoid.services.vision_encoder import to_encoder_input

    model = mujoco.MjModel.from_xml_string(_bearing_scene(render_px))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=render_px, width=render_px)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    try:
        for _ in range(n):
            theta = float(rng.uniform(-_FOV_HALF, _FOV_HALF))
            dist = float(rng.uniform(1.5, 4.0))
            data.mocap_pos[0] = [dist * math.cos(theta), dist * math.sin(theta), 0.35]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="cam")
            xs.append(to_encoder_input(renderer.render()))
            # nav image-x convention: the camera's right axis is world -Y, so world +Y (theta>0) lands image-LEFT
            # (negative bearing). Labelling this way lets the learned predictor drive vision_nav directly.
            ys.append(-theta / _FOV_HALF)
    finally:
        renderer.close()
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def train_bearing_predictor(x, y, *, epochs: int = 80, lr: float = 3e-3, seed: int = 0):
    """Fit encoder + a linear head to predict the goal bearing from pixels (JAX + a small hand-rolled Adam).

    Returns (params, test_mae, baseline_mae) -- baseline is the untrained encoder, so test_mae << baseline_mae
    is the proof the encoder LEARNED to localize the goal."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from virturoid.services.vision_encoder import TinyVisionEncoder, jax_encode, to_jax_params

    n = len(x)
    ntr = int(n * 0.8)
    xtr, ytr, xte, yte = jnp.asarray(x[:ntr]), jnp.asarray(y[:ntr]), jnp.asarray(x[ntr:]), jnp.asarray(y[ntr:])

    enc_keys = ("W1", "b1", "W2", "b2", "Wf", "bf")
    params = {**to_jax_params(TinyVisionEncoder(seed=seed)),
              "w_head": jnp.zeros((64,), jnp.float32), "b_head": jnp.array(0.0, jnp.float32)}

    def predict(p, batch):
        feats = jax.vmap(lambda im: jax_encode({k: p[k] for k in enc_keys}, im))(batch)
        return feats @ p["w_head"] + p["b_head"]

    def mse(p, batch, target):
        return jnp.mean((predict(p, batch) - target) ** 2)

    def mae(p, batch, target):
        return float(jnp.mean(jnp.abs(predict(p, batch) - target)))

    baseline_mae = mae(params, xte, yte)                       # untrained encoder + zero head

    grad_fn = jax.jit(jax.value_and_grad(mse))
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        _, grads = grad_fn(params, xtr, ytr)
        for k in params:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mhat = m[k] / (1 - b1 ** t)
            vhat = v[k] / (1 - b2 ** t)
            params[k] = params[k] - lr * mhat / (jnp.sqrt(vhat) + eps)

    return params, mae(params, xte, yte), baseline_mae


def learn_to_see_goal(*, n: int = 400, epochs: int = 80, seed: int = 0) -> dict:
    """End-to-end: render data, train, report. test_mae (normalized bearing units, -1..1) << baseline = learned."""
    params, report = train_goal_seer(n=n, epochs=epochs, seed=seed)
    return report


def train_goal_seer(*, n: int = 400, epochs: int = 80, seed: int = 0):
    """Train and RETURN (params, report) so the learned perception can drive vision_nav (not just self-report)."""
    x, y = gen_bearing_dataset(n, seed=seed)
    params, test_mae, baseline_mae = train_bearing_predictor(x, y, epochs=epochs, seed=seed)
    report = {"n": n, "epochs": epochs, "test_mae": round(test_mae, 4), "baseline_mae": round(baseline_mae, 4),
              "improvement_x": round(baseline_mae / test_mae, 2) if test_mae > 0 else None,
              "test_mae_deg": round(test_mae * _FOV_HALF * 180 / math.pi, 2),
              "learned": test_mae < 0.5 * baseline_mae}
    return params, report


def make_learned_bearing_fn(params: dict, goal_rgb=(0.1, 0.85, 0.15)):
    """Wrap trained params as a ``bearing_fn(frame) -> (bearing|None, frac)`` for vision_nav.run_vision_nav_episode.

    The BEARING (the precise 'where') is the learned encoder's prediction; visibility (is any goal colour present
    at all?) is a trivial colour-fraction gate, since the regressor only ever saw in-view goals. This is what
    lets the robot navigate end-to-end on LEARNED perception."""
    import jax
    import numpy as np

    from virturoid.services.vision_encoder import jax_encode, to_encoder_input

    enc = {k: params[k] for k in ("W1", "b1", "W2", "b2", "Wf", "bf")}
    w_head, b_head = params["w_head"], params["b_head"]
    predict = jax.jit(lambda enc_in: jax_encode(enc, enc_in) @ w_head + b_head)
    target = np.asarray(goal_rgb, np.float32)

    def bearing_fn(frame):
        frac = float((np.abs(np.asarray(frame, np.float32) / 255.0 - target).max(axis=2) < 0.28).mean())
        if frac < 0.0025:
            return None, frac                                    # no goal colour in view -> search
        return float(np.clip(float(predict(to_encoder_input(frame))), -1.0, 1.0)), frac

    return bearing_fn
