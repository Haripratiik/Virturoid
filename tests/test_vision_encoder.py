"""The tiny vision encoder ('small CV model') + camera pipeline."""
import numpy as np
import pytest

from virturoid.services.vision_encoder import TinyVisionEncoder, render_camera, to_encoder_input


def test_to_encoder_input_downsamples_and_keeps_color():
    img = np.zeros((128, 128, 3), np.uint8)
    img[:, :, 0] = 255                                  # all red
    enc = to_encoder_input(img)
    assert enc.shape == (16, 16, 3)
    assert enc[:, :, 0].mean() > 0.9 and enc[:, :, 1].mean() < 0.1
    assert 0.0 <= enc.min() and enc.max() <= 1.0


def test_tiny_vision_encoder_is_small_and_bounded():
    enc = TinyVisionEncoder()
    assert enc.n_params < 50000                         # genuinely tiny (the Squint recipe -- small = train more)
    feat = enc.encode(np.random.default_rng(0).random((16, 16, 3)))
    assert feat.shape == (64,)
    assert np.all(np.abs(feat) <= 1.0)                  # tanh-bounded


def test_encoder_is_deterministic_for_a_given_seed():
    img = np.random.default_rng(1).random((16, 16, 3))
    a = TinyVisionEncoder(seed=7).encode(img)
    b = TinyVisionEncoder(seed=7).encode(img)
    assert np.allclose(a, b)


def test_render_camera_sees_a_colored_object_if_gl_available():
    import mujoco

    xml = ('<mujoco><visual><global offwidth="128" offheight="128"/></visual><worldbody>'
           '<light pos="0 -1 4" dir="0 0.3 -1"/><camera name="cam" pos="0 -2.5 1.1" xyaxes="1 0 0 0 0.5 1"/>'
           '<geom type="plane" size="5 5 0.1" rgba="0.4 0.4 0.4 1"/>'
           '<geom type="box" pos="0 0 0.4" size="0.35 0.35 0.4" rgba="0.9 0.15 0.15 1"/></worldbody></mujoco>')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        img = render_camera(model, data, "cam")
    except Exception as exc:  # noqa: BLE001 - headless/no-GL environments can't render; skip rather than fail
        pytest.skip(f"no GL render context: {exc}")
    assert img.shape == (128, 128, 3)
    assert int(((img[:, :, 0] > 120) & (img[:, :, 1] < 90)).sum()) > 100   # the red box is visible to the camera


def test_jax_encoder_matches_numpy_is_trainable_and_batches():
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from virturoid.services.vision_encoder import jax_encode, to_jax_params

    enc = TinyVisionEncoder(seed=3)
    img = np.random.default_rng(5).random((16, 16, 3)).astype(np.float32)
    params = to_jax_params(enc)
    assert np.abs(enc.encode(img) - np.asarray(jax_encode(params, img))).max() < 1e-4      # JAX == NumPy
    grad = jax.grad(lambda p, im: jnp.mean(jax_encode(p, im) ** 2))(params, img)
    assert float(sum(jnp.sum(v ** 2) for v in grad.values())) > 0.0                         # differentiable
    batched = jax.jit(jax.vmap(jax_encode, in_axes=(None, 0)))(params, np.repeat(img[None], 8, axis=0))
    assert np.asarray(batched).shape == (8, 64)                                             # vmap + jit
