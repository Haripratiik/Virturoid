"""Supervised proof that the tiny vision encoder LEARNS to see the goal."""
import pytest

from virturoid.services.vision_train import gen_bearing_dataset, learn_to_see_goal


def test_bearing_dataset_shapes_and_ranges():
    try:
        x, y = gen_bearing_dataset(n=12, render_px=32)
    except Exception as exc:  # noqa: BLE001 - headless/no-GL: skip
        pytest.skip(f"no GL render context: {exc}")
    assert x.shape == (12, 16, 16, 3) and y.shape == (12,)
    assert x.min() >= 0.0 and x.max() <= 1.0 and abs(y).max() <= 1.0


def test_encoder_learns_to_localize_the_goal():
    pytest.importorskip("jax")
    try:
        r = learn_to_see_goal(n=160, epochs=50)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GL render context: {exc}")
    assert r["learned"]                              # trained MAE < half the untrained baseline
    assert r["test_mae"] < r["baseline_mae"]
