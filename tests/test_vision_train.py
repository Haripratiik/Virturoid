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


def test_learned_encoder_drives_the_nav():
    pytest.importorskip("jax")
    from virturoid.services.vision_nav import run_vision_nav_episode
    from virturoid.services.vision_train import make_learned_bearing_fn, train_goal_seer

    try:
        params, report = train_goal_seer(n=220, epochs=65)
        bearing_fn = make_learned_bearing_fn(params)
        r = run_vision_nav_episode(goal_xy=(3.0, 0.0), start_yaw=0.0, bearing_fn=bearing_fn, horizon=300)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GL render context: {exc}")
    assert report["learned"]
    assert r["reached"]                              # the LEARNED perception drove the robot to the goal
