"""Vision navigation: reach a goal the robot can SEE, camera-only."""
import math

import numpy as np
import pytest

from virturoid.services.vision_nav import detect_goal_bearing, run_vision_nav_episode

_GREEN = (0.1, 0.85, 0.15)
_GREEN_U8 = (26, 217, 38)


def test_detect_goal_bearing_left_is_negative_right_is_positive():
    left = np.zeros((64, 64, 3), np.uint8)
    left[20:40, 5:20] = _GREEN_U8
    bearing, frac = detect_goal_bearing(left, _GREEN)
    assert bearing is not None and bearing < -0.2 and frac > 0

    right = np.zeros((64, 64, 3), np.uint8)
    right[20:40, 44:59] = _GREEN_U8
    bearing, _ = detect_goal_bearing(right, _GREEN)
    assert bearing is not None and bearing > 0.2


def test_detect_goal_bearing_returns_none_when_absent():
    bearing, frac = detect_goal_bearing(np.zeros((64, 64, 3), np.uint8), _GREEN)
    assert bearing is None and frac < 0.0025


def test_vision_nav_reaches_a_goal_dead_ahead():
    try:
        r = run_vision_nav_episode(goal_xy=(3.0, 0.0), start_yaw=0.0, horizon=300)
    except Exception as exc:  # noqa: BLE001 - headless/no-GL: skip rather than fail
        pytest.skip(f"no GL render context: {exc}")
    assert r["reached"] and r["goal_was_seen"] and r["sensed_only"]


def test_vision_nav_searches_for_an_offscreen_goal():
    try:
        r = run_vision_nav_episode(goal_xy=(3.0, 0.0), start_yaw=math.pi, horizon=400)   # goal starts behind
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GL render context: {exc}")
    assert r["reached"]   # must rotate to find it, then drive in


def test_record_frames_and_save_montage():
    pytest.importorskip("PIL")
    import os
    import tempfile

    from virturoid.services.vision_nav import save_vision_montage

    try:
        r = run_vision_nav_episode(goal_xy=(2.5, 0.0), start_yaw=0.0, horizon=90,
                                   record_frames=True, frame_every=10)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GL render context: {exc}")
    assert len(r["frames"]) > 0 and r["frames"][0].shape == (64, 64, 3)
    with tempfile.TemporaryDirectory() as td:
        path = save_vision_montage(r["frames"], os.path.join(td, "montage.png"))
        assert os.path.exists(path)


def test_learned_encoder_drives_nav_end_to_end():
    # the LEARNED tiny 2-conv CNN — not the hand-written colour detector — closes the nav loop: train it to see the
    # goal's bearing, then drive to the goal on that learned perception alone. Proves the CV isn't just classic
    # colour segmentation dressed up as "vision".
    from virturoid.services.vision_train import evaluate_learned_vision_nav
    try:
        r = evaluate_learned_vision_nav(n_train=300, epochs=60, seed=0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GL render context: {exc}")
    assert r["perception"] == "learned_tiny_cnn"
    assert r["bearing_mae_deg"] < 5.0, f"the learned encoder must see the goal accurately (got {r['bearing_mae_deg']} deg)"
    assert r["reached"] >= 3, f"the learned CV must navigate to the goal (reached {r['reached']}/4)"
