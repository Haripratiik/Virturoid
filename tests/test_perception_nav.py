"""Perception (range-sensing) navigation: the robot reaches a goal in an obstacle field it can only SENSE."""
import mujoco

from virturoid.services.morphology_composer import compose_robot
from virturoid.services.perception_nav import (
    build_perception_scene, evaluate_perception_nav, reactive_heading, run_perception_nav_episode,
)


def test_reactive_heading_aims_at_goal_when_clear():
    # all rays open -> head straight for the goal
    assert abs(reactive_heading(0.0, [3.0] * 8)) < 0.1


def test_reactive_heading_steers_off_a_blocked_goal_direction():
    # obstacle straight ahead (ray 0) while the goal is straight ahead -> must deviate, not head into it
    assert abs(reactive_heading(0.0, [0.3] + [3.0] * 7)) > 0.3


def _mobile():
    return compose_robot("a mobile robot that delivers packages", llm=None)


def test_perception_nav_reaches_a_clear_goal_sensed_only():
    model = mujoco.MjModel.from_xml_string(build_perception_scene(_mobile(), [], (3.0, 0.0)))
    res = run_perception_nav_episode(model, (3.0, 0.0), horizon=5000)
    assert res["sensed_only"] is True
    assert res["reached"], res


def test_perception_nav_rounds_a_sensed_obstacle():
    # a single obstacle directly between robot and goal -- the hysteretic side-commit must round it
    model = mujoco.MjModel.from_xml_string(build_perception_scene(_mobile(), [(1.5, 0.0)], (3.0, 0.0)))
    res = run_perception_nav_episode(model, (3.0, 0.0), horizon=6000)
    assert res["reached"], res


def test_evaluate_perception_nav_reports_a_reach_rate():
    out = evaluate_perception_nav(_mobile(), n_scenes=3, seed=0)
    assert out["task"] == "perception_navigation"
    assert out["sensed_only"] is True
    assert 0.0 <= out["success_rate"] <= 1.0
    assert len(out["episodes"]) == 3
