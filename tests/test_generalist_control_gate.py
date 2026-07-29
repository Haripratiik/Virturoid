import importlib.util

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _chain_model(n_joints: int) -> str:
    bodies = []
    actuators = []
    for index in range(n_joints):
        bodies.append(
            f'<body name="b{index}" pos="0 0 .12"><joint name="j{index}" type="hinge" '
            f'axis="0 1 0" range="-1 1"/><geom type="capsule" size=".02 .06"/>'
        )
        actuators.append(f'<position name="a{index}" joint="j{index}" kp="20" ctrllimited="true" ctrlrange="-12 12"/>')
    return (
        '<mujoco><compiler angle="radian"/><worldbody><body name="root">'
        + "".join(bodies)
        + "</body>" * n_joints
        + "</body></worldbody><actuator>"
        + "".join(actuators)
        + "</actuator></mujoco>"
    )


@pytest.mark.skipif(not _MUJOCO, reason="MuJoCo not installed")
def test_four_body_gate_validates_tokens_and_cpu_throughput():
    from virturoid.services.generalist_control_gate import run_generalist_control_gate

    report = run_generalist_control_gate(
        [_chain_model(n) for n in (1, 2, 3, 4)],
        steps=30,
        min_steps_per_second=1.0,
        require_local_gpu=False,
    )
    assert report.passed, report.blockers
    assert report.distinct_topologies == 4
    assert {body.feature_dim for body in report.bodies} == {27}
    assert {body.joint_description_dim for body in report.bodies} == {14}


@pytest.mark.skipif(not _MUJOCO, reason="MuJoCo not installed")
def test_gate_refuses_an_under_sized_research_corpus():
    from virturoid.services.generalist_control_gate import run_generalist_control_gate

    report = run_generalist_control_gate(
        [_chain_model(n) for n in (1, 2, 3)],
        steps=10,
        min_steps_per_second=1.0,
        require_local_gpu=False,
    )
    assert not report.passed
    assert "needs_at_least_4_bodies" in report.blockers
