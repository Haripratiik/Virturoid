"""WS-F.1 — the sim-backend parity harness: sign-flip as a failing test (master_plan_v6 WS-F)."""
from __future__ import annotations

import numpy as np

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import parity_harness as PH


def _tiny():
    """A minimal 1-actuator body that compiles + steps fast."""
    return RobotGene(id="tiny", species="t.tiny", robot_class="manipulator", base_mount="table",
                     end_effector_type="none",
                     segments=[GeneSegment(name="base", parent=None, joint_type=None, length_m=0.2, radius_m=0.04,
                                           mass_kg=0.5),
                               GeneSegment(name="link", parent="base", joint_type="revolute", length_m=0.2,
                                           radius_m=0.03, mass_kg=0.2, joint_lower=-1.5, joint_upper=1.5,
                                           actuator_torque_nm=5.0, is_end_effector=True)])


def test_fixed_ctrl_sequence_is_deterministic():
    a = PH.fixed_ctrl_sequence(3, 50, seed=7)
    b = PH.fixed_ctrl_sequence(3, 50, seed=7)
    assert a.shape == (50, 3) and np.array_equal(a, b)
    assert not np.array_equal(a, PH.fixed_ctrl_sequence(3, 50, seed=8))   # different seed -> different control


def test_cpu_rollout_is_deterministic():
    g = _tiny()
    ctrl = PH.fixed_ctrl_sequence(1, 120, seed=1)
    r1 = PH.rollout_cpu_mujoco(g, ctrl)
    r2 = PH.rollout_cpu_mujoco(g, ctrl)
    assert r1["finite"] and np.array_equal(r1["qpos_traj"], r2["qpos_traj"])   # same engine + control = bit-parity
    assert r1["forward"] == r2["forward"]


def test_compare_agrees_cpu_vs_cpu():
    g = _tiny()
    ctrl = PH.fixed_ctrl_sequence(1, 120, seed=2)
    a = PH.rollout_cpu_mujoco(g, ctrl); b = PH.rollout_cpu_mujoco(g, ctrl)
    c = PH.compare(a, b)
    assert c["parity_ok"] and not c["sign_flip"] and c["max_state_abs_diff"] == 0.0


def test_compare_detects_the_sign_flip():
    """The WS-F bug: identical control, opposite forward travel -> flagged, parity fails."""
    a = {"backend": "cpu_mujoco", "qpos_traj": np.zeros((10, 3)), "forward": 0.535}
    b = {"backend": "mjx_single", "qpos_traj": np.zeros((10, 3)), "forward": -0.138}   # the diagnosed numbers
    c = PH.compare(a, b)
    assert c["sign_flip"] and not c["parity_ok"]
    assert "sign flip" in c["note"].lower()


def test_parity_check_cpu_only_is_not_established():
    """MJX is unavailable on this box -> recorded honestly, and cross-engine parity CANNOT be claimed alone."""
    rep = PH.parity_check(_tiny(), backends=("cpu_mujoco", "mjx_single"), steps=80).to_dict()
    assert rep["backends_run"] == ["cpu_mujoco"]
    assert "mjx_single" in rep["backends_unavailable"]
    assert rep["parity_ok"] is False and "NOT ESTABLISHED" in rep["gate"]


def test_parity_check_two_agreeing_backends_is_green(monkeypatch):
    """With two agreeing backends (a CPU clone stands in for the GPU box's MJX), parity is GREEN."""
    monkeypatch.setitem(PH.BACKENDS, "cpu_clone", PH.rollout_cpu_mujoco)
    rep = PH.parity_check(_tiny(), backends=("cpu_mujoco", "cpu_clone"), steps=80)
    assert rep.parity_ok and rep.comparisons and rep.comparisons[0]["parity_ok"]
    assert set(rep.backends_run) == {"cpu_mujoco", "cpu_clone"}
