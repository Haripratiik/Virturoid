"""verify_robot quotes a robustness margin only for the controller it ran.

A margin describes one operating point to the last bit of its floats (``ai_native_tools._fit_margin_for``: a
2.4e-5 change in ``freq`` flips the grounded authored cat from CREDIBLE WALK to ROLL-OVER). Verify used to read
``metadata['gait_fit']`` whenever it ran a tuned controller, and every training door lands its controller over
that record without touching it. MEASURED 2026-10-10: a controller landed through train_held, whose own margin
was 0.001, verified beside the earlier fit's 0.1 and that fit's note. train_held now lands on far more bodies
than it used to (it runs the product's per-body fit), so a create_robot body trained afterwards met this often.

The body is the design schema's quadruped, whose shipped default walks, and every test lands that same default.
The deploy-select then compares two identical rollouts and keeps the landing, so the tuned controller is what
verify runs. The mined flywheel hint is switched off: the bank is shared across the suite, and a hint that
walked further would take the run away from the controller under test.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="verify needs MuJoCo")

#: An operating point other than the shipped default, for the fit record the landing goes over.
_OTHER = {"freq": 2.4, "kp": 70.0}


@pytest.fixture()
def walker(monkeypatch):
    from virturoid.services.agent_design_tools import get_design_schema, submit_design
    monkeypatch.setattr("virturoid.services.ai_native_tools._mined_hint_params", lambda gene: {})
    return submit_design({"graph": get_design_schema({})["examples"]["quadruped"]})["robot_id"]


def _plant_fit(rid: str, params: dict, *, rel: float = 0.1, note: str = "THE EARLIER FIT'S NOTE") -> None:
    """The record create_robot's per-body fit leaves when it adopts ``params``."""
    from virturoid.schemas.gene import RobotGene
    from virturoid.services import session_state as S
    g = RobotGene.from_dict(S.get_robot(rid).to_dict())
    g.metadata = {**(g.metadata or {}), "gait_fit": {"adopted": True, "params": dict(params), "robustness_rel": rel,
                                                      "robustness_note": note, "fragile": False,
                                                      "horizon_steps": 6000}}
    assert S.commit_robot(rid, g, label="a create_robot-style fit")


def _verify(rid: str) -> dict:
    from virturoid.services.agent_tools import call_tool
    v = call_tool("verify_robot", {"robot_id": rid, "mode": "quick"})["result"]
    assert str(v.get("gait_source", "")).startswith("tuned_for_this_body"), v.get("gait_source")
    return v


def test_a_landing_over_an_earlier_fit_reports_its_own_margin(walker):
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    from virturoid.services.trained_controller import apply_trained_gait
    _plant_fit(walker, {**_DEFAULT_GAIT, **_OTHER})
    rep = apply_trained_gait(walker, dict(_DEFAULT_GAIT), door="train_held", credible=True, verdict="CREDIBLE WALK",
                             evidence={"robustness_rel": 0.001, "robustness_note": "THE LANDED POINT'S OWN NOTE"})
    assert rep["applied"], rep
    v = _verify(walker)
    assert v["robustness_rel"] == 0.001 and v["robustness_measured_by"] == "train_held", v
    assert v["robustness_note"] == "THE LANDED POINT'S OWN NOTE", v["robustness_note"]


def test_a_landing_nobody_measured_borrows_no_margin(walker):
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    _plant_fit(walker, {**_DEFAULT_GAIT, **_OTHER})
    env = call_tool("apply_gait", {"robot_id": walker, "params": dict(_DEFAULT_GAIT)})
    assert env["ok"], env
    v = _verify(walker)
    assert "robustness_rel" not in v, v.get("robustness_rel")
    assert "no robustness margin was measured for this controller" in v["robustness_note"], v["robustness_note"]


def test_the_fits_own_controller_still_carries_the_fits_margin(walker):
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    _plant_fit(walker, dict(_DEFAULT_GAIT))
    env = call_tool("apply_gait", {"robot_id": walker, "params": dict(_DEFAULT_GAIT)})
    assert env["ok"], env
    v = _verify(walker)
    assert v["robustness_rel"] == 0.1 and v["robustness_measured_by"] == "gait_fit", v
    assert v["robustness_note"] == "THE EARLIER FIT'S NOTE", v["robustness_note"]
