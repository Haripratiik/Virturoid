"""``apply_gait`` reported a landing that ``verify_robot`` did not see — and only on the customer's own robot.

MEASURED 2026-08-10 through ``agent_tools.call_tool`` on a REAL MuJoCo-Menagerie Unitree Go2, full mode:

    verify_robot (before)   gait_source flywheel_hint                   forward_m  1.019
    apply_gait              applied: true, gait_source_after "tuned_for_this_body::apply_gait",
                            next: "verify_robot ... now measures THESE parameters"
    verify_robot (after)    gait_source default_crawl                   forward_m  0.119

Two surfaces, each individually correct, contradicting each other with nothing in between. The parameters DID
land — the metadata carried them and verify DID run them — but ``_honest_gait``'s deploy-select preferred
another arm and OVERWROTE ``gait_source`` with that arm's name, which from the outside is indistinguishable
from an apply that never happened. And the number got 9x worse, because the arm the landing was measured
against was the SHIPPED DEFAULT while the arm it actually displaced was the mined FLYWHEEL HINT: a body with
no op-point of its own deploys the hint, and the instant one lands, the hint stops being considered at all.

The composed lane hid it. On a body we composed the applied controller usually wins, so the two surfaces agree
and nothing looks wrong; it took an imported body — where our generic priors are all mediocre and the ranking
is decided by tenths of a metre — to make the swap the common case.

WHAT THESE TESTS PIN
  * the write side verifies its own landing by READ-BACK before claiming ``applied`` (the defect class this
    repo has closed four times: "applied: true when the parameters are not installed");
  * the write side does not promise a ``gait_source`` it does not control;
  * the read side never erases a landed controller — when deploy-select prefers another arm it says so, with
    the landed controller's OWN measured number, that it is still installed, and how to undo it;
  * a landing cannot cost the body the controller it was already deploying.

They go through ``call_tool`` wherever the behaviour is end-to-end, because a unit test on one side of a
disagreement between two surfaces proves nothing about the disagreement.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="the verdict path needs MuJoCo")

_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")

#: In range for every one of ``gait_search``'s five intervals, and measured to travel almost nowhere on both a
#: composed quadruped and an imported Go2 — so the deploy-select rejects it and the disclosure path is exercised.
_LOSER = {"freq": 0.9, "hip_amp": 0.45, "knee_amp": 0.55, "kp": 25.0, "kd": 1.2}
#: Measured to out-travel both the shipped default and the mined hint on the composed dog (0.463 vs 0.384 m).
_WINNER = {"freq": 1.5, "hip_amp": 1.2, "knee_amp": 1.2, "kp": 60.0, "kd": 3.0}


@pytest.fixture()
def held_dog():
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    return S.put_robot(compose_robot("a four legged robot dog", llm=None), prompt="a four legged robot dog")


def _apply(rid: str, params: dict) -> dict:
    from virturoid.services.agent_tools import call_tool
    env = call_tool("apply_gait", {"robot_id": rid, "params": params})
    assert env.get("ok"), env
    return env["result"]


def _verify(rid: str) -> dict:
    from virturoid.services.agent_tools import call_tool
    env = call_tool("verify_robot", {"robot_id": rid, "mode": "quick"})
    assert env.get("ok"), env
    return env["result"]


# ------------------------------------------------------------------ the write side may only claim what it did
def test_apply_gait_confirms_its_landing_by_reading_it_back(held_dog):
    """``commit_robot`` returning True says the store accepted a write, not that the next reader sees it."""
    rep = _apply(held_dog, _WINNER)
    assert rep["applied"] is True, rep
    assert "read-back" in rep["verified_by"]
    from virturoid.services.trained_controller import held_gait
    assert held_gait(held_dog)["params"] == pytest.approx(_WINNER)


def test_a_landing_that_cannot_be_read_back_is_not_reported_as_applied(held_dog, monkeypatch):
    """The whole point of the read-back: when the store lies, this tool must not repeat the lie."""
    from virturoid.services import trained_controller as TC
    monkeypatch.setattr(TC, "held_gait", lambda _rid: {"params": {}, "provenance": None, "source": None})
    rep = TC.apply_trained_gait(held_dog, _WINNER, door="apply_gait", apply="always")
    assert rep["applied"] is False, rep
    assert "does NOT show" in rep["reason"]
    assert rep["error"] == "landing could not be verified by read-back"


def test_apply_gait_does_not_promise_a_gait_source_verify_may_not_report(held_dog):
    """It controls what is INSTALLED and what verify RUNS. It does not control what verify DEPLOYS."""
    rep = _apply(held_dog, _LOSER)
    assert rep["gait_source_after"] == "tuned_for_this_body::apply_gait"
    may = rep["verify_may_report"]
    assert "default_crawl" in may and "flywheel_hint" in may
    assert "NOT a failed landing" in may
    assert "deploy_select" in may


# ------------------------------------------------------------------ the read side may not erase what landed
def test_a_rejected_landing_is_named_by_verify_not_erased(held_dog):
    """The disagreement itself: verify names another arm, and must say what happened to the landed one."""
    _apply(held_dog, _LOSER)
    res = _verify(held_dog)
    if res.get("kind") != "legged":
        pytest.skip(f"verify routed to {res.get('kind')}")
    if res["gait_source"] == "tuned_for_this_body::apply_gait":
        pytest.skip("the deliberately-bad op-point won on this body; nothing was rejected to disclose")
    dsel = res.get("deploy_select")
    assert dsel, f"verify reported {res['gait_source']!r} and said nothing about the landed controller: {res}"
    assert dsel["rejected"]["gait_source"] == "tuned_for_this_body::apply_gait"
    assert dsel["rejected"]["params"] == pytest.approx(_LOSER)
    assert isinstance(dsel["rejected"]["forward_m"], float)      # its OWN number, measured on this body
    assert dsel["rejected"]["verdict"]
    assert dsel["still_installed"] is True
    assert "undo" in dsel["undo"] or "op:'undo'" in dsel["undo"]
    # and the robot really does still carry it -- "still_installed" is a claim, so check it
    from virturoid.services.trained_controller import held_gait
    assert held_gait(held_dog)["params"] == pytest.approx(_LOSER)


def test_a_deployed_landing_carries_no_disclosure(held_dog):
    """The happy path is byte-unchanged: when the landed controller wins there is nothing to disclose."""
    _apply(held_dog, _WINNER)
    res = _verify(held_dog)
    if res.get("kind") != "legged" or res["gait_source"] != "tuned_for_this_body::apply_gait":
        pytest.skip(f"the deploy-select preferred {res.get('gait_source')!r} on this body")
    assert "deploy_select" not in res


# ------------------------------------------------------------------ a landing may not cost the body its hint
def test_a_landing_cannot_cost_the_body_the_hint_it_was_already_deploying(held_dog, monkeypatch):
    """THE REGRESSION, at the unit the product decides it on.

    The mined hint is pinned rather than banked, because whether a scratch bank holds two walks near this body
    depends on test order — and the fact under test is not "the bank has hints", it is "when it does, landing a
    worse controller does not silently drop the robot onto a worse arm".
    """
    from virturoid.services import ai_native_tools as AIT
    from virturoid.services import session_state as S
    monkeypatch.setattr(AIT, "_mined_hint_params", lambda _g: dict(_WINNER))

    gene = S.get_robot(held_dog)
    with_hint = AIT._honest_gait(gene, steps=800)                # what the body deploys with no op-point of its own
    if with_hint["gait_source"] != "flywheel_hint":
        pytest.skip(f"the pinned hint lost to the shipped default on this body ({with_hint['gait_source']})")

    _apply(held_dog, _LOSER)
    after = AIT._honest_gait(S.get_robot(held_dog), steps=800)
    assert abs(after["forward_m"]) >= abs(with_hint["forward_m"]) - 1e-9, (
        f"landing a controller dropped this body from {with_hint['forward_m']} m to {after['forward_m']} m — "
        f"the hint it was deploying was never re-run")
    assert after["deploy_select"]["rejected"]["gait_source"] == "tuned_for_this_body::apply_gait"


# ------------------------------------------------------------------ the body the defect was found on
def _menagerie_go2():
    p = os.path.join(_MENAGERIE, "unitree_go2", "go2.xml")
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    from virturoid.services import session_state as S
    from virturoid.services.robot_import import import_robot
    gene = import_robot(p)["gene"]
    assert gene is not None
    return S.put_robot(gene, prompt="customer go2", label="imported")


def test_the_imported_go2_verdict_names_the_controller_landed_on_it():
    """On an IMPORTED body the provenance block is the surface the engineer reads, and it used to say

        {"whose": "ours, generic — fitted to no particular robot", "what": "default_crawl"}

    right after they had landed a controller on that robot. True of the rollout, false about the machine.
    """
    rid = _menagerie_go2()
    from virturoid.services import session_state as S
    try:
        before = _verify(rid)
        if before.get("kind") != "legged":
            pytest.skip(f"verify routed to {before.get('kind')}")
        _apply(rid, _LOSER)
        after = _verify(rid)
        if after["gait_source"] == "tuned_for_this_body::apply_gait":
            pytest.skip("the deliberately-bad op-point won on this body; nothing was rejected to disclose")
        dsel = after["deploy_select"]
        assert dsel["rejected"]["gait_source"] == "tuned_for_this_body::apply_gait"
        note = after["controller_provenance"]["a_controller_is_landed_on_this_robot"]
        assert "tuned_for_this_body::apply_gait is installed" in note
        assert "deploy_select" in note
        # ...and the landing did not make the customer's verdict worse than it was before it
        assert abs(float(after["forward_m"])) >= abs(float(before["forward_m"])) - 1e-9, (
            f"the apply dropped this robot from {before['forward_m']} m to {after['forward_m']} m")
    finally:
        S.forget_robot(rid)
