"""A TEN-MINUTE CALL THAT PRINTS NOTHING AND ADOPTS NOTHING IS THREE DEFECTS, NOT ONE.

MEASURED 2026-08-10 across 20 real ``create_robot`` calls through ``agent_tools.call_tool``: 0.5 s to 634 s,
silent throughout, and three bodies spent the whole 360-gait / 3-seed budget and adopted nothing. The stage
breakdown of the worst (a large quadruped, 663.7 s end to end) says where it goes:

    compose      43.6 s
    ground        0.0 s
    gait fit    615.4 s   -- 92.7%, and NOT one search: ~351 s on the authored body, then ~264 s more inside
                             the walkability gate, fitting the substitution candidate
    render        1.3 s

This file guards the three fixes, and it is written to make the ONE THING THAT MUST NOT HAPPEN impossible: a
build that quietly stops fitting. The fit is what makes an authored body walk (7/7 proportion variants,
docs/breaking_the_cotuning_wall.md), so every test here asserts on a DISCLOSURE, never on a duration -- a test
that asserted "this finishes in under N seconds" would go red on a loaded machine and, worse, would tempt
someone to buy the number by cutting the search.
"""
from __future__ import annotations

import importlib.util
import io
import time

import pytest

from virturoid.services import build_progress as BP
from virturoid.services import gait_flywheel as GF

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_needs_mujoco = pytest.mark.skipif(not _MUJOCO, reason="fitting/refusing a real body needs MuJoCo")


def _compose(prompt: str):
    from virturoid.services.morphology_composer import _compose_robot_impl
    return _compose_robot_impl(prompt, llm=None)


def _grounded(prompt: str):
    from virturoid.services.gene_build import ground_and_repair
    g = _compose(prompt)
    ground_and_repair(g)
    return g


# ---------------------------------------------------------------------------- 1. the wait is legible
def test_the_stage_table_records_every_stage_and_accumulates_repeats():
    """``create_robot`` runs the fit up to three times; three rows called "gait_fit" would read as three
    different stages to anyone summing the table, which is exactly the question the table exists to answer."""
    with BP.build_progress("t", printing=False) as rep:
        with BP.stage("compose"):
            pass
        with BP.stage("gait_fit"):
            pass
        with BP.stage("gait_fit"):          # the walkability gate's second fit
            pass
    names = [r["stage"] for r in rep.table()]
    assert names.count("gait_fit") == 1, f"repeats must accumulate into one row, got {rep.table()}"
    assert set(names) == {"compose", "gait_fit"}


def test_progress_prints_to_stderr_and_never_to_stdout(capsys):
    """LOAD-BEARING, not stylistic: ``virturoid.mcp_server`` speaks JSON-RPC over stdout, so one progress line
    there corrupts the protocol frame and takes the session down."""
    with BP.build_progress("t", printing=True) as rep:
        rep.say("something worth knowing")
        with BP.stage("compose"):
            pass
    cap = capsys.readouterr()
    assert cap.out == "", f"nothing may reach stdout, got {cap.out!r}"
    assert "compose" in cap.err and "something worth knowing" in cap.err


def test_a_progress_line_can_never_fail_a_build():
    """pytest's capture closes ``sys.stderr`` at teardown and a daemon heartbeat can outlive it by
    milliseconds. A build must not die because a line could not be printed."""
    rep = BP.BuildProgress("t", printing=True)
    closed = io.StringIO()
    closed.close()
    import sys
    old, sys.stderr = sys.stderr, closed
    try:
        rep.say("this write will raise inside")     # must swallow
        with rep.stage("compose"):
            pass
    finally:
        sys.stderr = old
    assert [r["stage"] for r in rep.table()] == ["compose"], "the timing must still be recorded"


def test_printing_is_off_under_pytest_and_forceable_either_way(monkeypatch):
    monkeypatch.delenv("VIRTUROID_PROGRESS", raising=False)
    assert BP.printing_enabled() is False, "the suite calls create_robot ~50 times; that is not a progress log"
    monkeypatch.setenv("VIRTUROID_PROGRESS", "1")
    assert BP.printing_enabled() is True
    monkeypatch.setenv("VIRTUROID_PROGRESS", "0")
    assert BP.printing_enabled() is False


# ---------------------------------------------------------------------------- 2. the budget
def test_the_product_default_is_a_real_ceiling_even_though_the_suite_pins_it_off(monkeypatch):
    """``tests/conftest.py`` pins ``VIRTUROID_GAIT_FIT_BUDGET_S=0`` so no walk assertion depends on machine
    speed. This asserts the PRODUCT default is not 0 — i.e. that nobody "fixed" the suite's pin by making
    unbounded the shipped behaviour, which would put the 634-second call straight back."""
    monkeypatch.delenv("VIRTUROID_GAIT_FIT_BUDGET_S", raising=False)
    assert GF._budget_from_env(None) == GF._DEFAULT_FIT_BUDGET_S
    assert GF._DEFAULT_FIT_BUDGET_S > 0
    monkeypatch.setenv("VIRTUROID_GAIT_FIT_BUDGET_S", "0")
    assert GF._budget_from_env(None) is None, "0 means unbounded"
    assert GF._budget_from_env(45) == 45.0, "an explicit argument outranks the environment"
    monkeypatch.setenv("VIRTUROID_GAIT_FIT_BUDGET_S", "not-a-number")
    assert GF._budget_from_env(None) == GF._DEFAULT_FIT_BUDGET_S, "garbage falls back, never to unbounded"


@_needs_mujoco
def test_the_deadline_stops_the_search_before_a_rollout_never_inside_one():
    """The one economy this codebase forbids is a shorter HORIZON (``fit_gait_for_body``: "a shorter horizon
    does not make the fit cheaper, it makes the answer wrong"). So the clock is checked at evaluation
    boundaries: with a deadline already past, the search must return having run ZERO candidates rather than
    having run truncated ones."""
    from virturoid.services.gait_search import search_gait
    g = _grounded("a quadruped robot dog")
    res = search_gait(g, generations=4, pop=4, steps=200, stop_on_credible=True,
                      deadline=time.monotonic() - 1.0)
    assert res.stopped_reason == "budget"
    assert res.n_evals == 0, f"an expired deadline must start no candidate at all, ran {res.n_evals}"


@_needs_mujoco
def test_a_search_the_clock_stopped_is_never_reported_as_a_finding_about_the_body():
    """THE HONESTY TEST OF THE WHOLE CHANGE. The sentence "none was still walking at the 6000-step horizon"
    reads to a customer as a measured negative about their design. It is only earned when the budget was
    actually spent looking. A budget the customer set must never come back to them as a fact about their
    robot."""
    g = _grounded("a quadruped robot dog")
    out = GF.fit_gait_for_body(g, budget_s=0.001, bank=False, deploy_steps=600, search_steps=600,
                               max_evals=8, warm_evals=4, seed_restarts=2)
    if out.get("reason") == "the shipped default gait is already a credible walk for this body":
        pytest.skip("this body never reaches the search, so there is no budget to test")
    assert out["stopped_by_budget"] is True, out
    assert "STOPPED BY THE BUILD BUDGET" in out["reason"], out["reason"]
    assert "NOTHING HERE SAYS THIS BODY CANNOT WALK" in out["reason"], out["reason"]
    assert "none was still walking" not in out["reason"], out["reason"]
    # ...and it says how to finish the search rather than leaving the reader stuck
    assert "gait_budget_s" in out["reason"]


@_needs_mujoco
def test_asking_for_an_unbounded_search_actually_gets_one(monkeypatch):
    """REGRESSION, and it was live for an hour of this task's own development. ``create_robot`` resolved the
    argument with a local ``or default``, so ``gait_budget_s: 0`` reached ``build_progress`` as "this build has
    no clock", the fitter then fell back to its OWN default, and a caller who explicitly asked for an unbounded
    search was silently capped at 180 s. MEASURED: an 8-legged spider requested unbounded, reported
    ``stopped_by_budget: true, budget_s: 180.0``. A control that does not do what it says is worse than none.

    The same path is what makes ``tests/conftest.py``'s ``VIRTUROID_GAIT_FIT_BUDGET_S=0`` pin effective, so
    this also guards the suite's determinism.
    """
    from virturoid.services import build_progress as P
    from virturoid.services.gait_flywheel import _budget_from_env

    monkeypatch.setenv("VIRTUROID_GAIT_FIT_BUDGET_S", "0")
    assert _budget_from_env(0) is None and _budget_from_env(None) is None
    with P.build_progress("create_robot", printing=False, budget_s=_budget_from_env(0)) as rep:
        assert rep.deadline is None, "an explicit 0 must leave the build with no clock at all"

    # ...and the fitter must not re-impose its OWN default inside an unbounded build. Env says 180 here, the
    # build says unbounded, and the build wins: this is precedence rule 3 in ``fit_gait_for_body``.
    monkeypatch.setenv("VIRTUROID_GAIT_FIT_BUDGET_S", "180")
    g = _grounded("a quadruped robot dog")
    with P.build_progress("create_robot", printing=False, budget_s=_budget_from_env(0)):
        out = GF.fit_gait_for_body(g, bank=False, deploy_steps=400, search_steps=400,
                                   max_evals=2, warm_evals=1, seed_restarts=1)
    if out.get("reason") == "the shipped default gait is already a credible walk for this body":
        pytest.skip("this body never reaches the search, so no budget is resolved")
    assert out["budget_s"] is None, f"the build asked for unbounded; the fit reported {out['budget_s']}"
    assert out["stopped_by_budget"] is False


@_needs_mujoco
def test_one_build_is_one_clock_not_one_clock_per_fit():
    """``create_robot`` can run the fitter three times. Three independent 180 s ceilings would honour "spend up
    to 180 seconds" while taking 540 — measured, the walkability gate alone was a second 264 s search."""
    g = _grounded("a quadruped robot dog")
    with BP.build_progress("t", printing=False, budget_s=0.001) as rep:
        rep.claim_deadline()                     # the first fit of the build started (and exhausted) the clock
        out = GF.fit_gait_for_body(g, budget_s=None, bank=False, deploy_steps=600, search_steps=600,
                                   max_evals=8, warm_evals=4, seed_restarts=2)
    if out.get("reason") == "the shipped default gait is already a credible walk for this body":
        pytest.skip("this body never reaches the search")
    # the build's exhausted clock wins even though this call asked for the (unbounded, suite-pinned) default
    assert out["stopped_by_budget"] is True, out


def test_the_clock_starts_at_the_first_SEARCH_not_at_the_build():
    """REGRESSION, found by measuring the fix rather than by reading it. Started at construction, the budget
    made a slow COMPOSER cancel the search: a biped whose compose took 253.65 s reached its fit with 0.0 s left
    and ran ZERO of 360 evaluations — 259 s of the customer's time for a body with no fitted controller AND no
    measurement of one. Nothing can interrupt the composer, so charging the search for it only moves the loss
    onto the stage that was doing useful work."""
    rep = BP.BuildProgress("t", printing=False, budget_s=60.0)
    assert rep.deadline is None, "constructing a build must not start the clock"
    first = rep.claim_deadline()
    assert first is not None and first > time.monotonic() + 55.0, "the first search gets the WHOLE budget"
    assert rep.claim_deadline() == first, "later fits share the remainder, they do not get a fresh budget"


def test_an_unbounded_build_never_produces_a_deadline():
    rep = BP.BuildProgress("t", printing=False, budget_s=0)
    assert rep.budget_s is None and rep.claim_deadline() is None


# ---------------------------------------------------------------------------- 3. refusing in advance
#
# "A SEARCH THAT CANNOT SUCCEED MUST NOT BE REPORTED AS A SEARCH THAT FAILED" is an existing rule here, and this
# task was asked to extend it to the biped: refuse 2 legs in advance, because the crawl engine cannot balance
# them. IT WAS TRIED AND THE FIRST MEASUREMENT TAKEN TO JUSTIFY IT REFUTED IT, so the refusal was withdrawn and
# these tests are what keeps it withdrawn. What shipped instead is the same instinct with a criterion that is
# MEASURED on the body rather than inferred from its class.


@_needs_mujoco
def test_a_biped_is_searchable_and_is_not_refused_in_advance():
    """THE RETRACTION, kept as a test so it cannot quietly come back.

    Measured 2026-08-10 through ``call_tool``: a composed 2-legged body (8 DOF, legs=2) ADOPTED an operating
    point at evaluation 1 — +3.732 m at the 6000-step horizon against the shipped default's 0.498, travel rate
    RISING +0.41 -> +0.62 m/1000 (accelerating, not decaying), surviving a 1e-2 perturbation of all five
    parameters at once, 4/4 probes. The bank agrees independently: two ``gait::humanoid::*`` locomotion rows,
    one at success 1.0, and a row only enters after beating the default at the deploy horizon.

    Task #206 measured ONE humanoid on which the crawl fell at step 487 while learned control held 6000/6000.
    That makes the biped the HARD case. "Hard" does not license refusing to look — that is the false-negative
    this module's own docstring warns costs "a body silently denied its own controller"."""
    m = GF.crawl_deployment_match(_compose("a bipedal walking robot"))
    if m.get("n_legs") != 2:
        pytest.skip(f"this prompt did not compose a 2-legged body (n_legs={m.get('n_legs')})")
    assert m["applicable"] is True and m["match"] is True
    assert "frontier" not in m, "the refuted 'frontier' axis must not come back as a field"


@_needs_mujoco
def test_a_quadruped_is_searchable_too():
    m = GF.crawl_deployment_match(_compose("a quadruped robot dog"))
    assert m["applicable"] is True and m["match"] is True


def test_a_body_whose_extremes_all_collapse_is_refused_in_three_rollouts(monkeypatch):
    """THE REPLACEMENT, and the criterion is a measurement of THIS body rather than a class it belongs to.

    Degeneracy is exactly "the output does not depend on the input": if the shipped default AND both extreme
    corners of the 5-parameter box all collapse inside 2% of the horizon, nothing between them can be ranked
    either. The post-hoc ``degenerate_search`` flag already caught this — after the entire budget. Measured on
    the live inchworm, that was "360 gaits (379 physics rollouts)" to establish a fact three rollouts settle.
    """
    from virturoid.services import gait_search as GS

    class _Seg:
        name = "s"
        shape = "capsule"
        length_m = radius_m = mass_kg = 0.1
        joint_type = "revolute"
        joint_axis = (0.0, 0.0, 1.0)
        joint_lower = joint_upper = 0.0
        mount_euler = (0.0, 0.0, 0.0)
        parent = None

    class _G:
        def __init__(self):
            self.metadata = {}
            self.id = "collapses"
            self.segments = [_Seg()]

    calls = []

    def _collapse(gene, params, *, steps=1200, reward_fn=None):
        calls.append(dict(params))
        return {"forward": 0.0, "survived": False, "credible": False, "verdict": "FELL",
                "steps_integrated": 12, "rates": {}, "height_ratio": 0.0, "fitness": -1.0}

    monkeypatch.setattr(GS, "evaluate_gait", _collapse)
    monkeypatch.delenv("VIRTUROID_SKIP_GAIT_FIT", raising=False)
    out = GF.fit_gait_for_body(_G(), deploy_steps=6000, bank=False)

    assert out["searched"] is False and out["n_evals"] == 0, out
    assert out["degenerate_search"] is True and out["n_rollouts"] == 3, out
    assert "NO SEARCH WAS RUN" in out["reason"] and "nothing here is a finding" in out["reason"]
    assert len(calls) == 3, "default + both corners, and nothing more"
    assert calls[1]["freq"] != calls[2]["freq"], "the two probes must be the EXTREMES, not the same point"


def test_the_probes_do_not_run_on_a_body_that_gets_anywhere(monkeypatch):
    """FAIL-OPEN, and the narrowness is the point: a body whose default rollout survives past the floor never
    pays for the probes, and any probe that survives cancels the refusal. A degeneracy test that could fire on
    a body which merely walks BADLY would be the false negative, not the fix."""
    from virturoid.services import gait_search as GS

    seen = []

    def _travels(gene, params, *, steps=1200, reward_fn=None):
        seen.append(dict(params))
        return {"forward": 0.4, "survived": True, "credible": False, "verdict": "SLIDE",
                "steps_integrated": steps, "rates": {}, "height_ratio": 0.9, "fitness": 0.1}

    monkeypatch.setattr(GS, "evaluate_gait", _travels)
    monkeypatch.delenv("VIRTUROID_SKIP_GAIT_FIT", raising=False)

    class _G:
        def __init__(self):
            self.metadata = {}
            self.id = "walks-badly"
            self.segments = [type("S", (), {"name": "s"})()]

    out = GF.fit_gait_for_body(_G(), deploy_steps=6000, bank=False, max_evals=1, warm_evals=1, seed_restarts=1)
    assert out.get("degenerate_search") is not True, out
    assert out.get("probe_rollouts") is None, "no probe may be paid for by a body that reached the horizon"


# ---------------------------------------------------------------------------- 4. through the tool
@_needs_mujoco
def test_create_robot_returns_the_breakdown_and_what_the_expensive_stage_bought():
    """``took_s`` alone says a call cost 664 seconds and nothing about which of four stages spent them."""
    from virturoid.services.agent_tools import call_tool
    res = call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]
    stages = {r["stage"]: r["seconds"] for r in res["stages"]}
    assert "compose" in stages and "gait_fit" in stages, stages
    assert res["build_seconds"] >= sum(stages.values()) - 1e-6
    assert res["stages"] == sorted(res["stages"], key=lambda r: -r["seconds"]), "slowest first, or it is a dump"
    fit = res.get("gait_fit")
    assert fit is not None and "reason" in fit, "the caller waited for this; it may not be hidden on the gene"


@_needs_mujoco
def test_turning_the_fit_off_is_disclosed_AND_actually_turns_it_off():
    """TWO defects in one line of arguments, and the second was found by this test.

    (1) ``tune_gait: false`` used to leave NO trace: the body carried no ``gait_fit`` at all, so a robot whose
    controller was never measured looked exactly like one whose default was measured and kept.

    (2) It also did not turn the fit OFF. ``create_robot`` skipped its own call and
    ``anatomy_compiler.ensure_walkable_quad`` then ran a full one anyway on bodies ``create_robot`` never sees
    — overwriting the disclosure with a real result, so the flag read as honoured while the clock said
    otherwise. The policy now lives on the build, where every fitter inside it can see it.
    """
    from virturoid.services.agent_tools import call_tool
    res = call_tool("create_robot", {"prompt": "a quadruped robot dog", "tune_gait": False})["result"]
    fit = res.get("gait_fit") or {}
    assert fit.get("skipped") is True, res.get("gait_fit")
    assert "TURNED OFF" in fit.get("reason", "") and "never measured" in fit.get("reason", "")
    assert fit.get("n_evals") == 0 and fit.get("searched") is False, fit
    # ...and no stage may have spent real search time
    stages = {r["stage"]: r["seconds"] for r in res["stages"]}
    assert stages.get("gait_fit", 0.0) < 5.0, stages


@_needs_mujoco
def test_a_biped_still_gets_a_real_search_through_the_tool():
    """The end-to-end half of the retraction: no refusal reaches the customer, and the body really is searched
    (or really does keep a default that already walks)."""
    from virturoid.services.agent_tools import call_tool
    res = call_tool("create_robot", {"prompt": "a bipedal walking robot"})["result"]
    if (res.get("appendages") or {}).get("legs") != 2:
        pytest.skip("this prompt did not compose a 2-legged body")
    fit = res.get("gait_fit") or {}
    assert fit.get("frontier") is not True, fit
    assert fit.get("searched") is True or "already a credible walk" in (fit.get("reason") or ""), fit


def test_the_budget_and_the_escape_hatch_are_both_on_the_wire():
    """A control the caller cannot discover is not a control. Both must appear in the advertised schema."""
    from virturoid.services.agent_tools import tool_specs
    spec = next(t for t in tool_specs() if t["name"] == "create_robot")
    props = spec["parameters"]["properties"]
    assert "gait_budget_s" in props and "tune_gait" in props, sorted(props)
    assert "0" in props["gait_budget_s"]["description"], "the unbounded escape must be documented"
