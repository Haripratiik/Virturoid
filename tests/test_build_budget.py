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
build that quietly stops fitting. The fit is what makes an authored body walk (7/7 proportion variants, commit 33e0800,
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


# ------------------------------------------------------- 2b. the budget a test can pin: EVALUATIONS
#
# The clock above bounds the WAIT. It cannot bound the ANSWER reproducibly, and the evidence is two runs of the
# SAME code on the SAME box on 2026-08-12: "an eight-legged spider robot" through ``call_tool``, both stopped by
# the same 180 s ceiling, stopped at 144 of 360 gaits on one run and 70 of 360 on the next. Neither number is a
# property of the robot. That is exactly why every test in section 2 asserts on a disclosure and never on a
# duration -- and why "a budgeted build discloses that it was budgeted" could not be pinned there at all. An
# evaluation is a unit the physics owns rather than the hardware, so these tests CAN assert the count.


def test_no_eval_budget_by_default_which_is_the_whole_point():
    """THE HARD CONSTRAINT OF TASK #291, guarded at the one place a cap could become the default. A number here
    instead of ``None`` would silently truncate the per-body fit for every build in the product, and the fit is
    what makes an authored body walk (7/7 proportion variants, commit 33e0800 -- six of the seven FELL BY ROLL-OVER at the
    shipped default and reached CREDIBLE WALK with an op-point of their own; the 4/4 authored-animal run in
    docs/breaking_the_cotuning_wall.md is a DIFFERENT experiment and does not carry this number)."""
    rep = BP.BuildProgress("t", printing=False)
    assert rep.evals_budget is None and rep.evals_remaining() is None
    rep.spend_evals(500)
    assert rep.evals_remaining() is None, "an unbounded build can never run out"
    assert rep.evals_spent == 500, "...but it still counts, so a caller can choose a cap next time"


def test_one_build_is_one_eval_ledger_not_one_per_fit():
    """The same 3N bug the clock has, in the other currency: ``create_robot`` runs the fitter up to three times,
    so three fits each honouring "spend up to 40 evaluations" spend 120."""
    rep = BP.BuildProgress("t", printing=False, evals_budget=40)
    assert rep.evals_remaining() == 40
    rep.spend_evals(25)                       # the authored body's fit
    assert rep.evals_remaining() == 15        # ...and the walkability gate's fit inherits the REMAINDER
    rep.spend_evals(15)
    assert rep.evals_remaining() == 0
    rep.spend_evals(9)
    assert rep.evals_remaining() == 0, "an overspent ledger reports empty, never negative"


def test_a_zero_or_negative_eval_budget_means_unbounded_like_the_clock():
    """One convention for both budgets. ``0`` on the clock already means "no ceiling"; a ``0`` here meaning "run
    no evaluations" would make the two controls read opposite ways from the same number."""
    assert BP.BuildProgress("t", printing=False, evals_budget=0).evals_budget is None
    assert BP.BuildProgress("t", printing=False, evals_budget=-5).evals_budget is None


@_needs_mujoco
def test_the_eval_budget_binds_exactly_and_is_reported_as_an_UNFINISHED_SEARCH():
    """THE HONESTY TEST, and the deterministic twin of the wall-clock one above.

    MEASURED 2026-08-12 on the grounded authored dog, ``evals_budget=8``: 8 evaluations, 14 physics rollouts,
    6.4 s -- against 124.1 s and 91 evaluations for the same body's unbudgeted fit through ``create_robot``. The
    count is asserted because it is machine-independent; the SECONDS deliberately are not.

    "none was still walking at the 6000-step horizon" would be a measurement this build did not make. A cap the
    customer set may not come back to them as a fact about their robot.
    """
    g = _grounded("a quadruped robot dog")
    with BP.build_progress("t", printing=False, budget_s=0, evals_budget=8) as rep:
        out = GF.fit_gait_for_body(g, bank=False)
    if out.get("reason") == "the shipped default gait is already a credible walk for this body":
        pytest.skip("this body never reaches the search, so there is no budget to test")
    assert out["n_evals"] == 8, f"the cap must bind exactly, ran {out['n_evals']}"
    assert rep.evals_spent == 8 and rep.evals_remaining() == 0
    assert out["stopped_by_eval_budget"] is True, out
    assert out["evals_budget"] == 8
    if not out["adopted"]:
        assert "STOPPED BY THE EVALUATION BUDGET" in out["reason"], out["reason"]
        assert "NOTHING HERE SAYS THIS BODY CANNOT WALK" in out["reason"], out["reason"]
        assert "none was still walking" not in out["reason"], out["reason"]
        assert "gait_max_evals" in out["reason"], "it must say how to finish the search"


@_needs_mujoco
def test_a_second_fit_in_the_same_build_is_refused_rather_than_given_a_fresh_cap():
    """The walkability gate fits AGAIN, on a different body, inside the same build. With the ledger already
    spent that second fit must run zero evaluations and say why -- and it must NOT get the clock's
    first-attempt exemption, which exists only because wall time is spent by stages a fit does not control."""
    g = _grounded("a quadruped robot dog")
    with BP.build_progress("t", printing=False, budget_s=0, evals_budget=6) as rep:
        GF.fit_gait_for_body(g, bank=False)
        spent_after_first = rep.evals_spent
        second = GF.fit_gait_for_body(_grounded("a quadruped robot dog"), bank=False)
    if second.get("reason") == "the shipped default gait is already a credible walk for this body":
        pytest.skip("this body never reaches the search")
    assert spent_after_first >= 6
    assert second["n_evals"] == 0, f"the second fit spent {second['n_evals']} evaluations of an empty ledger"
    assert second["stopped_by_eval_budget"] is True, second
    assert "STOPPED BY THE EVALUATION BUDGET" in second["reason"], second["reason"]


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
def test_the_per_body_fit_STILL_RUNS_by_default_and_this_is_the_non_negotiable_one():
    """EVERY OTHER TEST IN THIS FILE IS ABOUT MAKING A BUILD CHEAPER OR LOUDER. This one is the counterweight,
    and it is the constraint task #291 was given: none of that may be bought by quietly not fitting.

    The fit is what makes an authored body walk. MEASURED 2026-08-12 through ``call_tool`` with no arguments but
    the prompt, "a large quadruped robot": 91 evaluations / 150 rollouts, ADOPTED, 5.877 m still walking at step
    6000 against the shipped default's 0.01 m. Remove the fit and that body travels a centimetre.

    So: a default build of a legged body must SEARCH — not skip, not refuse, not be capped by a budget nobody
    asked for. The only alternative permitted is the one that means the search was unnecessary (the body's
    shipped default already walks), because then nothing was withheld.

    IT ALSO ASSERTS HOW MUCH SEARCH HAPPENED, AND THE FIRST VERSION DID NOT — which made it blind to the exact
    defect it exists to prevent. Mutation-tested 2026-08-13: changing the production call at
    ``ai_native_tools.py`` to ``fit_gait_for_body(gene, cache=True, warm_evals=2, max_evals=2,
    seed_restarts=1)`` cuts a 360-evaluation search to FOUR and discloses nothing — and this test PASSED, in
    12.8 s instead of the honest 132.4 s. Every assertion it had was about the ADVERTISED cap
    (``gait_max_evals is None``, ``stopped_by_eval_budget is not True``), and none of them is false when the
    search is simply told to be small. "The build got faster by quietly doing less work" is the one outcome
    task #291 named as worse than a 634 s build, so the floor below is the assertion that matters.

    ``_MIN_HONEST_EVALS`` is deliberately far under the measured 91: the point is to catch a search cut to a
    handful, not to pin a number that moves with the bank's contents (a warm start legitimately finishes early
    — measured 91 evals with a warm bank against 360 cold). A run that adopts at evaluation 1 is allowed
    through explicitly, because there the search ended by SUCCEEDING.
    """
    from virturoid.services.agent_tools import call_tool
    _MIN_HONEST_EVALS = 12
    res = call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]
    assert (res.get("appendages") or {}).get("legs", 0) >= 4, "this prompt must compose a legged body"
    assert res["gait_max_evals"] is None, "no build may arrive pre-capped"
    fit = res.get("gait_fit") or {}
    assert fit.get("skipped") is not True, fit
    assert fit.get("stopped_by_eval_budget") is not True, fit
    assert fit.get("searched") is True or "already a credible walk" in (fit.get("reason") or ""), fit
    # THE FLOOR IS ON THE BUILD LEDGER, NOT ON ``gait_fit``, and that is the whole trick. Mutation-tested: with
    # the search cut to 4, ``gait_fit`` comes back {searched: False, n_evals: 0, reason: "the shipped default
    # gait is already a credible walk for this body"} -- because the walkable gate runs its OWN fit afterwards
    # and OVERWRITES metadata['gait_fit'], erasing the shrunken search from the record entirely. Any assertion
    # guarded on ``fit["searched"]`` is therefore unreachable on the default path, which is how the first
    # version of this floor missed the mutation it was written for. ``gait_evals_spent`` is the build's own
    # ledger and survives the overwrite: it read 4 under the mutation and 336 on an honest cold run.
    #
    # The rule is a BAND, not a minimum, because zero is legitimate: a build that never needed to search says
    # so. What cannot be legitimate is a handful -- something looked, briefly, and nothing explains why it
    # stopped. That is the exact signature of a quietly shrunken search.
    spent = int(res.get("gait_evals_spent") or 0)
    if spent > 0:
        assert spent >= _MIN_HONEST_EVALS, (
            f"a default build spent {spent} evaluation(s) -- too few to be a real search and too many to be "
            f"none. Either the search was quietly shrunk (the #291 constraint this test exists for) or it "
            f"stopped for a reason that must be disclosed: gait_fit={fit}")
    if fit.get("searched") is True and not fit.get("adopted"):
        n_evals = int(fit.get("n_evals") or 0)
        assert n_evals >= _MIN_HONEST_EVALS, (
            f"the fit reports searching only {n_evals} evaluation(s) and adopting nothing: {fit}")


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
    assert "gait_max_evals" in props, "the machine-independent budget must be discoverable too"
    assert "gait_max_evals" not in (spec["parameters"].get("required") or []), "capping must never be required"
    assert "default" not in props["gait_max_evals"], (
        "an advertised default here would read as 'the search is capped out of the box'; it is not, and the "
        "uncapped search is what makes an authored body walk")


def test_the_advertised_default_of_every_control_is_the_one_the_handler_uses():
    """A MEASURED DEFECT, not a tidy-up: ``ensure_walkable`` was advertised ``default: False`` while
    ``create_robot`` reads ``args.get("ensure_walkable", True)``. Measured 2026-08-12, a plain
    ``create_robot({"prompt": "a large quadruped robot"})`` ran a ``walkable_gate`` stage for 0.77 s -- so the
    gate was ON while the schema said OFF.

    It is the WORST control in the tool to misdescribe. The walkability gate is the thing that can DISCARD THE
    CUSTOMER'S COMPOSED BODY and ship a template in its place (see 12b11dc, where a regression let exactly that
    happen again), so a caller reading the schema was told the opposite of the truth about who owns the robot
    that comes back.

    Asserted against the handler's own defaults rather than against literals, so the next control cannot drift
    the same way.
    """
    import inspect

    from virturoid.services import ai_native_tools as A
    from virturoid.services.agent_tools import tool_specs
    props = next(t for t in tool_specs() if t["name"] == "create_robot")["parameters"]["properties"]
    src = inspect.getsource(A.create_robot) + inspect.getsource(A._create_robot_stages)
    flags = [(k, v["default"]) for k, v in props.items()
             if isinstance(v.get("default"), bool)]
    assert {k for k, _ in flags} == {"ensure_walkable", "tune_gait"}, sorted(k for k, _ in flags)
    for name, advertised in flags:
        for literal in (f'args.get("{name}", True)', f'args.get("{name}", False)'):
            if literal in src:
                assert advertised is literal.endswith("True)"), (
                    f"{name}: schema says default {advertised!r}, the handler uses {literal}")
                break
        else:
            raise AssertionError(f"{name} advertises a default the handler does not visibly set")
    # ...and the one numeric default, which the handler resolves through the fitter rather than inline
    assert props["gait_budget_s"]["default"] == GF._DEFAULT_FIT_BUDGET_S


@_needs_mujoco
def test_a_budgeted_build_says_it_was_budgeted_all_the_way_out_through_the_tool():
    """END TO END, and deterministic: the cap is in candidates, so this asserts a COUNT rather than a clock.
    A partial search presented as a completed one is the defect; ``gait_max_evals`` echoed back on the result is
    what stops a reader mistaking the two.

    THIS TEST WAS DEAD WHEN WRITTEN, and mutation found it: neutering all three production sites that set
    ``stopped_by_eval_budget = True`` turned its two siblings red and left this one GREEN. Two independent
    reasons, both now closed. (1) The flag was not in ``ai_native_tools``' ``gait_fit`` projection whitelist, so
    it never reached the result at all -- and ``fit.get(...) is True`` on an absent key is False, i.e. the
    assertion could only ever have failed, never passed, and was never reached to fail. (2) The call used the
    DEFAULT path, where the walkable gate runs its own fit afterwards and overwrites ``metadata['gait_fit']``
    with ``{searched: false, n_evals: 0}`` -- so ``if fit.get("searched")`` never opened, and a 6-evaluation
    build and a 360-evaluation build returned byte-identical ``gait_fit`` dicts.

    So the budgeted half now runs with ``ensure_walkable: False``, which is the only way to observe the fit this
    test is about, and asserts unconditionally rather than behind an ``if`` that a later stage can close."""
    from virturoid.services.agent_tools import call_tool
    res = call_tool("create_robot", {"prompt": "a quadruped robot dog",
                                     "gait_max_evals": 6, "ensure_walkable": False})["result"]
    assert res["gait_max_evals"] == 6, res.get("gait_max_evals")
    assert res["gait_evals_spent"] <= 6, res["gait_evals_spent"]
    fit = res.get("gait_fit") or {}
    assert fit.get("searched") is True, (
        "the fit this test is about must be the one reported -- if a later stage overwrote it, this test is "
        f"measuring something else again: {fit}")
    assert fit["n_evals"] <= 6, fit
    if not fit.get("adopted"):
        assert fit.get("stopped_by_eval_budget") is True, (
            f"the machine-readable flag must survive the projection out to the caller, not just the prose: {fit}")
        assert "STOPPED BY THE EVALUATION BUDGET" in fit["reason"], fit["reason"]
    # ...and an UNCAPPED build must not start reporting a cap it does not have
    plain = call_tool("create_robot", {"prompt": "a quadruped robot dog", "tune_gait": False})["result"]
    assert plain["gait_max_evals"] is None, plain["gait_max_evals"]


def test_a_cache_replay_says_it_ran_no_rollouts_instead_of_billing_the_first_calls_work():
    """A memoized fit hands back the right ANSWER and must not hand back the first call's INVOICE.

    Found by adversarial review of #291 and measured 2026-08-13: with ``VIRTUROID_GAIT_FIT_CACHE=1`` -- the
    mode this repo's own test instructions require and the one the build path asks for -- a second structurally
    identical fit returned ``{searched: True, n_evals: 6}`` for a call that ran zero rollouts, with nothing to
    tell a reader which it was. The evaluation ledger was never overspent, so the budget itself held; what was
    wrong is a claim about work done. That is the same defect as every other number corrected today: true where
    it was taken, quoted somewhere it was not.

    The counts are deliberately KEPT rather than zeroed. They are the honest provenance of the answer being
    replayed, and blanking them would trade one misreading ("this call searched 6 gaits") for another ("this
    answer came from nothing"). ``replayed_from_cache`` is what makes them readable as history.
    """
    GF._FIT_CACHE.clear()
    kw = dict(cache=True, warm_evals=3, max_evals=3, seed_restarts=1, bank=False)
    first = GF.fit_gait_for_body(_grounded("a quadruped robot dog"), **kw)
    second = GF.fit_gait_for_body(_grounded("a quadruped robot dog"), **kw)
    assert second.get("adopted") == first.get("adopted"), "a replay must not change the answer"
    assert second.get("n_evals") == first.get("n_evals"), "the provenance of the answer is kept, not blanked"
    # ...and the first call must NOT be labelled a replay, or the flag says nothing
    assert not first.get("replayed_from_cache"), first
    assert first.get("evals_spent_by_this_call") is None, first
    # ...while the second says plainly that it spent nothing
    assert second.get("replayed_from_cache") is True, second
    assert second.get("evals_spent_by_this_call") == 0, second
    assert "REPLAYED" in (second.get("reason") or ""), second.get("reason")
    assert "ran no rollouts" in (second.get("reason") or ""), second.get("reason")
