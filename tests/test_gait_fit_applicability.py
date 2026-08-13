"""A SEARCH THAT CANNOT SUCCEED MUST NOT BE REPORTED AS A SEARCH THAT FAILED.

Defect D1 of ``docs/body_vs_controller_ruling.md``, measured on the live-model-designed inchworm:
``fit_gait_for_body`` reported

    "searched 360 gaits (379 physics rollouts) for this body over 3 seed attempt(s); none was still walking at
     the 6000-step horizon"

and it took **1.3 seconds**. 379 rollouts of 6000 steps in 1.3 s is arithmetically impossible; every one of them
was the same unactuated collapse, because the body is a limbless serial spine and ``crawl_gait_rollout``'s wave
gait drives ``build_appendage_map(...).legs`` — of which it has none. ``ai_native_tools._honest_gait`` routes
exactly that body to ``_honest_serpentine`` before it ever reads ``metadata['gait_params']``, so nothing the
search could have adopted would have reached the robot either. To a customer that sentence reads as a measured
negative about their design. It is a measurement artefact, the same class as a NaN twin issuing a certificate.

Two guards, and they are deliberately different in kind:

  * ``crawl_deployment_match`` — STRUCTURAL and a-priori. Compares the controller this fitter searches against
    the one the product will actually deploy, and refuses before paying for a single rollout. Exact where it
    fires, and it FAILS OPEN everywhere it cannot see the body.
  * the ``degenerate_search`` flag — EMPIRICAL and a-posteriori. A decline may only quote a horizon that some
    rollout actually reached. Knows nothing about body kinds, so it catches the classes the matrix does not.
"""
from __future__ import annotations

import importlib.util

import pytest

from virturoid.services import gait_flywheel as GF
from virturoid.services import gait_search as GS

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_needs_mujoco = pytest.mark.skipif(not _MUJOCO, reason="the structural matrix measures a compiled body")


class _Gene:
    """A body the matrix cannot read at all (no segments, no compilable model)."""
    def __init__(self):
        self.metadata = {}
        self.id = "test-gene"
        self.robot_class = "quadruped"


def _compose(prompt: str):
    from virturoid.services.morphology_composer import _compose_robot_impl
    return _compose_robot_impl(prompt, llm=None)


# --------------------------------------------------------------------------- the matrix, on real bodies


@_needs_mujoco
def test_a_limbless_serial_spine_names_the_controller_that_will_actually_drive_it():
    """The D1 row. The refusal has to say WHICH controller ships, or it is just another unexplained decline."""
    m = GF.crawl_deployment_match(_compose("a snake robot"))
    assert m["applicable"] is False and m["match"] is False
    assert m["deployed_controller"] == "morph_policy.serpentine_rollout"
    assert m["searched_controller"] == "morph_policy.crawl_gait_rollout"
    assert "SERIAL SPINE" in m["why"]


@_needs_mujoco
def test_a_quadruped_is_searchable_because_the_two_controllers_agree():
    m = GF.crawl_deployment_match(_compose("a quadruped robot dog"))
    assert m["applicable"] is True and m["match"] is True
    assert m["deployed_controller"] == m["searched_controller"] == "morph_policy.crawl_gait_rollout"
    assert m["n_legs"] == 4


@_needs_mujoco
def test_a_hexapod_is_searchable_too():
    """Leg COUNT is not the gate — having legs at all is."""
    m = GF.crawl_deployment_match(_compose("a hexapod robot"))
    assert m["applicable"] is True and (m["n_legs"] or 0) >= 4


@_needs_mujoco
def test_a_wheeled_body_is_refused_because_verify_drives_it():
    m = GF.crawl_deployment_match(_compose("a six wheeled rover"))
    assert m["applicable"] is False and m["kind"] == "mobile"
    assert "_honest_drive" in m["deployed_controller"]


@_needs_mujoco
def test_an_arm_is_refused_as_a_manipulator_and_never_as_a_spine():
    """ORDERING REGRESSION, and it was live for the first draft of this guard.

    A 6-DOF arm IS "a single non-branching revolute chain", so ``aquatic._is_serial_spine`` answers True for it.
    Only ``_honest_gait`` consults that predicate, and ``verify_robot`` has already routed a manipulator to
    ``_honest_reach`` long before ``_honest_gait`` is reached. Testing the spine first made the arm's disclosure
    name the serpentine controller — a refusal with the wrong reason on it, which is its own small dishonesty.
    """
    from virturoid.services.aquatic import _is_serial_spine
    arm = _compose("a 6-DOF robot arm with a gripper")
    assert _is_serial_spine(arm), "precondition: an arm trips the spine predicate, which is why order matters"
    m = GF.crawl_deployment_match(arm)
    assert m["applicable"] is False and m["kind"] == "manipulator"
    assert "serpentine" not in m["deployed_controller"]


@_needs_mujoco
def test_an_aerial_body_is_flown_not_walked():
    g = _compose("a quadruped robot dog")
    g.robot_class = "aerial"                     # the structural marker verify_robot dispatches on
    m = GF.crawl_deployment_match(g)
    assert m["applicable"] is False and "_honest_fly" in m["deployed_controller"]


@_needs_mujoco
def test_an_aquatic_body_is_swum_not_walked():
    g = _compose("a quadruped robot dog")
    g.robot_class = "aquatic"
    m = GF.crawl_deployment_match(g)
    assert m["applicable"] is False and "_honest_swim" in m["deployed_controller"]


def test_the_matrix_fails_open_on_a_body_it_cannot_read():
    """The cost of a false 'applicable' is one search. The cost of a false 'not applicable' is a body silently
    denied its own controller — so an unreadable body is searched, and says it could not be read."""
    m = GF.crawl_deployment_match(_Gene())
    assert m["applicable"] is True
    assert m["measured"] is False


# --------------------------------------------------------------------------- the fitter refuses, and says why


@_needs_mujoco
def test_the_fitter_refuses_a_serial_spine_without_running_one_rollout(monkeypatch):
    """The whole defect in one assertion: ZERO evaluations, not 360."""
    calls = []

    def _never(gene, params, *, steps=1200, reward_fn=None):
        calls.append(params)
        raise AssertionError("a body the crawl gait cannot drive must not be searched")

    monkeypatch.setattr(GS, "evaluate_gait", _never)
    out = GF.fit_gait_for_body(_compose("a snake robot"), bank=False, seed_restarts=3)
    assert calls == []
    assert out["ok"] is True and out["searched"] is False and out["adopted"] is False
    assert out["not_applicable"] is True and out["n_evals"] == 0
    assert out["deployed_controller"] == "morph_policy.serpentine_rollout"


@_needs_mujoco
def test_the_refusal_never_reads_as_a_finding_about_the_body():
    out = GF.fit_gait_for_body(_compose("a snake robot"), bank=False, seed_restarts=1)
    reason = out["reason"]
    assert "none was still walking" not in reason
    assert "NO SEARCH WAS RUN" in reason and "nothing here is a finding about this body" in reason
    # ...and "not applicable" must not be read as "not fittable": the serpentine controller exposes five
    # parameters of its own and has simply never had any of them fitted.
    assert out["fittable_in_principle"] is True
    assert "amp, wavenum, freq, kp, kd" in out["what_would_make_it_fittable"]


@_needs_mujoco
def test_the_refusal_is_recorded_on_the_body_so_a_later_reader_cannot_mistake_it():
    g = _compose("a snake robot")
    GF.fit_gait_for_body(g, bank=False, seed_restarts=1)
    fit = (getattr(g, "metadata", None) or {}).get("gait_fit") or {}
    assert fit.get("not_applicable") is True and fit.get("searched") is False
    assert "gait_params" not in (getattr(g, "metadata", None) or {}), "a refusal adopts nothing"


# --------------------------------------------------------------------------- the general degeneracy backstop


# TWO NETS CATCH DEGENERACY, AND THESE TESTS ARE ABOUT THE SECOND ONE.
#
# Since 2026-08-10 ``fit_gait_for_body`` also refuses UP FRONT: when the shipped default collapses inside
# ``_DEGENERATE_INTEGRATION_FRAC`` of the horizon it probes the two extreme CORNERS of the parameter box, and if
# those collapse together it returns ``searched: False`` in three rollouts rather than 379. That is a cheaper and
# better answer wherever it applies, and it is covered by tests/test_build_budget.py.
#
# It is not what the four tests below assert. They are the GENERAL post-search backstop -- the net for every case
# the up-front probe deliberately fails open on -- so the doubles hand the corner probes SILENCE about
# ``steps_integrated``, which is precisely the fail-open the probe documents ("an evaluator that does not report
# ``steps_integrated`` ... is searched normally"). The search then runs, and the backstop is what answers.
_CORNERS = ({k: GS._LO[k] for k in GF._FIT_PARAMS}, {k: GS._HI[k] for k in GF._FIT_PARAMS})


def _is_corner_probe(params) -> bool:
    """Is this the fitter's up-front probe of an extreme corner, rather than a rollout of the gait under test?"""
    return any(all(float(params.get(k, 1e30)) == float(v) for k, v in c.items()) for c in _CORNERS)


def _stub_never_credible(*, integrated, corners=None):
    """Every rollout falls. ``integrated`` is how much of the horizon it bought before it did (None = silent).

    ``corners`` is what the two EXTREME-CORNER probes report, and it defaults to ``None`` (silent) so these tests
    keep measuring the post-search backstop rather than the up-front refusal that would otherwise preempt it --
    see the block above. Pass a number to exercise the up-front branch instead.
    """
    def _ev(gene, params, *, steps=1200, reward_fn=None):
        r = {"fitness": -1.0, "forward": 0.0, "height_ratio": 0.1, "survived": False, "cadence": 0.0,
             "support_frac": 0.0, "credible": False, "verdict": "FELL", "reward_return": -1.0}
        depth = corners if _is_corner_probe(params) else integrated
        if depth is not None:
            r["steps_integrated"] = depth
        return r
    return _ev


def _stub_search_that_finds_nothing(**extra):
    # ``deadline`` and ``eval_cap`` accepted and ignored: the real ``_one_search`` takes the build's shared
    # gait-fit clock (2026-08-10) and its shared evaluation ledger (2026-08-12). A double without the first
    # raised TypeError inside ``fit_gait_for_body``'s own except, so these assertions about an HONEST DECLINE
    # came back as an error payload -- "gait fit FAILED for this body (TypeError) - this is an error, not a
    # finding", which is exactly the confusion this file exists to stop. Nothing below is budgeted: the suite
    # pins the clock off (tests/conftest.py) and sets no evaluation cap.
    def _one(gene, *, db, bank, warm_evals, max_evals, out, kw, deadline=None, eval_cap=None):
        return ({"beats_default": False, "params": {}, "forward_m": 0.0, "credible": False,
                 "survived": False, "n_evals": 120, **extra}, 120)
    return _one


def test_a_search_whose_rollouts_never_reach_the_horizon_retracts_the_horizon(monkeypatch):
    """1.3 s for 379 six-thousand-step rollouts, made legible. The unit is STEPS, not seconds, so the check is
    deterministic and does not flip on a loaded machine."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_never_credible(integrated=30))
    monkeypatch.setattr(GF, "_one_search", _stub_search_that_finds_nothing())
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=1)
    assert out["searched"] is True and out["adopted"] is False
    assert out["degenerate_search"] is True and out["deepest_rollout_steps"] == 30
    assert "MEASURED NOTHING" in out["reason"] and "step 30 of 6000" in out["reason"]
    assert "NOTHING HERE IS A FINDING" in out["reason"]


def test_a_real_decline_still_reads_as_a_decline(monkeypatch):
    """The backstop must not swallow the honest expensive negative — the grounded authored dog spends its whole
    budget establishing that it cannot walk, and that IS a finding about the body."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_never_credible(integrated=6000))
    monkeypatch.setattr(GF, "_one_search", _stub_search_that_finds_nothing())
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=1)
    assert out["degenerate_search"] is False and out["deepest_rollout_steps"] == 6000
    assert "none was still walking at the 6000-step horizon" in out["reason"]
    assert "the deepest rollout reached step 6000 of 6000" in out["reason"]


def test_the_deepest_rollout_is_a_maximum_over_every_attempt_not_only_the_default(monkeypatch):
    """A default that dies instantly does not make the SEARCH degenerate — a candidate may hold the body up far
    longer, and the winner's own deploy rollout is where that shows."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_never_credible(integrated=12))
    monkeypatch.setattr(GF, "_one_search", _stub_search_that_finds_nothing(deploy_steps_integrated=5200))
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=1)
    assert out["deepest_rollout_steps"] == 5200 and out["degenerate_search"] is False


def test_silence_about_integration_is_never_read_as_zero(monkeypatch):
    """An evaluator that does not report its work is UNKNOWN, not degenerate. Reading silence as zero would
    turn every contract test's honest decline — and any future evaluator — into a false alarm."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_never_credible(integrated=None))
    monkeypatch.setattr(GF, "_one_search", _stub_search_that_finds_nothing())
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=1)
    assert out["degenerate_search"] is False and out["deepest_rollout_steps"] is None
    assert "none was still walking at the 6000-step horizon" in out["reason"]
    assert "deepest rollout" not in out["reason"], "a depth nobody measured must not be invented"


# --------------------------------------------------------------------------- where the work number comes from


def test_evaluate_gait_reports_the_steps_a_real_rollout_integrated(monkeypatch):
    from virturoid.services import morph_policy as MP
    monkeypatch.setattr(MP, "crawl_gait_rollout", lambda gene, **kw: {
        "finite": True, "survived": False, "forward": 0.2, "height_ratio": 0.4, "alive": 1234,
        "cadence": 3.0, "support_frac": 0.5, "steps": 6000, "frame_every": 20})
    assert GS.evaluate_gait(_Gene(), dict(GF._DEFAULT_GAIT), steps=6000)["steps_integrated"] == 1234


def test_a_rollout_that_never_ran_a_step_reports_zero_not_a_full_horizon(monkeypatch):
    """``crawl_gait_rollout``'s no-graph early return claims ``alive == steps`` and ``survived: True`` for a body
    it never stepped. Taking ``alive`` at face value there would report a full horizon of physics that did not
    happen — so the presence of the ``steps`` key (which only the real rollout emits) is the "any physics at
    all" test."""
    from virturoid.services import morph_policy as MP
    monkeypatch.setattr(MP, "crawl_gait_rollout", lambda gene, **kw: {
        "finite": True, "survived": True, "forward": 0.0, "height_ratio": 1.0, "alive": 6000,
        "cadence": 0.0, "support_frac": 0.0, "upright_frac": 1.0, "n_feet": 0, "speed": 0.0})
    assert GS.evaluate_gait(_Gene(), dict(GF._DEFAULT_GAIT), steps=6000)["steps_integrated"] == 0
