"""ROBUSTNESS-AWARE GAIT SEARCH, and the error bar that must be printed next to the verdict (task #267).

THE DEFECT. ``search_gait`` optimised a scalar computed from ONE deterministic rollout, so the point it returned
was whichever single draw scored highest — and the highest point of a rough landscape is systematically a spike.
Measured on the grounded authored cat, the adopted operating point flips from ``+0.958 m CREDIBLE WALK`` to
``+0.500 m FELL by ROLL-OVER`` under a **2.4e-5 relative change in step frequency**, and a robust point that is
30% faster and survives 8/8 perturbations at 1e-2 exists in the same body's parameter space. Hardware tolerance
on a gain or a commanded frequency is ~1e-2, so the fragile point is not a controller anyone could deploy.

THE TWO HALVES OF THE FIX, both under test here:
  1. the SEARCH scores a candidate by its NEIGHBOURHOOD (``evaluate_gait_robust``), ranks robust-first
     (``_robust_key``), and the credible-early-stop fires only for a candidate whose perturbed copies walk too;
  2. the RESULT carries a per-parameter margin, because "1e-1 on every axis" and "below 1e-5 on four of five"
     used to print the same two words.

These are CONTRACT tests: the physics is stubbed so what is under test is the selection rule, not the simulator.
The measured before/after on real bodies lives in the task report, not here.
"""
from __future__ import annotations

import pytest

from virturoid.services import gait_flywheel as GF
from virturoid.services import gait_search as GS


class _Gene:
    def __init__(self):
        self.metadata = {}
        self.id = "robust-test-gene"
        self.robot_class = "quadruped"


def _res(fitness: float, credible: bool, forward: float = 1.0) -> dict:
    return {"fitness": float(fitness), "forward": float(forward), "height_ratio": 0.9,
            "survived": True, "credible": bool(credible), "cadence": 20.0, "support_frac": 0.9,
            "verdict": "CREDIBLE WALK" if credible else "SLIDE (feet barely lift / no real stepping)",
            "reward_return": float(fitness), "horizon_steps": 1500, "rates": {}, "holds_rate": None}


# --------------------------------------------------------------------------- the neighbourhood scorer


def test_a_candidate_is_scored_by_its_neighbourhood_not_its_luckiest_rollout(monkeypatch):
    """The knife edge, in miniature: credible at its own numbers, a fall one perturbation away."""
    seen = []

    def fake(_gene, params, **_kw):
        seen.append(dict(params))
        exact = params["freq"] == 2.0                     # ONLY the un-perturbed float walks
        return _res(9.0 if exact else -0.7, exact)

    monkeypatch.setattr(GS, "evaluate_gait", fake)
    r = GS.evaluate_gait_robust(_Gene(), {"freq": 2.0, "kp": 30.0}, rel=1e-2, n=4)
    assert r["credible"] is True, "the nominal rollout is credible — that is the whole trap"
    assert r["robust_credible"] is False
    assert r["robust_frac"] == "0/4"
    assert r["robust_fitness"] < r["fitness"], "the neighbourhood score must be worse than the lucky draw"
    assert r["n_rollouts"] == 5
    assert len(seen) == 5


def test_a_real_controller_keeps_its_score(monkeypatch):
    monkeypatch.setattr(GS, "evaluate_gait", lambda _g, _p, **_kw: _res(3.0, True))
    r = GS.evaluate_gait_robust(_Gene(), {"freq": 2.0, "kp": 30.0}, rel=1e-2, n=4)
    assert r["robust_credible"] is True
    assert r["robust_frac"] == "4/4"
    assert r["robust_fitness"] == pytest.approx(3.0)


def test_a_candidate_that_does_not_walk_pays_for_no_probes(monkeypatch):
    """The screen is what keeps this affordable: a fall cannot be robust, so confirming it is wasted budget."""
    calls = []
    monkeypatch.setattr(GS, "evaluate_gait",
                        lambda _g, p, **_kw: (calls.append(p), _res(-0.7, False))[1])
    r = GS.evaluate_gait_robust(_Gene(), {"freq": 2.0, "kp": 30.0}, rel=1e-2, n=4)
    assert len(calls) == 1
    assert r["n_rollouts"] == 1
    assert r["robust_frac"] is None
    assert r["robust_fitness"] == pytest.approx(-0.7)


def test_perturbation_is_relative_and_touches_only_the_named_parameters():
    import random
    p = {"freq": 2.0, "kp": 100.0, "note": "not a number"}
    q = GS.perturbed_params(p, 0.01, random.Random(0), keys=("freq", "kp"))
    assert q["note"] == "not a number"
    assert q["freq"] != 2.0 and abs(q["freq"] - 2.0) <= 0.01 * 2.0
    assert q["kp"] != 100.0 and abs(q["kp"] - 100.0) <= 0.01 * 100.0


# --------------------------------------------------------------------------- the selection rule


def test_a_controller_outranks_a_lucky_float_that_travelled_further():
    """Lexicographic, not weighted. No amount of extra distance buys a point out of being unrepeatable —
    measured on the cat, the fragile candidate beat its robust near-twin by 0.16% of travel."""
    lucky = {"fitness": 99.0, "robust_fitness": 40.0, "robust_rank": 0.0}
    real = {"fitness": 1.0, "robust_fitness": 1.0, "robust_rank": 1.0}
    assert GS._robust_key(real) > GS._robust_key(lucky)


def test_a_fragile_walk_still_outranks_a_body_that_never_walked(monkeypatch):
    """The regression the FIRST version of this key caused, pinned. Keying on "every copy walked" tied every
    non-robust candidate at 0 and let the mean fitness decide — so a non-credible SLIDE (fitness 0.3*forward,
    always >= 0) outranked a fragile but real WALK (~0.8 after two falls are averaged in). On the grounded
    authored cat that turned "adopts a fragile walk, flagged and unbanked" into "adopts nothing and ships the
    default, which falls": a strictly worse outcome dressed up as a stricter one."""
    def fake(_gene, params, **_kw):
        if params["freq"] == 2.0:
            return _res(3.8, True, forward=3.8)      # the knife-edge walk, at exactly its own float
        if params["freq"] == 3.0:
            return _res(1.2, False, forward=4.0)     # a SLIDE: survives, covers ground, never lifts a foot
        return _res(-0.7, False, forward=0.4)        # anything perturbed off the edge falls

    monkeypatch.setattr(GS, "evaluate_gait", fake)
    walk = GS.evaluate_gait_robust(_Gene(), {"freq": 2.0}, rel=1e-2, n=2)
    slide = GS.evaluate_gait_robust(_Gene(), {"freq": 3.0}, rel=1e-2, n=2)
    assert walk["robust_frac"] == "0/2" and slide["robust_frac"] is None
    assert slide["robust_fitness"] > walk["robust_fitness"], "the scalar really does invert here"
    assert GS._robust_key(walk) > GS._robust_key(slide), "a real walk must outrank a body that never walked"


def test_a_wider_basin_outranks_a_narrower_one():
    """Graded, not binary: 2-of-4 neighbours walking is real information about the basin and must beat 1-of-4."""
    wide = {"robust_rank": 0.5, "robust_fitness": 1.0}
    narrow = {"robust_rank": 0.25, "robust_fitness": 9.0}
    assert GS._robust_key(wide) > GS._robust_key(narrow)


def test_with_robust_scoring_off_the_ranking_is_exactly_todays():
    a, b = {"fitness": 2.0}, {"fitness": 1.0}
    assert GS._robust_key(a) > GS._robust_key(b)


def _scripted(script):
    """``evaluate_gait`` stub driven by call index; records the params of every call."""
    seen: list[dict] = []

    def fake(_gene, params, **_kw):
        seen.append(dict(params))
        return script(len(seen) - 1)
    return fake, seen


def test_a_fragile_credible_candidate_does_not_end_the_search(monkeypatch):
    """THE BUG, at the search level. Call 0 is the baseline; candidate A is credible and its two probes fall;
    candidate B is credible and its probes hold. The old rule shipped A."""
    def script(i):
        return {0: _res(-0.7, False),                 # baseline
                1: _res(9.0, True, forward=5.0),      # candidate A — credible...
                2: _res(-0.7, False), 3: _res(-0.7, False),   # ...and one perturbation from a fall
                4: _res(1.0, True, forward=1.0),      # candidate B — credible...
                5: _res(1.0, True), 6: _res(1.0, True),       # ...and so are its neighbours
                }.get(i, _res(-0.7, False))

    fake, seen = _scripted(script)
    monkeypatch.setattr(GS, "evaluate_gait", fake)
    r = GS.search_gait(_Gene(), generations=1, pop=2, steps=10, seed=0,
                       stop_on_credible=True, robust_rel=1e-2, robust_n=2)
    assert r.stopped_reason == "credible_walk"     # deliberately the SAME string; the bar changed, not the reason
    assert r.best_robust_credible is True
    assert r.best_robust_frac == "2/2"
    assert r.best_params == seen[4], "the ROBUST candidate must be the winner, not the further-travelling one"
    assert r.n_evals == 2 and r.n_rollouts == 7


def test_the_search_reports_the_rollouts_it_actually_ran(monkeypatch):
    """``n_evals`` counts candidates and is no longer the cost. Reporting it alone would hide the price of
    robustness-aware search exactly where it is paid."""
    fake, seen = _scripted(lambda i: _res(2.0, True))
    monkeypatch.setattr(GS, "evaluate_gait", fake)
    r = GS.search_gait(_Gene(), generations=1, pop=3, steps=10, robust_rel=1e-2, robust_n=2)
    assert r.n_evals == 3
    assert r.n_rollouts == len(seen) == 1 + 3 * 3      # baseline + three candidates, each with two probes


def test_robust_scoring_still_goes_through_the_worker_pool(monkeypatch):
    """Turning robustness on must not silently cancel a caller's ``workers``. The batch helper carries the
    neighbourhood spec as three plain numbers precisely so the parallel path is available to the mode that
    needs it most; this pins that the robust spec actually reaches the evaluator through it."""
    fake, seen = _scripted(lambda i: _res(2.0, True))
    monkeypatch.setattr(GS, "evaluate_gait", fake)
    out = GS._eval_batch(_Gene(), [{"freq": 2.0}, {"freq": 2.5}], 10, 1, robust=(1e-2, 2, 7))
    assert [r["robust_frac"] for r in out] == ["2/2", "2/2"]
    assert len(seen) == 6                                     # two candidates, each nominal + two probes


def test_robust_rel_none_leaves_the_old_search_untouched(monkeypatch):
    """The ablation guarantee: the before/after in the task report compares two code paths that are the same
    code with one argument changed, so any difference measured there is the argument."""
    fake, seen = _scripted(lambda i: _res(float(i), i == 3))
    monkeypatch.setattr(GS, "evaluate_gait", fake)
    r = GS.search_gait(_Gene(), generations=2, pop=3, steps=10, seed=1, stop_on_credible=True)
    assert r.stopped_reason == "credible_walk"
    assert r.n_evals == 3 and r.n_rollouts == len(seen)
    assert r.robust_rel is None and r.best_robust_frac is None


# --------------------------------------------------------------------------- the error bar


def _knife_edge_on(param: str):
    """Credible everywhere except when ``param`` moves by more than 1e-4 relative."""
    base = {"freq": 2.0, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 30.0, "kd": 2.0}

    def fake(_gene, params, **_kw):
        drift = abs(float(params[param]) - base[param]) / base[param]
        return _res(1.0, drift <= 1e-4)
    return base, fake


def test_the_margin_names_the_parameter_that_cannot_be_touched(monkeypatch):
    """The actionable half. A body that is flat on four axes and knife-edge on step frequency needs its step
    frequency pinned — a single joint scalar cannot say that, and CREDIBLE WALK says nothing at all."""
    base, fake = _knife_edge_on("freq")
    monkeypatch.setattr(GS, "evaluate_gait", fake)
    rob = GF.robustness_margin(_Gene(), base, steps=10)
    assert rob["robustness_rel"] is None, "jointly perturbed copies all move freq -> all fall"
    assert rob["per_param_rel"]["freq"] is None
    assert rob["per_param_rel"]["kp"] == pytest.approx(0.1)
    assert rob["per_param_rel"]["kd"] == pytest.approx(0.1)
    assert rob["n_rollouts"] > 0
    sentence = GF.margin_sentence(rob)
    assert "freq <0.001" in sentence and "kp 0.1" in sentence


def test_a_sturdy_point_says_so_in_the_same_sentence(monkeypatch):
    monkeypatch.setattr(GS, "evaluate_gait", lambda _g, _p, **_kw: _res(1.0, True))
    rob = GF.robustness_margin(_Gene(), {"freq": 2.0, "kp": 30.0}, steps=10)
    assert rob["robustness_rel"] == pytest.approx(0.1)
    assert list(rob["per_param_rel"].values()) == pytest.approx([0.1, 0.1])
    assert "survives a 0.1 relative perturbation" in GF.margin_sentence(rob)


def test_the_reported_margin_is_measured_on_draws_the_search_never_saw(monkeypatch):
    """An error bar quoted on the draws the winner was selected under is an error bar on its own training set.
    It would read HIGH exactly when the search had overfitted the probe — the one case it exists to catch."""
    searched: list[dict] = []
    measured: list[dict] = []

    sink = searched
    monkeypatch.setattr(GS, "evaluate_gait",
                        lambda _g, p, **_kw: (sink.append(dict(p)), _res(1.0, True))[1])
    GS.search_gait(_Gene(), generations=1, pop=2, steps=10, seed=0, robust_rel=1e-2, robust_n=3)

    sink = measured
    monkeypatch.setattr(GS, "evaluate_gait",
                        lambda _g, p, **_kw: (sink.append(dict(p)), _res(1.0, True))[1])
    GF.robustness_margin(_Gene(), {"freq": 2.0, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 30.0, "kd": 2.0},
                         steps=10, ladder=(1e-2,))

    def freqs(rows):
        return {round(float(r["freq"]), 12) for r in rows}
    assert freqs(searched) & freqs(measured) == set(), "the margin re-used the search's own perturbation draws"


# --------------------------------------------------------------------------- what the fitter hands back


def test_the_fit_publishes_the_per_parameter_margin_and_the_rollout_cost(monkeypatch):
    """``fit_gait_for_body`` is where a customer-facing verdict is minted, so it is where the error bar has to
    appear — not only in the search result that nothing downstream reads."""
    winner = {"freq": 2.0, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 30.0, "kd": 2.0}

    def fake(_gene, params, *, steps=1200, reward_fn=None):
        default = abs(float(params["freq"]) - 1.5) < 1e-9
        return _res(-0.7 if default else 1.0, not default)

    monkeypatch.setattr(GS, "evaluate_gait", fake)
    # ``deadline`` is accepted and ignored: the real ``_one_search`` takes the build's shared gait-fit clock
    # (2026-08-10), and a double missing it raised TypeError inside ``fit_gait_for_body``'s own except, turning
    # this assertion about the ROBUSTNESS MARGIN into an error payload. Nothing here is timed -- the suite pins
    # the budget off (tests/conftest.py) so a margin never becomes a function of machine speed.
    monkeypatch.setattr(GF, "_one_search",
                        lambda gene, *, db, bank, warm_evals, max_evals, out, kw, deadline=None: (
                            {"beats_default": True, "params": dict(winner), "forward_m": 1.0,
                             "credible": True, "survived": True, "n_evals": 5, "n_rollouts_attempt": 17}, 5))
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=1)
    assert out["adopted"] is True
    assert out["robustness_rel"] == pytest.approx(0.1)
    assert set(out["robustness_per_param"]) == set(winner)
    assert "survives a 0.1 relative perturbation" in out["robustness_note"]
    # The sample sizes must SURVIVE the trip. Rebuilding the margin field-by-field on the way out is how this
    # sentence once read "per parameter (n=None each)" — the one number that makes its two halves comparable.
    assert "n=None" not in out["robustness_note"]
    assert "probes)" in out["robustness_note"]
    assert out["robustness_note"] in out["reason"], "the margin must travel with the verdict, not beside it"
    assert out["n_rollouts"] >= 17, "the fit must report physics rollouts, not only candidates"
