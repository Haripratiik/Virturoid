"""A walk verdict must not be reported from a horizon that ends before the body falls.

THE DEFECT (measured, task #267, 2026-08-01). The grounded authored cat's adopted operating point was reported
as a CREDIBLE WALK and falls at step 2014. The verdict was measured at 1500. The credible walk *was* the fall,
happening 514 steps after we stopped looking.

The mechanism is not a bad simulator and not a knife-edge classifier — it is the optimised quantity. NET
DISPLACEMENT AT A FIXED HORIZON measures locomotion on the canonical template and drift-before-falling on the
animals, and at one horizon those are indistinguishable. Across horizons they are trivially distinguishable, by
the travel RATE (m per 1000 steps at 1500 / 2500 / 4000 / 6000):

    template DEFAULT   1.385 -> 1.440 -> 1.395 -> 1.278   CONSTANT   -> +7.67 m, alive at 6000
    template SEARCHED  2.58  -> 2.80  -> 2.99  -> 3.08    CONSTANT   -> +18.50 m, alive
    cat ADOPTED        0.639 -> 0.300 -> 0.188 -> 0.125   1/t DECAY  -> fell at 2014
    dog ADOPTED        0.953 -> 0.836 -> 0.503 -> 0.230   DECAY      -> alive, but stopped advancing

A 1/t profile IS the signature of "moved, then stopped": the numerator froze and the denominator kept growing.
So a credible walk now requires the RATE TO HOLD at a materially longer horizon, and the verdict carries the
horizon it was measured at plus a ROBUSTNESS MARGIN — how much relative perturbation the operating point
survives. The cat's was below 1e-5 on four of five parameters and the template's >= 1e-1 on all five; both
printed the same two words.

The pure tests below fix the CONTRACT (no simulator). The MuJoCo ones pin the measured defect itself.
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

FE = 5


def _frames(xs, *, roll_deg=0.0, pitch_deg=0.0, spikes=()):
    """A qpos trace [x, y, z, w, qx, qy, qz] with the given base-x samples and a constant attitude.

    ``spikes`` = {index: (roll_deg, pitch_deg)} overrides individual frames, for testing max-vs-sustained.
    """
    out = []
    for i, x in enumerate(xs):
        rd, pd = spikes.get(i, (roll_deg, pitch_deg)) if isinstance(spikes, dict) else (roll_deg, pitch_deg)
        a, b = math.radians(rd) / 2.0, math.radians(pd) / 2.0
        # small-angle composition is enough: the tests use pure roll OR pure pitch
        if pd:
            q = (math.cos(b), 0.0, math.sin(b), 0.0)
        else:
            q = (math.cos(a), math.sin(a), 0.0, 0.0)
        out.append([float(x), 0.0, 0.3, q[0], q[1], q[2], q[3]])
    return out


def _rollout(xs, *, steps, **kw):
    """A metrics dict shaped exactly like ``crawl_gait_rollout``'s, with scalars that all PASS, so the only
    thing under test is the horizon/settling/orientation logic."""
    base = {"survived": True, "upright_frac": 1.0, "height_ratio": 0.9, "cadence": 8.0, "support_frac": 0.5,
            "forward": float(xs[-1] - xs[0]), "steps": int(steps), "frame_every": FE,
            "qpos_frames": _frames(xs, **kw)}
    return base


def _constant(steps, v_per_1000=1.4):
    """Travel at a steady rate for the whole run — a body that is walking."""
    n = steps // FE
    return [v_per_1000 * (FE * (i + 1)) / 1000.0 for i in range(n)]


def _drift_then_stop(steps, stop_at=2000, v_per_1000=0.64):
    """Travel at a steady rate, then freeze — the measured shape of a body that moved and stopped."""
    n = steps // FE
    return [v_per_1000 * min(FE * (i + 1), stop_at) / 1000.0 for i in range(n)]


# ---------------------------------------------------------------------------------------------------------
# THE SETTLING CONTRACT
# ---------------------------------------------------------------------------------------------------------

def test_a_short_rollout_says_it_cannot_judge_settling_rather_than_guessing():
    """The gate is OFF below the minimum horizon. Failing a body for a question its trace cannot answer would
    be a second false verdict, and defaulting to 'holds' would be the first one again."""
    from virturoid.services.gait_quality import _SETTLE_MIN_STEPS, settling
    assert settling(_rollout(_constant(1500), steps=1500)) is None
    assert settling(_rollout(_constant(_SETTLE_MIN_STEPS), steps=_SETTLE_MIN_STEPS)) is not None


def test_a_metrics_only_dict_is_never_forced_through_the_settling_gate():
    """No trace, no horizon -> no settling claim. Back-compat for every caller that passes bare scalars."""
    from virturoid.services.gait_quality import classify, settling
    assert settling({"survived": True, "forward": 2.0}) is None
    assert classify({"survived": True, "upright_frac": 1.0, "height_ratio": 0.9, "cadence": 8.0,
                     "support_frac": 0.5, "forward": 0.8}) == "CREDIBLE WALK"


def test_a_constant_rate_holds_and_a_decaying_rate_does_not():
    from virturoid.services.gait_quality import settling
    good = settling(_rollout(_constant(6000), steps=6000))
    bad = settling(_rollout(_drift_then_stop(6000), steps=6000))
    assert good["holds_rate"] is True
    assert bad["holds_rate"] is False
    # the decay is the measured 1/t shape: once travel freezes at step 2000, only the denominator keeps
    # growing, so the rate at 6000 is exactly 2000/6000 of the un-frozen rate the early horizon saw.
    assert bad["rate_full"] == pytest.approx(bad["rate_early"] / 3.0, rel=0.05)


def test_a_body_that_moved_then_stopped_is_not_a_credible_walk():
    """The dog case: never falls, stays upright the whole way, clears every scalar gate — and has stopped."""
    from virturoid.services.gait_quality import classify
    v = classify(_rollout(_drift_then_stop(6000), steps=6000))
    assert not v.startswith("CREDIBLE"), v
    assert "DRIFTS THEN STALLS" in v
    assert "moved, then stopped" in v
    assert classify(_rollout(_constant(6000), steps=6000)) == "CREDIBLE WALK"


def test_the_rate_profile_travels_with_the_verdict():
    """A net displacement hides the thing that matters, so the profile is reported, not just the total."""
    from virturoid.services.gait_quality import settling, travel_rates
    r = _rollout(_constant(6000), steps=6000)
    rates = travel_rates(r)
    assert sorted(rates) == [1500, 3000, 6000]
    assert all(v == pytest.approx(1.4, rel=0.02) for v in rates.values())
    assert settling(r)["rates"] == rates


def test_a_body_that_starts_backwards_and_ends_up_walking_is_not_punished():
    """`holds_rate` is a COMPARISON, not a ratio: full/early would score a sign flip as a huge negative."""
    from virturoid.services.gait_quality import settling
    n = 6000 // FE
    xs = []
    for i in range(n):
        s_ = FE * (i + 1)
        xs.append(-0.5 * s_ / 1500.0 if s_ <= 1500 else -0.5 + 1.0 * (s_ - 1500) / 1000.0)
    s = settling(_rollout(xs, steps=6000))
    assert s["rate_early"] < 0 < s["rate_full"]
    assert s["holds_rate"] is True


# ---------------------------------------------------------------------------------------------------------
# THE LEVEL GATE MUST BE HORIZON-INVARIANT, or extending the horizon invents a NEW false verdict
# ---------------------------------------------------------------------------------------------------------

def test_one_bad_frame_in_a_long_run_is_not_a_lurch():
    """MEASURED on the grounded hexapod: pitch_max 17.5 (1500 steps) -> 21.5 (6000) -> 22.7 (12000) against a
    20 deg gate, while its P95 stays 17.0 and only 22 of 2400 frames ever exceed 20 deg, the first at step
    4280. That body walks 8.14 m dead straight at a steady rate. A max-based gate is monotone in the horizon,
    so it would have relabelled a real walker a LURCH purely because we looked longer."""
    from virturoid.services.gait_quality import classify, orientation_summary
    xs = _constant(6000)
    spikes = {i: (0.0, 22.7) for i in range(1000, 1010)}          # 10 of 1200 frames, late in the run
    r = _rollout(xs, steps=6000, pitch_deg=17.0, spikes=spikes)
    o = orientation_summary(r["qpos_frames"])
    assert o["pitch_max"] > 20.0 and o["pitch_p95"] < 20.0        # exactly the hexapod's shape
    assert classify(r) == "CREDIBLE WALK"


def test_a_sustained_rear_is_still_rejected():
    """The anti-Goodhart guard the level gate exists for survives the swap to a sustained statistic: a lurch
    pitches ~25 deg EVERY stride, so its P95 is ~25 too."""
    from virturoid.services.gait_quality import classify
    v = classify(_rollout(_constant(6000), steps=6000, pitch_deg=25.0))
    assert not v.startswith("CREDIBLE") and "LURCHES" in v


def test_a_single_catastrophic_frame_is_still_rejected():
    """Reading a percentile must not let a genuine tip through on the technicality that it was brief, so the
    single worst frame keeps an absolute ceiling."""
    from virturoid.services.gait_quality import _HARD_PITCH_MAX, classify
    spikes = {i: (0.0, _HARD_PITCH_MAX + 15.0) for i in range(600, 604)}
    v = classify(_rollout(_constant(6000), steps=6000, pitch_deg=3.0, spikes=spikes))
    assert not v.startswith("CREDIBLE"), v


# ---------------------------------------------------------------------------------------------------------
# THE VERDICT CARRIES ITS HORIZON
# ---------------------------------------------------------------------------------------------------------

def test_evaluate_gait_reports_the_horizon_it_measured_at():
    """A result that decides something must say how long anyone looked."""
    import virturoid.services.gait_search as GS

    def _fake(gene, *, steps, record_qpos=False, frame_every=5, **kw):
        return _rollout(_constant(steps), steps=steps)
    import virturoid.services.morph_policy as MP
    real = MP.crawl_gait_rollout
    MP.crawl_gait_rollout = _fake
    try:
        short = GS.evaluate_gait(None, {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5},
                                 steps=1500)
        long_ = GS.evaluate_gait(None, {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5},
                                 steps=6000)
    finally:
        MP.crawl_gait_rollout = real
    assert short["horizon_steps"] == 1500 and short["holds_rate"] is None and short["rates"] == {}
    assert long_["horizon_steps"] == 6000 and long_["holds_rate"] is True and len(long_["rates"]) == 3


def test_the_fitter_judges_at_the_horizon_it_searched_at():
    """MEASURED (task #267): screening cheaply and judging honestly adopts NOTHING — the cat spends 141 evals
    and the dog 174, because `stop_on_credible` stops at the first candidate the SEARCH horizon calls credible,
    which is precisely a drift-before-fall. Search 6000 / judge 6000 adopts a real walk (cat rate 0.769 ->
    0.636). An optimiser may never be scored at a shorter horizon than the one its answer is judged at."""
    import inspect

    from virturoid.services.gait_flywheel import _SETTLE_STEPS, fit_gait_for_body
    sig = inspect.signature(fit_gait_for_body).parameters
    assert sig["search_steps"].default == sig["deploy_steps"].default == _SETTLE_STEPS
    from virturoid.services.gait_quality import _SETTLE_MIN_STEPS
    assert _SETTLE_STEPS >= _SETTLE_MIN_STEPS, "the fit horizon must be long enough to be judged at"


# ---------------------------------------------------------------------------------------------------------
# THE MEASURED DEFECT, against the real simulator
# ---------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grounded_cat():
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a robot cat", ensure_walkable=False)
    ground_and_repair(g)
    return g


#: the operating point ``fit_gait_for_body`` adopted for the grounded cat and reported as +0.958 m CREDIBLE
#: WALK. It falls at step 2014.
_CAT_ADOPTED = {"freq": 2.213223629, "hip_amp": 0.770197134, "knee_amp": 0.967596586,
                "kp": 174.334675383, "kd": 12.353651216}


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_cats_reported_walk_is_a_fall_and_the_verdict_now_says_so(grounded_cat):
    """The defect itself, pinned. If this ever reads CREDIBLE at the settling horizon again, the product is
    once more certifying a fall as a walk."""
    from virturoid.services.gait_flywheel import _SETTLE_STEPS
    from virturoid.services.gait_quality import classify, settling
    from virturoid.services.morph_policy import crawl_gait_rollout
    r = crawl_gait_rollout(grounded_cat, steps=_SETTLE_STEPS, record_qpos=True, **_CAT_ADOPTED)
    assert r["alive"] < _SETTLE_STEPS, "this op-point is pinned BECAUSE the body falls; it no longer does"
    assert not classify(r).startswith("CREDIBLE"), classify(r)
    s = settling(r)
    assert s["holds_rate"] is False
    assert s["rate_full"] < 0.35 * s["rate_early"], f"the 1/t decay is the signature: {s['rates']}"


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_robustness_margin_separates_a_controller_from_one_lucky_float(grounded_cat):
    """The cat's adopted point does not survive even the coarsest rung; a 2.4e-5 relative change in step
    frequency flipped it from +0.958 m CREDIBLE WALK to +0.500 m FELL by ROLL-OVER. A verdict without this
    number cannot tell that apart from the template, which tolerates >= 1e-1 on all five parameters."""
    from virturoid.services.gait_flywheel import _SETTLE_STEPS, robustness_margin
    rob = robustness_margin(grounded_cat, _CAT_ADOPTED, steps=_SETTLE_STEPS, n=2)
    assert rob["robustness_rel"] is None
    assert "one lucky float" in rob["note"]
    assert all(v.startswith("0/") for v in rob["probes"].values()), rob["probes"]
