"""R1 (agentic platform plan, the load-bearing item): LLM-authored DSL rewards must actually STEER training,
selection must rank by a code-owned TRUSTED success (never the reward's own value), and the final verdict must
stay honest. Before R1 the gait search optimized a hardcoded fitness and the reward DSL could steer nothing.
"""
from __future__ import annotations

import importlib.util

import pytest

from virturoid.services.gait_search import evaluate_gait, reward_features_from_rollout, search_gait
from virturoid.services.reward_dsl import RewardCandidate, compile_reward, select_reward

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="the gait search needs MuJoCo")


def _dog():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a robot dog", llm=None)


def test_feature_extractor_reports_real_and_neutral_features_honestly():
    """The 6 measured features come from the rollout; the 4 unavailable ones are 0.0 (neutral), never faked."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    r = crawl_gait_rollout(_dog(), steps=300, record_qpos=True)
    f = reward_features_from_rollout(r)
    assert set(f) == {"forward_vel", "upright", "height_ratio", "contact_frac", "alive", "slip",
                      "foot_clearance", "energy", "action_smooth", "dist_to_goal"}
    assert f["foot_clearance"] == 0.0 and f["energy"] == 0.0 and f["action_smooth"] == 0.0  # unavailable -> neutral
    assert -1.0 <= f["upright"] <= 1.0 and 0.0 <= f["contact_frac"] <= 1.0                    # real, in-range
    # M6 (2026-07-24 audit): `alive` is the documented 0/1 survival flag, NOT the raw step count (was ~500x).
    assert f["alive"] in (0.0, 1.0), f"alive must be the 0/1 flag, got {f['alive']}"
    assert reward_features_from_rollout({"survived": True, "alive": 500})["alive"] == 1.0     # step count -> flag
    assert reward_features_from_rollout({"survived": False, "alive": 137})["alive"] == 0.0


def test_a_reward_steers_evaluate_gait_and_success_is_separate():
    """With a reward_fn the fitness IS the reward return; classify's credible/verdict are computed the same way
    regardless, so success is never the reward's to define."""
    g = _dog()
    params = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
    default = evaluate_gait(g, params, steps=400)
    with_reward = evaluate_gait(g, params, steps=400, reward_fn=compile_reward("100.0*height_ratio"))
    assert with_reward["fitness"] == pytest.approx(with_reward["reward_return"])       # reward steers fitness
    assert with_reward["fitness"] != default["fitness"]                                # a real reward changes it
    assert with_reward["credible"] == default["credible"]                              # success independent of reward


def test_different_rewards_produce_different_gaits():
    """The point of R1: swap the reward, get a different trained gait."""
    g = _dog()
    fast = search_gait(g, generations=3, pop=8, steps=350, seed=1,
                       reward_fn=compile_reward("max(0.0, forward_vel)*alive - 0.1*slip"))
    still = search_gait(g, generations=3, pop=8, steps=350, seed=1,
                        reward_fn=compile_reward("upright*alive - abs(forward_vel)"))
    assert fast.best_params != still.best_params
    assert abs(fast.best_forward - still.best_forward) > 1e-3    # the speed reward travels; the still one doesn't


def test_selection_ranks_by_trusted_success_not_reward_value():
    """A reward that racks up huge return but yields no credible walk must LOSE to one that produces real
    forward progress — ranking is by the code-owned trusted success, not the reward's own number."""
    from virturoid.services.reward_loop import _steered_rollout_fn
    g = _dog()
    rollout = _steered_rollout_fn(g, generations=2, pop=6, steps=300, seed=2)
    honest = RewardCandidate(name="honest", expr="max(0.0,forward_vel)*alive - 0.1*slip",
                             compiled=compile_reward("max(0.0,forward_vel)*alive - 0.1*slip"))
    hack = RewardCandidate(name="hack", expr="1000.0*height_ratio", compiled=compile_reward("1000.0*height_ratio"))
    sel = select_reward([honest, hack], rollout)
    assert sel["best"] is not None and sel["best"].name == "honest"       # trusted success wins, not the big number
    assert honest.reward_return < hack.reward_return or honest.trusted_success > hack.trusted_success


def test_full_loop_is_honest_and_never_banks_a_non_credible_gait():
    """The end-to-end loop returns a verdict consistent with classify(), and only a CREDIBLE gait is banked —
    it never claims a walk it didn't produce."""
    from virturoid.services.reward_loop import run_intelligent_reward_loop
    out = run_intelligent_reward_loop(_dog(), task="walk forward", llm=None, n_rewards=3,
                                      screen_generations=2, screen_pop=6, final_generations=3, final_pop=8,
                                      steps=400, seed=3, bank=False)
    assert out["ok"] is True
    assert out["credible"] == out["verdict"].startswith("CREDIBLE")       # the flag never disagrees with the verdict
    assert isinstance(out["reward_expr"], (str, type(None))) and out["n_candidates"] >= 1


def test_batched_reward_matches_scalar_and_vectorizes():
    """R1 MJX-port primitive: compile_reward_batched evaluates the SAME reward over arrays (numpy or jax.numpy),
    matching the scalar compiler element-wise, so the CPU search and the GPU trainer share one reward."""
    import numpy as np
    from virturoid.services.reward_dsl import (REWARD_FEATURES, _TEMPLATES, compile_reward,
                                               compile_reward_batched)
    expr = "exp(-(forward_vel - 0.4)**2 / 0.25) * alive + 0.2*upright - 0.1*energy"
    scalar, batched = compile_reward(expr), compile_reward_batched(expr)
    rng = np.random.default_rng(0)
    for _ in range(30):
        f = {k: float(rng.uniform(-1, 1)) for k in REWARD_FEATURES}
        fb = {k: np.array([f[k]]) for k in REWARD_FEATURES}
        assert abs(float(batched(fb)[0]) - scalar(f)) < 1e-9
    # vectorizes over a batch of "envs" and stays finite
    fb = {k: rng.uniform(-1, 1, 256) for k in REWARD_FEATURES}
    out = batched(fb)
    assert out.shape == (256,) and bool(np.all(np.isfinite(out)))
    # every template compiles under the batched backend too
    assert all(compile_reward_batched(e) is not None for e in _TEMPLATES.values())


def test_reward_dsl_rejects_exponent_dos():
    """B0b: 9**9**9 passes the allowlist but hangs eval building a 370M-digit int. A constant exponent >8 or a
    huge constant base is rejected fast (features are in [-1,1]); legit powers like **2 still compile."""
    import pytest as _pt
    from virturoid.services.reward_dsl import RewardSyntaxError, compile_reward
    for dos in ("9**9**9", "(10**9)**(10**9)", "forward_vel**100", "1000000000000**upright"):
        with _pt.raises(RewardSyntaxError):
            compile_reward(dos)
    assert compile_reward("(forward_vel - 0.4)**2") is not None      # squared-error pattern still legal
    # the RCE surface stays closed
    for rce in ("__import__('os')", "().__class__", "open('x')"):
        with _pt.raises(RewardSyntaxError):
            compile_reward(rce)
