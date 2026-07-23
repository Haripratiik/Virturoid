"""R2 (agentic platform plan WS-R): the reward loop must REFLECT -- decompose the reward into terms, name which
saturated, and feed that correction into the next proposal (the Eureka feedback loop). Reflection is what turns
a one-shot best-of-N into a loop that improves; these tests pin the decomposition, the diagnosis, and the
iterate-with-feedback behavior.
"""
from __future__ import annotations

import importlib.util

import pytest

from virturoid.services.reward_dsl import compile_reward
from virturoid.services.reward_reflection import (_split_additive_terms, reward_term_contributions)

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def test_split_additive_terms_handles_subtraction_and_nesting():
    terms = _split_additive_terms("a + 0.2*b - c*d + max(0.0, e)")
    signs = [s for s, _ in terms]
    assert signs == [1.0, 1.0, -1.0, 1.0]
    assert len(terms) == 4


def test_term_contributions_sum_to_the_whole_reward():
    """Decomposing an additive reward must be lossless: the term contributions sum to the reward's value."""
    expr = "exp(-(forward_vel - 0.4)**2 / 0.25) * alive + 0.2*upright - 0.1*energy"
    feats = {k: 0.0 for k in ("forward_vel", "upright", "height_ratio", "contact_frac", "alive", "slip",
                              "foot_clearance", "energy", "action_smooth", "dist_to_goal")}
    feats.update({"forward_vel": 0.15, "upright": 0.8, "alive": 1.0, "energy": 0.3})
    contribs = reward_term_contributions(expr, feats)
    whole = float(compile_reward(expr)(feats))
    assert sum(c["value"] for c in contribs) == pytest.approx(whole, abs=1e-3)


def test_reflection_names_the_saturated_term_and_diagnoses(monkeypatch):
    """A height/upright-farming reward that doesn't move must produce a reflection that flags the saturated
    features and tells the proposer to gate them behind forward progress."""
    if not _MUJOCO:
        pytest.skip("needs MuJoCo")
    from virturoid.services.gait_search import search_gait
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.reward_reflection import build_reflection_payload
    g = compose_robot("a robot dog", llm=None)
    res = search_gait(g, generations=3, pop=8, steps=400, seed=1,
                      reward_fn=compile_reward("2.0*height_ratio + upright"))
    payload = build_reflection_payload(g, res.best_params, "2.0*height_ratio + upright", steps=400)
    assert payload["credible"] == payload["verdict"].upper().startswith(("WALK", "CREDIBLE"))
    assert payload["dominant_term"] is not None
    assert isinstance(payload["diagnosis"], str) and len(payload["diagnosis"]) > 10
    # feature values are honest floats in the payload
    assert set(payload["features"]) >= {"forward_vel", "upright", "height_ratio"}


def test_reflection_text_is_injected_into_the_next_proposal():
    """The reflection block must reach the LLM proposer's prompt so the next round corrects the last failure."""
    from virturoid.services.reward_dsl import propose_rewards
    seen = {}

    class StubLLM:
        def complete(self, prompt):
            seen["prompt"] = prompt
            return "max(0.0, forward_vel)*alive - 0.1*slip"

    propose_rewards("walk forward", n=2, llm=StubLLM(), reflection="SATURATED: upright. Raise forward weight.")
    assert "SATURATED: upright" in seen["prompt"]
    assert "FEEDBACK" in seen["prompt"]


def test_loop_returns_reflection_and_iteration_log():
    """Even a single-iteration run must return a reflection payload + an iteration log (diagnostics always on)."""
    if not _MUJOCO:
        pytest.skip("needs MuJoCo")
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.reward_loop import run_intelligent_reward_loop
    g = compose_robot("a robot dog", llm=None)
    out = run_intelligent_reward_loop(g, task="walk forward", llm=None, n_rewards=3, iterations=1,
                                      screen_generations=2, screen_pop=6, final_generations=3, final_pop=8,
                                      steps=400, seed=3, bank=False)
    assert out["ok"] and "reflection" in out and out["iterations_run"] == 1
    assert out["reflection"]["credible"] == out["credible"]
    assert isinstance(out["iteration_log"], list) and out["iteration_log"]


def test_loop_iterates_when_not_credible_and_stops_early_when_credible():
    """iterations>1 must run multiple rounds while the gait is not credible, and stop the moment one is."""
    if not _MUJOCO:
        pytest.skip("needs MuJoCo")
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.reward_loop import run_intelligent_reward_loop
    g = compose_robot("a robot dog", llm=None)
    out = run_intelligent_reward_loop(g, task="walk forward", llm=None, n_rewards=3, iterations=3,
                                      screen_generations=2, screen_pop=6, final_generations=3, final_pop=8,
                                      steps=400, seed=3, bank=False)
    assert 1 <= out["iterations_run"] <= 3
    # if it ended credible, the last logged round must be the credible one (early stop); if never credible, it
    # used the whole budget.
    if out["credible"]:
        assert out["iteration_log"][-1]["credible"] is True
    else:
        assert out["iterations_run"] == 3
