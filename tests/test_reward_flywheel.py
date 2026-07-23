"""R6 (agentic platform plan WS-R / WS-F): the moat, threaded through the reward loop. A reward EXPRESSION that
certified a credible gait on one body must be banked keyed by morphology and RECALLED as a seed for the next
morphology-nearby body -- so the second robot's reward search starts from a reward already proven on something
shaped like it, not from generic templates. These tests pin bank->recall->seed and the honesty gates.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from types import SimpleNamespace

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composing bodies + vector memory needs MuJoCo")

_GAIT = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}


def _db():
    from virturoid.services.memory_db import MemoryDB
    return MemoryDB(os.path.join(tempfile.mkdtemp(), "moat.db"))


def _compose(prompt):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt, llm=None)


def _result(*, credible=True, forward=0.42, hr=0.9):
    return SimpleNamespace(best_survived=True, best_forward=forward, best_credible=credible,
                           best_params=dict(_GAIT), best_height_ratio=hr)


def test_bank_stores_reward_expr_in_base_config():
    import json
    from virturoid.services.gait_flywheel import bank_gait
    db = _db()
    sid = bank_gait(db, _compose("a robot dog"), _result(), reward_expr="max(0.0, forward_vel)*alive - 0.1*slip")
    assert sid is not None
    row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (sid,)).fetchone()
    bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
    assert bc["reward_expr"] == "max(0.0, forward_vel)*alive - 0.1*slip"
    db.close()


def test_recall_returns_a_reward_proven_on_a_nearby_body():
    from virturoid.services.gait_flywheel import bank_gait, recall_reward_hints
    db = _db()
    bank_gait(db, _compose("a robot dog"), _result(), reward_expr="max(0.0, forward_vel)*alive - 0.1*slip")
    hints = recall_reward_hints(db, _compose("a four-legged robot dog"), k=6)
    assert any(h["reward_expr"] == "max(0.0, forward_vel)*alive - 0.1*slip" for h in hints)
    assert all(0.0 <= h["similarity"] <= 1.0 for h in hints)
    db.close()


def test_recall_is_empty_on_a_cold_bank():
    from virturoid.services.gait_flywheel import recall_reward_hints
    db = _db()
    assert recall_reward_hints(db, _compose("a robot dog"), k=6) == []
    db.close()


def test_a_non_credible_result_banks_nothing_so_no_bad_reward_is_recalled():
    """The bank stays a bank of WORKING controllers -- a reward that produced a slide/fall is never banked, so it
    can never be recalled to poison the next body's search."""
    from virturoid.services.gait_flywheel import bank_gait, recall_reward_hints
    db = _db()
    assert bank_gait(db, _compose("a robot dog"), _result(credible=False), reward_expr="1000.0*height_ratio") is None
    assert recall_reward_hints(db, _compose("a robot dog"), k=6) == []
    db.close()


def test_loop_recalls_and_seeds_from_the_bank():
    """With a reward banked for a nearby body, the loop must recall it (reward_hints_recalled>0) and enter it
    into the same honest screen as the fresh proposals."""
    from virturoid.services.gait_flywheel import bank_gait
    from virturoid.services.reward_loop import run_intelligent_reward_loop
    db = _db()
    bank_gait(db, _compose("a robot dog"), _result(), reward_expr="max(0.0, forward_vel)*alive - 0.1*slip")
    out = run_intelligent_reward_loop(_compose("a four-legged robot dog"), task="walk forward", llm=None,
                                      n_rewards=3, screen_generations=1, screen_pop=4, final_generations=1,
                                      final_pop=4, steps=200, seed=5, bank=False, db=db)
    assert out["ok"] and out["reward_hints_recalled"] >= 1
    db.close()
