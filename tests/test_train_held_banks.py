"""Flywheel breakthrough R0 — train_held banks its winning gait (measured dead loop, §3.I1).

Before: run_train_gene_job found working gait params via the CPU search and DISCARDED them (never banked as a
reusable skill), so every train run threw away a hard-won controller. This asserts the loop is closed.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
def test_train_held_banks_a_credible_gait(tmp_path, monkeypatch):
    from virturoid.services import session_state as S
    from virturoid.services.agent_design_tools import run_train_gene_job
    from virturoid.services.morphology_composer import compose_robot

    # confine the flywheel DB writes to a temp dir (never the live build/memory)
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)

    g = compose_robot("a sturdy four-legged walking robot", ensure_walkable=True)
    rid = S.put_robot(g, prompt="a sturdy four-legged walking robot")
    out = run_train_gene_job({"robot_id": rid, "max_evals": 3})

    assert out.get("mode") == "gait_search"
    assert "banked_gait" in out                              # the contract now carries the banked skill id (or None)
    if out.get("solved"):
        # a solved search cleared the un-gameable forward+cadence+upright gates -> its winner MUST be banked
        assert out["banked_gait"], "a solved gait search must bank its winning params"
        from virturoid.services.memory_db import MemoryDB
        with MemoryDB(tmp_path / "virturoid_memory.db") as db:
            row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?",
                                  (out["banked_gait"],)).fetchone()
            assert row is not None                           # the skill is really persisted, recallable next build
