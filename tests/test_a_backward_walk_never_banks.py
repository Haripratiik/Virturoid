"""``forward`` IS SIGNED, and every rule that reads it has to be.

``morph_policy``'s ``forward`` is world-frame delta-x: +x is the direction the body set out in, so -0.5 means it
walked half a metre BACKWARD. ``gait_quality.classify`` has always known that — its scalar gate is ``fwd >= 0.3``,
signed — but three rules downstream of it compared ``abs(forward)`` instead, and an unsigned comparison ranks a
body walking backward faster ABOVE one walking forward:

  * ``learn_gait_flywheel``'s deploy comparison, ``abs(learned) > abs(default) + 0.02`` (fixed 2026-08-10)
  * ``bank_gait``'s admission gate, ``abs(best_forward) < 0.15`` -> return None
  * ``_rank``'s tie-break, which orders two search winners once sturdiness has tied

The last two are the ones that write and read the CORPUS, which is why they are tested here rather than left as a
comment: a bank does not tell you afterwards which of its rows walked the wrong way, and the row is recalled as a
warm start by the next morphologically-similar body. ``best_credible`` covers every caller that measured a
verdict, but it DEFAULTS TO TRUE for back-compat and ``corpus_factory`` / ``r2prime`` both hand-build their
result doubles — so the gate on this line is the only thing standing between a backward row and the moat.
"""
from __future__ import annotations

import os

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class _Result:
    """The duck-typed shape ``bank_gait`` reads, with NO ``best_credible`` — the back-compat caller."""

    def __init__(self, forward: float):
        self.best_params = {"freq": 2.0, "hip_amp": 0.7, "knee_amp": 1.1, "kp": 90.0, "kd": 4.0}
        self.best_forward = float(forward)
        self.best_height_ratio = 0.9
        self.best_survived = True


def test_a_body_that_walked_backward_is_not_banked(tmp_path):
    from virturoid.services.gait_flywheel import bank_gait
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a four legged robot dog", llm=None)
    with MemoryDB(tmp_path / "m.db") as db:
        assert bank_gait(db, gene, _Result(-0.9), door="test") is None, \
            "-0.9 m is nine tenths of a metre BACKWARD; |0.9| >= 0.15 admitted it"
        assert bank_gait(db, gene, _Result(+0.9), door="test") is not None, "a forward walk still banks"


def test_success_rate_is_never_earned_by_going_the_wrong_way(tmp_path):
    """The gate above is the first line; this is what it protects. ``success_rate`` is what ``mine_gait_hints``
    ranks by and what a warm start is chosen on, so a backward row scoring 0.6 does not merely sit in the bank —
    it outranks a forward row that travelled less."""
    from virturoid.services.gait_flywheel import bank_gait
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a four legged robot dog", llm=None)
    with MemoryDB(tmp_path / "m.db") as db:
        assert bank_gait(db, gene, _Result(-1.4), door="test") is None
        rows = [s for s in db.skills_for_class(gene.robot_class, limit=50)
                if str(s.get("skill_id", "")).startswith("gait::")]
        assert not rows, f"a backward walk reached the corpus: {rows}"
        assert bank_gait(db, gene, _Result(+1.4), door="test") is not None
        banked = [s for s in db.skills_for_class(gene.robot_class, limit=50)
                  if str(s.get("skill_id", "")).startswith("gait::")]
        assert len(banked) == 1 and float(banked[0]["success_rate"]) > 0.9, banked


def test_the_search_winner_tie_break_is_signed():
    """``_rank`` orders two winners by sturdiness, then by distance. Latent (it is only reached once
    ``beats_default`` holds, and that now needs a signed win by a credible winner) and asserted anyway, because
    latent is exactly how the unsigned form got copied to three sites in the first place."""
    from virturoid.services.gait_flywheel import _rank
    fwd = {"beats_default": True, "robustness_rel": 0.2, "forward_m": 0.4}
    back = {"beats_default": True, "robustness_rel": 0.2, "forward_m": -1.9}
    assert _rank(fwd) > _rank(back), "-1.9 m is a fall down the +x axis backwards, not a 1.9 m walk"
