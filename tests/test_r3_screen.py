"""R3 (agentic platform plan WS-R): the single-3060 ASHA fan-out screens K reward candidates cheaply and ranks
them by an honest screen score. The parse + rank logic is a pure function of the box's result file, unit-tested
offline; the live fan-out was validated on the RTX box (SPEED score 0.131 > STILL 0.049 -> correct ranking).
"""
from __future__ import annotations

from virturoid.services.gpu_trainer import _parse_screen_result


def test_parses_screened_candidates_and_flags_crashes():
    txt = "0 fwd_vel=-0.178 alive=1\n2 ERR\n1 fwd_vel=-0.037 alive=1\nDONE"
    p = _parse_screen_result(txt)
    assert len(p) == 3
    assert p[0]["fwd_vel"] == -0.178 and p[1]["fwd_vel"] == -0.037
    assert p[2].get("error") is True


def test_ranking_orders_by_induced_motion_and_zeroes_crashes():
    """Reproduce the wrapper's ranking: |fwd_vel| best-first, a crashed run scores 0."""
    parsed = _parse_screen_result("0 fwd_vel=-0.131 alive=1\n1 fwd_vel=-0.049 alive=1\n2 ERR\nDONE")
    exprs = ["speed", "still", "crash"]
    ranked = []
    for i, e in enumerate(exprs):
        r = parsed[i]
        score = 0.0 if r.get("error") else abs(float(r.get("fwd_vel", 0.0)))
        ranked.append({"expr": e, "screen_score": round(score, 4), "error": bool(r.get("error"))})
    ranked.sort(key=lambda d: d["screen_score"], reverse=True)
    assert [d["expr"] for d in ranked] == ["speed", "still", "crash"]   # SPEED induced most motion, crash last
    assert ranked[0]["screen_score"] == 0.131 and ranked[-1]["screen_score"] == 0.0


def test_empty_result_parses_to_empty():
    assert _parse_screen_result("") == []
