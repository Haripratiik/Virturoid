"""WS-D — the compounding-proof harness: N-sweep, 4 controls, honest verdict (master_plan_v6 §10.4)."""
from __future__ import annotations

from virturoid.services import compounding_proof as CP

_FAMILIES = 6


def _onehot(f):
    v = [0.0] * _FAMILIES
    v[f] = 1.0
    return v


def _world(per_family=2, n_held=6):
    """A synthetic corpus where a warm-start gait helps IFF it comes from the held body's family; embeddings are
    family one-hots (so nearest = same family). Corpus ordered by family so coverage grows with N."""
    embed, gait, fam = {}, {}, {}
    corpus, held = [], []
    for f in range(_FAMILIES):
        for j in range(per_family):
            cid = f"c{f}_{j}"; corpus.append(cid); embed[cid] = _onehot(f); gait[cid] = {"family": f}; fam[cid] = f
    for h in range(n_held):
        f = h % _FAMILIES; hid = f"h{h}"; held.append(hid); embed[hid] = _onehot(f); fam[hid] = f
    return corpus, held, embed, gait, fam


def _family_solver(fam):
    def solve(held_id, warm, seed):
        if warm is None:
            return {"solved": False, "time_to_threshold": 8.0}
        if warm.get("family") == fam[held_id]:
            return {"solved": True, "time_to_threshold": 1.0}
        return {"solved": False, "time_to_threshold": 8.0}
    return solve


# ---------------------------------------------------------------- retrieval conditions
def test_warm_start_conditions():
    import random
    embed = {"h": [1, 0], "a": [1, 0], "b": [0, 1]}
    gait = {"a": {"g": "A"}, "b": {"g": "B"}}
    rng = random.Random(0)
    assert CP.warm_start_for("h", [1, 0], ["a", "b"], condition="retrieval",
                             embed_by_id=embed, gait_by_id=gait, rng=rng) == {"g": "A"}   # nearest
    assert CP.warm_start_for("h", [1, 0], ["a", "b"], condition="no_retrieval",
                             embed_by_id=embed, gait_by_id=gait, rng=rng) is None
    shuf = CP.warm_start_for("h", [1, 0], ["a", "b"], condition="shuffled_label",
                             embed_by_id=embed, gait_by_id=gait, rng=rng)
    assert shuf == {"g": "B"}                              # nearest body 'a', but a DIFFERENT gait returned


def test_bootstrap_ci_bounds_the_mean():
    m, lo, hi = CP.bootstrap_ci([1.0, 1.0, 0.0, 0.0], seed=1)
    assert lo <= m <= hi and 0.0 <= lo <= hi <= 1.0
    assert CP.bootstrap_ci([0.7])[0] == 0.7               # single value degenerate


# ---------------------------------------------------------------- the curve + verdict
def test_curve_detects_real_compounding():
    corpus, held, embed, gait, fam = _world()
    curve = CP.run_compounding_curve(held, corpus, solve_fn=_family_solver(fam), embed_by_id=embed,
                                     gait_by_id=gait, seeds=3)
    v = curve.verdict
    assert v["proven"], v
    assert v["solve_rate_rises"] and v["time_to_threshold_falls"]
    assert v["separated_from_no_retrieval_at_maxN"] and not v["leakage_suspected"]
    # retrieval clearly beats the cold control at max N
    ret_last = [c for c in curve.checkpoints if c["condition"] == "retrieval"][-1]["solve_rate"]["mean"]
    cold_last = [c for c in curve.checkpoints if c["condition"] == "no_retrieval"][-1]["solve_rate"]["mean"]
    assert ret_last > cold_last


def test_flat_curve_is_reported_honestly_not_hidden():
    corpus, held, embed, gait, fam = _world()

    def blind_solver(held_id, warm, seed):
        return {"solved": False, "time_to_threshold": 8.0}   # warm-start never helps -> no compounding
    curve = CP.run_compounding_curve(held, corpus, solve_fn=blind_solver, embed_by_id=embed,
                                     gait_by_id=gait, seeds=2)
    assert not curve.verdict["proven"]                    # honest: flat curve is NOT proven
    assert "gates further factory investment" in curve.verdict["note"]


def test_shuffled_label_control_catches_leakage():
    corpus, held, embed, gait, fam = _world()

    def leaky_solver(held_id, warm, seed):
        # a LEAKY solver: any warm-start solves regardless of whether the gait matches the body
        return {"solved": bool(warm), "time_to_threshold": 1.0 if warm else 8.0}
    curve = CP.run_compounding_curve(held, corpus, solve_fn=leaky_solver, embed_by_id=embed,
                                     gait_by_id=gait, seeds=2)
    # shuffled-label helps as much as retrieval -> the leak detector fires, so compounding is NOT declared
    assert curve.verdict["leakage_suspected"]
    assert not curve.verdict["proven"]


def test_curriculum_ablation_reports_a_gap():
    corpus, held, embed, gait, fam = _world()
    # selection order = coverage-maximising (one body per family first); random will be shuffled internally
    selection = [f"c{f}_0" for f in range(_FAMILIES)] + [f"c{f}_1" for f in range(_FAMILIES)]
    out = CP.curriculum_ablation(held, corpus, solve_fn=_family_solver(fam), embed_by_id=embed,
                                 gait_by_id=gait, selection_order=selection, seeds=2)
    assert "curriculum_gap" in out and out["selection_final_solve_rate"] is not None
