"""``fit_gait_for_body`` must be robust to the RNG draw, must ship what it verified, and must not disguise a
crash as a finding. Three defects, all measured 2026-08-01 (task #265).

1. SEED FRAGILITY. Same grounded body, same code, same 96-eval budget: the authored dog reaches a CREDIBLE WALK
   on seeds 0/3/4 and FELLS by ROLL-OVER on 1/2; the cat wins 1 of 3. At a fixed ``seed=0`` a customer's robot
   walked or fell on a draw nobody chose. NOT a regression from fitting per body — it was always there, hidden
   because ``ensure_walkable_quad`` substituted the body before anything searched it.
2. VERIFY-ONE-SHIP-ANOTHER. The winner was validated unrounded and CACHED rounded to 4 decimals. On the grounded
   cat (seed 0) that was the difference between +0.958 m CREDIBLE WALK (freq 2.21322363) and +0.500 m FELL by
   ROLL-OVER (freq 2.2132) — a 2.4e-5 relative change. The adoption gate in ``learn_gait_flywheel`` already
   required `credible`; the gates never disagreed, the STORED NUMBERS did.
3. FAIL-OPEN. A blanket ``except`` turned an internal ``KeyError`` into a normal-looking "searched, adopted
   nothing" return in 0 s, so a crash read as a considered decline — the artefact the function exists to remove.

4. FIRST-CREDIBLE-WINS (added 2026-08-02, task #267). The fitter stopped at the first draw the horizon called
   credible and shipped it, so the robustness margin it computed was an annotation on a decision already taken.
   Measured on the grounded authored cat, that draw is 0/8 at rel 1e-2 and 1e-3 and falls at step 8754 against a
   6000-step verdict. Sturdiness now leads the ranking and a fragile win keeps searching.

These are unit tests: the physics is stubbed so the CONTRACT is what is under test, not the simulator.
"""
from __future__ import annotations

import pytest

from virturoid.services import gait_flywheel as GF
from virturoid.services import gait_search as GS


class _Gene:
    def __init__(self):
        self.metadata = {}
        self.id = "test-gene"
        self.robot_class = "quadruped"


_WINNER = {"freq": 2.21322363, "hip_amp": 0.77019713, "knee_amp": 0.96759659,
           "kp": 174.33467538, "kd": 12.35365122}


def _stub_evaluate(credible_for_winner: bool, *, sturdy: bool = True):
    """Default gait never credible (so a search happens); the winner's credibility is the knob under test.

    ``sturdy`` decides whether PERTURBED COPIES of the winner are credible too — i.e. whether the winner is a
    controller or one lucky float. ``robustness_margin`` probes exactly those copies, so this is the knob that
    tells the fitter's robustness gate apart from its credibility gate. A stub in which ONLY the exact winning
    float is credible describes a knife edge, and every op-point measured through it is fragile by construction.
    """
    tol = 0.2 if sturdy else 1e-9                 # relative half-width of the winner's credible neighbourhood
    def _ev(gene, params, *, steps=1200, reward_fn=None):
        is_winner = abs(float(params.get("freq", 0)) - _WINNER["freq"]) <= tol * _WINNER["freq"]
        ok = credible_for_winner and is_winner
        return {"fitness": 1.0 if ok else -1.0, "forward": 0.958 if ok else 0.5,
                "height_ratio": 1.0 if ok else 0.49, "survived": True,
                "cadence": 20.0, "support_frac": 0.9, "credible": ok,
                "verdict": "CREDIBLE WALK" if ok else "FELL by ROLL-OVER (roll 79 deg max)",
                "reward_return": 0.0}
    return _ev


def _stub_search(fail_first: int):
    """Fail ``fail_first`` attempts, then win. Records the seed each attempt was given."""
    seen: list[int] = []

    def _one(gene, *, db, bank, warm_evals, max_evals, out, kw):
        seen.append(int(kw["seed"]))
        won = len(seen) > fail_first
        return ({"beats_default": won, "params": dict(_WINNER), "forward_m": 0.958,
                 "credible": won, "survived": True, "n_evals": 7}, 7)
    return _one, seen


def test_a_failed_draw_is_retried_from_a_fresh_seed(monkeypatch):
    """A failed search is a failed DRAW, not a verdict on the body."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(True))
    one, seen = _stub_search(fail_first=2)
    monkeypatch.setattr(GF, "_one_search", one)
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=3)
    assert out["adopted"] is True
    assert out["seed_attempts"] == 3
    assert len(set(seen)) == 3, f"each attempt must use a DIFFERENT seed, got {seen}"
    assert out["n_evals"] == 21                      # cost is summed across attempts, honestly


def test_a_body_that_wins_first_time_pays_nothing_extra(monkeypatch):
    """A STURDY first win ends the search — the extra draws exist for fragility, not as a tax on success."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(True))
    one, seen = _stub_search(fail_first=0)
    monkeypatch.setattr(GF, "_one_search", one)
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=5)
    assert out["adopted"] is True and out["seed_attempts"] == 1 and len(seen) == 1
    assert out["robustness_rel"] is not None and out["fragile"] is False


def test_a_fragile_win_is_not_the_end_of_the_search(monkeypatch):
    """THE TASK #267 CONTRACT AT THE ADOPTION GATE. A credible walk that no perturbed copy of itself can repeat
    is one lucky float, and the old fitter stopped at the FIRST credible draw and shipped it. MEASURED
    2026-08-02 on the grounded authored cat: the point it adopts is 0/8 at rel 1e-2 AND 1e-3 and is on the floor
    by step 8754, while the verdict is measured at 6000 — the settling defect one horizon out. So a fragile win
    spends the remaining draws looking for a controller instead of banking a coincidence."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(True, sturdy=False))
    one, seen = _stub_search(fail_first=0)
    monkeypatch.setattr(GF, "_one_search", one)
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=3)
    assert len(seen) == 3, "a fragile win must not end the search"
    # ...and when no sturdier point exists, the body still keeps the walk it demonstrably has — declining a body
    # that travels and holds its rate would be the same dishonesty pointed the other way — but it is FLAGGED.
    assert out["adopted"] is True and out["fragile"] is True and out["robustness_rel"] is None
    assert "FRAGILE" in out["reason"] and "NOT banked" in out["reason"]


def test_restarts_are_bounded(monkeypatch):
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(True))
    one, seen = _stub_search(fail_first=99)
    monkeypatch.setattr(GF, "_one_search", one)
    out = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=3)
    assert out["adopted"] is False and out["seed_attempts"] == 3 and len(seen) == 3
    assert out["ok"] is True, "exhausting the budget is a DECLINE, not an error"


def test_the_exact_verified_parameters_are_what_get_stored(monkeypatch):
    """Not a rounded copy of them. Rounding to 4 dp flipped a real cat from CREDIBLE WALK to ROLL-OVER."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(True))
    one, _ = _stub_search(fail_first=0)
    monkeypatch.setattr(GF, "_one_search", one)
    g = _Gene()
    GF.fit_gait_for_body(g, bank=False)
    stored = g.metadata["gait_params"]
    for k, v in _WINNER.items():
        assert stored[k] == v, f"{k} was altered on the way into the cache: {stored[k]} != {v}"


def test_adoption_refuses_anything_the_credibility_gate_rejects(monkeypatch):
    """The stored controller is re-measured; a winner that does not reproduce is NOT adopted."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(False))   # winner fails re-measurement
    one, _ = _stub_search(fail_first=0)
    monkeypatch.setattr(GF, "_one_search", one)
    g = _Gene()
    out = GF.fit_gait_for_body(g, bank=False, seed_restarts=1)
    assert out["adopted"] is False
    assert "did not reproduce" in out["reason"]
    assert "gait_params" not in g.metadata, "a rejected gait must never reach the deploy cache"


def test_a_crash_is_reported_as_an_error_not_as_a_finding(monkeypatch):
    """A swallowed exception must not read as 'this body has no better gait'."""
    monkeypatch.setattr(GS, "evaluate_gait", _stub_evaluate(True))

    def _boom(*a, **k):
        raise KeyError("duty")
    monkeypatch.setattr(GF, "_one_search", _boom)
    g = _Gene()
    out = GF.fit_gait_for_body(g, bank=False)
    assert out["ok"] is False and out["adopted"] is False
    assert "KeyError" in out["error"]
    assert "FAILED" in out["reason"] and "never measured" in out["reason"]
    assert g.metadata["gait_fit"]["ok"] is False          # the disclosure carries it too
    # and a genuine DECLINE stays distinguishable
    one, _ = _stub_search(fail_first=99)
    monkeypatch.setattr(GF, "_one_search", one)
    decline = GF.fit_gait_for_body(_Gene(), bank=False, seed_restarts=1)
    assert decline["ok"] is True and decline["adopted"] is False and "error" not in decline


def test_a_body_that_already_walks_is_never_churned(monkeypatch):
    """The additive guarantee: the shipped default wins outright and no search runs."""
    def _ev(gene, params, *, steps=1200, reward_fn=None):
        return {"fitness": 1.0, "forward": 1.149, "height_ratio": 0.99, "survived": True,
                "cadence": 17.0, "support_frac": 0.98, "credible": True, "verdict": "CREDIBLE WALK",
                "reward_return": 1.0}
    monkeypatch.setattr(GS, "evaluate_gait", _ev)

    def _never(*a, **k):
        raise AssertionError("a body that already walks must not be searched")
    monkeypatch.setattr(GF, "_one_search", _never)
    out = GF.fit_gait_for_body(_Gene(), bank=False)
    assert out["ok"] is True and out["searched"] is False and out["adopted"] is False
