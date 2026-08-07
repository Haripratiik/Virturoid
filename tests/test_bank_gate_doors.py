"""EVERY door into the gait bank has to say what it knows about the row it wrote.

The fragility gate landed in ONE of ``bank_gait``'s callers. The other three — the ordinary verify path,
``r2prime.seed_corpus`` and ``reward_loop`` — passed no margin at all, so their rows entered stamped exactly like
rows that predate the gate entirely. Measured on the live bank at the time: 101 rows, ``gated_fraction`` 0.000.

These tests pin what each door can and cannot supply:

* ``r2prime.seed_corpus`` and ``reward_loop`` MEASURE one (they already spend hundreds of rollouts; the ladder is
  a few percent) and the reading is stamped on the row.
* the ordinary verify path does NOT, because measuring there is 4-12 settling-horizon rollouts on every build.
  It inherits the per-body fit's margin when the fit measured THESE EXACT parameters, and otherwise declares
  itself ungated WITH THE REASON — never silently.
* a row whose inherited margin says FRAGILE is not banked at all, which is what ``fit_gait_for_body`` and
  ``verify_robot`` already tell the customer happens.
* and no write may ever DOWNGRADE what is known: keep-best is ``>=`` on a saturating success rate, so an
  unmeasured re-bank of a gated body used to erase the stamp into something indistinguishable from history.
"""
from __future__ import annotations

import json
import tempfile
import types
from pathlib import Path

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.gait_flywheel import (BANK_GATE, BANK_GATE_FRAGILE, BANK_UNGATED, LOCOMOTION, bank_gait,
                                              gate_census, gate_of)

_P = {"freq": 1.8, "hip_amp": 0.8, "knee_amp": 1.1, "kp": 60.0, "kd": 4.0}
_STURDY = {"robustness_rel": 0.01, "probes": {"0.1": "3/4", "0.01": "4/4"}, "steps": 6000}
_FRAGILE = {"robustness_rel": None, "probes": {"0.1": "0/4", "0.01": "0/4", "0.001": "0/4"}, "steps": 6000}


def _db(tmp=None):
    from virturoid.services.memory_db import MemoryDB
    return MemoryDB(db_path=(Path(tmp or tempfile.mkdtemp(prefix="doors_")) / "m.db"))


def _quad(tag: str, *, leg_len: float = 0.18) -> RobotGene:
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.5, radius_m=0.1)]
    for i in range(4):
        prev = "torso"
        for j in range(3):
            nm = f"leg{i}_{j}"
            segs.append(GeneSegment(name=nm, parent=prev, joint_type="revolute",
                                    length_m=leg_len, radius_m=0.02))
            prev = nm
    segs[-1].is_end_effector = True
    return RobotGene(id=tag, species=f"t.{tag}", robot_class="quadruped", segments=segs,
                     base_mount="free", end_effector_type="none")


def _result(params, forward=1.2):
    return types.SimpleNamespace(best_survived=True, best_forward=float(forward), best_credible=True,
                                 best_params=dict(params), best_height_ratio=0.8)


def _bc(db, skill_id) -> dict:
    row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
    return json.loads(row["base_config"])


def _credible_rollout(forward=1.2) -> dict:
    """A rollout dict that ``gait_quality.classify`` calls a CREDIBLE WALK on the scalars alone."""
    return {"survived": True, "upright_frac": 0.95, "cadence": 2.0, "support_frac": 0.6,
            "forward": float(forward), "height_ratio": 0.85}


# ----------------------------------------------------------------------------- the three answers a row can give
def test_measured_fragile_and_undeclared_are_three_distinguishable_stamps():
    db = _db()
    a = bank_gait(db, _quad("sturdy"), _result(_P), robustness=_STURDY, door="unit")
    b = bank_gait(db, _quad("fragile", leg_len=0.22), _result({**_P, "freq": 2.2}), robustness=_FRAGILE,
                  door="unit")
    c = bank_gait(db, _quad("declared", leg_len=0.26), _result({**_P, "freq": 2.6}),
                  ungated_reason="measuring here costs 4-12 rollouts per build", door="unit")
    d = bank_gait(db, _quad("silent", leg_len=0.30), _result({**_P, "freq": 3.0}))
    assert gate_of(_bc(db, a)) == BANK_GATE and _bc(db, a)["robustness_rel"] == 0.01
    # measured AND FAILED is its own word: it cost rollouts to learn, and it is not "unknown"
    assert gate_of(_bc(db, b)) == BANK_GATE_FRAGILE and _bc(db, b)["robustness_rel"] is None
    assert _bc(db, b)["robustness_probes"] == _FRAGILE["probes"]
    # not measured, and the caller says WHY — a later mining run can exclude it by fact, not by inference
    assert gate_of(_bc(db, c)) == BANK_UNGATED
    assert "4-12 rollouts" in _bc(db, c)["bank_gate_reason"]
    # a caller that has not been taught the gate still reads as pre-gate, never as "fine"
    assert _bc(db, d)["bank_gate"] is None and gate_of(_bc(db, d)) == "ungated"
    assert _bc(db, d)["bank_gate_reason"] is None
    db.close()


def test_only_the_passing_stamp_survives_gated_only_mining():
    """``mine_gait_hints(gated_only=True)`` must keep the MEASURED-AND-PASSED rows only. A fragile row is a
    measurement, but it is a measurement that this operating point is one lucky float."""
    from virturoid.services.gait_hints import mine_gait_hints
    db = _db()
    for i in range(2):
        bank_gait(db, _quad(f"ok{i}", leg_len=0.18 + 0.01 * i), _result({**_P, "freq": 1.6 + 0.1 * i}),
                  robustness=_STURDY, door="unit")
    bank_gait(db, _quad("frg", leg_len=0.24), _result({**_P, "freq": 2.9}), robustness=_FRAGILE, door="unit")
    bank_gait(db, _quad("dec", leg_len=0.28), _result({**_P, "freq": 3.1}), ungated_reason="no margin here",
              door="unit")
    pooled = mine_gait_hints(db, robot_class="quadruped")
    only = mine_gait_hints(db, robot_class="quadruped", gated_only=True)
    assert pooled["n"] == 4 and (pooled["n_gated"], pooled["n_ungated"]) == (2, 2)
    assert only["n"] == 2
    db.close()


def test_a_write_may_never_downgrade_what_is_known_about_a_banked_row():
    """keep-best is ``>=`` on a success rate that SATURATES at ``_FWD_NORM``, so two good walks both score 1.0 and
    the later write wins by default. An unmeasured one winning erases the margin AND the stamp, leaving a row
    that reads exactly like one banked before the gate existed."""
    db = _db()
    g = _quad("keep")
    sid = bank_gait(db, g, _result(_P, forward=1.6), robustness=_STURDY, door="learn_gait_flywheel")
    assert gate_of(_bc(db, sid)) == BANK_GATE
    # a STRICTLY BETTER-SCORING unmeasured write for the same body
    again = bank_gait(db, g, _result({**_P, "kp": 99.0}, forward=99.0), ungated_reason="no margin", door="verify")
    assert again == sid                                   # the bank does hold this body's gait...
    bc = _bc(db, sid)
    assert gate_of(bc) == BANK_GATE and bc["robustness_rel"] == 0.01     # ...and it is still the measured one
    assert bc["gait_params"]["kp"] == 60.0 and bc["bank_door"] == "learn_gait_flywheel"
    # the reverse IS allowed: a measurement may always replace an unmeasured row
    u = _quad("up")
    bank_gait(db, u, _result(_P, forward=1.6), ungated_reason="no margin", door="verify")
    sid2 = bank_gait(db, u, _result(_P, forward=1.6), robustness=_STURDY, door="learn_gait_flywheel")
    assert gate_of(_bc(db, sid2)) == BANK_GATE
    db.close()


def test_the_census_answers_the_gated_fraction_per_door():
    db = _db()
    bank_gait(db, _quad("d1"), _result(_P), robustness=_STURDY, door="learn_gait_flywheel")
    bank_gait(db, _quad("d2", leg_len=0.2), _result({**_P, "freq": 2.0}), robustness=_FRAGILE, door="reward_loop")
    bank_gait(db, _quad("d3", leg_len=0.22), _result({**_P, "freq": 2.2}), ungated_reason="r", door="verify_robot")
    bank_gait(db, _quad("d4", leg_len=0.24), _result({**_P, "freq": 2.4}))
    c = gate_census(db)
    assert c["rows"] == 4 and c["gated_fraction"] == 0.25
    assert c["by_gate"] == {BANK_GATE: 1, BANK_GATE_FRAGILE: 1, BANK_UNGATED: 1, "ungated": 1}
    assert c["by_door"]["learn_gait_flywheel"]["gated_fraction"] == 1.0
    assert c["by_door"]["verify_robot"]["gated_fraction"] == 0.0
    assert c["by_door"]["unnamed"]["rows"] == 1           # rows from before doors were named stay visible
    db.close()


# ----------------------------------------------------------------------------- DOOR 1: the ordinary verify path
def test_verify_door_banks_explicitly_ungated_when_no_margin_exists(tmp_path, monkeypatch):
    """The common case, measured on the product: a body the shipped default already walks never searches, so
    ``fit_gait_for_body`` short-circuits and there is NO margin to inherit. That row must say so on its face."""
    from virturoid.services import ai_native_tools as AIT
    from virturoid.services import memory_db as MD
    monkeypatch.setattr(MD, "DEFAULT_DB_PATH", tmp_path / "m.db")
    sid = AIT._auto_bank_gait(_quad("nofit"), _credible_rollout(), {})
    with _db(tmp_path) as db:
        bc = _bc(db, sid)
    assert gate_of(bc) == BANK_UNGATED and bc["bank_door"] == "verify_robot"
    assert "every build" in bc["bank_gate_reason"]        # the COST is the reason, stated on the row
    assert bc["robustness_rel"] is None


def test_verify_door_inherits_the_margin_the_per_body_fit_already_paid_for(tmp_path, monkeypatch):
    """When the fit searched, adopted and measured THESE parameters, the error bar is already in hand — banking
    it gated costs zero extra rollouts, which is the only way this door can ever be gated."""
    from virturoid.services import ai_native_tools as AIT
    from virturoid.services import gait_flywheel as GF
    from virturoid.services import memory_db as MD
    monkeypatch.setattr(MD, "DEFAULT_DB_PATH", tmp_path / "m.db")
    monkeypatch.setattr(GF, "robustness_margin", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the verify path must never pay for a margin")))
    g = _quad("fitted")
    g.metadata = {"gait_params": dict(_P),
                  "gait_fit": {"adopted": True, "searched": True, "params": dict(_P), "horizon_steps": 6000,
                               "robustness_rel": 0.01, "robustness_probes": _STURDY["probes"],
                               "robustness_per_param": {"freq": 0.01}, "fragile": False}}
    sid = AIT._auto_bank_gait(g, _credible_rollout(), dict(_P))
    with _db(tmp_path) as db:
        bc = _bc(db, sid)
    assert gate_of(bc) == BANK_GATE and bc["bank_door"] == "verify_robot"
    assert bc["robustness_rel"] == 0.01 and bc["robustness_steps"] == 6000


def test_a_margin_is_never_borrowed_across_a_different_operating_point(tmp_path, monkeypatch):
    """These walks sit on a knife edge — a 2.4e-5 relative change in ``freq`` has flipped a grounded body from
    +0.958 m CREDIBLE WALK to +0.500 m FELL. So an inherited error bar requires EXACT parameter identity; a
    near-miss must fall back to ungated rather than describe a controller nobody measured."""
    from virturoid.services import ai_native_tools as AIT
    from virturoid.services import memory_db as MD
    monkeypatch.setattr(MD, "DEFAULT_DB_PATH", tmp_path / "m.db")
    g = _quad("nearmiss")
    g.metadata = {"gait_fit": {"adopted": True, "params": dict(_P), "horizon_steps": 6000,
                               "robustness_rel": 0.1, "robustness_probes": {"0.1": "4/4"}}}
    sid = AIT._auto_bank_gait(g, _credible_rollout(), {**_P, "freq": _P["freq"] * (1 + 2.4e-5)})
    with _db(tmp_path) as db:
        bc = _bc(db, sid)
    assert gate_of(bc) == BANK_UNGATED and bc["robustness_rel"] is None


def test_the_verify_door_does_not_bank_an_operating_point_its_own_fit_called_fragile(tmp_path, monkeypatch):
    """MEASURED 2026-08-07 on the composed ``a robot dog``: the fitter adopted freq 2.6189 / kp 59.57, measured
    it 0/4 at every rung, and ``learn_gait_flywheel`` refused it — the bank stayed empty. Verify then deployed
    the same point, got CREDIBLE WALK at its shorter horizon, and banked it. Both ``fit_gait_for_body``'s
    disclosure and verify's own ``robustness_note`` say the gait is NOT banked for reuse; this makes that true."""
    from virturoid.services import ai_native_tools as AIT
    from virturoid.services import memory_db as MD
    monkeypatch.setattr(MD, "DEFAULT_DB_PATH", tmp_path / "m.db")
    g = _quad("lucky")
    g.metadata = {"gait_fit": {"adopted": True, "params": dict(_P), "horizon_steps": 6000, "fragile": True,
                               "robustness_rel": None, "robustness_probes": _FRAGILE["probes"]}}
    assert AIT._auto_bank_gait(g, _credible_rollout(), dict(_P)) is None
    with _db(tmp_path) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM skills WHERE task_type=?", (LOCOMOTION,)).fetchone()[0] == 0


# ----------------------------------------------------------- DOOR 2 and DOOR 3: the doors that CAN afford to pay
def test_r2prime_seed_corpus_measures_the_margin_and_stamps_it(tmp_path, monkeypatch):
    """Seeding one reference already costs ``seeds`` searches; the 4-12 rollout ladder on the ONE point it banks
    is a few percent on top. A fragile reference is still banked (dropping references would confound "retrieval
    helps" with "we curated harder") — but stamped, so gated-only mining never sees it."""
    from virturoid.services import gait_flywheel as GF
    from virturoid.services import gait_search as GS
    from virturoid.services import r2prime as R
    bodies = {"ref-sturdy": _quad("r2sturdy"), "ref-fragile": _quad("r2fragile", leg_len=0.26)}
    seen = {}

    monkeypatch.setattr(R, "_compose", lambda p: bodies[p])
    monkeypatch.setattr(GS, "search_gait", lambda g, **k: types.SimpleNamespace(best_params=dict(_P)))
    # the SEARCHED point beats the shipped-default candidate, so ``best`` is the one under test
    monkeypatch.setattr(GS, "evaluate_gait",
                        lambda g, p, **k: {**_credible_rollout(1.4 if p == _P else 0.9), "credible": True,
                                           "verdict": "CREDIBLE WALK"})

    def _margin(gene, params, *, steps, per_param=True, **kw):
        seen[gene.id] = {"steps": steps, "per_param": per_param, "params": dict(params)}
        return dict(_STURDY if gene.id == "r2sturdy" else _FRAGILE, steps=steps)

    monkeypatch.setattr(GF, "robustness_margin", _margin)
    with _db(tmp_path) as db:
        out = R.seed_corpus(db, prompts=("ref-sturdy", "ref-fragile"), seeds=1, deploy_steps=1200)
        gates = {r["prompt"]: gate_of(_bc(db, r["skill_id"])) for r in out}
        rels = {r["prompt"]: _bc(db, r["skill_id"])["robustness_rel"] for r in out}
    # the margin is measured on the point actually banked, at the horizon it was selected at
    assert seen["r2sturdy"] == {"steps": 1200, "per_param": False, "params": dict(_P)}
    assert gates == {"ref-sturdy": BANK_GATE, "ref-fragile": BANK_GATE_FRAGILE}
    assert rels == {"ref-sturdy": 0.01, "ref-fragile": None}
    assert out[0]["robustness_rel"] == 0.01 and out[1]["robustness_probes"] == _FRAGILE["probes"]


def test_reward_loop_measures_the_margin_and_reports_it_next_to_the_verdict(tmp_path, monkeypatch):
    """The loop spends n_rewards screening searches plus a final one before it banks anything, so the ladder is
    a few percent — and "credible" alone has shipped both a controller and one lucky float under the same word."""
    from virturoid.services import gait_flywheel as GF
    from virturoid.services import reward_loop as RL
    from virturoid.services import reward_reflection as RR
    g = _quad("rl")
    seen = {}

    def _round(gene, task, **kw):
        return {"reward_source": "template", "reward_name": "fwd", "reward_expr": "forward_vel*alive",
                "final": _result(_P), "ranked": [], "n_gamed": 0, "n_candidates": 1,
                "v": {"verdict": "CREDIBLE WALK", "credible": True, "forward": 1.2, "height_ratio": 0.85}}

    def _margin(gene, params, *, steps, per_param=True, **kw):
        seen.update(steps=steps, per_param=per_param, params=dict(params))
        return dict(_STURDY, steps=steps)

    monkeypatch.setattr(RL, "_one_round", _round)
    monkeypatch.setattr(RR, "build_reflection_payload", lambda *a, **k: {})
    monkeypatch.setattr(GF, "robustness_margin", _margin)
    with _db(tmp_path) as db:
        out = RL.run_intelligent_reward_loop(g, llm=None, steps=777, bank=True, db=db)
        sid = db.conn.execute("SELECT skill_id FROM skills WHERE task_type=?", (LOCOMOTION,)).fetchone()["skill_id"]
        bc = _bc(db, sid)
    assert out["banked"] is True and out["robustness_rel"] == 0.01
    assert "0.01 relative perturbation" in out["robustness_note"]      # the error bar rides WITH the verdict
    assert seen == {"steps": 777, "per_param": False, "params": dict(_P)}   # the verdict's own horizon
    assert gate_of(bc) == BANK_GATE and bc["bank_door"] == "reward_loop"
    assert bc["reward_expr"] == "forward_vel*alive"                    # R6 recall is unaffected
