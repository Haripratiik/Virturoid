"""A banked gait has to say WHICH GATE let it in.

The fragility gate (bank only an operating point whose perturbed copies also walk) landed 2026-08-04 and lives in
ONE of the four callers of ``bank_gait``. So the live bank holds rows admitted under it and rows admitted with the
fragility question never asked, and until now nothing on the row distinguished them. Pooling those two kinds is
not a hypothetical error — it is exactly how ``duty``, a coordinate ``crawl_gait_rollout`` never read, came to be
reported to operators as the bank's tightest-clustered parameter (#265/#266).

These tests pin the provenance: the stamp is written when a margin was measured, absent when it was not, never
guessed; the miner always reports the split and can be restricted to the measured rows; and the corpus factory's
gait write refuses a row whose margin is missing or fragile.
"""
from __future__ import annotations

import tempfile
import types
from pathlib import Path

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import corpus_factory as CF
from virturoid.services.gait_flywheel import BANK_GATE, LOCOMOTION, bank_gait, gate_of


def _db():
    from virturoid.services.memory_db import MemoryDB
    return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="bankgate_")) / "m.db")


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


def _base_config(db, skill_id):
    import json
    row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
    return json.loads(row["base_config"])


_P = {"freq": 1.8, "hip_amp": 0.8, "knee_amp": 1.1, "kp": 60.0, "kd": 4.0}


def test_a_measured_margin_stamps_the_row_and_an_unmeasured_one_does_not():
    db = _db()
    gated = bank_gait(db, _quad("gated"), _result(_P),
                      robustness={"robustness_rel": 0.01, "probes": {"0.1": "3/4", "0.01": "4/4"}, "steps": 6000})
    plain = bank_gait(db, _quad("plain", leg_len=0.24), _result({**_P, "freq": 2.4}))
    bc_gated, bc_plain = _base_config(db, gated), _base_config(db, plain)
    assert bc_gated["bank_gate"] == BANK_GATE and bc_gated["robustness_rel"] == 0.01
    assert bc_gated["robustness_probes"] == {"0.1": "3/4", "0.01": "4/4"}
    assert gate_of(bc_gated) == BANK_GATE
    # ...and the row nobody measured says so rather than defaulting to "fine": a missing error bar is the whole
    # failure mode, so it must never read as a passing one.
    assert bc_plain["bank_gate"] is None and bc_plain["robustness_rel"] is None
    assert gate_of(bc_plain) == "ungated"
    assert gate_of({}) == "ungated" and gate_of(None) == "ungated"
    db.close()


def test_the_miner_always_reports_the_split_and_can_be_restricted_to_measured_rows():
    from virturoid.services.gait_hints import mine_gait_hints
    db = _db()
    for i in range(3):
        bank_gait(db, _quad(f"g{i}", leg_len=0.18 + 0.02 * i), _result({**_P, "freq": 1.6 + 0.1 * i}),
                  robustness={"robustness_rel": 0.1, "probes": {"0.1": "4/4"}, "steps": 6000})
    for i in range(5):
        bank_gait(db, _quad(f"u{i}", leg_len=0.30 + 0.02 * i), _result({**_P, "freq": 2.5 + 0.1 * i}))
    pooled = mine_gait_hints(db, robot_class="quadruped")
    assert (pooled["n_gated"], pooled["n_ungated"]) == (3, 5)
    assert pooled["n"] == 8 and "fragility gate" in pooled["note"] and "both pooled" in pooled["note"]
    only = mine_gait_hints(db, robot_class="quadruped", gated_only=True)
    assert only["n"] == 3 and (only["n_gated"], only["n_ungated"]) == (3, 5)
    assert "only the gated rows were used" in only["note"]
    # The two corpora are genuinely different data, not the same numbers relabelled: the ungated rows sit an
    # octave away in step frequency, so restricting the source has to move the mined prior.
    assert only["prior"]["freq"] < pooled["prior"]["freq"] - 0.3
    db.close()


def test_the_factory_verifier_fits_against_the_NIGHTS_memory_not_the_products(tmp_path, monkeypatch):
    """A "clean rebuild" that warm-starts every fit from the corpus it is replacing is not a clean rebuild.

    ``fit_gait_for_body(db=None)`` falls back to opening ``build/memory`` — the live product bank — so the
    verifier has to hand it the night's own DB explicitly. This pins that it does, and that the verify result
    carries the operating point AND its margin forward to the bank step rather than making it re-search.
    """
    from virturoid.services import gait_flywheel as GF

    seen = {}

    def _fake_fit(gene, *, db=None, bank=True, **kw):
        seen["db_path"] = str(getattr(db, "path", db))
        seen["bank"] = bank
        return {"ok": True, "searched": True, "adopted": True, "params": dict(_P), "confirm_forward_m": 2.5,
                "robustness": {"robustness_rel": 0.01, "probes": {"0.01": "4/4"}, "steps": 6000}}

    monkeypatch.setattr(GF, "fit_gait_for_body", _fake_fit)
    monkeypatch.setattr(CF, "_compiles_and_credible",
                        lambda g: {"schema_valid": True, "compiles": True, "credible": False, "fitness": 0.0,
                                   "verdict": "FELL", "kind": "legged", "source": "physics"})
    res = CF.gait_fit_verify_fn(tmp_path)(_quad("night"))
    assert str(tmp_path) in seen["db_path"], seen        # the night's DB, never build/memory
    assert seen["bank"] is False                          # banking happens after the gate stack, not in verify
    assert res["credible"] and res["gait_source"] == "fitted"
    assert res["gait_params"] == _P and res["robustness"]["robustness_rel"] == 0.01
    # ...and that result is exactly what the bank step consumes, so nothing is re-searched or re-measured.
    CF.gait_bank_fn(_quad("night"), res, tmp_path)
    from virturoid.services.memory_db import MemoryDB
    with MemoryDB(tmp_path / "virturoid_memory.db") as db:
        row = db.conn.execute("SELECT skill_id FROM skills WHERE task_type=?", (LOCOMOTION,)).fetchone()
        assert gate_of(_base_config(db, row["skill_id"])) == BANK_GATE


def test_the_factory_gait_write_refuses_an_unmeasured_or_fragile_operating_point(tmp_path):
    """``gait_bank_fn`` is the write ``default_bank_fn`` never made — and it must not become a door around the
    gate. A verify result with no margin, or one that failed the finest rung, banks nothing."""
    from virturoid.services.memory_db import MemoryDB

    def n_skills():
        with MemoryDB(tmp_path / "virturoid_memory.db") as db:
            return db.conn.execute("SELECT COUNT(*) FROM skills WHERE task_type=?", (LOCOMOTION,)).fetchone()[0]

    ok = {"gait_params": dict(_P), "fitness": 1.4,
          "robustness": {"robustness_rel": 0.01, "probes": {"0.01": "4/4"}, "steps": 6000}}
    CF.gait_bank_fn(_quad("fragile"), {**ok, "robustness": {"robustness_rel": None, "probes": {"0.001": "0/4"}}},
                    tmp_path)
    CF.gait_bank_fn(_quad("unmeasured"), {"gait_params": dict(_P), "fitness": 1.4}, tmp_path)
    CF.gait_bank_fn(_quad("noparams"), {**ok, "gait_params": None}, tmp_path)
    assert n_skills() == 0
    CF.gait_bank_fn(_quad("sturdy"), ok, tmp_path)
    assert n_skills() == 1
    with MemoryDB(tmp_path / "virturoid_memory.db") as db:
        sid = db.conn.execute("SELECT skill_id FROM skills WHERE task_type=?", (LOCOMOTION,)).fetchone()["skill_id"]
        assert gate_of(_base_config(db, sid)) == BANK_GATE
        # the body sub-space write survives too — novelty / nearest-bodies still see the admitted body
        assert db.conn.execute("SELECT COUNT(*) FROM vectors WHERE obj_type='body'").fetchone()[0] == 1
