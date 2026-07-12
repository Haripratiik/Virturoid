"""WS-C.2 — the corpus factory night: ordered gate stack, checkpointing, dashboards (master_plan_v6 §10.2)."""
from __future__ import annotations

import json

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import corpus_factory as CF


def _legged(n_legs, tag, *, total_len=2.0):
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.5, radius_m=0.1)]
    for i in range(n_legs):
        prev = "torso"
        for j in range(3):
            nm = f"leg{i}_{j}"
            segs.append(GeneSegment(name=nm, parent=prev, joint_type="revolute",
                                    length_m=total_len / (n_legs * 3), radius_m=0.02))
            prev = nm
    segs[-1].is_end_effector = True                       # gene.validate() needs exactly one end effector
    return RobotGene(id=tag, species=f"t.{tag}", robot_class="quadruped", segments=segs,
                     base_mount="free", end_effector_type="none")


def _aquatic():
    return RobotGene(id="aq", species="t.aq", robot_class="aquatic", base_mount="free",
                     segments=[GeneSegment(name="s0", parent=None, joint_type=None),
                               GeneSegment(name="s1", parent="s0", joint_type="revolute", is_end_effector=True)])


def _fake_verify(credible_ids):
    """A physics-tagged verify that marks only ``credible_ids`` credible (fast, no MuJoCo)."""
    def verify(gene):
        return {"schema_valid": True, "compiles": True, "source": "physics",
                "credible": gene.id in credible_ids, "fitness": 0.6 if gene.id in credible_ids else 0.0,
                "verdict": "CREDIBLE" if gene.id in credible_ids else "SLIDE", "kind": "legged"}
    return verify


def test_gate_stack_admits_novel_credible_and_rejects_the_rest(tmp_path):
    # NB: >=6 limb chains is a reserved held-out niche, so novel bodies stay at 4-5 legs
    novel_a, dup_a = _legged(4, "A"), _legged(4, "A")                              # dup_a is identical to A
    novel_b = _legged(5, "B")                                                      # 5 legs: novel niche, not held
    bad = _legged(4, "BAD", total_len=4.0)                                         # distinct+novel but non-credible
    held = _aquatic()                                                              # aquatic niche -> held_out gate
    queue = [novel_a, dup_a, novel_b, bad, held]
    it = iter(queue)
    res = CF.run_factory_night(lambda ctx: next(it, None),
                               config=CF.FactoryConfig(max_bodies=10),
                               manifest_path=tmp_path / "corpus.json",
                               verify_fn=_fake_verify({"A", "B"}))
    admitted_keys = {a["body_key"] for a in res.admitted}
    from virturoid.services.heldout_set import body_key
    assert body_key(novel_a) in admitted_keys and body_key(novel_b) in admitted_keys
    assert res.rejected.get("held_out", 0) >= 1            # aquatic body held out
    assert res.rejected.get("verdict", 0) >= 1            # non-credible rejected
    assert res.rejected.get("novelty", 0) >= 1            # the duplicate rejected
    assert len(res.admitted) == 2                         # both credible+novel bodies admitted
    assert res.annecs == len(res.admitted)               # ANNECS = novel-AND-solved this night
    assert res.coverage["total_bodies"] == 2             # exactly the two admitted bodies in the corpus


def test_held_out_aware_proposer_saves_the_budget(tmp_path):
    """v7-C1: the held-out-aware wrapper resamples past reserved-niche bodies (the measured live-LLM waste,
    25/30) so the proposal budget reaches admittable bodies — without ever weakening the gate."""
    # a raw proposer that mostly yields held-out aquatic bodies, with one clean legged body buried in it
    queue = [_aquatic(), _aquatic(), _legged(4, "GOOD"), _aquatic()]
    it = iter(queue)
    raw = lambda ctx: next(it, None)
    wrapped = CF.held_out_aware(raw, max_resample=4)
    got = wrapped({})                                          # must skip the two aquatic bodies -> the legged one
    gene = got[0] if isinstance(got, tuple) else got
    from virturoid.services.heldout_set import is_held_out, body_key
    assert not is_held_out(gene) and body_key(gene) == body_key(_legged(4, "GOOD"))
    # bank-exhausted -> None (never loops forever); an all-held-out bank yields the last (gate still rejects it)
    assert CF.held_out_aware(lambda ctx: None)({}) is None
    allheld = iter([_aquatic(), _aquatic()])
    out = CF.held_out_aware(lambda ctx: next(allheld, None), max_resample=1)({})
    assert out is None or is_held_out(out[0] if isinstance(out, tuple) else out)   # gate stays authoritative

    # end-to-end: the wrapper lifts admits when the raw proposer wastes slots on held-out bodies
    q2 = [_aquatic(), _legged(4, "A"), _aquatic(), _legged(5, "B"), _aquatic()]
    it2 = iter(q2)
    res = CF.run_factory_night(CF.held_out_aware(lambda ctx: next(it2, None)),
                               config=CF.FactoryConfig(max_bodies=2),   # only 2 slots — both must land real bodies
                               manifest_path=tmp_path / "c.json", verify_fn=_fake_verify({"A", "B"}))
    assert len(res.admitted) == 2 and res.rejected.get("held_out", 0) == 0   # no slot wasted on a held-out body


def test_checkpoint_persists_and_resumes(tmp_path):
    mp = tmp_path / "corpus.json"
    it1 = iter([_legged(4, "A")])
    r1 = CF.run_factory_night(lambda ctx: next(it1, None), manifest_path=mp, verify_fn=_fake_verify({"A"}))
    assert len(r1.admitted) == 1 and r1.annecs == 1
    saved = json.loads(mp.read_text())
    assert saved["nights"] == 1 and len(saved["admitted"]) == 1
    # a SECOND night resumes from the manifest: the prior body seeds the grid, ANNECS accumulates
    it2 = iter([_legged(5, "B")])                          # 5 legs: novel niche, not a held-out many-limb body
    r2 = CF.run_factory_night(lambda ctx: next(it2, None), manifest_path=mp, verify_fn=_fake_verify({"B"}))
    assert r2.annecs == 2 and json.loads(mp.read_text())["nights"] == 2


def test_real_data_anchor_rejects_a_predicted_label(tmp_path):
    def predicted_verify(gene):
        return {"schema_valid": True, "compiles": True, "source": "metric_predicted",   # NOT a physics label
                "credible": True, "fitness": 0.9, "verdict": "PREDICTED"}
    it = iter([_legged(4, "P")])
    res = CF.run_factory_night(lambda ctx: next(it, None), manifest_path=tmp_path / "c.json",
                               verify_fn=predicted_verify)
    assert not res.admitted and res.rejected.get("real_data_anchor", 0) == 1


def test_proposer_failure_is_isolated_not_fatal(tmp_path):
    calls = {"n": 0}

    def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("proposer hiccup")
        if calls["n"] == 2:
            return _legged(4, "OK")
        return None
    res = CF.run_factory_night(flaky, manifest_path=tmp_path / "c.json", verify_fn=_fake_verify({"OK"}))
    assert len(res.admitted) == 1                          # the night survived the proposer error
    assert any(d.get("stage") == "propose" for d in res.diagnostics)


@pytest.mark.slow
def test_gait_search_verify_is_physics_tagged_and_can_solve(tmp_path):
    """The §10.2 VERIFY-BUILD: a gait search returns a physics-sourced verdict; a walkable body can be solved."""
    from virturoid.services.corpus_factory import gait_search_verify
    from virturoid.services.morphology_composer import compose_robot
    walker = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    res = gait_search_verify(walker, max_evals=4, steps=400)
    assert res["source"] == "physics"                     # every eval is a real rollout, never a prediction
    assert isinstance(res["credible"], bool) and "schema_valid" in res


@pytest.mark.slow
def test_night_with_real_physics_and_banking(tmp_path):
    """End-to-end: real physics verify + default banking makes admitted bodies retrievable corpus."""
    from virturoid.services.morphology_composer import compose_robot
    walker = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    it = iter([walker])
    res = CF.run_factory_night(lambda ctx: next(it, None), manifest_path=tmp_path / "corpus.json",
                               memory_dir=tmp_path, bank_fn=CF.default_bank_fn,
                               config=CF.FactoryConfig(max_bodies=1))
    # whatever the verdict, the night completes, checkpoints, and reports honest dashboards
    assert isinstance(res.coverage, dict) and res.wall_s >= 0
    assert (tmp_path / "corpus.json").exists()
    if res.admitted:                                       # if the walker was credible, it's now banked corpus
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import BODY, RoboticsVectorMemory
        with MemoryDB(tmp_path / "virturoid_memory.db") as db:
            assert RoboticsVectorMemory(db).count(BODY) >= 1
