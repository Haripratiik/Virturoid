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


def test_novelty_is_decided_before_the_expensive_verify(tmp_path):
    """#281: novelty is pure structure and the verifier is a 39 s gait fit, so a duplicate must never reach it.
    The measured night rejected 9 proposals for novelty AFTER paying to fit each one a controller."""
    calls = []

    def counting_verify(gene):
        calls.append(gene.id)
        return {"schema_valid": True, "compiles": True, "source": "physics", "credible": True,
                "fitness": 0.6, "verdict": "CREDIBLE", "kind": "legged"}
    it = iter([_legged(4, "A"), _legged(4, "DUP")])          # DUP is structurally identical to A
    res = CF.run_factory_night(lambda ctx: next(it, None), manifest_path=tmp_path / "c.json",
                               verify_fn=counting_verify)
    assert len(res.admitted) == 1 and res.rejected.get("novelty") == 1
    assert calls == ["A"], f"the duplicate was verified before it was deduped: {calls}"


def test_a_failed_draft_does_not_end_the_night(tmp_path):
    """A proposer hiccup (a composition that raised) used to return None through the wrapper, and
    ``run_factory_night`` reads None as 'the bank is exhausted' — so one transient failure forfeited the rest
    of the budget silently. A None mid-resample is a failed DRAFT, not an exhausted bank."""
    queue = [_aquatic(), None, _legged(4, "A"), _aquatic(), _legged(5, "B")]
    it = iter(queue)
    wrapped = CF.held_out_aware(lambda ctx: next(it, None), max_resample=4)
    res = CF.run_factory_night(wrapped, config=CF.FactoryConfig(max_bodies=2),
                               manifest_path=tmp_path / "c.json", verify_fn=_fake_verify({"A", "B"}))
    assert len(res.admitted) == 2 and res.rejected.get("held_out", 0) == 0
    assert res.proposer["failed_drafts"] == 1 and res.proposer["reserved_drafts"] == 2


def test_the_guard_steers_the_proposer_instead_of_only_filtering_it(tmp_path):
    """The whole point of #281: a reserved draft must come back to the proposer WITH ITS REASON, and the guard's
    constraints must reach the proposer before it draws at all."""
    seen_context, told = {}, []

    def proposer(ctx):
        seen_context.update(ctx)
        return next(it, None)
    it = iter([_aquatic(), _legged(4, "A")])
    wrapped = CF.held_out_aware(proposer, max_resample=3,
                                on_reserved=lambda g, p, why: told.append(why))
    CF.run_factory_night(wrapped, config=CF.FactoryConfig(max_bodies=1),
                         manifest_path=tmp_path / "c.json", verify_fn=_fake_verify({"A"}))
    assert told and told[0]["niche"] == "aquatic" and told[0]["reason"]
    assert seen_context["guard"]["max_limbs"] == 5           # the brief, handed over before any body exists
    assert "avoid_niches" in seen_context["guard"] and "corpus_keys" in seen_context


def test_guard_brief_is_the_guards_own_constraints():
    from virturoid.services.heldout_set import design_constraints
    assert CF.guard_brief() == design_constraints()


def test_proposer_stats_report_drafts_per_slot(tmp_path):
    """drafts/slot is the number that says whether steering works: 1.0 means every draft became a proposal."""
    it = iter([_legged(4, "A"), _legged(5, "B")])
    wrapped = CF.held_out_aware(lambda ctx: next(it, None))
    res = CF.run_factory_night(wrapped, config=CF.FactoryConfig(max_bodies=2),
                               manifest_path=tmp_path / "c.json", verify_fn=_fake_verify({"A", "B"}))
    assert res.proposer["drafts"] == 2 and res.proposer["drafts_per_slot"] == 1.0
    assert res.to_dict()["proposer"]["slots"] == 2


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


def _night_module():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import corpus_factory_night
    return corpus_factory_night


def test_the_legged_target_grid_spans_the_axes_that_actually_move_the_body():
    """The grid's cells are the MEASURED-effective axes (body-plan route, size word, biped stature); an inert
    axis in here is a slot spent to re-bank a body the corpus already has."""
    night = _night_module()
    targets = night._legged_targets()
    assert len(targets) == len({p for p, _ in targets})                # no duplicate prompt in the grid
    assert {t["route"] for _, t in targets} == {"template", "anatomy", "biped"}
    assert min(night._BIPED_STATURES_M) > 1.03                         # below this the biped builder clamps flat
    # interleaved: the first three targets are three DIFFERENT body plans, not three sizes of one
    assert len({t["route"] for _, t in targets[:3]}) == 3
    # every target must name a body the guard can admit — a reserved cell is a cell that can never be banked
    from virturoid.services.heldout_set import MANY_LIMB_MIN
    assert MANY_LIMB_MIN >= 6 and not any("six-legged" in p or "eight-legged" in p for p, _ in targets)


def test_night_proposer_spends_a_draft_not_a_slot_on_a_reserved_body(monkeypatch):
    """The load-bearing behaviour: a body the guard reserves is composed CHEAP, recognised, and its target
    retired — it never reaches the expensive walkability stage and never occupies a proposal slot."""
    night = _night_module()
    from virturoid.services import anatomy_compiler, morphology_composer
    drafted, finalized = [], []

    def fake_compose(prompt, *, llm="auto", ensure_walkable=False, strict_llm=False, **kw):
        drafted.append(prompt)
        assert ensure_walkable is False, "the draft stage must not pay for walkability"
        return _aquatic() if len(drafted) <= 2 else _legged(4, f"OK{len(drafted)}")

    def fake_walk(gene, prompt="", **kw):
        finalized.append(prompt)
        return gene
    monkeypatch.setattr(morphology_composer, "compose_robot", fake_compose)
    monkeypatch.setattr(anatomy_compiler, "ensure_walkable_quad", fake_walk)

    propose = night._offline_proposer(False, ("legged",), llm=None)
    got = propose({"thinnest_classes": ["legged"], "corpus_keys": set()})
    from virturoid.services.heldout_set import is_held_out
    gene, prompt = got
    assert not is_held_out(gene, prompt=prompt)
    assert len(drafted) == 3 and finalized == [drafted[2]]     # 2 reserved drafts, 1 finalize — not 3 finalizes
    assert propose.stats == {"drafts_composed": 3, "reserved_drafts": 2, "duplicate_drafts": 0,
                             "failed_drafts": 0, "finalized": 1}
    # the two reserved targets are RETIRED, not re-rolled: the next call draws fresh prompts
    propose({"thinnest_classes": ["legged"], "corpus_keys": set()})
    assert drafted[3] not in drafted[:3]


def test_night_proposer_dedups_structurally_before_paying_to_verify(monkeypatch):
    """A body the corpus already holds is refused at draft time, where it costs ~0 s, instead of downstream at
    the novelty gate where it would already have cost a gait fit."""
    night = _night_module()
    from virturoid.services import anatomy_compiler, morphology_composer
    from virturoid.services.heldout_set import body_key
    n = {"i": 0}

    def fake_compose(prompt, **kw):
        n["i"] += 1
        return _legged(4, "SAME") if n["i"] <= 2 else _legged(5, "NEW")
    monkeypatch.setattr(morphology_composer, "compose_robot", fake_compose)
    monkeypatch.setattr(anatomy_compiler, "ensure_walkable_quad", lambda g, p="", **kw: g)
    propose = night._offline_proposer(False, ("legged",), llm=None)
    got = propose({"corpus_keys": {body_key(_legged(4, "SAME"))}})
    assert body_key(got[0]) == body_key(_legged(5, "NEW"))
    assert propose.stats["duplicate_drafts"] == 2 and propose.stats["finalized"] == 1


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
