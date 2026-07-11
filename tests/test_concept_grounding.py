"""WS-G + WS-H — concept×embedding grounding (exact-alias preserved) + visibility (master_plan_v6 §WS-G/H)."""
from __future__ import annotations

import json


def _seed(memdir):
    """A verified concept 'wibbot' whose banked body is a quadruped in the species tree."""
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morphology_composer import compose_robot
    quad = compose_robot("a sturdy four-legged walking robot", ensure_walkable=True)
    with MemoryDB(memdir / "virturoid_memory.db") as db:
        db.upsert_species_node("test.quad", robot_class="quadruped", genes=quad.to_dict(), buildable=True)
        db.observe_concept("wibbot", "a wibbot robot", aliases=["wibbot", "wib bot"])
        db.promote_concept("wibbot", execution_family="quadruped", task_type="locomotion",
                           species_pattern="test.quad", success_rate=0.9, target_success_rate=0.5)


# ---------------------------------------------------------------- exact-alias routing preserved
def test_exact_alias_routes_deterministically(tmp_path):
    _seed(tmp_path)
    from virturoid.services.concept_grounding import ground_concept
    r = ground_concept(tmp_path, "wibbot")
    assert r["routed"] and r["route"]["execution_family"] == "quadruped"
    r2 = ground_concept(tmp_path, "wib bot")                 # an LLM-proposed EXACT alias also routes
    assert r2["routed"]


def test_novel_concept_gets_advisory_grounding_never_a_route(tmp_path):
    _seed(tmp_path)
    from virturoid.services.concept_grounding import ground_concept
    from virturoid.services.morphology_composer import compose_robot
    query = compose_robot("a small four-legged robot dog")   # a NOVEL word, but a quadruped-shaped first attempt
    r = ground_concept(tmp_path, "a brand new zorblax", query_gene=query)
    assert r["routed"] is False                              # a new word NEVER inherits a route
    assert r["grounding"] and r["grounding"][0]["concept"] == "wibbot"   # grounded by the nearest verified concept
    assert "not a route" in r["note"].lower() or "advisory" in r["note"].lower()
    # the grounding carries the physics outcome, not a silent substitution
    assert "quadruped" in r["grounding"][0]["outcome"]


def test_novel_concept_without_corpus_is_honest(tmp_path):
    from virturoid.services.concept_grounding import ground_concept
    r = ground_concept(tmp_path, "a first ever robot")       # empty corpus
    assert r["routed"] is False and r["grounding"] == []
    assert "genuinely new" in r["note"]


def test_concept_neighbors_rank_by_similarity(tmp_path):
    _seed(tmp_path)
    from virturoid.services.concept_grounding import concept_neighbors
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morphology_composer import compose_robot
    with MemoryDB(tmp_path / "virturoid_memory.db") as db:
        nb = concept_neighbors(db, compose_robot("a quadruped robot"), k=3)
    assert nb and 0.0 <= nb[0]["similarity"] <= 1.0 and nb[0]["concept"] == "wibbot"


# ---------------------------------------------------------------- WS-H visibility
def test_brain_metrics_includes_concepts(tmp_path):
    _seed(tmp_path)
    from virturoid.services.flywheel_status import brain_metrics
    from virturoid.services.memory_db import MemoryDB
    with MemoryDB(tmp_path / "virturoid_memory.db") as db:
        bm = brain_metrics(db)
    assert bm["concepts"]["verified"] >= 1 and bm["concepts"]["total"] >= 1


def test_corpus_factory_status_traces_to_the_manifest(tmp_path):
    from virturoid.services.flywheel_status import corpus_factory_status
    assert corpus_factory_status(tmp_path)["nights"] == 0     # no manifest -> honest empty
    manifest = {"nights": 2, "cumulative_annecs": 5,
                "admitted": [{"niche": ["legged", 3, 2, 2]}, {"niche": ["mobile", 3, 0, 1]}],
                "rejected_totals": {"verdict": 4}}
    (tmp_path / "corpus_factory.json").write_text(json.dumps(manifest), encoding="utf-8")
    st = corpus_factory_status(tmp_path)
    assert st["nights"] == 2 and st["cumulative_annecs"] == 5 and st["corpus_size"] == 2
    assert st["class_balance"] == {"legged": 1, "mobile": 1}


def test_moat_status_surfaces_the_factory(tmp_path):
    from virturoid.services.flywheel_status import moat_status
    (tmp_path / "corpus_factory.json").write_text(
        json.dumps({"nights": 1, "cumulative_annecs": 3, "admitted": []}), encoding="utf-8")
    st = moat_status(tmp_path)
    assert "corpus_factory" in st and st["corpus_factory"]["cumulative_annecs"] == 3
    assert "Factory: 3 verified bodies" in st["summary"]
