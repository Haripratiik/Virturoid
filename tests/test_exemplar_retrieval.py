"""WS-B.2 — query-specific verified-exemplar retrieval, scores OMITTED (master_plan_v6 §8.2.2)."""
from __future__ import annotations

import json

import pytest

from virturoid.services import exemplar_retrieval as ER


def _seed_corpus(memdir):
    """Seed a small VERIFIED body corpus (buildable=True) + one unbuilt body (buildable=False)."""
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.robotics_vector_memory import BODY, RoboticsVectorMemory, embed_body
    with MemoryDB(memdir / "virturoid_memory.db") as db:
        vm = RoboticsVectorMemory(db)
        vm.upsert(BODY, "verified_quad", embed_body(compose_robot("a large quadruped robot")),
                  {"robot_class": "quadruped", "buildable": True})
        vm.upsert(BODY, "verified_arm", embed_body(compose_robot("a 6-dof manipulator arm")),
                  {"robot_class": "manipulator", "buildable": True})
        vm.upsert(BODY, "UNBUILT_quad", embed_body(compose_robot("a small quadruped robot dog")),
                  {"robot_class": "quadruped", "buildable": False})


def test_structural_summary_is_score_free():
    from virturoid.services.morphology_composer import compose_robot
    s = ER.structural_summary(compose_robot("a four-legged walking robot"))
    assert set(s).isdisjoint(ER._FORBIDDEN_SCORE_FIELDS)
    assert s["n_segments"] > 0 and "shape_counts" in s and "dof" in s


def test_verified_exemplars_are_query_specific_and_score_free(tmp_path, monkeypatch):
    _seed_corpus(tmp_path)
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)
    from virturoid.services.morphology_composer import compose_robot
    ex = ER.verified_exemplars(compose_robot("a small quadruped robot dog"), k=3)
    assert ex, "a verified body should ground a quadruped query"
    ids = [e["exemplar"] for e in ex]
    assert "UNBUILT_quad" not in ids                        # verified-only: the unbuilt body is never returned
    # query-specific: the quadruped exemplar outranks the manipulator for a quadruped query
    assert ids[0] == "verified_quad"
    # NO score field anywhere in the payload (the RoboMorph mode-collapse guard)
    blob = json.dumps(ex).lower()
    for forbidden in ("fitness", "success", "reward", "forward_m", "\"verdict\"", "credible"):
        assert forbidden not in blob, f"exemplar payload leaked a score field: {forbidden}"


def test_exemplar_grounding_omitted_without_a_draft():
    assert ER.exemplar_grounding({}) == {}                  # no draft graph -> no query-specific retrieval
    assert ER.exemplar_grounding({"robot_class": "quadruped"}) == {}


def test_exemplar_grounding_from_a_draft_graph(tmp_path, monkeypatch):
    _seed_corpus(tmp_path)
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)
    draft = {"robot_class": "quadruped", "name": "draft",
             "parts": [{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
                       {"name": "leg1", "role": "leg", "parent": "torso", "attach": "front_bottom",
                        "aim": "down_out", "size": 0.4, "girth": 0.02, "segments": 4, "joint": "revolute",
                        "symmetry": "left_right"}]}
    g = ER.exemplar_grounding({"graph": draft})
    assert "verified_exemplars" in g and g["verified_exemplars"]
    assert "omitted" in g["note"].lower()                   # the note states scores are omitted on purpose


def test_get_design_schema_surfaces_verified_exemplars(tmp_path, monkeypatch):
    _seed_corpus(tmp_path)
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)
    from virturoid.services.agent_design_tools import get_design_schema
    draft = {"robot_class": "quadruped", "parts": [{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
             {"name": "leg1", "role": "leg", "parent": "torso", "aim": "down_out", "size": 0.4, "girth": 0.02,
              "segments": 4, "joint": "revolute", "symmetry": "left_right"}]}
    schema = get_design_schema({"graph": draft, "robot_class": "quadruped"})
    assert schema["ok"]
    assert "verified_exemplars" in schema                   # query-specific grounding rode into the schema
