"""WS-B.4 — the coercion audit surfaces every compiler mutation (master_plan_v6 §8.2.4)."""
from __future__ import annotations

from virturoid.services import coercion_audit as CA


def _graph(**over):
    parts = [{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
             {"name": "leg1", "role": "leg", "parent": "torso", "attach": "front_bottom",
              "aim": "down_out", "size": 0.4, "girth": 0.02, "segments": 4, "joint": "revolute"}]
    g = {"robot_class": "quadruped", "name": "t", "parts": parts}
    g.update(over)
    return g


def test_detect_reports_girth_default_and_joint_aim_defaults():
    # a leg that omits girth, joint AND aim -> three convenience defaults surfaced
    g = _graph(parts=[{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
                      {"name": "leg1", "role": "leg", "parent": "torso", "size": 0.4, "segments": 4}])
    coer = CA.detect_coercions(g)
    fields = {(c["part"], c["field"], c["kind"]) for c in coer}
    assert ("leg1", "girth", "default") in fields
    assert ("leg1", "joint", "default") in fields
    assert ("leg1", "aim", "default") in fields
    d = next(c for c in coer if c["part"] == "leg1" and c["field"] == "girth")
    assert abs(d["applied"] - 0.18 * 0.4) < 1e-6           # ~0.18×size


def test_detect_reports_size_and_girth_clamps_as_physics_required():
    g = _graph(parts=[{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
                      {"name": "nub", "role": "leg", "parent": "torso", "size": 0.01, "girth": 0.001,
                       "joint": "revolute", "aim": "down_out"}])
    coer = CA.detect_coercions(g)
    size_c = next(c for c in coer if c["part"] == "nub" and c["field"] == "size")
    girth_c = next(c for c in coer if c["part"] == "nub" and c["field"] == "girth")
    assert size_c["kind"] == "clamp" and size_c["physics_required"] and size_c["applied"] == CA.SIZE_FLOOR_M
    assert girth_c["kind"] == "clamp" and girth_c["physics_required"]


def test_body_width_cap_is_an_override_and_aspect_opts_out():
    fat = _graph(parts=[{"name": "torso", "role": "body", "size": 0.5, "girth": 0.30},   # 0.30 > 0.32×0.5=0.16
                        {"name": "leg1", "role": "leg", "parent": "torso", "size": 0.4, "girth": 0.02,
                         "joint": "revolute", "aim": "down_out"}])
    caps = [c for c in CA.detect_coercions(fat) if c["field"] == "girth" and c["kind"] == "cap"]
    assert len(caps) == 1 and caps[0]["physics_required"]
    # opting out with an explicit aspect removes the override
    fat["parts"][0]["aspect"] = "wide"
    assert not [c for c in CA.detect_coercions(fat) if c["kind"] == "cap"]


def test_clean_design_has_no_coercions():
    assert CA.detect_coercions(_graph()) == []             # every field explicit + in-band -> nothing coerced


def test_summarize_splits_the_two_plan_classes():
    g = _graph(parts=[{"name": "torso", "role": "body", "size": 0.5, "girth": 0.30},
                      {"name": "leg1", "role": "leg", "parent": "torso", "size": 0.4}])
    s = CA.summarize(CA.detect_coercions(g))
    assert s["n"] >= 2
    assert len(s["overrides"]) == 1                        # the torso width cap
    assert all(c["physics_required"] for c in s["physics_required"])
    assert all(not c["physics_required"] for c in s["conveniences"])


def test_audit_table_is_complete_and_classified():
    tbl = CA.audit_table()
    assert len(tbl) >= 6
    for row in tbl:
        assert row["kind"] in ("clamp", "default", "cap")
        assert isinstance(row["physics_required"], bool) and isinstance(row["surfaced"], bool)
        assert row["rationale"]
    # every runtime-detectable kind is represented in the static table
    assert {"clamp", "default", "cap"} <= {row["kind"] for row in tbl}


def test_rules_track_the_compiler_behaviourally():
    """Drift guard: build a graph whose values SHOULD be coerced and assert the realized gene matches the audit's
    prediction (so the rules can't silently diverge from anatomy_compiler)."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    g = {"robot_class": "quadruped", "name": "t",
         "parts": [{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
                   {"name": "leg1", "role": "leg", "parent": "torso", "size": 0.01,   # below SIZE_FLOOR
                    "girth": 0.02, "segments": 1, "joint": "revolute", "aim": "down_out",
                    "symmetry": "left_right"}]}
    assert any(c["field"] == "size" and c["kind"] == "clamp" for c in CA.detect_coercions(g))
    gene = build_from_anatomy(g)
    leg_segs = [s for s in gene.segments if s.name.startswith("leg1")]
    assert leg_segs, "leg did not compile"
    # the realized leg segment length is clamped to the floor the audit reported (not the sub-floor 0.01)
    assert max(s.length_m for s in leg_segs) >= CA.SIZE_FLOOR_M - 1e-9


def test_submit_design_surfaces_coercions():
    from virturoid.services.agent_design_tools import submit_design
    g = _graph(parts=[{"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
                      {"name": "leg1", "role": "leg", "parent": "torso", "aim": "down_out",
                       "size": 0.4, "segments": 4, "joint": "revolute", "symmetry": "left_right"},
                      {"name": "leg2", "role": "leg", "parent": "torso", "attach": "rear_bottom",
                       "aim": "down_out", "size": 0.4, "segments": 4, "joint": "revolute",
                       "symmetry": "left_right"}])
    out = submit_design({"graph": g})
    assert out["ok"], out.get("error")
    assert "coercions" in out                               # legs omit girth -> surfaced, not silent
    assert any(c["field"] == "girth" and c["kind"] == "default" for c in out["coercions"])
