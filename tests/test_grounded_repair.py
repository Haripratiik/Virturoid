"""WS-B.1 — grounded repair: error-typed feedback, repair/resample routing, 2-round cap (§8.2.1)."""
from __future__ import annotations

from virturoid.services import grounded_repair as GR


# ---------------------------------------------------------------- routing (the load-bearing §8.2.1 finding)
def test_routing_repairs_structural_and_resamples_semantic():
    assert GR.action_for("schema") == "repair"
    assert GR.action_for("compile") == "repair"
    assert GR.action_for("spawn") == "repair"
    assert GR.action_for("verdict") == "resample"          # semantic failures repair poorly -> resample
    assert GR.action_for(None) == "done"


# ---------------------------------------------------------------- typed diagnostics
def test_schema_diagnostics_are_typed():
    d = GR.diagnostics(None, {"error_class": "schema", "schema_issue": "unknown parent 'ghost'"})
    assert d["error_class"] == "schema" and "ghost" in d["issue"] and d["fix"]


def test_verdict_diagnostics_carry_physics_and_a_fix_direction():
    row = {"error_class": "verdict", "credible": False, "kind": "legged",
           "verdict": "LURCHES (pitch 34 / roll 17 deg)", "fitness_raw": 1.4,
           "verdict_detail": {"roll_max_deg": 17, "pitch_max_deg": 34, "support_frac": 0.4}}
    d = GR.diagnostics(None, row)
    assert d["error_class"] == "verdict"
    assert d["pitch_max_deg"] == 34 and d["roll_max_deg"] == 17     # physics surfaced
    assert "widen" in d["fix"].lower() and "com" in d["fix"].lower()   # a physics-grounded fix direction


def test_slide_and_drive_hints_key_off_the_failure_mode():
    slide = GR.diagnostics(None, {"error_class": "verdict", "credible": False,
                                  "verdict": "SLIDE (feet barely lift)", "verdict_detail": {"support_frac": 0.1}})
    assert "slide" in slide["fix"].lower() and "revolute" in slide["fix"]
    tip = GR.diagnostics(None, {"error_class": "verdict", "credible": False, "verdict": "TIPPED", "verdict_detail": {}})
    assert "wheel" in tip["fix"].lower()


# ---------------------------------------------------------------- the loop (injected propose; physics judges)
def _stub_evaluate(sequence):
    """An evaluate() that returns pre-scripted rows by call order (no physics)."""
    calls = {"i": 0}

    def ev(gene):
        r = sequence[min(calls["i"], len(sequence) - 1)]
        calls["i"] += 1
        return r
    return ev


def test_loop_stops_immediately_when_first_design_is_credible():
    ev = _stub_evaluate([{"credible": True, "error_class": None}])
    out = GR.repair_loop("x", lambda p, feedback, resample: "G", evaluate=ev)
    assert out["credible"] and out["repair_iters"] == 0


def test_loop_repairs_a_compile_failure_then_succeeds():
    seen = []

    def propose(p, feedback, resample):
        seen.append((None if feedback is None else feedback["action"], resample))
        return "G"
    ev = _stub_evaluate([{"credible": False, "error_class": "compile", "compile_error": "bad"},
                         {"credible": True, "error_class": None}])
    out = GR.repair_loop("x", propose, evaluate=ev)
    assert out["credible"] and out["repair_iters"] == 1
    assert seen[1] == ("repair", False)                    # a compile failure was REPAIRED (not resampled)


def test_loop_resamples_a_verdict_failure_and_caps_at_two_rounds():
    actions = []

    def propose(p, feedback, resample):
        if feedback is not None:
            actions.append((feedback["action"], resample))
        return "G"
    ev = _stub_evaluate([{"credible": False, "error_class": "verdict", "verdict": "SLIDE"}])  # never improves
    out = GR.repair_loop("x", propose, evaluate=ev)
    assert not out["credible"]
    assert out["repair_iters"] == GR.MAX_ROUNDS            # capped
    assert actions == [("resample", True), ("resample", True)]   # verdict failures were RESAMPLED, not patched


def test_loop_integrates_with_real_physics_evaluate():
    """End-to-end with the real funnel: a designer that ignores feedback and keeps proposing the same walking
    quadruped either passes round 0 or caps — the loop must terminate and report an honest repair_iters."""
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    out = GR.repair_loop("a simple four-legged walking robot",
                         lambda p, feedback, resample: g, verify=True)
    assert out["repair_iters"] <= GR.MAX_ROUNDS
    assert isinstance(out["credible"], bool) and len(out["history"]) == out["repair_iters"] + 1
