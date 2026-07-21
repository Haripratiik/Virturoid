"""A zero-token evaluator must still get a ROBOT, and must be told how it was designed (MVP audit B2).

Measured before: `POST /api/jobs {kind: autonomous_build}` with no LLM backend returned
"LLM planner is not configured or reachable; no robot was generated" — feasible=False, zero package files. A
fresh clone has only .env.example, so the flagship build path was dead on arrival for anyone evaluating the repo.

The offline heuristic composer (the SAME one create_robot uses) was already wired and builds a real,
physics-verified body — it was simply never reached, because `plan_build(require_llm=True)` returns
routing_confidence="llm_unavailable" and that fell into the clarify branch. Now that case re-plans offline.

Honesty is the other half: an offline body must never be passed off as LLM-authored, and the note has to reach
reports/autonomy_report.json (the gene route writes its own report, so an in-memory-only note would vanish from
the artifact a reviewer opens).
"""
from __future__ import annotations



import pytest

from virturoid.services import autonomous_build as AB


def test_llm_unavailable_replans_offline_instead_of_giving_up(monkeypatch):
    """The routing decision itself, without paying for a full build.

    NB conftest sets VIRTUROID_ALLOW_HEURISTIC_FALLBACK=1 session-wide; the PRODUCT default is "0", which is the
    only configuration where the llm_unavailable branch can be reached. Force it, or this asserts nothing."""
    monkeypatch.setenv("VIRTUROID_ALLOW_HEURISTIC_FALLBACK", "0")
    calls = []

    class _Intent:
        def __init__(self, conf):
            self.routing_confidence = conf
            self.concept = "quadruped"; self.concept_aliases = []; self.robot_class = "quadruped"
            self.task_family = "locomotion"; self.buildable = conf != "llm_unavailable"; self.gaps = []

    def fake_plan_build(prompt, llm=None, require_llm=False):
        calls.append({"llm": llm, "require_llm": require_llm})
        return _Intent("llm_unavailable" if require_llm else "task_inferred")

    monkeypatch.setattr("virturoid.services.intent_planner.plan_build", fake_plan_build)
    monkeypatch.setattr(AB, "_maybe_gene_build", lambda *a, **k: None)
    monkeypatch.setattr(AB, "build_robot_package_from_prompt", lambda **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):          # we only care that it got PAST the clarify branch
        AB.autonomous_build("a four-legged walking robot", output_dir="build/pt_offline_probe")
    assert len(calls) == 2, calls                            # first strict (require_llm), then the offline re-plan
    assert calls[0]["require_llm"] is True and calls[1]["require_llm"] is False
    assert calls[1]["llm"] is None                           # re-planned with the offline heuristic composer


def test_offline_note_is_re_persisted_not_just_held_in_memory(tmp_path, monkeypatch):
    """The gene route WRITES its report before returning, so stamping the note must also re-write it — otherwise
    the provenance exists only in memory and reports/autonomy_report.json silently omits how the body was
    designed. (Unit-boundary: capture the write; _write_report's own validation is its business.)"""
    wrote = []
    monkeypatch.setattr(AB, "_write_report", lambda out, rep: wrote.append((out, rep)))
    rep = AB.AutonomyReport(id="t", prompt="p", robot_class="quadruped", species="q.c", task_type="locomotion",
                            target_success_rate=0.7, feasible=True)
    AB._note_offline_design(rep, tmp_path)
    assert any("OFFLINE" in n for n in rep.notes)
    assert len(wrote) == 1 and wrote[0][0] == tmp_path       # re-persisted to the artifact a reviewer opens
    AB._note_offline_design(rep, tmp_path)                   # idempotent - never stamped or rewritten twice
    assert sum("OFFLINE" in n for n in rep.notes) == 1 and len(wrote) == 1


def test_provenance_never_fails_a_completed_build(tmp_path, monkeypatch):
    """A write/validation failure must not sink a build that already succeeded."""
    monkeypatch.setattr(AB, "_write_report", lambda out, rep: (_ for _ in ()).throw(ValueError("invalid")))
    rep = AB.AutonomyReport(id="t", prompt="p", robot_class="quadruped", species="q.c", task_type="locomotion",
                            target_success_rate=0.7, feasible=True)
    AB._note_offline_design(rep, tmp_path)                   # must not raise
    assert any("OFFLINE" in n for n in rep.notes)            # in-memory provenance still stands


def test_an_llm_backed_build_is_never_labelled_offline(monkeypatch):
    """Guard against the note leaking onto real LLM designs (that would be its own honesty bug)."""
    class _Ok:
        routing_confidence = "task_inferred"; concept = "quadruped"; concept_aliases = []
        robot_class = "quadruped"; task_family = "locomotion"; buildable = True; gaps = []
    monkeypatch.setattr("virturoid.services.intent_planner.plan_build", lambda *a, **k: _Ok())
    seen = {}
    def fake_gene_build(*a, **k):
        seen["hit"] = True
        return AB.AutonomyReport(id="g", prompt="p", robot_class="quadruped", species="q.c",
                                 task_type="locomotion", target_success_rate=0.7, feasible=True)
    monkeypatch.setattr(AB, "_maybe_gene_build", fake_gene_build)
    out = AB.autonomous_build("a four-legged walking robot", output_dir="build/pt_offline_probe2")
    assert seen.get("hit") and not any("OFFLINE" in (n or "") for n in (out.notes or []))


@pytest.mark.slow
def test_zero_token_build_produces_a_real_package(tmp_path, monkeypatch):
    """End-to-end: no LLM anywhere -> a real, multi-file robot package (was 0 files)."""
    monkeypatch.setenv("VIRTUROID_NO_INTERNAL_LLM", "1")
    monkeypatch.setenv("VIRTUROID_NO_LOCAL_ENV", "1")
    monkeypatch.setenv("VIRTUROID_ALLOW_HEURISTIC_FALLBACK", "0")   # the PRODUCT default (conftest presets "1")
    rep = AB.autonomous_build("a four-legged walking robot that patrols a warehouse", output_dir=tmp_path)
    assert rep.feasible and rep.robot_class
    assert sum(1 for p in tmp_path.rglob("*") if p.is_file()) > 20
    assert any("OFFLINE" in (n or "") for n in (rep.notes or []))
