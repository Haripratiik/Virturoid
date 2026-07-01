"""Knowledge context — assemble a retrieved PRIOR-KNOWLEDGE block for an LLM role (Phase 2, the AURA move).

Phase 2 of the robotics-native-AI plan (``docs/robotics_native_ai_plan.md``): every LLM design role used to
reason BLIND. Here we assemble the block that un-blinds it — the body made legible (``describe_robot``) plus
the top retrieved tips/lessons/skills for THIS body (``recall_knowledge``), formatted as a compact
"PRIOR KNOWLEDGE" preface a role prepends to its prompt. This is the retrieval that moved AURA from 38% to
99%: same reasoner, given what worked before on similar bodies.

Two entry points, by what the caller has:

* ``assemble_prior_knowledge(memory, gene=..., task_type=...)`` — the z_body case (a critic/reward/planner
  role that HAS a body): full morphology-keyed recall (nearest bodies + tips + lessons + skills).
* ``assemble_prompt_context(memory, prompt=...)`` — the Designer case (``species_agent`` proposing a gene, so
  no body yet): lighter prior-art recall by task text (nearest prior runs + reusable skills).

A cheap **Selector** (top-k above a similarity floor) keeps the block short — dumping every match degrades the
generator (AURA). Verified-only knowledge is what the store holds, so the block is grounded, not hallucinated.
Pure-Python; returns ``""`` when there's nothing to inject, so callers prepend unconditionally + offline-safe.
"""

from __future__ import annotations

from pathlib import Path

from virturoid.schemas.gene import RobotGene


def _resolve_vm(memory):
    """Return ``(vector_memory, closer)`` from a memory dir path, a MemoryDB, or a RoboticsVectorMemory.

    ``closer`` is a DB to close when we opened it ourselves (else None). A missing DB opens empty -> the
    assembler returns "" (offline-safe); it never raises on absent memory.
    """
    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
    if isinstance(memory, RoboticsVectorMemory):
        return memory, None
    if isinstance(memory, (str, Path)):
        from virturoid.services.memory_db import MemoryDB
        db = MemoryDB(Path(memory) / "virturoid_memory.db")
        return RoboticsVectorMemory(db), db
    return RoboticsVectorMemory(memory), None   # a MemoryDB / connection


def _select(hits: list[dict], k: int, min_sim: float) -> list[dict]:
    """The Selector: top-k above a similarity floor (dumping all matches degrades the generator)."""
    return [h for h in hits if h.get("similarity", 0.0) >= min_sim][:k]


def assemble_prior_knowledge(memory, *, gene: RobotGene, task_type: str | None = None,
                             failure_code: str | None = None, k: int = 3, min_sim: float = 0.12) -> str:
    """The PRIOR-KNOWLEDGE block for a role that HAS a body: body summary + morphology-keyed recall.

    Returns a compact multi-line string (or ``""`` if nothing relevant). Grounded entirely in the vector
    store (verified-only) + the gene (``describe_robot``). Safe to prepend to any role's user prompt.
    """
    vm, closer = _resolve_vm(memory)
    try:
        from virturoid.services.robot_serializer import describe_robot
        know = vm.recall_knowledge(gene, task_type, failure_code=failure_code, k=k)
        try:
            bodies = _select(vm.nearest_bodies(gene, k=k, min_sim=min_sim), k, min_sim)
        except Exception:  # noqa: BLE001 - retrieval is best-effort; never break the role
            bodies = []
        tips = _select(know.get("tips", []), k, min_sim)
        lessons = _select(know.get("lessons", []), k, min_sim)
        skills = _select(know.get("skills", []), k, min_sim)
        if not (tips or lessons or skills or bodies):
            return ""

        lines = ["PRIOR KNOWLEDGE (retrieved for this body — proven on similar robots; physics still decides):"]
        summary = describe_robot(gene)["summary_text"].splitlines()
        if summary:
            lines.append(f"Body: {summary[0]}")
        if bodies:
            lines.append("Similar prior builds: " + ", ".join(
                f"{b['obj_id']} ({b['meta'].get('robot_class', '?')})" for b in bodies))
        for t in tips:
            m = t["meta"]
            lines.append(f"- TIP ({m.get('audience', 'builder')}): {m.get('tip')}")
        for le in lessons:
            m = le["meta"]
            imp = m.get("improvement")
            imp_s = f" (+{imp:.2f})" if isinstance(imp, (int, float)) else ""
            lines.append(f"- LESSON {m.get('failure_code')}: apply '{m.get('operator')}'"
                         + (f" — {m.get('root_cause')}" if m.get("root_cause") else "") + imp_s)
        for s in skills:
            m = s["meta"]
            lines.append(f"- REUSABLE SKILL: {m.get('task_type')} on {m.get('species')} "
                         f"(success {float(s.get('similarity', 0)):.2f} match) — reuse its recipe to warm-start")
        return "\n".join(lines)
    finally:
        if closer is not None:
            closer.close()


def assemble_prompt_context(memory, *, prompt: str, task_type: str | None = None,
                            robot_class: str | None = None, k: int = 3, min_sim: float = 0.12) -> str:
    """The lighter PRIOR-ART block for the Designer (proposing a gene, no body yet): nearest prior runs +
    reusable skills by task text. Returns ``""`` when nothing relevant."""
    vm, closer = _resolve_vm(memory)
    try:
        from virturoid.services.robotics_vector_memory import RUN, embed_task
        try:
            vm.index_runs()
            if vm.count("skill") == 0:
                vm.index_skills()
        except Exception:  # noqa: BLE001
            pass
        runs = _select(vm.nearest(RUN, embed_task(prompt, task_type, robot_class), k=k), k, min_sim)
        skills = _select(vm.nearest("skill", embed_task(prompt, task_type, robot_class), k=k), k, min_sim)
        if not (runs or skills):
            return ""
        lines = ["PRIOR ART (similar past work — reuse/amend rather than start cold):"]
        for r in runs:
            m = r["meta"]
            sr = m.get("success_rate")
            lines.append(f"- prior build: {m.get('task_type')} / {m.get('robot_class')}"
                         + (f" (success {sr:.0%})" if isinstance(sr, (int, float)) else ""))
        for s in skills:
            m = s["meta"]
            lines.append(f"- reusable skill: {m.get('task_type')} on {m.get('species')} ({m.get('robot_class')})")
        return "\n".join(lines)
    finally:
        if closer is not None:
            closer.close()
