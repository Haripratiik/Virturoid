"""Search operators (breakthrough plan H3) — the LLM-owned SEMANTIC layer of the design-search harness.

The harness spine (design_search.py) owns budget/selection/stopping; these operators own *what to try next*.
Three roles, each a schema-validated LLM call (offline-safe; None → the caller falls back to a heuristic):

* Proposer — emits K design EDITS, prompted with **diversity reflection**: it sees every prior candidate and is
  told to be maximally different from all of them. This one prompt mechanic is the single biggest measured lever
  in morphology search (RoboMoRe: 487 vs 137 vs 13 — ~38× over random; docs/breakthrough_research_plan.md §3).
* Critic — screens ONE proposal *before* any evaluation is spent (Voyager's most-important feedback channel;
  Debate2Create's untrained thesis screen). Kills obviously-bad proposals for free.
* Diagnostician — reads the H1 diagnosis artifact and names which knob to turn next (MLE-STAR's measured-
  attribution: edit the block that matters, don't rewrite the whole thing).

A design EDIT is a small typed dict the evaluate() adapter applies to a base design: ``{edit_kind, params,
rationale}`` over a fixed vocabulary (cpg gait params / reward weights / gains / gene amendment) — a constrained
space keeps the LLM's proposals valid (RoboMorph/Eureka lesson). ``make_llm_proposer`` wires all three into the
harness's ``propose(parent, history)`` signature. Pure-Python; the LLM is injected, so it unit-tests with a
fake that records prompts (the Phase-2 pattern)."""

from __future__ import annotations

EDIT_KINDS = ("cpg", "reward", "gains", "gene_amend")

_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "edit_kind": {"type": "string", "enum": list(EDIT_KINDS)},
        "params": {"type": "object"},
        "rationale": {"type": "string"},
    },
    "required": ["edit_kind", "params"],
    "additionalProperties": False,
}
_PROPOSALS_SCHEMA = {
    "type": "object",
    "properties": {"proposals": {"type": "array", "items": _EDIT_SCHEMA}},
    "required": ["proposals"], "additionalProperties": False,
}
_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"viable": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["viable"], "additionalProperties": False,
}

_PROPOSER_SYSTEM = (
    "You are Virturoid's design-search Proposer. You edit a robot's controller/body to pass a physics gate. "
    "Output ONLY JSON {proposals:[{edit_kind, params, rationale}]}. edit_kind is one of: 'cpg' (gait prior: "
    "params may set calf_phase, freq, thigh_amp, calf_amp), 'reward' (params = weight names→values), 'gains' "
    "(params: adaptive true/false, kp, kd), 'gene_amend' (params: operator like lengthen_reach, and its args). "
    "CRITICAL: propose edits MAXIMALLY DIFFERENT from every prior candidate listed — cover distinct hypotheses, "
    "do not repeat a prior edit or make a tiny variation of it. Ground each in the diagnosis. Physics decides; "
    "propose boldly and diversely.")
_DIAGNOSTICIAN_SYSTEM = (
    "You are Virturoid's design-search Diagnostician. Read the evaluation diagnosis and name the SINGLE most "
    "promising knob to turn next. Output ONLY JSON {edit_kind, params, rationale} — one targeted edit.")
_CRITIC_SYSTEM = (
    "You are Virturoid's design-search Critic. Given a proposed edit + the current diagnosis, judge whether it "
    "is worth spending a physics evaluation on (viable + non-duplicative + addresses the failure). Output ONLY "
    "JSON {viable, reason}. Be willing to reject: a wasted evaluation is expensive, an LLM call is not.")


def _priors_block(prior_specs: list) -> str:
    if not prior_specs:
        return "(no prior candidates yet)"
    return "\n".join(f"  {i}. {s.get('edit_kind', '?')}: {s.get('params', {})}" for i, s in enumerate(prior_specs))


def propose_designs(task: str, artifact: dict, prior_specs: list, llm, *, memory_block: str = "",
                    k: int = 3) -> list[dict]:
    """K diverse design edits given the task, the H1 artifact, prior candidates (diversity reflection), and
    optional retrieved-memory block. Returns [] if no backend or on failure (caller falls back to heuristic)."""
    if llm is None:
        return []
    user = (
        (f"{memory_block}\n\n" if memory_block else "")
        + f"Task: {task}\n"
        + f"Latest evaluation diagnosis:\n{artifact.get('summary_text', '')}\n"
        + f"Suggested next-action menu: {artifact.get('next_actions', [])}\n"
        + f"PRIOR CANDIDATES (be maximally different from ALL of these):\n{_priors_block(prior_specs)}\n"
        + f"Propose {k} maximally-different viable edits as JSON {{proposals:[...]}}.")
    try:
        raw = llm.complete_json(_PROPOSER_SYSTEM, user, _PROPOSALS_SCHEMA)
        return [p for p in (raw.get("proposals") or []) if p.get("edit_kind") in EDIT_KINDS][:k]
    except Exception:  # noqa: BLE001 - offline/parse failure -> heuristic fallback in the caller
        return []


def diagnose_next_edit(task: str, artifact: dict, llm) -> dict | None:
    """One targeted edit the Diagnostician recommends from the artifact, or None (no backend/failure)."""
    if llm is None:
        return None
    user = f"Task: {task}\nDiagnosis:\n{artifact.get('summary_text', '')}\nName the single best next edit as JSON."
    try:
        raw = llm.complete_json(_DIAGNOSTICIAN_SYSTEM, user, _EDIT_SCHEMA)
        return raw if raw.get("edit_kind") in EDIT_KINDS else None
    except Exception:  # noqa: BLE001
        return None


def critique_design(spec: dict, task: str, artifact: dict, llm) -> dict:
    """Screen a proposal before any evaluation is spent. Returns {viable, reason}. Fails OPEN (viable=True) with
    no backend — the physics gate is the real filter; the critic is a cheap accelerator, never a hard blocker."""
    if llm is None:
        return {"viable": True, "reason": "no critic backend (physics gate will decide)"}
    user = (f"Task: {task}\nCurrent diagnosis:\n{artifact.get('summary_text', '')}\n"
            f"Proposed edit: {spec}\nIs it worth a physics evaluation? JSON {{viable, reason}}.")
    try:
        raw = llm.complete_json(_CRITIC_SYSTEM, user, _CRITIC_SCHEMA)
        return {"viable": bool(raw.get("viable", True)), "reason": raw.get("reason", "")}
    except Exception:  # noqa: BLE001
        return {"viable": True, "reason": "critic error -> defer to physics gate"}


def make_llm_proposer(task: str, llm, *, memory_block: str = "", k: int = 3, screen: bool = True,
                      heuristic=None):
    """Adapt the operators into the harness's ``propose(parent, history) -> spec`` callable.

    Maintains a queue of proposals; refills via ``propose_designs`` using the parent's artifact + ALL prior
    specs (diversity reflection) whenever empty. If ``screen``, each proposal is Critic-screened and skipped if
    non-viable. ``heuristic(parent, history) -> spec`` is the offline/empty fallback (the LLM-free N1 path)."""
    queue: list[dict] = []
    seen: list[dict] = []

    def propose(parent, history):
        artifact = parent.artifact if parent is not None else {"summary_text": "(no evaluation yet — cold start)",
                                                               "next_actions": []}
        # refill the queue with a fresh diverse batch when empty
        tries = 0
        while not queue and tries < 2:
            tries += 1
            batch = propose_designs(task, artifact, seen, llm, memory_block=memory_block, k=k)
            for spec in batch:
                if screen and not critique_design(spec, task, artifact, llm).get("viable", True):
                    seen.append(spec)                          # remember rejects too (don't re-propose them)
                    continue
                queue.append(spec)
            if not batch:
                break
        if queue:
            spec = queue.pop(0)
            seen.append(spec)
            return spec
        return heuristic(parent, history) if heuristic is not None else None   # LLM-free fallback / stop

    return propose
