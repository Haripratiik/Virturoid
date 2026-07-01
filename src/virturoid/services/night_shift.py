"""Night-shift engine core (breakthrough plan WS2 / N1) — the autonomous self-improvement loop.

The Dreamer mode: run the design-search harness on autopilot over a stream of candidate (body, task) proposals,
bank every verified win to the flywheel, and track the open-endedness metrics — so the platform is measurably
better every morning. This is the SAME harness as Engineer mode (design_search + engineer_mode + banking); the
only difference is that a proposal policy supplies the work instead of a user.

Week-1 core is LLM-FREE and GPU-agnostic (the evaluators are injected): give it proposals + a per-candidate
evaluator and it runs bounded searches, banks solved wins (verified-only), and reports:
* ``banked`` — verified wins written to memory,
* ``novel`` (ANNECS-V) — verified wins that were NEW (a fresh banked tip; a repeat solve on a known body
  writes nothing and is not counted) — the POET-style open-endedness counter,
* a resumable ``journal`` (G8 robustness): each candidate's outcome is persisted, so a crash/restart continues
  instead of re-doing banked work.

Pure-Python orchestration; the heavy lifting is in the injected ``evaluate_for``. Unit-tests with fakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from virturoid.services.engineer_mode import run_engineer_search
from virturoid.services.harness_banking import bank_search_result


@dataclass
class NightReport:
    candidates_run: int = 0
    banked: int = 0
    novel: int = 0                       # ANNECS-V: verified AND newly-banked
    budget_used: int = 0
    results: list = field(default_factory=list)
    stopped_reason: str = "proposals_exhausted"


def _load_journal(path):
    if path is None or not Path(path).exists():
        return {}
    done = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _append_journal(path, rec):
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def run_night_shift(proposals, evaluate_for, *, memory_dir, llm=None, budget_evals: int = 200,
                    per_candidate_evals: int = 8, gate_target: float = 0.30, journal_path=None,
                    on_result=None) -> NightReport:
    """Run the night shift over ``proposals`` (an iterable of candidate dicts, each ``{id, gene, task,
    task_type?, gates?}``). ``evaluate_for(candidate) -> evaluate`` builds the per-candidate physics evaluator
    (fidelity-ladder adapter in production, a test double in tests). Bounded by ``budget_evals`` total search
    evaluations. ``journal_path`` (G8) makes it resumable — already-journaled candidate ids are skipped."""
    from virturoid.services.robotics_vector_memory import cosine, embed_body
    done = _load_journal(journal_path)
    rep = NightReport()
    banked_embeds: list = []                                  # ANNECS-V: distinct banked bodies (embedding-space)

    for cand in proposals:
        cid = cand["id"]
        if cid in done:                                       # G8: resume — don't re-run a journaled candidate
            continue
        if rep.budget_used >= budget_evals:
            rep.stopped_reason = "budget_exhausted"
            break
        gene, task = cand["gene"], cand.get("task", "walk")
        task_type = cand.get("task_type", "locomotion")
        remaining = budget_evals - rep.budget_used
        search = run_engineer_search(
            task=task, gene=gene, evaluate=evaluate_for(cand), llm=llm, memory_dir=memory_dir,
            task_type=task_type, gates=cand.get("gates"), max_evals=min(per_candidate_evals, remaining),
            heuristic=cand.get("heuristic"))
        rep.budget_used += search.n_evals
        rep.candidates_run += 1

        banked = {"banked": False}
        if search.solved:
            banked = bank_search_result(search, gene=gene, memory_dir=memory_dir, task_type=task_type,
                                        gate_target=cand.get("gate_target", gate_target))
            if banked.get("banked"):
                rep.banked += 1
                # ANNECS-V novelty = a distinct body in MORPHOLOGY-EMBEDDING space (not tip-text): a near-identical
                # body already banked this run is coverage, not novelty (POET's minimal-criterion discipline).
                z = embed_body(gene)
                if all(cosine(z, e) < 0.98 for e in banked_embeds):
                    rep.novel += 1
                    banked_embeds.append(z)

        rec = {"id": cid, "solved": search.solved, "banked": bool(banked.get("banked")),
               "evals": search.n_evals, "stopped": search.stopped_reason,
               "best_mode": (search.best.artifact["failure_mode"] if search.best else None)}
        rep.results.append(rec)
        _append_journal(journal_path, rec)
        if on_result is not None:
            on_result(rec)

    return rep
