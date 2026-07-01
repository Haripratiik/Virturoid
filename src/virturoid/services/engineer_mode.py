"""Engineer mode (breakthrough plan §4, Mode 1) — the user-driven entry point of the design-search harness.

Ties the harness bricks together for ONE body + task: assemble the morphology-keyed PRIOR-KNOWLEDGE block
(H5, memory-in-the-loop — reusing the Phase-2 ``assemble_prior_knowledge``), build the LLM proposer with that
memory + a heuristic fallback (H3), and run the search (H4) whose selection is the honesty gate (H1/H2).

This is the loop that replaces the fixed ``autonomous_build`` pipeline as the product's core: a prompt starts a
bounded design search that reasons over retrieved experience, proposes diverse edits, verifies each in physics,
and stops on a gate pass / budget / no-improvement — returning the winner plus an honest tree of what it tried.
``evaluate`` is injected (fidelity-ladder adapter in production, a test double in tests), so this unit-tests
without MuJoCo/LLM. Memory read here (H5) closes the flywheel's missing in-loop consumption.
"""

from __future__ import annotations

from virturoid.services.design_search import SearchReport, run_design_search
from virturoid.services.search_operators import make_llm_proposer


def run_engineer_search(*, task: str, gene, evaluate, llm=None, memory=None, memory_dir=None,
                        gates: dict | None = None, task_type: str = "locomotion", max_evals: int = 20,
                        patience: int = 6, k: int = 3, heuristic=None) -> SearchReport:
    """Run one Engineer-mode design search over ``gene`` for ``task``.

    ``evaluate(spec) -> result`` runs a candidate (fidelity-ladder adapter / test double). ``llm`` drives the
    Proposer/Critic (None → the ``heuristic`` proposer, i.e. the LLM-free path). ``memory``/``memory_dir`` (a
    RoboticsVectorMemory/MemoryDB or a dir) feeds the morphology-keyed PRIOR-KNOWLEDGE block into every proposal
    prompt (H5). Returns the ``SearchReport`` (best node + honest tree)."""
    memory_block = ""
    src = memory if memory is not None else memory_dir
    if src is not None and gene is not None:
        try:
            from virturoid.services.knowledge_context import assemble_prior_knowledge
            memory_block = assemble_prior_knowledge(src, gene=gene, task_type=task_type)
        except Exception:  # noqa: BLE001 - memory is best-effort; a missing store must not break the search
            memory_block = ""

    propose = make_llm_proposer(task, llm, memory_block=memory_block, k=k, screen=True, heuristic=heuristic)
    return run_design_search(propose=propose, evaluate=evaluate, task_type=task_type, gates=gates,
                             max_evals=max_evals, patience=patience)
