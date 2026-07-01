"""Design-search harness (breakthrough plan H4 / night-shift N1 core) — the AIDE/AlphaEvolve-shaped spine.

The research is unanimous (docs/breakthrough_research_plan.md §3): the way to beat "Claude + a simulator" is
NOT a free-roaming agent (that loses 2× — AIDE vs OpenHands) but a thin CODED harness that owns budget,
candidate selection, and stopping, while the LLM owns every SEMANTIC decision. This is that harness.

* CODE owns: the budget (max_evals), SELECTION (``fitness_from_artifact`` over the honesty gates, on the
  eval the proposer never controls — AIRA's held-out-selection discipline), and STOPPING (gate pass / budget
  out / no-improvement-for-k).
* The caller owns SEMANTICS via two injected callables:
    - ``propose(parent_node | None, history) -> spec`` — the design operator (an LLM Proposer with diversity
      reflection later; a heuristic mutator or a test double now). ``spec`` is opaque to the harness.
    - ``evaluate(spec) -> result dict`` — runs the candidate down the fidelity ladder (surrogate → quick sim →
      full train) or a real/fake evaluator. Returns a rollout/eval result the Diagnosis Artifact (H1) reads.

Greedy parent selection (AIRA: greedy is fine before the operators are good; don't buy tree search first).
Every node carries its H1 artifact + fitness + lineage, so the whole search persists into the MAP-Elites
archive + provenance ledger. Pure-Python, fully injectable → unit-tests with no MuJoCo/LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.services.diagnosis_artifact import build_diagnosis_artifact, fitness_from_artifact


@dataclass
class Node:
    """One design-search node: a candidate + its verified evaluation, ready to persist to the archive."""
    node_id: int
    spec: object                      # the candidate design (gene/reward/config) — opaque to the harness
    parent_id: int | None
    result: dict
    artifact: dict                    # the H1 diagnosis artifact
    fitness: float
    verdict: str


@dataclass
class SearchReport:
    best: Node | None
    nodes: list = field(default_factory=list)
    n_evals: int = 0
    stopped_reason: str = ""
    solved: bool = False

    def tree(self) -> list[dict]:
        """Compact, honest tree of what was tried (for the report / archive / UI): id, parent, fitness, mode."""
        return [{"id": n.node_id, "parent": n.parent_id, "fitness": n.fitness, "verdict": n.verdict,
                 "failure_mode": n.artifact.get("failure_mode")} for n in self.nodes]


def run_design_search(*, propose, evaluate, task_type: str = "locomotion", gates: dict | None = None,
                      max_evals: int = 20, patience: int = 6, on_node=None) -> SearchReport:
    """Run the harness. Returns a ``SearchReport``. ``on_node(node)`` is an optional progress callback (e.g. to
    bank verified nodes or stream to a UI). Stops on: a passing node (solved), ``max_evals`` exhausted, the
    proposer returning ``None`` (exhausted), or ``patience`` consecutive evals with no fitness improvement."""
    nodes: list[Node] = []
    best: Node | None = None
    no_improve = 0
    stopped_reason = "budget_exhausted"

    for i in range(max_evals):
        parent = best                                     # greedy: improve the best node so far
        spec = propose(parent, nodes)
        if spec is None:
            stopped_reason = "proposer_exhausted"
            break
        result = evaluate(spec)
        artifact = build_diagnosis_artifact(result, task_type=task_type, gates=gates,
                                            history=[n.artifact["metrics"] for n in nodes] or None)
        fit = fitness_from_artifact(artifact)
        node = Node(node_id=i, spec=spec, parent_id=(parent.node_id if parent else None),
                    result=result, artifact=artifact, fitness=fit, verdict=artifact["verdict"])
        nodes.append(node)
        if on_node is not None:
            on_node(node)

        if best is None or fit > best.fitness + 1e-9:
            best, no_improve = node, 0
        else:
            no_improve += 1

        if artifact["verdict"] == "pass":                 # the gate is the search's accept — not a post-hoc report
            stopped_reason = "solved"
            break
        if no_improve >= patience:
            stopped_reason = "no_improvement"
            break

    return SearchReport(best=best, nodes=nodes, n_evals=len(nodes), stopped_reason=stopped_reason,
                        solved=bool(best is not None and best.verdict == "pass"))
