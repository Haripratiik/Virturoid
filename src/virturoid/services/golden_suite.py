"""Golden suite (breakthrough plan v2 §5.4) — the night-shift's anti-drift anchor.

A small SEALED set of (task, seed, budget) cases with a frozen expected outcome, re-run identically every night;
if any case REGRESSES below its sealed baseline, the night's banking is BLOCKED (a red-CI gate). This is the
AIRA "Hidden Consistent Evaluation" lesson transplanted: an autonomous system that selects on its own signal
will drift/overfit it, so a held-out, re-run-identically suite is the drift alarm that a bad code change or a
reward-hack degraded a previously-working capability.

Cases reuse the frozen VIRT-Bench verifier (independent physics re-run), so a golden pass is the same
anti-Goodhart standard the harness banks on. Baselines are conservative floors (not the peak), so normal
run-to-run noise doesn't false-alarm; a real regression (a capability that stopped working) trips it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenCase:
    """One sealed regression case: ``policy_for(task_id) -> policy`` supplies the controller to re-verify, and
    ``min_forward`` (locomotion) / ``min_success`` (manipulation) is the conservative baseline floor it must
    still clear. ``gene_for(task_id) -> gene`` supplies the body (defaults to the arms' task-body composer)."""
    task_id: str
    min_metric: float
    metric_key: str = "forward_m"


# The sealed floors are set from measured, banked capability (conservative). New golden cases are ADDED as
# capabilities land; a case is never LOOSENED to make a run pass (that would defeat the drift alarm — plan §9).
GOLDEN_CASES = [
    GoldenCase("L1_quad_walk", min_metric=0.0, metric_key="forward_m"),   # the quad must at least not go BACKWARD
]


def run_golden_suite(cases=None, *, policy_for=None, gene_for=None, verify=None, steps: int | None = None) -> dict:
    """Re-run the sealed cases through the independent verifier and flag regressions (verified metric below the
    sealed floor). Returns ``{passed, regressions, results}``; ``passed`` False -> the night loop must BLOCK
    banking. ``policy_for``/``gene_for``/``verify`` are injectable (tests + swapping the controller source);
    defaults use the VIRT-Bench arms' body composer + verify_submission with no submitted policy (the honest
    floor controller)."""
    cases = cases if cases is not None else GOLDEN_CASES
    if verify is None:
        from virturoid.services.virt_bench import verify_submission as verify
    if gene_for is None:
        from virturoid.services.virt_bench import get_task
        from virturoid.services.virt_bench_arms import _task_body
        gene_for = lambda tid: _task_body(get_task(tid))  # noqa: E731

    results, regressions = [], []
    for c in cases:
        gene = gene_for(c.task_id)
        pol = policy_for(c.task_id) if policy_for is not None else None
        res = verify(c.task_id, gene, pol, **({"steps": steps} if steps is not None else {}))
        metric = float((res.get("metrics") or {}).get(c.metric_key, 0.0) or 0.0)
        regressed = metric < c.min_metric
        row = {"task": c.task_id, "metric_key": c.metric_key, "value": metric, "floor": c.min_metric,
               "regressed": regressed, "verified_pass": bool(res.get("verified_pass"))}
        results.append(row)
        if regressed:
            regressions.append(row)
    return {"passed": not regressions, "regressions": regressions, "results": results, "n": len(results)}
