"""Pre-registration manifest for the VIRT-Bench head-to-head (breakthrough plan v2 §3.7 / M5).

A self-built benchmark is only credible if the tasks, seeds, budgets, metrics, and GRADER CODE are frozen and
hashed BEFORE the run, and predictions are declared up front (MLE-bench / METR / OSF practice). This module
builds that manifest: SHA256 of the frozen task registry + the verifier/scoreboard module source ("the grader
is code, and the grader is versioned"), the per-arm seed + budget tables, the arm definitions, and the declared
predictions. Writing it, committing it, and git-tagging the commit (``virtbench-prereg-vN``) with the manifest
hash in the README is the timestamp; a later run that doesn't hash-match isn't a replication.
"""

from __future__ import annotations

import hashlib
import json


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _module_hash(module) -> str:
    """Hash a module's SOURCE so the grader is versioned (a change to the verifier invalidates the pre-reg)."""
    import inspect
    try:
        return _sha256(inspect.getsource(module))
    except (OSError, TypeError):
        return "unavailable"


def build_prereg_manifest(*, arms=("A", "A+", "B"), seeds=(20260701, 20260702, 20260703),
                          budget=None, predictions=None, split: str | None = None) -> dict:
    """Freeze the current VIRT-Bench into a pre-registration manifest. Hashes the task registry + the verifier
    and arms MODULE SOURCE, records the seed table (>=3 per MLE-bench canon), per-arm budgets (identical totals
    incl. LLM calls), arm definitions, and the declared predictions + delta formulas. Deterministic (no clock/
    rng) so re-running before any edit reproduces the same ``manifest_hash`` -- the replication check."""
    from virturoid.services import virt_bench, virt_bench_arms
    tasks = [t for t in virt_bench.VIRT_TASKS if split is None or t["split"] == split]
    task_json = json.dumps(tasks, sort_keys=True)
    budget = budget or {"search_max_evals": 12, "gpu_iters": 100, "env_steps_cap": 4_000_000,
                        "llm_call_cap_total": 200}
    predictions = predictions or {"A": "2-4/12", "A+": "4-7/12", "B": "9-12/12",
                                  "transfer_delta": ">=3x repeat-task speedup (memory on vs off)",
                                  "honesty_delta": "~0 for B (claimed==verified)"}
    arm_defs = {
        "A0": "the literal Claude+MCP control: an LLM agent drives the PRIMITIVE MCP tools (inspect + simulate) "
              "and submits a controller — NO verified search harness, NO memory (plan v3 M1)",
        "A": "fixed pipeline: compose body + default controller, no search, NO memory (writes disabled)",
        "A+": "A + memory infra ON but COLD (no flywheel warm-start)",
        "B": "full harness: search + transfer-recall memory warm from the flywheel",
    }
    body = {
        "benchmark": "VIRT-Bench",
        "arm_ids": list(arms),                                # the exact arm set is part of the frozen protocol
        "arms": ({a: arm_defs[a] for a in arms if a in arm_defs} or {a: a for a in arms}),
        "tasks": tasks,
        "task_registry_sha256": _sha256(task_json),
        "verifier_sha256": _module_hash(virt_bench),          # the grader is code, and it is versioned
        "arms_sha256": _module_hash(virt_bench_arms),
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "budget_per_arm": budget,
        "metrics": ["verified_solve", "pass_cubed", "transfer_delta", "honesty_delta"],
        "delta_formulas": {
            "transfer_delta": "(budget_to_solve_A - budget_to_solve_B) / budget_to_solve_A",
            "honesty_delta": "claimed_solve_rate - verified_solve_rate (per arm, per family)",
        },
        "contamination_controls": ["per-arm fresh memory root (hashed at arm start)",
                                   "per-(arm,task,seed) PRNG derivation", "arms in separate processes",
                                   "stateless pinned-version LLM calls", "verify at the task's FROZEN horizon"],
        "predictions": predictions,
    }
    body["manifest_hash"] = _sha256(json.dumps(body, sort_keys=True))
    return body


def write_prereg(path: str, **kw) -> str:
    """Build the manifest and write it as pretty JSON; return the ``manifest_hash`` (to git-tag + cite in the
    README). Commit this file, then ``git tag virtbench-prereg-vN`` — that is the external timestamp."""
    from pathlib import Path
    m = build_prereg_manifest(**kw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    return m["manifest_hash"]
