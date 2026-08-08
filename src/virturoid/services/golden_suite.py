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

from virturoid.services.install_paths import anchored


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


# ANCHORED to the install (see ``services.install_paths``). These are RATCHETED floors -- the drift alarm that
# protects what the flywheel just learned -- and both halves fail silently when the path moves with the shell's
# pwd: ``seal_golden_floor`` writes a registry nothing will read, and ``load_golden_cases`` finds no file and
# quietly falls back to the base floors, so the alarm keeps passing at exactly the moment it stopped ratcheting.
_DEFAULT_FLOORS = str(anchored("build/golden_floors.json"))


def seal_golden_floor(task_id: str, verified_value: float, *, metric_key: str = "forward_m", margin: float = 0.9,
                      path: str = _DEFAULT_FLOORS) -> float:
    """Ratchet the golden floor for a NEWLY-BANKED capability to ``margin * verified`` (plan gap-closure N21):
    the drift alarm now protects what the flywheel just learned. NEVER lowers an existing floor (anti-Goodhart —
    a case is only ever tightened). Persisted to ``path`` so it survives across nights. Returns the sealed floor."""
    import json
    from pathlib import Path
    p = Path(path)
    floors = {}
    if p.exists():
        try:
            floors = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt registry must not crash a night; start fresh
            floors = {}
    key = f"{task_id}:{metric_key}"
    new_floor = round(margin * float(verified_value), 4)
    if key not in floors or new_floor > float(floors[key]["min_metric"]):
        floors[key] = {"task_id": task_id, "metric_key": metric_key, "min_metric": new_floor}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(floors, indent=2, sort_keys=True), encoding="utf-8")
    return float(floors[key]["min_metric"])


def load_golden_cases(path: str = _DEFAULT_FLOORS) -> list:
    """The base sealed cases PLUS any ratcheted floors persisted from banking nights (N21). A persisted floor that
    is HIGHER than the base replaces it (tighter is fine); the base list is never dropped or loosened."""
    import json
    from pathlib import Path
    cases = {f"{c.task_id}:{c.metric_key}": c for c in GOLDEN_CASES}
    p = Path(path)
    if p.exists():
        try:
            for key, f in json.loads(p.read_text(encoding="utf-8")).items():
                base = cases.get(key)
                floor = float(f["min_metric"])
                if base is None or floor > base.min_metric:
                    cases[key] = GoldenCase(f["task_id"], min_metric=floor, metric_key=f["metric_key"])
        except Exception:  # noqa: BLE001 - a corrupt registry falls back to the base cases (never fewer)
            pass
    return list(cases.values())


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
        pol = policy_for(c.task_id) if policy_for is not None else None
        # If a real ``policy_for`` is supplied (the night recalls the BANKED capability) but returns None, there is
        # no banked policy to protect for this case yet -> vacuous PASS (skip), so a fresh system can still bank its
        # FIRST capability instead of the sealed floor false-alarming on the random default controller.
        if policy_for is not None and pol is None:
            continue
        gene = gene_for(c.task_id)
        res = verify(c.task_id, gene, pol, **({"steps": steps} if steps is not None else {}))
        metric = float((res.get("metrics") or {}).get(c.metric_key, 0.0) or 0.0)
        regressed = metric < c.min_metric
        row = {"task": c.task_id, "metric_key": c.metric_key, "value": metric, "floor": c.min_metric,
               "regressed": regressed, "verified_pass": bool(res.get("verified_pass"))}
        results.append(row)
        if regressed:
            regressions.append(row)
    return {"passed": not regressions, "regressions": regressions, "results": results, "n": len(results)}
