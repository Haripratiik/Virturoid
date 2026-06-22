"""Task verifier: deterministic critics that reject an infeasible TaskSpec BEFORE it runs (LLM-Modulo —
the LLM proposes, the verifier disposes). Honest feasibility without learned value functions: it checks
the spec against the capability registry and the robot's actual morphology, so "make this wheeled base
grasp a cup" is rejected with a specific reason instead of silently mis-running.
"""

from __future__ import annotations

from virturoid.services.capability_registry import REGISTRY
from virturoid.services.task_matched_eval import robot_kind


def verify_task(spec, gene) -> dict:
    """Return ``{ok, issues, kind}``. ``issues`` are specific, actionable strings (the gaps to surface)."""
    issues: list[str] = []
    issues.extend(spec.validate())                       # structural

    kind = robot_kind(gene)
    established = set()                                  # facts available as we walk the steps in order

    for step in spec.steps:
        sig = REGISTRY.get(step.skill)
        # 1. registry membership
        if sig is None:
            issues.append(f"step {step.step_id!r}: unknown skill {step.skill!r} (not in capability registry)")
            continue
        # 2. morphology gate — the feasibility check no hard-coded handler gives
        if sig.morphologies and kind not in sig.morphologies:
            issues.append(f"step {step.step_id!r}: skill {step.skill!r} needs morphology "
                          f"{list(sig.morphologies)}, but this robot is '{kind}'")
        # 3. causal chain — preconditions satisfied by a prior step's effects (most v1 skills need none)
        for need in sig.requires:
            if need not in established:
                issues.append(f"step {step.step_id!r}: skill {step.skill!r} requires {need.value} which no "
                              f"prior step establishes")
        established |= set(sig.establishes)

    # 4. grounding — entities referenced by predicates must be declared
    declared = {e.id for e in spec.entities}
    for p in spec.goal + spec.failure:
        for ent in p.referenced_entities():
            if ent not in declared:
                issues.append(f"predicate {p.op.value} references undeclared entity {ent!r}")

    # 5. goal reachability — every goal fact must be establishable by some step
    for g in spec.goal:
        if g.op not in established:
            issues.append(f"goal {g.op.value} is not established by any step — the plan can't achieve it")

    return {"ok": not issues, "issues": issues, "kind": kind}
