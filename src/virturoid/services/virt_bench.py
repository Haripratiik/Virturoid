"""VIRT-Bench (breakthrough plan WS3) — the head-to-head benchmark that makes "outperforms Claude+MCP" a
falsifiable, externally-verified claim (docs/breakthrough_research_plan.md §7).

The core of a credible benchmark is an INDEPENDENT verifier that neither arm owns: it takes a *submitted
artifact* (a gene + a controller) for a frozen task, RE-RUNS it in physics, and applies the honesty gates —
never trusting the arm's self-reported success (the false-success literature: agents fabricate results in
8/10 tasks). This module is that verifier + the frozen task registry. Metrics an arm can't fake: verified
solve, and — with a shared memory dir across the suite — the transfer delta (repeat tasks) and honesty delta
(claimed vs verified). Reuses the H1 diagnosis artifact as the gate, so verification is the same anti-Goodhart
standard the harness selects on.

This is the proof AND the development compass: every task the platform fails is the next build priority. First
version verifies locomotion (recipe rollout) + manipulation (task-matched eval); the full external-container +
seed-disjoint protocol (§7) layers on top.
"""

from __future__ import annotations

from virturoid.services.diagnosis_artifact import build_diagnosis_artifact

# Frozen task registry (a first slice of the §7 suite; dev vs held-out split is a protocol layer above this).
# Plan v2 §3.1/N4: ``steps`` (eval horizon) and ``seed`` are FROZEN INSIDE the task so an arm cannot pick a
# flattering horizon (gates are horizon-dependent: forward_m 0.4 @900 != @2400). The verifier ignores any
# caller-supplied horizon by default and uses these. ``seed`` is disjoint from any training seed.
#
# WS3 (plan v3): ``upright`` is now MORPHOLOGY-RELATIVE. The deploy verifier's upright_frac counts steps at the
# body's natural stance height (upright_height_ratio: quad/biped 0.7*z0, hexapod 0.55, octopod+ 0.5) -- because
# no hexapod-RL result gates a low, statically-stable tripod as "not upright" (they walk at ~0.15 m with no
# upright reward at all; arXiv:2511.03167). The bar moves SIDEWAYS not down: many-legged tasks add a ``support``
# gate (tripod stepping-support fraction) so a low body must still show real feet-up/feet-down stepping, never a
# flat slide. Quad L1 is byte-identical (tau stays 0.7, no support gate).
VIRT_TASKS = [
    {"id": "L1_quad_walk", "family": "locomotion", "task_type": "locomotion",
     "prompt": "design a quadruped that walks", "gates": {"forward_m": 0.5, "cadence": 3.0, "upright": 0.6},
     "split": "dev", "steps": 900, "seed": 20260701},
    {"id": "L2_hex_walk", "family": "locomotion", "task_type": "locomotion",
     "prompt": "design a hexapod that walks",
     "gates": {"forward_m": 0.4, "cadence": 3.0, "upright": 0.6, "support": 0.3},
     "split": "held_out", "steps": 900, "seed": 20260701},
    {"id": "L3_octopod_walk", "family": "locomotion", "task_type": "locomotion",
     "prompt": "design an octopod that walks",
     "gates": {"forward_m": 0.4, "cadence": 3.0, "upright": 0.6, "support": 0.3},
     "split": "held_out", "steps": 900, "seed": 20260701},
    {"id": "L4_decapod_walk", "family": "locomotion", "task_type": "locomotion",
     "prompt": "design a ten-legged robot that walks",
     "gates": {"forward_m": 0.4, "cadence": 3.0, "upright": 0.6, "support": 0.3},
     "split": "dev", "steps": 900, "seed": 20260701},
    {"id": "M1_arm_grasp", "family": "manipulation", "task_type": "grasp",
     "prompt": "design an arm that picks up a 3 cm cube", "gates": {"success_rate": 0.6}, "split": "dev",
     "steps": 900, "seed": 20260701},
    {"id": "M2_arm_sort", "family": "manipulation", "task_type": "pick_place_sort",
     "prompt": "design an arm that sorts blocks by color", "gates": {"success_rate": 0.6}, "split": "held_out",
     "steps": 900, "seed": 20260701},
    {"id": "M3_arm_stack", "family": "manipulation", "task_type": "pick_place_sort",
     "prompt": "design an arm that stacks blocks into a tower", "gates": {"success_rate": 0.6}, "split": "held_out",
     "steps": 900, "seed": 20260701},
    {"id": "M4_arm_transport", "family": "manipulation", "task_type": "pick_place_sort",
     "prompt": "design an arm that transports blocks into a bin", "gates": {"success_rate": 0.6}, "split": "dev",
     "steps": 900, "seed": 20260701},
]

_BY_ID = {t["id"]: t for t in VIRT_TASKS}


def get_task(task_id: str) -> dict:
    if task_id not in _BY_ID:
        raise KeyError(f"unknown VIRT-Bench task {task_id!r}; known: {sorted(_BY_ID)}")
    return _BY_ID[task_id]


def list_tasks(split: str | None = None) -> list[dict]:
    return [t for t in VIRT_TASKS if split is None or t["split"] == split]


def verify_submission(task_id: str, gene, policy=None, *, steps: int | None = None,
                      decimation: int | None = None, seed: int | None = None) -> dict:
    """Independently verify a submitted (gene, controller) against a frozen task by RE-RUNNING it in physics and
    applying the honesty gate. Returns ``{task, verified_pass, failure_mode, metrics, summary, verifier}`` — the
    arm's own claim is never consulted. ``policy`` is a MorphPolicy (locomotion) / controller handle; ``None`` =
    the scripted/default controller (an honest floor).

    Plan v2 §3.1/§3.2: the eval HORIZON and SEED are FROZEN in the task (``steps`` overrides only for tests);
    the result records ``verifier`` = {sim, steps, seed, decimation} so a number can never be mistaken for one
    produced under a different protocol (the deploy gap makes CPU-vs-MJX provenance load-bearing). ``decimation``
    is the CONTROL rate the submitted policy expects (must match how it was trained — plan v2 T1.1); defaults to
    the policy's banked ``decimation`` metadata, else 1."""
    task = get_task(task_id)
    eval_steps = int(task.get("steps", 900) if steps is None else steps)
    # SEED is frozen in the task (§3.1); a caller may OVERRIDE it for the multi-seed protocol (plan gap-closure
    # N18 / WS4 -- ≥3 seeds for mean±SEM), which is recorded in ``verifier`` so provenance stays honest.
    seed = int(task.get("seed", 0) if seed is None else seed)
    dec = int(decimation if decimation is not None else getattr(policy, "decimation", 1) or 1)
    lpf = float(getattr(policy, "action_lpf", 0.0) or 0.0)    # deploy at the policy's banked LPF (T1.2, deploy==train)
    sf = bool(getattr(policy, "sphere_feet", False))          # deploy with the policy's banked foot model (T1.4)
    verifier = {"sim": "cpu-mujoco", "steps": eval_steps, "seed": seed, "decimation": dec,
                "action_lpf": lpf, "sphere_feet": sf}
    if task["family"] == "locomotion":
        from virturoid.services.morph_policy import recipe_rollout_morph
        r = recipe_rollout_morph(gene, policy, steps=eval_steps, seed=seed, decimation=dec,
                                 action_lpf=lpf, sphere_feet=sf)
        result = {"forward": r.get("forward"), "cadence": r.get("cadence"),
                  "upright_frac": r.get("upright_frac"), "support_frac": r.get("support_frac"),
                  "survived": r.get("survived")}
        verifier["upright_tau"] = r.get("upright_tau")        # per-body upright height threshold (WS3 provenance)
    else:
        from virturoid.services.task_matched_eval import evaluate_robot
        # MANIPULATION controller channel (plan gap-closure N22/WS6): an arm may SUBMIT searched skill parameters
        # (a dict: grip/PD gains, phase_steps, approach offsets) as ``policy`` -> the verifier re-runs the skill
        # WITH them. ``None`` (or a non-dict) = the default built-in skill (the honest floor). Either way the
        # verifier owns the physics re-run and the gate; the arm never grades itself.
        controller_params = policy if isinstance(policy, dict) else None
        res = evaluate_robot(gene, prompt=task["prompt"], controller_params=controller_params)
        result = {"success_rate": res.get("value"), "metric": res.get("metric")}

    art = build_diagnosis_artifact(result, task_type=task["task_type"], gates=task["gates"])
    return {"task": task_id, "verified_pass": art["verdict"] == "pass", "failure_mode": art["failure_mode"],
            "metrics": art["metrics"], "summary": art["summary_text"], "verifier": verifier}
