"""Demonstration amplifier — the dossier's "Bet 1" and the Training Improvement plan's Phase-1 highest-leverage lever.

MimicGen / RoboCasa / DexFlyWheel lesson: the cheapest way to cut expensive RL exploration is to turn ONE
(or a few) successful trajectory into MANY *physics-validated* variants, then behavior-clone / residual-train
on that data instead of searching from scratch. `training_ladder` already names ``demonstration_amplifier`` as
the first teacher source for pick-place/sort/tool-use — this module makes it real.

The engine is deliberately body/task-agnostic. It takes injected callables:

    scene_sampler(rng, seed_episode) -> new scene_params      # perturb the scene
    retarget_fn(seed_episode, scene_params) -> plan           # adapt the seed trajectory to the new scene
    replay_fn(plan, scene_params) -> {success, ...}           # run it in physics
    trusted_success(replay_result) -> bool                    # the TRUSTED gate (keep only real successes)

so grasp / push / reach / pick-place all plug in. It records full lineage (every attempt, accepted or not)
and the named 10x proof metric ``demo_amplification_yield`` (kept demonstrations per seed). A concrete MuJoCo
grasp instance (:func:`grasp_amplifier_fns` / :func:`amplify_grasp`) proves the yield is real on physics using
the scripted top-down grasp as the teacher — no pre-trained policy required.

The engine is deterministic for a given seed and has no numpy / backend dependency (AGENTS.md layering); only
the concrete grasp instance imports MuJoCo, lazily.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable


class _Rng:
    """Tiny deterministic RNG shim: uses numpy if present, else the standard library. ``uniform`` only."""

    def __init__(self, seed: int) -> None:
        try:
            import numpy as np  # noqa: F401
            self._np = np.random.default_rng(seed)
            self._py = None
        except Exception:  # noqa: BLE001 - numpy is optional for the engine
            import random
            self._np = None
            self._py = random.Random(seed)

    def uniform(self, low: float, high: float) -> float:
        if self._np is not None:
            return float(self._np.uniform(low, high))
        return float(self._py.uniform(low, high))


@dataclass
class Variant:
    """One amplification attempt (accepted or rejected) — the lineage record."""

    variant_id: str
    seed_id: str
    scene_params: dict
    accepted: bool
    return_: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id, "seed_id": self.seed_id, "scene_params": self.scene_params,
            "accepted": self.accepted, "return": round(self.return_, 4), "reason": self.reason,
        }


@dataclass
class AmplificationResult:
    """Result of amplifying seed demonstrations: kept (validated) episodes + full lineage + yield metrics."""

    task: str
    seed_ids: list[str]
    attempted: int
    kept: int
    episodes: list[dict] = field(default_factory=list)   # kept, demonstration-shaped (pass-through trajectory)
    lineage: list[Variant] = field(default_factory=list)  # EVERY attempt, accepted or not

    @property
    def seed_count(self) -> int:
        return len(self.seed_ids)

    @property
    def yield_per_seed(self) -> float:
        """demo_amplification_yield: validated demonstrations produced per seed demonstration."""
        return round(self.kept / self.seed_count, 4) if self.seed_count else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of retargeted variants that passed the trusted success gate."""
        return round(self.kept / self.attempted, 4) if self.attempted else 0.0

    def report(self) -> dict:
        rejected: dict[str, int] = {}
        for v in self.lineage:
            if not v.accepted:
                key = v.reason or "unspecified"
                rejected[key] = rejected.get(key, 0) + 1
        return {
            "task": self.task,
            "seed_count": self.seed_count,
            "attempted": self.attempted,
            "kept": self.kept,
            "demo_amplification_yield": self.yield_per_seed,   # named 10x proof metric
            "acceptance_rate": self.acceptance_rate,
            "rejected_by_reason": rejected,
            "seed_ids": list(self.seed_ids),
        }


def _make_episode(variant_id: str, seed_id: str, seed_ep: dict, scene: dict, result: dict, task: str) -> dict:
    """Build a demonstration-dataset-shaped episode from an accepted replay.

    If the replay recorded a full ``trajectory`` (obs/actions/...), pass it through so the variant is directly
    trainable; otherwise keep a compact *physics-validated record* (scene + outcome + lineage) — still a real
    datum for scene-difficulty ranking, curriculum, and provenance.
    """
    episode: dict[str, Any] = {
        "episode_id": variant_id,
        "amplified_from": seed_id,
        "success": True,
        "scene_params": dict(scene),
        "task_language": seed_ep.get("task_language", task),
        "robot_genome_id": seed_ep.get("robot_genome_id", ""),
        "source": "amplified",
        "return": float(result.get("return", result.get("dense", 0.0))),
    }
    trajectory = result.get("trajectory")
    if isinstance(trajectory, dict):
        episode.update(trajectory)                              # obs/actions/rewards/robot_qpos/... -> trainable
    return episode


def amplify_demonstrations(
    seed_episodes: list[dict],
    *,
    n_variants: int,
    scene_sampler: Callable[[Any, dict], dict],
    retarget_fn: Callable[[dict, dict], Any],
    replay_fn: Callable[[Any, dict], dict],
    trusted_success: Callable[[dict], bool] | None = None,
    task: str = "amplified",
    seed: int = 0,
) -> AmplificationResult:
    """Amplify ``seed_episodes`` into ``n_variants`` retargeted, physics-filtered variants each.

    For every seed x variant: sample a scene, retarget the seed to it, replay it, and KEEP ONLY variants that
    pass ``trusted_success`` (default: ``result["success"]``). Records every attempt in the lineage. Deterministic
    for a given ``seed``.
    """
    if n_variants < 0:
        raise ValueError("n_variants must be >= 0")
    if not seed_episodes:
        return AmplificationResult(task=task, seed_ids=[], attempted=0, kept=0)
    gate = trusted_success or (lambda r: bool(r.get("success")))
    rng = _Rng(seed)

    kept: list[dict] = []
    lineage: list[Variant] = []
    attempted = 0
    seed_ids: list[str] = []
    for si, seed_ep in enumerate(seed_episodes):
        seed_id = seed_ep.get("episode_id") or f"seed_{si:03d}"
        seed_ids.append(seed_id)
        for vi in range(n_variants):
            attempted += 1
            scene = scene_sampler(rng, seed_ep)
            plan = retarget_fn(seed_ep, scene)
            result = replay_fn(plan, scene) or {}
            accepted = bool(gate(result))
            variant_id = f"{seed_id}__amp_{vi:03d}"
            lineage.append(Variant(
                variant_id=variant_id, seed_id=seed_id, scene_params=dict(scene), accepted=accepted,
                return_=float(result.get("return", result.get("dense", 0.0))),
                reason=str(result.get("reason") or ("" if accepted else "rejected")),
            ))
            if accepted:
                kept.append(_make_episode(variant_id, seed_id, seed_ep, scene, result, task))
    return AmplificationResult(
        task=task, seed_ids=seed_ids, attempted=attempted, kept=len(kept), episodes=kept, lineage=lineage,
    )


def write_amplification(result: AmplificationResult, out_dir: str) -> str:
    """Write ``<out_dir>/amplification_report.json`` (report + kept-episode index + full lineage). Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "amplification_report.json")
    payload = {
        **result.report(),
        "kept_episode_ids": [ep["episode_id"] for ep in result.episodes],
        "lineage": [v.to_dict() for v in result.lineage],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------------------------------------------
# Concrete MuJoCo grasp instance — proves the yield is real on physics, using the SCRIPTED grasp as the teacher.
# --------------------------------------------------------------------------------------------------------------

_DEFAULT_REGION = {"x": (0.40, 0.48), "y": (-0.08, 0.08)}


def _default_arm_gene():
    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper
    return add_parallel_gripper(tabletop_arm_gene())


def _resolve_grasp_teacher(gene, policy):
    """Resolve the grasp TEACHER: an explicit policy, else this body's banked grasp residual, else the scripted
    base. Returns ``(policy, residual_scale, fin_residual_scale)`` — the scripted base (policy=None) makes no
    contact for most arms, so the banked residual is the real teacher when present."""
    rs, frs = 0.12, 0.04
    if policy is not None:
        return policy, rs, frs
    try:
        from virturoid.services.grasp_skill import banked_grasp_policy
        pol, scales = banked_grasp_policy(gene)
        if pol is not None:
            scales = scales or {}
            return pol, float(scales.get("residual_scale", rs)), float(scales.get("fin_residual_scale", frs))
    except Exception:  # noqa: BLE001 - fall back to the scripted base
        pass
    return None, rs, frs


def grasp_amplifier_fns(gene, *, policy=None, residual_scale: float = 0.12, fin_residual_scale: float = 0.04,
                        region: dict | None = None, jitter: tuple[float, float] = (0.03, 0.05), ep_len: int = 260):
    """Build (scene_sampler, retarget_fn, replay_fn) that amplify a top-down grasp via REAL MuJoCo physics.

    The teacher is the supplied ``policy`` (a per-token MorphPolicy grasp residual). The retarget is
    object-relative: the grasp re-plans IK to the sampled object xy, so replaying the teacher at the perturbed
    position IS the retargeted trajectory. Variants are clipped to the reachable ``region`` so the amplifier
    never proposes an unreachable (guaranteed-fail) scene.
    """
    region = region or _DEFAULT_REGION
    rx, ry = tuple(region["x"]), tuple(region["y"])
    jx, jy = jitter

    def scene_sampler(rng, seed_ep: dict) -> dict:
        sp = seed_ep.get("scene_params") or {}
        cx, cy = sp.get("box_x"), sp.get("box_y")
        if cx is None or cy is None:                            # no seed anchor -> sample the whole region
            return {"box_x": rng.uniform(*rx), "box_y": rng.uniform(*ry)}
        return {
            "box_x": min(max(cx + rng.uniform(-jx, jx), rx[0]), rx[1]),
            "box_y": min(max(cy + rng.uniform(-jy, jy), ry[0]), ry[1]),
        }

    def retarget_fn(seed_ep: dict, scene: dict) -> dict:
        return {"box_x": scene["box_x"], "box_y": scene["box_y"]}

    def replay_fn(plan: dict, scene: dict) -> dict:
        from virturoid.services.grasp_skill import morph_grasp_rollout
        r = morph_grasp_rollout(gene, policy, plan["box_x"], plan["box_y"], ep_len=ep_len,
                                residual_scale=residual_scale, fin_residual_scale=fin_residual_scale)
        return {
            "success": bool(r.get("success")), "return": float(r.get("dense", 0.0)),
            "reason": r.get("reason") or "", "lifted": r.get("lifted"), "contacts": r.get("contacts"),
        }

    return scene_sampler, retarget_fn, replay_fn


def record_grasp_seeds(gene=None, *, policy=None, positions=None, ep_len: int = 260) -> list[dict]:
    """Record a few SUCCESSFUL grasp seeds with the resolved teacher (the 'few demos' the amplifier grows from)."""
    from virturoid.services.grasp_skill import morph_grasp_rollout
    gene = gene or _default_arm_gene()
    policy, rs, frs = _resolve_grasp_teacher(gene, policy)
    positions = positions or [(0.44, 0.0), (0.44, 0.05), (0.44, -0.05)]
    seeds: list[dict] = []
    for i, (gx, gy) in enumerate(positions):
        r = morph_grasp_rollout(gene, policy, gx, gy, ep_len=ep_len,
                                residual_scale=rs, fin_residual_scale=frs)
        if r.get("success"):
            seeds.append({
                "episode_id": f"grasp_seed_{i:03d}", "success": True,
                "scene_params": {"box_x": float(gx), "box_y": float(gy)},
                "task_language": "grasp the cube and lift it off the table",
                "robot_genome_id": getattr(gene, "id", ""),
                "source": "banked_teacher" if policy is not None else "scripted_teacher",
            })
    return seeds


def amplify_grasp(gene=None, *, seeds: list[dict] | None = None, n_variants: int = 8,
                  region: dict | None = None, policy=None, ep_len: int = 260, seed: int = 0) -> AmplificationResult:
    """End-to-end concrete instance: record (or accept) grasp seeds, then amplify them on real physics.

    Uses this body's banked grasp residual as the teacher when no policy is supplied (the scripted base alone
    rarely makes contact). Returns an :class:`AmplificationResult` whose ``demo_amplification_yield`` is measured,
    not asserted; if no competent teacher can produce a seed success, returns an empty result rather than
    fabricating demonstrations.
    """
    gene = gene or _default_arm_gene()
    policy, rs, frs = _resolve_grasp_teacher(gene, policy)
    if seeds is None:
        seeds = record_grasp_seeds(gene, policy=policy, ep_len=ep_len)
    scene_sampler, retarget_fn, replay_fn = grasp_amplifier_fns(
        gene, policy=policy, residual_scale=rs, fin_residual_scale=frs, region=region, ep_len=ep_len)
    return amplify_demonstrations(
        seeds, n_variants=n_variants, scene_sampler=scene_sampler, retarget_fn=retarget_fn,
        replay_fn=replay_fn, task="grasp", seed=seed,
    )


# --------------------------------------------------------------------------------------------------------------
# Concrete MuJoCo LOCOMOTION instance — amplifies one walking body into many cadence-varied, physics-validated
# gait demonstrations. Robust on CPU (the wave-gait engine walks any leg count), so it is the default proof.
# --------------------------------------------------------------------------------------------------------------

def _walkable_gene(prompt: str):
    from virturoid.services.anatomy_compiler import ensure_walkable_quad
    from virturoid.services.morphology_composer import compose_robot
    return ensure_walkable_quad(compose_robot(prompt), prompt)


def gait_amplifier_fns(gene, *, base_freq: float = 1.4, freq_jitter: float = 0.9, steps: int = 1200,
                       min_forward: float = 0.05, min_height_ratio: float = 0.6):
    """(scene_sampler, retarget_fn, replay_fn) that amplify a walking gait across command variations.

    A "scene" is a locomotion command (gait cadence); the retargeted plan is that cadence; the replay runs the
    one wave-gait engine and the TRUSTED gate keeps a variant only if it walks forward, upright, and survives.
    Wide cadence jitter makes extreme cadences fall, so the gate genuinely filters (not everything passes).
    """
    from virturoid.services.morph_policy import crawl_gait_rollout

    def scene_sampler(rng, seed_ep: dict) -> dict:
        return {"freq": round(max(0.5, base_freq + rng.uniform(-freq_jitter, freq_jitter)), 3)}

    def retarget_fn(seed_ep: dict, scene: dict) -> dict:
        return {"freq": scene["freq"]}

    def replay_fn(plan: dict, scene: dict) -> dict:
        r = crawl_gait_rollout(gene, steps=steps, freq=plan["freq"])
        fwd = float(r.get("forward", 0.0))
        hr = float(r.get("height_ratio", 0.0))
        survived = bool(r.get("survived"))
        ok = fwd >= min_forward and hr >= min_height_ratio and survived
        reason = "" if ok else ("fell" if not survived else "low" if hr < min_height_ratio else "too_slow")
        return {"success": ok, "return": fwd, "reason": reason, "forward": fwd, "height_ratio": round(hr, 3)}

    return scene_sampler, retarget_fn, replay_fn


def amplify_gait(gene=None, *, prompt: str = "a quadruped robot dog", seeds: list[dict] | None = None,
                 n_variants: int = 6, base_freq: float = 1.4, steps: int = 1200, seed: int = 0) -> AmplificationResult:
    """End-to-end locomotion amplifier: one walking body -> many cadence-varied, physics-validated gait demos.

    ``demo_amplification_yield`` is measured on real MuJoCo rollouts of the one wave-gait engine.
    """
    gene = gene or _walkable_gene(prompt)
    scene_sampler, retarget_fn, replay_fn = gait_amplifier_fns(gene, base_freq=base_freq, steps=steps)
    if seeds is None:                                          # the seed is the body walking at the base cadence
        base = replay_fn({"freq": base_freq}, {"freq": base_freq})
        seeds = [{"episode_id": "gait_seed_000", "success": bool(base["success"]),
                  "scene_params": {"freq": base_freq}, "task_language": prompt,
                  "robot_genome_id": getattr(gene, "id", ""), "source": "wave_gait_teacher"}]
        if not base["success"]:
            return AmplificationResult(task="gait", seed_ids=[], attempted=0, kept=0)
    return amplify_demonstrations(
        seeds, n_variants=n_variants, scene_sampler=scene_sampler, retarget_fn=retarget_fn,
        replay_fn=replay_fn, task="gait", seed=seed,
    )
