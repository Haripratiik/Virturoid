"""Skill flywheel (M3 G4): bank trained control policies + warm-start new builds from them.

This is the flywheel applied to *policies* — the third leg of Virturoid's compounding memory, beside
the design flywheel (``MemoryDB.designs`` / ``best_design``) and the reasoned-redesign flywheel
(``MemoryDB.lessons`` / ``recall_lesson``). When a robot is trained to do a task, the resulting policy
is banked with everything needed to reuse it: the gene/species it ran on, the process-aware reward
recipe, the body<->task-matched region it is valid in, the base-controller config, and a pointer to
the saved policy params. A later build on a related body/task **retrieves the nearest banked skill and
warm-starts** its policy instead of learning from scratch — so the next robot learns faster than the
last (startup plan §13 memory, §32 training; skill-library transfer arXiv:2411.11714).

The trainer (``scripts/mjx_residual_grasp.py``) emits a one-line ``SKILL_CARD {json}`` at the end of a
run; ``bank_skill_from_card`` turns that into a banked skill. ``warm_start_for`` is the read side the
next trainer calls to decide what to seed from. Both are pure-Python and offline (no GPU/JAX) — the
GPU only produces the params file the card points to.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Trainers whose saved params can warm-start a *future* run of a compatible body. The tabletop
# CEM reach trainer (``policy_trainer.train_arm_controller_from_export``) re-initialises its
# linear policy every run and cannot consume prior weights, so its banked skills warm-start in
# 'config' mode (reuse the success record + base config), not 'params' mode. Param-reusing
# trainers (the MJX PPO / residual-grasp leg) bank a real ``params_path`` for direct transfer.
_PARAM_REUSE_TRAINERS = frozenset({"mjx_ppo", "mjx_morph_ppo", "mjx_residual", "residual_grasp"})


def skill_id_for(task_type: str, species: str | None, robot_class: str | None = None) -> str:
    """Stable skill id from task + species (falls back to robot_class). Idempotent banking key."""
    base = species or robot_class or "unknown"
    slug = _SLUG_RE.sub(".", f"{task_type}.{base}".lower()).strip(".")
    return f"{task_type}__{slug}" if not slug.startswith(f"{task_type}.") else slug


def bank_skill_from_card(memory, card: dict | str, *, params_path: str | None = None,
                         notes: str | None = None) -> str:
    """Bank a trained skill from a trainer SKILL_CARD (dict or JSON string). Returns the skill_id.

    ``params_path`` overrides the card's pointer (e.g. once the params file is copied off the GPU box).
    Idempotent: re-banking the same skill_id keeps the best success (MemoryDB.record_skill).
    """
    if isinstance(card, str):
        card = json.loads(card[len("SKILL_CARD "):] if card.startswith("SKILL_CARD ") else card)
    task_type = card["task_type"]
    species = card.get("species")
    robot_class = card.get("robot_class", "manipulator")
    sid = skill_id_for(task_type, species, robot_class)
    memory.record_skill(
        sid, robot_class, task_type,
        success_rate=float(card.get("success_rate", 0.0)),
        params_path=params_path or card.get("params_path"),
        species=species, gene_id=card.get("gene_id"),
        reward_spec=card.get("reward_spec"), region=card.get("region"),
        base_config=card.get("base_config"),
        obs_dim=card.get("obs_dim"), act_dim=card.get("act_dim"), notes=notes,
    )
    return sid


def warm_start_for(memory, robot_class: str, task_type: str, *, species: str | None = None,
                   obs_dim: int | None = None, act_dim: int | None = None,
                   min_success: float = 0.3) -> dict | None:
    """Pick the best banked skill to warm-start a new (class, task, body) policy — the flywheel read.

    Strategy (best-effort, never blocks a build):
      1. exact class+task, IO-compatible (same obs/act dims) → direct param warm-start;
      2. exact class+task, any IO → reuse its reward recipe + region + base config (re-init params);
      3. cross-body similar skill (same task, nearest species) → transfer candidate.
    Returns the chosen skill dict augmented with ``warm_start`` ("params" | "config" | "transfer"),
    or ``None`` when nothing banked clears ``min_success`` (the trainer then cold-starts).
    """
    exact_io = memory.recall_skill(robot_class, task_type, species=species, obs_dim=obs_dim, act_dim=act_dim)
    if exact_io and (exact_io.get("success_rate") or 0) >= min_success and exact_io.get("params_path"):
        return {**exact_io, "warm_start": "params"}
    exact = memory.recall_skill(robot_class, task_type, species=species)
    if exact and (exact.get("success_rate") or 0) >= min_success:
        return {**exact, "warm_start": "config"}
    for cand in memory.similar_skills(task_type, species or "", limit=3):
        if (cand.get("success_rate") or 0) >= min_success:
            return {**cand, "warm_start": "transfer"}
    return None


def synthesize_skill_card(policy, robot_class: str, task_type: str, *, species: str | None = None,
                          gene_id: str | None = None, params_path: str | None = None,
                          trainer: str = "cem_reach") -> dict:
    """Turn a freshly ``TrainedPolicy`` into a SKILL_CARD dict ready for :func:`bank_skill_from_card`.

    This is what closes the skill-flywheel *write* loop inside ``autonomous_build(train=True)``: the
    in-process trainer produces a ``TrainedPolicy`` object (not the trainer-CLI's ``SKILL_CARD`` line),
    so we mint the same card shape from it. The card is honest about what was actually trained:

      * ``reward_spec`` is empty — the tabletop reach trainer has no process-aware reward recipe;
      * ``region`` is ``None`` — the body<->task reachable region is not computed by this trainer;
      * ``params_path`` is kept only when ``trainer`` can warm-start from its own weights
        (:data:`_PARAM_REUSE_TRAINERS`). The CEM reach trainer re-initialises each run, so its
        params are recorded under ``artifact_path`` for provenance and the skill warm-starts in
        'config' mode — never claiming a param reuse the trainer can't perform.
    """
    ev = getattr(policy, "evaluation", None)
    success_rate = float(getattr(ev, "success_rate", 0.0) or 0.0)
    obs_dim = len(policy.observation_space) or len(policy.input_features)
    act_dim = len(policy.action_dimensions)
    base_config = {
        "policy_type": getattr(policy, "policy_type", ""),
        "control_frequency_hz": getattr(policy, "control_frequency_hz", 0.0),
        "low_level_control_frequency_hz": getattr(policy, "low_level_control_frequency_hz", 0.0),
        "safety_clamps": dict(getattr(policy, "safety_clamps", {}) or {}),
        "trainer": trainer,
    }
    consumes_params = trainer in _PARAM_REUSE_TRAINERS
    return {
        "skill_id": skill_id_for(task_type, species, robot_class),
        "task_type": task_type,
        "robot_class": robot_class,
        "species": species,
        "gene_id": gene_id,
        "success_rate": success_rate,
        "obs_dim": obs_dim or None,
        "act_dim": act_dim or None,
        "reward_spec": {},
        "region": None,
        "base_config": base_config,
        "params_path": params_path if consumes_params else None,
        "artifact_path": params_path,
        "trainer": trainer,
    }


def append_manifest(memory_dir, row: dict) -> Path:
    """Append one row to ``<memory_dir>/flywheel_manifest.jsonl`` (the human-auditable skill ledger).

    One JSON line per banked skill — ``(skill_id, success_rate, n_episodes, ckpt_hash, cycle,
    warm_started, ...)`` — so a person (or a later flywheel-runner) can read the compounding-memory
    trail without opening the SQLite DB. Best-effort callers should still wrap this; it only fails
    if the path is unwritable. Returns the manifest path.
    """
    manifest = Path(memory_dir) / "flywheel_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return manifest
