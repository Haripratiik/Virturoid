"""Proposer distillation (plan Phase E): turn the flywheel into local-model training data.

Per docs/ai_architecture_plan.md, the local LLM "proposer" is distilled LAST — from the
accumulated (prompt -> physics-verified gene/design -> success) rows the platform logs. This
module builds that supervised dataset: each high-success build becomes a chat example whose
target is the structured gene the physics loop verified. Fine-tuning a small local model
(e.g. Nemotron) on it lets the platform propose good genes cheaply at scale.

This module produces the **dataset** (real, here) and documents the **training recipe**. The
actual LoRA/SFT run needs a GPU and is external (see ``build_sft_dataset`` docstring) — we do
not fake a trained model. Gate distillation on volume: only worth it once enough verified rows
exist and the task distribution has stabilized.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.gene import RobotGene

_SYSTEM = (
    "You are Virturoid's robot proposer. Given a task, output ONLY a JSON robot gene: a "
    "kinematic tree of segments (name, parent, shape, length_m, joint_type, joint_axis, "
    "actuator_torque_nm, is_end_effector), a base_mount, and an end_effector_type. The gene "
    "must be a valid tree (one root, no cycles) that a physics simulator can build."
)


def gene_to_target(gene: RobotGene) -> dict:
    """Compact, schema-faithful JSON target the model is trained to emit."""
    return {
        "species": gene.species,
        "robot_class": gene.robot_class,
        "base_mount": gene.base_mount,
        "end_effector_type": gene.end_effector_type,
        "segments": [
            {k: v for k, v in {
                "name": s.name, "parent": s.parent, "shape": s.shape,
                "length_m": s.length_m, "radius_m": s.radius_m, "mass_kg": s.mass_kg,
                "joint_type": s.joint_type,
                "joint_axis": list(s.joint_axis) if s.joint_type in ("revolute", "prismatic") else None,
                "actuator_torque_nm": s.actuator_torque_nm,
                "is_end_effector": s.is_end_effector or None,
            }.items() if v is not None}
            for s in gene.segments
        ],
    }


def sft_example(prompt: str, gene: RobotGene) -> dict:
    """One supervised chat example: (system, user prompt) -> verified gene JSON."""
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(gene_to_target(gene), separators=(",", ":"))},
        ]
    }


def build_sft_dataset(pairs: list[tuple[str, RobotGene]], *, min_success: float | None = None,
                      successes: list[float] | None = None) -> list[dict]:
    """Build SFT examples from (prompt, gene) pairs, optionally filtered by success.

    To actually distill a local model from this: write the returned examples to JSONL, then
    LoRA-SFT a small base (e.g. Nemotron) with trl ``SFTTrainer`` on a GPU, merge, and serve
    (vLLM or GGUF→Ollama). That run is external compute — this function only builds the data.
    """
    examples = []
    for i, (prompt, gene) in enumerate(pairs):
        if min_success is not None and successes is not None and successes[i] < min_success:
            continue
        examples.append(sft_example(prompt, gene))
    return examples


def export_sft_jsonl(examples: list[dict], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return len(examples)


def dataset_from_memory(memory_dir: Path, *, min_success: float = 0.8) -> list[dict]:
    """Build SFT examples from the memory flywheel's design rows (high-success only).

    Uses the verified design dataset; for full gene targets the build must have logged the
    gene (gene-built runs). Returns whatever verified rows exist — grows with usage.
    """
    from virturoid.services.memory_db import MemoryDB

    db_path = Path(memory_dir) / "virturoid_memory.db"
    if not db_path.exists():
        return []
    examples = []
    with MemoryDB(db_path) as db:
        for row in db.training_dataset(min_success=min_success):
            # design rows store a design dict; emit a prompt->design SFT example (gene targets
            # accrue as more gene-built runs are logged).
            examples.append({"messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": json.dumps(row["design"], separators=(",", ":"))},
            ]})
    return examples
