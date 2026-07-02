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


def _round_floats(obj, ndigits: int = 2):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _distinct_key(example: dict) -> tuple:
    """A (prompt-family, normalized-design) fingerprint for de-duplication: the prompt's first few words + the
    assistant design JSON with floats rounded, so near-identical designs collapse to one."""
    msgs = example.get("messages", [])
    prompt = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    design = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    try:
        norm = json.dumps(_round_floats(json.loads(design)), sort_keys=True)
    except Exception:  # noqa: BLE001 - a non-JSON design de-dups on its raw string
        norm = str(design)
    return (" ".join(str(prompt).lower().split()[:4]), norm)


def dedup_distinct(examples: list[dict]) -> list[dict]:
    """Keep only DISTINCT (prompt-family, design) examples (plan v3 WS12). Rejection-sampling SFT gains are driven
    by the count of DISTINCT verified solutions, NOT raw volume — after de-dup, 100 samples yield only ~5 distinct
    paths/problem (Yuan et al. 2023, arXiv:2308.01825); LIMA (2305.11206) shows quantity-without-diversity
    plateaus. So the flywheel's banked designs must be de-duplicated before they count toward distillation."""
    seen: set = set()
    out: list[dict] = []
    for ex in examples:
        k = _distinct_key(ex)
        if k in seen:
            continue
        seen.add(k)
        out.append(ex)
    return out


def distillation_readiness(examples: list[dict], *, target_distinct: int = 1000) -> dict:
    """Is the banked corpus ready to self-distill a small proposer? Reports DISTINCT-design count + prompt-family
    coverage and gates on the evidence anchor (~1000 distinct: LIMA 2305.11206 / ReST^EM 2312.06585). Below that,
    DIVERSITY dominates raw count and a run may only shift schema/format adherence (RFT 2308.01825) — so the
    honest verdict is 'collect more distinct designs', not 'train now'. The SFT run itself is external GPU."""
    distinct = dedup_distinct(examples)
    fams = {_distinct_key(ex)[0] for ex in distinct}
    n = len(distinct)
    ready = n >= int(target_distinct)
    if not examples:
        verdict = "no verified designs banked yet — the flywheel must run to produce rejection-sampled successes"
    elif ready:
        verdict = (f"{n} distinct designs across {len(fams)} prompt-families >= {target_distinct} — ready to "
                   "rejection-sampling-SFT an ~8B base for 1-2 rounds (ReST^EM: gains saturate after 1-2)")
    else:
        verdict = (f"{n} distinct designs across {len(fams)} prompt-families (< {target_distinct}); collect more — "
                   "diversity dominates raw count; a run now may only shift schema adherence, not capability")
    return {"n_examples": len(examples), "n_distinct": n, "n_prompt_families": len(fams),
            "target_distinct": int(target_distinct), "ready": ready, "verdict": verdict}


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
