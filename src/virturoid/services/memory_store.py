"""Minimal robot-build memory.

The plan treats memory as a core product feature: remember what design worked
for a task so future builds reuse it. This is a small file-backed store keyed by
(robot_class, task_type) that lets the autonomous builder warm-start from a prior
converged design instead of searching from scratch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from virturoid.services.memory_db import default_memory_dir

#: One rule for where memory lives, owned by ``memory_db`` -- see the reasoning there.
#: Duplicating the ``build/memory`` literal here is how the two would drift apart, and
#: this module's JSON records sit in the same directory as that module's sqlite file.
DEFAULT_MEMORY_DIR = default_memory_dir()


def _key(robot_class: str, task_type: str) -> str:
    return f"{robot_class}__{task_type}".replace("/", "_")


def save_build_record(
    robot_class: str,
    task_type: str,
    converged_design: dict,
    success_rate: float,
    prompt: str,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
) -> Path:
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{_key(robot_class, task_type)}.json"
    existing = _read(path)
    # Keep the best-performing design seen for this task.
    if existing and existing.get("success_rate", -1) >= success_rate:
        return path
    record = {
        "robot_class": robot_class,
        "task_type": task_type,
        "prompt": prompt,
        "converged_design": converged_design,
        "success_rate": success_rate,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def append_design_record(
    prompt: str,
    robot_class: str,
    task_type: str,
    design: dict,
    success_rate: float,
    source: str,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
) -> Path:
    """Append one (prompt -> physics-optimized design -> success) row to the design dataset.

    This is the data flywheel from docs/ai_layer_plan.md §6: every real build adds a
    supervised training row that can later distill the local model (LoRA) so it proposes
    near-optimal designs directly. ``source`` records what produced the design
    (e.g. "physics_codesign", "ai_designer+codesign").
    """
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / "design_dataset.jsonl"
    row = {
        "prompt": prompt,
        "robot_class": robot_class,
        "task_type": task_type,
        "design": design,
        "success_rate": success_rate,
        "source": source,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def find_similar_design(robot_class: str, task_type: str, memory_dir: Path = DEFAULT_MEMORY_DIR) -> dict | None:
    """Return a prior converged design for the same robot class and task, if any."""
    path = Path(memory_dir) / f"{_key(robot_class, task_type)}.json"
    record = _read(path)
    if record and record.get("converged_design"):
        return record
    # Transfer (plan §8.6): fall back to any prior design of the same robot class.
    return _best_for_class(robot_class, memory_dir)


def _best_for_class(robot_class: str, memory_dir: Path) -> dict | None:
    """Best-performing prior design for a robot class across tasks (cross-task transfer)."""
    memory_dir = Path(memory_dir)
    if not memory_dir.exists():
        return None
    best = None
    for path in memory_dir.glob(f"{robot_class}__*.json"):
        record = _read(path)
        if record and record.get("converged_design"):
            if best is None or record.get("success_rate", -1) > best.get("success_rate", -1):
                best = {**record, "transferred": True}
    return best


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
