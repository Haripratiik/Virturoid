"""Train and accept a small contact-grasp skill on the exported MVP manipulator.

This is deliberately modest: it tunes a few physically meaningful controller parameters for the same exported
MJCF model used by the readiness ledger, then evaluates the best setting on held-out scenes. It is not a full
neural grasp/place policy yet, but it moves training beyond reach-only by optimizing a real contact skill.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.services.mujoco_runner import mujoco_available
from virturoid.services.physics_evaluator import run_contact_grasp_evaluation_from_export


def train_contact_grasp_skill_from_export(
    package_dir: Path,
    *,
    train_scene_uris: list[str] | None = None,
    eval_scene_uris: list[str] | None = None,
    max_attempts: int = 4,
    min_success_rate: float = 0.5,
    memory_dir: Path | None = None,
    robot_class: str = "manipulator",
    species: str | None = None,
) -> dict:
    """Tune contact-grasp controller parameters and write skill/acceptance artifacts."""
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to train contact grasping.")

    package_dir = Path(package_dir)
    memory = _load_memory_skill(memory_dir, robot_class, species)
    train_scene_uris = train_scene_uris or ["simulation/scene_set.json"]
    eval_scene_uris = eval_scene_uris or ["simulation/baseline_scene_set.json", "simulation/holdout_scene_set.json"]
    baseline_params = {
        "arm_kp": 70.0,
        "arm_kd": 9.0,
        "finger_kp": 220.0,
        "finger_kd": 12.0,
        "finger_target": 0.012,
        "lift_offset_m": 0.16,
    }
    candidates = [
        baseline_params,
        {**baseline_params, "finger_target": 0.024, "finger_kp": 320.0},
        {**baseline_params, "finger_target": 0.032, "finger_kp": 420.0, "lift_offset_m": 0.18},
        {**baseline_params, "arm_kp": 90.0, "arm_kd": 10.0, "finger_target": 0.038, "finger_kp": 500.0, "finger_kd": 18.0, "lift_offset_m": 0.20},
    ]
    if memory and memory.get("controller_params"):
        candidates.insert(0, dict(memory["controller_params"]))

    baseline = run_contact_grasp_evaluation_from_export(
        package_dir, scene_uris=train_scene_uris, max_attempts=max_attempts,
        controller_params=baseline_params, write_report=False)
    scored = []
    for index, params in enumerate(candidates):
        train_eval = run_contact_grasp_evaluation_from_export(
            package_dir, scene_uris=train_scene_uris, max_attempts=max_attempts,
            controller_params=params, write_report=False)
        scored.append({
            "candidate": index,
            "params": params,
            "success_rate": train_eval["success_rate"],
            "mean_lift_m": train_eval["mean_lift_m"],
            "score": float(train_eval["success_rate"]) + 0.1 * float(train_eval["mean_lift_m"]),
        })
    best = max(scored, key=lambda item: item["score"])
    final_eval = run_contact_grasp_evaluation_from_export(
        package_dir, scene_uris=eval_scene_uris, max_attempts=max_attempts,
        controller_params=best["params"], write_report=True)
    accepted = bool(
        final_eval["success_rate"] >= min_success_rate
        and final_eval["mean_lift_m"] > 0.03
        and final_eval["total_attempts"] > 0
    )

    skill_dir = package_dir / "software" / "skills"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "contact_grasp_skill.json"
    skill = {
        "id": f"contact_grasp_skill_{final_eval['robot_genome_id']}",
        "robot_genome_id": final_eval["robot_genome_id"],
        "skill_type": "contact_grasp_lift",
        "controller_params": best["params"],
        "training_method": "deterministic_parameter_search",
        "accepted": accepted,
        "evaluation": {
            "success_rate": final_eval["success_rate"],
            "mean_lift_m": final_eval["mean_lift_m"],
            "max_lift_m": final_eval["max_lift_m"],
            "total_attempts": final_eval["total_attempts"],
        },
    }
    skill_path.write_text(json.dumps(skill, indent=2), encoding="utf-8")
    banked_skill_id = _bank_memory_skill(
        memory_dir,
        robot_class=robot_class,
        species=species,
        skill=skill,
        success_rate=float(final_eval["success_rate"]),
        params_path=str(skill_path),
    )

    report = {
        "accepted": accepted,
        "skill_uri": "software/skills/contact_grasp_skill.json",
        "method": "deterministic_parameter_search",
        "memory_reused": bool(memory),
        "memory_skill_id": (memory or {}).get("skill_id") or banked_skill_id,
        "memory_warm_start": (memory or {}).get("warm_start"),
        "thresholds": {"min_success_rate": min_success_rate, "min_mean_lift_m": 0.03},
        "baseline": {
            "params": baseline_params,
            "success_rate": baseline["success_rate"],
            "mean_lift_m": baseline["mean_lift_m"],
        },
        "best_training_candidate": best,
        "controller_params": best["params"],
        "evaluation": {
            "success_rate": final_eval["success_rate"],
            "mean_lift_m": final_eval["mean_lift_m"],
            "max_lift_m": final_eval["max_lift_m"],
            "total_attempts": final_eval["total_attempts"],
        },
        "improved_over_baseline": (
            final_eval["success_rate"] > baseline["success_rate"]
            or final_eval["mean_lift_m"] > baseline["mean_lift_m"]
        ),
        "banked_skill_id": banked_skill_id,
        "notes": [
            "This trains contact-grasp controller parameters against real MuJoCo contact lifts on the exported model.",
            "It complements the reach policy; it is not yet a learned full pick/place sequence policy.",
        ],
    }
    out = package_dir / "reports" / "manipulation_training_acceptance_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_memory_skill(memory_dir: Path | None, robot_class: str, species: str | None) -> dict | None:
    if memory_dir is None:
        return None
    try:
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.skill_flywheel import warm_start_for

        with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
            recalled = warm_start_for(db, robot_class, "contact_grasp_lift", species=species, min_success=0.5)
        if not recalled:
            return None
        base = recalled.get("base_config") or {}
        params = base.get("controller_params") or {}
        if not params and recalled.get("params_path") and Path(recalled["params_path"]).exists():
            try:
                params = json.loads(Path(recalled["params_path"]).read_text(encoding="utf-8")).get("controller_params", {})
            except Exception:  # noqa: BLE001
                params = {}
        if not params:
            return None
        return {**recalled, "controller_params": params}
    except Exception:  # noqa: BLE001 - memory is advisory; training still works cold
        return None


def _bank_memory_skill(
    memory_dir: Path | None,
    *,
    robot_class: str,
    species: str | None,
    skill: dict,
    success_rate: float,
    params_path: str,
) -> str | None:
    if memory_dir is None:
        return None
    try:
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.skill_flywheel import skill_id_for

        sid = skill_id_for("contact_grasp_lift", species, robot_class)
        with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
            db.record_skill(
                sid,
                robot_class,
                "contact_grasp_lift",
                species=species,
                gene_id=skill.get("robot_genome_id"),
                success_rate=success_rate,
                params_path=params_path,
                base_config={
                    "trainer": "deterministic_parameter_search",
                    "controller_params": skill.get("controller_params") or {},
                    "evaluation": skill.get("evaluation") or {},
                },
                notes="MVP contact grasp skill: tuned slide-joint gripper gains and lift target.",
            )
        return sid
    except Exception:  # noqa: BLE001 - memory is advisory; artifact training still succeeded
        return None
