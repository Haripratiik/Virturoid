"""Generic prompt-to-robot build orchestration.

This module is intentionally thin: it chooses the morphology and then delegates
to the robot-class package writer. The arm MVP remains the most complete package
today, but the product entrypoint is now robot-class aware instead of arm-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.services.morphology_selector import select_morphology_template
from virturoid.services.package_contract_builder import PACKAGE_CONTRACT_URI, write_robot_package_contract
from virturoid.services.package_writer_registry import write_package_for_template
from virturoid.services.readiness_report import READINESS_REPORT_URI, write_mvp_readiness_report
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.task_builder import build_task_graph
from virturoid.services.workbench_ui import WORKBENCH_UI_URI, write_workbench_ui


@dataclass
class AutonomousBuildResult:
    output_dir: str
    prompt: str
    requirements_id: str
    task_type: str
    selected_morphology_template_id: str
    selected_robot_class: str
    selected_species: str
    package_type: str
    package_valid: bool | None
    summary_uri: str
    artifacts: dict[str, str]
    training: dict | None
    notes: list[str]
    readiness: dict | None = None
    evaluation: dict | None = None
    compute: dict | None = None
    # Species-tree honesty: what the request mapped to vs. what was actually buildable.
    # When these differ from selected_*, the build was done as the nearest buildable
    # relative and the system says so (never a silent mislabel).
    requested_robot_class: str = ""
    requested_species: str = ""
    species_exact: bool = True
    species_note: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


def build_robot_package_from_prompt(
    prompt: str,
    output_dir: Path,
    payload_kg: float | None = None,
    reach_m: float | None = None,
    sensor: str | None = None,
    train: bool = False,
    evaluate: bool = False,
    co_optimize: bool = False,
    perceive: bool = False,
) -> AutonomousBuildResult:
    """Build the best currently implemented robot package for a prompt."""
    requirements = build_requirements_from_prompt(prompt, payload_kg=payload_kg, reach_m=reach_m, sensor=sensor)
    task = build_task_graph(requirements)
    templates = morphology_template_catalog()
    selection = select_morphology_template(requirements, task, templates)
    output_dir = Path(output_dir)

    # The selector already traced the species tree to the nearest *buildable* species
    # (selection.selected_template is always buildable), keeping the requested species for
    # honest reporting. So a humanoid request builds its nearest buildable relative instead
    # of crashing, and we still say what was actually requested.
    build_template = selection.selected_template

    package = write_package_for_template(
        requirements=requirements,
        task=task,
        template=build_template,
        output_dir=output_dir,
        train=train,
    )

    contract = write_robot_package_contract(
        package.written_dir,
        package_type=package.package_type,
        robot_class=build_template.robot_class,
        species=build_template.species_pattern,
        morphology_template_id=build_template.id,
        task_type=task.task_type,
        artifacts=package.artifacts,
        training=package.training,
    )
    artifacts = {
        **package.artifacts,
        "robot_package_contract": PACKAGE_CONTRACT_URI,
        "mvp_readiness_report": READINESS_REPORT_URI,
        "workbench": WORKBENCH_UI_URI,
    }
    package_valid = bool(package.package_valid and contract.ok)

    evaluation = _maybe_evaluate(package.written_dir, perceive=perceive) if evaluate else None
    if evaluation and evaluation.get("ran"):
        artifacts["physics_evaluation_report"] = "reports/physics_evaluation_report.json"

    if co_optimize:
        codesign = _maybe_co_optimize(package.written_dir, requirements.prompt, build_template.robot_class)
        if codesign and codesign.get("ran"):
            artifacts["codesign_optimization_report"] = "reports/codesign_optimization_report.json"
        if evaluation is None:
            evaluation = {}
        evaluation["hardware_codesign"] = codesign

    result = AutonomousBuildResult(
        output_dir=str(package.written_dir),
        prompt=requirements.prompt,
        requirements_id=requirements.id,
        task_type=task.task_type,
        selected_morphology_template_id=build_template.id,
        selected_robot_class=build_template.robot_class,
        selected_species=build_template.species_pattern,
        package_type=package.package_type,
        package_valid=package_valid,
        summary_uri="reports/autonomous_build_summary.json",
        artifacts=artifacts,
        training=package.training,
        notes=package.notes + ([selection.species_note] if not selection.species_exact else []),
        requested_robot_class=selection.requested_robot_class,
        requested_species=selection.requested_species,
        species_exact=selection.species_exact,
        species_note=selection.species_note,
        evaluation=evaluation,
        compute={
            "physics_executed": bool(evaluate or co_optimize),
            "training_executed": bool(train),
            "note": (
                "Real MuJoCo physics ran for the requested evaluate/co-optimize/train steps."
                if (evaluate or co_optimize or train)
                else "Fast scaffold only: NO physics/optimization/training ran. "
                "Use --evaluate / --co-optimize / --train, or `python -m virturoid.autobuild`, for the real loop."
            ),
        },
    )
    readiness = write_mvp_readiness_report(
        package.written_dir,
        summary=result.to_dict(),
        artifacts=artifacts,
        training=package.training,
        requested_training=train,
    )
    result.readiness = {
        "uri": READINESS_REPORT_URI,
        "ready": readiness.ready,
        "score": readiness.score,
        "failed_required_gates": [
            gate.key for gate in readiness.gates if gate.required and gate.status != "pass"
        ],
    }
    write_workbench_ui(package.written_dir, summary=result.to_dict(), artifacts=artifacts, training=package.training)
    _write_summary(package.written_dir, result)
    return result


def _maybe_evaluate(written_dir: Path, perceive: bool = False) -> dict:
    """Run a real pick-and-place evaluation when the package supports it."""
    from virturoid.services.mujoco_runner import mujoco_available

    compiled = written_dir / "simulation" / "mujoco" / "compiled_scene_index.json"
    genome = written_dir / "robot" / "robot_genome.json"
    if not (compiled.exists() and genome.exists()):
        return {"ran": False, "reason": "Package has no compiled scenes/genome to evaluate."}
    if not mujoco_available():
        return {"ran": False, "reason": "MuJoCo not installed; run `pip install mujoco` to evaluate."}
    from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

    perception = None
    if perceive:
        from virturoid.services.perception_adapter import default_perception

        perception = default_perception(seed=7)
    report = run_physics_pick_place_evaluation(written_dir, perception=perception)
    return {
        "ran": True,
        "perception": getattr(perception, "name", "privileged_sim_pose"),
        "controller": report.controller,
        "total_episodes": report.total_episodes,
        "success_rate": report.success_rate,
        "blocks_placed": report.blocks_placed,
        "blocks_total": report.blocks_total,
        "failure_clusters": [
            {"label": c.failure_label, "count": c.count, "suggested_regression": c.suggested_regression}
            for c in report.failure_clusters
        ],
    }


def _maybe_co_optimize(written_dir: Path, prompt: str, robot_class: str) -> dict:
    """Co-optimize hardware against real task success (manipulator packages only)."""
    from virturoid.services.mujoco_runner import mujoco_available

    if robot_class != "manipulator":
        return {"ran": False, "reason": f"Hardware co-design not implemented for robot class '{robot_class}'."}
    if not mujoco_available():
        return {"ran": False, "reason": "MuJoCo not installed; run `pip install mujoco` to co-optimize."}
    from virturoid.services.codesign_optimizer import co_optimize_hardware, write_codesign_report

    report = co_optimize_hardware(prompt=prompt)
    write_codesign_report(report, written_dir)
    return {
        "ran": True,
        "baseline_success_rate": report.baseline.success_rate,
        "best_success_rate": report.best.success_rate,
        "success_improvement": report.success_improvement,
        "candidates_evaluated": report.candidates_evaluated,
        "changed_parameters": report.changed_parameters,
    }


def _write_summary(output_dir: Path, result: AutonomousBuildResult) -> Path:
    path = output_dir / result.summary_uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
