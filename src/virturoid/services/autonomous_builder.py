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

PRODUCT_READINESS_LEDGER_URI = "reports/product_readiness_ledger.json"


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

    # If the morphology-template path has no exact builder for the requested species (e.g. a quadruped),
    # build the REAL morphology with the general engine instead of silently falling back to the arm. The
    # selector already traced the request honestly (species_exact is False); compose_robot classifies and
    # composes any body, and build_gene_package emits the scenes a learned controller replays.
    # MOBILE BASES also go through the general engine: the dedicated foundation-template writer emitted a
    # meshless URDF (so Robot mode showed only placeholders) plus an untagged scene set (task_type=None,
    # which broke episode replay). The gene navigation package is clean -- no URDF (so it auto-plays the
    # Episode like the maze_nav demo) and task_type=navigation. Only manipulators keep the mesh-bearing
    # template writer below.
    if not selection.species_exact or selection.requested_robot_class == "mobile_base":
        return _build_via_general_engine(requirements, task, selection, output_dir, train)

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
        "product_readiness_ledger": PRODUCT_READINESS_LEDGER_URI,
        "robot_package_contract": PACKAGE_CONTRACT_URI,
        "mvp_readiness_report": READINESS_REPORT_URI,
        "workbench": WORKBENCH_UI_URI,
    }
    package_valid = bool(package.package_valid and contract.ok)

    evaluation = _maybe_evaluate(package.written_dir, perceive=perceive) if evaluate else None
    if evaluation and evaluation.get("ran"):
        artifacts["physics_evaluation_report"] = "reports/physics_evaluation_report.json"
        artifacts["grasp_evaluation_report"] = "reports/grasp_evaluation_report.json"
        artifacts["manipulation_fidelity_gap_report"] = "reports/manipulation_fidelity_gap_report.json"

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
    result.readiness = _ledger_backed_readiness(
        package.written_dir, build_template.robot_class, legacy_readiness=readiness)
    write_workbench_ui(package.written_dir, summary=result.to_dict(), artifacts=artifacts, training=package.training)
    _write_summary(package.written_dir, result)
    return result


def _build_via_general_engine(requirements, task, selection, output_dir: Path, train: bool) -> AutonomousBuildResult:
    """Build the REAL requested morphology with the general engine when no morphology-template writer exists for
    it yet (quadruped, legged, etc.). compose_robot classifies and composes any body; build_gene_package emits
    the package (genome, CAD, compiled scenes) including the locomotion scenes a learned gait replays."""
    from virturoid.services.gene_build import build_gene_package
    from virturoid.services.morphology_composer import compose_robot

    gene = compose_robot(requirements.prompt, llm=None)
    summary = build_gene_package(gene, requirements.prompt, output_dir)
    written_dir = Path(output_dir)
    _try_write_robot_urdf(written_dir)  # best-effort: lets the package list + render in Robot mode
    # Save the bare-robot model so the locomotion replay runs on the SAME model the gait was trained on. The
    # compiled SCENE model reorders geoms/joints, which mismaps the per-joint gait into a lurch; the bare model
    # matches training, so the quad actually walks. Best-effort; replay falls back to the scene if absent.
    try:
        from virturoid.services.morph_policy import robot_mjcf

        (written_dir / "simulation").mkdir(parents=True, exist_ok=True)
        (written_dir / "simulation" / "robot_only.xml").write_text(robot_mjcf(gene), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    species = getattr(gene, "species", None) or selection.requested_species
    robot_class = getattr(gene, "robot_class", None) or selection.requested_robot_class
    # Report the builder HONESTLY. When the selector picked a template of the SAME class as what we built (e.g.
    # a mobile_base navigation prompt -> the mobile template), keep it. When it was TRACED to a mismatched class
    # (a quadruped is not buildable-by-template yet, so is_buildable traced it to the ARM), name the general
    # engine instead of the arm — otherwise the report contradicts its own "real morphology, not an arm fallback"
    # note. Nothing resolves this id against the catalog (stored + displayed only).
    _sel = getattr(selection, "selected_template", None)
    engine_template_id = (_sel.id if _sel is not None and getattr(_sel, "robot_class", None) == robot_class
                          else f"general_engine.{robot_class}")
    task_type = summary.get("task_type", task.task_type) if isinstance(summary, dict) else task.task_type
    has_scenes = (written_dir / "simulation" / "mujoco" / "compiled_scene_index.json").exists()

    artifacts = {
        "robot_genome": "robot/robot_genome.json",
        "scene_set": "simulation/scene_set.json",
        "compiled_scene_index": "simulation/mujoco/compiled_scene_index.json",
        "product_readiness_ledger": PRODUCT_READINESS_LEDGER_URI,
        "robot_package_contract": PACKAGE_CONTRACT_URI,
        "mvp_readiness_report": READINESS_REPORT_URI,
        "workbench": WORKBENCH_UI_URI,
    }
    if (written_dir / "robot" / "robot.urdf").exists():
        artifacts["robot_urdf"] = "robot/robot.urdf"

    # Contract + readiness + workbench are best-effort here: the template-based validator expects the ARM's
    # required artifacts, which a locomotion package legitimately lacks. The package is valid when the general
    # engine produced compiled scenes (what the episode replay needs); a contract hiccup must not fail the build.
    try:
        write_robot_package_contract(
            written_dir, package_type="general_engine_package", robot_class=robot_class, species=species,
            morphology_template_id=engine_template_id, task_type=task_type,
            artifacts=artifacts, training=None,
        )
    except Exception:  # noqa: BLE001
        pass

    result = AutonomousBuildResult(
        output_dir=str(written_dir),
        prompt=requirements.prompt,
        requirements_id=requirements.id,
        task_type=task_type,
        selected_morphology_template_id=engine_template_id,
        selected_robot_class=robot_class,
        selected_species=species,
        package_type="general_engine_package",
        package_valid=bool(has_scenes),
        summary_uri="reports/autonomous_build_summary.json",
        artifacts=artifacts,
        training=None,
        notes=[
            "No morphology-template builder exists for this species yet, so it was built with the general "
            "engine (compose_robot + build_gene_package) -- the real requested morphology, not an arm fallback.",
            f"Built {species} ({robot_class}); task {task_type}. Replay an episode to see the gait.",
        ],
        requested_robot_class=selection.requested_robot_class,
        requested_species=selection.requested_species,
        species_exact=True,
        species_note="",
        compute={"physics_executed": True, "training_executed": False,
                 "note": "Built and compiled by the general engine; replay an episode to see the gait."},
    )
    try:
        readiness = write_mvp_readiness_report(
            written_dir, summary=result.to_dict(), artifacts=artifacts, training=None, requested_training=train,
        )
        result.readiness = {
            "uri": READINESS_REPORT_URI, "ready": readiness.ready, "score": readiness.score,
            "failed_required_gates": [g.key for g in readiness.gates if g.required and g.status != "pass"],
        }
        result.readiness = _ledger_backed_readiness(
            written_dir, robot_class, gene=gene, legacy_readiness=readiness)
    except Exception:  # noqa: BLE001
        pass
    try:
        write_workbench_ui(written_dir, summary=result.to_dict(), artifacts=artifacts, training=None)
        _write_summary(written_dir, result)
    except Exception:  # noqa: BLE001
        pass
    return result


def _try_write_robot_urdf(written_dir: Path) -> None:
    """Best-effort robot/robot.urdf from the package genome so the package lists + renders in Robot mode.
    Episode replay does not need it, so any failure here is non-fatal."""
    try:
        import json as _json

        from virturoid.schemas.robot import RobotGenome
        from virturoid.services.urdf_exporter import write_robot_urdf

        genome_path = written_dir / "robot" / "robot_genome.json"
        if (written_dir / "robot" / "robot.urdf").exists() or not genome_path.exists():
            return
        genome = RobotGenome.from_dict(_json.loads(genome_path.read_text(encoding="utf-8")))
        write_robot_urdf(genome, written_dir)
    except Exception:  # noqa: BLE001 - URDF is a nicety; episode replay works without it
        pass


def _ledger_backed_readiness(package_dir: Path, robot_class: str, *, legacy_readiness=None, gene=None) -> dict:
    """Expose the Product Readiness Ledger as the public readiness verdict.

    The MVP scorecard is still useful for package-shape diagnostics, but it can pass over placeholder CAD or
    unrun physics. The ledger is the export truth: `ready` means `safe_to_export`.
    """
    legacy = None
    if legacy_readiness is not None:
        legacy = {
            "uri": READINESS_REPORT_URI,
            "ready": bool(legacy_readiness.ready),
            "score": legacy_readiness.score,
            "failed_required_gates": [
                gate.key for gate in legacy_readiness.gates if gate.required and gate.status != "pass"
            ],
        }
    try:
        from virturoid.services.readiness_ledger import write_product_readiness_ledger
        ledger = write_product_readiness_ledger(package_dir, robot_class=robot_class, gene=gene, enforce=False)
        out = {
            "uri": PRODUCT_READINESS_LEDGER_URI,
            "ready": ledger.safe_to_export,
            "safe_to_export": ledger.safe_to_export,
            "highest_attained": ledger.highest_attained,
            "failed_required_gates": ledger.validate(),
            "legacy_mvp_readiness": legacy,
        }
        return out
    except Exception as exc:  # noqa: BLE001 - do not break builds if the additive ledger probe crashes
        fallback = legacy or {"uri": READINESS_REPORT_URI, "ready": False, "score": 0, "failed_required_gates": []}
        return {**fallback, "safe_to_export": False, "ledger_error": str(exc)}


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
