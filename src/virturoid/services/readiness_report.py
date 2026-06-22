from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from virturoid.schemas.readiness import MvpReadinessReport, ReadinessGate

READINESS_REPORT_URI = "reports/mvp_readiness_report.json"

MVP_GOAL = (
    "Autonomously build a robot package from a prompt, generate task scenes to train it, "
    "compile simulator scenes, and export a learned controller when training is requested."
)


def write_mvp_readiness_report(
    package_dir: Path,
    *,
    summary: dict[str, Any],
    artifacts: dict[str, str],
    training: dict | None,
    requested_training: bool,
) -> MvpReadinessReport:
    report = build_mvp_readiness_report(
        package_dir,
        summary=summary,
        artifacts=artifacts,
        training=training,
        requested_training=requested_training,
    )
    validation = report.validate()
    if not validation.ok and report.ready:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{report.id} was marked ready but failed validation: {issues}")

    path = Path(package_dir) / READINESS_REPORT_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    # Co-locate the HONEST binding export gate so this soft scorecard can't read as truth on its own: ready/score
    # are contract/compile/eval gates (kept for back-compat), but `safe_to_export` comes from the Product Readiness
    # Ledger, which is False over placeholder CAD or unrun physics. A reviewer reading this JSON sees both.
    out = report.to_dict()
    try:
        from virturoid.services.readiness_ledger import build_product_readiness_ledger
        led = build_product_readiness_ledger(package_dir, robot_class=str(summary.get("selected_robot_class", "")))
        out["safe_to_export"] = bool(led.safe_to_export)
        out["honest_ledger"] = "reports/product_readiness_ledger.json"
        out["note"] = ("legacy readiness scorecard; the binding export gate is `safe_to_export` "
                       "(from the Product Readiness Ledger), never `ready`/`score` alone.")
    except Exception:  # noqa: BLE001 - the ledger is additive truth, never a write blocker
        pass
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return report


def build_mvp_readiness_report(
    package_dir: Path,
    *,
    summary: dict[str, Any],
    artifacts: dict[str, str],
    training: dict | None,
    requested_training: bool,
) -> MvpReadinessReport:
    package_dir = Path(package_dir)
    robot_class = str(summary.get("selected_robot_class", ""))
    package_type = str(summary.get("package_type", ""))

    contract = _read_json(package_dir / "reports" / "robot_package_contract.json")
    scene_set = _read_json(package_dir / "simulation" / "scene_set.json")
    compiled_index = _read_json(package_dir / "simulation" / "mujoco" / "compiled_scene_index.json")
    simulator_contract = _read_json(package_dir / "simulation" / "simulator_contract.json")
    perception_config = _read_json(package_dir / "simulation" / "perception_config.json")
    world_model = _read_json(package_dir / "simulation" / "world_model_contract.json")
    synthetic_manifest = _read_json(package_dir / "datasets" / "synthetic_observation_manifest.json")
    synthetic_index = _read_json(package_dir / "datasets" / "synthetic_observations" / "index.json")
    world_state_index = _read_json(package_dir / "simulation" / "world_state_index.json")

    scenes = scene_set.get("scenes", []) if isinstance(scene_set, dict) else []
    compiled_scenes = compiled_index.get("scenes", []) if isinstance(compiled_index, dict) else []
    controller_required = requested_training or bool(training and training.get("ran"))
    manipulator_route = robot_class == "manipulator"

    gates = [
        _gate(
            "generic_robot_route",
            "Generic robot-class route selected",
            bool(robot_class and summary.get("selected_morphology_template_id")),
            f"{robot_class or 'unknown'} via {summary.get('selected_morphology_template_id', 'unknown')}",
            "reports/autonomous_build_summary.json",
        ),
        _gate(
            "package_contract_ok",
            "Shared package contract passes",
            bool(contract.get("ok")),
            f"{len(contract.get('artifact_checks', []))} artifact checks recorded.",
            "reports/robot_package_contract.json",
        ),
        _gate(
            "package_valid",
            "Route-specific package validation passes",
            bool(summary.get("package_valid")),
            "Package writer and contract agree the package is valid.",
            "reports/autonomous_build_summary.json",
        ),
        _gate(
            "robot_generated",
            "Robot genome generated",
            (package_dir / "robot" / "robot_genome.json").exists(),
            "Robot Genome is present for downstream CAD/sim/control tooling.",
            "robot/robot_genome.json",
        ),
        _gate(
            "robot_model_exported",
            "Robot model exported",
            _any_artifact_exists(package_dir, artifacts, ["robot_urdf", "mujoco_xml"])
            or (package_dir / "robot" / "robot.urdf").exists(),
            "At least one simulator/robot model export exists.",
            artifacts.get("robot_urdf") or artifacts.get("mujoco_xml") or "robot/robot.urdf",
        ),
        _gate(
            "scenes_generated",
            "Task scenes generated",
            len(scenes) > 0,
            f"{len(scenes)} scene(s) in the scene set.",
            "simulation/scene_set.json",
        ),
        _gate(
            "compiled_simulation_scenes",
            "Simulator scenes compiled",
            len(compiled_scenes) > 0 and _compiled_scene_files_exist(package_dir, compiled_scenes),
            f"{len(compiled_scenes)} compiled scene(s) in the MuJoCo index.",
            "simulation/mujoco/compiled_scene_index.json",
        ),
        _gate(
            "generic_package_boundary",
            "Package boundary is not arm-only",
            package_type in {"mvp_manipulator_training_package", "mobile_base_foundation_package"},
            f"Current package type: {package_type or 'unknown'}.",
            "reports/robot_package_contract.json",
        ),
        _gate(
            "cv_perception_contract",
            "CV perception contract ready",
            bool(perception_config.get("vision_annotations") and perception_config.get("synthetic_dataset")),
            f"{len(perception_config.get('vision_annotations', []))} annotation spec(s) declared.",
            "simulation/perception_config.json",
        ),
        _gate(
            "world_model_contract",
            "World model contract ready",
            bool(world_model.get("observable_entities") and world_model.get("physical_parameter_targets")),
            f"{len(world_model.get('observable_entities', []))} observable entity/entities and {len(world_model.get('physical_parameter_targets', []))} physical parameter target(s).",
            "simulation/world_model_contract.json",
        ),
        _gate(
            "synthetic_observation_dataset",
            "Synthetic observation dataset manifest ready",
            bool(synthetic_manifest.get("vision_annotations") and synthetic_manifest.get("scene_sets")),
            f"{len(synthetic_manifest.get('scene_sets', []))} scene set(s) bound for CV data generation.",
            "datasets/synthetic_observation_manifest.json",
        ),
        _gate(
            "generated_cv_annotations",
            "Synthetic CV annotations generated",
            bool(synthetic_index.get("frames")) and _annotation_files_exist(package_dir, synthetic_index.get("frames", [])),
            f"{synthetic_index.get('frame_count', 0)} generated observation frame(s).",
            "datasets/synthetic_observations/index.json",
        ),
        _gate(
            "generated_world_state_snapshots",
            "World-state snapshots generated",
            bool(world_state_index.get("scene_states")) and _world_state_files_exist(package_dir, world_state_index.get("scene_states", [])),
            f"{world_state_index.get('scene_state_count', 0)} scene state snapshot(s).",
            "simulation/world_state_index.json",
        ),
    ]

    gates.append(
        _gate(
            "training_config_present",
            "Training run config present",
            (package_dir / "training" / "training_run_config.json").exists(),
            "Manipulator route exposes a simulator training contract." if manipulator_route else "Not required for this foundation route yet.",
            "training/training_run_config.json",
            required=manipulator_route,
            skipped=not manipulator_route,
        )
    )
    gates.append(
        _gate(
            "simulator_contract_ready",
            "Simulator handoff contract ready",
            bool(simulator_contract and simulator_contract.get("scene_contracts")),
            f"{len(simulator_contract.get('scene_contracts', [])) if simulator_contract else 0} simulator scene contract(s).",
            "simulation/simulator_contract.json",
            required=manipulator_route,
            skipped=not manipulator_route,
        )
    )
    gates.append(
        _gate(
            "controller_exported",
            "Learned controller exported",
            _controller_bundle_exists(package_dir),
            _controller_detail(training, requested_training),
            "software/controller/controller_bundle.json",
            required=controller_required,
            skipped=not controller_required,
        )
    )

    required_gates = [gate for gate in gates if gate.required]
    passed_required = [gate for gate in required_gates if gate.status == "pass"]
    ready = len(required_gates) == len(passed_required)
    score = round((len(passed_required) / len(required_gates)) * 100) if required_gates else 0

    report = MvpReadinessReport(
        id=f"mvp_readiness_{robot_class or 'unknown'}",
        package_dir=str(package_dir),
        robot_class=robot_class,
        package_type=package_type,
        goal=MVP_GOAL,
        ready=ready,
        score=score,
        gates=gates,
        next_steps=_next_steps(gates),
    )
    return report


def _gate(
    key: str,
    label: str,
    passes: bool,
    detail: str,
    evidence_uri: str | None,
    *,
    required: bool = True,
    skipped: bool = False,
) -> ReadinessGate:
    status = "skip" if skipped else "pass" if passes else "fail"
    return ReadinessGate(
        key=key,
        label=label,
        status=status,
        required=required,
        detail=detail,
        evidence_uri=evidence_uri,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _any_artifact_exists(package_dir: Path, artifacts: dict[str, str], keys: list[str]) -> bool:
    return any((package_dir / artifacts[key]).exists() for key in keys if key in artifacts)


def _compiled_scene_files_exist(package_dir: Path, compiled_scenes: list[dict[str, Any]]) -> bool:
    for scene in compiled_scenes:
        uri = scene.get("mujoco_xml")
        if not uri or not (package_dir / uri).exists():
            return False
    return True


def _annotation_files_exist(package_dir: Path, frames: list[dict[str, Any]]) -> bool:
    return bool(frames) and all((package_dir / str(frame.get("annotation_uri", ""))).exists() for frame in frames)


def _world_state_files_exist(package_dir: Path, records: list[dict[str, Any]]) -> bool:
    return bool(records) and all((package_dir / str(record.get("world_state_uri", ""))).exists() for record in records)


def _controller_bundle_exists(package_dir: Path) -> bool:
    return (
        (package_dir / "software" / "controller" / "controller_bundle.json").exists()
        and (package_dir / "software" / "controller" / "controller.py").exists()
    )


def _controller_detail(training: dict | None, requested_training: bool) -> str:
    if training and training.get("ran"):
        success_rate = training.get("eval_success_rate", "unknown")
        return f"Training ran and exported a controller bundle; eval success rate: {success_rate}."
    if requested_training and training and not training.get("ran"):
        return str(training.get("reason", "Training was requested but did not run."))
    if requested_training:
        return "Training was requested, but no training status was recorded."
    return "Controller export is optional until the build is run with --train."


def _next_steps(gates: list[ReadinessGate]) -> list[str]:
    steps = []
    for gate in gates:
        if gate.required and gate.status != "pass":
            steps.append(f"Fix readiness gate `{gate.key}`: {gate.detail}")
    if not steps:
        steps.append("Use this package as the MVP baseline and start expanding robot-class writers beyond the manipulator.")
    return steps
