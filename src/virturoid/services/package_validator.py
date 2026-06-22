from __future__ import annotations

import json
import ast
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


@dataclass
class PackageCheck:
    name: str
    status: str
    message: str


@dataclass
class PackageValidationReport:
    package_dir: str
    checks: list[PackageCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status == "pass" for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "package_dir": self.package_dir,
            "ok": self.ok,
            "checks": [check.__dict__ for check in self.checks],
        }


def validate_mvp_export_package(package_dir: Path) -> PackageValidationReport:
    report = PackageValidationReport(package_dir=str(package_dir))
    expected_files = [
        "manifest.json",
        "project.json",
        "design/morphology_template.json",
        "design/builder_dispatch.json",
        "design/robot_build_blueprint.json",
        "bom/bom.json",
        "bom/part_resolution_report.json",
        "power/power_architecture.json",
        "robot/robot_genome.json",
        "robot/robot.urdf",
        "simulation/baseline_scene_set.json",
        "simulation/scene_set.json",
        "simulation/regression_scene_set.json",
        "simulation/holdout_scene_set.json",
        "simulation/revision_scene_set.json",
        "simulation/perception_config.json",
        "simulation/world_model_contract.json",
        "simulation/simulator_contract.json",
        "datasets/synthetic_observation_manifest.json",
        "datasets/synthetic_observations/index.json",
        "simulation/world_state_index.json",
        "simulation/mujoco/mvp_scene.xml",
        "simulation/mujoco/compiled_scene_index.json",
        "simulation/previews/index.html",
        "simulation/previews/variation_scene.svg",
        "training/training_manifest.json",
        "training/training_objective.json",
        "training/training_curriculum.json",
        "training/training_run_config.json",
        "training/training_run_config_revised.json",
        "training/training_run_config_next.json",
        "training/active_training_inputs.json",
        "runs/mvp_training/dry_run_result.json",
        "runs/mvp_training/replay_index.json",
        "runs/mvp_training/logs/episode_trace.jsonl",
        "runs/mvp_training_revision/dry_run_result.json",
        "runs/mvp_training_revision/replay_index.json",
        "runs/mvp_training_revision/logs/episode_trace.jsonl",
        "runs/mvp_training_next/dry_run_result.json",
        "runs/mvp_training_next/replay_index.json",
        "runs/mvp_training_next/logs/episode_trace.jsonl",
        "reports/design_validation_report.json",
        "reports/mvp_evaluation_report.json",
        "reports/training_feedback.json",
        "reports/redesign_revision.json",
        "reports/improvement_report.json",
        "reports/promotion_decision.json",
        "software/policy_plan.json",
        "software/policy_plan_revised.json",
        "software/ros2_test_harness_placeholder.json",
        "cad/exact/robot_assembly.step",
        "cad/parametric/robot_arm.py",
        "cad/build_plan.json",
    ]
    for relative_path in expected_files:
        _check_exists(report, package_dir / relative_path, relative_path)

    _check_json(report, package_dir / "manifest.json", "manifest_json")
    _check_json(report, package_dir / "project.json", "project_json")
    _check_json(report, package_dir / "design" / "morphology_template.json", "morphology_template_json")
    _check_json(report, package_dir / "design" / "builder_dispatch.json", "builder_dispatch_json")
    _check_json(report, package_dir / "design" / "robot_build_blueprint.json", "robot_build_blueprint_json")
    _check_json(report, package_dir / "bom" / "part_resolution_report.json", "part_resolution_report_json")
    _check_json(report, package_dir / "cad" / "build_plan.json", "fabrication_build_plan_json")
    _check_json(report, package_dir / "training" / "training_manifest.json", "training_manifest_json")
    _check_json(report, package_dir / "training" / "training_objective.json", "training_objective_json")
    _check_json(report, package_dir / "training" / "training_curriculum.json", "training_curriculum_json")
    _check_json(report, package_dir / "training" / "training_run_config.json", "training_run_config_json")
    _check_json(report, package_dir / "training" / "training_run_config_revised.json", "training_run_config_revised_json")
    _check_json(report, package_dir / "training" / "training_run_config_next.json", "training_run_config_next_json")
    _check_json(report, package_dir / "training" / "active_training_inputs.json", "active_training_inputs_json")
    _check_json(report, package_dir / "software" / "policy_plan.json", "policy_plan_json")
    _check_json(report, package_dir / "software" / "policy_plan_revised.json", "policy_plan_revised_json")
    _check_json(report, package_dir / "simulation" / "revision_scene_set.json", "revision_scene_set_json")
    _check_json(report, package_dir / "simulation" / "perception_config.json", "perception_config_json")
    _check_json(report, package_dir / "simulation" / "world_model_contract.json", "world_model_contract_json")
    _check_json(report, package_dir / "simulation" / "simulator_contract.json", "simulator_contract_json")
    _check_json(report, package_dir / "datasets" / "synthetic_observation_manifest.json", "synthetic_observation_manifest_json")
    _check_json(report, package_dir / "datasets" / "synthetic_observations" / "index.json", "synthetic_observation_index_json")
    _check_json(report, package_dir / "simulation" / "world_state_index.json", "world_state_index_json")
    _check_json(report, package_dir / "simulation" / "mujoco" / "compiled_scene_index.json", "compiled_scene_index_json")
    _check_json(report, package_dir / "runs" / "mvp_training" / "dry_run_result.json", "dry_run_result_json")
    _check_json(report, package_dir / "runs" / "mvp_training" / "replay_index.json", "replay_index_json")
    _check_json(report, package_dir / "runs" / "mvp_training_revision" / "dry_run_result.json", "revised_dry_run_result_json")
    _check_json(report, package_dir / "runs" / "mvp_training_revision" / "replay_index.json", "revised_replay_index_json")
    _check_json(report, package_dir / "runs" / "mvp_training_next" / "dry_run_result.json", "next_dry_run_result_json")
    _check_json(report, package_dir / "runs" / "mvp_training_next" / "replay_index.json", "next_replay_index_json")
    _check_json(report, package_dir / "reports" / "design_validation_report.json", "design_validation_report_json")
    _check_design_validation_gates(report, package_dir)
    _check_json(report, package_dir / "reports" / "training_feedback.json", "training_feedback_json")
    _check_json(report, package_dir / "reports" / "redesign_revision.json", "redesign_revision_json")
    _check_json(report, package_dir / "reports" / "improvement_report.json", "improvement_report_json")
    _check_json(report, package_dir / "reports" / "promotion_decision.json", "promotion_decision_json")
    _check_jsonl(report, package_dir / "runs" / "mvp_training" / "logs" / "episode_trace.jsonl", "episode_trace_jsonl")
    _check_jsonl(report, package_dir / "runs" / "mvp_training_revision" / "logs" / "episode_trace.jsonl", "revised_episode_trace_jsonl")
    _check_jsonl(report, package_dir / "runs" / "mvp_training_next" / "logs" / "episode_trace.jsonl", "next_episode_trace_jsonl")
    _check_xml(report, package_dir / "robot" / "robot.urdf", "urdf_xml", expected_root="robot")
    _check_xml(report, package_dir / "simulation" / "mujoco" / "mvp_scene.xml", "mujoco_xml", expected_root="mujoco")
    _check_python_source(report, package_dir / "cad" / "parametric" / "robot_arm.py", "parametric_cad_source_python")
    _check_training_config_references(report, package_dir, "training/training_run_config.json", "training_config_references")
    _check_training_config_references(report, package_dir, "training/training_run_config_revised.json", "revised_training_config_references")
    _check_training_config_references(report, package_dir, "training/training_run_config_next.json", "next_training_config_references")
    _check_fabrication_plan_references(report, package_dir)
    _check_builder_dispatch_contract(report, package_dir)
    _check_robot_build_blueprint_references(report, package_dir)
    _check_perception_contract(report, package_dir)
    _check_world_model_contract(report, package_dir)
    _check_synthetic_observation_dataset(report, package_dir)
    _check_simulator_contract_references(report, package_dir)
    _check_training_objective_contract(report, package_dir)
    _check_compiled_scene_index_references(report, package_dir)
    _check_training_curriculum_references(report, package_dir)
    _check_active_inputs_references(report, package_dir)
    _check_replay_index_references(report, package_dir, "runs/mvp_training/replay_index.json", "replay_index_references")
    _check_replay_index_references(report, package_dir, "runs/mvp_training_revision/replay_index.json", "revised_replay_index_references")
    _check_replay_index_references(report, package_dir, "runs/mvp_training_next/replay_index.json", "next_replay_index_references")
    _check_visual_meshes(report, package_dir)
    return report


def write_validation_report(package_dir: Path) -> Path:
    report = validate_mvp_export_package(package_dir)
    path = package_dir / "reports" / "package_validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def _check_design_validation_gates(report: PackageValidationReport, package_dir: Path) -> None:
    label = "design_validation_gates"
    try:
        data = json.loads((package_dir / "reports" / "design_validation_report.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck(label, "fail", f"{label} unreadable: {exc}"))
        return
    if not data.get("ok", False):
        failures = ", ".join(data.get("blocking_failures", [])) or "unknown"
        report.checks.append(PackageCheck(label, "fail", f"Design is infeasible; failing gates: {failures}."))
        return
    report.checks.append(PackageCheck(label, "pass", "Design feasibility gates passed."))


def _check_exists(report: PackageValidationReport, path: Path, label: str) -> None:
    if path.exists():
        report.checks.append(PackageCheck(label, "pass", f"{label} exists."))
    else:
        report.checks.append(PackageCheck(label, "fail", f"{label} is missing."))


def _check_json(report: PackageValidationReport, path: Path, label: str) -> None:
    try:
        with path.open("r", encoding="utf-8") as file:
            json.load(file)
    except Exception as exc:  # noqa: BLE001 - report any parse/load problem as validation data.
        report.checks.append(PackageCheck(label, "fail", f"{label} is not valid JSON: {exc}"))
        return
    report.checks.append(PackageCheck(label, "pass", f"{label} parses as JSON."))


def _check_jsonl(report: PackageValidationReport, path: Path, label: str) -> None:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            json.loads(line)
    except Exception as exc:  # noqa: BLE001 - report any parse/load problem as validation data.
        report.checks.append(PackageCheck(label, "fail", f"{label} is not valid JSONL: {exc}"))
        return
    if not lines:
        report.checks.append(PackageCheck(label, "fail", f"{label} has no episode trace rows."))
        return
    report.checks.append(PackageCheck(label, "pass", f"{label} parses as JSONL with {len(lines)} rows."))


def _check_xml(report: PackageValidationReport, path: Path, label: str, expected_root: str) -> None:
    try:
        root = ElementTree.parse(path).getroot()
    except Exception as exc:  # noqa: BLE001 - report any parse/load problem as validation data.
        report.checks.append(PackageCheck(label, "fail", f"{label} is not valid XML: {exc}"))
        return
    if root.tag != expected_root:
        report.checks.append(PackageCheck(label, "fail", f"{label} root is {root.tag}, expected {expected_root}."))
        return
    report.checks.append(PackageCheck(label, "pass", f"{label} parses with expected root {expected_root}."))


def _check_python_source(report: PackageValidationReport, path: Path, label: str) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck(label, "fail", f"{label} is not valid Python: {exc}"))
        return
    report.checks.append(PackageCheck(label, "pass", f"{label} parses as Python."))


def _check_training_config_references(report: PackageValidationReport, package_dir: Path, config_uri: str, check_name: str) -> None:
    config_path = package_dir / config_uri
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck(check_name, "fail", f"Cannot read training config: {exc}"))
        return

    refs = [config.get("robot_model", {}).get("uri")]
    policy_artifact = config.get("policy_artifact", {}).get("uri")
    if policy_artifact:
        refs.append(policy_artifact)
    perception_artifact = config.get("perception_artifact", {}).get("uri")
    if perception_artifact:
        refs.append(perception_artifact)
    objective_artifact = config.get("objective_artifact", {}).get("uri")
    if objective_artifact:
        refs.append(objective_artifact)
    curriculum_artifact = config.get("curriculum_artifact", {}).get("uri")
    if curriculum_artifact:
        refs.append(curriculum_artifact)
    compiled_scene_artifact = config.get("compiled_scene_artifact", {}).get("uri")
    if compiled_scene_artifact:
        refs.append(compiled_scene_artifact)
    refs.extend(item.get("uri") for item in config.get("scene_artifacts", []))
    training_manifest = config.get("training_manifest", {}).get("uri")
    if training_manifest:
        refs.append(training_manifest)

    missing = [ref for ref in refs if ref and not (package_dir / ref).exists()]
    if missing:
        report.checks.append(
            PackageCheck(check_name, "fail", f"Missing referenced artifacts: {', '.join(missing)}")
        )
        return
    report.checks.append(PackageCheck(check_name, "pass", f"{config_uri} references existing artifacts."))


def _check_active_inputs_references(report: PackageValidationReport, package_dir: Path) -> None:
    active_path = package_dir / "training" / "active_training_inputs.json"
    try:
        active_inputs = json.loads(active_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("active_inputs_references", "fail", f"Cannot read active training inputs: {exc}"))
        return

    refs = [
        active_inputs.get("training_config", {}).get("uri"),
        active_inputs.get("policy_plan", {}).get("uri"),
        active_inputs.get("scene_set", {}).get("uri"),
    ]
    missing = [ref for ref in refs if ref and not (package_dir / ref).exists()]
    if missing:
        report.checks.append(PackageCheck("active_inputs_references", "fail", f"Missing active input artifacts: {', '.join(missing)}"))
        return
    report.checks.append(PackageCheck("active_inputs_references", "pass", "Active training inputs reference existing artifacts."))


def _check_fabrication_plan_references(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        plan = json.loads((package_dir / "cad" / "build_plan.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("fabrication_plan_references", "fail", f"Cannot inspect build plan: {exc}"))
        return

    refs = [
        plan.get("parametric_source", {}).get("uri"),
        plan.get("exact_assembly_artifact", {}).get("uri"),
    ]
    visual_mesh_dir = plan.get("visual_mesh_dir", {}).get("uri")
    if visual_mesh_dir:
        refs.append(visual_mesh_dir)
    missing = [ref for ref in refs if ref and not (package_dir / ref).exists()]
    if missing:
        report.checks.append(PackageCheck("fabrication_plan_references", "fail", f"Missing build-plan artifacts: {', '.join(missing)}"))
        return

    operations = plan.get("assembly_operations", [])
    checks = plan.get("fabrication_checks", [])
    if not operations or not checks:
        report.checks.append(PackageCheck("fabrication_plan_references", "fail", "Build plan lacks operations or checks."))
        return
    report.checks.append(PackageCheck("fabrication_plan_references", "pass", f"Build plan references {len(operations)} operations and {len(checks)} checks."))


def _check_robot_build_blueprint_references(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        blueprint = json.loads((package_dir / "design" / "robot_build_blueprint.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("robot_build_blueprint_references", "fail", f"Cannot inspect robot build blueprint: {exc}"))
        return

    refs = [
        artifact.get("uri")
        for handoff in blueprint.get("handoffs", [])
        for group in ("inputs", "outputs")
        for artifact in handoff.get(group, [])
    ]
    missing = [ref for ref in refs if ref and not (package_dir / ref).exists()]
    if missing:
        report.checks.append(PackageCheck("robot_build_blueprint_references", "fail", f"Blueprint references missing artifacts: {', '.join(missing)}"))
        return
    if len(blueprint.get("decisions", [])) < 5:
        report.checks.append(PackageCheck("robot_build_blueprint_references", "fail", "Blueprint does not include enough design decisions."))
        return
    if not blueprint.get("morphology_template_id") or not blueprint.get("robot_class"):
        report.checks.append(PackageCheck("robot_build_blueprint_references", "fail", "Blueprint does not name morphology template and robot class."))
        return
    report.checks.append(
        PackageCheck(
            "robot_build_blueprint_references",
            "pass",
            f"Blueprint references {len(refs)} handoff artifact(s) across {len(blueprint.get('handoffs', []))} stage(s).",
        )
    )


def _check_builder_dispatch_contract(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        dispatch = json.loads((package_dir / "design" / "builder_dispatch.json").read_text(encoding="utf-8"))
        morphology = json.loads((package_dir / "design" / "morphology_template.json").read_text(encoding="utf-8"))
        robot = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("builder_dispatch_contract", "fail", f"Cannot inspect builder dispatch: {exc}"))
        return

    if dispatch.get("selected_morphology_template_id") != morphology.get("id"):
        report.checks.append(PackageCheck("builder_dispatch_contract", "fail", "Dispatch selected template does not match exported morphology template."))
        return
    if robot.get("morphology_template_id") != morphology.get("id"):
        report.checks.append(PackageCheck("builder_dispatch_contract", "fail", "Robot genome does not reference exported morphology template."))
        return
    if dispatch.get("dispatch_status") != "implemented":
        report.checks.append(PackageCheck("builder_dispatch_contract", "fail", "Selected builder dispatch is not implemented."))
        return
    if not dispatch.get("implemented_templates"):
        report.checks.append(PackageCheck("builder_dispatch_contract", "fail", "Dispatch does not list implemented templates."))
        return
    if not dispatch.get("planned_templates"):
        report.checks.append(PackageCheck("builder_dispatch_contract", "fail", "Dispatch does not preserve planned future templates."))
        return
    report.checks.append(
        PackageCheck(
            "builder_dispatch_contract",
            "pass",
            f"Builder dispatch selected {dispatch['selected_builder_service']} with {len(dispatch.get('planned_templates', []))} planned template(s).",
        )
    )


def _check_perception_contract(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        perception = json.loads((package_dir / "simulation" / "perception_config.json").read_text(encoding="utf-8"))
        dry_run = json.loads((package_dir / "runs" / "mvp_training" / "dry_run_result.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("perception_contract", "fail", f"Cannot inspect perception contract: {exc}"))
        return

    stream_keys = {stream.get("observation_key") for stream in perception.get("sensor_streams", [])}
    result_streams = set(dry_run.get("observation_streams", []))
    if not stream_keys:
        report.checks.append(PackageCheck("perception_contract", "fail", "Perception config has no sensor streams."))
        return
    if not result_streams.issubset(stream_keys):
        missing = sorted(result_streams - stream_keys)
        report.checks.append(PackageCheck("perception_contract", "fail", f"Dry run names unknown perception streams: {', '.join(missing)}"))
        return
    report.checks.append(PackageCheck("perception_contract", "pass", f"Dry run uses {len(result_streams)} declared perception stream(s)."))


def _check_world_model_contract(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        world_model = json.loads((package_dir / "simulation" / "world_model_contract.json").read_text(encoding="utf-8"))
        perception = json.loads((package_dir / "simulation" / "perception_config.json").read_text(encoding="utf-8"))
        manifest = json.loads((package_dir / "datasets" / "synthetic_observation_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("world_model_contract", "fail", f"Cannot inspect world model contract: {exc}"))
        return

    if world_model.get("perception_config_id") != perception.get("id"):
        report.checks.append(PackageCheck("world_model_contract", "fail", "World model does not reference the exported perception config."))
        return
    dataset_uri = world_model.get("synthetic_dataset_manifest_uri")
    if not dataset_uri or not (package_dir / dataset_uri).exists():
        report.checks.append(PackageCheck("world_model_contract", "fail", "World model does not reference an existing synthetic dataset manifest."))
        return
    if manifest.get("world_model_contract_uri") != "simulation/world_model_contract.json":
        report.checks.append(PackageCheck("world_model_contract", "fail", "Synthetic dataset manifest does not link back to the world model contract."))
        return
    if not world_model.get("observable_entities"):
        report.checks.append(PackageCheck("world_model_contract", "fail", "World model has no observable entities."))
        return
    if not world_model.get("state_variables"):
        report.checks.append(PackageCheck("world_model_contract", "fail", "World model has no state variables."))
        return
    if not world_model.get("physical_parameter_targets"):
        report.checks.append(PackageCheck("world_model_contract", "fail", "World model has no Physical AI parameter targets."))
        return
    if not perception.get("vision_annotations") or not perception.get("synthetic_dataset"):
        report.checks.append(PackageCheck("world_model_contract", "fail", "Perception config lacks CV annotations or synthetic dataset spec."))
        return
    report.checks.append(
        PackageCheck(
            "world_model_contract",
            "pass",
            f"World model maps {len(world_model['observable_entities'])} observable entities and {len(world_model['physical_parameter_targets'])} physical parameter target(s).",
        )
    )


def _check_synthetic_observation_dataset(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        index = json.loads((package_dir / "datasets" / "synthetic_observations" / "index.json").read_text(encoding="utf-8"))
        world_state_index = json.loads((package_dir / "simulation" / "world_state_index.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", f"Cannot inspect synthetic observation dataset: {exc}"))
        return

    frames = index.get("frames", [])
    if not frames:
        report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", "Synthetic observation index has no frames."))
        return
    if index.get("frame_count") != len(frames):
        report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", "Synthetic observation frame count does not match frame list."))
        return
    missing = []
    annotation_count = 0
    for frame in frames:
        annotation_uri = frame.get("annotation_uri")
        if not annotation_uri or not (package_dir / annotation_uri).exists():
            missing.append(annotation_uri or "<missing annotation_uri>")
            continue
        annotation = json.loads((package_dir / annotation_uri).read_text(encoding="utf-8"))
        if not annotation.get("annotations"):
            report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", f"{annotation_uri} has no object annotations."))
            return
        world_state_uri = annotation.get("world_state_uri")
        if not world_state_uri or not (package_dir / world_state_uri).exists():
            report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", f"{annotation_uri} references missing world state."))
            return
        annotation_count += len(annotation["annotations"])
    if missing:
        report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", f"Missing annotation files: {', '.join(missing[:5])}"))
        return
    if world_state_index.get("scene_state_count") != len(world_state_index.get("scene_states", [])):
        report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", "World-state index count does not match scene-state list."))
        return
    missing_states = [
        state.get("world_state_uri")
        for state in world_state_index.get("scene_states", [])
        if not state.get("world_state_uri") or not (package_dir / state["world_state_uri"]).exists()
    ]
    if missing_states:
        report.checks.append(PackageCheck("synthetic_observation_dataset", "fail", f"Missing world-state files: {', '.join(missing_states[:5])}"))
        return
    report.checks.append(
        PackageCheck(
            "synthetic_observation_dataset",
            "pass",
            f"Synthetic dataset has {len(frames)} frame(s), {annotation_count} object annotation(s), and {world_state_index.get('scene_state_count', 0)} world-state snapshot(s).",
        )
    )


def _check_simulator_contract_references(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        contract = json.loads((package_dir / "simulation" / "simulator_contract.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("simulator_contract_references", "fail", f"Cannot inspect simulator contract: {exc}"))
        return

    missing = [
        item.get("uri")
        for item in contract.get("required_inputs", [])
        if not item.get("uri") or not (package_dir / item["uri"]).exists()
    ]
    if missing:
        report.checks.append(PackageCheck("simulator_contract_references", "fail", f"Simulator contract references missing inputs: {', '.join(missing)}"))
        return
    scene_contracts = contract.get("scene_contracts", [])
    if not scene_contracts:
        report.checks.append(PackageCheck("simulator_contract_references", "fail", "Simulator contract has no scene contracts."))
        return
    failed_scenes = [scene["scene_id"] for scene in scene_contracts if scene.get("xml_parse_status") != "pass"]
    if failed_scenes:
        report.checks.append(PackageCheck("simulator_contract_references", "fail", f"Simulator contract has invalid scene XML: {', '.join(failed_scenes)}"))
        return
    capabilities = {item.get("backend"): item for item in contract.get("backend_capabilities", [])}
    if "dry_run" not in capabilities or not capabilities["dry_run"].get("available"):
        report.checks.append(PackageCheck("simulator_contract_references", "fail", "Simulator contract must include available dry-run capability."))
        return
    if "mujoco" not in capabilities:
        report.checks.append(PackageCheck("simulator_contract_references", "fail", "Simulator contract must report MuJoCo capability status."))
        return
    report.checks.append(
        PackageCheck(
            "simulator_contract_references",
            "pass",
            f"Simulator contract validates {len(scene_contracts)} scene XML contract(s).",
        )
    )


def _check_training_objective_contract(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        objective = json.loads((package_dir / "training" / "training_objective.json").read_text(encoding="utf-8"))
        dry_run = json.loads((package_dir / "runs" / "mvp_training" / "dry_run_result.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("training_objective_contract", "fail", f"Cannot inspect training objective contract: {exc}"))
        return

    objective_terms = {term.get("name") for term in objective.get("reward_terms", [])}
    result_terms = set(dry_run.get("reward_terms", []))
    if not objective_terms:
        report.checks.append(PackageCheck("training_objective_contract", "fail", "Training objective has no reward terms."))
        return
    if not result_terms.issubset(objective_terms):
        missing = sorted(result_terms - objective_terms)
        report.checks.append(PackageCheck("training_objective_contract", "fail", f"Dry run names unknown reward terms: {', '.join(missing)}"))
        return
    if not objective.get("termination_rules"):
        report.checks.append(PackageCheck("training_objective_contract", "fail", "Training objective has no termination rules."))
        return
    report.checks.append(PackageCheck("training_objective_contract", "pass", f"Dry run uses {len(result_terms)} declared reward term(s)."))


def _check_compiled_scene_index_references(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        index = json.loads((package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("compiled_scene_index_references", "fail", f"Cannot inspect compiled scene index: {exc}"))
        return

    scenes = index.get("scenes", [])
    if index.get("scene_count") != len(scenes):
        report.checks.append(
            PackageCheck(
                "compiled_scene_index_references",
                "fail",
                f"Compiled scene count {index.get('scene_count')} does not match {len(scenes)} index entries.",
            )
        )
        return
    if not scenes:
        report.checks.append(PackageCheck("compiled_scene_index_references", "fail", "Compiled scene index has no scene entries."))
        return

    missing = []
    invalid_xml = []
    purposes = set()
    for entry in scenes:
        purpose = entry.get("purpose")
        if purpose:
            purposes.add(purpose)
        xml_uri = entry.get("mujoco_xml")
        if not xml_uri:
            missing.append("<empty mujoco_xml>")
            continue
        xml_path = package_dir / xml_uri
        if not xml_path.exists():
            missing.append(xml_uri)
            continue
        try:
            root = ElementTree.parse(xml_path).getroot()
        except Exception as exc:  # noqa: BLE001
            invalid_xml.append(f"{xml_uri}: {exc}")
            continue
        if root.tag != "mujoco":
            invalid_xml.append(f"{xml_uri}: root {root.tag}")

    if missing:
        report.checks.append(PackageCheck("compiled_scene_index_references", "fail", f"Missing compiled scene XML files: {', '.join(missing)}"))
        return
    if invalid_xml:
        report.checks.append(PackageCheck("compiled_scene_index_references", "fail", f"Invalid compiled scene XML files: {', '.join(invalid_xml)}"))
        return

    required_purposes = {"baseline", "variation", "regression", "holdout", "revision"}
    missing_purposes = sorted(required_purposes - purposes)
    if missing_purposes:
        report.checks.append(
            PackageCheck(
                "compiled_scene_index_references",
                "fail",
                f"Compiled scene index is missing purposes: {', '.join(missing_purposes)}",
            )
        )
        return
    report.checks.append(
        PackageCheck(
            "compiled_scene_index_references",
            "pass",
            f"Compiled scene index references {len(scenes)} MuJoCo scene XML file(s).",
        )
    )


def _check_training_curriculum_references(report: PackageValidationReport, package_dir: Path) -> None:
    try:
        curriculum = json.loads((package_dir / "training" / "training_curriculum.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck("training_curriculum_references", "fail", f"Cannot inspect training curriculum: {exc}"))
        return

    compiled_index_uri = curriculum.get("compiled_scene_index_uri")
    if not compiled_index_uri or not (package_dir / compiled_index_uri).exists():
        report.checks.append(PackageCheck("training_curriculum_references", "fail", "Training curriculum does not reference an existing compiled scene index."))
        return

    bindings = [
        binding
        for stage in curriculum.get("stages", [])
        for binding in stage.get("scene_bindings", [])
    ]
    if not bindings:
        report.checks.append(PackageCheck("training_curriculum_references", "fail", "Training curriculum has no scene bindings."))
        return

    missing = []
    for binding in bindings:
        for key in ("scene_set_uri", "compiled_scene_xml_uri"):
            uri = binding.get(key)
            if not uri or not (package_dir / uri).exists():
                missing.append(uri or f"<missing {key}>")
    if missing:
        report.checks.append(PackageCheck("training_curriculum_references", "fail", f"Training curriculum references missing artifacts: {', '.join(missing)}"))
        return

    total_episodes = sum(stage.get("planned_episodes", 0) for stage in curriculum.get("stages", []))
    if total_episodes <= 0:
        report.checks.append(PackageCheck("training_curriculum_references", "fail", "Training curriculum does not plan any episodes."))
        return
    report.checks.append(
        PackageCheck(
            "training_curriculum_references",
            "pass",
            f"Training curriculum binds {len(bindings)} scene(s) across {len(curriculum.get('stages', []))} stage(s).",
        )
    )


def _check_replay_index_references(report: PackageValidationReport, package_dir: Path, replay_uri: str, check_name: str) -> None:
    replay_path = package_dir / replay_uri
    try:
        replay_index = json.loads(replay_path.read_text(encoding="utf-8"))
        trace_uri = replay_index["episode_trace_uri"]
        trace_path = package_dir / trace_uri
        trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PackageCheck(check_name, "fail", f"Cannot inspect replay index references: {exc}"))
        return

    if replay_index.get("episode_count") != len(trace_rows):
        report.checks.append(
            PackageCheck(
                check_name,
                "fail",
                f"Replay index episode count {replay_index.get('episode_count')} does not match {len(trace_rows)} trace rows.",
            )
        )
        return
    if not replay_index.get("adapter_name"):
        report.checks.append(PackageCheck(check_name, "fail", "Replay index does not name a simulator adapter."))
        return
    compiled_index_uri = replay_index.get("compiled_scene_index_uri")
    if not compiled_index_uri or not (package_dir / compiled_index_uri).exists():
        report.checks.append(PackageCheck(check_name, "fail", "Replay index does not reference an existing compiled scene index."))
        return
    missing_compiled_xml = [
        uri for uri in replay_index.get("compiled_scene_xml_uris", [])
        if not (package_dir / uri).exists()
    ]
    if missing_compiled_xml:
        report.checks.append(PackageCheck(check_name, "fail", f"Replay index references missing compiled scene XML files: {', '.join(missing_compiled_xml)}"))
        return
    missing_trace_xml = [
        row.get("compiled_scene_xml_uri", "")
        for row in trace_rows
        if not row.get("compiled_scene_xml_uri") or not (package_dir / row.get("compiled_scene_xml_uri", "")).exists()
    ]
    if missing_trace_xml:
        report.checks.append(PackageCheck(check_name, "fail", "One or more episode traces lack an existing compiled scene XML reference."))
        return
    report.checks.append(
        PackageCheck(
            check_name,
            "pass",
            f"Replay index references {len(trace_rows)} trace rows from {replay_index['adapter_name']}.",
        )
    )


def _check_visual_meshes(report: PackageValidationReport, package_dir: Path) -> None:
    mesh_dir = package_dir / "cad" / "mesh" / "visual"
    stl_files = list(mesh_dir.glob("*.stl")) if mesh_dir.exists() else []
    if not stl_files:
        report.checks.append(PackageCheck("visual_meshes", "fail", "No visual STL meshes were exported."))
        return
    report.checks.append(PackageCheck("visual_meshes", "pass", f"{len(stl_files)} visual STL meshes were exported."))
