from __future__ import annotations

import json
from pathlib import Path
from html import escape
from typing import Any

from virturoid.services.package_validator import validate_mvp_export_package


def build_mvp_summary_markdown(project: dict[str, Any], package_valid: bool | None = None) -> str:
    requirements = project["requirements"]
    robot = project["robot_genome"]
    bom = project["bom"]
    selected_components = project["selected_components"]
    blueprint = project.get("robot_build_blueprint")
    morphology_template = project.get("morphology_template")
    builder_dispatch = project.get("builder_dispatch")
    part_resolution = project.get("part_resolution")
    fabrication_plan = project.get("fabrication_plan")
    task = project["task_graph"]
    scene_sets = [
        project["baseline_scene_set"],
        project["scene_set"],
        project["regression_scene_set"],
        project["holdout_scene_set"],
    ]
    evaluation_run = project["evaluation_run"]
    training_manifest = project["training_manifest"]
    training_objective = project.get("training_objective")
    training_run_config = project["training_run_config"]
    policy_plan = project["policy_plan"]
    perception_config = project.get("perception_config")
    world_model_contract = project.get("world_model_contract")
    dry_run_result = _load_dry_run_result(project)
    revised_dry_run_result = _load_revised_dry_run_result(project)
    next_training_config = _load_next_training_config(project)
    next_dry_run_result = _load_next_dry_run_result(project)
    training_feedback = _load_training_feedback(project)
    redesign_revision = _load_redesign_revision(project)
    improvement_report = _load_improvement_report(project)
    promotion_decision = _load_promotion_decision(project)
    compatibility_report = project["compatibility_report"]
    power_architecture = project.get("power_architecture")
    compiled_scene_index = _load_compiled_scene_index(project)
    training_curriculum = _load_training_curriculum(project)
    simulator_contract = _load_simulator_contract(project)

    package_status = "unknown" if package_valid is None else ("valid" if package_valid else "invalid")
    lines = [
        "# Virturoid MVP Robot Arm Summary",
        "",
        "## Request",
        "",
        f"- Prompt: {requirements.prompt}",
        f"- Environment: {requirements.environment}",
        f"- Payload: {requirements.payload_kg} kg",
        f"- Reach: {requirements.reach_m} m",
        f"- Sensors: {', '.join(requirements.sensor_requirements)}",
        "",
        "## Robot",
        "",
        f"- Name: {robot.name}",
        f"- Species: {robot.species}",
        f"- Links: {len(robot.links)}",
        f"- Joints: {len(robot.joints)}",
        f"- Sensors: {len(robot.sensors)}",
        f"- End effectors: {', '.join(robot.end_effectors)}",
        "",
        "## Selected Parts",
        "",
    ]
    for item in bom.items:
        component = next((candidate for candidate in selected_components if candidate.id == item.component_id), None)
        component_name = component.normalized_name if component else item.component_id
        lines.append(f"- {item.role}: {component_name} x{item.quantity}")

    if blueprint is not None:
        lines.extend(["", "## Robot Build Blueprint", ""])
        lines.append("- Blueprint: `design/robot_build_blueprint.json`")
        lines.append(f"- Robot class: {blueprint.robot_class}")
        lines.append(f"- Morphology template: {blueprint.morphology_template_id}")
        lines.append(f"- Morphology: {blueprint.morphology}")
        lines.append(f"- Candidate templates: {', '.join(blueprint.candidate_morphology_template_ids)}")
        lines.append(f"- Autonomous scope items: {len(blueprint.autonomy_scope)}")
        for decision in blueprint.decisions:
            lines.append(f"- {decision.area}: {decision.decision}")
    if morphology_template is not None:
        lines.extend(["", "## Morphology Template", ""])
        lines.append("- Template: `design/morphology_template.json`")
        lines.append(f"- Robot class: {morphology_template.robot_class}")
        lines.append(f"- Species pattern: {morphology_template.species_pattern}")
        lines.append(f"- Supported tasks: {', '.join(morphology_template.supported_task_types)}")
        lines.append(f"- Required modules: {', '.join(module.role for module in morphology_template.required_modules)}")
    if builder_dispatch is not None:
        lines.extend(["", "## Builder Dispatch", ""])
        lines.append("- Dispatch: `design/builder_dispatch.json`")
        lines.append(f"- Selected builder: {builder_dispatch.selected_builder_service}")
        lines.append(f"- Status: {builder_dispatch.dispatch_status}")
        lines.append(f"- Implemented templates: {len(builder_dispatch.implemented_templates)}")
        lines.append(f"- Planned templates: {len(builder_dispatch.planned_templates)}")

    if part_resolution is not None:
        lines.extend(["", "## Part Resolution", ""])
        if part_resolution.requested_part_mentions:
            lines.append(f"- Requested part mentions: {', '.join(part_resolution.requested_part_mentions)}")
        else:
            lines.append("- Requested part mentions: none detected")
        for part in part_resolution.resolved_parts:
            component = next((candidate for candidate in selected_components if candidate.id == part.component_id), None)
            component_name = component.normalized_name if component else part.component_id
            limits = ", ".join(f"{key}={value}" for key, value in part.technical_limits.items() if value is not None)
            lines.append(f"- {part.role}: {component_name} ({part.source}) - {part.reason}")
            if limits:
                lines.append(f"- {part.role} limits: {limits}")
        for warning in part_resolution.warnings:
            lines.append(f"- WARNING: {warning}")

    lines.extend(
        [
            "",
            "## Compatibility",
            "",
        ]
    )
    for check in compatibility_report.checks:
        lines.append(f"- {check.status.upper()}: {check.check} - {check.reason}")

    if fabrication_plan is not None:
        lines.extend(["", "## Fabrication Build Plan", ""])
        lines.append(f"- Parametric CAD source: `{fabrication_plan.parametric_source.uri}`")
        lines.append(f"- Exact assembly: `{fabrication_plan.exact_assembly_artifact.uri}`")
        lines.append(f"- Estimated printed mass: {fabrication_plan.estimated_printed_mass_kg} kg")
        for check in fabrication_plan.fabrication_checks:
            lines.append(f"- {check.status.upper()}: {check.check} - {check.reason}")
        for operation in fabrication_plan.assembly_operations[:8]:
            lines.append(f"- {operation.step_id}: {operation.operation} -> {operation.target}")

    if power_architecture is not None:
        lines.extend(["", "## Power Architecture", ""])
        for rail in power_architecture.rails:
            roles = ", ".join(rail.component_roles)
            lines.append(
                f"- {rail.name}: {rail.nominal_voltage_v} V, peak {rail.estimated_peak_current_a} A, "
                f"recommended regulator {rail.recommended_regulator_current_a} A, headroom {rail.headroom_percent}% ({roles})"
            )
        if power_architecture.battery:
            battery = power_architecture.battery
            lines.append(
                f"- Battery estimate: {battery.capacity_wh} Wh pack, {battery.usable_capacity_wh} Wh usable, "
                f"{battery.estimated_average_power_w} W average, ~{battery.estimated_runtime_minutes} minutes runtime"
            )
        for warning in power_architecture.warnings:
            lines.append(f"- WARNING: {warning}")

    lines.extend(
        [
            "",
            "## Task",
            "",
            f"- Type: {task.task_type}",
            f"- Required skills: {', '.join(task.required_skills)}",
            f"- Objects: {', '.join(task.objects)}",
            "",
            "## Scene Sets",
            "",
        ]
    )
    for scene_set in scene_sets:
        lines.append(f"- {scene_set.purpose}: {len(scene_set.scenes)} scenes ({scene_set.id})")

    if compiled_scene_index is not None:
        lines.extend(
            [
                "",
                "## Compiled MuJoCo Scenes",
                "",
                "- Index: `simulation/mujoco/compiled_scene_index.json`",
                f"- Scene XML files: {compiled_scene_index.get('scene_count', 0)}",
            ]
        )
        for entry in compiled_scene_index.get("scenes", [])[:5]:
            lines.append(f"- {entry['purpose']}: `{entry['mujoco_xml']}`")

    if simulator_contract is not None:
        lines.extend(
            [
                "",
                "## Simulator Contract",
                "",
                "- Contract: `simulation/simulator_contract.json`",
                f"- Scene XML contracts: {len(simulator_contract.get('scene_contracts', []))}",
            ]
        )
        for capability in simulator_contract.get("backend_capabilities", []):
            lines.append(
                f"- {capability['backend']}: {capability['status']} "
                f"(available={str(capability['available']).lower()})"
            )

    if training_curriculum is not None:
        lines.extend(
            [
                "",
                "## Training Curriculum",
                "",
                "- Curriculum: `training/training_curriculum.json`",
                f"- Planned episodes: {sum(stage.get('planned_episodes', 0) for stage in training_curriculum.get('stages', []))}",
                f"- Success checks: {len(training_curriculum.get('success_checks', []))}",
                f"- Safety checks: {len(training_curriculum.get('safety_checks', []))}",
            ]
        )
        for stage in training_curriculum.get("stages", []):
            lines.append(f"- {stage['purpose']}: {stage['scene_count']} scenes, {stage['planned_episodes']} episodes")

    lines.extend(
        [
            "",
            "## Training Package",
            "",
            f"- Backend: {training_manifest.backend}",
            f"- Total scenes: {training_manifest.total_scene_count}",
            f"- Robot model: {training_run_config.robot_model.uri if training_run_config.robot_model else 'missing'}",
            f"- Policy plan: {training_run_config.policy_artifact.uri if training_run_config.policy_artifact else 'missing'}",
            f"- Output directory: {training_run_config.output_dir}",
            f"- Policy steps: {', '.join(step.step_id for step in policy_plan.steps)}",
        ]
    )
    if dry_run_result is not None:
        lines.extend(
            [
                f"- Simulator adapter: {dry_run_result['adapter_name']} ({dry_run_result['backend']})",
                f"- Dry-run planned episodes: {dry_run_result['total_planned_episodes']}",
                f"- Dry-run estimated success rate: {dry_run_result['estimated_success_rate']}",
                f"- Replay index: `{dry_run_result['replay_index_uri']}`",
                f"- Episode trace: `{dry_run_result['episode_trace_uri']}`",
            ]
        )
        for group in dry_run_result.get("group_results", []):
            lines.append(f"- {group['purpose']}: {group['planned_episodes']} planned episodes across {group['scene_count']} scenes")

    if perception_config is not None:
        lines.extend(["", "## Perception Config", ""])
        for stream in perception_config.sensor_streams:
            modalities = ", ".join(stream.output_modalities)
            outputs = ", ".join(stream.simulated_outputs)
            lines.append(f"- {stream.name}: {stream.sensor_type}, modalities {modalities}, outputs {outputs}")
        for annotation in perception_config.vision_annotations:
            lines.append(
                f"- CV annotation {annotation.name}: {', '.join(annotation.annotation_modalities)} "
                f"for labels {', '.join(annotation.label_space)}"
            )
        if perception_config.synthetic_dataset is not None:
            lines.append(f"- Synthetic dataset root: `{perception_config.synthetic_dataset.output_root}`")
        for rule in perception_config.texture_randomization:
            lines.append(f"- Texture {rule.target}: {', '.join(rule.variants)}")

    if world_model_contract is not None:
        lines.extend(["", "## World Model Contract", ""])
        lines.append("- Contract: `simulation/world_model_contract.json`")
        lines.append(f"- Synthetic observation manifest: `{world_model_contract.synthetic_dataset_manifest_uri}`")
        lines.append("- Synthetic observation index: `datasets/synthetic_observations/index.json`")
        lines.append("- World-state index: `simulation/world_state_index.json`")
        lines.append(f"- Observable entities: {len(world_model_contract.observable_entities)}")
        lines.append(f"- State variables: {len(world_model_contract.state_variables)}")
        lines.append(f"- Physical AI targets: {len(world_model_contract.physical_parameter_targets)}")
        for target in world_model_contract.physical_parameter_targets[:6]:
            lines.append(f"- {target.name}: {target.optimization_role} from {target.learning_signal}")

    if training_objective is not None:
        lines.extend(["", "## Training Objective", ""])
        lines.append(f"- Objective type: {training_objective.objective_type}")
        lines.append(f"- Metrics: {', '.join(training_objective.metrics)}")
        for reward in training_objective.reward_terms:
            lines.append(f"- Reward {reward.name}: {reward.expression} (weight {reward.weight})")
        for rule in training_objective.termination_rules[:6]:
            lines.append(f"- Termination {rule.name}: {rule.expression} -> {rule.terminal_status}")

    if revised_dry_run_result is not None:
        lines.extend(
            [
                f"- Revised dry-run planned episodes: {revised_dry_run_result['total_planned_episodes']}",
                f"- Revised dry-run estimated success rate: {revised_dry_run_result['estimated_success_rate']}",
                f"- Revised replay index: `{revised_dry_run_result['replay_index_uri']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            f"- Episodes: {len(evaluation_run.episodes)}",
            f"- Failures: {len(evaluation_run.failures)}",
        ]
    )
    for failure in evaluation_run.failures[:5]:
        lines.append(f"- {failure.failure_type}: {failure.summary}")

    if training_feedback is not None:
        lines.extend(
            [
                "",
                "## Training Feedback",
                "",
                f"- Status: {training_feedback['overall_status']}",
            ]
        )
        for recommendation in training_feedback.get("recommendations", []):
            actions = "; ".join(recommendation.get("suggested_actions", []))
            lines.append(
                f"- {recommendation['priority'].upper()} {recommendation['target']}: "
                f"{recommendation['summary']} Actions: {actions}"
            )

    if redesign_revision is not None:
        lines.extend(
            [
                "",
                "## Redesign Revision",
                "",
                f"- Revised policy: `{redesign_revision['revised_policy_artifact']['uri']}`",
                f"- Revised scene set: `{redesign_revision['revised_scene_artifact']['uri']}`",
            ]
        )
        for item in redesign_revision.get("applied_recommendations", []):
            lines.append(f"- {item['recommendation_id']}: {', '.join(item.get('changes', []))}")

    if improvement_report is not None:
        lines.extend(
            [
                "",
                "## Improvement Report",
                "",
                f"- Outcome: {improvement_report['outcome']}",
                f"- Basis: {improvement_report['comparison_basis']}",
            ]
        )
        for metric in improvement_report.get("metrics", []):
            lines.append(f"- {metric['name']}: {metric['before']} -> {metric['after']} (delta {metric['delta']})")
        for action in improvement_report.get("next_actions", []):
            lines.append(f"- Next action: {action}")

    if promotion_decision is not None:
        lines.extend(
            [
                "",
                "## Promotion Decision",
                "",
                f"- Decision: {promotion_decision['decision']}",
                f"- Selected training config: `{promotion_decision['selected_training_config']['uri']}`",
                f"- Selected policy: `{promotion_decision['selected_policy_plan']['uri']}`",
                f"- Selected scene set: `{promotion_decision['selected_scene_set']['uri']}`",
                f"- Rationale: {promotion_decision['rationale']}",
            ]
        )

    if next_training_config is not None and next_dry_run_result is not None:
        lines.extend(
            [
                "",
                "## Next Training Run",
                "",
                f"- Config: `training/training_run_config_next.json`",
                f"- Policy: `{next_training_config['policy_artifact']['uri']}`",
                f"- Scene set: `{next_training_config['scene_artifacts'][0]['uri']}`",
                f"- Output directory: {next_training_config['output_dir']}",
                f"- Planned episodes: {next_dry_run_result['total_planned_episodes']}",
                f"- Estimated success rate: {next_dry_run_result['estimated_success_rate']}",
                f"- Replay index: `{next_dry_run_result['replay_index_uri']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Export Validation",
            "",
            f"- Package status: {package_status}",
            "",
            "## Key Artifacts",
            "",
            "- `robot/robot.urdf`",
            "- `design/morphology_template.json`",
            "- `design/builder_dispatch.json`",
            "- `design/robot_build_blueprint.json`",
            "- `simulation/mujoco/mvp_scene.xml`",
            "- `simulation/mujoco/compiled_scene_index.json`",
            "- `simulation/simulator_contract.json`",
            "- `simulation/previews/index.html`",
            "- `simulation/perception_config.json`",
            "- `simulation/world_model_contract.json`",
            "- `datasets/synthetic_observation_manifest.json`",
            "- `datasets/synthetic_observations/index.json`",
            "- `simulation/world_state_index.json`",
            "- `bom/part_resolution_report.json`",
            "- `cad/build_plan.json`",
            "- `cad/parametric/robot_arm.py`",
            "- `training/training_manifest.json`",
            "- `training/training_objective.json`",
            "- `training/training_curriculum.json`",
            "- `training/training_run_config.json`",
            "- `training/training_run_config_revised.json`",
            "- `training/training_run_config_next.json`",
            "- `training/active_training_inputs.json`",
            "- `software/policy_plan.json`",
            "- `software/policy_plan_revised.json`",
            "- `runs/mvp_training/dry_run_result.json`",
            "- `runs/mvp_training/replay_index.json`",
            "- `runs/mvp_training/logs/episode_trace.jsonl`",
            "- `runs/mvp_training_revision/dry_run_result.json`",
            "- `runs/mvp_training_revision/replay_index.json`",
            "- `runs/mvp_training_revision/logs/episode_trace.jsonl`",
            "- `runs/mvp_training_next/dry_run_result.json`",
            "- `runs/mvp_training_next/replay_index.json`",
            "- `runs/mvp_training_next/logs/episode_trace.jsonl`",
            "- `reports/training_feedback.json`",
            "- `reports/redesign_revision.json`",
            "- `reports/improvement_report.json`",
            "- `reports/promotion_decision.json`",
            "- `simulation/revision_scene_set.json`",
            "- `cad/exact/robot_assembly.step`",
            "- `cad/mesh/visual/*.stl`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_mvp_summary_report(project: dict[str, Any], output_dir: Path) -> Path:
    package_valid = validate_mvp_export_package(output_dir).ok
    path = output_dir / "reports" / "mvp_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_mvp_summary_markdown({**project, "_output_dir": output_dir}, package_valid=package_valid), encoding="utf-8")
    return path


def build_mvp_html_report(project: dict[str, Any], package_valid: bool | None = None) -> str:
    requirements = project["requirements"]
    robot = project["robot_genome"]
    bom = project["bom"]
    selected_components = project["selected_components"]
    blueprint = project.get("robot_build_blueprint")
    morphology_template = project.get("morphology_template")
    builder_dispatch = project.get("builder_dispatch")
    part_resolution = project.get("part_resolution")
    fabrication_plan = project.get("fabrication_plan")
    task = project["task_graph"]
    policy_plan = project["policy_plan"]
    perception_config = project.get("perception_config")
    world_model_contract = project.get("world_model_contract")
    training_manifest = project["training_manifest"]
    training_objective = project.get("training_objective")
    evaluation_run = project["evaluation_run"]
    dry_run_result = _load_dry_run_result(project)
    revised_dry_run_result = _load_revised_dry_run_result(project)
    next_training_config = _load_next_training_config(project)
    next_dry_run_result = _load_next_dry_run_result(project)
    training_feedback = _load_training_feedback(project)
    redesign_revision = _load_redesign_revision(project)
    improvement_report = _load_improvement_report(project)
    promotion_decision = _load_promotion_decision(project)
    compatibility_report = project["compatibility_report"]
    power_architecture = project.get("power_architecture")
    compiled_scene_index = _load_compiled_scene_index(project)
    training_curriculum = _load_training_curriculum(project)
    simulator_contract = _load_simulator_contract(project)
    package_status = "unknown" if package_valid is None else ("valid" if package_valid else "invalid")
    status_class = "ok" if package_valid else "bad"

    part_rows = []
    for item in bom.items:
        component = next((candidate for candidate in selected_components if candidate.id == item.component_id), None)
        component_name = component.normalized_name if component else item.component_id
        part_rows.append(
            f"<tr><td>{escape(item.role)}</td><td>{escape(component_name)}</td><td>{item.quantity}</td></tr>"
        )
    part_resolution_html = ""
    if part_resolution is not None:
        part_resolution_rows = []
        for part in part_resolution.resolved_parts:
            component = next((candidate for candidate in selected_components if candidate.id == part.component_id), None)
            component_name = component.normalized_name if component else part.component_id
            limits = "; ".join(f"{key}={value}" for key, value in part.technical_limits.items() if value is not None)
            part_resolution_rows.append(
                f"<tr><td>{escape(part.role)}</td><td>{escape(component_name)}</td><td>{escape(part.source)}</td><td>{escape(part.reason)}</td><td>{escape(limits)}</td></tr>"
            )
        requested_mentions = ", ".join(part_resolution.requested_part_mentions) or "none detected"
        warnings = "".join(f"<li>{escape(warning)}</li>" for warning in part_resolution.warnings)
        if not warnings:
            warnings = "<li>No part-resolution warnings.</li>"
        part_resolution_html = f"""
  <h2>Part Resolution</h2>
  <p><a href="../bom/part_resolution_report.json">bom/part_resolution_report.json</a></p>
  <p>Requested part mentions: {escape(requested_mentions)}</p>
  <table>
    <thead><tr><th>Role</th><th>Component</th><th>Source</th><th>Reason</th><th>Technical Limits</th></tr></thead>
    <tbody>{''.join(part_resolution_rows)}</tbody>
  </table>
  <ul>{warnings}</ul>
"""
    blueprint_html = ""
    if blueprint is not None:
        decision_rows = "".join(
            f"<tr><td>{escape(decision.area)}</td><td>{escape(decision.decision)}</td><td>{escape(decision.rationale)}</td></tr>"
            for decision in blueprint.decisions
        )
        blueprint_html = f"""
  <h2>Robot Build Blueprint</h2>
  <p><a href="../design/robot_build_blueprint.json">design/robot_build_blueprint.json</a></p>
  <p>Robot class: <code>{escape(blueprint.robot_class)}</code> | Template: <code>{escape(blueprint.morphology_template_id)}</code> | Handoffs: {len(blueprint.handoffs)}</p>
  <table>
    <thead><tr><th>Area</th><th>Decision</th><th>Rationale</th></tr></thead>
    <tbody>{decision_rows}</tbody>
  </table>
"""
    morphology_html = ""
    if morphology_template is not None:
        module_rows = "".join(
            f"<tr><td>{escape(module.role)}</td><td>{escape(module.description)}</td><td>{escape(', '.join(module.compatible_component_categories))}</td></tr>"
            for module in morphology_template.required_modules
        )
        morphology_html = f"""
  <h2>Morphology Template</h2>
  <p><a href="../design/morphology_template.json">design/morphology_template.json</a></p>
  <p>Robot class: <code>{escape(morphology_template.robot_class)}</code> | Species pattern: <code>{escape(morphology_template.species_pattern)}</code></p>
  <table>
    <thead><tr><th>Module</th><th>Description</th><th>Component Categories</th></tr></thead>
    <tbody>{module_rows}</tbody>
  </table>
"""
    builder_dispatch_html = ""
    if builder_dispatch is not None:
        planned_rows = "".join(
            f"<tr><td>{escape(item.morphology_template_id)}</td><td>{escape(item.robot_class)}</td><td>{escape(item.builder_service)}</td></tr>"
            for item in builder_dispatch.planned_templates
        )
        builder_dispatch_html = f"""
  <h2>Builder Dispatch</h2>
  <p><a href="../design/builder_dispatch.json">design/builder_dispatch.json</a></p>
  <p>Selected builder: <code>{escape(builder_dispatch.selected_builder_service)}</code> | Status: <code>{escape(builder_dispatch.dispatch_status)}</code></p>
  <table>
    <thead><tr><th>Planned Template</th><th>Robot Class</th><th>Builder Service</th></tr></thead>
    <tbody>{planned_rows}</tbody>
  </table>
"""
    fabrication_html = ""
    if fabrication_plan is not None:
        fabrication_rows = "".join(
            f"<tr><td>{escape(check.status)}</td><td>{escape(check.check)}</td><td>{escape(check.reason)}</td></tr>"
            for check in fabrication_plan.fabrication_checks
        )
        operation_rows = "".join(
            f"<tr><td>{escape(operation.step_id)}</td><td>{escape(operation.operation)}</td><td>{escape(operation.target)}</td></tr>"
            for operation in fabrication_plan.assembly_operations[:10]
        )
        fabrication_html = f"""
  <h2>Fabrication Build Plan</h2>
  <p><a href="../cad/build_plan.json">cad/build_plan.json</a> | <a href="../cad/parametric/robot_arm.py">cad/parametric/robot_arm.py</a></p>
  <p>Estimated printed mass: {fabrication_plan.estimated_printed_mass_kg} kg</p>
  <table>
    <thead><tr><th>Status</th><th>Check</th><th>Reason</th></tr></thead>
    <tbody>{fabrication_rows}</tbody>
  </table>
  <table>
    <thead><tr><th>Step</th><th>Operation</th><th>Target</th></tr></thead>
    <tbody>{operation_rows}</tbody>
  </table>
"""

    scene_rows = []
    for group in training_manifest.scene_groups:
        scene_rows.append(
            f"<tr><td>{escape(group.purpose)}</td><td>{escape(group.scene_set_id)}</td><td>{group.scene_count}</td><td>{escape(group.usage)}</td></tr>"
        )
    compiled_scene_html = ""
    if compiled_scene_index is not None:
        compiled_rows = "".join(
            f"<tr><td>{escape(entry['purpose'])}</td><td>{escape(entry['scene_id'])}</td><td><a href=\"../{escape(entry['mujoco_xml'])}\">{escape(entry['mujoco_xml'])}</a></td><td>{entry['object_count']}</td></tr>"
            for entry in compiled_scene_index.get("scenes", [])[:10]
        )
        compiled_scene_html = f"""
  <h2>Compiled MuJoCo Scenes</h2>
  <p><a href="../simulation/mujoco/compiled_scene_index.json">simulation/mujoco/compiled_scene_index.json</a></p>
  <p>Scene XML files: {compiled_scene_index.get('scene_count', 0)}</p>
  <table>
    <thead><tr><th>Purpose</th><th>Scene</th><th>MJCF XML</th><th>Objects</th></tr></thead>
    <tbody>{compiled_rows}</tbody>
  </table>
"""
    simulator_contract_html = ""
    if simulator_contract is not None:
        capability_rows = "".join(
            f"<tr><td>{escape(item['backend'])}</td><td>{escape(item['adapter_name'])}</td><td>{escape(item['status'])}</td><td>{str(item['available']).lower()}</td><td>{escape(item.get('reason', ''))}</td></tr>"
            for item in simulator_contract.get("backend_capabilities", [])
        )
        simulator_contract_html = f"""
  <h2>Simulator Contract</h2>
  <p><a href="../simulation/simulator_contract.json">simulation/simulator_contract.json</a></p>
  <p>Scene XML contracts: {len(simulator_contract.get('scene_contracts', []))}</p>
  <table>
    <thead><tr><th>Backend</th><th>Adapter</th><th>Status</th><th>Available</th><th>Reason</th></tr></thead>
    <tbody>{capability_rows}</tbody>
  </table>
"""
    curriculum_html = ""
    if training_curriculum is not None:
        curriculum_rows = "".join(
            f"<tr><td>{escape(stage['purpose'])}</td><td>{escape(stage['usage'])}</td><td>{stage['scene_count']}</td><td>{stage['planned_episodes']}</td></tr>"
            for stage in training_curriculum.get("stages", [])
        )
        curriculum_html = f"""
  <h2>Training Curriculum</h2>
  <p><a href="../training/training_curriculum.json">training/training_curriculum.json</a></p>
  <p>Success checks: {len(training_curriculum.get('success_checks', []))} | Safety checks: {len(training_curriculum.get('safety_checks', []))}</p>
  <table>
    <thead><tr><th>Purpose</th><th>Usage</th><th>Scenes</th><th>Episodes</th></tr></thead>
    <tbody>{curriculum_rows}</tbody>
  </table>
"""

    policy_rows = "".join(
        f"<tr><td>{escape(step.step_id)}</td><td>{escape(step.skill)}</td><td>{escape(step.command)}</td><td>{escape(step.success_condition)}</td></tr>"
        for step in policy_plan.steps
    )

    compatibility_items = "".join(
        f"<li><strong>{escape(check.status.upper())}</strong>: {escape(check.check)} - {escape(check.reason)}</li>"
        for check in compatibility_report.checks
    )
    power_rows = ""
    if power_architecture is not None:
        power_rows = "".join(
            f"<tr><td>{escape(rail.name)}</td><td>{rail.nominal_voltage_v}</td><td>{rail.estimated_peak_current_a}</td><td>{rail.recommended_regulator_current_a}</td><td>{rail.headroom_percent}%</td><td>{escape(', '.join(rail.component_roles))}</td></tr>"
            for rail in power_architecture.rails
        )
    battery_html = ""
    if power_architecture is not None and power_architecture.battery is not None:
        battery = power_architecture.battery
        battery_html = (
            f"<p>Battery estimate: {battery.capacity_wh} Wh pack, {battery.usable_capacity_wh} Wh usable, "
            f"{battery.estimated_average_power_w} W average draw, ~{battery.estimated_runtime_minutes} minutes runtime.</p>"
        )
    failure_items = "".join(
        f"<li><strong>{escape(failure.failure_type)}</strong>: {escape(failure.summary)}</li>"
        for failure in evaluation_run.failures[:8]
    )
    if not failure_items:
        failure_items = "<li>No failures recorded.</li>"
    dry_run_html = ""
    if dry_run_result is not None:
        rows = "".join(
            f"<tr><td>{escape(group['purpose'])}</td><td>{group['scene_count']}</td><td>{group['planned_episodes']}</td><td>{group['simulated_success_rate']}</td></tr>"
            for group in dry_run_result.get("group_results", [])
        )
        dry_run_html = f"""
  <h2>Dry-Run Training Result</h2>
  <p>Simulator adapter: <code>{escape(dry_run_result['adapter_name'])}</code> ({escape(dry_run_result['backend'])})</p>
  <p>Planned episodes: {dry_run_result['total_planned_episodes']} | Estimated success rate: {dry_run_result['estimated_success_rate']}</p>
  <p>Replay artifacts: <a href="../{escape(dry_run_result['replay_index_uri'])}">replay index</a> | <a href="../{escape(dry_run_result['episode_trace_uri'])}">episode trace</a></p>
  <table>
    <thead><tr><th>Purpose</th><th>Scenes</th><th>Episodes</th><th>Estimated Success</th></tr></thead>
    <tbody>{rows}</tbody>
</table>
"""
    perception_html = ""
    if perception_config is not None:
        perception_rows = "".join(
            f"<tr><td>{escape(stream.name)}</td><td>{escape(stream.sensor_type)}</td><td>{escape(', '.join(stream.output_modalities))}</td><td>{escape(', '.join(stream.simulated_outputs))}</td></tr>"
            for stream in perception_config.sensor_streams
        )
        annotation_rows = "".join(
            f"<tr><td>{escape(annotation.name)}</td><td>{escape(', '.join(annotation.annotation_modalities))}</td><td>{escape(', '.join(annotation.label_space))}</td></tr>"
            for annotation in perception_config.vision_annotations
        )
        texture_items = "".join(
            f"<li>{escape(rule.target)}: {escape(', '.join(rule.variants))}</li>"
            for rule in perception_config.texture_randomization
        )
        dataset_root = perception_config.synthetic_dataset.output_root if perception_config.synthetic_dataset else "not declared"
        perception_html = f"""
  <h2>Perception Config</h2>
  <p><a href="../simulation/perception_config.json">simulation/perception_config.json</a></p>
  <p>Synthetic dataset root: <code>{escape(dataset_root)}</code></p>
  <table>
    <thead><tr><th>Stream</th><th>Sensor Type</th><th>Modalities</th><th>Simulated Outputs</th></tr></thead>
    <tbody>{perception_rows}</tbody>
  </table>
  <table>
    <thead><tr><th>CV Annotation</th><th>Modalities</th><th>Labels</th></tr></thead>
    <tbody>{annotation_rows}</tbody>
  </table>
  <ul>{texture_items}</ul>
"""
    world_model_html = ""
    if world_model_contract is not None:
        target_rows = "".join(
            f"<tr><td>{escape(target.name)}</td><td>{escape(target.target)}</td><td>{escape(target.optimization_role)}</td><td>{escape(target.learning_signal)}</td></tr>"
            for target in world_model_contract.physical_parameter_targets
        )
        world_model_html = f"""
  <h2>World Model Contract</h2>
  <p><a href="../simulation/world_model_contract.json">simulation/world_model_contract.json</a></p>
  <p><a href="../{escape(world_model_contract.synthetic_dataset_manifest_uri)}">{escape(world_model_contract.synthetic_dataset_manifest_uri)}</a></p>
  <p><a href="../datasets/synthetic_observations/index.json">datasets/synthetic_observations/index.json</a></p>
  <p><a href="../simulation/world_state_index.json">simulation/world_state_index.json</a></p>
  <p>Observable entities: {len(world_model_contract.observable_entities)} | State variables: {len(world_model_contract.state_variables)}</p>
  <table>
    <thead><tr><th>Physical AI Target</th><th>Target</th><th>Role</th><th>Learning Signal</th></tr></thead>
    <tbody>{target_rows}</tbody>
  </table>
"""
    objective_html = ""
    if training_objective is not None:
        reward_rows = "".join(
            f"<tr><td>{escape(term.name)}</td><td>{escape(term.kind)}</td><td>{term.weight}</td><td>{escape(term.expression)}</td></tr>"
            for term in training_objective.reward_terms
        )
        termination_rows = "".join(
            f"<tr><td>{escape(rule.name)}</td><td>{escape(rule.expression)}</td><td>{escape(rule.terminal_status)}</td></tr>"
            for rule in training_objective.termination_rules
        )
        objective_html = f"""
  <h2>Training Objective</h2>
  <p><a href="../training/training_objective.json">training/training_objective.json</a></p>
  <p>Metrics: {escape(', '.join(training_objective.metrics))}</p>
  <table>
    <thead><tr><th>Reward</th><th>Kind</th><th>Weight</th><th>Expression</th></tr></thead>
    <tbody>{reward_rows}</tbody>
  </table>
  <table>
    <thead><tr><th>Rule</th><th>Expression</th><th>Status</th></tr></thead>
    <tbody>{termination_rows}</tbody>
  </table>
"""
    feedback_html = ""
    if training_feedback is not None:
        feedback_rows = "".join(
            f"<tr><td>{escape(item['priority'])}</td><td>{escape(item['target'])}</td><td>{escape(item['summary'])}</td><td>{escape('; '.join(item.get('suggested_actions', [])))}</td></tr>"
            for item in training_feedback.get("recommendations", [])
        )
        feedback_html = f"""
  <h2>Training Feedback</h2>
  <p>Status: <code>{escape(training_feedback['overall_status'])}</code> | <a href="training_feedback.json">reports/training_feedback.json</a></p>
  <table>
    <thead><tr><th>Priority</th><th>Target</th><th>Summary</th><th>Suggested Actions</th></tr></thead>
    <tbody>{feedback_rows}</tbody>
  </table>
"""
    revised_dry_run_html = ""
    if revised_dry_run_result is not None:
        revised_rows = "".join(
            f"<tr><td>{escape(group['purpose'])}</td><td>{group['scene_count']}</td><td>{group['planned_episodes']}</td><td>{group['simulated_success_rate']}</td></tr>"
            for group in revised_dry_run_result.get("group_results", [])
        )
        revised_dry_run_html = f"""
  <h2>Revised Dry-Run Training Result</h2>
  <p>Planned episodes: {revised_dry_run_result['total_planned_episodes']} | Estimated success rate: {revised_dry_run_result['estimated_success_rate']}</p>
  <p>Replay artifacts: <a href="../{escape(revised_dry_run_result['replay_index_uri'])}">replay index</a> | <a href="../{escape(revised_dry_run_result['episode_trace_uri'])}">episode trace</a></p>
  <table>
    <thead><tr><th>Purpose</th><th>Scenes</th><th>Episodes</th><th>Estimated Success</th></tr></thead>
    <tbody>{revised_rows}</tbody>
  </table>
"""
    redesign_html = ""
    if redesign_revision is not None:
        redesign_rows = "".join(
            f"<tr><td>{escape(item['recommendation_id'])}</td><td>{escape(item['target'])}</td><td>{escape('; '.join(item.get('changes', [])))}</td></tr>"
            for item in redesign_revision.get("applied_recommendations", [])
        )
        redesign_html = f"""
  <h2>Redesign Revision</h2>
  <p><a href="redesign_revision.json">reports/redesign_revision.json</a></p>
  <p>Revised policy: <a href="../{escape(redesign_revision['revised_policy_artifact']['uri'])}">{escape(redesign_revision['revised_policy_artifact']['uri'])}</a></p>
  <p>Revised scene set: <a href="../{escape(redesign_revision['revised_scene_artifact']['uri'])}">{escape(redesign_revision['revised_scene_artifact']['uri'])}</a></p>
  <table>
    <thead><tr><th>Recommendation</th><th>Target</th><th>Changes</th></tr></thead>
    <tbody>{redesign_rows}</tbody>
  </table>
"""
    improvement_html = ""
    if improvement_report is not None:
        improvement_rows = "".join(
            f"<tr><td>{escape(item['name'])}</td><td>{item['before']}</td><td>{item['after']}</td><td>{item['delta']}</td><td>{escape(item['interpretation'])}</td></tr>"
            for item in improvement_report.get("metrics", [])
        )
        next_actions = "".join(
            f"<li>{escape(action)}</li>"
            for action in improvement_report.get("next_actions", [])
        )
        improvement_html = f"""
  <h2>Improvement Report</h2>
  <p>Outcome: <code>{escape(improvement_report['outcome'])}</code> | <a href="improvement_report.json">reports/improvement_report.json</a></p>
  <p>{escape(improvement_report['comparison_basis'])}</p>
  <table>
    <thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Delta</th><th>Interpretation</th></tr></thead>
    <tbody>{improvement_rows}</tbody>
  </table>
  <ul>{next_actions}</ul>
"""
    promotion_html = ""
    if promotion_decision is not None:
        gates = "".join(
            f"<li>{escape(name)}: {str(value).lower()}</li>"
            for name, value in promotion_decision.get("gates", {}).items()
        )
        promotion_html = f"""
  <h2>Promotion Decision</h2>
  <p>Decision: <code>{escape(promotion_decision['decision'])}</code> | <a href="promotion_decision.json">reports/promotion_decision.json</a></p>
  <p>{escape(promotion_decision['rationale'])}</p>
  <ul>{gates}</ul>
  <p>Active inputs: <a href="../training/active_training_inputs.json">training/active_training_inputs.json</a></p>
  <table>
    <thead><tr><th>Input</th><th>Artifact</th></tr></thead>
    <tbody>
      <tr><td>Training config</td><td>{escape(promotion_decision['selected_training_config']['uri'])}</td></tr>
      <tr><td>Policy plan</td><td>{escape(promotion_decision['selected_policy_plan']['uri'])}</td></tr>
      <tr><td>Scene set</td><td>{escape(promotion_decision['selected_scene_set']['uri'])}</td></tr>
    </tbody>
  </table>
"""
    next_run_html = ""
    if next_training_config is not None and next_dry_run_result is not None:
        next_run_html = f"""
  <h2>Next Training Run</h2>
  <p><a href="../training/training_run_config_next.json">training/training_run_config_next.json</a></p>
  <p>Policy: <a href="../{escape(next_training_config['policy_artifact']['uri'])}">{escape(next_training_config['policy_artifact']['uri'])}</a></p>
  <p>Scene set: <a href="../{escape(next_training_config['scene_artifacts'][0]['uri'])}">{escape(next_training_config['scene_artifacts'][0]['uri'])}</a></p>
  <p>Planned episodes: {next_dry_run_result['total_planned_episodes']} | Estimated success rate: {next_dry_run_result['estimated_success_rate']}</p>
  <p>Replay artifacts: <a href="../{escape(next_dry_run_result['replay_index_uri'])}">replay index</a> | <a href="../{escape(next_dry_run_result['episode_trace_uri'])}">episode trace</a></p>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Virturoid MVP Robot Arm Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #20242a; line-height: 1.45; }}
    header {{ border-bottom: 1px solid #d8dde5; margin-bottom: 24px; padding-bottom: 16px; }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #d8dde5; padding: 8px; text-align: left; }}
    th {{ background: #f4f6f8; }}
    .status {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: 700; }}
    .ok {{ background: #e4f6ea; color: #176b35; }}
    .bad {{ background: #fde7e7; color: #9b1c1c; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .panel {{ border: 1px solid #d8dde5; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    code {{ background: #eef1f5; padding: 2px 4px; border-radius: 4px; }}
    a {{ color: #1457a8; }}
  </style>
</head>
<body>
  <header>
    <h1>Virturoid MVP Robot Arm Report</h1>
    <p><span class="status {status_class}">Package {escape(package_status)}</span></p>
    <p>{escape(requirements.prompt)}</p>
  </header>

  <section class="grid">
    <div class="panel">
      <h2>Robot</h2>
      <p><strong>{escape(robot.name)}</strong></p>
      <p>Species: <code>{escape(robot.species)}</code></p>
      <p>Links: {len(robot.links)} | Joints: {len(robot.joints)} | Sensors: {len(robot.sensors)}</p>
    </div>
    <div class="panel">
      <h2>Task</h2>
      <p>Type: <code>{escape(task.task_type)}</code></p>
      <p>Payload: {requirements.payload_kg} kg | Reach: {requirements.reach_m} m</p>
      <p>Environment: {escape(requirements.environment or "unknown")}</p>
    </div>
  </section>

  <h2>Selected Parts</h2>
  <table>
    <thead><tr><th>Role</th><th>Component</th><th>Qty</th></tr></thead>
    <tbody>{''.join(part_rows)}</tbody>
  </table>

  {blueprint_html}
  {morphology_html}
  {builder_dispatch_html}
  {part_resolution_html}
  {fabrication_html}

  <h2>Training Scene Groups</h2>
  <table>
    <thead><tr><th>Purpose</th><th>Scene Set</th><th>Scenes</th><th>Usage</th></tr></thead>
    <tbody>{''.join(scene_rows)}</tbody>
  </table>
  {compiled_scene_html}
  {simulator_contract_html}
  {curriculum_html}

  <h2>Policy Plan</h2>
  <p><a href="../software/policy_plan.json">software/policy_plan.json</a></p>
  <table>
    <thead><tr><th>Step</th><th>Skill</th><th>Command</th><th>Success Condition</th></tr></thead>
    <tbody>{policy_rows}</tbody>
  </table>

  <h2>Compatibility</h2>
  <ul>{compatibility_items}</ul>

  <h2>Power Architecture</h2>
  <table>
    <thead><tr><th>Rail</th><th>Voltage</th><th>Peak Current</th><th>Regulator</th><th>Headroom</th><th>Roles</th></tr></thead>
    <tbody>{power_rows}</tbody>
  </table>
  {battery_html}

  <h2>Evaluation Failures</h2>
  <ul>{failure_items}</ul>

  {dry_run_html}
  {perception_html}
  {world_model_html}
  {objective_html}
  {feedback_html}
  {redesign_html}
  {revised_dry_run_html}
  {improvement_html}
  {promotion_html}
  {next_run_html}

  <h2>Key Artifacts</h2>
  <ul>
    <li><a href="../robot/robot.urdf">robot/robot.urdf</a></li>
    <li><a href="../design/morphology_template.json">design/morphology_template.json</a></li>
    <li><a href="../design/builder_dispatch.json">design/builder_dispatch.json</a></li>
    <li><a href="../design/robot_build_blueprint.json">design/robot_build_blueprint.json</a></li>
    <li><a href="../simulation/mujoco/mvp_scene.xml">simulation/mujoco/mvp_scene.xml</a></li>
    <li><a href="../simulation/mujoco/compiled_scene_index.json">simulation/mujoco/compiled_scene_index.json</a></li>
    <li><a href="../simulation/simulator_contract.json">simulation/simulator_contract.json</a></li>
    <li><a href="../simulation/previews/index.html">simulation/previews/index.html</a></li>
    <li><a href="../simulation/perception_config.json">simulation/perception_config.json</a></li>
    <li><a href="../simulation/world_model_contract.json">simulation/world_model_contract.json</a></li>
    <li><a href="../datasets/synthetic_observation_manifest.json">datasets/synthetic_observation_manifest.json</a></li>
    <li><a href="../datasets/synthetic_observations/index.json">datasets/synthetic_observations/index.json</a></li>
    <li><a href="../simulation/world_state_index.json">simulation/world_state_index.json</a></li>
    <li><a href="../bom/part_resolution_report.json">bom/part_resolution_report.json</a></li>
    <li><a href="../cad/build_plan.json">cad/build_plan.json</a></li>
    <li><a href="../cad/parametric/robot_arm.py">cad/parametric/robot_arm.py</a></li>
    <li><a href="../training/training_manifest.json">training/training_manifest.json</a></li>
    <li><a href="../training/training_objective.json">training/training_objective.json</a></li>
    <li><a href="../training/training_curriculum.json">training/training_curriculum.json</a></li>
    <li><a href="../training/training_run_config.json">training/training_run_config.json</a></li>
    <li><a href="../training/training_run_config_revised.json">training/training_run_config_revised.json</a></li>
    <li><a href="../training/training_run_config_next.json">training/training_run_config_next.json</a></li>
    <li><a href="../training/active_training_inputs.json">training/active_training_inputs.json</a></li>
    <li><a href="../software/policy_plan.json">software/policy_plan.json</a></li>
    <li><a href="../software/policy_plan_revised.json">software/policy_plan_revised.json</a></li>
    <li><a href="../simulation/revision_scene_set.json">simulation/revision_scene_set.json</a></li>
    <li><a href="../runs/mvp_training/dry_run_result.json">runs/mvp_training/dry_run_result.json</a></li>
    <li><a href="../runs/mvp_training/replay_index.json">runs/mvp_training/replay_index.json</a></li>
    <li><a href="../runs/mvp_training/logs/episode_trace.jsonl">runs/mvp_training/logs/episode_trace.jsonl</a></li>
    <li><a href="../runs/mvp_training_revision/dry_run_result.json">runs/mvp_training_revision/dry_run_result.json</a></li>
    <li><a href="../runs/mvp_training_revision/replay_index.json">runs/mvp_training_revision/replay_index.json</a></li>
    <li><a href="../runs/mvp_training_revision/logs/episode_trace.jsonl">runs/mvp_training_revision/logs/episode_trace.jsonl</a></li>
    <li><a href="../runs/mvp_training_next/dry_run_result.json">runs/mvp_training_next/dry_run_result.json</a></li>
    <li><a href="../runs/mvp_training_next/replay_index.json">runs/mvp_training_next/replay_index.json</a></li>
    <li><a href="../runs/mvp_training_next/logs/episode_trace.jsonl">runs/mvp_training_next/logs/episode_trace.jsonl</a></li>
    <li><a href="training_feedback.json">reports/training_feedback.json</a></li>
    <li><a href="redesign_revision.json">reports/redesign_revision.json</a></li>
    <li><a href="improvement_report.json">reports/improvement_report.json</a></li>
    <li><a href="promotion_decision.json">reports/promotion_decision.json</a></li>
    <li><a href="../power/power_architecture.json">power/power_architecture.json</a></li>
    <li><a href="../cad/exact/robot_assembly.step">cad/exact/robot_assembly.step</a></li>
    <li><a href="../cad/mesh/visual/">cad/mesh/visual/</a></li>
    <li><a href="package_validation_report.json">reports/package_validation_report.json</a></li>
    <li><a href="mvp_summary.md">reports/mvp_summary.md</a></li>
  </ul>
</body>
</html>
"""


def write_mvp_html_report(project: dict[str, Any], output_dir: Path) -> Path:
    package_valid = validate_mvp_export_package(output_dir).ok
    path = output_dir / "reports" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_mvp_html_report({**project, "_output_dir": output_dir}, package_valid=package_valid), encoding="utf-8")
    return path


def _load_dry_run_result(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "runs" / "mvp_training" / "dry_run_result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_revised_dry_run_result(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "runs" / "mvp_training_revision" / "dry_run_result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_next_training_config(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "training" / "training_run_config_next.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_next_dry_run_result(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "runs" / "mvp_training_next" / "dry_run_result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_training_feedback(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "reports" / "training_feedback.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_redesign_revision(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "reports" / "redesign_revision.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_improvement_report(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "reports" / "improvement_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_promotion_decision(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "reports" / "promotion_decision.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_compiled_scene_index(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "simulation" / "mujoco" / "compiled_scene_index.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_training_curriculum(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "training" / "training_curriculum.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_simulator_contract(project: dict[str, Any]) -> dict[str, Any] | None:
    output_dir = project.get("_output_dir")
    if output_dir is None:
        return None
    path = Path(output_dir) / "simulation" / "simulator_contract.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
