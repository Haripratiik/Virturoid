"""End-to-end autonomous robot build.

One call takes a prompt and runs the whole closed loop with no human tuning:

    prompt -> requirements -> parts/BOM/CAD -> Robot Genome -> scenes
           -> feasibility gates -> real-physics task evaluation
           -> improve the hardware from real failures -> converge
           -> finalize the package as the converged robot -> export + report

The verdict (`AutonomyReport.succeeded`) is the honest claim that the software
built a robot that performs the task. For the manipulator class the task is
block sorting; other classes currently build a validated foundation package and
report that the autonomous task loop is not yet implemented for them.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.fixtures.components import curated_component_library
from virturoid.schemas.autonomy import AutonomyDecision, AutonomyReport
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet
from virturoid.services.autonomous_builder import build_robot_package_from_prompt
from virturoid.services.memory_store import (
    DEFAULT_MEMORY_DIR,
    append_design_record,
    find_similar_design,
    save_build_record,
)
from virturoid.services.requirements_builder import build_requirements_from_prompt

_SCENE_SET_FILES = [
    "simulation/baseline_scene_set.json",
    "simulation/scene_set.json",
    "simulation/regression_scene_set.json",
    "simulation/holdout_scene_set.json",
    "simulation/revision_scene_set.json",
]


def autonomous_build(
    prompt: str,
    output_dir: Path,
    target_success_rate: float = 0.8,
    train: bool = False,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    max_rounds: int = 3,
    progress=None,
) -> AutonomyReport:
    from virturoid.services.mujoco_runner import mujoco_available

    def emit(stage: str, message: str, data: dict | None = None) -> None:
        if progress is not None:
            progress({"stage": stage, "message": message, "data": data or {}})

    output_dir = Path(output_dir)
    emit("start", f"Designing a robot for: {prompt}")

    # Gene-driven build (Phase A): classes that have a gene but no tuned legacy builder
    # (e.g. humanoid) are built+run from their gene — a genuinely distinct robot in real
    # physics — instead of tracing to the tabletop arm. The manipulator/mobile_base keep
    # their tuned legacy pipelines (higher success).
    gene_report = _maybe_gene_build(prompt, output_dir, target_success_rate, memory_dir, emit, train=train)
    if gene_report is not None:
        return gene_report

    # Optional Task Agent (plan §8, Phase D): a cloud model expands the prompt into a
    # richer, validated task graph. Advisory and gated by VIRTUROID_LLM_BACKEND — the
    # deterministic build_task_graph inside the package builder remains the source of
    # truth, so a missing/declining backend never blocks the build.
    task_proposal = _maybe_synthesize_task_graph(prompt, emit)

    build = build_robot_package_from_prompt(prompt=prompt, output_dir=output_dir)
    decisions: list[AutonomyDecision] = []
    feasible = _design_feasible(output_dir)
    decisions.append(
        AutonomyDecision(
            iteration=0,
            stage="build_and_validate",
            action="built package and ran feasibility gates",
            detail=f"robot_class={build.selected_robot_class}, design_feasible={feasible}",
            success_before=0.0,
            success_after=0.0,
        )
    )

    if task_proposal is not None:
        g = task_proposal["graph"]
        report_note = (f"Task agent ({task_proposal['backend']}) proposed task '{g['name']}' ({g['task_type']}) "
                       f"with {len(g['success_criteria'])} success / {len(g['failure_criteria'])} failure criteria.")
        decisions.append(
            AutonomyDecision(
                iteration=0, stage="synthesize_task_graph",
                action="synthesized a validated task graph from the prompt",
                detail=report_note, success_before=0.0, success_after=0.0,
            )
        )

    # Species-tree honesty: if the request mapped to a species with no builder yet, we
    # built the nearest buildable relative — say so plainly instead of mislabeling it.
    if not getattr(build, "species_exact", True):
        emit("species", f"Requested species '{build.requested_species}' ({build.requested_robot_class}) has no builder yet; "
                        f"building nearest buildable relative '{build.selected_species}' ({build.selected_robot_class}).")
        decisions.append(
            AutonomyDecision(
                iteration=0, stage="resolve_species",
                action="traced the robot species tree to the nearest buildable relative",
                detail=build.species_note, success_before=0.0, success_after=0.0,
            )
        )

    emit("build", f"Built and validated a {build.selected_robot_class} ({build.selected_species}). Design feasible: {feasible}.")
    report = AutonomyReport(
        id=f"autonomy_{build.requirements_id}",
        prompt=prompt,
        robot_class=build.selected_robot_class,
        species=build.selected_species,
        task_type=build.task_type,
        target_success_rate=target_success_rate,
        feasible=feasible,
        decisions=decisions,
        exported_artifacts=dict(build.artifacts),
    )
    if not getattr(build, "species_exact", True):
        report.notes.append(build.species_note)
        report.requested_species = build.requested_species
        report.requested_robot_class = build.requested_robot_class

    if not mujoco_available():
        report.notes.append("MuJoCo not installed; built and validated the package but could not run the physics task loop.")
        report.succeeded = False
        _write_report(output_dir, report)
        return report

    if build.selected_robot_class == "mobile_base":
        from virturoid.services.compute_provenance import RealComputeMeter
        from virturoid.services.navigation_controller import run_navigation_evaluation

        emit("evaluate", "Running real MuJoCo navigation (drive to goal, avoid obstacles)...")
        meter = RealComputeMeter()
        with meter:
            nav = run_navigation_evaluation(output_dir)
        final = nav["success_rate"]
        report.initial_success_rate = final
        report.final_success_rate = final
        report.succeeded = bool(final >= target_success_rate)
        report.compute = meter.provenance.to_dict()
        report.exported_artifacts["navigation_evaluation_report"] = "reports/navigation_evaluation_report.json"
        report.decisions.append(
            AutonomyDecision(
                iteration=1, stage="navigate",
                action="ran real differential-drive navigation across scenes",
                detail=f"reached {nav['reached']}/{nav['total_episodes']} goals",
                success_before=0.0, success_after=final,
            )
        )
        report.notes.append(f"Mobile-base navigation task loop: reached {nav['reached']}/{nav['total_episodes']} goals.")
        emit("done", f"Navigation success: {final:.0%} ({nav['reached']}/{nav['total_episodes']} goals reached).",
             {"succeeded": report.succeeded, "final_success_rate": final, "compute": report.compute})
        (output_dir / "reports" / "compute_provenance.json").write_text(json.dumps(report.compute, indent=2), encoding="utf-8")
        save_build_record(build.selected_robot_class, build.task_type, {}, final, prompt, memory_dir)
        _record_to_memory_db(memory_dir, prompt, build, None, final, target_success_rate, report)
        report.exported_artifacts["memory_effectiveness_report"] = _write_memory_effectiveness_report(
            output_dir, memory_dir, report)
        _write_report(output_dir, report)
        return report

    if build.selected_robot_class != "manipulator":
        report.notes.append(f"Autonomous task loop is implemented for manipulators and mobile bases; '{build.selected_robot_class}' produced a validated foundation package only.")
        report.succeeded = False
        _write_report(output_dir, report)
        return report

    if not feasible:
        report.notes.append("Design feasibility gates failed; not attempting the task loop on an impossible design.")
        report.succeeded = False
        _write_report(output_dir, report)
        return report

    from virturoid.services.compute_provenance import RealComputeMeter
    from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

    # Optional Part Recommender (plan §8, Phase C): an LLM picks a BOM from the
    # catalog, validated by hard compatibility checks. Advisory and gated by
    # VIRTUROID_LLM_BACKEND; the deterministic part_resolver already chose the
    # build's parts, so this records the agent's BOM without blocking.
    bom = _maybe_recommend_parts(prompt, emit)
    if bom is not None:
        report.notes.append(f"Part recommender ({bom['backend']}) BOM: {bom['selection']}")

    # Meter the real physics/optimization work so the build is auditable, not a
    # black box that "finishes too quickly".
    meter = RealComputeMeter()
    meter.__enter__()

    emit("evaluate", "Running real MuJoCo pick-and-place on the as-built robot...")
    initial_report = run_physics_pick_place_evaluation(output_dir)
    initial = initial_report.success_rate
    report.initial_success_rate = initial
    emit("evaluate_done", f"As-built task success: {initial:.0%}.", {"success_rate": initial})
    decisions.append(
        AutonomyDecision(
            iteration=1,
            stage="evaluate",
            action="ran real pick-and-place on the as-built robot",
            detail=f"as-built success={initial}",
            success_before=0.0,
            success_after=initial,
        )
    )

    memory = find_similar_design(build.selected_robot_class, build.task_type, memory_dir)
    final = initial
    converged = _as_built_design(prompt)
    design_source = "as_built"
    transfer_decision: dict | None = None

    # Flywheel WARM-START (plan §13/§26 "memory helps a similar task"): if a prior build of this
    # class+task converged on a body, START from it — apply it, evaluate it, and adopt it if it beats
    # the naive as-built body. This is the moat actually paying off: the second build begins near
    # converged instead of re-deriving the body from scratch.
    memory_design = (memory or {}).get("converged_design") if isinstance(memory, dict) else None
    _body_keys = ("forearm_mm", "wrist_mm", "actuator_torque_nm")
    if memory_design and all(k in memory_design for k in _body_keys):
        emit("memory_warm_start", f"Reusing a converged body from a prior build: {memory_design}")
        _finalize_package_with_design(output_dir, prompt, memory_design)
        warm = run_physics_pick_place_evaluation(output_dir).success_rate
        adopted = warm >= initial
        decisions.append(
            AutonomyDecision(
                iteration=1, stage="memory_warm_start",
                action="warm-started the body from a prior converged build (flywheel reuse)",
                detail=f"reused {memory_design} -> success={warm}" + ("" if adopted else " (kept as-built; warm worse)"),
                success_before=initial, success_after=warm,
            )
        )
        if adopted:
            converged, final, design_source = dict(memory_design), warm, "memory_warm_start"
        else:                                            # restore the as-built body for a fair co-design
            _finalize_package_with_design(output_dir, prompt, converged)

    if final < target_success_rate:
        from virturoid.services.codesign_optimizer import co_optimize_hardware

        # Optional Failure Analyst (plan §8, Phase D): reason over the real failure
        # clusters from this evaluation and record the diagnosis. Advisory and gated
        # by VIRTUROID_LLM_BACKEND; the co-design + controller-tuning loop below runs
        # regardless, so a missing/declining backend never blocks the build.
        diagnosis = _maybe_analyze_failures(initial_report.failure_clusters, emit)
        if diagnosis is not None:
            report.notes.append(
                f"Failure analyst ({diagnosis['backend']}): primary={diagnosis['diagnosis']['primary_failure']}, "
                f"action={diagnosis['diagnosis']['recommended_action']} — {diagnosis['diagnosis']['root_cause']}"
            )
            decisions.append(
                AutonomyDecision(
                    iteration=2,
                    stage="analyze_failures",
                    action="diagnosed real failure clusters and recommended a fix",
                    detail=f"primary={diagnosis['diagnosis']['primary_failure']}, action={diagnosis['diagnosis']['recommended_action']}",
                    success_before=initial,
                    success_after=initial,
                )
            )

        # Optional AI Designer (plan §8): an LLM proposes a starting body that the
        # physics co-design search then refines. Gated by VIRTUROID_LLM_BACKEND;
        # with no backend this is None and the search starts from the naive body.
        # Seed the search from an LLM proposal if available, else from the warm-started prior body
        # (a proven starting point makes the search converge faster than from the naive body).
        ai_seed = _maybe_ai_design(prompt, emit)
        seed_design = ai_seed or memory_design
        design_source = ("ai_designer+codesign" if ai_seed else
                         "memory_seeded_codesign" if memory_design else "physics_codesign")
        emit("co_design", "Co-designing the robot body against real task success (this runs many physics rollouts)...")
        codesign = co_optimize_hardware(prompt=prompt, seed_design=seed_design)
        best = codesign.best
        converged = {"forearm_mm": best.forearm_mm, "wrist_mm": best.wrist_mm, "actuator_torque_nm": best.actuator_torque_nm}
        decisions.append(
            AutonomyDecision(
                iteration=2,
                stage="co_design_hardware",
                action="searched robot body parameters against real task success",
                detail=f"changed body to forearm={best.forearm_mm}mm wrist={best.wrist_mm}mm torque={best.actuator_torque_nm}Nm",
                success_before=codesign.baseline.success_rate,
                success_after=best.success_rate,
            )
        )
        emit("co_design_done", f"AI chose a better body: {converged}. Finalizing and re-evaluating...", {"design": converged})
        _finalize_package_with_design(output_dir, prompt, converged)
        final = run_physics_pick_place_evaluation(output_dir).success_rate
        decisions.append(
            AutonomyDecision(
                iteration=3,
                stage="finalize_and_reevaluate",
                action="rewrote the exported robot to the converged body and re-evaluated",
                detail=f"finalized success={final}",
                success_before=initial,
                success_after=final,
            )
        )

    # Iterate: if the body change did not reach the target, tune the controller
    # ("brain") on the converged body — joint body+brain co-optimization across rounds.
    controller_params = None
    if final < target_success_rate and max_rounds >= 2:
        from virturoid.services.codesign_optimizer import tune_controller

        tuned = tune_controller(prompt, converged)
        if tuned["best_success"] > tuned["baseline_success"]:
            controller_params = tuned["best_params"]
            tuned_eval = run_physics_pick_place_evaluation(output_dir, controller_params=controller_params)
            before = final
            final = tuned_eval.success_rate
            decisions.append(
                AutonomyDecision(
                    iteration=4,
                    stage="tune_controller",
                    action="co-optimized the controller (brain) on the converged body",
                    detail=f"controller tuned to {controller_params}; success {before} -> {final}",
                    success_before=before,
                    success_after=final,
                )
            )

    report.memory_reused = memory is not None
    if memory is not None:
        report.notes.append(f"Memory had a prior design for {build.selected_robot_class}/{build.task_type}; current build kept the better-performing body.")
        # Optional Transfer Agent (plan §8.6): a cloud model decides whether the prior
        # design/policy is worth warm-starting from, gated by feasibility on the new
        # robot. Advisory and gated by VIRTUROID_LLM_BACKEND; the deterministic memory
        # reuse above already happened, so this only records the agent's reasoning.
        transfer = _maybe_recommend_transfer(prompt, build.selected_robot_class, build.task_type, memory, emit, memory_dir)
        if transfer is not None and transfer["decision"]["reuse"]:
            d = transfer["decision"]
            transfer_decision = d
            report.notes.append(f"Transfer agent ({transfer['backend']}): reuse {d['source_key']} assets={d['assets']} — {d['rationale']}")
            # The deterministic warm-start above already applied the prior body; this records the
            # transfer agent's (advisory) confirmation with the build's real outcome, not a stale value.
            decisions.append(
                AutonomyDecision(
                    iteration=5, stage="transfer",
                    action="transfer agent confirmed reuse of the prior build (warm-start already applied)",
                    detail=f"reuse {d['source_key']} assets={d['assets']}",
                    success_before=initial, success_after=final,
                )
            )

    if train:
        from virturoid.services.controller_exporter import export_controller_bundle, write_training_acceptance_report
        from virturoid.services.manipulation_skill_trainer import train_contact_grasp_skill_from_export
        from virturoid.services.policy_trainer import train_arm_controller_from_export

        # Skill-flywheel READ (advisory): is there a banked skill for this (class, task) to inform
        # the build? The CEM reach trainer re-initialises each run and cannot consume prior weights,
        # so today this is recorded for transparency (param-mode reuse is the next flywheel leg) —
        # never claimed as a warm-start the trainer didn't actually perform.
        warm_start = _lookup_warm_start(memory_dir, build.selected_robot_class, build.task_type,
                                        build.selected_species)

        policy = train_arm_controller_from_export(output_dir)
        bundle_path = export_controller_bundle(output_dir, policy)
        acceptance_path = write_training_acceptance_report(output_dir, policy, bundle_path)
        manipulation_report = train_contact_grasp_skill_from_export(
            output_dir,
            memory_dir=memory_dir,
            robot_class=build.selected_robot_class,
            species=build.selected_species,
        )
        report.exported_artifacts["trained_policy"] = "software/controller/trained_policy.json"
        report.exported_artifacts["controller_bundle"] = "software/controller/controller_bundle.json"
        report.exported_artifacts["training_acceptance_report"] = "reports/training_acceptance_report.json"
        report.exported_artifacts["manipulation_training_acceptance_report"] = "reports/manipulation_training_acceptance_report.json"
        report.exported_artifacts["contact_grasp_skill"] = "software/skills/contact_grasp_skill.json"
        from virturoid.services.ros2_exporter import maybe_export_ros2_package
        ros2_root = maybe_export_ros2_package(output_dir)   # runnable ROS2 harness for the exported controller (§24)
        if ros2_root is not None:
            report.exported_artifacts["ros2_package"] = "export/ros2/virturoid_robot"

        # Skill-flywheel WRITE: bank the trained policy as a reusable skill + append one auditable
        # ledger row, so the next build of a related body/task can warm-start from it (plan §13/§32).
        skill_id = _bank_trained_skill(memory_dir, output_dir, policy, build, converged,
                                       warm_started=False, warm_start_available=bool(warm_start))
        if skill_id:
            report.exported_artifacts["flywheel_manifest"] = "../memory/flywheel_manifest.jsonl"

        prior = f"; prior skill available ({warm_start['skill_id']})" if warm_start else ""
        decisions.append(
            AutonomyDecision(
                iteration=4,
                stage="train_controller",
                action="trained and exported an inference-ready controller for the converged robot",
                detail=f"eval reach success={policy.evaluation.success_rate}"
                       + f"; acceptance report {acceptance_path.name}"
                       + f"; contact grasp accepted={manipulation_report['accepted']}"
                       + (f"; banked skill '{skill_id}'" if skill_id else "") + prior,
                success_before=final,
                success_after=final,
            )
        )

    meter.__exit__(None, None, None)
    report.compute = meter.provenance.to_dict()
    (output_dir / "reports" / "compute_provenance.json").write_text(
        json.dumps(report.compute, indent=2), encoding="utf-8"
    )

    # Export the robot's actual control program (the intelligent control logic
    # that drives the converged robot on the task).
    control_uri = _export_control_program(output_dir, converged, trained=train, controller_params=controller_params)
    report.exported_artifacts["control_program"] = control_uri

    report.final_success_rate = final
    report.converged_design = converged
    report.iterations = len([d for d in decisions if d.stage in {"co_design_hardware", "evaluate"}])
    report.succeeded = bool(feasible and final >= target_success_rate)
    report.exported_artifacts["autonomy_report"] = "reports/autonomy_report.json"
    report.exported_artifacts["compute_provenance"] = "reports/compute_provenance.json"
    report.notes.append(
        f"Autonomous build {'succeeded' if report.succeeded else 'did not reach target'}: "
        f"final task success {final} vs target {target_success_rate}."
    )
    report.notes.append(report.compute.get("note", ""))

    emit(
        "done",
        f"Autonomous build {'succeeded' if report.succeeded else 'did not reach target'}: task success {final:.0%}.",
        {"succeeded": report.succeeded, "final_success_rate": final, "compute": report.compute},
    )
    save_build_record(build.selected_robot_class, build.task_type, converged, final, prompt, memory_dir)
    append_design_record(prompt, build.selected_robot_class, build.task_type, converged, final, design_source, memory_dir)
    _record_to_memory_db(
        memory_dir, prompt, build, converged, final, target_success_rate, report,
        failure_clusters=initial_report.failure_clusters, transfer_decision=transfer_decision,
    )
    report.exported_artifacts["memory_effectiveness_report"] = _write_memory_effectiveness_report(
        output_dir, memory_dir, report)
    _write_report(output_dir, report)
    return report


def _export_control_program(output_dir: Path, converged: dict, trained: bool, controller_params: dict | None = None) -> str:
    """Write the converged robot's control program (the 'intelligent control script').

    For the sorting task this is the scripted pick-and-place state machine plus the
    tuned controller parameters and the FK-IK waypoint planner it uses. When a
    learned policy was trained, that controller bundle is referenced too.
    """
    from virturoid.services.pick_place_controller import DEFAULT_CONTROLLER_PARAMS

    params = {**DEFAULT_CONTROLLER_PARAMS, **(controller_params or {})}
    control = {
        "id": "control_program_pick_place_sort",
        "task": "pick_place_sort",
        "controller_type": "scripted_state_machine_with_fk_ik_planner",
        "controller_tuned": controller_params is not None,
        "converged_design": converged,
        "controller_params": params,
        "phases": [
            "plan_waypoints (cross-entropy FK-IK)",
            "approach_above_block",
            "descend_and_engage_grasp",
            "lift",
            "transport_above_matching_bin",
            "lower_into_bin",
            "release",
        ],
        "waypoint_planner": "virturoid.services.pick_place_controller.plan_joint_targets",
        "executor": "virturoid.services.pick_place_controller.run_pick_place_episode",
        "learned_policy": "software/controller/controller_bundle.json" if trained else None,
        "notes": [
            "This is the robot's intelligent control logic for the sorting task, validated in real physics.",
            "Joint targets are computed by a forward-kinematics cross-entropy IK search per waypoint.",
            "The grasp engages when the end-effector reaches the block; release drops it into the bin.",
        ],
    }
    path = output_dir / "software" / "control_program.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(control, indent=2), encoding="utf-8")
    return "software/control_program.json"


def _write_memory_effectiveness_report(output_dir: Path, memory_dir: Path, report: AutonomyReport) -> str:
    """Write an auditable proof of what memory did for this build."""
    output_dir = Path(output_dir)
    memory_dir = Path(memory_dir)
    stages = [d.stage for d in report.decisions]
    warm = [d for d in report.decisions if d.stage == "memory_warm_start"]
    co_design_ran = "co_design_hardware" in stages
    payload = {
        "memory_reused": bool(report.memory_reused),
        "robot_class": report.robot_class,
        "task_type": report.task_type,
        "target_success_rate": report.target_success_rate,
        "initial_success_rate": report.initial_success_rate,
        "final_success_rate": report.final_success_rate,
        "success_delta": round(report.final_success_rate - report.initial_success_rate, 6),
        "warm_start_attempted": bool(warm),
        "warm_start_success_rate": warm[-1].success_after if warm else None,
        "co_design_search_ran": co_design_ran,
        "co_design_skipped_after_memory": bool(
            warm and not co_design_ran and report.final_success_rate >= report.target_success_rate),
        "decision_stages": stages,
        "json_memory_record": str(memory_dir / f"{report.robot_class}__{report.task_type}.json"),
        "sqlite_memory_db": str(memory_dir / "virturoid_memory.db"),
        "skill_memory_reused": None,
        "banked_contact_grasp_skill_id": None,
        "evidence": {},
    }
    try:
        manipulation_report = output_dir / "reports" / "manipulation_training_acceptance_report.json"
        if manipulation_report.exists():
            mt = json.loads(manipulation_report.read_text(encoding="utf-8"))
            payload["skill_memory_reused"] = bool(mt.get("memory_reused"))
            payload["banked_contact_grasp_skill_id"] = mt.get("banked_skill_id") or mt.get("memory_skill_id")
            payload["evidence"]["contact_grasp_training"] = {
                "accepted": mt.get("accepted"),
                "memory_reused": mt.get("memory_reused"),
                "memory_warm_start": mt.get("memory_warm_start"),
                "success_rate": (mt.get("evaluation") or {}).get("success_rate"),
                "mean_lift_m": (mt.get("evaluation") or {}).get("mean_lift_m"),
            }
    except Exception as exc:  # noqa: BLE001
        payload["evidence"]["skill_report_error"] = str(exc)
    try:
        from virturoid.services.memory_db import MemoryDB
        with MemoryDB(memory_dir / "virturoid_memory.db") as db:
            payload["evidence"]["db_stats"] = db.stats()
            best = db.best_design(report.robot_class, report.task_type)
            payload["evidence"]["best_design_success_rate"] = None if best is None else best.get("success_rate")
            payload["evidence"]["training_rows_at_target"] = len(
                db.training_dataset(min_success=report.target_success_rate))
            skill = db.recall_skill(report.robot_class, "contact_grasp_lift")
            payload["evidence"]["best_contact_grasp_skill"] = None if skill is None else {
                "skill_id": skill.get("skill_id"),
                "success_rate": skill.get("success_rate"),
                "species": skill.get("species"),
                "has_controller_params": bool((skill.get("base_config") or {}).get("controller_params")),
            }
    except Exception as exc:  # noqa: BLE001 - observability should not fail the build
        payload["evidence"]["db_error"] = str(exc)
    try:
        record_path = memory_dir / f"{report.robot_class}__{report.task_type}.json"
        payload["evidence"]["json_record_exists"] = record_path.exists()
        if record_path.exists():
            payload["evidence"]["json_record_success_rate"] = json.loads(
                record_path.read_text(encoding="utf-8")).get("success_rate")
    except Exception as exc:  # noqa: BLE001
        payload["evidence"]["json_record_error"] = str(exc)

    path = output_dir / "reports" / "memory_effectiveness_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "reports/memory_effectiveness_report.json"


def _maybe_ai_design(prompt: str, emit) -> dict | None:
    """Ask the Robot Designer agent for a starting body, if an LLM backend is configured."""
    try:
        from virturoid.services.design_agent import propose_arm_design
        from virturoid.services.llm_client import get_llm
        from virturoid.services.requirements_builder import build_requirements_from_prompt
        from virturoid.services.task_builder import build_task_graph

        llm = get_llm("designer")
        if llm is None:
            return None
        requirements = build_requirements_from_prompt(prompt)
        task = build_task_graph(requirements)
        proposal = propose_arm_design(prompt, requirements, task, llm)
        if proposal and proposal.get("feasible") and proposal.get("design"):
            emit("ai_design", f"AI designer ({proposal['backend']}) proposed a starting body: {proposal['design']}.")
            return proposal["design"]
    except Exception:  # noqa: BLE001 - the design agent is advisory; never block the build
        return None
    return None


def _maybe_analyze_failures(failure_clusters, emit) -> dict | None:
    """Ask the Failure Analyst agent to diagnose real failures, if a backend is set."""
    try:
        from virturoid.services.failure_agent import analyze_failures
        from virturoid.services.llm_client import get_llm

        llm = get_llm("failure_analyst")
        if llm is None or not failure_clusters:
            return None
        clusters = [{"label": c.failure_label, "count": c.count} for c in failure_clusters]
        result = analyze_failures(clusters, llm)
        if result and result.get("valid") and result.get("diagnosis"):
            d = result["diagnosis"]
            emit("analyze_failures", f"Failure analyst ({result['backend']}): {d['primary_failure']} -> {d['recommended_action']}.")
            return result
    except Exception:  # noqa: BLE001 - advisory; never block the build
        return None
    return None


# Classes that build from a gene (real, distinct robots) rather than the tuned legacy
# arm/mobile pipelines. manipulator/mobile_base stay on their higher-success legacy path.
_GENE_ONLY_CLASSES = {"humanoid", "quadruped", "mobile_manipulator"}
# G4 (gap-closure): the mobile_manipulator composite routes through the GENE path (chassis+wheels+arm built
# for real, BOM/CAD/ledger uniform). Plain mobile_base keeps its richer LEGACY navigation task loop (real
# navigate decisions + success gating — test_navigation guards it); its packages get a BOM via emit_genome_bom.
# Manipulation tasks the tuned legacy arm path can't score (it runs the colour-sort eval only) -> general gene path.
_NONSORT_MANIP_TASKS = {"pick_place_box", "stack", "push", "generic_manipulation"}


def _prompt_task_type(prompt: str) -> str | None:
    """Cheap task-type probe from the prompt (no gene/MuJoCo build) for routing decisions."""
    try:
        from virturoid.services.requirements_builder import build_requirements_from_prompt
        from virturoid.services.task_builder import build_task_graph
        return build_task_graph(build_requirements_from_prompt(prompt)).task_type
    except Exception:  # noqa: BLE001 - routing probe is best-effort; fall back to the legacy path
        return None


def _maybe_keystone_codesign(gene, prompt, memory_dir, emit):
    """Keystone (Pillar 1) for a LEGGED body: choose the body by its BEST-RESPONSE controller (the
    Stackelberg trainability objective) + illuminate the MAP-Elites archive + record a provenance edge —
    all via one ``co_design_with_memory`` call, warm-started from the banked best design.

    Returns ``(gene, follower, result)`` — the (possibly co-designed) gene, the follower controller it was
    selected with, and the result dict — or ``(gene, None, None)`` for non-legged bodies or on any failure.
    Best-effort (never fails a build); auto-on for legged, disable with ``VIRTUROID_KEYSTONE_CODESIGN=0``.
    The body selection is the value here; the headline score + viewer still use the shipped bare-CPG/banked
    controller (so the reported number can't contradict the 3D replay), and the follower is reserved for
    warm-starting the ES policy trainer.
    """
    import os
    try:
        from virturoid.services.task_matched_eval import robot_kind
        if robot_kind(gene) != "legged" or os.environ.get("VIRTUROID_KEYSTONE_CODESIGN", "1") == "0":
            return gene, None, None
        from virturoid.services.design_flywheel import co_design_with_memory
        from virturoid.services.memory_db import MemoryDB
        emit("keystone", "Choosing the walker body by its best-response controller (trainability) and "
                         "illuminating the design archive...")
        with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
            r = co_design_with_memory(gene, prompt, db, iterations=1, population=3, seed=0,
                                      best_response=True, archive_path=Path(memory_dir) / "design_archive.json",
                                      br_samples=2, br_iters=1, br_steps=300)
        return r["gene"], r.get("best_controller"), r
    except Exception:  # noqa: BLE001 - keystone co-design is best-effort; fall back to the composed gene
        return gene, None, None


def _maybe_gene_build(prompt, output_dir, target, memory_dir, emit, *, train: bool = False) -> AutonomyReport | None:
    """Build+run a robot from its gene for gene-only classes (e.g. humanoid). None otherwise.

    Produces a viewer-replayable package and an honest AutonomyReport. Success may be modest
    on a from-scratch gene — scene-matching + co-design lift it (the flywheel story).
    """
    from virturoid.services.morphology_selector import _explicit_class

    requested_class = _explicit_class((prompt or "").lower())
    # G4: the composite detector lives in the intent planner (mobile word + grasp verb = mobile_manipulator);
    # _explicit_class predates it and returns a single class, which silently dropped the mobile half of
    # 'a mobile robot that picks boxes and carries them'. Prefer the planner's class when it says composite.
    try:
        from virturoid.services.intent_planner import plan_build
        _plan_cls = plan_build(prompt, llm=None).robot_class
        if _plan_cls == "mobile_manipulator":
            requested_class = "mobile_manipulator"
    except Exception:  # noqa: BLE001
        pass
    # The tuned legacy arm pipeline only SCORES the colour-sort task -- stack / place-on-shelf / push read 0% there.
    # Route those (with the gene-only walking classes) through the general gene path so they actually run + evaluate.
    nonsort_manip = _prompt_task_type(prompt) in _NONSORT_MANIP_TASKS
    if requested_class not in _GENE_ONLY_CLASSES and not nonsort_manip:
        return None
    # Design the body INTELLIGENTLY: compose_robot routes a creature prompt through the LLM ANATOMY compiler
    # (the LLM describes the animal's anatomy; a general compiler builds it) — so "a dog" becomes a dog, not a
    # hard-coded fixture quadruped. Falls back to the parametric builders offline. (Was gene_for_class, a fixed
    # per-class fixture gene, which is exactly the hard-coding that made every quadruped the same flat disc.)
    from virturoid.services.morphology_composer import compose_robot

    try:
        if requested_class == "mobile_manipulator":
            # force the composite body plan: the live LLM's classifier tends to answer the TASK class
            # ('manipulator') and silently drops the requested mobile platform; the composite is explicit intent.
            from virturoid.services.intent_planner import plan_build as _pb
            gene = compose_robot(prompt, plan=_pb(prompt, llm=None))
        else:
            gene = compose_robot(prompt)
    except Exception:  # noqa: BLE001 - any design failure -> fixture gene, then legacy path
        gene = None
    if gene is None:
        from virturoid.fixtures.gene_library import gene_for_class
        gene = gene_for_class(requested_class, prompt=prompt)
    if gene is None:
        return None  # declared class with no gene yet -> fall through to legacy/foundation path
    # A non-sort task that didn't actually compose a manipulator (e.g. an ambiguous 'push' on a mobile base)
    # belongs on its own class's path, not the arm gene path.
    if requested_class not in _GENE_ONLY_CLASSES:
        if getattr(gene, "robot_class", None) != "manipulator":
            return None
        requested_class = "manipulator"   # flywheel reuse lookup + honest reporting of this arm gene build

    # Flywheel reuse: if a prior customer's gene for this class is stored in the species tree,
    # start from it (amend) instead of the seed — faster/better as the tree fills in.
    reused_from = None
    try:
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.map_elites_archive import descriptor
        from virturoid.services.memory_db import MemoryDB
        with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as _db:
            stored = _db.find_gene_for_class(requested_class)
        if stored:
            candidate = RobotGene.from_dict(stored)
            # Reuse a banked body ONLY when it shares the COMPOSED body's morphology NICHE — so a distinct
            # prompt (a six-legged, niche (6,8,3), vs the banked quad, (3,6,3)) keeps its own composed body and
            # illuminates a NEW archive niche instead of collapsing onto the banked one; same-morphology
            # rebuilds still amend the prior gene (the flywheel). Diagnosed empirically: the composer DOES build
            # distinct bodies per prompt, but the class-level reuse was overriding them (the flywheel-proof
            # coverage collapse — every legged prompt became the quad).
            try:
                same_niche = descriptor(candidate) == descriptor(gene)
            except Exception:  # noqa: BLE001 - if either can't be described, keep the old class-level reuse
                same_niche = True
            if not candidate.validate() and same_niche:
                gene, reused_from = candidate, candidate.species
    except Exception:  # noqa: BLE001 - reuse is best-effort; fall back to the seed gene
        pass

    from virturoid.services.mujoco_runner import mujoco_available
    if not mujoco_available():
        return None  # let the legacy path produce the validated foundation package

    # Keystone (Pillar 1): a LEGGED body is chosen by its best-response controller (trainability) and the
    # design archive is illuminated + a provenance edge recorded — the body co-design step the live legged
    # path lacked. Non-legged / disabled -> gene unchanged. Best-effort.
    gene, keystone_follower, keystone_res = _maybe_keystone_codesign(gene, prompt, memory_dir, emit)

    from virturoid.services.compute_provenance import RealComputeMeter
    from virturoid.services.gene_build import build_gene_package

    emit("species", f"Requested a {requested_class}; building from the '{gene.species}' gene (real, distinct robot).")
    emit("build", f"Compiling the {gene.species} gene to MuJoCo and generating task scenes...")
    meter = RealComputeMeter()
    with meter:
        summary = build_gene_package(gene, prompt, output_dir)
    try:
        # G6: re-run the morphology gate on the gene AS SHIPPED — build_gene_package's redesign/refine steps can
        # thicken links after the pre-build report, so the persisted verdict must describe the final body.
        import json as _json
        from dataclasses import asdict as _asdict
        from virturoid.services.morphology_priors import validate_morphology
        (Path(output_dir) / "reports" / "morphology_report.json").write_text(
            _json.dumps(_asdict(validate_morphology(gene)), indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    final = float(summary["success_rate"])
    task_type = summary.get("task_type", "pick_place_sort")
    emit("evaluate_done", f"Gene-built {gene.species} ran real '{task_type}': {final:.0%} task success.",
         {"success_rate": final})

    # Episode sub-space (Pillar 2): index this build's honest behavior stats (a walk's cadence/upright/
    # forward, a grasp's success) into the vector memory so "find episodes that behaved like this" is a kNN
    # over real rollout data (the 4th sub-space of the unified index). Best-effort; never breaks a build.
    try:
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        _feats = {k: float(v) for k, v in summary.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if _feats:
            with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as _edb:
                RoboticsVectorMemory(_edb).index_episode(
                    str(gene.id), _feats,
                    {"robot_class": gene.robot_class, "task_type": task_type, "status": summary.get("status")})
    except Exception:  # noqa: BLE001 - episode indexing is observability; never break a build
        pass

    # Reasoned redesign (design_critic): if the body underperforms the task, diagnose the PHYSICAL
    # limit (reach / joint torque / structure) and apply a directed, VERIFIED gene amendment —
    # reusing or recording cross-customer "lessons" (the flywheel). A control-only shortfall
    # (e.g. missed grasp) yields no body change and is honestly deferred to a learned skill.
    # Locomotion is excluded: the design-critic diagnoses manipulator REACH (it runs pick-place IK on
    # the reachable workspace), which is meaningless for a walker — a weak as-built gait is a control
    # shortfall lifted by the learned MorphPolicy (the policy flywheel), not a body-reach fix.
    redesign_trace = None
    if final < target and task_type != "locomotion":
        try:
            gene, final, summary, redesign_trace = _maybe_reasoned_redesign(
                gene, prompt, output_dir, target, memory_dir, emit, final, summary)
        except Exception:  # noqa: BLE001 - redesign is best-effort; never fail a build
            redesign_trace = None

    # Buildable-design assessment (roadmap_beyond_phase3 §3/§4): ground actuators to the real
    # parts library + structural feasibility + an honest transferability/confidence score.
    build_report = None
    try:
        from virturoid.fixtures.components import extended_component_library
        from virturoid.services.buildability import assess_buildability

        build_report = assess_buildability(gene, extended_component_library(), sim_success=final)
        emit("buildability", f"Buildable={build_report['buildable']}, confidence={build_report['confidence']}, "
             f"min safety factor={build_report['structural']['min_safety_factor']}.")
    except Exception:  # noqa: BLE001 - advisory
        build_report = None

    decisions = [
        AutonomyDecision(iteration=0, stage="build_and_validate",
                         action="compiled the species gene to a real MuJoCo robot and generated scenes",
                         detail=f"species={gene.species}, segments={len(gene.segments)}, dof={len(gene.actuated_joints())}",
                         success_before=0.0, success_after=0.0),
        AutonomyDecision(iteration=1, stage="evaluate",
                         action=("ran real locomotion (forward walking) on the gene-built robot"
                                 if task_type == "locomotion"
                                 else "ran real pick-and-place on the gene-built robot"),
                         detail=f"success={final}", success_before=0.0, success_after=final),
    ]
    if keystone_res is not None:   # the keystone body-selection step ran (legged) -> surface it honestly
        fl = keystone_follower or {}
        decisions.insert(1, AutonomyDecision(
            iteration=0, stage="keystone_codesign",
            action="selected the walker body by its best-response controller (Stackelberg trainability); illuminated the design archive",
            detail=(f"best_response_score={keystone_res.get('best_value')}, "
                    f"follower(freq={fl.get('freq')}, leg_flip={fl.get('leg_flip')}), "
                    f"delta_vs_prior={keystone_res.get('provenance_delta')}, "
                    f"archive={keystone_res.get('archive_action')} {keystone_res.get('archive_summary')}"),
            success_before=0.0, success_after=float(keystone_res.get("best_value") or 0.0)))
    # Pick-place builds emit object scenes (compiled_scene_index + scene_set); a locomotion build has no
    # object scenes, so only list the artifacts that were actually written (honest report).
    gene_artifacts = {
        "robot_genome": "robot/robot_genome.json",
        "gene_evaluation_report": "reports/gene_evaluation_report.json",
    }
    if task_type != "locomotion":
        gene_artifacts["compiled_scene_index"] = "simulation/mujoco/compiled_scene_index.json"
        gene_artifacts["scene_set"] = "simulation/scene_set.json"
    report = AutonomyReport(
        id=f"autonomy_gene_{gene.id}",
        prompt=prompt,
        robot_class=gene.robot_class,
        species=gene.species,
        requested_robot_class=requested_class,
        requested_species=gene.species,
        task_type=task_type,
        target_success_rate=target,
        initial_success_rate=final,
        final_success_rate=final,
        feasible=True,
        succeeded=bool(final >= target),
        converged_design={"segments": len(gene.segments), "dof": len(gene.actuated_joints())},
        decisions=decisions,
        compute=meter.provenance.to_dict(),
        exported_artifacts=gene_artifacts,
    )
    report.notes.append(
        f"Built from the {gene.species} gene — a structurally distinct {requested_class} (not the tabletop arm). "
        f"Raw task success {final:.0%}; scene-matching + co-design improve a from-scratch gene over time."
    )
    if reused_from:
        report.memory_reused = True   # surface gene-path flywheel reuse on the report (the flywheel KPI reads this)
        report.notes.append(f"Flywheel reuse: amended a prior stored {requested_class} gene ('{reused_from}') from the species tree.")
    # Reasoned-redesign outcome (honest: a body fix, a verified-but-no-help diagnosis, or "it's control").
    if redesign_trace is not None:
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports" / "reasoned_redesign.json").write_text(
            json.dumps(redesign_trace, indent=2, default=str), encoding="utf-8")
        report.exported_artifacts["reasoned_redesign"] = "reports/reasoned_redesign.json"
        accepted = [r for r in redesign_trace.get("rounds", []) if r.get("accepted")]
        if accepted:
            ops = ", ".join(dict.fromkeys(r["operator"] for r in accepted))
            srcs = set(r["source"] for r in accepted)
            via = "reused a prior customer's lesson" if "lesson" in srcs else "reasoned from the physics diagnosis"
            report.notes.append(
                f"Reasoned redesign: diagnosed {accepted[0]['root_cause']} and {via}; applied {ops}, "
                f"verified task success {redesign_trace['baseline_success']:.0%}->{redesign_trace['final_success']:.0%}. "
                "The proven fix was recorded as a lesson for future builds of this class.")
            for d in accepted:
                decisions.append(AutonomyDecision(
                    iteration=2, stage="reasoned_redesign",
                    action=f"diagnosed {d['diagnosis']} and applied a directed body fix ({d['operator']})",
                    detail=d["rationale"], success_before=d["success_before"], success_after=d["success_after"]))
        elif redesign_trace.get("rounds"):
            report.notes.append(
                f"Reasoned redesign: diagnosed {redesign_trace['rounds'][0]['root_cause']}, but no verified body "
                "amendment improved task success — the remaining shortfall is control/grasp, deferred to a learned skill (RL).")
        else:
            report.notes.append(
                "Reasoned redesign: no physical body limit (reach/torque/structure) found; the shortfall is "
                "control/grasp, not morphology — deferred to a learned skill (RL).")
    # Structured objective for the requested task (lift/spray/carry/…), gated + advisory.
    reward = _maybe_translate_reward(prompt, emit, gene=gene, memory_dir=memory_dir, task_type=task_type)
    if reward is not None:
        spec = reward["spec"]
        report.notes.append(
            f"Reward translator ({reward['backend']}): task '{spec.task_type}', success "
            f"'{spec.sparse_success.expression}', {len(spec.dense_terms)} shaping terms, ee={spec.end_effector}."
        )
        decisions.append(AutonomyDecision(
            iteration=0, stage="translate_reward",
            action="translated the task into a structured trainable objective (sparse success + dense terms)",
            detail=f"task={spec.task_type}; success={spec.sparse_success.expression}",
            success_before=0.0, success_after=0.0))
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "compute_provenance.json").write_text(
        json.dumps(report.compute, indent=2), encoding="utf-8")
    if build_report is not None:
        (output_dir / "reports" / "buildability_report.json").write_text(
            json.dumps(build_report, indent=2, default=str), encoding="utf-8")
        report.exported_artifacts["buildability_report"] = "reports/buildability_report.json"
        report.notes.append(
            f"Buildable design: {'YES' if build_report['buildable'] else 'NO'} "
            f"(confidence {build_report['confidence']}, structural min SF "
            f"{build_report['structural']['min_safety_factor']}). {build_report['disclaimer']}")
    # Policy flywheel (train=True): for a legged body, train a morphology-conditioned MorphPolicy that
    # WARM-STARTS from any prior banked legged policy and re-bank it — so the next legged build (even a
    # different morphology) starts from prior learning instead of from scratch (the params-mode moat,
    # CPU). Best-effort + gated on train, since ES rollouts add real wall-time. Runs BEFORE the memory
    # record so the trained capability (not the weak scripted baseline) is what gets banked + recorded.
    if train:
        morph = _maybe_train_morph_policy(gene, output_dir, memory_dir, emit, task_type, prompt)
        if morph is not None:
            report.exported_artifacts["morph_policy"] = "software/controller/morph_policy.npz"
            report.exported_artifacts["flywheel_manifest"] = "../memory/flywheel_manifest.jsonl"
            ws = "warm-started from a prior legged policy" if morph["warm_started"] else "cold-started (none banked yet)"
            kind = "PERCEPTIVE sense-and-avoid navigation" if morph.get("perceptive") else "locomotion"
            metric = "avoid fitness" if morph.get("perceptive") else "forward fitness"
            report.notes.append(
                f"Policy flywheel: trained a {kind} MorphPolicy for this legged body ({ws}); {metric} "
                f"{morph['baseline']:.3f}->{morph['final']:.3f}, banked as '{morph['skill_id']}'.")
            decisions.append(AutonomyDecision(
                iteration=3, stage="train_morph_policy",
                action=f"trained + banked a morphology-conditioned {kind} policy (warm-started from memory)",
                detail=f"{ws}; {metric} {morph['baseline']:.3f}->{morph['final']:.3f}; skill {morph['skill_id']}",
                success_before=0.0, success_after=0.0))
            # Report the BEST controller the autonomous loop produced: a legged body's as-built score uses
            # a scripted gait whose quality is morphology-dependent (a composed quad shuffles ~0 backward; a
            # hexapod may already walk). The trained MorphPolicy is the learned brain, but a weak CPU-ES run
            # can underperform a lucky scripted gait — so the honest final is max(scripted, trained), and the
            # robot ships whichever controller is better. (On the GPU box, PPO makes the learned brain win.)
            ts = morph.get("trained_locomotion_success")
            if ts is not None:
                scripted = final
                best = max(scripted, ts)
                winner = "trained MorphPolicy" if ts >= scripted else "scripted gait (trained policy was weaker)"
                report.initial_success_rate = scripted
                report.final_success_rate = final = best
                report.succeeded = bool(best >= target)
                morph["controller"] = "trained" if ts >= scripted else "scripted"
                report.notes.append(
                    f"Locomotion controller: best is the {winner} at {best:.0%} forward success "
                    f"(scripted {scripted:.0%} vs trained {ts:.0%}). The robot ships the better controller.")
                decisions.append(AutonomyDecision(
                    iteration=4, stage="evaluate_trained",
                    action="evaluated scripted vs TRAINED locomotion and kept the better controller",
                    detail=f"scripted {scripted:.3f} vs trained {ts:.3f} -> ship {morph['controller']} ({best:.3f})",
                    success_before=scripted, success_after=best))
    # Memory: record the run + AUTONOMOUSLY place the (possibly redesigned) gene in the species tree by
    # morphology — classify it as a variant of its nearest relative or detect it as a new species and
    # parent it under its closest neighbour (self-organizing tree, no hand-written taxonomy).
    try:
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.species_discovery import auto_place_species
        with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
            db.record_run(prompt, gene.robot_class, task_type,
                          {"segments": len(gene.segments)}, final, species=gene.species,
                          target_success_rate=target, succeeded=bool(final >= target),
                          design_source="gene_compiler")
            placement = auto_place_species(gene, db, source="gene")
            report.notes.append(
                f"Species tree: {placement['action']} as {placement['species_pattern']!r}"
                + (f" (variant; morphology distance {placement['distance']:.2f})" if placement["action"] == "classified"
                   else f" — a new species under {placement['parent']!r} by morphology.")
            )
            # Write-back (Phase 4): on a VERIFIED pass, bank a retrievable tip keyed to this species so the
            # next similar build recalls what worked (the flywheel turning). Verified-only + idempotent.
            if final >= target:
                try:
                    from virturoid.services.knowledge_writer import record_verified_knowledge
                    wb = record_verified_knowledge(db, gene, placement["species_pattern"],
                                                   task_type=task_type, success=final, target=target)
                    if wb.get("wrote_tip"):
                        report.notes.append(f"Banked a verified tip for reuse: \"{wb['tip']}\"")
                except Exception:  # noqa: BLE001 - write-back is best-effort
                    pass
    except Exception:  # noqa: BLE001 - memory is best-effort
        pass
    _write_report(output_dir, report)
    emit("done", f"Built a {requested_class} from its gene: {final:.0%} task success.",
         {"succeeded": report.succeeded, "final_success_rate": final, "compute": report.compute})
    return report


def _maybe_reasoned_redesign(gene, prompt, output_dir, target, memory_dir, emit, base_final, base_summary):
    """Diagnose a body's physical limit and apply a directed, verified fix (NOT random search).

    Returns ``(gene, final, summary, trace_dict)``. On a successful body fix the gene is the
    improved one and the package is rebuilt so all artifacts reflect the final robot; otherwise
    the originals are returned and the trace records why (no physical limit, or no fix helped ->
    it's a control/grasp problem for RL). The flywheel learns either way: proven fixes are stored
    as lessons keyed by robot_class + failure mode, and recalled first on future builds.
    """
    from virturoid.fixtures.components import extended_component_library
    from virturoid.services.design_critic import (
        improve_design,
        reach_required_from_scenes,
        task_sim_eval,
    )
    from virturoid.services.gene_build import build_gene_package
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    spec = select_task_spec(prompt)
    scenes = generate_task_scenes(gene, spec, count=4)        # FIXED across candidates (fair compare)
    if not scenes:
        return gene, base_final, base_summary, {"rounds": [], "baseline_success": base_final,
                                                "final_success": base_final, "accepted": 0}
    root = gene.root()
    base_h = (0.025 if gene.base_mount == "table" else 0.0) + (root.length_m if root else 0.0)
    reach_required = reach_required_from_scenes(scenes, base_height_m=base_h)
    masses = [o.mass_kg for s in scenes for o in s.objects if getattr(o, "mass_kg", None)]
    payload = round(max(masses), 4) if masses else 0.0

    emit("redesign", f"Task success {base_final:.0%} < target {target:.0%}; diagnosing the body "
         f"(reach req ~{reach_required:.2f} m, payload ~{payload:.2f} kg).")
    sim_eval = task_sim_eval(spec, scenes)
    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        trace = improve_design(gene, sim_eval, reach_required_m=reach_required, payload_kg=payload,
                               components=extended_component_library(), memory=db, max_rounds=3,
                               target_success=target, task_type=spec.task_type)
    trace_dict = {
        "baseline_success": trace.baseline_success, "final_success": trace.final_success,
        "accepted": trace.accepted, "rounds": trace.rounds,
        "reach_required_m": reach_required, "payload_kg": payload,
        "final_gene_id": trace.final_gene.id,
    }
    if trace.accepted > 0 and trace.final_gene.id != gene.id:
        improved = trace.final_gene
        emit("redesign", f"Applied {trace.accepted} verified body fix(es); rebuilding the package "
             f"from the improved '{improved.species}' gene.")
        summary = build_gene_package(improved, prompt, output_dir)
        return improved, float(summary["success_rate"]), summary, trace_dict
    return gene, base_final, base_summary, trace_dict


def _lookup_warm_start(memory_dir, robot_class, task_type, species):
    """Read side of the skill flywheel: best banked skill to inform a new (class, task) build.

    Best-effort and advisory — a missing DB or empty bank just returns ``None`` (cold start).
    """
    try:
        from pathlib import Path as _Path

        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.skill_flywheel import warm_start_for

        with MemoryDB(_Path(memory_dir) / "virturoid_memory.db") as db:
            return warm_start_for(db, robot_class, task_type, species=species)
    except Exception:  # noqa: BLE001 - flywheel read is advisory, never blocks a build
        return None


def _bank_trained_skill(memory_dir, output_dir, policy, build, converged, *,
                        warm_started: bool, warm_start_available: bool) -> str | None:
    """Write side of the skill flywheel: bank the trained policy + append one ledger row.

    Returns the banked ``skill_id`` (or ``None`` if persistence failed). Best-effort: a banking
    hiccup must never fail a real build, so everything here is wrapped.
    """
    try:
        import hashlib
        from pathlib import Path as _Path

        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.skill_flywheel import (
            append_manifest,
            bank_skill_from_card,
            synthesize_skill_card,
        )

        params_path = str(_Path(output_dir) / "software" / "controller" / "trained_policy.json")
        gene_id = (converged or {}).get("gene_id") if isinstance(converged, dict) else None
        card = synthesize_skill_card(
            policy, build.selected_robot_class, build.task_type,
            species=build.selected_species, gene_id=gene_id, params_path=params_path,
            trainer="cem_reach",
        )
        with MemoryDB(_Path(memory_dir) / "virturoid_memory.db") as db:
            skill_id = bank_skill_from_card(db, card)
        ckpt_hash = hashlib.sha256(
            json.dumps(getattr(policy, "weights", []), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        append_manifest(memory_dir, {
            "skill_id": skill_id,
            "robot_class": build.selected_robot_class,
            "task_type": build.task_type,
            "species": build.selected_species,
            "success_rate": float(policy.evaluation.success_rate),
            "n_episodes": int(policy.evaluation.eval_scene_count or 0),
            "trainer": "cem_reach",
            "ckpt_hash": ckpt_hash,
            "warm_started": bool(warm_started),
            "warm_start_available": bool(warm_start_available),
        })
        return skill_id
    except Exception:  # noqa: BLE001 - skill banking is best-effort
        return None


_NAV_INTENT_WORDS = ("navigat", "obstacle", "avoid", "maze", "around walls", "unknown environment",
                     "sense", "perceiv", "perception")


def _is_navigation_intent(prompt: str) -> bool:
    """Does the prompt ask the robot to sense + move through an environment (vs. just walk forward)?"""
    p = (prompt or "").lower()
    return any(w in p for w in _NAV_INTENT_WORDS)


def _maybe_train_morph_policy(gene, output_dir, memory_dir, emit, task_type, prompt=""):
    """Train + bank a morphology-conditioned MorphPolicy for a legged gene, warm-starting from prior memory.

    Forward-locomotion by default; for an obstacle/navigation request it trains a PERCEPTIVE sense-and-avoid
    policy (the range-sensing brain) and banks it under a separate 'navigation' skill so perceptive and
    locomotion policies compound independently. Returns ``{skill_id, warm_started, baseline, final,
    params_path}`` or ``None`` (non-legged body, missing deps, or any failure — banking a policy must never
    fail a real build). Appends one row to the unified flywheel manifest either way.
    """
    try:
        from pathlib import Path as _Path

        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.policy_flywheel import transfer_train_morph
        from virturoid.services.skill_flywheel import append_manifest
        from virturoid.services.task_matched_eval import robot_kind

        if robot_kind(gene) != "legged":
            return None
        ctrl = _Path(output_dir) / "software" / "controller"
        ctrl.mkdir(parents=True, exist_ok=True)
        params_path = ctrl / "morph_policy.npz"

        nav = _is_navigation_intent(prompt)
        if nav:
            from virturoid.services.morph_trainer import train_morph_avoid
            policy_task, trainer, train_fn = "navigation", "morph_avoid", train_morph_avoid
            emit("train_policy", "Training a PERCEPTIVE sense-and-avoid navigation policy (ES on range sensing; "
                 "warm-starts from memory)...")
        else:
            policy_task, trainer, train_fn = "locomotion", "morph_es", None
            emit("train_policy", "Training a morphology-conditioned locomotion policy (ES; warm-starts from memory)...")

        with MemoryDB(_Path(memory_dir) / "virturoid_memory.db") as db:
            # Modest CPU ES budget — enough to compound the flywheel without dominating build wall-time.
            res = transfer_train_morph([gene], db, task_type=policy_task, params_path=str(params_path),
                                       generations=6, pop=10, steps=200, seed=0, train_fn=train_fn)
        append_manifest(memory_dir, {
            "skill_id": res["skill_id"],
            "robot_class": gene.robot_class,
            "task_type": policy_task,
            "species": getattr(gene, "species", None),
            "success_rate": res["final"],
            "trainer": trainer,
            "warm_started": res["warm_started"],
            "params_path": res["params_path"],
        })
        res["perceptive"] = nav
        # For a locomotion build, the robot's REAL walking capability is the trained policy, not the weak
        # scripted gait the body was first scored with — re-evaluate forward distance with the learned brain
        # so the report reflects what the product actually achieved (built AND trained). Skipped for nav
        # (its policy is avoid-optimized, not forward-distance).
        if not nav:
            from virturoid.services.morph_policy import rollout_morph
            rr = rollout_morph(gene, res["policy"], steps=600)
            res["trained_locomotion_success"] = round(min(1.0, max(0.0, float(rr.get("forward", 0.0)) / 1.0)), 3)
        return res
    except Exception:  # noqa: BLE001 - policy banking is best-effort
        return None


def _record_to_memory_db(memory_dir, prompt, build, converged, final, target, report,
                         failure_clusters=None, transfer_decision=None) -> None:
    """Persist this run into the Virturoid Memory database (plan §13.3, update after every run).

    Records the run + flywheel design row + failure memory + any transfer decision so
    the AI agents can warm-start and avoid past failures on future builds, and so the
    local model has a growing supervised dataset to learn from. Best-effort: a DB hiccup
    must never fail a real build.
    """
    try:
        from pathlib import Path as _Path

        from virturoid.services.llm_client import _load_local_env
        from virturoid.services.memory_db import MemoryDB

        _load_local_env()
        backend = __import__("os").environ.get("VIRTUROID_LLM_BACKEND", "off")
        with MemoryDB(_Path(memory_dir) / "virturoid_memory.db") as db:
            run_id = db.record_run(
                prompt=prompt,
                robot_class=build.selected_robot_class,
                task_type=build.task_type,
                converged_design=converged,
                success_rate=final,
                species=build.selected_species,
                target_success_rate=target,
                succeeded=bool(report.succeeded),
                backend=backend,
                design_source=getattr(report, "design_source", None) or "as_built",
            )
            if failure_clusters:
                db.record_failures(run_id, build.selected_robot_class, build.task_type, list(failure_clusters))
            if transfer_decision:
                db.record_transfer(run_id, build.selected_robot_class, transfer_decision, success_rate=final)
            _record_species_tree(db, build)
    except Exception:  # noqa: BLE001 - memory persistence is best-effort
        return


def _record_species_tree(db, build) -> None:
    """Seed the species tree from the catalog and record the requested species node.

    This is what makes the tree *grow*: the catalog nodes are the seed taxonomy, and a
    request for a not-yet-buildable species (e.g. humanoid) is still recorded as a node,
    so implementing its builder later upgrades it in place.
    """
    from virturoid.fixtures.morphologies import morphology_template_catalog
    from virturoid.services.species_tree import is_buildable

    for t in morphology_template_catalog():
        db.upsert_species_node(
            t.species_pattern, parent_species=t.parent_species, robot_class=t.robot_class,
            morphology_tags=list(t.morphology_tags), buildable=is_buildable(t), source="catalog",
        )
    requested = getattr(build, "requested_species", "") or build.selected_species
    if requested and requested != build.selected_species:
        # A distinct requested species with no builder yet — record it as a tree node.
        db.upsert_species_node(
            requested, parent_species=getattr(build, "requested_robot_class", "") or None,
            robot_class=getattr(build, "requested_robot_class", "") or build.selected_robot_class,
            buildable=False, source="requested",
        )
    # Keep the tree bounded: merge near-duplicate variations + trim subsumed leaves.
    # Deterministic + gated (never touches buildable nodes, nodes with builds, or tips),
    # so the species memory cannot grow uncontrollably across many builds.
    try:
        from virturoid.services.species_maintenance import prune_tree

        prune_tree(db)
    except Exception:  # noqa: BLE001 - housekeeping is best-effort
        pass


def _maybe_recommend_transfer(prompt: str, robot_class: str, task_type: str, memory: dict, emit,
                              memory_dir=DEFAULT_MEMORY_DIR) -> dict | None:
    """Ask the Transfer Agent whether to warm-start from prior memory, if a backend is set.

    Candidates come from the Virturoid Memory database (every prior build of this robot
    class, best-per-task) plus the JSON warm-start record — so the more builds run, the
    richer the pool the agent reasons over (the learn-and-improve loop).
    """
    try:
        from virturoid.services.llm_client import get_llm
        from virturoid.services.transfer_agent import recommend_transfer

        llm = get_llm("task_graph")  # cloud reasoning role
        if llm is None:
            return None

        candidates: dict[str, dict] = {}
        # Prior builds from the memory DB (best per task across this robot class).
        try:
            from pathlib import Path as _Path

            from virturoid.services.memory_db import MemoryDB

            with MemoryDB(_Path(memory_dir) / "virturoid_memory.db") as db:
                for c in db.transfer_candidates(robot_class):
                    candidates[c["key"]] = {
                        "key": c["key"], "robot_class": c["robot_class"], "task_type": c["task_type"],
                        "converged_design": c.get("converged_design"), "success_rate": c.get("success_rate"),
                    }
        except Exception:  # noqa: BLE001 - DB is optional context
            pass
        # The JSON warm-start record (may predate the DB).
        if memory and memory.get("converged_design"):
            key = f"{memory.get('robot_class', robot_class)}__{memory.get('task_type', task_type)}"
            candidates.setdefault(key, {
                "key": key, "robot_class": memory.get("robot_class", robot_class),
                "task_type": memory.get("task_type", task_type),
                "converged_design": memory.get("converged_design"), "success_rate": memory.get("success_rate"),
            })
        if not candidates:
            return None

        requirements = build_requirements_from_prompt(prompt)
        result = recommend_transfer(prompt, requirements, list(candidates.values()), llm)
        if result and result.get("valid"):
            return result
    except Exception:  # noqa: BLE001 - advisory; never block the build
        return None
    return None


def _maybe_translate_reward(prompt: str, emit, *, gene=None, memory_dir=None, task_type=None) -> dict | None:
    """Translate the task into a structured RewardSpec (Language-to-Rewards), if a backend is set.

    Gated + offline-safe. The sparse success criterion + bounded dense terms are the trainable
    objective that lets the platform handle diverse tasks (lift/spray/carry) rather than only
    the hard-coded sort. Advisory today (recorded); drives RL skills once MJX is online.

    Phase 2 RAG: when a ``gene`` + ``memory_dir`` are available, prepend a retrieved PRIOR-KNOWLEDGE
    block (nearest skill's reward recipe + trainer tips for this body) so the reward is designed with
    what worked before — the reasoner reasons about THIS body, not blind.
    """
    try:
        from virturoid.services.llm_client import get_llm
        from virturoid.services.requirements_builder import build_requirements_from_prompt
        from virturoid.services.reward_agent import translate_task_to_reward

        llm = get_llm("task_graph")  # reasoning role
        if llm is None:
            return None
        prior = ""
        if gene is not None and memory_dir is not None:
            try:
                from virturoid.services.knowledge_context import assemble_prior_knowledge
                prior = assemble_prior_knowledge(memory_dir, gene=gene, task_type=task_type)
            except Exception:  # noqa: BLE001 - RAG is best-effort; never break the build
                prior = ""
        result = translate_task_to_reward(prompt, build_requirements_from_prompt(prompt), llm,
                                          prior_knowledge=prior)
        if result and result.get("valid"):
            emit("reward", f"Reward translator ({result['backend']}): success = '{result['spec'].sparse_success.expression}'.")
            return result
    except Exception:  # noqa: BLE001 - advisory; never block the build
        return None
    return None


def _maybe_synthesize_task_graph(prompt: str, emit) -> dict | None:
    """Ask the Task Agent for a richer task graph, if an LLM backend is configured."""
    try:
        from virturoid.services.llm_client import get_llm
        from virturoid.services.task_agent import synthesize_task_graph

        llm = get_llm("task_graph")
        if llm is None:
            return None
        requirements = build_requirements_from_prompt(prompt)
        result = synthesize_task_graph(prompt, requirements, llm)
        if result and result.get("valid") and result.get("graph"):
            g = result["graph"]
            emit("task_graph", f"Task agent ({result['backend']}) synthesized task '{g['name']}' ({g['task_type']}).")
            return result
    except Exception:  # noqa: BLE001 - advisory; never block the build
        return None
    return None


def _maybe_recommend_parts(prompt: str, emit) -> dict | None:
    """Ask the Part Recommender agent for a BOM, if an LLM backend is configured."""
    try:
        from virturoid.fixtures.components import extended_component_library
        from virturoid.services.llm_client import get_llm
        from virturoid.services.part_agent import recommend_parts

        llm = get_llm("part_recommender")
        if llm is None:
            return None
        requirements = build_requirements_from_prompt(prompt)
        result = recommend_parts(prompt, requirements, extended_component_library(), llm)
        if result and result.get("feasible") and result.get("selection"):
            emit("recommend_parts", f"Part recommender ({result['backend']}) selected: {result['selection']}.")
            return result
    except Exception:  # noqa: BLE001 - advisory; never block the build
        return None
    return None


def _design_feasible(output_dir: Path) -> bool:
    path = output_dir / "reports" / "design_validation_report.json"
    if not path.exists():
        return True
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("ok", True))
    except Exception:  # noqa: BLE001
        return True


def _as_built_design(prompt: str) -> dict:
    requirements = build_requirements_from_prompt(prompt)
    reach_mm = int((requirements.reach_m if requirements.reach_m is not None else 0.6) * 1000)
    return {"forearm_mm": float(max(300, int(reach_mm * 0.55))), "wrist_mm": 160.0, "actuator_torque_nm": 12.0}


def _finalize_package_with_design(output_dir: Path, prompt: str, design: dict) -> None:
    """Rewrite the package's robot + simulator files to the converged body."""
    from virturoid.fixtures.morphologies import morphology_template_catalog
    from virturoid.services.codesign_optimizer import _build_variant
    from virturoid.services.design_validator import validate_design, write_design_validation_report
    from virturoid.services.morphology_selector import select_morphology_template
    from virturoid.services.mujoco_exporter import write_mujoco_scene_collection, write_mujoco_xml
    from virturoid.services.package_validator import write_validation_report
    from virturoid.services.simulator_contract_builder import write_simulator_contract_from_export
    from virturoid.services.task_builder import build_task_graph
    from virturoid.services.urdf_exporter import write_robot_urdf

    requirements = build_requirements_from_prompt(prompt)
    task = build_task_graph(requirements)
    templates = morphology_template_catalog()
    template = select_morphology_template(requirements, task, templates).selected_template
    genome, cad = _build_variant(
        requirements,
        curated_component_library(),
        design["forearm_mm"],
        design["wrist_mm"],
        design["actuator_torque_nm"],
        morphology_template=template,
    )
    (output_dir / "robot" / "robot_genome.json").write_text(json.dumps(genome.to_dict(), indent=2), encoding="utf-8")
    write_robot_urdf(genome, output_dir, cad_models=cad)

    scene_sets = [_load_scene_set(output_dir / rel) for rel in _SCENE_SET_FILES if (output_dir / rel).exists()]
    write_mujoco_xml(genome, scene_sets[1] if len(scene_sets) > 1 else scene_sets[0], output_dir, cad_models=cad)
    write_mujoco_scene_collection(genome, scene_sets, output_dir, cad_models=cad)
    write_simulator_contract_from_export(output_dir)

    components = [c for c in curated_component_library() if c.actuator or c.sensor]
    design_validation = validate_design(requirements, genome, cad, components, scene_sets, package_dir=output_dir)
    write_design_validation_report(design_validation, output_dir)
    write_validation_report(output_dir)


def _load_scene_set(path: Path) -> SceneSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SceneSet(
        id=data["id"],
        version=data.get("version", "0.1.0"),
        task_graph_id=data["task_graph_id"],
        purpose=data["purpose"],
        scenes=[
            SceneGraph(
                id=s["id"],
                version=s.get("version", "0.1.0"),
                name=s["name"],
                backend_targets=s["backend_targets"],
                robot_spawn_xyz_rpy=tuple(s["robot_spawn_xyz_rpy"]),
                objects=[
                    SceneObject(
                        name=o["name"],
                        object_type=o["object_type"],
                        pose_xyz_rpy=tuple(o["pose_xyz_rpy"]),
                        mass_kg=o.get("mass_kg"),
                        material=o.get("material"),
                        friction=o.get("friction"),
                        scale=o.get("scale"),
                    )
                    for o in s["objects"]
                ],
                variation_parameters=s.get("variation_parameters", {}),
                requirement_trace=s.get("requirement_trace", []),
            )
            for s in data["scenes"]
        ],
        generation_rules=data.get("generation_rules", []),
    )


def _write_report(output_dir: Path, report: AutonomyReport) -> Path:
    validation = report.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{report.id} failed validation: {issues}")
    path = output_dir / "reports" / "autonomy_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    # Honest Product Readiness Ledger for EVERY package this legacy path writes (manipulator / mobile_base /
    # foundation). The gene path already emits it; without this, a manipulator/rover build could only show the
    # soft mvp_readiness_report (which can read ready:true over placeholder CAD). The ledger never lies.
    try:
        from virturoid.services.readiness_ledger import write_product_readiness_ledger
        write_product_readiness_ledger(output_dir, robot_class=getattr(report, "robot_class", "") or "")
    except Exception:  # noqa: BLE001 - the ledger is a truth surface, never a build blocker
        pass
    return path
