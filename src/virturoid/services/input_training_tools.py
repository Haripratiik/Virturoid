"""Agent tools for the input-ingestion + training-improvement plans (W2 / I4 / T-E).

The agent (Claude/Codex via MCP) is the brain, so these plan capabilities must be callable, not just importable.
This module registers them into the same ``TOOLS`` registry as the rest (``agent_tools`` does ``TOOLS.update``):

  * interpret_prompt        — Input Compiler Phase 0: provenance-tracked interpretation of a prompt
  * inspect_project_bundle  — Phase 2: classify a dropped folder/zip -> Project Graph dashboard
  * import_robot_model      — Phase 1: faithful + inferred-RobotGene import report for a model file
  * plan_training           — dossier Training Brain: the deterministic three-phase ladder for a task
  * check_perception_leakage— Training Improvement Phase 0: privileged-state leakage gate + perception rung
  * amplify_demonstrations  — dossier Bet 1: turn a walking body into many physics-validated gait demos
  * data_dividends          — the flywheel "what did we improve?" ledger summary

Handlers take ``args: dict`` and return JSON-able dicts (errors as ``{"error": ...}``), matching the registry.
"""

from __future__ import annotations

import os


def _interpret_prompt(args: dict) -> dict:
    from virturoid.services.input_evidence import interpret_prompt
    prompt = (args or {}).get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    interp = interpret_prompt(
        prompt,
        payload_kg=args.get("payload_kg"), reach_m=args.get("reach_m"), sensor=args.get("sensor"))
    return interp.to_dict()


def _inspect_project_bundle(args: dict) -> dict:
    from virturoid.services.input_classifier import project_graph_summary, scan_folder, scan_zip
    path = (args or {}).get("path", "")
    if not path:
        return {"error": "path is required (a project folder or a .zip)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    max_files = int(args.get("max_files", 20000))
    try:
        bundle = (scan_zip(path, max_files=max_files) if path.lower().endswith(".zip")
                  else scan_folder(path, max_files=max_files))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not scan project: {exc}"}
    return project_graph_summary(bundle)


def _import_robot_model(args: dict) -> dict:
    from virturoid.services.robot_import_report import build_import_report
    path = (args or {}).get("path", "")
    if not path:
        return {"error": "path is required (a .urdf/.mjcf/.xml model file)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    report = build_import_report(path, robot_id=args.get("robot_id"), species=args.get("species"))
    return report.to_dict()


def _plan_training(args: dict) -> dict:
    from virturoid.services.training_ladder import ladder_report, plan_training
    task = (args or {}).get("task", "")
    if not task:
        return {"error": "task is required (a task family or free-text task)"}
    plan = plan_training(
        task,
        robot_genome_id=args.get("robot_genome_id", "unspecified_body"),
        task_graph_id=args.get("task_graph_id", "unspecified_task"),
        scene_set_refs=args.get("scene_set_refs"),
        gpu_available=bool(args.get("gpu_available", True)),
        deployable=bool(args.get("deployable", True)))
    return ladder_report(plan)


def _check_perception_leakage(args: dict) -> dict:
    from virturoid.schemas.observation_contract import ObservationContract, PerceptionRung
    from virturoid.services.perception_leakage import training_plan_report
    args = args or {}
    rung = args.get("perception_rung", PerceptionRung.RUNG_0_PRIVILEGED.value)
    try:
        rung_enum = PerceptionRung(rung)
    except ValueError:
        return {"error": f"unknown perception_rung '{rung}'"}
    contract = ObservationContract(
        id=args.get("id", "contract_adhoc"),
        task_graph_id=args.get("task_graph_id", "task"),
        scene_set_id=args.get("scene_set_id", "scenes"),
        robot_genome_id=args.get("robot_genome_id", "body"),
        policy_observation_keys=list(args.get("policy_observation_keys", [])),
        privileged_label_keys=list(args.get("privileged_label_keys", [])),
        required_modalities=list(args.get("required_modalities", [])),
        deploy_modalities=list(args.get("deploy_modalities", [])),
        leakage_policy=args.get("leakage_policy", "strict"),
        perception_rung=rung_enum,
        train_scene_seeds=list(args.get("train_scene_seeds", [])),
        heldout_scene_seeds=list(args.get("heldout_scene_seeds", [])),
        randomization_logged=bool(args.get("randomization_logged", False)))
    return training_plan_report(contract)


def _amplify_demonstrations(args: dict) -> dict:
    from virturoid.services.demonstration_amplifier import amplify_gait
    args = args or {}
    prompt = args.get("prompt", "a quadruped robot dog")
    res = amplify_gait(
        prompt=prompt,
        n_variants=int(args.get("n_variants", 6)),
        base_freq=float(args.get("base_freq", 1.4)),
        steps=int(args.get("steps", 1200)),
        seed=int(args.get("seed", 0)))
    return res.report()


def _data_dividends(args: dict) -> dict:
    from virturoid.services.data_dividend import dividend_summary
    memory_dir = (args or {}).get("memory_dir")
    if memory_dir:
        return dividend_summary(memory_dir=memory_dir)
    return dividend_summary()


INPUT_TRAINING_TOOLS: dict[str, dict] = {
    "interpret_prompt": {
        "description": "Input Compiler (Phase 0): parse a build prompt into a provenance-tracked interpretation — "
                       "every field tagged explicit/parsed/inferred/defaulted, with intake questions for genuine "
                       "conflicts (e.g. a weight budget mis-read as a carry payload). No physics.",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}, "payload_kg": {"type": "number"},
            "reach_m": {"type": "number"}, "sensor": {"type": "string"}}},
        "handler": _interpret_prompt, "heavy": False,
    },
    "inspect_project_bundle": {
        "description": "Classify a dropped enterprise project FOLDER or .zip into a Project Graph: recognized vs "
                       "unrecognized files by category (models/meshes/cad/ros/policies/logs/bom), checksums, the "
                       "first runnable sim target, and blockers. Local-only, metadata only (no file contents).",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a project folder or a .zip"},
            "max_files": {"type": "integer", "default": 20000}}},
        "handler": _inspect_project_bundle, "heavy": False,
    },
    "import_robot_model": {
        "description": "Enterprise import report for a .urdf/.mjcf/.xml model: runs the FAITHFUL native lane "
                       "(compiles as-is) and the inferred RobotGene lane (editable, lossy) side by side, with every "
                       "warning classified into a concrete fix, plus per-axis readiness scores. Real MuJoCo compile.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string"}, "robot_id": {"type": "string"}, "species": {"type": "string"}}},
        "handler": _import_robot_model, "heavy": False,
    },
    "plan_training": {
        "description": "The Training Brain: compile a task into a deterministic three-phase ladder (cheap "
                       "reuse -> distill/adapt -> expensive RL) with teacher sources, skill decomposition, backend "
                       "budget, deploy-sim checkpoint selection, and evaluator-certified banking rules. No training.",
        "parameters": {"type": "object", "required": ["task"], "properties": {
            "task": {"type": "string", "description": "a task family or free-text task"},
            "robot_genome_id": {"type": "string"}, "task_graph_id": {"type": "string"},
            "gpu_available": {"type": "boolean", "default": True},
            "deployable": {"type": "boolean", "default": True}}},
        "handler": _plan_training, "heavy": False,
    },
    "check_perception_leakage": {
        "description": "Privileged-state leakage gate + perception-rung report for an observation contract: rejects "
                       "a deployable plan whose policy cheats with simulator truth (object poses, goal truth, "
                       "segmentation the robot won't have) or whose held-out seeds overlap training. No training.",
        "parameters": {"type": "object", "properties": {
            "policy_observation_keys": {"type": "array", "items": {"type": "string"}},
            "privileged_label_keys": {"type": "array", "items": {"type": "string"}},
            "required_modalities": {"type": "array", "items": {"type": "string"}},
            "deploy_modalities": {"type": "array", "items": {"type": "string"}},
            "train_scene_seeds": {"type": "array", "items": {"type": "integer"}},
            "heldout_scene_seeds": {"type": "array", "items": {"type": "integer"}},
            "leakage_policy": {"type": "string", "enum": ["strict", "permissive"]},
            "randomization_logged": {"type": "boolean"}}},
        "handler": _check_perception_leakage, "heavy": False,
    },
    "amplify_demonstrations": {
        "description": "Demonstration amplifier (the highest-leverage training lever): turn ONE walking body into "
                       "many cadence-varied, PHYSICS-VALIDATED gait demonstrations, keeping only variants that walk "
                       "forward + upright + survive. Returns the measured demo_amplification_yield. Real MuJoCo, slow.",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string", "default": "a quadruped robot dog"},
            "n_variants": {"type": "integer", "default": 6},
            "base_freq": {"type": "number", "default": 1.4}, "seed": {"type": "integer", "default": 0}}},
        "handler": _amplify_demonstrations, "heavy": True,
    },
    "data_dividends": {
        "description": "The flywheel ledger summary: across every run, which reusable priors (skill/reward/body/"
                       "sensor/failure-repair/...) were improved, how many became reusable by default, and the "
                       "reuse-conversion rate — the 'software gets better as people use it' moat view.",
        "parameters": {"type": "object", "properties": {
            "memory_dir": {"type": "string", "description": "shared memory dir (default build/memory)"}}},
        "handler": _data_dividends, "heavy": False,
    },
}
