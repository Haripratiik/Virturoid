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
    report = res.report()
    if args.get("bank_dividend") and res.kept > 0:              # close the flywheel: record what this run improved
        from virturoid.services.data_dividend import compute_dividend, record_dividend
        rec = compute_dividend(
            run_id=f"amp_{prompt[:24]}", improved_prior_type="demonstration_dataset",
            improved_prior_ref=f"gait_demos::{prompt[:40]}",
            before_metrics={"validated_demos": 1}, after_metrics={"validated_demos": res.kept},
            key_metric="validated_demos", evidence_refs=[v.variant_id for v in res.lineage if v.accepted])
        path = (record_dividend(rec, memory_dir=args["memory_dir"]) if args.get("memory_dir")
                else record_dividend(rec))
        report["data_dividend"] = {"reusable_by_default": rec.reusable_by_default,
                                   "measured_delta": rec.measured_delta, "ledger": path}
    return report


def _import_bom(args: dict) -> dict:
    from virturoid.services.bom_importer import parse_bom_file
    path = (args or {}).get("path", "")
    if not path:
        return {"error": "path is required (a .csv/.json/.yaml/.xlsx BOM)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    return parse_bom_file(path).to_dict()


def _sandbox_policy(args: dict) -> dict:
    from virturoid.services.policy_importer import static_parse_python
    from virturoid.services.policy_sandbox import sandbox_policy_step
    args = args or {}
    source = args.get("source")
    if not source:
        return {"error": "source (an inline Python controller defining e.g. def act(obs): ...) is required"}
    entrypoint = args.get("entrypoint")
    action_dim = args.get("action_dim")
    if not entrypoint or action_dim is None:                   # recover them via the P1 static parse
        spec = static_parse_python(source)
        entrypoint = entrypoint or spec.entrypoint
        action_dim = action_dim if action_dim is not None else spec.action_dim
    if not entrypoint:
        return {"error": "no callable entrypoint found; pass 'entrypoint'"}
    return sandbox_policy_step(
        source, entrypoint=entrypoint, observation=args.get("observation", []),
        action_dim=action_dim, safety_limits=args.get("safety_limits"),
        timeout=float(args.get("timeout", 10.0)))


def _import_onnx_policy(args: dict) -> dict:
    from virturoid.services.policy_native_adapter import inspect_onnx, run_onnx_policy
    args = args or {}
    path = args.get("path", "")
    if not path:
        return {"error": "path is required (a .onnx policy file)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    if args.get("observation") is None:                        # no obs -> just inspect the IO contract
        return inspect_onnx(path)
    return run_onnx_policy(path, args["observation"], action_dim=args.get("action_dim"),
                           safety_limits=args.get("safety_limits"))


def _import_controller_interface(args: dict) -> dict:
    from virturoid.services.ros2_control_parser import controller_interface_from_ros2_control, parse_ros2_control
    args = args or {}
    xml = args.get("xml")
    path = args.get("path")
    if not xml and path:
        if not os.path.exists(path):
            return {"error": f"path not found: {path}"}
        with open(path, encoding="utf-8") as handle:
            xml = handle.read()
    if not xml:
        return {"error": "provide 'path' to a URDF/xacro with a <ros2_control> block, or inline 'xml'"}
    ctrl_yaml = None
    ypath = args.get("controller_yaml_path")
    if ypath and os.path.exists(ypath):
        with open(ypath, encoding="utf-8") as handle:
            ctrl_yaml = handle.read()
    elif args.get("controller_yaml"):
        ctrl_yaml = args["controller_yaml"]
    parsed = parse_ros2_control(xml)
    spec = controller_interface_from_ros2_control(xml, controller_yaml=ctrl_yaml)
    return {"interface": spec.to_dict(), "hardware_plugins": parsed["hardware_plugins"],
            "sensors": parsed["sensors"], "warnings": parsed["warnings"]}


def _import_cad(args: dict) -> dict:
    from virturoid.services.cad_importer import import_cad
    path = (args or {}).get("path", "")
    if not path:
        return {"error": "path is required (a .stl/.obj mesh; .step is deferred)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    return import_cad(path, material=args.get("material", "abs")).to_dict()


def _ingest_project(args: dict) -> dict:
    """INGESTION AGENT (Part B): a robotics team drops a project FOLDER/ZIP of their existing robot (URDF/MJCF +
    optional BOM/CAD) plus an NLP description ("aluminum body, carbon-fiber legs, 5 kg payload, 6-DOF arm"), and
    gets back ONE unified, immediately-editable RobotGene held in the session -- the user's stated materials +
    load already applied, with a project graph + BOM/CAD summary + honest per-step warnings. It ORCHESTRATES the
    importers already built (classifier, robot_import, bom_importer, cad_importer) + nlp_properties + edit_operators."""
    import shutil

    args = args or {}
    path = args.get("project_path") or args.get("path")
    description = (args.get("description") or args.get("nlp") or "").strip()
    if not path and not description:
        return {"error": "provide project_path (a folder or .zip of the robot's files) and/or description (NLP)"}

    result: dict = {"robot_id": None, "materials_applied": [], "payload_kg": None,
                    "applied_ops": [], "skipped_ops": [], "warnings": [], "notes": []}
    workdir: str | None = None
    scan_root: str | None = None
    bundle = None

    # 1) scan the project into a Project Graph (extract a zip to a temp dir so its model is importable)
    if path:
        if not os.path.exists(path):
            return {"error": f"path not found: {path}"}
        from virturoid.services.input_classifier import project_graph_summary, scan_folder
        try:
            if path.lower().endswith(".zip"):
                import tempfile
                import zipfile
                workdir = tempfile.mkdtemp(prefix="ingest_")
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(workdir)
                scan_root = workdir
            else:
                scan_root = os.path.abspath(path)
            bundle = scan_folder(scan_root)
            result["project_graph"] = project_graph_summary(bundle)
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"project scan failed: {exc}")

    # 2) import the robot model -> the inferred, EDITABLE RobotGene (the twin we amend)
    gene = None
    model_rel = (result.get("project_graph") or {}).get("first_runnable_sim_target")
    if model_rel and scan_root:
        model_path = os.path.join(scan_root, model_rel.replace("/", os.sep))
        try:
            from virturoid.services.robot_import import import_robot
            imp = import_robot(model_path, robot_id=args.get("robot_id"))
            gene = imp.get("gene")
            result["import"] = {"source": model_rel, "robot_class": imp.get("robot_class"),
                                "species": imp.get("species"), "valid": imp.get("valid"),
                                "warnings": list(imp.get("warnings", []))[:8]}
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"model import failed ({model_rel}): {exc}")

    # 3) parse the NLP description into typed, provenance-tagged properties + edit ops
    from virturoid.services.nlp_properties import extract_properties
    props = extract_properties(description)
    result["nlp"] = props.to_dict()
    result["notes"].extend(props.notes)

    # fallback: no importable model but a description -> compose an editable robot from the words (still ingest-able)
    if gene is None and description:
        try:
            from virturoid.services.morphology_composer import compose_robot
            gene = compose_robot(description)
            result["notes"].append("no importable robot model found -> composed an editable robot from the description")
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"could not compose a robot from the description: {exc}")
    if gene is None:
        result["warnings"].append("no robot could be produced (no importable model and no usable description)")
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        return result

    # 4) apply the stated materials (only to parts that EXIST -- no fabrication) + payload, one gate per op
    from virturoid.services.edit_operators import apply_op, segments_for_group
    try:
        from virturoid.services.grounded_physics import ground_gene
        ground_gene(gene)                                        # real baseline torques/mass so set_payload is grounded
    except Exception:  # noqa: BLE001
        pass
    for op in props.ops:
        name, a = op["op"], dict(op.get("args") or {})
        if name == "set_material" and a.get("group") != "all":
            try:
                if not segments_for_group(gene, a["group"]):
                    result["skipped_ops"].append({**op, "reason": f"this robot has no '{a['group']}' part"})
                    continue
            except Exception:  # noqa: BLE001
                pass
        try:
            gene, diff = apply_op(gene, name, a)
            result["applied_ops"].append({"op": name, "args": a})
            if name == "set_material":
                result["materials_applied"].append({"group": a["group"], "material": a["material"]})
            elif name == "set_payload":
                result["payload_kg"] = a.get("payload_kg")
                if diff.get("warning"):
                    result["warnings"].append(diff["warning"])
        except Exception as exc:  # noqa: BLE001 - a bad op (incl. EditError) is skipped + noted, never aborts ingest
            result["skipped_ops"].append({**op, "reason": str(exc)})

    # 5) hold the unified robot in the session so it's immediately editable / verifiable
    from virturoid.services import session_state as S
    result["robot_id"] = S.put_robot(gene, prompt=(description[:120] or "ingested robot"),
                                     label="ingested", robot_id=args.get("robot_id"))

    # 6) fold in the BOM + CAD the user shipped (summaries; provenance for reconciliation)
    if bundle is not None:
        arts = getattr(bundle, "artifacts", [])
        bom_ref = next((x.extracted_refs[0] for x in arts
                        if getattr(x, "media_type", "") == "bom" and x.extracted_refs), None)
        cad_ref = next((x.extracted_refs[0] for x in arts
                        if getattr(x, "media_type", "") == "cad" and x.extracted_refs), None)
        # a user's OWN control scripts / policies -> surface them so the agent can run+improve via adopt_control_script
        ctl_refs = [x.extracted_refs[0] for x in arts
                    if getattr(x, "media_type", "") in ("controller", "policy") and x.extracted_refs]
        if ctl_refs and scan_root:
            result["control_scripts"] = [os.path.join(scan_root, r.replace("/", os.sep)) for r in ctl_refs][:8]
            result["notes"].append(f"found {len(ctl_refs)} control/policy file(s) -> call adopt_control_script "
                                   "{robot_id, script_path} to RUN them in sim and IMPROVE them")
        if bom_ref and scan_root:
            try:
                from virturoid.services.bom_importer import parse_bom_file
                b = parse_bom_file(os.path.join(scan_root, bom_ref.replace("/", os.sep))).to_dict()
                result["bom"] = {"source": bom_ref, "line_items": b.get("line_item_count") or len(b.get("items", [])),
                                 "total_mass_kg": b.get("total_mass_kg")}
            except Exception as exc:  # noqa: BLE001
                result["warnings"].append(f"BOM import failed ({bom_ref}): {exc}")
        if cad_ref and scan_root:
            try:
                from virturoid.services.cad_importer import import_cad
                mat = (result["materials_applied"][0]["material"].split("_")[0]
                       if result["materials_applied"] else "abs")
                c = import_cad(os.path.join(scan_root, cad_ref.replace("/", os.sep)),
                               material=mat if mat in ("abs", "pla", "aluminum", "steel", "nylon") else "abs").to_dict()
                result["cad"] = {"source": cad_ref, "dimensions_m": c.get("dimensions_m") or c.get("bbox_m"),
                                 "est_mass_kg": c.get("est_mass_kg") or c.get("mass_kg")}
            except Exception as exc:  # noqa: BLE001
                result["warnings"].append(f"CAD import failed ({cad_ref}): {exc}")

    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    result["summary"] = (f"Ingested -> {result['robot_id']}: "
                         f"{len(result['materials_applied'])} material(s) applied, "
                         f"payload={result['payload_kg']} kg, {len(result['warnings'])} warning(s). Ready to edit/verify.")
    return result


def _adopt_control_script(args: dict) -> dict:
    """A user's OWN control script / policy: RUN it on a held robot in real physics, then IMPROVE it (gait search
    warm-started from their params). Accepts a control-script .py path, a policy-params .json path, or inline
    params; statically parses the .py (safety/entrypoint) and reads its params where present. Returns measured
    before/after + the improved params + an honest verdict (keeps the user's controller if tuning can't beat it)."""
    import json

    args = args or {}
    rid = args.get("robot_id")
    if not rid:
        return {"error": "robot_id is required (the held robot to run/improve the controller on)"}
    from virturoid.services import session_state as _S
    gene = _S.get_robot(rid)
    if gene is None:
        return {"error": f"no held robot {rid}"}

    params = args.get("params") if isinstance(args.get("params"), dict) else None
    script_meta = None
    ppath = args.get("params_path") or args.get("script_path")
    if params is None and not ppath:
        return {"error": "provide params (inline dict), params_path (a policy_params.json), or script_path (a .py controller)"}
    if ppath:
        if not os.path.exists(ppath):
            return {"error": f"path not found: {ppath}"}
        # a .py controller -> statically parse it (entrypoint/deps/safety) and look for a sibling params json
        if ppath.endswith(".py"):
            try:
                from virturoid.services.policy_importer import static_parse_python
                spec = static_parse_python(open(ppath, encoding="utf-8").read(), source_ref=os.path.basename(ppath))
                script_meta = {"entrypoint": spec.entrypoint, "dependencies": spec.dependencies[:8],
                               "warnings": list(getattr(spec, "warnings", []))[:5]}
            except Exception as exc:  # noqa: BLE001
                script_meta = {"parse_error": str(exc)}
            sib = [os.path.join(os.path.dirname(ppath), n) for n in ("policy_params.json", "params.json")]
            jp = next((c for c in sib if os.path.exists(c)), None)
            if jp:
                params = json.load(open(jp, encoding="utf-8"))
        elif params is None:
            try:
                params = json.load(open(ppath, encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                return {"error": f"could not read control params from {ppath}: {exc}"}
    params = params or {}                                       # no readable params -> adopter falls back to a mid gait

    from virturoid.services.control_adopter import adopt_control_script
    out = adopt_control_script(gene, params, steps=int(args.get("steps", 800)),
                               generations=int(args.get("generations", 6)), pop=int(args.get("pop", 16)),
                               seed=int(args.get("seed", 0)))
    if script_meta:
        out["control_script"] = script_meta
    return out


def _import_dataset(args: dict) -> dict:
    from virturoid.services.dataset_importer import import_dataset
    path = (args or {}).get("path", "")
    if not path:
        return {"error": "path is required (a dataset dir/file: lerobot/hdf5/mcap/bag/db3/npz)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    return import_dataset(path).to_dict()


def _learn_gait(args: dict) -> dict:
    from virturoid.services.gait_flywheel import learn_gait_flywheel
    from virturoid.services.memory_db import MemoryDB
    args = args or {}
    rid = args.get("robot_id")
    if rid:                                                    # learn for the EXACT held body (what verify runs)
        from virturoid.services import session_state as _S
        gene = _S.get_robot(rid)                               # get_robot returns the RobotGene directly (or None)
        if gene is None:
            return {"error": f"no held robot {rid}"}
        prompt = getattr(gene, "robot_class", "legged")
    else:
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.morphology_composer import compose_robot
        prompt = args.get("prompt", "a quadruped robot dog")
        gene = ensure_walkable_quad(compose_robot(prompt), prompt)
    # Go through the FLYWHEEL: recall a specific prior for this body -> screened warm-start -> bank -> provenance.
    # So repeated/adjacent bodies compound (the bank persists in the shared memory dir), not cold-start every time.
    db = MemoryDB(**({"db_path": args["db_path"]} if args.get("db_path") else {}))
    out = learn_gait_flywheel(
        gene, db, generations=int(args.get("generations", 8)), pop=int(args.get("pop", 20)),
        steps=int(args.get("steps", 900)), seed=int(args.get("seed", 0)), workers=int(args.get("workers", 1)))
    out["prompt"] = prompt
    out["deployable"] = True                                   # the learned params ARE the deploy controller
    return out


def _classify_failure(args: dict) -> dict:
    from virturoid.services.failure_classifier import repairs_for_metrics
    args = args or {}
    metrics = args.get("metrics")
    if not isinstance(metrics, dict):
        return {"error": "metrics (an episode-metrics dict) is required"}
    return repairs_for_metrics(metrics, domain=args.get("domain", "auto"))


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
            "base_freq": {"type": "number", "default": 1.4}, "seed": {"type": "integer", "default": 0},
            "bank_dividend": {"type": "boolean", "default": False,
                              "description": "also record a DataDividend (the demonstration_dataset prior improved)"},
            "memory_dir": {"type": "string"}}},
        "handler": _amplify_demonstrations, "heavy": True,
    },
    "import_bom": {
        "description": "Import a user Bill of Materials (.csv/.json/.yaml/.xlsx): classify arbitrary columns to "
                       "canonical fields, normalize units (g->kg), merge duplicate parts, and return typed line "
                       "items with provenance — ready to override/augment the generated BOM. No physics.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a .csv/.json/.yaml/.xlsx BOM file"}}},
        "handler": _import_bom, "heavy": False,
    },
    "sandbox_policy": {
        "description": "PolicyImporter Tier P2: run an inbound Python controller ONCE against a simulated "
                       "observation inside an isolated, AST-gated subprocess (no network/keys/os; numpy only — "
                       "torch/onnx route to a native adapter), and validate the action is dimension-correct, "
                       "finite, and within safety limits. Fail-closed. The entrypoint/action_dim are auto-detected.",
        "parameters": {"type": "object", "required": ["source"], "properties": {
            "source": {"type": "string", "description": "inline Python controller (e.g. def act(obs): ...)"},
            "observation": {"type": "array", "description": "one observation vector"},
            "entrypoint": {"type": "string"}, "action_dim": {"type": "integer"},
            "safety_limits": {"type": "object"}, "timeout": {"type": "number", "default": 10.0}}},
        "handler": _sandbox_policy, "heavy": False,
    },
    "import_onnx_policy": {
        "description": "PolicyImporter Tier P3: load an inbound ONNX policy and (given an observation) run one "
                       "inference, validating the action's dimension/finiteness/limits. ONNX is a computation "
                       "GRAPH (safe to load, unlike a torch pickle). Without an observation, returns the model's "
                       "declared input/output tensor contract. No physics.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a .onnx policy file"},
            "observation": {"type": "array", "description": "one observation vector (omit to just inspect IO)"},
            "action_dim": {"type": "integer"}, "safety_limits": {"type": "object"}}},
        "handler": _import_onnx_policy, "heavy": False,
    },
    "import_controller_interface": {
        "description": "Extract a controller contract from a ROS 2 robot: parse the <ros2_control> URDF/xacro tag "
                       "(+ optional controller_manager YAML) into joint command/state interfaces, joint order, "
                       "safety limits, hardware plugin, and control rate — the truth needed to map an imported "
                       "policy to actuators. Static parse (no xacro execution). No physics.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "a .urdf/.xacro/.xml with a <ros2_control> block"},
            "xml": {"type": "string", "description": "inline URDF/xacro XML (alternative to path)"},
            "controller_yaml_path": {"type": "string"}, "controller_yaml": {"type": "string"}}},
        "handler": _import_controller_interface, "heavy": False,
    },
    "import_cad": {
        "description": "Import a CAD mesh (.stl binary/ascii or .obj): recover triangle/vertex count, the "
                       "axis-aligned bounding box + real dimensions, a unit-scale guess (mm-modeled parts read "
                       "~1000x too big), a mesh volume, and an inertial mass estimate from an assumed density "
                       "(with a warning). STEP/IGES is honestly deferred to a CAD kernel. No physics.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a .stl or .obj mesh file"},
            "material": {"type": "string", "enum": ["abs", "pla", "aluminum", "steel", "nylon"],
                         "description": "density prior for the mass estimate (default abs)"}}},
        "handler": _import_cad, "heavy": False,
    },
    "ingest_project": {
        "description": "INGESTION AGENT: bring in a user's EXISTING robot end-to-end. Point it at a project "
                       "FOLDER or .zip (their URDF/MJCF + optional BOM/CAD) and/or an NLP description ('aluminum "
                       "body, carbon-fiber legs, 5 kg payload, 6-DOF arm'); it classifies the folder, imports the "
                       "model into an editable RobotGene, parses the description into typed materials/payload, "
                       "APPLIES them (only to parts that exist -- no fabrication), folds in the BOM/CAD, and holds "
                       "ONE unified robot in the session ready to verify/edit. Returns robot_id + what was applied/"
                       "skipped + honest warnings. Composes from the description if no model is importable.",
        "parameters": {"type": "object", "properties": {
            "project_path": {"type": "string", "description": "a project folder or .zip of the robot's files"},
            "description": {"type": "string", "description": "NLP about the robot (materials per part, payload, DOF)"},
            "robot_id": {"type": "string", "description": "reuse/replace a specific held robot id (optional)"}}},
        "handler": _ingest_project, "heavy": False,
    },
    "adopt_control_script": {
        "description": "A user's OWN control script / policy: RUN it on a held robot in real physics, then IMPROVE "
                       "it. Point it at a control-script .py (statically parsed for entrypoint/deps/safety), a "
                       "policy_params.json, or inline params; the sim rolls out the imported controller AND runs a "
                       "gait search WARM-STARTED from the user's params, returning a measured before/after + the "
                       "improved params. Honest: it keeps the user's controller if tuning can't credibly beat it. "
                       "Measured: an imported CPG that shuffled 0.34 m (not credible) improved to a 0.62 m credible "
                       "walk (1.8x). Real MuJoCo, slow.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string", "description": "the held robot to run/improve the controller on"},
            "script_path": {"type": "string", "description": "a .py control script (a sibling policy_params.json is auto-found)"},
            "params_path": {"type": "string", "description": "a policy_params.json (freq/amplitude/... )"},
            "params": {"type": "object", "description": "inline control params (alternative to a path)"},
            "generations": {"type": "integer", "default": 6}, "pop": {"type": "integer", "default": 16},
            "steps": {"type": "integer", "default": 800}, "seed": {"type": "integer", "default": 0}}},
        "handler": _adopt_control_script, "heavy": True,
    },
    "import_dataset": {
        "description": "Import a demonstration/log dataset (LeRobot dir, robomimic .hdf5, .mcap/.bag/.db3 log, or "
                       "virturoid npz) into a typed spec: episodes, rate, modalities, and candidate observation/"
                       "action keys to seed training. Grounded where deps allow, honest where they don't. No physics.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a dataset directory or file"}}},
        "handler": _import_dataset, "heavy": False,
    },
    "learn_gait": {
        "description": "LEARN a deployable walk for a legged body: CEM-optimize the crawl-gait controller's "
                       "parameters (freq/amps/duty/gains) with an un-gameable fitness (forward travel counts only "
                       "when upright + survived). Unlike an MJX policy, the result is CPU-deployable BY "
                       "CONSTRUCTION (it IS the deploy controller). Returns the best gait + measured forward/height. "
                       "Real MuJoCo, slow.",
        "parameters": {"type": "object", "properties": {
            "robot_id": {"type": "string", "description": "learn for THIS held robot (what verify_robot runs); "
                                                          "omit to compose from prompt"},
            "prompt": {"type": "string", "default": "a quadruped robot dog"},
            "generations": {"type": "integer", "default": 10}, "pop": {"type": "integer", "default": 20},
            "steps": {"type": "integer", "default": 1000}, "seed": {"type": "integer", "default": 0}}},
        "handler": _learn_gait, "heavy": True,
    },
    "classify_failure": {
        "description": "Turn raw episode metrics (survived/height_ratio/forward/contacts/lifted/...) into a "
                       "failure label AND the concrete curriculum repairs for it (closes metrics -> failure -> "
                       "repair). Rule-based cold start; refines with a learned classifier as episodes accrue. "
                       "No physics.",
        "parameters": {"type": "object", "required": ["metrics"], "properties": {
            "metrics": {"type": "object", "description": "episode metrics dict"},
            "domain": {"type": "string", "enum": ["auto", "legged", "manipulator"], "default": "auto"}}},
        "handler": _classify_failure, "heavy": False,
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
