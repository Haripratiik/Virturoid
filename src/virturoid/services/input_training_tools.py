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


def _model_compile_probe():
    """A real-MuJoCo-compile probe for the model picker (input_classifier.rank_model_candidates).

    Reuses ``model_import.import_model`` -- the SAME machinery ingest uses on the file it eventually picks -- so
    the counts the picker scores on are the counts the customer will actually get. Results are memoised because
    the picker probes a handful of near-tied candidates and ingest then imports the winner again. Returns None
    when MuJoCo is unavailable, which simply drops the compiled evidence and leaves the static ranking."""
    cache: dict[str, dict | None] = {}

    def probe(abs_path: str) -> dict | None:
        if abs_path not in cache:
            try:
                from virturoid.services.model_import import import_model
                r = import_model(abs_path)
                cache[abs_path] = ({"ok": True, "bodies": int(r.get("parts") or 0),
                                    "actuators": int(r.get("actuated") or 0)} if r.get("ok")
                                   else {"ok": False, "bodies": 0, "actuators": 0})
            except Exception:  # noqa: BLE001 - no MuJoCo / unreadable file -> no compiled evidence, not a crash
                cache[abs_path] = None
        return cache[abs_path]

    return probe


# Classes that ALREADY carry legged meaning. ``robot_import._infer_class`` decides the family from the limbs
# that actually hold the body up (#244) and is right across the Menagerie corpus, so the #214 reconciliation
# below must never overwrite one of these.
_LEGGED_CLASSES = frozenset({"quadruped", "hexapod", "octopod", "legged", "humanoid", "biped", "bipedal"})
# The walkable reference template is a QUADRUPED fan/crawl recipe -- only these families can adopt it.
_QUAD_TEMPLATE_CLASSES = frozenset({"quadruped", "hexapod", "octopod", "legged"})


def _needs_legged_reconciliation(gene) -> bool:
    """True only for the #214 case: a body whose STRUCTURE is legged while its class string says otherwise.

    That happens when MuJoCo fuses a URDF quadruped's static torso into the world, leaving a fixed base that
    the string classifier reads as an arm. It does NOT happen to an imported humanoid, whose class the
    structural classifier already got right -- and treating it as if it did is what renamed every biped a
    quadruped on the way in."""
    try:
        from virturoid.services.task_matched_eval import robot_kind
        if robot_kind(gene) != "legged":
            return False
    except Exception:  # noqa: BLE001 - a classification guess must never break an ingest
        return False
    return (getattr(gene, "robot_class", "") or "").strip().lower() not in _LEGGED_CLASSES


def _legged_family(gene) -> str:
    """Which legged family a body belongs to, from the limbs that actually CARRY it -- the same ground-contact
    signal ``robot_import._infer_class`` uses -- never a hard-coded guess.

    An inconclusive count returns the honest generic ``"legged"`` rather than inventing a family, because the
    family drives the BOM, the spec sheet, the verify rubric and the walkable-template offer."""
    n = 0
    try:
        import mujoco

        from virturoid.services.appendage_map import build_appendage_map
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        n = int(build_appendage_map(mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))).n_legs)
    except Exception:  # noqa: BLE001 - no MuJoCo / an uncompilable twin -> fall through to the generic
        n = 0
    if n >= 6:
        return "hexapod"
    if n >= 3:
        return "quadruped"
    if n == 2:
        return "humanoid"
    return "legged"


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
            # PICK THE MODEL BY EVIDENCE. ingest already compiles models, so it contributes the compiled
            # body/actuator counts the metadata-only inspect_project_bundle cannot.
            result["project_graph"] = project_graph_summary(bundle, compile_probe=_model_compile_probe())
            # Auto-read a NOTES / README the customer dropped in the folder into the NLP payload, so specs
            # written in notes.md ("aluminum body, carbon-fiber legs, 5 kg payload") are parsed the same as a
            # typed description -- a drop-a-folder customer shouldn't have to re-type what's already in the box
            # (2026-07-24 audit). Capped so a long README can't swamp the props extractor.
            for _nm in ("notes.md", "notes.txt", "README.md", "readme.md", "README.txt"):
                _np = os.path.join(scan_root, _nm)
                try:
                    if os.path.isfile(_np):
                        with open(_np, encoding="utf-8", errors="replace") as _f:
                            _txt = _f.read(4000).strip()
                        if _txt:
                            description = (description + "\n" + _txt).strip() if description else _txt
                            result["notes"].append(f"folded {_nm} from the project into the NLP description")
                            break
                except OSError:
                    pass
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

    # 2-faithful) FAITHFUL LANE (P0, plan §P0.2): before we touch the lossy gene twin, load the customer's model
    # AS-IS through the repair pass and keep it. This is the body with THEIR link names + dimensions (FL_hip,
    # FL_thigh, ...); the gene twin is a re-derived approximation good for edits but not for fidelity. We DISCLOSE
    # which lanes ran so an ingesting agent (and the customer) can see exactly what was preserved vs approximated.
    result["lanes_attempted"] = []
    result["lane_used"] = None
    if model_rel and scan_root:
        model_path = os.path.join(scan_root, model_rel.replace("/", os.sep))
        try:
            from virturoid.services.model_import import import_model, reroot_free_base
            fm = import_model(model_path)
            result["lanes_attempted"].append("faithful")
            if fm.get("ok"):
                mjcf, rerooted = reroot_free_base(fm["mjcf"])
                names = fm.get("body_names", [])
                result["faithful"] = {
                    "ok": True, "parts": fm.get("parts"), "actuated": fm.get("actuated"),
                    "free_base": bool(fm.get("free_base")) or rerooted,
                    "reroot_applied": rerooted, "repairs": fm.get("repairs", []),
                    "link_names_preserved": [n for n in names if n][:12], "mjcf": mjcf,
                }
                result["lane_used"] = "faithful"
                if fm.get("repairs"):
                    result["notes"].append("faithful lane: applied " + ", ".join(
                        f"{r['kind']}(x{r['count']})" for r in fm["repairs"]) + " — every change is listed in "
                        "result['faithful']['repairs']")
                if rerooted:
                    result["notes"].append("faithful lane: the imported body was bolted to the world (fixed base) "
                                           "— added a free base so it can locomote; original geometry unchanged")
            else:
                result["faithful"] = {"ok": False, "note": fm.get("note"), "repairs": fm.get("repairs", [])}
        except Exception as exc:  # noqa: BLE001 - faithful lane is additive; the gene twin still stands
            result["warnings"].append(f"faithful lane unavailable: {exc}")

    # 2b) a legged import that can't stand on its own stance gets the SAME walkable-stance treatment a composed
    # body gets (a wide fanned stance) so it can actually be SIMULATED and its controller improved. A well-formed
    # import that already walks is left untouched; no-op for arms / mobile bases.
    if gene is not None and result.get("lane_used") is None:
        result["lane_used"] = "gene_fallback"                 # only the lossy twin is runnable -> say so
    if gene is not None:
        result["lanes_attempted"].append("gene")
    # B2 (2026-07-24 audit): NEVER silently swap the customer's imported body for a canonical template before we
    # verify it. That made the ingest verdict describe OUR template's walk, not the customer's robot -- the exact
    # honesty failure the strategy is built to avoid. We KEEP the imported geometry (with the customer's link
    # names) as the held robot, report ITS OWN honest verdict, and OFFER a walkable reference template as a
    # clearly-labelled, opt-in alternative the customer can adopt -- never a hidden substitution.
    if gene is not None:
        try:
            from virturoid.services.task_matched_eval import evaluate_robot, robot_kind
            _kind = robot_kind(gene)
            # #214: reconcile a fixed-base legged import that the string classifier mislabelled an arm, so the
            # BOM/fusion/verify pipeline all treat it consistently as the legged body it structurally is.
            #
            # 2026-08-01: this fired for ANY class outside ("quadruped", "legged", "hexapod") and always wrote
            # the literal "quadruped", so every imported HUMANOID -- a family robot_import._infer_class had
            # already read correctly off the limbs holding it up (#244) -- was renamed a quadruped here. Measured
            # on unitree_g1: bom.json and spec_sheet.json said `robot_class: quadruped`, the honesty ledger said
            # "the imported QUADRUPED does not walk credibly", and the walkable-template offer quoted 0.681 m --
            # the generic quad template, the same number offered for a Go2 -- i.e. it offered to turn a biped
            # into a different animal. A class that already means legged is now left alone, and when the remap
            # does fire the family comes from the body's own leg count instead of a constant.
            if _needs_legged_reconciliation(gene):
                _fam = _legged_family(gene)
                result["notes"].append(f"reclassified the import from '{gene.robot_class}' to '{_fam}' (it "
                                       f"stands on limbs off a common base, not on wheels or a bench mount)")
                gene.robot_class = _fam
        except Exception:  # noqa: BLE001
            _kind = getattr(gene, "robot_class", "")
    if gene is not None and _kind == "legged":
        try:
            own = float(evaluate_robot(gene).get("value", 0.0))
            result["imported_verdict"] = {"walks_as_imported": own >= 0.5, "distance_m": round(own, 3),
                                          "body": "the customer's own imported geometry (not a substitute)"}
            if own < 0.5:
                # measure (do NOT adopt) what a walkable template would do, so the offer is honest + quantified.
                # The reference template is a QUADRUPED fan/crawl recipe, so it is only offered to quadruped-
                # family bodies: proposing it for a biped is not a fixed version of the customer's robot, it is
                # a different animal (and it quoted the generic quad's distance as if it were theirs).
                _cls = (getattr(gene, "robot_class", "") or "").strip().lower()
                if _cls in _QUAD_TEMPLATE_CLASSES:
                    try:
                        from virturoid.services.anatomy_compiler import ensure_walkable_quad
                        tmpl = ensure_walkable_quad(gene, "imported quadruped", force=True)
                        tval = float(evaluate_robot(tmpl).get("value", 0.0))
                        result["walkable_template_offer"] = {
                            "available": tval > own, "template_distance_m": round(tval, 3),
                            "how_to_adopt": "call amend/edit with op 'adopt_walkable_template', or train the "
                                            "imported body directly with train_reward — its geometry is "
                                            "preserved either way"}
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    result["walkable_template_offer"] = {
                        "available": False,
                        "why": f"the walkable reference template is a QUADRUPED fan/crawl recipe; this import is "
                               f"a {_cls or 'legged'} body, so adopting it would hand back a different animal "
                               f"rather than a walking version of yours",
                        "how_to_proceed": "train the imported body directly with train_reward — for a biped, "
                                          "dynamic walking is a learned-control problem, not a stance tweak"}
                # ...and say so consistently: the ledger must not advertise a template the offer just declined
                _offered = bool((result.get("walkable_template_offer") or {}).get("available"))
                result["notes"].append(
                    f"the imported {getattr(gene, 'robot_class', 'legged')} does not walk credibly as imported "
                    f"({round(own, 3)} m under the scripted gait) -- its ORIGINAL geometry is kept as-is; "
                    + ("a walkable reference template is available as an explicit opt-in (see "
                       "walkable_template_offer), and " if _offered else
                       "no walkable reference template fits this body (see walkable_template_offer), so ")
                    + "train_reward can learn a gait for the real body.")
        except Exception:  # noqa: BLE001 - the honest-verdict probe is best-effort; never block the ingest
            pass

    # 3) parse the NLP description into typed, provenance-tagged properties + edit ops
    from virturoid.services.nlp_properties import extract_properties
    props = extract_properties(description)
    result["nlp"] = props.to_dict()
    result["notes"].extend(props.notes)

    # fallback: no importable model but a description -> compose an editable robot from the words (still ingest-able)
    composed_from_description = False
    if gene is None and description:
        try:
            from virturoid.services.morphology_composer import compose_robot
            gene = compose_robot(description)
            composed_from_description = True
            result["notes"].append("no importable robot model found -> composed an editable robot from the description")
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"could not compose a robot from the description: {exc}")
    if gene is None:
        result["ok"] = False
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

    # 6b) SUBSTITUTION GATE. A GENERATED body must never be presentable as the customer's own robot.
    #
    # Measured before this gate: dropping boston_dynamics_spot/ returned
    #     ok=True  "Ingested -> robot_5eecc737: lane=faithful, 0 material(s) applied, ... 0 warning(s)."
    # while the held robot was a composed quadruped and Spot had never been read. `lane=faithful` was true of the
    # scene wrapper we happened to compile, not of the body we held -- the summary conflated the two -- and the
    # only trace of the swap was a line in `notes`, which the customer-facing summary counts nothing from.
    #
    # We do NOT return an error, because composing from words is a documented, legitimate mode (description-only
    # ingest asks for exactly that). What must be impossible is MISTAKING one for the other, so:
    #   * `lane_used` describes the HELD ROBOT, and can never read "faithful" for a body we generated;
    #   * a compose that happened *after the customer handed us files* is a SUBSTITUTION -- it sets ok=False, a
    #     top-level `substituted` block, and a warning (so the "N warning(s)" count can never be 0);
    #   * `summary` -- the one line an agent relays -- leads with it.
    _design_source = str(getattr(gene, "design_source", "") or "").lower()
    if composed_from_description or (result.get("lane_used") == "faithful"
                                     and _design_source.startswith("anatomy")):
        result["lane_used"] = "composed_from_description"
    result["held_robot_design_source"] = _design_source or "unknown"
    result["ok"] = True
    if composed_from_description and path:
        seen = list((result.get("project_graph") or {}).get("robot_models") or [])
        why = (f"none of the {len(seen)} model file(s) in the project could be imported"
               if seen else "the project contained no robot description file (URDF/MJCF/SDF/USD)")
        result["ok"] = False
        result["substituted"] = True
        result["substitution"] = {
            "held_robot_is": "a robot GENERATED from the text description — NOT the robot in your project",
            "reason": why,
            "project_path": str(path),
            "model_files_seen": seen[:8],
            "import_errors": [w for w in result["warnings"] if "import failed" in w][:4],
            "how_to_fix": "point ingest_project directly at the model file, or fix the import errors above; "
                          "inspect_project_bundle shows every candidate and why each was ranked where it was",
        }
        result["warnings"].insert(0, f"SUBSTITUTED ROBOT: {why}, so the held robot {result['robot_id']} was "
                                     f"GENERATED from your description and is NOT your robot.")

    # 7) INGESTION REPORT (P0, plan §P0.7): the honest three-ledger account of what the folder yielded —
    # what we UNDERSTOOD, what we GUESSED, what we DROPPED — the same honesty contract as composition_notes.
    report = _ingestion_report(result)
    result["ingestion_report"] = report
    _persist_ingestion_report(report, scan_root if (path and not str(path).lower().endswith(".zip")) else path)

    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    lane = result.get("lane_used") or "none"
    if result.get("substituted"):
        result["summary"] = (
            f"SUBSTITUTED - {result['robot_id']} IS NOT YOUR ROBOT: {result['substitution']['reason']}, so this "
            f"body was GENERATED from your description (lane={lane}). Your model was not ingested. "
            f"{len(result['warnings'])} warning(s); see result['substitution'] for how to fix it.")
    else:
        result["summary"] = (f"Ingested -> {result['robot_id']}: lane={lane}, "
                             f"{len(result['materials_applied'])} material(s) applied, "
                             f"payload={result['payload_kg']} kg, {len(result['warnings'])} warning(s). "
                             f"Ready to edit/verify.")
    return result


def _ingestion_report(result: dict) -> dict:
    """Turn an ingest result into the understood/guessed/dropped ledger a customer can trust."""
    understood: list[str] = []
    guessed: list[str] = []
    dropped: list[str] = list(result.get("warnings", []))

    faithful = result.get("faithful") or {}
    if faithful.get("ok"):
        kept = faithful.get("link_names_preserved") or []
        understood.append(f"faithful model loaded: {faithful.get('parts')} bodies, "
                          f"{faithful.get('actuated')} actuators; link names preserved e.g. {kept[:4]}")
        for rep in faithful.get("repairs", []):
            understood.append(f"repair applied ({rep['kind']}): {rep['detail']}")
        if faithful.get("reroot_applied"):
            understood.append("added a free base so the fixed-base import can locomote")
    elif faithful:
        dropped.append(f"faithful model did not load: {faithful.get('note')}")

    imp = result.get("import") or {}
    if imp:
        guessed.append(f"editable gene twin inferred as {imp.get('robot_class')} "
                       f"(approximate — use the faithful model for fidelity)")
    for note in result.get("notes", []):
        (guessed if "could not" in note or "adopted" in note else understood).append(note)
    if result.get("bom"):
        understood.append(f"BOM: {result['bom'].get('line_items')} line items, "
                          f"{result['bom'].get('total_mass_kg')} kg")
    if result.get("cad"):
        understood.append(f"CAD: dims {result['cad'].get('dimensions_m')}")
    if result.get("control_scripts"):
        guessed.append(f"{len(result['control_scripts'])} control/policy file(s) found — not yet run "
                       "(call adopt_control_script)")
    return {"understood": understood, "guessed": guessed, "dropped": dropped,
            "lane_used": result.get("lane_used"), "lanes_attempted": result.get("lanes_attempted", [])}


def _persist_ingestion_report(report: dict, project_dir) -> None:
    """Write ingestion_report.json next to the project (best-effort; a write failure never sinks an ingest)."""
    import json
    try:
        base = project_dir if (project_dir and os.path.isdir(str(project_dir))) else "build"
        out = os.path.join(str(base), "ingestion_report.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - reporting is additive
        pass


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
    params_source = "inline" if params else None
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
                params_source = f"sibling:{os.path.basename(jp)}"
        elif params is None:
            try:
                params = json.load(open(ppath, encoding="utf-8"))
                params_source = os.path.basename(ppath)
            except Exception as exc:  # noqa: BLE001
                return {"error": f"could not read control params from {ppath}: {exc}"}

    # HONEST FAIL (F8): if the caller pointed at a .py controller but we could NOT extract runnable parameters
    # from it (no inline params, no sibling policy_params.json / params.json), the adopter would run a FALLBACK
    # gait — and claiming that "improved your controller" is a lie about a controller we never ran. Say so.
    if not params and ppath and str(ppath).endswith(".py"):
        looked = os.path.join(os.path.dirname(ppath), "policy_params.json")
        return {"ok": True, "adopted": False, "params_source": None, "control_script": script_meta,
                "reason": (f"could not extract runnable parameters from {os.path.basename(ppath)}: a Python "
                           f"controller's behaviour lives in its code, which is not executed for safety. Provide "
                           f"a policy_params.json next to it (looked at {looked}), pass params inline, or export "
                           f"an ONNX policy and use import_onnx_policy. The controller was NOT run and nothing "
                           f"was improved."),
                "how_to_fix": "supply params inline, ship a sibling policy_params.json, or use import_onnx_policy"}
    params = params or {}

    from virturoid.services.control_adopter import adopt_control_script
    out = adopt_control_script(gene, params, steps=int(args.get("steps", 800)),
                               generations=int(args.get("generations", 6)), pop=int(args.get("pop", 16)),
                               seed=int(args.get("seed", 0)))
    out.setdefault("adopted", True)
    out["params_source"] = params_source
    if script_meta:
        out["control_script"] = script_meta
    return out


def _list_parts(args: dict) -> dict:
    """The real catalog: every part (or one category) with its STRUCTURED specs — motor peak/rated torque + no-load
    RPM + voltage; camera resolution/fps/FOV; lidar channels/range/FOV; imu DOF/rate; compute TOPS; power Wh — so a
    part choice or swap is grounded in real numbers, not a headline string."""
    from virturoid.services.component_catalog import PART_CATEGORIES, list_parts
    args = args or {}
    cat = args.get("category")
    if cat and cat not in PART_CATEGORIES:
        return {"error": f"unknown category '{cat}'; valid: {list(PART_CATEGORIES)}"}
    parts = list_parts(cat)
    return {"category": cat or "all", "count": len(parts), "categories": list(PART_CATEGORIES), "parts": parts}


def _part_specs(args: dict) -> dict:
    """The real STRUCTURED specs of ONE named part (exact or substring, e.g. 'AK80-9' or 'Ouster OS1')."""
    from virturoid.services.component_catalog import part_specs
    name = (args or {}).get("part") or (args or {}).get("name", "")
    if not name:
        return {"error": "part (a catalog part name or substring) is required"}
    spec = part_specs(name)
    return spec if spec else {"error": f"no catalog part matching '{name}'; call list_parts to see options"}


def _pin_part(args: dict) -> dict:
    """SPECIFY an exact part for a held robot: pin a real catalog part for a category (e.g. category='lidar',
    part='Ouster OS1-32', or category='actuator', part='AK80-9'). The BOM is rebuilt using the pinned part and the
    pin travels with the robot/export. A part that doesn't exist, or whose category doesn't match, is rejected."""
    from virturoid.services import session_state as _S
    from virturoid.services.component_catalog import resolve_part
    args = args or {}
    rid = args.get("robot_id")
    if not rid:
        return {"error": "robot_id is required (the held robot to pin the part on)"}
    gene = _S.get_robot(rid)
    if gene is None:
        return {"error": f"no held robot {rid}"}
    name = args.get("part") or args.get("name", "")
    part = resolve_part(name)
    if part is None:
        return {"error": f"no catalog part matching '{name}'; call list_parts to see the options"}
    pcat = "actuator" if hasattr(part, "peak_torque_nm") else part.category
    category = args.get("category") or pcat
    if category != pcat:
        return {"error": f"'{part.name}' is a {pcat}, not a {category} — pick a {category} or set category='{pcat}'"}
    md = dict(getattr(gene, "metadata", None) or {})
    pins = dict(md.get("pinned_parts") or {})
    pins[category] = part.name
    md["pinned_parts"] = pins
    gene.metadata = md
    from virturoid.services.bom_builder import build_bom
    bom = build_bom(gene, task=args.get("task", ""), pins=pins)
    gene.metadata["bom"] = bom
    _S.commit_robot(rid, gene, label=f"pin {category}={part.name}")
    return {"robot_id": rid, "pinned": {category: part.name}, "specs": part.to_spec(),
            "all_pins": pins, "bom_pins": bom.get("pins"), "bom_totals": bom.get("totals")}


def _train_camera_policy(args: dict) -> dict:
    """Train the CV IN-HOUSE on a held camera-equipped robot's OWN onboard camera: render its functional robot_cam
    over many target placements, encode with the tiny 2-conv CNN, and fit a readout that reads the target's bearing
    — then BANK the learned vision policy on the robot so it deploys with it. Returns the held-out accuracy vs an
    untrained baseline (the un-gameable proof the CV is trained on THIS robot's camera). Real MuJoCo renders; slow."""
    from virturoid.services import session_state as _S
    args = args or {}
    rid = args.get("robot_id")
    if not rid:
        return {"error": "robot_id is required (the held camera-equipped robot to train vision on)"}
    gene = _S.get_robot(rid)
    if gene is None:
        return {"error": f"no held robot {rid}"}
    from virturoid.services.camera_perception import robot_camera_part
    part = robot_camera_part(gene)
    if part is None:
        return {"error": "this robot carries no camera in its BOM; give it a camera (a nav/inspect task) or "
                         "pin_part a camera first, then train its vision"}
    from virturoid.services.robot_vision import train_robot_vision
    tr = train_robot_vision(gene, n=int(args.get("n_train", 280)), seed=int(args.get("seed", 0)))
    md = dict(getattr(gene, "metadata", None) or {})
    md["vision_policy"] = {"trained": True, "enc_seed": int(tr["enc_seed"]),
                           "readout": [float(v) for v in tr["readout"]], "camera_part": part.name,
                           "bearing_mae_deg": tr["test_mae_deg"],
                           "method": "tiny-CNN features + fitted readout on the robot's own camera"}
    gene.metadata = md
    _S.commit_robot(rid, gene, label="train camera vision")
    return {"robot_id": rid, "trained_in_house": True, "banked": True, "camera_part": part.name,
            "bearing_mae_deg": tr["test_mae_deg"], "baseline_norm": tr["baseline_mae"],
            "improvement_x": tr["improvement_x"], "learned": tr["learned"],
            "verdict": (f"trained the robot's vision on its own {part.name} camera to {tr['test_mae_deg']} deg "
                        f"bearing error ({tr['improvement_x']}x baseline)" if tr["learned"]
                        else "vision training did not clearly beat the baseline this budget")}


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
    "list_parts": {
        "description": "The real part catalog with STRUCTURED specs: motors (peak/rated torque, no-load RPM, "
                       "voltage, gear ratio), cameras (resolution MP, fps, FOV, range), lidar (channels/beams, "
                       "range m, FOV, point rate), imu (DOF, rate), compute (TOPS, RAM), power (Wh, rail V), "
                       "grippers, wheels. Optionally filter by category. The queryable options for a build or a swap.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "actuator|camera|lidar|imu|force_torque|thermal|gps|"
                                                          "microphone|compute|power|drive_motor|wheel|gripper"}}},
        "handler": _list_parts, "heavy": False,
    },
    "part_specs": {
        "description": "The real STRUCTURED specs of one named part (exact or substring, e.g. 'AK80-9' or "
                       "'Ouster OS1') — the actual datasheet numbers (torque range, RPM, resolution, range, "
                       "channels, FOV, ...), not a headline string.",
        "parameters": {"type": "object", "required": ["part"], "properties": {
            "part": {"type": "string", "description": "a catalog part name or substring"}}},
        "handler": _part_specs, "heavy": False,
    },
    "pin_part": {
        "description": "SPECIFY an exact part for a held robot: pin a real catalog part for a category (e.g. "
                       "{category:'lidar', part:'Ouster OS1-32'} or {category:'actuator', part:'AK80-9'}). The BOM "
                       "is rebuilt with the pinned part (and the pin travels with the robot/export). A part that "
                       "isn't in the catalog, or whose category doesn't match, is rejected with a teaching error.",
        "parameters": {"type": "object", "required": ["robot_id", "part"], "properties": {
            "robot_id": {"type": "string"}, "part": {"type": "string", "description": "a catalog part name/substring"},
            "category": {"type": "string", "description": "the part's category (inferred from the part if omitted)"},
            "task": {"type": "string", "description": "task text so the rest of the sensor suite stays task-adaptive"}}},
        "handler": _pin_part, "heavy": False,
    },
    "import_dataset": {
        "description": "Import a demonstration/log dataset (LeRobot dir, robomimic .hdf5, .mcap/.bag/.db3 log, or "
                       "virturoid npz) into a typed spec: episodes, rate, modalities, and candidate observation/"
                       "action keys to seed training. Grounded where deps allow, honest where they don't. No physics.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a dataset directory or file"}}},
        "handler": _import_dataset, "heavy": False,
    },
    "train_camera_policy": {
        "description": "Train COMPUTER VISION in-house on a held camera-equipped robot's OWN onboard camera: render "
                       "its functional robot_cam over many target placements, encode with the tiny 2-conv CNN, and fit "
                       "a readout that reads the target's bearing from THIS robot's camera — then BANK the learned "
                       "vision policy on the robot (it deploys with it). Returns the held-out bearing error vs an "
                       "untrained baseline (the un-gameable proof the CV is trained on this robot's actual camera). "
                       "Errors if the robot carries no camera. Real MuJoCo renders, slow.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string", "description": "the held camera-equipped robot to train vision on"},
            "n_train": {"type": "integer", "default": 280, "description": "rendered training frames"},
            "seed": {"type": "integer", "default": 0}}},
        "handler": _train_camera_policy, "heavy": True,
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
