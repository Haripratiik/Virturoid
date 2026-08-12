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
import re


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


def _actuator_count(robot_id: str):
    """``(n_actuators, "")`` for a held robot, or ``(None, why_not)``. Compiles the gene — MuJoCo's own count."""
    from virturoid.services import session_state as _S
    gene = _S.get_robot(str(robot_id or ""))
    if gene is None:
        return None, f"no held robot '{robot_id}'; create_robot / submit_design / ingest_project first"
    try:
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        return int(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene)).nu), ""
    except Exception as exc:  # noqa: BLE001 - no MuJoCo / a body that will not compile: say so, never guess
        return None, f"could not count actuators on '{robot_id}': {type(exc).__name__}: {exc}"


def _import_onnx_policy(args: dict) -> dict:
    """INSPECT + VALIDATE an inbound ONNX policy. Deliberately NOT a deployment — see ``policy_native_adapter``.

    ``robot_id`` is what makes "validated" mean something about the customer's robot rather than about an
    abstract vector: the action width is checked against the actuator count MuJoCo reports for THAT body, so a
    policy trained for a 12-DOF quadruped and pointed at an 18-DOF hexapod is caught here instead of later.
    """
    from virturoid.services.policy_native_adapter import DEPLOYMENT, inspect_onnx, run_onnx_policy
    args = args or {}
    path = args.get("path", "")
    if not path:
        return {"error": "path is required (a .onnx policy file)"}
    if not os.path.exists(path):
        return {"error": f"path not found: {path}"}
    action_dim, checked_against = args.get("action_dim"), None
    if args.get("robot_id"):
        n, why = _actuator_count(args["robot_id"])
        if n is None:
            return {"error": why}
        if action_dim is not None and int(action_dim) != n:
            return {"error": f"action_dim={int(action_dim)} contradicts robot '{args['robot_id']}', which has {n} "
                             f"actuators — drop action_dim and the robot's own count is used"}
        action_dim, checked_against = n, {"robot_id": args["robot_id"], "n_actuators": n}
    if args.get("observation") is None:                        # no obs -> just inspect the IO contract
        out = inspect_onnx(path)
    else:
        out = run_onnx_policy(path, args["observation"], action_dim=action_dim,
                              safety_limits=args.get("safety_limits"))
    if checked_against:
        out["action_dim_checked_against"] = checked_against
    # SAY IT AT THE TOP LEVEL TOO. An agent that reads only the headline of a tool called "import_onnx_policy"
    # must not come away believing the robot now runs this policy; the nested block is the detail, this is the
    # sentence. Nothing in the repo can deploy an imported ONNX policy, and pretending otherwise by omission is
    # the same defect as train_reward's GPU arm claiming a bank nothing wrote to.
    out["deployed"] = False
    out["deployment_note"] = DEPLOYMENT["what_this_tool_does"] + " " + DEPLOYMENT["verify_robot_still_measures"]
    return out


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


# Classes that ALREADY carry legged meaning, and the families that can adopt the QUADRUPED fan/crawl walkable
# template. Both now come from ``body_kind`` -- the one place body classification lives -- rather than being
# re-listed here, which is how the copies drifted apart in the first place.
from virturoid.services.body_kind import (  # noqa: E402
    LEGGED_CLASSES as _LEGGED_CLASSES,
    QUAD_TEMPLATE_CLASSES as _QUAD_TEMPLATE_CLASSES,
    family_from_legs,
    measured_legs,
)


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
    family drives the BOM, the spec sheet, the verify rubric and the walkable-template offer.

    The count and the ladder are both ``body_kind``'s -- this function used to carry its own copy of each, and
    the ladder disagreed with ``robot_import._infer_class``'s copy at 3 legs."""
    return family_from_legs(measured_legs(gene)) or "legged"


#: a per-link mass has to move by more than this before we call the customer's number "replaced" (kg).
_MASS_TOL_KG = 1e-3

#: a scheme'd URL or a bare host/path (`arxiv.org/abs/...`) -- text that is never a robot specification.
_URL_RE = re.compile(
    r"(?:\w+://\S+)|(?:\b(?:www\.)?[\w.-]+\.(?:org|com|net|io|edu|gov|ai|dev|co|uk|de|cn)\b(?:/\S*)?)",
    re.IGNORECASE)


def _strip_urls(text: str) -> str:
    """Remove links before a dropped README is read as a materials spec.

    The fold-in below exists so a customer needn't retype specs already written in notes.md. But a README is
    not a spec sheet -- it is prose, badges and CITATIONS -- and the property extractor matches substrings.
    Measured on the real Menagerie G1: its README cites ``url={https://arxiv.org/abs/2502.08844}``, the
    ``/abs/`` in that arXiv link parsed as ABS PLASTIC, and ingest silently applied
    ``set_material(group='all', material='abs_plastic')`` to an imported humanoid -- which re-derives every
    link mass, so Unitree's 33.341 kg became 30.780 kg of our plastic estimate because of a citation URL.

    A URL can never be a material, a payload or a dimension, so it is dropped before parsing. Prose that
    really does say "aluminium body, 5 kg payload" is untouched."""
    try:
        return _URL_RE.sub(" ", text or "").strip()
    except Exception:  # noqa: BLE001 - never let sanitising break an ingest
        return text or ""


def _link_shape(gene) -> dict:
    """``{link_name: (mass_kg, radius_m, length_m)}`` — the snapshot that lets ingest state, as a measurement
    rather than a hope, whether the body it hands back is still the body the customer handed us."""
    try:
        return {s.name: (float(s.mass_kg or 0.0), float(s.radius_m or 0.0), float(s.length_m or 0.0))
                for s in gene.segments}
    except Exception:  # noqa: BLE001 - provenance accounting must never break an ingest
        return {}


def _reconcile_mass_provenance(gene, before: dict, result: dict) -> dict:
    """Make ``metadata['mass_source']`` tell the truth about the masses the gene is actually carrying.

    An imported robot arrives stamped ``mass_source='source_model'``: those per-link masses are the
    manufacturer's own, read straight off the customer's model. That flag is LOAD-BEARING, not decorative --
    ``gene_build.grounding_config`` turns it into ``preserve_mass=True`` on every later re-ground, so once it
    is set, every downstream door (build, export, spec sheet, BOM, certificate) treats whatever masses it
    finds as authoritative and refuses to touch them.

    So a wrong ``mass_source`` is SELF-SEALING: if ingest replaced the customer's masses with our own estimate
    and left the flag reading "source_model", the estimate is not merely mislabelled, it is LOCKED IN as the
    customer's own measurement, permanently and silently, and every artifact downstream cites the
    manufacturer for a number the manufacturer never published. Measured before this: an ingested Go2 held
    29.031 kg of our aluminium-and-catalog-motor guesswork under a label that said 15.206 kg of Unitree's.

    Therefore: when the numbers move, the label moves with them. Returns the ledger either way, so the honest
    case is visible too -- "we preserved it" is a claim that should also be measured, not assumed.
    """
    meta = getattr(gene, "metadata", None)
    after = _link_shape(gene)
    if not isinstance(meta, dict) or not before or not after:
        return {}
    src_total = sum(v[0] for v in before.values())
    now_total = sum(v[0] for v in after.values())
    changed = [n for n in (set(before) | set(after))
               if abs(after.get(n, (0.0,))[0] - before.get(n, (0.0,))[0]) > _MASS_TOL_KG]
    preserved = not changed and abs(now_total - src_total) <= _MASS_TOL_KG
    claimed = str(meta.get("mass_source") or "") or None
    prov = {"claimed_on_import": claimed, "mass_kg_as_imported": round(src_total, 3),
            "mass_kg_held": round(now_total, 3), "delta_kg": round(now_total - src_total, 3),
            "n_links_mass_changed": len(changed), "preserved": bool(preserved)}
    if claimed == "source_model" and not preserved:
        meta["mass_source"] = "virturoid_estimate"
        meta["mass_source_replaced"] = {
            "was": "source_model", "source_model_mass_kg": round(src_total, 3),
            "held_mass_kg": round(now_total, 3), "n_links": len(changed),
            "why": "ingest re-derived per-link mass (material grounding and/or an applied edit op), so these "
                   "masses are Virturoid's estimate and must not be re-preserved as the manufacturer's",
        }
        prov["corrected_to"] = "virturoid_estimate"
        _finding(result, _WARN,
            f"MASSES REPLACED - the manufacturer's per-link masses "
            f"({round(src_total, 3)} kg over {len(before)} links) were "
            f"REPLACED during ingest by Virturoid's derived masses ({round(now_total, 3)} kg, "
            f"{len(changed)} link(s) changed) -- metadata['mass_source'] is now 'virturoid_estimate', not "
            f"'source_model', so nothing downstream cites your manufacturer for our number")
    elif preserved and claimed == "source_model":
        prov["note"] = ("your model's own per-link masses are held unchanged; grounding sized actuators "
                        "around them instead of re-deriving them")
    # AND NAME THE MATERIAL HONESTLY. `metadata['grounding']` records the density the held masses were derived
    # at, which is right and load-bearing -- but it is OUR pick unless the customer asked for it, and their
    # request can perfectly well have been SKIPPED ("this robot has no 'torso' part") while a re-ground stamped
    # a global material anyway. Report the record and whether they chose it; never let the stamp imply consent.
    #
    # Only an IMPORT gets the warning. On a body we composed from the customer's words there is no
    # manufacturer to misquote -- picking a default density is the design decision they asked us to make --
    # and warning about it on every description-only ingest would be noise that buries the real one.
    rec = meta.get("grounding") if isinstance(meta.get("grounding"), dict) else None
    if rec:
        asked = {str(m.get("material") or "") for m in (result.get("materials_applied") or [])}
        mat = str(rec.get("material") or "")
        prov["material_masses_derived_at"] = mat
        prov["material_requested_by_customer"] = bool(mat and mat in asked)
        if mat and mat not in asked and meta.get("imported_from"):
            _finding(result, _WARN,
                f"MATERIAL NOT YOURS - the held masses were derived at '{mat}' — Virturoid's default for this "
                f"step, NOT a material you specified"
                + (f" (your material request was skipped: {result['skipped_ops'][0].get('reason')})"
                   if result.get("skipped_ops") else "")
                + "; set it explicitly with the set_material edit op if that is wrong")
    return prov


# ---------------------------------------------------------------------------------------------------------
# INGEST FINDINGS -> the one line an agent actually relays.
#
# This is the SECOND time the same defect shape has been found in this function. 60d452d closed it for the
# substitution gate ("substitution can no longer report success"); the simulability gate added in 9fbdcdc
# reintroduced it verbatim, because the honesty lived IN THE GATE instead of in the summary. The measured
# symptom, on the one genuinely-unsimulable Menagerie package:
#
#     ok=True   "Ingested -> adv_flybody_ingest: lane=faithful, 0 material(s) applied, payload=None kg,
#                1 warning(s). Ready to edit/verify."
#
# ...for a twin that does not compile in MuJoCo. A COUNT hides its contents: "1 warning(s)" reads the same
# whether the warning is "your CAD file had an odd unit" or "this robot cannot be stepped at all".
#
# So the summary is no longer written by the gates. It is DERIVED, and the derivation makes the honest
# outcome the DEFAULT rather than something each new gate has to remember:
#
#   * ``result['warnings']`` IS the findings ledger. A future gate that only appends a string -- which is
#     what every gate here already does, so it is what the next one will do -- still cannot produce a clean
#     summary: an unclassified warning is treated as ``warn`` and its TEXT reaches the line.
#   * ``_finding(result, _ERROR, ...)`` additionally flips ``ok`` to False and makes that finding LEAD the
#     summary. Refusing is a SEVERITY on a finding, not a bespoke branch a gate author has to write.
#   * "Ready to edit/verify" is emitted by exactly ONE branch: the one with no findings at all.
#
# _ERROR means the caller cannot proceed as asked with the robot we are handing back -- it is not their
# robot, or it cannot be stepped, or there is none. _WARN means the ingest stands but something the caller
# must know was dropped, replaced or guessed. A bad VERDICT (a body that does not walk) is neither: that is
# ``imported_verdict``, and it correctly leaves ``ok`` True.
# ---------------------------------------------------------------------------------------------------------
_ERROR, _WARN = "error", "warn"


def _finding(result: dict, severity: str, message: str, *, lead: bool = False) -> None:
    """Record one ingest finding at a severity the summary is structurally required to honour.

    Writes BOTH ``findings`` (typed) and ``warnings`` (the flat list every existing consumer already reads,
    including ``_ingestion_report``'s ``dropped`` ledger), so severity is additive and nothing downstream
    has to change to keep working. ``lead=True`` puts it first -- for the case where two errors are true at
    once and one of them is the more specific account of the other.
    """
    entry = {"severity": severity, "message": message}
    findings = result.setdefault("findings", [])
    warnings = result.setdefault("warnings", [])
    if lead:
        findings.insert(0, entry)
        warnings.insert(0, message)
    else:
        findings.append(entry)
        warnings.append(message)


def _finalize_ingest(result: dict) -> dict:
    """Derive ``ok``, ``error`` and ``summary`` from the findings ledger.

    The ONLY place any of the three is written, and every ``return`` of a built result goes through it, so a
    gate cannot report a success it did not earn -- nor a refusal with no message, which is what the two
    early ``return`` paths used to produce (``ok=False``, no ``summary`` at all, and an envelope reading
    "ingest_project reported failure without a reason").
    """
    typed = {}
    for f in result.get("findings") or []:
        typed.setdefault(f["message"], f)
    # Rebuild in warning order, promoting any bare `warnings.append` a gate made without a severity.
    ordered = [typed.get(w) or {"severity": _WARN, "message": w} for w in (result.get("warnings") or [])]
    for f in result.get("findings") or []:                     # a finding with no warning line (shouldn't happen)
        if f not in ordered:
            ordered.append(f)
    result["findings"] = ordered

    errors = [f for f in ordered if f.get("severity") == _ERROR]
    rid = result.get("robot_id")
    lane = result.get("lane_used") or "none"
    # The twin importer's own notes are NOT ingest findings -- they describe how the editable approximation
    # was derived (unsupported joint types, multi-joint bodies) and every real model carries a few. But they
    # are not nothing either, and they were invisible: "0 warning(s)" was the line for a Go2 whose import
    # raised 3. Counted separately so neither number can stand in for the other.
    approximations = len(((result.get("import") or {}).get("warnings")) or [])
    tail = (f" [robot={rid}; lane={lane}; {len(ordered)} finding(s)"
            + (f"; {approximations} twin-approximation note(s) in result['import']['warnings']"
               if approximations else "")
            + "; full list in result['findings']]")

    if errors:
        result["ok"] = False
        result["error"] = errors[0]["message"]
        result["summary"] = errors[0]["message"] + tail
    elif ordered:
        result["ok"] = True                                    # the ingest stands -- but it is not clean
        result["summary"] = (f"Ingested -> {rid}: lane={lane}, "
                             f"{len(result.get('materials_applied') or [])} material(s) applied, "
                             f"payload={result.get('payload_kg')} kg. NOT CLEAN - {ordered[0]['message']}"
                             + tail)
    else:
        result["ok"] = True
        result["summary"] = (f"Ingested -> {rid}: lane={lane}, "
                             f"{len(result.get('materials_applied') or [])} material(s) applied, "
                             f"payload={result.get('payload_kg')} kg, no findings"
                             + (f" ({approximations} twin-approximation note(s) in "
                                f"result['import']['warnings'])" if approximations else "")
                             + ". Ready to edit/verify.")
    return result


def _ingest_project(args: dict) -> dict:
    """INGESTION AGENT (Part B): a robotics team drops a project FOLDER/ZIP of their existing robot (URDF/MJCF +
    optional BOM/CAD) plus an NLP description ("aluminum body, carbon-fiber legs, 5 kg payload, 6-DOF arm"), and
    gets back ONE unified, immediately-editable RobotGene held in the session -- the user's stated materials +
    load already applied, with a project graph + BOM/CAD summary + honest per-step warnings. It ORCHESTRATES the
    importers already built (classifier, robot_import, bom_importer, cad_importer) + nlp_properties + edit_operators.

    THE CUSTOMER'S FOLDER IS READ-ONLY FOR THE WHOLE OF THIS CALL. Two writes into it were measured on a real
    Menagerie Go2 (a prep copy in ``model_import``, the report in ``_persist_ingestion_report``); both are
    fixed at their own site, and this wrapper is what stops the third. It has to sit at the DOOR rather than
    at each writer, because an ingest fans out through the classifier, three importers, the grounding pass
    and the edit operators -- "every write reachable from an ingest" is not a list anyone can keep current by
    reading. A refusal is recorded and surfaced in ``source_folder_protection`` rather than swallowed.
    """
    from virturoid.services import source_guard
    target = (args or {}).get("project_path") or (args or {}).get("path")
    with source_guard.read_only(target) as guard:
        result = _ingest_project_inner(args)
        report = guard.report()
    if report and isinstance(result, dict):
        result["source_folder_protection"] = report
        _finding(result, _WARN, f"WE TRIED TO WRITE INTO YOUR PROJECT FOLDER - {report['n_blocked_writes']} "
                                f"write(s) into {report['source_folder']} were refused; your files are "
                                f"unchanged. This is a Virturoid defect — see source_folder_protection.")
        _finalize_ingest(result)
    return result


def _ingest_project_inner(args: dict) -> dict:
    """The ingest itself. Always called through :func:`_ingest_project`, which owns the read-only guarantee."""
    import shutil

    args = args or {}
    path = args.get("project_path") or args.get("path")
    description = (args.get("description") or args.get("nlp") or "").strip()
    if not path and not description:
        return {"error": "provide project_path (a folder or .zip of the robot's files) and/or description (NLP)"}

    result: dict = {"robot_id": None, "materials_applied": [], "payload_kg": None, "applied_ops": [],
                    "skipped_ops": [], "warnings": [], "findings": [], "notes": []}
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
                        _txt = _strip_urls(_txt)
                        if _txt:
                            description = (description + "\n" + _txt).strip() if description else _txt
                            result["notes"].append(f"folded {_nm} from the project into the NLP description")
                            break
                except OSError:
                    pass
        except Exception as exc:  # noqa: BLE001
            _finding(result, _ERROR, f"PROJECT NOT READ - scanning {path} failed ({exc}), so NONE of the files "
                                     f"you handed us were opened; anything held below came from the description")

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
                                # A twin that cannot be STEPPED is a different failure from one whose schema is
                                # off, and the ingesting agent has to be able to tell: everything it does next
                                # (verify, certify, cost, calibrate) is computed by stepping this model.
                                "simulable": imp.get("simulable", True),
                                "simulation_check": imp.get("simulation_check") or {},
                                "warnings": list(imp.get("warnings", []))[:8]}
            if not imp.get("simulable", True):
                # AN ERROR, NOT A WARNING. The twin recorded here IS the held robot -- `S.put_robot(gene)`
                # below holds this object, and verify/certify/BOM/cost/calibrate all produce their numbers by
                # STEPPING it. A model that does not step cannot produce any of them, so "Ready to edit/verify"
                # is false about the only robot this call hands back. (The faithful MJCF, when it loaded, is
                # returned in the payload for native simulation -- but it is not what is held.)
                _why = " ".join(str((imp.get("simulation_check") or {}).get("reason") or "").split())[:220]
                _finding(result, _ERROR,
                         f"NOT SIMULABLE - the editable twin inferred from {model_rel} cannot be stepped in "
                         f"MuJoCo ({_why}). It is the HELD robot, and every number downstream (verdict, "
                         f"certificate, BOM, spec sheet, calibration gap) comes from stepping it, so it is NOT "
                         f"ready to edit or verify; result['faithful']['mjcf'] holds your model as-is for "
                         f"native simulation.")
        except Exception as exc:  # noqa: BLE001
            # No twin at all from the customer's model. Whatever is held below came from somewhere else.
            _finding(result, _ERROR, f"MODEL NOT IMPORTED - {model_rel} could not be turned into an editable "
                                     f"robot ({exc})")

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
                # A CLEAN refusal by the faithful lane raised nothing at all before this: it reached only
                # `_ingestion_report`'s `dropped` list, which the summary counts nothing from, so a customer
                # whose model we could not load as-is still read "0 warning(s). Ready to edit/verify."
                _finding(result, _WARN,
                         f"YOUR MODEL DID NOT LOAD AS-IS - the faithful lane refused {model_rel} "
                         f"({fm.get('note')}), so the only body here is our re-derived approximation of it, "
                         f"not your geometry or your link names")
        except Exception as exc:  # noqa: BLE001 - faithful lane is additive; the gene twin still stands
            _finding(result, _WARN, f"YOUR MODEL DID NOT LOAD AS-IS - the faithful lane failed on {model_rel} "
                                    f"({exc}); the held body is our re-derived approximation, not your geometry")

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
            # WHOSE CONTROLLER PRODUCED THIS NUMBER IS PART OF THE NUMBER. Until 2026-08-12 this read
            # ``walks_as_imported: own >= 0.5`` over "the customer's own imported geometry", which a customer
            # reads as "my robot walks". It is OUR scripted gait driving THEIR body -- the very attribution
            # ``verify_robot`` refuses to make (it returns ``locomotion_verdict: None`` and the "NO LOCOMOTION
            # VERDICT -- we do not have your robot's controller" lead for the same robot, in the same session).
            # Two surfaces answering the same question with different framings is how #215/#218 happened.
            #
            # The bare ``>= 0.5`` is also the wrong gate on its own: ``forward`` is world-frame delta-x, so a
            # body going round in a circle books its far side as travel. ``classify`` is the code-owned
            # un-gameable verdict and is what the rest of the product is judged by, so it is what is reported
            # here. MEASURED on the tests/test_customer_ingest fixture quad: 1.604 m, CREDIBLE WALK
            # (straightness 0.825, upright_frac 1.0, cadence 33.33) -- the threshold and the classifier agree
            # on THIS body, which is exactly why the disagreement had gone unnoticed.
            _ev = evaluate_robot(gene)
            own = float(_ev.get("value", 0.0))
            try:
                from virturoid.services import gait_quality as _gq
                _own_verdict = _gq.classify(_ev.get("detail") or {}) or None
            except Exception:  # noqa: BLE001 - the distance still stands if the classifier cannot be run
                _own_verdict = None
            result["imported_verdict"] = {
                "walks_under_our_scripted_gait": own >= 0.5, "distance_m": round(own, 3),
                "verdict": _own_verdict,
                "body": "the customer's own imported geometry (not a substitute)",
                "controller": "OURS, not yours -- we do not have your controller. This is what YOUR BODY did "
                              "under a gait we wrote for it, which is why verify_robot withholds a locomotion "
                              "verdict for this robot. adopt_control_script runs your parameters on it; "
                              "train_reward learns a controller for the real body."}
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
    # EVERY note `nlp_properties` emits is of one shape: "we read this requirement and did NOT act on it"
    # (a payload above the safe amend range, an unlabelled mass we refused to treat as a load, a stated DOF
    # count). They landed in `notes`, which the summary counts nothing from -- so a customer who wrote
    # "carries a 60 kg payload" read `payload=None kg ... Ready to edit/verify` with their requirement
    # nowhere in the line. Unapplied is a finding, not a footnote.
    for _n in props.notes:
        _finding(result, _WARN, f"STATED BUT NOT APPLIED - {_n}")

    # fallback: no importable model but a description -> compose an editable robot from the words (still ingest-able)
    composed_from_description = False
    if gene is None and description:
        try:
            from virturoid.services.morphology_composer import compose_robot
            gene = compose_robot(description)
            composed_from_description = True
            result["notes"].append("no importable robot model found -> composed an editable robot from the description")
        except Exception as exc:  # noqa: BLE001
            _finding(result, _ERROR, f"NOTHING COMPOSED - the description could not be turned into a robot "
                                     f"either ({exc})")
    if gene is None:
        # This exit used to set ok=False and return with NO `summary` key at all, so the one line an agent
        # relays did not exist and the envelope read "ingest_project reported failure without a reason".
        # It goes through the same finalizer as every other exit now.
        _finding(result, _ERROR, "NO ROBOT INGESTED - no importable model and no usable description, so this "
                                 "call is holding nothing")
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        return _finalize_ingest(result)

    # 4) apply the stated materials (only to parts that EXIST -- no fabrication) + payload, one gate per op
    from virturoid.services.edit_operators import apply_op, segments_for_group

    # GROUND THE ROBOT WE WERE HANDED -- DO NOT RE-DERIVE THE CUSTOMER'S MASSES.
    #
    # This step exists so `set_payload` has a real torque/mass baseline. It used to call `ground_gene(gene)`
    # BARE, which takes the defaults material="aluminum", fill=0.3, preserve_mass=False -- so every link's
    # mass was thrown away and recomputed as (primitive volume x aluminium density x 0.3) + one of OUR catalog
    # motors, on a body whose manufacturer masses ALREADY include its motors. Every actuator was counted
    # twice, against a shape we approximated, at a density nobody asked for. Measured through this tool on
    # real Menagerie models:
    #
    #     Go2   15.206 -> 29.031 kg      Panda  17.452 -> 50.425 kg
    #     G1    33.341 -> 42.339 kg      UR5e   20.995 -> 25.772 kg
    #
    # `grounded_physics.ground_gene`'s own docstring explains exactly why this call needs `preserve_mass` and
    # names the Go2 as the example -- the knowledge was already in the file; only this call site never used it.
    #
    # `gene_build.ground_and_repair` is the SINGLE grounding path both exit doors (package build and
    # `export_held`) already share, and it reads `grounding_config(gene)`, which sets preserve_mass=True for
    # `metadata['mass_source'] == 'source_model'` (an import) and otherwise reproduces the body's own recorded
    # material. Routing through it means the ingest door grounds a body the same way the export door will, so
    # the robot the customer verifies cannot change weight or shape on its way out.
    #
    # It also stops us stamping a material nobody chose: `ground_gene` only records `metadata['grounding']`
    # when it actually derived the masses, so a preserved import is no longer labelled "aluminum, fill 0.3"
    # while the ingest report says the aluminium request was SKIPPED for want of a 'torso' part.
    #
    # SCOPED TO BODIES WHOSE MASSES ARE AUTHORITATIVE, i.e. an import. A body we COMPOSED from the customer's
    # words has no manufacturer mass to protect, and moving its grounding is not free: `ground_gene` PINS
    # `torque_req_nm` on the first ground, so a different first call sizes different motors and every later
    # step compounds it -- and per [[body-and-gait-are-co-tuned]] a silent mass move trades walk verdicts
    # across the design bench.
    #
    # It also cannot be measured here. The composed path is NOT DETERMINISTIC: three identical calls with
    # "a quadruped robot dog, aluminum body, 5 kg payload" returned 46.172 / 49.715 / 22.368 kg over 23-25
    # segments, so no before/after number on that path would mean anything, and a change we cannot measure is
    # a change we should not make while fixing something else. The composed path therefore keeps exactly the
    # call it had, and this fix cannot reach it. (That 2.2x spread is a real defect, but a different one.)
    _shape_before = _link_shape(gene)
    _authoritative = str((getattr(gene, "metadata", None) or {}).get("mass_source") or "") == "source_model"
    try:
        if _authoritative:
            from virturoid.services.gene_build import ground_and_repair
            ground_and_repair(gene)                              # honours preserve_mass; same path as export
        else:
            from virturoid.services.grounded_physics import ground_gene
            ground_gene(gene)                                    # composed body: unchanged baseline
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
                    # The summary quotes `payload=N kg` as an accomplished fact. When the actuator catalog
                    # cannot actually reach that load the number is an aspiration, so the qualification has
                    # to travel with it rather than sitting behind a count.
                    _finding(result, _WARN, f"PAYLOAD OVER THE ACTUATOR ENVELOPE - {diff['warning']}")
        except Exception as exc:  # noqa: BLE001 - a bad op (incl. EditError) is skipped + noted, never aborts ingest
            result["skipped_ops"].append({**op, "reason": str(exc)})

    # 4b) RECONCILE THE PROVENANCE WITH THE NUMBERS -- after the ops, not just after grounding, because an
    # applied `set_material` / `set_payload` re-derives mass too (`edit_operators._reground_and_gate`). Whatever
    # replaced the customer's masses, the label must name it; see :func:`_reconcile_mass_provenance`.
    result["mass_provenance"] = _reconcile_mass_provenance(gene, _shape_before, result)
    # ...and say the same about their GEOMETRY. Grounding grows a link that is too thin to house the actuator
    # driving it -- correct for a body WE drew, but on an imported robot those radii are the customer's own
    # measurements, so a silent change is a different robot handed back under their name (#215/B2). Disclosed,
    # not suppressed: the growth is what keeps the held body identical to the one the export door emits.
    if str((getattr(gene, "metadata", None) or {}).get("imported_from") or ""):
        _now = _link_shape(gene)
        _grew = [n for n, v in _now.items()
                 if n in _shape_before and v[1] - _shape_before[n][1] > 1e-6]
        if _grew:
            _worst = max(_grew, key=lambda n: _now[n][1] / max(_shape_before[n][1], 1e-9))
            result["notes"].append(
                f"grounding widened {len(_grew)} of your link(s) so each can physically house the actuator "
                f"that drives it (largest: '{_worst}' radius "
                f"{round(_shape_before[_worst][1], 4)} -> {round(_now[_worst][1], 4)} m); lengths and masses "
                f"are untouched, and this is the same body the export door produces")
            result["geometry_changed_links"] = sorted(_grew)[:12]

    # 4c) SAY WHERE THIS BODY CAME FROM. `design_source` defaults to "unknown" on RobotGene, and the import
    # path never overwrote it -- so `get_robot` on a Go2 read straight off a named file answered
    # `design_source: "unknown"`, which is not modesty, it is the one fact about provenance we most certainly
    # had. "imported" is the existing vocabulary (`desktop._DESIGN_SOURCE_LABEL` already renders it as
    # "imported model"), and it is deliberately NOT in `design_cassette.MODEL_AUTHORED_SOURCES`, so an
    # imported body still counts as not-model-authored in the design-bench funnel.
    #
    # Keyed on `metadata['imported_from']` -- the flag `robot_import` sets when a real model file produced
    # this body -- so it can never fire on a body composed from the description. That matters: the
    # substitution gate in 6b reads `design_source` to catch an anatomy-composed body wearing a "faithful"
    # lane label, and stamping it here would have blinded exactly that gate.
    if str((getattr(gene, "metadata", None) or {}).get("imported_from") or ""):
        if str(getattr(gene, "design_source", "") or "").lower() in ("", "unknown"):
            gene.design_source = "imported"

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
                # Their BOM is the cost/actuator ground truth. Dropping it silently means every downstream
                # cost and mass figure comes from OUR catalog while the customer believes theirs was read.
                _finding(result, _WARN, f"YOUR BOM WAS NOT READ - {bom_ref} could not be parsed ({exc}); cost, "
                                        f"part and mass figures below come from Virturoid's catalog, not yours")
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
                _finding(result, _WARN, f"YOUR CAD WAS NOT READ - {cad_ref} could not be imported ({exc}); no "
                                        f"dimension or mass evidence from it reached the held robot")

    # 6b) SUBSTITUTION GATE. A GENERATED body must never be presentable as the customer's own robot.
    #
    # Measured before this gate: dropping boston_dynamics_spot/ returned
    #     ok=True  "Ingested -> robot_5eecc737: lane=faithful, 0 material(s) applied, ... 0 warning(s)."
    # while the held robot was a composed quadruped and Spot had never been read. `lane=faithful` was true of the
    # scene wrapper we happened to compile, not of the body we held -- the summary conflated the two -- and the
    # only trace of the swap was a line in `notes`, which the customer-facing summary counts nothing from.
    #
    # We do NOT raise, because composing from words is a documented, legitimate mode (description-only ingest
    # asks for exactly that). What must be impossible is MISTAKING one for the other, so:
    #   * `lane_used` describes the HELD ROBOT, and can never read "faithful" for a body we generated;
    #   * a compose that happened *after the customer handed us files* is a SUBSTITUTION -- an _ERROR finding,
    #     which is what flips ok=False and puts the refusal at the FRONT of the summary. This gate is now
    #     spelled the same way every other gate is; see `_finding` / `_finalize_ingest`.
    _design_source = str(getattr(gene, "design_source", "") or "").lower()
    if composed_from_description or (result.get("lane_used") == "faithful"
                                     and _design_source.startswith("anatomy")):
        result["lane_used"] = "composed_from_description"
    result["held_robot_design_source"] = _design_source or "unknown"
    if composed_from_description and path:
        _pg = result.get("project_graph") or {}
        seen = list(_pg.get("robot_models") or [])
        # A project whose ONLY robot description is a USD or SDF is the single likeliest first contact we have
        # (USD is the format NVIDIA trained the industry on). "no robot description file" would be a lie about
        # a folder that visibly contains robot.usd, and a bare "unsupported" would waste their afternoon — so
        # name the format, and name the conversion that works.
        _unsupported = list(_pg.get("unsupported_models") or [])
        _guidance = list(_pg.get("unsupported_model_guidance") or [])
        if seen:
            why = f"none of the {len(seen)} model file(s) in the project could be imported"
        elif _unsupported:
            why = (f"the project's only robot description(s) are in a format Virturoid cannot read "
                   f"({', '.join(sorted(_unsupported)[:3])}) — it reads URDF and MJCF")
        else:
            why = "the project contained no robot description file (URDF/MJCF)"
        result["substituted"] = True
        result["substitution"] = {
            "held_robot_is": "a robot GENERATED from the text description — NOT the robot in your project",
            "reason": why,
            "project_path": str(path),
            "model_files_seen": seen[:8],
            "unsupported_model_files": _unsupported[:8],
            "import_errors": [f["message"] for f in (result.get("findings") or [])
                              if f["message"].startswith(("MODEL NOT IMPORTED", "PROJECT NOT READ"))][:4],
            "how_to_fix": ("; ".join(_guidance) if _guidance else
                           "point ingest_project directly at the model file, or fix the import errors above; "
                           "inspect_project_bundle shows every candidate and why each was ranked where it was"),
        }
        # LEADS the summary: when the model also failed to import (an _ERROR of its own) this is the more
        # specific account of the same event, and it is the one the customer has to see first.
        _finding(result, _ERROR,
                 f"SUBSTITUTED - {result['robot_id']} IS NOT YOUR ROBOT: {why}, so this body was GENERATED "
                 f"from your description. Your model was not ingested; see result['substitution'] for how to "
                 f"fix it.", lead=True)

    # 7) VERDICT + INGESTION REPORT. The summary/ok/error come first so the report -- and the JSON it persists
    # next to the customer's project -- carries the same severities the caller was handed, not a flat list.
    _finalize_ingest(result)
    report = _ingestion_report(result)
    result["ingestion_report"] = report
    written = _persist_ingestion_report(
        report, scan_root if (path and not str(path).lower().endswith(".zip")) else path)
    if written:
        result["ingestion_report_path"] = written

    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
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
    # `dropped` is a flat list of strings, so the JSON persisted next to the customer's project could not say
    # which of its lines was "we could not parse your CSV" and which was "this robot cannot be stepped".
    # The typed ledger travels with it.
    return {"understood": understood, "guessed": guessed, "dropped": dropped,
            "findings": list(result.get("findings") or []), "ok": result.get("ok"),
            "summary": result.get("summary"),
            "lane_used": result.get("lane_used"), "lanes_attempted": result.get("lanes_attempted", [])}


def _persist_ingestion_report(report: dict, project_dir) -> str | None:
    """Write ingestion_report.json UNDER build/, and return where it landed.

    It used to be written "next to the project", which put a Virturoid artifact inside the folder the
    customer pointed us at -- measured on a real Menagerie Go2, an ingest left
    ``unitree_go2/ingestion_report.json`` in their tree. Convenient, and not ours to do: that folder can be
    a git checkout with a dirty-tree gate, a read-only mount, or a vendor drop nobody may modify. Keyed on
    the source path by ``source_guard.staging_dir`` so two projects never overwrite each other's report.

    Returns the path so the ingest result can NAME it -- a report the customer cannot find is a report that
    did not happen, and moving it silently would just trade one honesty gap for another.
    """
    import json
    try:
        from virturoid.services import source_guard
        out = source_guard.staging_dir(project_dir or "project", kind="ingest") / "ingestion_report.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        return str(out)
    except Exception:  # noqa: BLE001 - reporting is additive
        return None


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
    # ...AND LAND IT. Found by the same sweep that caught ``train_held`` (2026-08-08): this tool fits a
    # controller to the held body, warm-started from the CUSTOMER'S OWN parameters, and said "improved the
    # user's controller" while writing nothing — so the very next verify_robot re-measured the controller the
    # robot had before, exactly the defect 248afca closed for the other three doors. Same contract
    # (``trained_controller``): credible-gated, undoable, artifact-only under apply='never'. The gate is
    # ``beat_imported``, which is already the tool's own honest bar (a CREDIBLE walk that travels further than
    # the user's controller) — an "improvement" that failed it must never overwrite what the customer shipped.
    from virturoid.services.trained_controller import apply_trained_gait
    out["applied_to_robot"] = apply_trained_gait(
        rid, out.get("improved_params") or {}, door="adopt_control_script",
        apply=str(args.get("apply") or "auto").lower(),
        credible=bool(out.get("beat_imported")), verdict=str(out.get("verdict") or ""),
        evidence={"forward_m": (out.get("improved") or {}).get("forward_m"),
                  "imported_forward_m": (out.get("utilised") or {}).get("forward_m"),
                  "params_source": params_source})
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
    from virturoid.services.sensor_provenance import camera_is_ours_to_add
    part = robot_camera_part(gene)
    if part is None:
        # On the CUSTOMER'S imported machine the reason is a refusal, not a gap in their config — say which,
        # or "give it a camera" reads as our tooling being unconfigured rather than their robot having none.
        allowed, why = camera_is_ours_to_add(gene)
        if not allowed:
            return {"error": f"{why}. To train vision on a camera you intend to fit, pin_part it first "
                             f"(pinned_parts.camera) — that records the camera as YOUR addition, not ours"}
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
    # ...and "deployable" now means DEPLOYED. ``learn_gait_flywheel`` fits an operating point to this exact body
    # and hands back the numbers; nothing wrote them onto the held gene, so ``verify_robot`` afterwards ran the
    # controller the robot had BEFORE training. Landing it here is what makes the ``deployable: True`` above a
    # fact rather than a promise. See ``trained_controller`` for the contract (credible-gated, undoable).
    if rid:
        from virturoid.services.trained_controller import apply_trained_gait
        out["applied_to_robot"] = apply_trained_gait(
            rid, out.get("params") or {}, door="learn_gait", apply=str(args.get("apply") or "auto").lower(),
            credible=bool(out.get("survived")) and bool(out.get("beats_default")),
            verdict=str(out.get("stopped_reason") or ""),
            evidence={"forward_m": out.get("forward_m"), "default_forward_m": out.get("default_forward_m"),
                      "n_evals": out.get("n_evals"), "robustness_rel": out.get("robustness_rel")})
    else:
        out["applied_to_robot"] = {"applied": False, "reason": "no robot_id was given, so this gait was learned "
                                                              "for a body composed on the fly and there is no "
                                                              "held robot to apply it to; pass robot_id to train "
                                                              "the robot you are actually holding"}
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
        "description": "PolicyImporter Tier P3: INSPECT and VALIDATE an inbound ONNX policy -- load it (a "
                       "computation GRAPH, safe, unlike a torch pickle), read its declared input/output tensor "
                       "contract, and, given an observation, run ONE inference and check the action's "
                       "dimension/finiteness/safety-limits. Pass robot_id and the action width is checked "
                       "against that held robot's actual actuator count. IT DOES NOT DEPLOY THE POLICY: nothing "
                       "here makes it the robot's controller, verify_robot afterwards still measures OUR "
                       "scripted controller on your body, and the result says so (`deployed: false` + a "
                       "`deployment` block naming what an .onnx cannot tell us -- observation layout, joint "
                       "order, torque-vs-position, action scale). To get a controller that DOES deploy: "
                       "train_held with mode='gpu_rl' / train_reward with train_backend='gpu' (a native "
                       "MorphPolicy, banked and then deployed by verify_robot), or adopt_control_script if your "
                       "controller is parameterised. No physics.",
        "parameters": {"type": "object", "required": ["path"], "properties": {
            "path": {"type": "string", "description": "a .onnx policy file"},
            "observation": {"type": "array", "description": "one observation vector (omit to just inspect IO)"},
            "robot_id": {"type": "string", "description": "a held robot to validate the action width against "
                                                          "(its MuJoCo actuator count becomes action_dim)"},
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
                       "walk (1.8x). Real MuJoCo, slow. The improved controller is COMMITTED to the held robot "
                       "when it credibly beat the user's own (so the next verify_robot reports gait_source "
                       "'tuned_for_this_body::adopt_control_script'); otherwise the customer's controller stands "
                       "and applied_to_robot says why. Undo with edit_robot op:'undo'.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string", "description": "the held robot to run/improve the controller on"},
            "script_path": {"type": "string", "description": "a .py control script (a sibling policy_params.json is auto-found)"},
            "params_path": {"type": "string", "description": "a policy_params.json (freq/amplitude/... )"},
            "params": {"type": "object", "description": "inline control params (alternative to a path)"},
            "apply": {"type": "string", "enum": ["auto", "always", "never"], "default": "auto",
                      "description": "'auto' commits the improved controller only when it credibly beat the "
                                     "imported one; 'never' returns it without touching the robot; 'always' "
                                     "commits it regardless and says so"},
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
                       "CONSTRUCTION (it IS the deploy controller) -- and with a robot_id it is COMMITTED to that "
                       "held robot, so the next verify_robot measures the gait you just learned and reports "
                       "gait_source 'tuned_for_this_body::learn_gait' (undo with edit_robot op:'undo'; pass "
                       "apply:'never' for a dry run). Consults the flywheel for a prior on this morphology "
                       "first and banks a credible result after. Returns the best gait + measured "
                       "forward/height. Real MuJoCo, slow -- the full budget; adapt_gait is the cheap version.",
        "parameters": {"type": "object", "properties": {
            "robot_id": {"type": "string", "description": "learn for THIS held robot (what verify_robot runs); "
                                                          "omit to compose from prompt -- but then there is no "
                                                          "held robot for the result to land on"},
            "prompt": {"type": "string", "default": "a quadruped robot dog"},
            "apply": {"type": "string", "enum": ["auto", "always", "never"], "default": "auto",
                      "description": "'auto' commits the learned gait to the held robot when it beat the default "
                                     "and survived; 'never' returns it without touching the robot; 'always' "
                                     "commits it regardless and says so"},
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
