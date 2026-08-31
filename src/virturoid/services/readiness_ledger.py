"""Build + write the Product Readiness Ledger by running HONEST per-stage probes over a built package.

Each probe answers "is this stage provably real?" by inspecting artifacts on disk (and, for physics, the
gene). Probes REUSE existing checks rather than reinventing: ``cad_geometry.is_real_cad`` for CAD, a rendered-
pixel scan for CV, ``gene_validation.validate_gene_design`` + a spawn-penetration check for physics. Probes
are best-effort: a missing optional dependency yields an honest non-real status (scaffolded/not_run), never a
crash and never a false 'attained'. The default landing is EMIT-ONLY (``enforce=False``): the ledger reports
the truth; flipping ``enforce=True`` later turns a dishonest 'ready' into a build-time error.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.readiness_ledger import (
    ATTAINED, BELOW_GATE, COLLISION, DRY_RUN_ONLY, NOT_REQUIRED, NOT_RUN, PLACEHOLDER, SCAFFOLDED,
    ProductReadinessLedger, StageRecord,
)

# CV artifacts that count as REAL rendered data (not JSON manifests).
_PIXEL_SUFFIXES = (".png", ".exr", ".pcd", ".npy", ".npz", ".jpg", ".jpeg")

# ALLOWLIST of grasp models that count as a REAL (non-idealized) physics grasp. The honesty gate fails CLOSED:
# only these certify; an idealized pin, a missing disclosure, or an unknown value reads BELOW_GATE.
_REAL_GRASP_MODELS = {"contact"}

# the 'actually walked' distance bar — shared with gene_build._WALK_MIN_FORWARD_M so the readiness gate and the
# build's own 'walked' label can never disagree (the gate used to be more lenient at 0.1 m).
_WALK_MIN_FORWARD_M = 0.15


def build_product_readiness_ledger(package_dir, *, robot_class: str = "", gene=None,
                                   require=None, enforce: bool = False) -> ProductReadinessLedger:
    """Probe a package directory and return its honest ledger. ``gene`` (a RobotGene) enables the real
    physics/spawn probe on the gene path. ``require`` overrides which stages gate ``safe_to_export``."""
    pkg = Path(package_dir)
    if require is None:
        # Default required gates: a build is 'safe' only with real CAD + a real (non-dry-run) physics pass on
        # top of schema validity + compiled sim + a REAL BOM (G2, fidelity gap-closure: the e2e test caught
        # packages with ZERO parts list marked safe_to_export=True — a robot with no parts cannot be built).
        # CV + controller are required only when present/requested.
        require = ["schema_valid", "sim_compiled", "real_cad_exported", "physics_evaluated", "bom_present",
                   "actuators_feasible"]
    stages = [
        _probe_schema_valid(pkg),
        _probe_sim_compiled(pkg),
        _probe_real_cad(pkg),
        _probe_rendered_cv(pkg),
        _probe_physics(pkg, gene),
        _probe_controller(pkg),
        _probe_bom(pkg),
        _probe_actuators_feasible(pkg),
    ]
    require = list(require)
    # A control stack that FAILED its datasheet-torque audit must block safe_to_export, not merely be noted:
    # this is the artifact a customer runs on hardware. Scoped to the failure so a passing/absent audit never
    # adds a gate a build could not previously have had to clear.
    ctrl = next((s for s in stages if s.stage == "controller_exported"), None)
    if ctrl is not None and ctrl.status == BELOW_GATE and "controller_exported" not in require:
        require.append("controller_exported")
    return ProductReadinessLedger(package_dir=str(pkg), robot_class=robot_class, stages=stages,
                                  required=require, enforce=enforce)


def write_product_readiness_ledger(package_dir, *, robot_class: str = "", gene=None,
                                   require=None, enforce: bool = False) -> ProductReadinessLedger:
    """Build the ledger, write ``reports/product_readiness_ledger.json``, and (when ``enforce``) RAISE if a
    required stage is provably fake — the single honest chokepoint. Returns the ledger."""
    ledger = build_product_readiness_ledger(package_dir, robot_class=robot_class, gene=gene,
                                            require=require, enforce=enforce)
    out = Path(package_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "product_readiness_ledger.json").write_text(json.dumps(ledger.to_dict(), indent=2), encoding="utf-8")
    if enforce:
        issues = ledger.validate()
        if issues:
            raise ValueError("product readiness ledger BLOCKED export — dishonest 'ready' over fake stages: "
                             + "; ".join(issues))
    return ledger


# --------------------------------------------------------------------------- probes

# Categories that make a robot ACTUATED / able to move: joint actuators (arm/legged), drive motors (mobile),
# rotors (aerial). A body needs at least one — but which KIND depends on its morphology, so the gate must accept
# all of them (a drone is propelled by ROTORS not joint actuators; a rover by DRIVE MOTORS — both are complete).
_PROPULSION_CATEGORIES = ("actuator", "drive_motor", "rotor")


def _probe_bom(pkg: Path) -> StageRecord:
    """G2 (fidelity gap-closure): a package is only buildable with a REAL bill of materials — present, parseable,
    with at least one PROPULSION line (a joint actuator, a drive motor, OR a rotor — whichever its morphology
    uses to move). Fail-closed: no BOM / empty lines / nothing that moves it -> not attained."""
    p = pkg / "robot" / "bill_of_materials.json"
    if not p.is_file():
        return StageRecord("bom_present", PLACEHOLDER, "no robot/bill_of_materials.json — a robot with no parts "
                                                       "list cannot be built")
    try:
        bom = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return StageRecord("bom_present", PLACEHOLDER, "bill_of_materials.json does not parse")
    lines = bom.get("lines") or bom.get("items") or []
    n_act = sum(1 for ln in lines if (ln.get("category") or ln.get("kind")) in _PROPULSION_CATEGORIES)
    if not lines:
        return StageRecord("bom_present", PLACEHOLDER, "BOM has zero line items")
    if n_act == 0:
        return StageRecord("bom_present", BELOW_GATE,
                           f"{len(lines)} lines but NO propulsion (actuator/drive-motor/rotor) lines")
    return StageRecord("bom_present", ATTAINED, f"{len(lines)} lines incl. {n_act} propulsion line(s)",
                       evidence={"line_items": len(lines), "propulsion_lines": n_act})


def _probe_schema_valid(pkg: Path) -> StageRecord:
    """Core artifacts exist + parse. Lenient across the MVP and gene package layouts."""
    candidates = [pkg / "robot" / "robot_genome.json",
                  pkg / "reports" / "package_validation_report.json"]
    found = [c for c in candidates if c.is_file()]
    if not found:
        # any json that parses under the package is a minimal signal the build wrote structured artifacts
        anyjson = next((p for p in pkg.rglob("*.json")), None)
        if anyjson is None:
            return StageRecord("schema_valid", SCAFFOLDED, "no structured artifacts written")
    for c in found:
        try:
            json.loads(c.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return StageRecord("schema_valid", PLACEHOLDER, f"{c.name} does not parse")
    # If a package_validation_report exists, honor its ok flag.
    pvr = pkg / "reports" / "package_validation_report.json"
    if pvr.is_file():
        try:
            ok = bool(json.loads(pvr.read_text(encoding="utf-8")).get("ok"))
            return StageRecord("schema_valid", ATTAINED if ok else PLACEHOLDER,
                               "package_validation_report.ok" if ok else "package validation failed")
        except Exception:  # noqa: BLE001
            pass
    return StageRecord("schema_valid", ATTAINED, "core artifacts parse")


def _probe_sim_compiled(pkg: Path) -> StageRecord:
    """Scenes compiled to MJCF. Attained if a compiled scene index lists scenes (and >=1 xml exists)."""
    idx = next((p for p in pkg.rglob("compiled_scene_index.json")), None)
    if idx is None:
        return StageRecord("sim_compiled", NOT_RUN, "no compiled_scene_index.json")
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return StageRecord("sim_compiled", PLACEHOLDER, "compiled_scene_index.json does not parse")
    scenes = data.get("scenes") or []
    xml_ok = any((pkg / s.get("mujoco_xml", "")).is_file() for s in scenes)
    if scenes and xml_ok:
        return StageRecord("sim_compiled", ATTAINED, f"{len(scenes)} compiled scenes",
                           {"scene_count": len(scenes)})
    return StageRecord("sim_compiled", PLACEHOLDER, "index lists no loadable scene xml")


def _probe_real_cad(pkg: Path) -> StageRecord:
    """Real B-rep STEP present (not the 479-byte placeholder). Uses cad_geometry.is_real_cad."""
    steps = list(pkg.rglob("*.step"))
    if not steps:
        return StageRecord("real_cad_exported", NOT_RUN, "no .step files")
    try:
        from virturoid.services.cad_geometry import is_real_cad
    except Exception:  # noqa: BLE001
        return StageRecord("real_cad_exported", NOT_RUN, "cad checker unavailable")
    real = [s for s in steps if is_real_cad(s)]
    if real:
        return StageRecord("real_cad_exported", ATTAINED, f"{len(real)}/{len(steps)} STEP files are real B-rep",
                           {"real_step_count": len(real), "total_step_count": len(steps)})
    return StageRecord("real_cad_exported", PLACEHOLDER,
                       f"{len(steps)} STEP file(s) are placeholders/stubs (no real B-rep)")


def _probe_rendered_cv(pkg: Path) -> StageRecord:
    """Real rendered pixels/point-clouds on disk, not just a manifest. not_required when no CV was requested."""
    obs_dirs = [p for p in pkg.rglob("synthetic_observations") if p.is_dir()]
    obs_dirs += [p for p in pkg.rglob("observations") if p.is_dir()]
    manifest = next((p for p in pkg.rglob("synthetic_observation_manifest.json")), None)
    pixels = []
    for d in obs_dirs:
        for f in d.rglob("*"):
            if f.suffix.lower() in _PIXEL_SUFFIXES and f.is_file() and f.stat().st_size > 256:
                pixels.append(f)
                break
        if pixels:
            break
    if pixels:
        return StageRecord("rendered_cv_data", ATTAINED, "rendered pixel/point-cloud files present")
    if manifest is not None or obs_dirs:
        return StageRecord("rendered_cv_data", PLACEHOLDER,
                           "CV manifest/contract present but NO rendered pixel files on disk")
    return StageRecord("rendered_cv_data", NOT_REQUIRED, "no perception/CV stage in this build")


def _probe_actuators_feasible(pkg: Path) -> StageRecord:
    """B3d (2026-07-24 audit): surface the BOM's power/torque feasibility as a GATE, not a buried note.

    ``bom_certificate.json`` (bom_certificate.certify_policy_on_bom) grades every joint's measured torque/speed
    demand against the REAL selected servo across 6 gates (peak torque, thermal-continuous, speed, torque-speed
    envelope, mechanical power, selection headroom). Before this it sat on disk and nothing gated on it, so a
    robot whose actuators can't physically supply the demanded power still exported green. Now: cert present +
    ``pass:true`` -> ATTAINED; present + ``pass:false`` -> BELOW_GATE (blocks export, with the failing gate
    named); absent -> NOT_REQUIRED (no cert was run -> stay silent, never block a body we didn't assess)."""
    cert = next((p for p in pkg.rglob("bom_certificate.json")), None)
    if cert is None:
        return StageRecord("actuators_feasible", NOT_REQUIRED, "no BOM feasibility certificate in this build")
    try:
        data = json.loads(cert.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return StageRecord("actuators_feasible", NOT_REQUIRED, "bom_certificate.json does not parse")
    if data.get("pass") is True:
        return StageRecord("actuators_feasible", ATTAINED,
                           f"real BOM actuators cleared all feasibility gates ({data.get('n_gates_pass')}/6)",
                           {"n_gates_pass": data.get("n_gates_pass")})
    failed = [g.get("name") for g in (data.get("gates") or []) if not g.get("passed")]
    return StageRecord("actuators_feasible", BELOW_GATE,
                       "BOM actuators are power/torque INFEASIBLE for the demanded motion: "
                       + (", ".join(str(f) for f in failed) or "below feasibility threshold"),
                       {"n_gates_pass": data.get("n_gates_pass"), "failed_gates": failed})


def _requested_task_failure(pkg: Path) -> StageRecord | None:
    """B3d: return a BELOW_GATE verdict if the REQUESTED-task evaluation ran for real and fell below its bar,
    else None. This is what makes gate strictness CONSISTENT across archetypes -- the task the user asked for
    (in gene_evaluation_report / physics_evaluation_report) vetoes export whether it's locomotion (didn't walk)
    or a scored task (<25% success), so a generic grasp-lift or nav certification can never mask a failing
    requested task. Returns None when the requested task PASSED or ran only as a dry-run (let the normal flow
    decide), so this only ever BLOCKS, never upgrades."""
    gene_eval = next((p for p in pkg.rglob("gene_evaluation_report.json")), None)
    phys = next((p for p in pkg.rglob("physics_evaluation_report.json")), None)
    for rep in (phys, gene_eval):
        if rep is None:
            continue
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if str(data.get("adapter_name", "")).lower().startswith("dry_run"):
            continue                                        # a dry-run isn't a real verdict -> don't veto on it
        if str(data.get("task_type", "")) == "locomotion":
            fwd = float(data.get("forward_m", 0.0)); up = bool(data.get("upright", False))
            status = str(data.get("status", ""))
            walked = (status == "walked") if status else (fwd >= _WALK_MIN_FORWARD_M and up)
            if not walked:
                return StageRecord("physics_evaluated", BELOW_GATE,
                                   f"requested locomotion did not walk (forward {fwd:.2f} m, status "
                                   f"'{status or up}') in {rep.name}",
                                   {"forward_m": fwd, "upright": up, "status": status})
            return None
        sr = data.get("success_rate")
        if sr is not None and float(sr) < 0.25:
            return StageRecord("physics_evaluated", BELOW_GATE,
                               f"requested task success only {float(sr):.0%} in {rep.name}", {"success_rate": sr})
        return None
    return None


def _probe_physics(pkg: Path, gene) -> StageRecord:
    """A REAL (non-dry-run) physics evaluation + a passed spawn-penetration check. Uses the gene when given."""
    # spawn penetration via the gene-design gate (fatal/high stability/kinematic flags == not physics-ready).
    if gene is not None:
        try:
            from virturoid.services.gene_validation import validate_gene_design
            report = validate_gene_design(gene)
            if report["checks"].get("stability") is False:
                return StageRecord("physics_evaluated", COLLISION,
                                   "rest pose is statically unstable / penetrates (gene_validation)")
            bad = [f for f in report["risk_flags"] if f["severity"] in ("fatal", "high")]
            if bad:
                return StageRecord("physics_evaluated", COLLISION,
                                   "gene_validation flagged: " + bad[0]["detail"], {"flags": len(bad)})
        except Exception:  # noqa: BLE001
            pass
    # §15.3 EXPORT GATE (strongest signal): a gated EvaluationRun (evaluation_loop.gated_evaluation_run) wrote
    # an explicit pass/fail. "Physics ran" is NOT "task passed" — if the real tiered run failed the gate
    # (baseline/randomized below threshold), this stage is BELOW_GATE and blocks safe_to_export.
    ev = next((p for p in pkg.rglob("evaluation_run.json")), None)
    if ev is not None:
        try:
            data = json.loads(ev.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = None
        if isinstance(data, dict) and "export_ready" in data:
            evidence = {"tier_success": data.get("tier_success"),
                        "regression_cause_confirmed_rate": data.get("regression_cause_confirmed_rate")}
            if data.get("export_ready") is True:
                return StageRecord("physics_evaluated", ATTAINED,
                                   "real tiered evaluation cleared the §15.3 export gate", evidence)
            return StageRecord("physics_evaluated", BELOW_GATE,
                               "real physics ran but task success is below the export gate: "
                               + "; ".join(data.get("export_blockers") or ["below threshold"]), evidence)

    # B3d (2026-07-24 audit): the REQUESTED task is authoritative for the gate. Before crediting a GENERIC
    # capability certification (a contact grasp+lift of an arbitrary block, or a nav route), veto on the task the
    # user actually asked for: an arm asked to sort/transport that scored ~0% must be EXPORT-BLOCKED exactly like
    # a dog that won't walk — a passing generic grasp cert cannot mask a failing requested task. (Was: the grasp
    # cert was checked first and short-circuited to ATTAINED, so the arm exported green while the dog didn't —
    # inconsistent strictness.) The §15.3 gated run above already decides when present, so this only applies when
    # there is no explicit gated verdict.
    veto = _requested_task_failure(pkg)
    if veto is not None:
        return veto

    # A REAL (non-pinned) contact grasp+lift, certified by grasp_eval.evaluate_grasp_lift (closes the fingers and
    # lifts the object by FRICTION — no _pin_block). This is the honest manipulation capability: if it passed, the
    # physics stage ATTAINS on a real grasp even though the higher-level sort/transport demo still uses the pin.
    nav_rep = next((p for p in pkg.rglob("navigation_evaluation_report.json")), None)
    if nav_rep is not None:
        try:
            nd = json.loads(nav_rep.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            nd = {}
        sr = nd.get("success_rate")
        if sr is not None:
            evidence = {
                "success_rate": sr,
                "reached": nd.get("reached"),
                "total_episodes": nd.get("total_episodes"),
            }
            if float(sr) >= 0.8:
                return StageRecord("physics_evaluated", ATTAINED,
                                   f"real navigation cleared the route gate in {nav_rep.name}", evidence)
            return StageRecord("physics_evaluated", BELOW_GATE,
                               f"real navigation only reached {float(sr):.0%} in {nav_rep.name}", evidence)

    grasp_rep = next((p for p in pkg.rglob("grasp_evaluation_report.json")), None)
    if grasp_rep is not None:
        try:
            gd = json.loads(grasp_rep.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            gd = {}
        gm = str(gd.get("grasp_model", "")); gsr = gd.get("success_rate")
        if gm in _REAL_GRASP_MODELS and gsr is not None:
            if float(gsr) >= 0.5:
                return StageRecord("physics_evaluated", ATTAINED,
                                   f"real CONTACT grasp+lift certified ({float(gsr):.0%} success, no pin) in {grasp_rep.name}",
                                   {"success_rate": gsr, "grasp_model": gm, "mean_lift_m": gd.get("mean_lift_m")})
            return StageRecord("physics_evaluated", BELOW_GATE,   # a real grasp ran but didn't clear the bar
                               f"real contact grasp ran but only {float(gsr):.0%} success in {grasp_rep.name}",
                               {"success_rate": gsr, "grasp_model": gm})

    # a real evaluation report (gene path) or physics_evaluation_report (mvp path), NOT a dry-run.
    gene_eval = next((p for p in pkg.rglob("gene_evaluation_report.json")), None)
    phys = next((p for p in pkg.rglob("physics_evaluation_report.json")), None)
    for rep in (phys, gene_eval):
        if rep is None:
            continue
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        adapter = str(data.get("adapter_name", "")).lower()
        if adapter.startswith("dry_run"):
            return StageRecord("physics_evaluated", DRY_RUN_ONLY, f"{rep.name} is a dry-run result")
        # A REAL evaluation the robot FAILED is not export-ready physics. A walker that fell or drifted backward,
        # or a task at ~0% success, must read BELOW_GATE, not ATTAINED — otherwise safe_to_export would be True
        # over a robot that doesn't do its task.
        if str(data.get("task_type", "")) == "locomotion":
            fwd = float(data.get("forward_m", 0.0)); up = bool(data.get("upright", False))
            status = str(data.get("status", ""))
            # ATTAINED only for a GENUINE walk. Prefer the honest 'status' the build now writes (already gated on
            # sustained-upright + a real foot cadence + distance >= _WALK_MIN_FORWARD_M); fall back to the shared
            # distance bar for older reports. A 0.10-0.15 m drift or an upright slide no longer certifies (the gate
            # was more lenient (0.1 m) than the build's own 'walked' label — now unified).
            walked = (status == "walked") if status else (fwd >= _WALK_MIN_FORWARD_M and up)
            if walked:
                return StageRecord("physics_evaluated", ATTAINED, f"walked forward upright in {rep.name}",
                                   {"forward_m": fwd, "upright": up, "status": status})
            return StageRecord("physics_evaluated", BELOW_GATE,
                               f"did not walk (forward {fwd:.2f} m, status '{status or up}') in {rep.name}",
                               {"forward_m": fwd, "upright": up, "status": status})
        sr = data.get("success_rate")
        if sr is not None and float(sr) < 0.25:
            return StageRecord("physics_evaluated", BELOW_GATE, f"task success only {float(sr):.0%} in {rep.name}",
                               {"success_rate": sr})
        # GRASP HONESTY GATE — fail CLOSED via an ALLOWLIST of known-real grasp models. A manipulation report
        # ATTAINS only if it discloses a real (non-pinned) grasp; an IDEALIZED pin (block teleported to the gripper),
        # a MISSING disclosure, or any unrecognized value reads BELOW_GATE. (A blocklist on just "idealized_pin"
        # would silently ATTAIN a report that simply dropped the disclosure — the fail-OPEN hole the audit found.)
        gm = str(data.get("grasp_model", "")); task = str(data.get("task_type", ""))
        is_manip = bool(gm) or any(w in task for w in ("pick_place", "grasp", "sort", "place", "transport"))
        if is_manip and gm not in _REAL_GRASP_MODELS:
            why = ("the grasp is IDEALIZED (block pinned to the gripper on contact), not a real full-contact grasp"
                   if gm == "idealized_pin" else
                   f"the manipulation grasp model is undisclosed/unrecognized ('{gm or 'missing'}')")
            return StageRecord("physics_evaluated", BELOW_GATE, f"real physics ran in {rep.name} but {why}",
                               {"success_rate": sr, "grasp_model": gm or None})
        return StageRecord("physics_evaluated", ATTAINED, f"real evaluation in {rep.name}",
                           {"success_rate": sr})
    return StageRecord("physics_evaluated", NOT_RUN, "no real physics evaluation report")


def _control_stack_torque_failure(pkg: Path) -> StageRecord | None:
    """Return a BELOW_GATE verdict if the shipped control stack FAILED its datasheet-torque audit, else None.

    The control stage used to attain on FILE PRESENCE alone, while the compiler's docstring and the deployment
    guide both told the customer a command past a joint's datasheet torque had been checked for. Nothing ran that
    check, so the ladder was crediting a green it had not earned. Now the ledger reads the audit's own verdict in
    ``reports/script_validation.json -> torque_audit`` and refuses to call the stage real when it failed. A
    'no_reference'/absent audit does NOT downgrade the stage (nothing was measured, so nothing is claimed) --
    it is disclosed in the detail instead."""
    val = next((p for p in pkg.rglob("script_validation.json")), None)
    if val is None:
        return None
    try:
        audit = (json.loads(val.read_text(encoding="utf-8")) or {}).get("torque_audit") or {}
    except Exception:  # noqa: BLE001
        return None
    if audit.get("status") != "fail":
        return None
    viol = [str(v) for v in (audit.get("violations") or [])]
    return StageRecord("controller_exported", BELOW_GATE,
                       "the shipped control stack FAILED its datasheet-torque audit: "
                       + ("; ".join(viol[:3]) or "a command could exceed a joint's actuator peak torque"),
                       {"torque_audit_violations": len(viol), "detail": viol[:5]})


def _probe_controller(pkg: Path) -> StageRecord:
    """A trained/exported controller bundle whose companion control stack passed the datasheet-torque audit.
    not_required when training wasn't requested (no bundle present)."""
    veto = _control_stack_torque_failure(pkg)
    if veto is not None:
        return veto                      # a stack that can overdrive a motor is not an exported controller
    bundle = next((p for p in pkg.rglob("controller_bundle.json")), None)
    ctrl = next((p for p in pkg.rglob("controller.py")), None)
    val = next((p for p in pkg.rglob("script_validation.json")), None)
    astat, n_joints = None, 0
    if val is not None:
        try:
            _a = ((json.loads(val.read_text(encoding="utf-8")) or {}).get("torque_audit") or {})
            astat, n_joints = _a.get("status"), int(_a.get("n_joints") or 0)
        except Exception:  # noqa: BLE001
            astat = None
    # A pass over ZERO joints is not a passed safety check -- every check is trivially true over an empty set.
    # It must read as NOT APPLICABLE here too, or the ledger re-launders the vacuous green the audit refused.
    audited = astat == "pass" and n_joints > 0
    if audited:
        note = " + control stack passed the datasheet-torque audit"
    elif astat == "not_applicable" or (astat == "pass" and n_joints == 0):
        note = " (no actuated joints: the datasheet-torque audit does not apply -- nothing was checked)"
    elif val is not None:
        note = " (control stack present but its datasheet-torque audit did NOT run)"
    else:
        note = ""
    if bundle is not None or ctrl is not None:
        return StageRecord("controller_exported", ATTAINED, "controller bundle present" + note,
                           {"torque_audit_passed": audited})
    return StageRecord("controller_exported", NOT_REQUIRED, "no trained controller requested" + note,
                       {"torque_audit_passed": audited})
