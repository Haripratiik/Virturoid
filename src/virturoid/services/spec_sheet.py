"""Robot spec sheet: an auto-derived capability + cost summary for a built package.

A user who builds a robot wants to know, at a glance, what it actually IS and can DO -- mass, footprint, power
draw, parts cost, actuator torque, sensors, and task performance. This aggregates the data the build already
produces (the Robot Genome + the Bill of Materials + the physics evaluation / verification certificate) into one
``spec_sheet.json`` + human-readable ``spec_sheet.md``. Pure read-over-artifacts; no rebuild, no training.

Three rules, all learned from packages that actually shipped:

1. LOOK EVERYWHERE THE PIPELINE ACTUALLY WRITES. Nothing here is computed fresh -- every number already exists
   in the package. This sheet used to look for exactly ONE BOM filename (``robot/bill_of_materials.json``),
   which is what the *package builder* writes -- while ``export_held`` writes ``bom.json`` at the package root
   and the MVP/blueprint writers write ``bom/bom.json``. So every ``export_held`` package shipped a
   ``reports/spec_sheet.json`` whose mass, actuators, size, power, cost and peak torque were ALL ``null`` and
   whose sensing/compute lists were empty, with a fully populated ``bom.json`` sitting one directory away. The
   same disconnect hit the model (``simulation/**.xml`` vs a root ``robot.xml``) and the performance numbers
   (``reports/gene_evaluation_report.json`` vs ``verification_certificate.json``). Read all of them.

2. NEVER SHIP A BARE NULL. A number we genuinely cannot derive gets an entry in ``unavailable``: the dotted
   field path mapped to what is missing and what the engineer can run to get it. The markdown prints that
   sentence in the table cell instead of ``None``. ``| Mass | None kg |`` tells a robotics engineer nothing;
   "no bill of materials in this package -- re-export with the 'bom' format" tells them what to do next. The
   value slots stay number-or-null so the typed consumers (Studio's Inspector, the UI package list) keep
   formatting them as numbers; the reason travels beside them, never instead of them.

3. SAY IT LOUDLY WHEN THE PACKAGE CONTRADICTS ITSELF. ``consistency`` cross-checks this sheet against the BOM
   and the verification certificate travelling in the SAME package: robot class, DOF, the actual motor
   selection, peak torque, and the structural mass the walk was verified at. A package once shipped two
   different motor selections; a spec sheet that quietly averages over a contradiction is worse than none.

MASS IS TWO NUMBERS, ON PURPOSE. ``mass_kg`` is the AS-BUILT parts mass from the BOM (structure + motors +
electronics + power). The physics verdict is measured on the simulated body, which carries the STRUCTURE only
-- the sim body does not embody the actuators or the battery. Those two numbers legitimately differ (often
~2x), so both are printed, labelled, and the ratio is stated. Reporting only one of them is how a spec sheet
implies a robot was verified at a mass it will never have.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SCENE_WORDS = ("floor", "ground", "plane", "block", "cube", "goal", "obstacle", "wall", "table", "target", "bin")

# Every exit door writes these under a different name/location. See rule 1 above.
_BOM_NAMES = ("bill_of_materials.json", "bom.json")
_EVAL_NAMES = ("gene_evaluation_report.json",)
_CERT_NAMES = ("verification_certificate.json",)
_GENOME_NAMES = ("robot_genome.json",)

# XML in a robot package is not necessarily a MuJoCo model: the ROS2/ament package manifest is XML too, and
# sorts FIRST under ``export/``. Skip the known non-models so the bounding box is taken from the real model.
_NOT_MJCF = ("package.xml", "pom.xml", "plugin.xml", "manifest.xml", "catkin.xml")

_ACTUATOR_CATEGORIES = ("actuator", "drive_motor", "rotor")
_SENSOR_CATEGORIES = ("camera", "imu", "lidar", "sensor", "force_torque", "depth")

_NO_BOM = ("no bill of materials in this package (looked for {names}) -- re-export including the 'bom' format, "
           "or rebuild the package, to get mass / parts cost / power draw / actuator torque")


def _first(output_dir: Path, *names: str) -> dict | None:
    """The first PARSEABLE json under ``output_dir`` matching any of ``names``, in the order given.

    Multiple names because the same artifact is written under different names by different exit doors; a
    parse failure falls through to the next candidate rather than sinking the whole sheet.
    """
    for name in names:
        for hit in sorted(Path(output_dir).rglob(name)):
            try:
                loaded = json.loads(hit.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(loaded, dict):
                return loaded
    return None


def _model_candidates(output_dir: Path) -> list[Path]:
    """Every plausible MuJoCo model in the package, best-first: the package builder puts it under
    ``simulation/``, ``export_held`` writes ``robot.xml`` at the package root, and the URDF writer uses
    ``robot/``. Trying only the first location is what left ``size_m`` null on every export."""
    root = Path(output_dir)
    seen: set[Path] = set()
    out: list[Path] = []
    for group in (sorted((root / "simulation").rglob("*.xml")),
                  sorted(root.glob("*.xml")),
                  sorted((root / "robot").rglob("*.xml")),
                  sorted(root.rglob("*.xml"))):
        for p in group:
            if p.name.lower() in _NOT_MJCF or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _extent_of(path: Path) -> dict | None:
    """Robot bounding box (length/width/height in m) from one compiled MJCF, excluding scene props
    (floor/blocks/goal/...) so the size reflects the ROBOT, not the task scene."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    keep = []
    for g in range(model.ngeom):
        bid = int(model.geom_bodyid[g])
        if bid == 0:
            continue  # worldbody (floor/lights)
        nm = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
        if any(w in nm for w in _SCENE_WORDS):
            continue
        keep.append(g)
    if not keep:
        return None
    pos = np.asarray(data.geom_xpos)[keep]
    rad = np.asarray(model.geom_rbound)[keep]  # bounding radius per geom
    lo = (pos - rad[:, None]).min(axis=0)
    hi = (pos + rad[:, None]).max(axis=0)
    ext = hi - lo
    return {"length_m": round(float(ext[0]), 3), "width_m": round(float(ext[1]), 3),
            "height_m": round(float(ext[2]), 3)}


def _robot_extent(output_dir: Path) -> tuple[dict | None, str | None]:
    """``(size dict, None)`` on success, or ``(None, actionable reason)``. Never raises: the size is a
    nicety, but a MISSING size still owes the reader an explanation rather than a null."""
    try:
        import mujoco  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return None, (f"not derived: the mujoco package is not importable here ({type(exc).__name__}) -- "
                      "install mujoco to get the robot's bounding box from the shipped model")
    cands = _model_candidates(Path(output_dir))
    if not cands:
        return None, ("not derived: this package ships no MuJoCo model (looked under simulation/, the package "
                      "root and robot/) -- re-export including the 'mjcf' format to get the bounding box")
    errs = []
    for p in cands[:6]:
        try:
            ext = _extent_of(p)
        except Exception as exc:  # noqa: BLE001 - try the next candidate model
            errs.append(f"{p.name}: {type(exc).__name__}")
            continue
        if ext:
            return ext, None
        errs.append(f"{p.name}: no robot geoms (all filtered as scene props)")
    return None, (f"not derived: {len(cands)} XML file(s) in this package but none yielded a robot bounding box "
                  f"({'; '.join(errs[:3])}) -- re-export the 'mjcf' format")


def _bom_facts(bom: dict) -> dict:
    """Everything the spec sheet needs out of the BOM, in the BOM's own units. No re-derivation, no margins."""
    lines = [ln for ln in (bom.get("lines") or []) if isinstance(ln, dict)]
    torques = []
    for line in lines:
        if line.get("category") in _ACTUATOR_CATEGORIES:
            mt = re.search(r"peak\s+([\d.]+)\s*Nm", str(line.get("detail", "")))
            if mt:
                torques.append(float(mt.group(1)))
    return {
        "lines": lines,
        "torques": torques,
        # propulsion = whatever moves THIS body: joint actuators (arm/legged), drive motors (mobile), rotors
        # (aerial) -- so a drone's spec shows its rotors and a rover's its drive motors, not an empty section.
        "actuator_types": sorted({str(ln["part"]) for ln in lines
                                  if ln.get("category") in _ACTUATOR_CATEGORIES and ln.get("part")}),
        "sensors": sorted({str(ln["part"]) for ln in lines
                           if ln.get("category") in _SENSOR_CATEGORIES and ln.get("part")}),
        "compute": sorted({str(ln["part"]) for ln in lines
                           if ln.get("category") == "compute" and ln.get("part")}),
        # the STRUCTURE-only mass: this, not the full parts mass, is what the simulated body weighs, so it is
        # the number that must agree with the certificate. See the module docstring on mass.
        "structure_mass_kg": (round(sum(float(ln.get("mass_kg") or 0.0)
                                        for ln in lines if ln.get("category") == "material"), 3)
                              if any(ln.get("category") == "material" for ln in lines) else None),
        "actuator_selection": sorted((str(ln.get("part")), int(ln.get("qty") or 0))
                                     for ln in lines if ln.get("category") == "actuator"),
    }


def _genome_dof(genome: dict) -> int | None:
    joints = [j for j in (genome.get("joints") or []) if isinstance(j, dict)]
    if not joints:
        return None
    actuated = [j for j in joints if str(j.get("joint_type", "")).lower() in ("revolute", "prismatic")]
    return len(actuated) or len(joints)   # untyped legacy genomes list only actuated joints


def _cert_checks(cert: dict) -> dict:
    v = cert.get("verdict") if isinstance(cert.get("verdict"), dict) else {}
    return v.get("checks") if isinstance(v.get("checks"), dict) else {}


def build_spec_sheet(output_dir) -> dict:
    """Aggregate genome + BOM + evaluation + verification certificate into a capability/cost spec dict."""
    output_dir = Path(output_dir)
    genome = _first(output_dir, *_GENOME_NAMES) or {}
    bom = _first(output_dir, *_BOM_NAMES) or {}
    ev = _first(output_dir, *_EVAL_NAMES) or {}
    cert = _first(output_dir, *_CERT_NAMES) or {}
    totals = bom.get("totals") or {}
    facts = _bom_facts(bom)
    unavailable: dict[str, str] = {}
    no_bom = _NO_BOM.format(names=" / ".join(_BOM_NAMES))

    size, size_why = _robot_extent(output_dir)
    if size_why:
        unavailable["physical.size_m"] = size_why
    for field, key in (("physical.mass_kg", "mass_kg"), ("physical.actuators", "actuators"),
                       ("power_and_cost.est_power_draw_w", "est_power_w"),
                       ("power_and_cost.est_parts_cost_usd", "price_usd")):
        if totals.get(key) is None:
            unavailable[field] = no_bom if not bom else (
                f"not in this package's bill of materials: totals.{key} is absent -- rebuild the BOM "
                f"(export format 'bom') so the sheet can quote it")
    if not facts["torques"]:
        unavailable["actuation.peak_joint_torque_nm"] = no_bom if not bom else (
            "not derived: no actuator/drive/rotor line in the BOM carries a 'peak <N> Nm' rating -- the parts "
            "are listed but unrated, so no peak joint torque can be quoted without inventing one")
    if not facts["actuator_types"]:
        unavailable["actuation.actuator_types"] = no_bom if not bom else (
            "empty: the BOM lists no actuator / drive_motor / rotor lines for this body")
    if not facts["sensors"]:
        unavailable["sensing"] = no_bom if not bom else (
            "empty: the BOM selects no camera / IMU / LiDAR / depth / force-torque parts for this robot -- the "
            "sensor suite is task-adaptive, so a task prompt that asks for perception will populate it")
    if not facts["compute"]:
        unavailable["compute"] = no_bom if not bom else (
            "empty: the BOM selects no compute module for this robot")

    perf = _performance(ev, cert, genome, unavailable)

    spec = {
        "name": genome.get("name") or genome.get("species") or bom.get("robot_class"),
        "species": genome.get("species"),
        "robot_class": genome.get("robot_class") or bom.get("robot_class")
                       or ((cert.get("identity") or {}).get("robot_class")),
        "dof": bom.get("dof") or _genome_dof(genome) or (cert.get("identity") or {}).get("dof"),
        "physical": {
            "mass_kg": totals.get("mass_kg"),                     # AS-BUILT: structure + motors + electronics
            "actuators": totals.get("actuators"),
            "size_m": size,
        },
        "power_and_cost": {
            "est_power_draw_w": totals.get("est_power_w"),
            "est_parts_cost_usd": totals.get("price_usd"),
        },
        "actuation": {
            "actuator_types": facts["actuator_types"],
            "peak_joint_torque_nm": round(max(facts["torques"]), 1) if facts["torques"] else None,
        },
        "sensing": facts["sensors"],
        "compute": facts["compute"],
        "performance": perf,
    }
    spec["physical"]["mass_breakdown"] = _mass_breakdown(totals, facts, cert)
    spec["consistency"] = _consistency(spec, genome, bom, facts, cert)
    spec["unavailable"] = unavailable
    spec["sources"] = {k: v for k, v in (("genome", bool(genome)), ("bill_of_materials", bool(bom)),
                                         ("evaluation_report", bool(ev)),
                                         ("verification_certificate", bool(cert)))}
    spec["summary"] = _summary(spec)
    return spec


def _performance(ev: dict, cert: dict, genome: dict, unavailable: dict) -> dict:
    """Task performance, from the evaluation report if the package has one and otherwise from the VERIFICATION
    CERTIFICATE that travels in every exported package -- which carries a signed rollout (verdict, forward
    distance, cadence, support fraction, tilt) that this sheet used to ignore entirely."""
    checks = _cert_checks(cert)
    verdict = cert.get("verdict") if isinstance(cert.get("verdict"), dict) else {}
    task = ev.get("task_type")
    if not task:
        task = "locomotion" if checks.get("forward_m") is not None else genome.get("robot_class")
    perf: dict = {"task": task, "success_rate": ev.get("success_rate")}
    if ev.get("task_type") == "locomotion":
        perf.update({"forward_m": round(float(ev.get("forward_m") or ev.get("distance_m") or 0.0), 2),
                     "cadence_hz": ev.get("cadence_hz"), "gait_stability_frac": ev.get("upright_frac"),
                     "measured_by": "reports/gene_evaluation_report.json"})
    elif checks:
        # The certificate is a real, signed rollout on the body this package ships -- quote it verbatim under
        # its own names rather than re-labelling it as the evaluation report's metrics.
        perf.update({k: checks[k] for k in ("forward_m", "speed_mps", "support_frac", "height_ratio",
                                            "roll_max_deg", "pitch_max_deg", "survived") if k in checks})
        if checks.get("cadence") is not None:
            perf["cadence_hz"] = checks["cadence"]
        perf["verdict"] = verdict.get("verdict")
        perf["credible"] = verdict.get("credible")
        perf["gait_source"] = verdict.get("gait_source")
        perf["measured_by"] = ("verification_certificate.json -- a physics rollout re-measured on the body "
                               "this package ships, at export time")
    if isinstance(ev.get("perception"), dict):
        perf["perception_nav_success"] = ev["perception"].get("success_rate")  # sensed-only reach rate
    if perf.get("success_rate") is None:
        if verdict.get("verdict"):
            unavailable["performance.success_rate"] = (
                f"not scored: this package carries a physics verdict ('{verdict.get('verdict')}'"
                + (f", {checks.get('forward_m')} m forward" if checks.get("forward_m") is not None else "")
                + ") but no task-success evaluation -- a success RATE needs a scored target, so build a full "
                  "package (build_gene_package) or run evaluate_robot to get one")
        else:
            unavailable["performance.success_rate"] = (
                "not measured: this package carries neither reports/gene_evaluation_report.json nor a "
                "verification_certificate.json -- re-export including the 'certificate' format, or build a "
                "full package, to get a measured verdict")
    return perf


def _mass_breakdown(totals: dict, facts: dict, cert: dict) -> dict:
    """As-built parts mass vs the mass the physics verdict was actually measured at. See the module docstring."""
    sim = ((cert.get("model_sanity") or {}) if isinstance(cert.get("model_sanity"), dict) else {}).get(
        "total_mass_kg")
    out: dict = {"as_built_parts_kg": totals.get("mass_kg"),
                 "structure_only_kg": facts["structure_mass_kg"],
                 "simulated_body_kg": sim}
    if totals.get("mass_kg") and sim:
        out["as_built_over_simulated"] = round(float(totals["mass_kg"]) / float(sim), 2)
        out["note"] = ("The physics verdict in this package was measured on the SIMULATED body "
                       f"({sim} kg = structure only); the robot you build weighs {totals['mass_kg']} kg once "
                       "motors, electronics and power are bolted on. Treat the verdict as a structural result "
                       "until the actuator masses are embodied.")
    elif not sim:
        out["note"] = ("no verification certificate in this package, so there is no simulated-body mass to "
                       "compare the as-built parts mass against")
    return out


def _consistency(spec: dict, genome: dict, bom: dict, facts: dict, cert: dict) -> dict:
    """Cross-check this sheet against the other artifacts SHIPPING IN THE SAME PACKAGE. A contradiction here
    is a pipeline bug worth far more than the spec sheet -- one package once shipped two motor selections."""
    checks: list[dict] = []
    bad: list[str] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            bad.append(f"{name}: {detail}")

    ident = cert.get("identity") if isinstance(cert.get("identity"), dict) else {}

    def agree(name: str, values: dict, fmt=str) -> None:
        vals = {k: v for k, v in values.items() if v not in (None, "")}
        if len(vals) < 2:
            return
        add(name, len(set(map(fmt, vals.values()))) == 1,
            ("agree: " if len(set(map(fmt, vals.values()))) == 1 else "DISAGREE: ")
            + ", ".join(f"{k}={fmt(v)}" for k, v in vals.items()))

    agree("robot_class", {"genome": genome.get("robot_class"), "bom": bom.get("robot_class"),
                          "certificate": ident.get("robot_class")})
    agree("dof", {"bom": bom.get("dof"), "genome": _genome_dof(genome), "certificate": ident.get("dof")})

    cert_bom = (cert.get("margins") or {}).get("bom") if isinstance(cert.get("margins"), dict) else None
    if facts["actuator_selection"] and cert_bom:
        from_cert = sorted((str(b.get("part")), int(b.get("qty") or 0)) for b in cert_bom
                           if isinstance(b, dict))
        add("motor_selection", facts["actuator_selection"] == from_cert,
            f"bill of materials {facts['actuator_selection']} vs certificate margins.bom {from_cert}"
            if facts["actuator_selection"] != from_cert
            else f"one selection, {len(from_cert)} SKU(s), agreed by the BOM and the certificate")
        cert_peak = max((float(b.get("peak_nm") or 0.0) for b in cert_bom if isinstance(b, dict)), default=None)
        sheet_peak = (spec.get("actuation") or {}).get("peak_joint_torque_nm")
        if cert_peak and sheet_peak is not None:
            add("peak_joint_torque_nm", abs(cert_peak - float(sheet_peak)) <= 0.15,
                f"spec sheet {sheet_peak} Nm vs certificate margins.bom {cert_peak} Nm")

    struct = facts["structure_mass_kg"]
    sim = ((cert.get("model_sanity") or {}) if isinstance(cert.get("model_sanity"), dict) else {}).get(
        "total_mass_kg")
    if struct and sim:
        # The BOM's structural lines ARE the simulated body. If these two disagree the package is describing
        # two different robots; the actuator/electronics delta on top of them is expected and is NOT a failure.
        add("structural_mass", abs(struct - float(sim)) <= max(0.02 * float(sim), 0.05),
            f"BOM structural lines {struct} kg vs the certified simulated body {sim} kg")

    return {"ok": not bad, "checked": len(checks), "checks": checks, "contradictions": bad}


def _summary(spec: dict) -> str:
    cls = spec.get("robot_class") or "robot"
    dof = spec.get("dof")
    mass = (spec.get("physical") or {}).get("mass_kg")
    cost = (spec.get("power_and_cost") or {}).get("est_parts_cost_usd")
    bits = [f"{dof}-DOF {cls}" if dof else str(cls)]
    if mass:
        bits.append(f"{mass:.1f} kg")
    if cost:
        bits.append(f"~${cost:,.0f} parts")
    perf = spec.get("performance") or {}
    sr = perf.get("success_rate")
    if perf.get("task") == "locomotion":
        # 'walks' is gated upstream on forward + upright + cadence; the raw cadence_hz (total foot-lifts/sec) is
        # left to the detailed table rather than this one-liner, where it reads misleadingly high. For a legged
        # body locomotion IS the task, so success_rate is the fraction of the forward-distance target reached ON
        # A CREDIBLE WALK (0 if it did NOT walk). Never print "walks (0% task success)" -- the audit's self-
        # contradiction: a 0 means it did not achieve a credible walk, so don't claim it walks.
        if sr:
            bits.append(f"walks ({sr:.0%} of the distance target)")
        elif sr is not None:
            bits.append("does not reach a credible walk")
        elif perf.get("credible") is True:
            # No scored success rate, but the certificate signed a rollout -- quote THAT rather than the bare
            # word "walks", which this sheet used to print on no evidence at all.
            fwd = perf.get("forward_m")
            bits.append(f"certified {str(perf.get('verdict') or 'walk').lower()}"
                        + (f" ({fwd} m in sim)" if fwd is not None else ""))
        elif perf.get("credible") is False:
            bits.append(f"NOT certified: {perf.get('verdict') or 'no credible walk'}")
        elif perf.get("cadence_hz"):
            bits.append("walks")
        else:
            bits.append("gait not measured in this package")
    elif sr is not None:
        bits.append(f"{sr:.0%} task success at {perf.get('task')}")
    out = ", ".join(bits) + "."
    contradictions = (spec.get("consistency") or {}).get("contradictions") or []
    if contradictions:
        out += (f" INCONSISTENT PACKAGE -- {len(contradictions)} artifact(s) in this package contradict this "
                f"sheet: {'; '.join(contradictions)}")
    return out


def _cell(spec: dict, path: str, value, suffix: str = "") -> str:
    """A markdown cell that is either the number or the actionable reason it is missing -- never ``None``."""
    if value not in (None, [], {}):
        return f"{value}{suffix}"
    why = (spec.get("unavailable") or {}).get(path)
    return why or "n/a"


def _markdown(spec: dict) -> str:
    p = spec.get("physical") or {}
    pc = spec.get("power_and_cost") or {}
    act = spec.get("actuation") or {}
    perf = spec.get("performance") or {}
    size = p.get("size_m") or {}
    sz = (f"{size.get('length_m')} x {size.get('width_m')} x {size.get('height_m')} m"
          if size else _cell(spec, "physical.size_m", None))
    types = ", ".join(act.get("actuator_types") or []) or _cell(spec, "actuation.actuator_types", None)
    cost = pc.get("est_parts_cost_usd")
    cost_cell = _cell(spec, "power_and_cost.est_parts_cost_usd",
                      f"${float(cost):,.2f}" if isinstance(cost, (int, float)) else None)
    torque_cell = _cell(spec, "actuation.peak_joint_torque_nm", act.get("peak_joint_torque_nm"), " Nm")
    power_cell = _cell(spec, "power_and_cost.est_power_draw_w", pc.get("est_power_draw_w"), " W")
    mass_cell = _cell(spec, "physical.mass_kg", p.get("mass_kg"), " kg")
    sensors_cell = ", ".join(spec.get("sensing") or []) or _cell(spec, "sensing", None)
    compute_cell = ", ".join(spec.get("compute") or []) or _cell(spec, "compute", None)
    success_cell = _cell(spec, "performance.success_rate", perf.get("success_rate"))
    lines = [
        f"# {spec.get('name') or 'Robot'} - Spec Sheet",
        "",
        f"_{spec.get('summary', '')}_",
        "",
        "| Spec | Value |",
        "| --- | --- |",
        f"| Class | {spec.get('robot_class')} |",
        f"| Degrees of freedom | {spec.get('dof')} |",
        f"| Mass (as built) | {mass_cell} |",
        f"| Size (LxWxH) | {sz} |",
        f"| Actuators | {_cell(spec, 'physical.actuators', p.get('actuators'))} ({types}) |",
        f"| Peak joint torque | {torque_cell} |",
        f"| Est. power draw | {power_cell} |",
        f"| Est. parts cost | {cost_cell} |",
        f"| Sensors | {sensors_cell} |",
        f"| Compute | {compute_cell} |",
        f"| Task | {perf.get('task')} |",
        f"| Task success | {success_cell} |",
    ]
    if perf.get("verdict"):
        lines.append(f"| Physics verdict | {perf.get('verdict')} (gait: {perf.get('gait_source')}, "
                     f"source: {perf.get('measured_by')}) |")
    if perf.get("task") == "locomotion":
        # Name the stability metric that was ACTUALLY measured. The evaluation report reports an upright
        # fraction; the certificate reports a support fraction. They are different quantities, so printing
        # either one under the label "upright" would misreport whichever it wasn't.
        if perf.get("gait_stability_frac") is not None:
            stability = f"upright {perf['gait_stability_frac']}"
        elif perf.get("support_frac") is not None:
            stability = f"support {perf['support_frac']}"
        else:
            stability = "stability not reported"
        lines.append(f"| Gait | forward {perf.get('forward_m')} m, cadence {perf.get('cadence_hz')} Hz, "
                     f"{stability} |")
    mb = p.get("mass_breakdown") or {}
    if mb.get("note"):
        lines += ["", f"**Mass:** as built {mb.get('as_built_parts_kg')} kg "
                      f"(structure {mb.get('structure_only_kg')} kg + motors/electronics/power). "
                      f"{mb['note']}"]
    cons = spec.get("consistency") or {}
    if cons.get("contradictions"):
        lines += ["", "## INCONSISTENT PACKAGE", "",
                  "These artifacts ship in this same package and contradict this sheet. Do not order parts "
                  "against it until they are reconciled:", ""]
        lines += [f"- {c}" for c in cons["contradictions"]]
    elif cons.get("checked"):
        lines += ["", f"_Cross-checked against the bill of materials and the verification certificate in this "
                      f"package: {cons['checked']}/{cons['checked']} agree "
                      f"({', '.join(c['check'] for c in cons.get('checks') or [])})._"]
    if spec.get("unavailable"):
        lines += ["", "## Not in this package", ""]
        lines += [f"- **{k}** -- {v}" for k, v in sorted(spec["unavailable"].items())]
    lines.append("")
    lines.append("_Estimates derived from the generated design + real-component bill of materials + the physics "
                 "evaluation. Not a manufacturing guarantee._")
    return "\n".join(lines)


def write_spec_sheet(output_dir) -> Path | None:
    """Write reports/spec_sheet.json + spec_sheet.md. Returns the json path, or None if there's nothing to spec."""
    output_dir = Path(output_dir)
    spec = build_spec_sheet(output_dir)
    if not spec.get("robot_class") and not spec.get("dof"):
        return None
    out = output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec_sheet.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out / "spec_sheet.md").write_text(_markdown(spec), encoding="utf-8")
    return out / "spec_sheet.json"
