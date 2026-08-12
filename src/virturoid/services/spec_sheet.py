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

MASS IS THREE NUMBERS, ON PURPOSE — AND TWO OF THEM SHOULD NOW AGREE. ``mass_kg`` is the AS-BUILT parts mass
from the BOM; ``structure_mass_kg`` is the raw stock a fabricator buys (its material lines); ``simulated_body_kg``
is what the physics verdict was measured on. On a body we composed, ``grounded_physics.embody_component_masses``
bolts the parts list's own battery, compute and sensors onto the links that carry them, so AS-BUILT and SIMULATED
are the same robot and ``as_built_over_simulated`` reads 1.0. Print all three anyway, and say when they diverge:
an IMPORTED robot is deliberately left at the customer's own masses, so its parts list is their machine PLUS
whatever this BOM proposes adding, and the ratio is a real difference rather than a bookkeeping one.

A NUMBER NAMED AFTER THE ROBOT MUST BE READ FROM THE ROBOT. ``actuation.peak_joint_torque_nm`` reported
``360.0`` for a customer's real Menagerie Unitree Go2 whose own ``robot.xml``, shipping in the same package,
declares ``forcerange="-45.43 45.43"`` — 8x, because the only torque this sheet could see was the one it
regexed out of a BOM line's detail string, and that is the rating of the CATALOG PART the BOM matched to those
joints. An actuator is selected as the smallest part that COVERS its joint, so a part rating sits at or above
the joint's limit by construction: reading one as the other can only ever overstate. The robot's number now
comes from the BOM's ``joint_limits`` block or the genome's own joint efforts (see ``_robot_peak_torque``);
the part's rating still ships, on its own row, named ``catalog_part_peak_torque_nm``. Same treatment as mass:
three numbers, each labelled, none standing in for another. And ``consistency`` now cross-checks the sheet's
peak torque against the genome's declared efforts — the check that would have caught this, and did not exist.

MONEY GETS THE SAME SPLIT. ``est_parts_cost_usd`` is ``totals.price_usd``, which on an imported machine used
to include $6,480 of catalog EQUIVALENTS for motors the customer already owns and asked us to keep. It is now
the BILL, with ``already_fitted_usd`` beside it.

This paragraph used to say the sim body "carries the STRUCTURE only ... does not embody the actuators or the
battery", and ``structure_mass_kg`` was read off material lines that had every link's motor inside them. Both
halves were wrong in the same direction: grounding has embodied actuator mass for some time, and summing
finished link masses under "material" billed those motors a second time (measured: an authored horse simulated
at 15.951 kg with a 29.212 kg parts list, 10.410 kg of it its own motors, counted twice).
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
#: written by ``export_gate.write_refusal_package`` when the exported twin cannot be stepped. A spec sheet is a
#: physics summary; a package that has declared itself unsimulable has no physics to summarise, and this sheet
#: is a pure aggregator, so it would happily print the mass and the peak torque anyway (measured on flybody:
#: "66-DOF hexapod, 5.1 kg, ~$5,598 parts"). Enforced here as well as at the export door so that ANY package
#: carrying the refusal is safe, whichever writer assembled it.
_REFUSAL_NAMES = ("export_refusal.json",)

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
    # THE CATALOG PART's rating — a fact about the motor we would buy, NOT about the robot. Kept because a
    # buyer wants it; named ``part_torques`` because it was previously called ``torques`` and printed under
    # the label "Peak joint torque". See _robot_peak_torque.
    part_torques = []
    for line in lines:
        if line.get("category") in _ACTUATOR_CATEGORIES:
            mt = re.search(r"peak\s+([\d.]+)\s*Nm", str(line.get("detail", "")))
            if mt:
                part_torques.append((float(mt.group(1)), str(line.get("part") or "")))
    return {
        "lines": lines,
        "part_torques": part_torques,
        # propulsion = whatever moves THIS body: joint actuators (arm/legged), drive motors (mobile), rotors
        # (aerial) -- so a drone's spec shows its rotors and a rover's its drive motors, not an empty section.
        "actuator_types": sorted({str(ln["part"]) for ln in lines
                                  if ln.get("category") in _ACTUATOR_CATEGORIES and ln.get("part")}),
        "sensors": sorted({str(ln["part"]) for ln in lines
                           if ln.get("category") in _SENSOR_CATEGORIES and ln.get("part")}),
        # ...and which of them the BOM itself marks as NOT fitted. Same defect one surface along: a Go2 that
        # compiles to ncam 0 / nsensor 0 had `sensing: [Bosch BNO055, Intel RealSense D435i]` printed flat
        # under "Sensors", while bom.json two directories away said `proposed: true` on both lines.
        "sensors_proposed": sorted({str(ln["part"]) for ln in lines
                                    if ln.get("category") in _SENSOR_CATEGORIES and ln.get("part")
                                    and ln.get("proposed")}),
        "compute": sorted({str(ln["part"]) for ln in lines
                           if ln.get("category") == "compute" and ln.get("part")}),
        # The STRUCTURE-only mass — raw stock, no motors, no electronics. It is NOT the number the certificate
        # has to match: the simulated body carries its parts too, so the full parts mass is. See the module
        # docstring on mass.
        "structure_mass_kg": (round(sum(float(ln.get("mass_kg") or 0.0)
                                        for ln in lines if ln.get("category") == "material"), 3)
                              if any(ln.get("category") == "material" for ln in lines) else None),
        "actuator_selection": sorted((str(ln.get("part")), int(ln.get("qty") or 0))
                                     for ln in lines if ln.get("category") == "actuator"),
    }


def _genome_joint_efforts(genome: dict) -> list[tuple[float, str]]:
    """``[(effort, joint name)]`` the GENOME declares — the robot's own limits, in the same package.

    On an imported machine these are the manufacturer's, read out of the customer's file at import; on a body
    we composed they are the fitted actuator's peak, written by ``gene_build``. Either way they describe THIS
    ROBOT, which is what a field called "peak joint torque" claims to.

    REVOLUTE ONLY. A prismatic joint's ``effort`` is a LINEAR FORCE in newtons, and a max taken across both
    would put newtons into a field named ``_nm`` -- the same category error as the one this file is fixing,
    one unit along. ``bom_builder._aggregate_joint_limits`` splits them for the same reason.
    """
    out = []
    for j in (genome.get("joints") or []):
        if not isinstance(j, dict):
            continue
        if str(j.get("joint_type", "revolute")).lower() not in ("revolute", ""):
            continue
        eff = (j.get("limit") or {}).get("effort") if isinstance(j.get("limit"), dict) else None
        try:
            val = abs(float(eff))
        except (TypeError, ValueError):
            continue
        if val > 0:
            out.append((val, str(j.get("name") or "")))
    return out


def _robot_peak_torque(bom: dict, genome: dict, facts: dict) -> tuple[float | None, str, str | None]:
    """``(peak N.m, how we know, reason it is unavailable)`` — THE ROBOT'S capability, in priority order.

    THE DEFECT THIS FUNCTION EXISTS FOR (2026-08-12, real Menagerie Unitree Go2 through ``call_tool`` +
    ``export_held``): this sheet reported ``actuation.peak_joint_torque_nm: 360.0`` for a robot whose own
    ``robot.xml`` — shipping three files away in the same package — declares ``forcerange="-45.43 45.43"``.
    The 360 was the rating of the catalog part the BOM matched to those joints ("Unitree M107 (B2/H1-class),
    peak 360 Nm"), regexed out of the part's detail string. A part is selected as the SMALLEST one that COVERS
    its joint, so its rating is >= the joint's by construction: reading it as the robot's capability can only
    overstate, never understate, and here it overstated by 8x on a customer's own machine.

    So: the BOM's ``joint_limits`` block first (bom_builder states it explicitly, with provenance), then the
    genome's declared joint efforts, both of which describe the ROBOT. The catalog rating is used ONLY as a
    last resort and ONLY on a robot we designed, where the joint limit really is the peak of the motor we
    chose — and it is labelled even then. On an imported machine with neither block, we refuse: an 8x overstate
    on hardware somebody is going to command torque to is worse than an empty cell that says why.
    """
    jl = bom.get("joint_limits") if isinstance(bom.get("joint_limits"), dict) else {}
    if jl.get("peak_joint_torque_nm") is not None:
        src = ("read from your own model's declared limits"
               if jl.get("declared_by_your_model") else "the peak of the actuator this design fitted")
        return (round(float(jl["peak_joint_torque_nm"]), 2),
                f"{src} (worst joint: {jl.get('peak_joint')})", None)
    efforts = _genome_joint_efforts(genome)
    if efforts:
        val, name = max(efforts)
        return round(val, 2), f"the robot genome's declared joint effort limit (worst joint: {name})", None
    if facts["part_torques"] and not _is_imported(bom):
        val, part = max(facts["part_torques"])
        return (round(val, 2),
                f"the peak rating of the actuator this design fitted ({part}) — on a robot we designed the "
                f"joint limit IS the selected motor's peak, because we chose the motor", None)
    if facts["part_torques"]:
        val, part = max(facts["part_torques"])
        return None, "", (
            f"not derived: this package carries no joint-limit block and no genome joint efforts, and this is "
            f"YOUR machine rather than a design of ours — so the only torque available is {part}'s catalog "
            f"rating ({val:g} Nm), which is the smallest part that COVERS your joints and therefore overstates "
            f"them. Re-export from a re-imported robot so the BOM carries joint_limits, or read the "
            f"forcerange in the shipped model")
    return None, "", None


def _is_imported(bom: dict) -> bool:
    """Is the BOM describing the CUSTOMER'S machine? ``bom_builder`` says so in ``spec_provenance.robot_is``
    (and, on older packages, by excluding the actuator lines from the mass because they are already in it)."""
    prov = bom.get("spec_provenance") if isinstance(bom.get("spec_provenance"), dict) else {}
    if prov.get("robot_is"):
        return "imported" in str(prov["robot_is"])
    return any(isinstance(ln, dict) and ln.get("category") == "actuator" and ln.get("in_mass_total") is False
               for ln in (bom.get("lines") or []))


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
    refusal = _first(output_dir, *_REFUSAL_NAMES) or {}
    if refusal.get("simulable") is False:
        return {"name": None, "robot_class": None, "dof": None, "refused": True,
                "summary": ("NO SPEC SHEET: this package declares its twin NOT SIMULABLE "
                            f"({refusal.get('reason')}). Every headline a spec sheet prints — mass, peak "
                            "torque, power draw, verdict — is a claim about physics, and no physics was run "
                            "on this body. See NOT_SIMULABLE.md."),
                "not_simulable": {"reason": refusal.get("reason"),
                                  "simulation_check": refusal.get("simulation_check")}}
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
    robot_peak, peak_source, peak_why = _robot_peak_torque(bom, genome, facts)
    if robot_peak is None:
        unavailable["actuation.peak_joint_torque_nm"] = peak_why or (no_bom if not bom else (
            "not derived: this package states no joint torque limit -- neither the BOM's joint_limits block nor "
            "the genome's joint effort limits are present, and the actuator lines carry no 'peak <N> Nm' rating. "
            "Re-export including the 'bom' format so the robot's own limits travel with it"))
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
            # THE BILL, not the catalog valuation of a machine the customer already owns. ``bom_builder`` keeps
            # ``price_usd`` to the lines it actually asks them to buy and quotes the rest beside it; both
            # travel here so the sheet can never present one as the other.
            "est_parts_cost_usd": totals.get("price_usd"),
            **({"already_fitted_usd": totals.get("already_fitted_usd"),
                "catalog_list_price_usd": totals.get("catalog_list_price_usd"),
                "cost_note": totals.get("already_fitted_note")}
               if totals.get("already_fitted_usd") else {}),
        },
        "actuation": {
            "actuator_types": facts["actuator_types"],
            # On an imported machine ``actuator_types`` is a list of parts WE would buy to match the motors the
            # customer already has, not a list of the motors it has. Say which.
            **({"equivalents_for_your_motors": True} if (facts["actuator_types"] and _is_imported(bom)) else {}),
            # THE ROBOT's limit. Never a part rating -- see _robot_peak_torque.
            "peak_joint_torque_nm": robot_peak,
            **({"peak_joint_torque_source": peak_source} if robot_peak is not None else {}),
            # ...and the part rating, on its own line, named for what it is. It is genuinely useful (it is the
            # headroom the certificate's margin gates grade against); it is simply not the robot's number.
            **({"catalog_part_peak_torque_nm": round(max(facts["part_torques"])[0], 1),
                "catalog_part": max(facts["part_torques"])[1],
                "catalog_part_note": (
                    "the RATING OF THE PART we would buy, not this robot's capability: an actuator is selected "
                    "as the smallest one that covers its joint, so this sits at or above the joint limit above")}
               if facts["part_torques"] else {}),
        },
        "sensing": facts["sensors"],
        # the RECOMMENDATION stays in ``sensing`` (that is the useful part); which of it is not on the machine
        # travels beside it, so no consumer can print the list as an inventory again.
        **({"sensing_proposed": facts["sensors_proposed"]} if facts["sensors_proposed"] else {}),
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
    """As-built parts mass vs the mass the physics verdict was actually measured at. See the module docstring.

    A MISSING certificate and a certificate that produced no simulated mass are different facts, and this used
    to print the first sentence for both. Measured on an unsimulable flybody twin: the sheet said "no
    verification certificate in this package" in ``mass_breakdown.note`` while ``sources.verification_certificate``
    read ``true`` eight lines below it, in the same file — the certificate was there, VOID, and its
    ``model_sanity`` had no mass because the model never compiled. Name the real reason instead.
    """
    sanity = cert.get("model_sanity") if isinstance(cert.get("model_sanity"), dict) else {}
    sim = sanity.get("total_mass_kg")
    out: dict = {"as_built_parts_kg": totals.get("mass_kg"),
                 "structure_only_kg": facts["structure_mass_kg"],
                 "simulated_body_kg": sim}
    if totals.get("mass_kg") and sim:
        ratio = round(float(totals["mass_kg"]) / float(sim), 2)
        out["as_built_over_simulated"] = ratio
        if abs(float(totals["mass_kg"]) - float(sim)) <= max(0.05, 0.01 * float(sim)):
            out["note"] = (f"The physics verdict in this package was measured on a body weighing {sim} kg, and "
                           f"the parts list adds up to {totals['mass_kg']} kg — THE SAME ROBOT. The motors, "
                           f"electronics and power below are on the simulated links that carry them, so the "
                           f"verdict is not a structural result standing in for a heavier machine.")
        else:
            out["note"] = (f"The physics verdict in this package was measured on a {sim} kg body, while its "
                           f"parts list adds up to {totals['mass_kg']} kg ({ratio}x). On an IMPORTED robot that "
                           f"is expected — the simulated body is the customer's own machine and the parts list "
                           f"is that machine PLUS what this BOM proposes adding. Anywhere else it means the "
                           f"body was not re-grounded after its parts changed.")
    elif not sim and not cert:
        out["note"] = ("no verification certificate in this package, so there is no simulated-body mass to "
                       "compare the as-built parts mass against")
    elif not sim:
        why = (cert.get("voided_reason") or "; ".join(str(i) for i in (sanity.get("issues") or []))
               or "its model_sanity block carries no total_mass_kg")
        out["certificate_valid"] = cert.get("valid")
        out["note"] = (f"this package DOES carry a verification certificate, but it reports no simulated-body "
                       f"mass ({str(why)[:220]}), so the as-built parts mass above cannot be compared against "
                       f"one — and the certificate itself is "
                       + ("VOID" if cert.get("valid") is False else "incomplete on this point") + ".")
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
        # LIKE FOR LIKE. ``margins.bom`` grades the ORDERED PART, so it is compared against the part rating --
        # NOT against the robot's joint limit, which is a different quantity and is checked below. Comparing
        # them here is what made the 360 N.m claim look CONSISTENT: two artifacts agreeing about a part number
        # while the robot's own model, in the same package, said 45.43.
        cert_peak = max((float(b.get("peak_nm") or 0.0) for b in cert_bom if isinstance(b, dict)), default=None)
        part_peak = (spec.get("actuation") or {}).get("catalog_part_peak_torque_nm")
        if cert_peak and part_peak is not None:
            add("actuator_part_peak_nm", abs(cert_peak - float(part_peak)) <= 0.15,
                f"spec sheet part rating {part_peak} Nm vs certificate margins.bom {cert_peak} Nm")

    # THE CHECK THAT WOULD HAVE CAUGHT IT. The robot's peak joint torque, as this sheet reports it, against the
    # joint effort limits in the genome shipping in the SAME package. On the Go2 that is 360.0 vs 45.43 -- a
    # contradiction between two files a customer opens one after the other, and nothing was comparing them.
    sheet_peak = (spec.get("actuation") or {}).get("peak_joint_torque_nm")
    efforts = _genome_joint_efforts(genome)
    if sheet_peak is not None and efforts:
        worst, jname = max(efforts)
        ok = abs(float(worst) - float(sheet_peak)) <= max(0.01, 0.02 * float(worst))
        add("peak_joint_torque_nm", ok,
            f"spec sheet {sheet_peak} Nm vs the robot genome's own worst joint effort limit {round(worst, 2)} Nm"
            + (f" at {jname}" if jname else "")
            + ("" if ok else " -- these describe the SAME robot; a sheet quoting a bigger number is quoting a "
                            "catalog part's rating instead of the machine's capability"))

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
    pc = spec.get("power_and_cost") or {}
    cost = pc.get("est_parts_cost_usd")
    bits = [f"{dof}-DOF {cls}" if dof else str(cls)]
    if mass:
        bits.append(f"{mass:.1f} kg")
    if cost:
        # "~$7,614 parts" on an imported machine read as the price of the robot in front of them, and $6,480 of
        # it was motors already bolted to it. The headline is the BILL; what they own is named, not folded in.
        already = pc.get("already_fitted_usd")
        bits.append(f"~${cost:,.0f} parts to buy" + (f" (+${already:,.0f} already fitted)" if already else ""))
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
    if pc.get("already_fitted_usd"):
        cost_cell += (f" to buy (+ ${float(pc['already_fitted_usd']):,.2f} of catalog equivalents for hardware "
                      f"you already own, not billed)")
    torque_cell = _cell(spec, "actuation.peak_joint_torque_nm", act.get("peak_joint_torque_nm"), " Nm")
    if act.get("peak_joint_torque_source"):
        torque_cell += f" -- {act['peak_joint_torque_source']}"
    power_cell = _cell(spec, "power_and_cost.est_power_draw_w", pc.get("est_power_draw_w"), " W")
    mass_cell = _cell(spec, "physical.mass_kg", p.get("mass_kg"), " kg")
    # A PART THE BOM MARKS "NOT FITTED TO YOUR ROBOT" MUST NOT BE PRINTED AS ONE OF ITS SENSORS.
    _proposed = set(spec.get("sensing_proposed") or [])
    sensors_cell = ", ".join(f"{s} (PROPOSED -- not on your robot)" if s in _proposed else s
                             for s in (spec.get("sensing") or [])) or _cell(spec, "sensing", None)
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
        # ON THE CUSTOMER'S MACHINE THESE ARE OUR PART NUMBERS FOR THEIR MOTORS, not the motors it has.
        f"| Actuators | {_cell(spec, 'physical.actuators', p.get('actuators'))} "
        f"({'catalog equivalents for your own motors: ' if act.get('equivalents_for_your_motors') else ''}"
        f"{types}) |",
        f"| Peak joint torque (THIS ROBOT) | {torque_cell} |",
        f"| Est. power draw | {power_cell} |",
        f"| Est. parts cost | {cost_cell} |",
        f"| Sensors | {sensors_cell} |",
        f"| Compute | {compute_cell} |",
        f"| Task | {perf.get('task')} |",
        f"| Task success | {success_cell} |",
    ]
    if act.get("catalog_part_peak_torque_nm") is not None:
        # A SEPARATE, CLEARLY-LABELLED LINE. It is the part's number; putting it in the row above is the whole
        # defect (a 45.43 N.m Go2 reported as 360.0 N.m because that is what the motor we would sell can do).
        lines.append(f"| Actuator part rating (NOT the robot) | {act['catalog_part_peak_torque_nm']} Nm "
                     f"({act.get('catalog_part')}) -- {act.get('catalog_part_note')} |")
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
    if pc.get("cost_note"):
        lines += ["", f"**Cost:** {pc['cost_note']}"]
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
    if spec.get("refused") or (not spec.get("robot_class") and not spec.get("dof")):
        return None
    out = output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec_sheet.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out / "spec_sheet.md").write_text(_markdown(spec), encoding="utf-8")
    return out / "spec_sheet.json"
