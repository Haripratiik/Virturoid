"""Robot spec sheet: an auto-derived capability + cost summary for a built package.

A user who builds a robot wants to know, at a glance, what it actually IS and can DO -- mass, footprint, power
draw, parts cost, actuator torque, sensors, and task performance. This aggregates the data the build already
produces (the Robot Genome + the Bill of Materials + the physics evaluation report) into one ``spec_sheet.json``
+ human-readable ``spec_sheet.md``. Pure read-over-artifacts; no rebuild, no training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SCENE_WORDS = ("floor", "ground", "plane", "block", "cube", "goal", "obstacle", "wall", "table", "target", "bin")


def _first(output_dir: Path, name: str) -> dict | None:
    hits = sorted(Path(output_dir).rglob(name))
    return json.loads(hits[0].read_text(encoding="utf-8")) if hits else None


def _robot_extent(output_dir: Path) -> dict | None:
    """Best-effort robot bounding box (length/width/height in m) from the compiled MJCF, excluding scene props
    (floor/blocks/goal/...) so the size reflects the ROBOT, not the task scene."""
    try:
        import mujoco
        import numpy as np

        xmls = sorted((Path(output_dir) / "simulation").rglob("*.xml"))
        if not xmls:
            return None
        model = mujoco.MjModel.from_xml_path(str(xmls[0]))
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
    except Exception:  # noqa: BLE001 - size is a nicety; never break the spec sheet
        return None


def build_spec_sheet(output_dir) -> dict:
    """Aggregate genome + BOM + evaluation into a capability/cost spec dict."""
    output_dir = Path(output_dir)
    genome = _first(output_dir, "robot_genome.json") or {}
    bom = _first(output_dir, "bill_of_materials.json") or {}
    ev = _first(output_dir, "gene_evaluation_report.json") or {}
    totals = bom.get("totals", {})
    lines = bom.get("lines", [])

    torques = []
    for line in lines:
        if line.get("category") == "actuator":
            mt = re.search(r"peak\s+([\d.]+)\s*Nm", line.get("detail", ""))
            if mt:
                torques.append(float(mt.group(1)))
    actuator_types = sorted({line["part"] for line in lines if line.get("category") == "actuator"})
    sensors = sorted({line["part"] for line in lines
                      if line.get("category") in ("camera", "imu", "lidar", "sensor", "force_torque", "depth")})
    compute = sorted({line["part"] for line in lines if line.get("category") == "compute"})

    task = ev.get("task_type") or genome.get("robot_class")
    perf: dict = {"task": task, "success_rate": ev.get("success_rate")}
    if ev.get("task_type") == "locomotion":
        perf.update({"forward_m": round(float(ev.get("forward_m") or ev.get("distance_m") or 0.0), 2),
                     "cadence_hz": ev.get("cadence_hz"), "gait_stability_frac": ev.get("upright_frac")})
    if isinstance(ev.get("perception"), dict):
        perf["perception_nav_success"] = ev["perception"].get("success_rate")  # sensed-only reach rate

    spec = {
        "name": genome.get("name") or genome.get("species"),
        "species": genome.get("species"),
        "robot_class": genome.get("robot_class") or bom.get("robot_class"),
        "dof": bom.get("dof") or len([j for j in genome.get("joints", [])]),
        "physical": {
            "mass_kg": totals.get("mass_kg"),
            "actuators": totals.get("actuators"),
            "size_m": _robot_extent(output_dir),
        },
        "power_and_cost": {
            "est_power_draw_w": totals.get("est_power_w"),
            "est_parts_cost_usd": totals.get("price_usd"),
        },
        "actuation": {
            "actuator_types": actuator_types,
            "peak_joint_torque_nm": round(max(torques), 1) if torques else None,
        },
        "sensing": sensors,
        "compute": compute,
        "performance": perf,
    }
    spec["summary"] = _summary(spec)
    return spec


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
    if perf.get("task") == "locomotion" and perf.get("cadence_hz"):
        # 'walks' is gated upstream on forward + upright + cadence; the raw cadence_hz (total foot-lifts/sec) is
        # left to the detailed table rather than this one-liner, where it reads misleadingly high.
        bits.append(f"walks ({sr:.0%} task success)" if sr is not None else "walks")
    elif sr is not None:
        bits.append(f"{sr:.0%} task success at {perf.get('task')}")
    return ", ".join(bits) + "."


def _markdown(spec: dict) -> str:
    p = spec.get("physical") or {}
    pc = spec.get("power_and_cost") or {}
    act = spec.get("actuation") or {}
    perf = spec.get("performance") or {}
    size = p.get("size_m") or {}
    sz = (f"{size.get('length_m')} x {size.get('width_m')} x {size.get('height_m')} m"
          if size else "n/a")
    lines = [
        f"# {spec.get('name') or 'Robot'} - Spec Sheet",
        "",
        f"_{spec.get('summary', '')}_",
        "",
        "| Spec | Value |",
        "| --- | --- |",
        f"| Class | {spec.get('robot_class')} |",
        f"| Degrees of freedom | {spec.get('dof')} |",
        f"| Mass | {p.get('mass_kg')} kg |",
        f"| Size (LxWxH) | {sz} |",
        f"| Actuators | {p.get('actuators')} ({', '.join(act.get('actuator_types') or []) or 'n/a'}) |",
        f"| Peak joint torque | {act.get('peak_joint_torque_nm')} Nm |",
        f"| Est. power draw | {pc.get('est_power_draw_w')} W |",
        f"| Est. parts cost | ${pc.get('est_parts_cost_usd')} |",
        f"| Sensors | {', '.join(spec.get('sensing') or []) or 'n/a'} |",
        f"| Compute | {', '.join(spec.get('compute') or []) or 'n/a'} |",
        f"| Task | {perf.get('task')} |",
        f"| Task success | {perf.get('success_rate')} |",
    ]
    if perf.get("task") == "locomotion":
        lines.append(f"| Gait | forward {perf.get('forward_m')} m, cadence {perf.get('cadence_hz')} Hz, "
                     f"upright {perf.get('gait_stability_frac')} |")
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
