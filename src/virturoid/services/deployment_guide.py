"""Deployment guide: a 'build this robot for real' walkthrough generated from the package artifacts.

Virturoid designs a robot in simulation AND emits everything needed to make it physical -- a real-component bill
of materials, a URDF, an installable ROS2 package, a control program. This stitches those into ONE actionable
``reports/deployment_guide.md``: order these parts, mount this actuator at this joint, build + launch the ROS2
package, and here are the honest sim-to-real caveats. Pure aggregation over artifacts; no rebuild.
"""
from __future__ import annotations

import json
from pathlib import Path


def _load(output_dir: Path, name: str) -> dict | None:
    hits = sorted(Path(output_dir).rglob(name))
    if not hits:
        return None
    try:
        return json.loads(hits[0].read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def build_deployment_guide(output_dir) -> str:
    output_dir = Path(output_dir)
    spec = _load(output_dir, "spec_sheet.json") or {}
    bom = _load(output_dir, "bill_of_materials.json") or {}
    ledger = _load(output_dir, "product_readiness_ledger.json") or {}
    name = spec.get("name") or (bom.get("robot_class") or "robot")
    totals = bom.get("totals", {})
    lines = bom.get("lines", [])
    actuator_map = bom.get("actuator_map", {})

    has_ros2 = bool(list((output_dir / "export").rglob("package.xml"))) if (output_dir / "export").exists() else False
    has_control = (output_dir / "software" / "control_program.json").exists()
    has_urdf = (output_dir / "robot" / "robot.urdf").exists()
    fusion = _load(output_dir, "fusion_manifest.json")
    scripts = _load(output_dir, "script_manifest.json")
    script_val = _load(output_dir, "script_validation.json")

    out: list[str] = []
    out.append(f"# Build *{name}* for real - deployment guide")
    out.append("")
    if spec.get("summary"):
        out.append(f"_{spec['summary']}_")
        out.append("")
    out.append("Virturoid designed this robot in simulation and produced everything below so you can build the "
               "physical version. The steps go sim -> parts -> assembly -> software.")
    out.append("")

    # 1. Parts
    cost = totals.get("price_usd")
    out.append(f"## 1. Order the parts{f' (~${cost:,.0f})' if cost else ''}")
    out.append("")
    if lines:
        out.append("| Part | Category | Qty | Unit price | Notes |")
        out.append("| --- | --- | --- | --- | --- |")
        for line in lines:
            up = line.get("unit_price_usd")
            out.append(f"| {line.get('part','?')} | {line.get('category','')} | {line.get('qty','')} | "
                       f"{('$' + format(up, ',.0f')) if up else '-'} | {line.get('detail','')} |")
        out.append("")
        out.append(f"Totals: **{totals.get('mass_kg','?')} kg**, **~${cost:,.0f}** parts, "
                   f"**~{totals.get('est_power_w','?')} W** peak draw." if cost else "")
    else:
        out.append("_No bill of materials was generated for this build (legacy path). Build it through the gene "
                   "path to get a real per-joint BOM._")
    out.append("")

    # 2. Assembly
    out.append("## 2. Assemble the chassis")
    out.append("")
    if actuator_map:
        out.append("Mount one actuator per joint as specified by the design:")
        out.append("")
        out.append("| Joint | Actuator |")
        out.append("| --- | --- |")
        for joint, act in actuator_map.items():
            out.append(f"| {joint} | {act} |")
        out.append("")
    out.append(f"Use `robot/robot.urdf` (the kinematic description) as the assembly reference."
               if has_urdf else "A URDF was not emitted for this build; use the MJCF in `simulation/` as the "
               "kinematic reference.")
    out.append("")

    # 3. Software
    out.append("## 3. Flash the software")
    out.append("")
    if has_ros2:
        out.append("This build ships an installable ROS2 (ament_python) package:")
        out.append("")
        out.append("```")
        out.append("colcon build --packages-select virturoid_robot")
        out.append("ros2 launch virturoid_robot evaluate.launch.py")
        out.append("```")
        out.append("")
    if has_control:
        out.append("The learned/derived controller is exported at `software/control_program.json` "
                   "(a downstream PD / ros2_control loop tracks its joint targets).")
        out.append("")
    if fusion:
        kind = fusion.get("kind", "robot")
        sens = fusion.get("sensors", [])
        # "CARRIES n SENSOR(S)" IS A CLAIM OF POSSESSION, and on an imported robot the suite is a QUOTE: a
        # Menagerie Go2 (ncam 0, nsensor 0) was told it "carries 2 sensor(s)" it has none of. The proposed count
        # comes straight off the compiler's per-entry flag, so this line cannot drift from the config it describes.
        n_prop = sum(1 for s in sens if s.get("proposed"))
        carries = (f"This {kind} carries {len(sens)} sensor(s)" if not n_prop else
                   (f"{n_prop} of these {len(sens)} sensor(s) are PROPOSED ADDITIONS — hardware your own model "
                    f"does not declare, so their topics have no publisher until you fit them"))
        out.append(f"**Sensor fusion (state estimation)** — compiled from the BOM, not hand-written. {carries}; "
                   f"`fusion/` ships the deployable stack:")
        out.append("")
        for f in fusion.get("files", []):
            out.append(f"- `fusion/{f}`")
        fused = fusion.get("fused_states") or {}
        if fused:
            out.append("")
            out.append("Fused state estimates: " + ", ".join(
                f"**{st}** ({', '.join(srcs)})" for st, srcs in fused.items()) + ".")
        for miss in (fusion.get("missing") or [])[:3]:
            out.append(f"- ⚠️ {miss}")
        out.append("")
        out.append("```")
        out.append("ros2 launch virturoid_robot sensor_fusion.launch.py")
        out.append("```")
        out.append("")
    if scripts:
        n = len(scripts.get("scripts", []))
        passed = script_val.get("all_pass") if script_val else None
        audit = (script_val or {}).get("torque_audit") or {}
        astat = audit.get("status")
        badge = "" if passed is None else (" — all **compiled + sim-dry-run** ✅" if passed
                                           else " — ⚠️ some scripts failed validation")
        out.append(f"**Operational control scripts** — {n} generated from the robot, not hand-written{badge}. "
                   f"Under `software/scripts/`:")
        out.append("")
        out.append("| Script | Role |")
        out.append("| --- | --- |")
        _ROLE = {"obs_assembler.py": "builds the exact observation vector the policy trained on",
                 "safety_filter.py": "clamps every command to each actuator's peak torque + joint limits",
                 "state_machine.py": "estop / stand / active / fall-damping supervisory logic",
                 "watchdog.py": "trips estop on a stalled loop, comms timeout, or joint-limit breach",
                 "teleop.py": "keyboard/joystick velocity teleop stub",
                 "calibrate.py": "captures each joint's encoder zero offset"}
        for s in scripts.get("scripts", []):
            out.append(f"| `{s}` | {_ROLE.get(s, 'operational control script')} |")
        out.append("")
        # The datasheet-torque claim is stated ONLY when the audit that proves it actually ran and passed.
        # (Before: this paragraph asserted it unconditionally — an inflated control_config.json shipped with the
        # sentence still under it, and nothing had checked.)
        if astat == "not_applicable" or (astat == "pass" and not audit.get("n_joints")):
            # A pass over ZERO joints is every check trivially true over an empty set. Printing the full
            # "audit passed / the filter was EXECUTED at 10x each joint's peak" paragraph for a jointless body
            # (a quadcopter) claimed a safety check that never ran on anything.
            out.append("➖ **Datasheet-torque audit: not applicable** — this robot has no actuated joints, so "
                       "there is no per-joint datasheet torque to check and nothing was executed. This is NOT a "
                       "passed safety check; if you add actuated joints, rebuild so the audit can run.")
        elif astat == "pass":
            out.append(f"✅ **Datasheet-torque audit passed** ({audit.get('n_joints', '?')} joints). Every torque "
                       f"ceiling in `control_config.json` was re-derived from the BOM actuator sized for that "
                       f"joint, and `safety_filter.py` was **executed** on a command at "
                       f"{audit.get('overdrive_x', 10):g}× each joint's datasheet peak: nothing above the peak "
                       f"came back out, and `audit()` named every breach.")
        elif astat == "fail":
            out.append("⚠️ **This control stack FAILED the datasheet-torque audit — do not run it on hardware "
                       "until these are resolved:**")
            for v in (audit.get("violations") or [])[:6]:
                out.append(f"  - {v}")
        else:
            out.append("⚠️ The datasheet-torque audit **did not run** for this build, so the torque ceilings in "
                       "`control_config.json` have **not** been checked against the BOM actuators. Treat them as "
                       "unverified.")
        if audit.get("unaudited_scripts"):
            out.append(f"  - Not covered by that audit: "
                       f"{', '.join('`' + s + '`' for s in audit['unaudited_scripts'])} "
                       f"(not generated by Virturoid). The audit also cannot check that your own code routes its "
                       f"commands through `safety_filter.py` — that is on you.")
        out.append("")
    if not (has_ros2 or has_control or fusion or scripts):
        out.append("_No ROS2 package or control program was exported for this build._")
        out.append("")

    # 4. Caveats
    out.append("## 4. Honest sim-to-real caveats")
    out.append("")
    safe = ledger.get("safe_to_export")
    if ledger:
        out.append(f"- Export readiness: **{'cleared' if safe else 'NOT cleared'}** "
                   f"(highest attained: {ledger.get('highest_attained','?')}).")
        for issue in (ledger.get("issues") or [])[:5]:
            out.append(f"- {issue}")
    out.append("- These are simulation-derived estimates. Real actuator backlash, sensor noise, friction, and "
               "battery sag will differ; expect to re-tune the controller on hardware (the policy is "
               "domain-randomized to ease this, but a sim-to-real gap remains).")
    out.append("- The bill of materials sizes real off-the-shelf components but is not a manufacturing guarantee.")
    out.append("")
    return "\n".join(out)


def write_deployment_guide(output_dir) -> Path | None:
    """Write reports/deployment_guide.md. Returns the path, or None if there's nothing to deploy."""
    output_dir = Path(output_dir)
    if not (list(output_dir.rglob("bill_of_materials.json")) or list(output_dir.rglob("spec_sheet.json"))):
        return None
    md = build_deployment_guide(output_dir)
    out = output_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "deployment_guide.md"
    path.write_text(md, encoding="utf-8")
    return path
