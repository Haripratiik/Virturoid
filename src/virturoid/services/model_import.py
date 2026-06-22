"""Import an EXISTING robot model (MJCF or URDF — i.e. a CAD-exported robot) so the platform can iterate
on it: simulate it, learn control for it (MorphPolicy / MJX-GPU), evaluate it, and bank skills + tips —
the same flywheel as composed robots. This is the "I already have a robot, improve it" half of the goal.

MuJoCo loads both MJCF and URDF natively; we normalize to an MJCF string (ensuring a ground plane) that
flows through ``robot_mjcf`` everywhere. Meshed URDFs need their mesh/asset files alongside the file.
"""

from __future__ import annotations

from pathlib import Path


def _ensure_floor(xml: str) -> str:
    if 'type="plane"' in xml:
        return xml
    floor = '    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.30 0.34 0.40 1"/>\n'
    return xml.replace("<worldbody>", "<worldbody>\n" + floor, 1)


def _add_actuators(mjcf: str, model) -> tuple:
    """Add a motor per movable joint when the imported model has none — URDFs import joints (with effort
    limits) but no MuJoCo actuators, so without this the robot has DOFs but nothing to drive them. Lets an
    imported CAD model be CONTROLLED and improved, not just simulated."""
    import mujoco

    motors = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (2, 3):                    # 2=slide, 3=hinge (skip free/ball)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name:
                motors.append(f'    <motor joint="{name}" ctrlrange="-1 1" gear="20"/>')
    if not motors:
        return mjcf, model
    block = "  <actuator>\n" + "\n".join(motors) + "\n  </actuator>\n"
    mjcf2 = mjcf.replace("</mujoco>", block + "</mujoco>", 1)
    return mjcf2, mujoco.MjModel.from_xml_string(mjcf2)


def import_model(path: str) -> dict:
    """Load an MJCF/URDF robot file. Returns ``{mjcf, name, parts, actuated, free_base, ok, note}`` —
    ``mjcf`` is a normalized MJCF string (ground ensured) usable everywhere a composed gene is."""
    import os
    import tempfile

    import mujoco

    p = Path(path)
    if not p.exists():
        return {"ok": False, "note": f"file not found: {path}"}
    sfx = p.suffix.lower()
    try:
        if sfx in (".xml", ".mjcf"):
            mjcf = _ensure_floor(p.read_text(encoding="utf-8"))
            model = mujoco.MjModel.from_xml_string(mjcf)            # validate it loads
        elif sfx == ".urdf":
            model = mujoco.MjModel.from_xml_path(str(p))            # MuJoCo compiles URDF
            tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False); tmp.close()
            mujoco.mj_saveLastXML(tmp.name, model)
            mjcf = _ensure_floor(Path(tmp.name).read_text(encoding="utf-8"))
            os.unlink(tmp.name)
            model = mujoco.MjModel.from_xml_string(mjcf)            # re-validate the normalized MJCF
        else:
            return {"ok": False, "note": f"unsupported format {sfx!r} — use .xml/.mjcf or .urdf"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "note": f"could not load model: {exc}"}

    if model.nu == 0:                          # URDF joints import without motors -> add them so it's drivable
        try:
            mjcf, model = _add_actuators(mjcf, model)
        except Exception:  # noqa: BLE001 - keep the importable model even if actuator synthesis fails
            pass

    free = any(int(model.jnt_type[j]) == 0 for j in range(model.njnt))   # free joint -> can move/locomote
    note = "ready to iterate on" if model.nu > 0 else "no actuators — can simulate but not learn control"
    return {"ok": True, "mjcf": mjcf, "name": p.stem, "parts": int(model.nbody) - 1,
            "actuated": int(model.nu), "free_base": bool(free), "note": note}
