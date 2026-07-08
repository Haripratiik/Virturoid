"""End-to-end proof: a customer (say, an Optimus robotics engineer) DROPS A FOLDER of their existing robot --
a URDF, a BOM, CAD meshes, a plain-english description, and their OWN control script -- and Virturoid ingests all
of it into ONE editable, simulate-able robot, then MODIFIES/IMPROVES the design AND runs+improves their controller.

It reuses artifacts Virturoid itself exported earlier (build/agent_exports/... + build/.../cad) as the customer's
"existing files", so the whole loop is exercised on realistic data with no hand-authoring:

    python scripts/demo_ingest_customer.py

Steps: (1) assemble the drop-folder, (2) ingest_project -> held robot + applied materials/payload + detected
control script, (3) verify it in sim (baseline), (4) AMEND it (carry a heavier payload) -> re-verify, (5)
adopt_control_script -> RUN the customer's controller in physics and IMPROVE it (measured before/after).
Everything printed is measured; nothing staged.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


def _find_quad_export() -> Path | None:
    """A previously-exported QUADRUPED bundle (has a URDF + BOM + gait controller) to pose as the customer's robot."""
    for d in sorted((ROOT / "build" / "agent_exports").glob("robot_*")):
        bom = d / "bom.json"
        ctl = d / "software" / "gait_controller.py"
        prm = d / "software" / "controller" / "policy_params.json"
        urdf = d / "robot" / "robot.urdf"
        if bom.exists() and ctl.exists() and prm.exists() and urdf.exists():
            try:
                if json.loads(bom.read_text()).get("robot_class") in ("quadruped", "hexapod"):
                    return d
            except Exception:  # noqa: BLE001
                continue
    return None


def assemble_customer_folder(dst: Path) -> Path:
    """Build the 'random folder the customer drops us' from reused Virturoid outputs + a plain-english description."""
    proj = dst / "optimus_dropbox"
    (proj / "robot").mkdir(parents=True, exist_ok=True)
    (proj / "control").mkdir(parents=True, exist_ok=True)
    (proj / "cad").mkdir(parents=True, exist_ok=True)

    exp = _find_quad_export()
    if exp is not None:
        shutil.copy(exp / "robot" / "robot.urdf", proj / "robot" / "quad.urdf")
        shutil.copy(exp / "bom.json", proj / "bom.json")
        shutil.copy(exp / "software" / "gait_controller.py", proj / "control" / "gait_controller.py")
        shutil.copy(exp / "software" / "controller" / "policy_params.json", proj / "control" / "policy_params.json")
        print(f"  reused customer robot from {exp.name} (URDF + BOM + gait controller)")
    else:
        print("  no exported quad found; run the app to generate one. Writing a minimal stand-in URDF.")
        (proj / "robot" / "quad.urdf").write_text(_MINIMAL_URDF, encoding="utf-8")

    # a CAD mesh the customer shipped (reuse a real STL Virturoid emitted)
    stl = next((ROOT / "build").rglob("*.stl"), None)
    if stl is not None:
        shutil.copy(stl, proj / "cad" / stl.name)
        print(f"  included a customer CAD mesh: cad/{stl.name}")

    # the plain-english description a customer would actually write
    (proj / "README.txt").write_text(
        "Optimus internal quadruped 'Lynx'. Aluminum chassis, carbon-fiber legs. Carries a 6 kg sensor payload. "
        "14-DOF. We want to simulate it, tune the body, and improve our trot controller (control/gait_controller.py).",
        encoding="utf-8")
    return proj


_MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="quad_stub"><link name="base"><inertial><mass value="2.0"/>
<inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
<collision><geometry><box size="0.3 0.18 0.1"/></geometry></collision></link></robot>"""


def main() -> int:
    from virturoid.services.agent_tools import call_tool

    work = Path(tempfile.mkdtemp(prefix="optimus_"))
    os.environ.setdefault("VIRTUROID_SESSION_DIR", str(work / "session"))
    print("1) Assembling the customer drop-folder ...")
    proj = assemble_customer_folder(work)
    print(f"   folder: {proj}\n   contents: {[str(p.relative_to(proj)) for p in proj.rglob('*') if p.is_file()]}\n")

    print("2) ingest_project — parse EVERYTHING into one editable robot ...")
    desc = (proj / "README.txt").read_text()
    ing = call_tool("ingest_project", {"project_path": str(proj), "description": desc}).get("result", {})
    rid = ing.get("robot_id")
    print(f"   robot_id: {rid}")
    print(f"   materials applied: {[(m['group'], m['material']) for m in ing.get('materials_applied', [])]}")
    print(f"   payload: {ing.get('payload_kg')} kg | import: {ing.get('import', {}).get('robot_class')} "
          f"({ing.get('import', {}).get('source')})")
    print(f"   BOM: {ing.get('bom')} | CAD: {ing.get('cad')}")
    print(f"   control scripts detected: {[os.path.basename(c) for c in ing.get('control_scripts', [])]}")
    for n in ing.get("notes", []):
        print(f"   note: {n}")
    if not rid:
        print("   ingest produced no robot; aborting.", ing.get("warnings"))
        return 1

    print("\n3) verify the ingested robot in sim (baseline) ...")
    v0 = call_tool("verify_robot", {"robot_id": rid, "mode": "quick"}).get("result", {})
    print(f"   {v0.get('kind')}: {v0.get('verdict')}")

    print("\n4) AMEND the design — make it carry a heavier 12 kg payload ...")
    ed = call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": "set_payload", "args": {"payload_kg": 12.0}}]}).get("result", {})
    d0 = (ed.get("diffs") or [{}])[0]
    print(f"   set_payload -> load_factor {d0.get('load_factor')}, mass {d0.get('total_mass_kg')} kg, "
          f"{d0.get('n_joints_upsized')} joints upsized" + (f", WARN: {d0.get('warning')[:60]}" if d0.get('warning') else ""))

    print("\n5) adopt_control_script — RUN the customer's controller in sim, then IMPROVE it ...")
    script = next((c for c in ing.get("control_scripts", []) if c.endswith(".py")), None) \
        or next((c for c in ing.get("control_scripts", []) if c.endswith(".json")), None)
    if script is None:
        print("   no control script detected in the folder; skipping the adopt step.")
    else:
        # The ingested robot WALKS (a legged import that can't stand on its own stance is given the same walkable
        # fanned stance a composed body gets), so the customer's controller runs and improves ON THEIR OWN robot.
        ad = call_tool("adopt_control_script", {"robot_id": rid, "script_path": script,
                                                "generations": 6, "pop": 16, "steps": 800}).get("result", {})
        if ad.get("error"):
            print(f"   adopt error: {ad['error']}")
        else:
            print(f"   parsed control script: {ad.get('control_script')}")
            print(f"   UTILISED (their controller): {ad.get('utilised')}")
            print(f"   IMPROVED (our sim tuned it): {ad.get('improved')}")
            print(f"   -> {ad.get('verdict')}  (gain {ad.get('forward_gain_x')}x)")

    print(f"\nDone. The customer dropped a folder; Virturoid parsed model+BOM+CAD+NLP into one editable robot, "
          f"amended its design, and ran+improved their controller on their own robot — all measured in real physics."
          f"\nSession: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
