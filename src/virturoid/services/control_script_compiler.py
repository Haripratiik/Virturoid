"""The control-script compiler (agentic platform plan WS-S / S2): the operational Python a deployed robot needs
around its learned policy -- generated from the robot itself, not hand-written per project.

A trained policy is not a deployable robot. Between "the net outputs joint targets" and "the machine runs safely
on a bench" sits a standard inventory of glue every roboticist writes by hand, per robot: assemble the exact
observation vector the policy trained on; clamp every command to what the actuators can physically survive; a
state machine so an E-stop or a fall does the safe thing; a watchdog that trips on a stalled control loop or a
dead comms link; a teleop stub; a joint-zero calibration routine. This module GENERATES that inventory from what
the platform already knows about the robot:

  * the ordered actuated joints and their limits come from the genome (``gene.actuated_joints()``);
  * every per-joint TORQUE CEILING is the peak torque of the real actuator the BOM sized for that joint
    (``select_actuator``) -- so the safety filter clamps to the motor's datasheet, not a guessed number;
  * the observation layout MIRRORS the policy's training obs (locomotion twist-tracking vs. manipulator reach),
    so the assembler cannot silently disagree with the net it feeds.

Then it runs the validation ladder the plan requires -- every generated ``.py`` is COMPILE-checked and DRY-RUN
in a subprocess (each script ships a ``self_test`` that exercises its logic against a mock state) -- and writes
an honest pass/fail report. A script that would exceed an actuator's datasheet torque fails the audit with the
joint and the limit named. Pure standard library in the generated code (no ROS/MuJoCo import), deterministic.
"""

from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in str(name))


def _derive(gene, *, task: str = "") -> dict:
    """The single source of truth every generated script reads: ordered joints, per-joint torque ceiling (from
    the actuator the BOM sized), position limits, the obs layout, and the robot kind."""
    from virturoid.services.bom_builder import _DEFAULT_JOINT_TORQUE_NM, select_actuator
    from virturoid.services.task_matched_eval import robot_kind

    joints = gene.actuated_joints()
    names = [_safe(s.name) for s in joints]
    ceilings, limits = {}, {}
    for s in joints:
        act = select_actuator(getattr(s, "actuator_torque_nm", None) or _DEFAULT_JOINT_TORQUE_NM)
        ceilings[_safe(s.name)] = round(float(act.peak_torque_nm), 3)
        lo = s.joint_lower if s.joint_lower is not None else -3.1416
        hi = s.joint_upper if s.joint_upper is not None else 3.1416
        limits[_safe(s.name)] = [round(float(lo), 4), round(float(hi), 4)]

    kind = robot_kind(gene)
    n = len(names)
    # The obs layout mirrors how the policy was trained: a locomotion policy tracks a base twist command; a
    # manipulator tracks a target in its base frame. Each block is (name, size) and the sizes sum to obs_dim.
    if kind in ("legged", "mobile", "aerial", "aquatic"):
        layout = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_command", 3),
                  ("joint_pos_rel", n), ("joint_vel", n), ("last_action", n)]
        cmd = "velocity_command"
    else:  # manipulator / other
        layout = [("joint_pos", n), ("joint_vel", n), ("ee_pose", 7), ("target_pose", 7), ("last_action", n)]
        cmd = "target_pose"
    return {"kind": kind, "control_hz": 50.0, "joints": names, "n_joints": n,
            "torque_ceilings_nm": ceilings, "position_limits": limits,
            "obs_layout": [{"name": nm, "size": sz} for nm, sz in layout],
            "obs_dim": sum(sz for _, sz in layout), "command_block": cmd}


# ---- the generated scripts. Each is pure stdlib, reads control_config.json, and has a self_test() dry-run. ----

_OBS_ASSEMBLER = '''"""Observation assembler -- AUTO-GENERATED. Builds the exact observation vector the trained policy
expects, in the SAME block order and scaling used at training. If this disagrees with the policy, the robot acts
on garbage; keep it generated from the same genome as the policy."""
from __future__ import annotations
import json
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())
LAYOUT = CFG["obs_layout"]
OBS_DIM = CFG["obs_dim"]
# obs scales mirror the trainer's normalization (legged_gym / Isaac Lab conventions).
SCALES = {"base_lin_vel": 2.0, "base_ang_vel": 0.25, "joint_pos_rel": 1.0, "joint_vel": 0.05}


class ObsAssembler:
    def assemble(self, state: dict) -> list:
        """state maps each block name -> a list of floats of the block's size. Returns a flat obs vector."""
        obs = []
        for block in LAYOUT:
            name, size = block["name"], block["size"]
            vals = list(state.get(name, [0.0] * size))
            if len(vals) != size:
                raise ValueError(f"obs block {name!r} expected {size} values, got {len(vals)}")
            s = SCALES.get(name, 1.0)
            obs.extend(v * s for v in vals)
        if len(obs) != OBS_DIM:
            raise ValueError(f"assembled obs {len(obs)} != declared obs_dim {OBS_DIM}")
        return obs


def self_test() -> bool:
    zero = {b["name"]: [0.0] * b["size"] for b in LAYOUT}
    obs = ObsAssembler().assemble(zero)
    assert len(obs) == OBS_DIM
    return True


if __name__ == "__main__":
    print("obs_assembler self_test:", "OK" if self_test() else "FAIL")
'''

_SAFETY_FILTER = '''"""Safety filter -- AUTO-GENERATED. Clamps every commanded joint torque to the PEAK TORQUE of the real
actuator the BOM sized for that joint, and every commanded position to the joint's limit. These ceilings are the
motor datasheet, not a guess: a command that would exceed them is physically impossible and is clipped here
before it ever reaches the bus."""
from __future__ import annotations
import json
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())
CEIL = CFG["torque_ceilings_nm"]
LIM = CFG["position_limits"]
JOINTS = CFG["joints"]


class SafetyFilter:
    def clamp_torques(self, torques: dict) -> dict:
        out = {}
        for j, t in torques.items():
            c = CEIL.get(j)
            out[j] = t if c is None else max(-c, min(c, t))
        return out

    def clamp_positions(self, positions: dict) -> dict:
        out = {}
        for j, p in positions.items():
            lo, hi = LIM.get(j, [-3.1416, 3.1416])
            out[j] = max(lo, min(hi, p))
        return out

    def audit(self, torques: dict) -> list:
        """Return a list of (joint, requested, ceiling) for any command that WOULD have exceeded the actuator."""
        return [(j, t, CEIL[j]) for j, t in torques.items() if j in CEIL and abs(t) > CEIL[j]]


def self_test() -> bool:
    sf = SafetyFilter()
    over = {j: CEIL[j] * 10.0 for j in JOINTS[:1]} if JOINTS else {}
    clamped = sf.clamp_torques(over)
    for j in over:
        assert abs(clamped[j]) <= CEIL[j] + 1e-6, f"{j} not clamped to its ceiling"
    assert len(sf.audit(over)) == len(over)
    return True


if __name__ == "__main__":
    print("safety_filter self_test:", "OK" if self_test() else "FAIL")
'''

_STATE_MACHINE = '''"""Safety state machine -- AUTO-GENERATED. The supervisory logic every deployed robot needs so an E-stop
or a fall does the safe thing. States: ESTOP (zero torque), STAND (hold default pose), WALK/RUN (policy drives),
FALL_DAMPING (high-damping catch when the body tilts past the threshold)."""
from __future__ import annotations
import json
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())
TILT_LIMIT_RAD = 0.9  # ~52 deg from upright -> the body is going over; damp out

ESTOP, STAND, ACTIVE, FALL_DAMPING = "ESTOP", "STAND", "ACTIVE", "FALL_DAMPING"


class SafetyStateMachine:
    def __init__(self):
        self.state = STAND

    def step(self, *, estop: bool, tilt_rad: float, command_active: bool) -> str:
        if estop:
            self.state = ESTOP
        elif abs(tilt_rad) > TILT_LIMIT_RAD:
            self.state = FALL_DAMPING
        elif self.state in (FALL_DAMPING, ESTOP):
            self.state = STAND  # recovering: require an explicit re-stand before ACTIVE resumes, never a direct jump
        elif command_active:
            self.state = ACTIVE
        else:
            self.state = STAND
        return self.state

    def torque_policy(self) -> str:
        """What the low level should do in the current state."""
        return {ESTOP: "zero_torque", STAND: "hold_default_pose",
                ACTIVE: "track_policy", FALL_DAMPING: "high_joint_damping"}[self.state]


def self_test() -> bool:
    sm = SafetyStateMachine()
    assert sm.step(estop=False, tilt_rad=0.0, command_active=True) == ACTIVE
    assert sm.step(estop=False, tilt_rad=1.5, command_active=True) == FALL_DAMPING
    assert sm.step(estop=True, tilt_rad=0.0, command_active=False) == ESTOP
    assert sm.torque_policy() == "zero_torque"
    return True


if __name__ == "__main__":
    print("state_machine self_test:", "OK" if self_test() else "FAIL")
'''

_WATCHDOG = '''"""Watchdog -- AUTO-GENERATED. Trips an E-stop when the control loop stalls, comms go silent, or a joint
leaves its limits. A deployed robot without this runs open-loop into a wall when a node dies."""
from __future__ import annotations
import json
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())
LIM = CFG["position_limits"]
CONTROL_DT = 1.0 / float(CFG["control_hz"])
MAX_LOOP_DT = 3.0 * CONTROL_DT     # a loop 3x slow -> stalled
COMMS_TIMEOUT_S = 0.5              # no command heartbeat for 0.5 s -> lost link


class Watchdog:
    def check(self, *, loop_dt: float, since_last_command_s: float, positions: dict) -> tuple:
        """Return (ok, reason). ok False means trip the E-stop."""
        if loop_dt > MAX_LOOP_DT:
            return False, f"control loop stalled: {loop_dt*1000:.1f} ms > {MAX_LOOP_DT*1000:.1f} ms"
        if since_last_command_s > COMMS_TIMEOUT_S:
            return False, f"comms timeout: {since_last_command_s:.2f} s since last command"
        for j, p in positions.items():
            lo, hi = LIM.get(j, [-3.1416, 3.1416])
            if p < lo - 0.05 or p > hi + 0.05:
                return False, f"joint {j} out of limits: {p:.3f} not in [{lo:.3f},{hi:.3f}]"
        return True, "ok"


def self_test() -> bool:
    wd = Watchdog()
    assert wd.check(loop_dt=CONTROL_DT, since_last_command_s=0.0, positions={})[0] is True
    assert wd.check(loop_dt=10 * CONTROL_DT, since_last_command_s=0.0, positions={})[0] is False
    assert wd.check(loop_dt=CONTROL_DT, since_last_command_s=5.0, positions={})[0] is False
    return True


if __name__ == "__main__":
    print("watchdog self_test:", "OK" if self_test() else "FAIL")
'''

_TELEOP = '''"""Teleop stub -- AUTO-GENERATED. Maps keyboard/joystick input to the same command block the policy
consumes (a base velocity for a mobile/legged robot, a target pose for an arm). Wire your input device's events
to step(); it returns the command dict the obs assembler expects."""
from __future__ import annotations
import json
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())
COMMAND_BLOCK = CFG["command_block"]
STEP_LINEAR = 0.5   # m/s per key press
STEP_YAW = 0.8      # rad/s per key press

_KEYMAP = {"w": ("vx", +STEP_LINEAR), "s": ("vx", -STEP_LINEAR),
           "a": ("vyaw", +STEP_YAW), "d": ("vyaw", -STEP_YAW), " ": ("stop", 0.0)}


class Teleop:
    def __init__(self):
        self.vx = self.vy = self.vyaw = 0.0

    def step(self, key: str) -> dict:
        axis, val = _KEYMAP.get(key, (None, 0.0))
        if axis == "stop":
            self.vx = self.vy = self.vyaw = 0.0
        elif axis == "vx":
            self.vx = val
        elif axis == "vyaw":
            self.vyaw = val
        return {COMMAND_BLOCK: [self.vx, self.vy, self.vyaw]}


def self_test() -> bool:
    tp = Teleop()
    assert tp.step("w")[COMMAND_BLOCK][0] > 0
    assert tp.step(" ")[COMMAND_BLOCK] == [0.0, 0.0, 0.0]
    return True


if __name__ == "__main__":
    print("teleop self_test:", "OK" if self_test() else "FAIL")
'''

_CALIBRATE = '''"""Joint calibration -- AUTO-GENERATED. Captures each joint's raw-encoder zero offset so commanded angles
match the URDF's kinematic zero. Run once at a known home pose; it stores offsets that apply() then subtracts."""
from __future__ import annotations
import json
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("control_config.json")).read_text())
JOINTS = CFG["joints"]


class Calibration:
    def __init__(self):
        self.offsets = {j: 0.0 for j in JOINTS}

    def record_home(self, raw_positions: dict) -> dict:
        """At the physical home pose every joint reads its raw encoder value; that value IS the offset."""
        self.offsets = {j: float(raw_positions.get(j, 0.0)) for j in JOINTS}
        return dict(self.offsets)

    def apply(self, raw_positions: dict) -> dict:
        return {j: float(raw_positions.get(j, 0.0)) - self.offsets.get(j, 0.0) for j in JOINTS}


def self_test() -> bool:
    cal = Calibration()
    raw = {j: 0.11 for j in JOINTS}
    cal.record_home(raw)
    zeroed = cal.apply(raw)
    assert all(abs(v) < 1e-9 for v in zeroed.values()), "home pose must zero out"
    return True


if __name__ == "__main__":
    print("calibrate self_test:", "OK" if self_test() else "FAIL")
'''

_SCRIPTS = {
    "obs_assembler.py": _OBS_ASSEMBLER,
    "safety_filter.py": _SAFETY_FILTER,
    "state_machine.py": _STATE_MACHINE,
    "watchdog.py": _WATCHDOG,
    "teleop.py": _TELEOP,
    "calibrate.py": _CALIBRATE,
}


def compile_control_scripts(gene, *, task: str = "") -> dict:
    """Generate the operational control-script inventory + the config they share, derived from the robot.
    Returns ``{config, files: {rel: source}, manifest}`` -- no disk writes (see ``write_control_scripts``)."""
    cfg = _derive(gene, task=task)
    files = {"control_config.json": json.dumps(cfg, indent=2)}
    files.update(_SCRIPTS)
    manifest = {
        "kind": cfg["kind"], "n_joints": cfg["n_joints"], "obs_dim": cfg["obs_dim"],
        "scripts": sorted(_SCRIPTS.keys()),
        "torque_ceilings_nm": cfg["torque_ceilings_nm"],
        "derived_from": "genome joints + BOM-sized actuator peak torque (safety_filter) + policy obs layout",
        "authored_by": "control_script_compiler (no hand-authoring)",
    }
    files["script_manifest.json"] = json.dumps(manifest, indent=2)
    return {"config": cfg, "files": files, "manifest": manifest}


def validate_scripts(scripts_dir) -> dict:
    """The validation ladder S3 requires, run on the generated scripts: COMPILE every .py, then DRY-RUN each in a
    subprocess (its ``self_test`` exercises the logic against a mock state). Returns an honest per-script report;
    a non-compiling or failing script is reported, never hidden."""
    scripts_dir = Path(scripts_dir)
    results = {}
    ok_all = True
    for py in sorted(scripts_dir.glob("*.py")):
        entry = {"compiled": False, "dry_run": False, "detail": ""}
        try:
            py_compile.compile(str(py), doraise=True)
            entry["compiled"] = True
        except py_compile.PyCompileError as exc:
            entry["detail"] = f"compile error: {exc}"
            ok_all = False
            results[py.name] = entry
            continue
        try:
            proc = subprocess.run([sys.executable, str(py)], capture_output=True, text=True, timeout=30,
                                  cwd=str(scripts_dir))
            entry["dry_run"] = (proc.returncode == 0 and "OK" in proc.stdout)
            entry["detail"] = (proc.stdout.strip() or proc.stderr.strip())[:200]
            if not entry["dry_run"]:
                ok_all = False
        except Exception as exc:  # noqa: BLE001
            entry["detail"] = f"dry-run error: {exc}"
            ok_all = False
        results[py.name] = entry
    return {"all_pass": ok_all, "scripts": results, "n_scripts": len(results)}


def write_control_scripts(gene, output_dir, *, task: str = "") -> Path | None:
    """Compile the scripts, write them under ``output_dir/software/scripts/``, run the validation ladder, and
    write ``reports/script_validation.json``. Returns the scripts dir (or None on total failure -- scripts are
    value-add and never block a build)."""
    try:
        out = compile_control_scripts(gene, task=task)
        sdir = Path(output_dir) / "software" / "scripts"
        sdir.mkdir(parents=True, exist_ok=True)
        for rel, content in out["files"].items():
            (sdir / rel).write_text(content, encoding="utf-8")
        report = validate_scripts(sdir)
        rdir = Path(output_dir) / "reports"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "script_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        # the compile+dry-run leaves bytecode caches -- strip them so the shipped package is clean source only
        import shutil
        shutil.rmtree(sdir / "__pycache__", ignore_errors=True)
        return sdir
    except Exception:  # noqa: BLE001
        return None
