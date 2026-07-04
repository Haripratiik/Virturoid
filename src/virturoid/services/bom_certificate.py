"""Executable-on-BOM certificate (breakthrough v5, B1 / wedge #2). After a policy is trained, we ask the honest
question almost no sim-first stack answers: *could the motion this policy produces actually run on the real motors
we would ship in the BOM?* A naive trainer clamps joint torque to a constant, so a policy freely commands peak
torque at high joint speed -- an operating point NO real servo can reach (the torque-speed curve). Such a policy
looks great in sim and shatters on hardware.

This module rolls the trained policy out on the (grounded) body while recording the APPLIED per-joint torque and
joint speed each step, sizes the smallest real catalog servo that covers each joint's peak demand (with a real
1.3x safety margin -- the part we would actually order), and grades the trajectory against that datasheet on six
gates. G4 (the torque-speed envelope) is the one a constant-clamp sim cannot see and is the crux of the wedge.

Every number is measured, not asserted. The certificate is a real artifact (``bom_certificate.json``) a customer
or reviewer can audit: per joint it names the motor, its datasheet limits, and the policy's realized demand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from virturoid.services.actuator_model import available_torque, knee_speed
from virturoid.services.component_catalog import select_actuator
from virturoid.services.morph_policy import recipe_rollout_morph


@dataclass
class GateResult:
    name: str
    passed: bool
    measured: float
    limit: float
    detail: str


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else 0.0


def certify_policy_on_bom(gene, policy=None, *, steps: int = 900, n_seeds: int = 3, cpg: dict | None = None,
                          margin: float = 1.3, knee_frac: float = 0.5, envelope_tol: float = 0.02,
                          out_path: str | None = None, **rollout_kw) -> dict:
    """Roll ``policy`` (or the scripted CPG if None) out on ``gene`` for ``n_seeds`` seeds, size a real servo per
    joint from its peak torque demand, and grade the realized motion against that servo's datasheet on 6 gates.

    Returns a JSON-able dict: ``{"pass": bool, "gates": [...], "joints": [...], "bom": [...], "summary": str}``.
    ``envelope_tol`` is the allowed fraction of driving samples that may exceed the torque-speed envelope (G4).
    """
    taus, qvs, survived_all, clamps = [], [], True, None
    for s in range(max(1, n_seeds)):
        r = recipe_rollout_morph(gene, policy, steps=steps, seed=s, cpg=cpg,
                                 record_actuation=True, **rollout_kw)
        if r.get("tau_cmd") is None or len(r["tau_cmd"]) == 0:
            survived_all = False
            continue
        taus.append(np.asarray(r["tau_cmd"], dtype=float))     # (T_s, n_joints)
        qvs.append(np.asarray(r["qvel"], dtype=float))
        clamps = np.asarray(r["clamps"], dtype=float)          # per-joint torque capacity -> the SHIPPED servo
        survived_all = survived_all and bool(r.get("survived", False))

    if not taus:                                               # the policy never produced a controllable step
        cert = {"pass": False, "gates": [], "joints": [], "bom": [],
                "summary": "UNCERTIFIABLE: the policy produced no finite actuation trace (fell immediately)."}
        if out_path:
            _write(out_path, cert)
        return cert

    tau = np.concatenate(taus, axis=0)                         # (sum_T, n_joints) APPLIED torque
    qv = np.concatenate(qvs, axis=0)
    cert = grade_actuation(tau, qv, clamps=clamps, margin=margin, knee_frac=knee_frac, envelope_tol=envelope_tol,
                           survived_all=survived_all, steps=steps, n_seeds=len(taus))
    if out_path:
        _write(out_path, cert)
    return cert


def grade_actuation(tau, qv, *, clamps=None, margin: float = 1.3, knee_frac: float = 0.5,
                    envelope_tol: float = 0.02, survived_all: bool = True, steps: int = 0, n_seeds: int = 1) -> dict:
    """Pure grader (no physics): given APPLIED per-joint torque ``tau`` and joint speed ``qv``, both ``(T, n)``,
    grade the trained motion against the SHIPPED servo per joint on 6 datasheet gates. Split out so it is
    deterministically testable on synthetic traces and reused by any rollout source (CPU eval, MJX replay,
    hardware log).

    ``clamps`` = per-joint torque CAPACITY (the compiled joint's torque limit = the grounding-selected actuator's
    peak). We size the servo that covers it -- the SAME part the control-side ``real_actuator`` clamp enforces, so
    the certificate grades exactly the motor the robot ships, and a policy trained under that clamp respects the
    torque-speed envelope (G4) by construction. When ``clamps`` is None we fall back to sizing from the joint's
    peak DEMAND with the safety margin (synthetic tests + un-grounded bodies)."""
    tau = np.asarray(tau, dtype=float); qv = np.asarray(qv, dtype=float)
    n_joints = tau.shape[1]

    joints, bom, viol_frac_all = [], [], []
    worst_headroom, thermal_ratio, speed_ratio, power_ratio = np.inf, 0.0, 0.0, 0.0
    seen = {}
    for j in range(n_joints):
        tj = np.abs(tau[:, j]); wj = np.abs(qv[:, j])
        demand_peak = _pct(tj, 99.0)                           # p99 to shrug off single-step spikes
        demand_rms = float(np.sqrt(np.mean(tau[:, j] ** 2)))
        speed_peak = _pct(wj, 99.0)
        # grade against the SHIPPED servo (covers the joint's torque capacity) so the cert matches the clamp; else
        # size the real part from the measured duty (peak w/ margin, continuous, AND speed) for un-grounded bodies
        servo = (select_actuator(float(clamps[j]), margin=1.0) if clamps is not None
                 else select_actuator(demand_peak, margin=margin,
                                      required_speed_radps=speed_peak, continuous_torque_nm=demand_rms))
        tau_pk, tau_rated, qd_max = servo.peak_torque_nm, servo.rated_torque_nm, servo.max_speed_radps
        qd_knee = knee_speed(qd_max, frac=knee_frac)

        # G4 core: at each DRIVING sample, is the applied torque within the speed-dependent envelope?
        driving = (tau[:, j] * qv[:, j]) >= 0.0
        avail = available_torque(qv[:, j], tau_pk, qd_knee, qd_max, xp=np)
        over_env = driving & (tj > avail + 1e-6)
        n_drive = int(np.sum(driving))
        viol_frac = float(np.sum(over_env)) / max(1, n_drive)
        viol_frac_all.append(viol_frac)

        # every gate as a ratio (>1.0 == infeasible) so the certificate is legible, not just a boolean
        headroom = tau_pk / max(demand_peak, 1e-6)
        worst_headroom = min(worst_headroom, headroom)
        thermal_ratio = max(thermal_ratio, demand_rms / max(tau_rated, 1e-6))
        speed_ratio = max(speed_ratio, speed_peak / max(qd_max, 1e-6))
        power_peak = float(np.max(tau[:, j] * qv[:, j])) if len(tau) else 0.0
        power_ceiling = tau_pk * qd_knee                       # ~rated mechanical power (peak torque at the knee)
        power_ratio = max(power_ratio, power_peak / max(power_ceiling, 1e-6))

        joints.append({"joint": j, "servo": servo.name, "demand_peak_nm": round(demand_peak, 3),
                       "demand_rms_nm": round(demand_rms, 3), "speed_peak_radps": round(speed_peak, 3),
                       "servo_peak_nm": tau_pk, "servo_rated_nm": tau_rated, "servo_noload_radps": qd_max,
                       "envelope_violation_frac": round(viol_frac, 4), "headroom": round(headroom, 2)})
        seen[servo.name] = seen.get(servo.name, 0) + 1
    for name, qty in sorted(seen.items()):
        a = next(x for x in _CAT() if x.name == name)
        bom.append({"part": name, "qty": qty, "peak_nm": a.peak_torque_nm, "mass_kg": a.mass_kg,
                    "price_usd": a.price_usd, "vendor": a.vendor})

    worst_viol = max(viol_frac_all) if viol_frac_all else 0.0
    gates = [
        GateResult("G1_peak_torque", worst_headroom >= 1.0, round(1.0 / worst_headroom, 3), 1.0,
                   "every joint's p99 torque demand is within its servo's peak torque"),
        GateResult("G2_thermal_continuous", thermal_ratio <= 1.0, round(thermal_ratio, 3), 1.0,
                   "RMS torque within the servo's rated (continuous) torque -- won't overheat"),
        GateResult("G3_speed_feasible", speed_ratio <= 1.0, round(speed_ratio, 3), 1.0,
                   "p99 joint speed within the servo's no-load speed"),
        GateResult("G4_torque_speed_envelope", worst_viol <= envelope_tol, round(worst_viol, 4), envelope_tol,
                   "driving-quadrant torque respects the speed-dependent envelope (the transfer killer)"),
        GateResult("G5_mechanical_power", power_ratio <= 1.0, round(power_ratio, 3), 1.0,
                   "peak mechanical output power within the motor's power ceiling"),
        GateResult("G6_selection_headroom", worst_headroom >= 1.15, round(worst_headroom, 2), 1.15,
                   "the weakest joint keeps real BOM safety margin over its demand"),
    ]
    ok = all(g.passed for g in gates)
    n_pass = sum(g.passed for g in gates)
    summary = (f"{'CERTIFIED' if ok else 'NOT CERTIFIED'}: {n_pass}/6 gates pass on the real BOM "
               f"({sum(b['qty'] for b in bom)} motors, {len(bom)} SKUs). "
               f"Worst torque-speed envelope violation {worst_viol:.1%}; weakest joint headroom {worst_headroom:.2f}x. "
               f"survived_all={survived_all}.")
    return {"pass": ok, "n_gates_pass": n_pass, "gates": [asdict(g) for g in gates],
            "joints": joints, "bom": bom, "survived_all": survived_all, "n_seeds": n_seeds,
            "steps": steps, "margin": margin, "summary": summary}


def _CAT():
    from virturoid.services.component_catalog import ACTUATORS
    return ACTUATORS


def _write(path: str, cert: dict) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
