"""Certificate v2 — tiered sim-to-real evidence (master_plan_v6 §9 / WS-E).

Tier-0 (verdict_certificate) is an honest sim verdict; hardware credibility is the next frontier. The research
verdict: a nominal sim verdict alone is weak (Simulation Optimization Bias is positive *by construction*), but the
gap is dominated by a small, known set of quantities — mostly the ACTUATOR (armature, damping, friction, latency,
PD) and MASS. So Tier-1 makes the certificate carry the evidence a bring-up engineer actually checks, ranked by
credibility-per-FLOP (§9.1):

  1. **Frozen-policy DR sweep** — re-verify the SAME controller across Monte-Carlo draws of mass / actuator
     strength / payload (the dominant factors), report pass-rate + per-parameter **failure boundaries** + a
     **Clopper-Pearson** bound w.r.t. the sampled distribution.
  2. **Actuator-fidelity level** (L0..L3) — we ship **L1** (datasheet torque-speed clamp): declared, and the
     verdict was earned under the clamp (URDF-only actuators *fail to walk* on real hardware — ETH 2025).
  3. **Per-joint torque/thermal margins** from the verified trajectory — reused from ``bom_certificate`` (peak vs
     motor peak, RMS vs continuous rating, speed, mechanical power). The numbers a builder computes first.
  4. **Numerics sensitivity** — re-verdict at a halved rollout budget (a knife-edge-result alarm). This is a
     BUDGET probe, not the halved-timestep integrator sweep §9.1 asks for; it says so on every certificate and
     it reports ``probed: False`` on bodies it cannot run on, instead of a free ``invariant: True``.
  5. **Model-sanity gate** — certificate VOID unless green: physically-valid inertias (triangle inequality),
     joint limits present, finite BOM-consistent mass.

Design rules (§9.2): margins + failure boundaries, never bare booleans; the **scope / not_modeled** block is
load-bearing; ``use_history.predictivity_srcc`` starts **null** ("unmeasured") and becomes the certificate's own
credibility flywheel as real builds close the loop.

**Honest limits (§9.3) are DERIVED, not declared.** ``scope.valid_iff``, ``scope.not_modeled``,
``scope.claim_ceiling`` and ``honest_limits`` used to be four string constants — byte-identical on every
certificate for every robot, in this same module that calls them "mandatory, not boilerplate". ``claim_ceiling``
asserted unconditionally that "the failure modes we can model were swept and passed with the stated margins",
which was false on every certificate built with ``run_dr=False`` (nothing was swept), on every DR sweep that
passed less than all of its draws, and on every body whose headline verdict was FELL. They are now computed from
what THIS run actually did; the residue that genuinely cannot be measured (failure categories outside
rigid-body sim, the standing sim-optimism caveat) is kept but LABELLED ``declared`` and separated from the
measured half, which carries the number it came from.
"""
from __future__ import annotations

import math
import random
from dataclasses import replace

# The rung labels are CLAIMS, so they say only what the code does. L1 used to read "datasheet torque-speed
# clamp + PD + latency"; the clamp and the PD are real (``actuator_model.clamp_torque``, the recipe gains) and
# the LATENCY was not -- there is no actuation-delay representation anywhere in the compiled model or the
# rollout loop, which ``sysid.calibration.model_represents_actuation_delay`` now checks rather than assumes.
# A label is the shortest thing on a certificate and the most likely to be quoted, so it does not get to
# claim a term the simulator does not carry.
ACTUATOR_LEVELS = {0: "L0 ideal (no actuator model)",
                   1: "L1 datasheet torque-speed clamp + PD (SHIPPED; actuation latency NOT modelled)",
                   2: "L2 bench-identified actuator",
                   3: "L3 learned actuator-network"}

# The community DR envelopes that empirically produced zero-shot transfer (scaled to the factors we sweep at
# gene level — mass/actuator/payload dominate per ETH 2025). Friction µ and latency are model-level (Tier-2).
DR_RANGES = {"mass_scale": (0.85, 1.15), "motor_scale": (0.90, 1.10), "payload_frac": (0.0, 0.15)}


# ------------------------------------------------------------------ Clopper-Pearson (pure, no scipy)
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) in [0,1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse regularised incomplete beta by bisection (deterministic)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if reg_incomplete_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k_fail: int, n: int, *, alpha: float = 0.05) -> dict:
    """Exact Clopper-Pearson two-sided CI on the FAILURE probability given ``k_fail`` failures in ``n`` trials
    (w.r.t. the sampled distribution). k=0 gives the headline "0/n ⇒ ≤ hi failure at 95%"."""
    if n <= 0:
        return {"p_fail": None, "lo": None, "hi": None, "n": 0, "k_fail": 0}
    lo = 0.0 if k_fail == 0 else _beta_quantile(alpha / 2, k_fail, n - k_fail + 1)
    hi = 1.0 if k_fail == n else _beta_quantile(1 - alpha / 2, k_fail + 1, n - k_fail)
    return {"p_fail": round(k_fail / n, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n, "k_fail": k_fail,
            "note": f"{k_fail}/{n} failures -> true failure rate in [{lo:.3f}, {hi:.3f}] at {int((1-alpha)*100)}% "
                    f"confidence, w.r.t. the sampled perturbation distribution ONLY"}


# ------------------------------------------------------------------ model sanity (VOID gate)
def model_sanity(gene) -> dict:
    """Certificate VOID unless green: physically-valid inertias (triangle inequality), joint limits present,
    finite positive mass. The first things a bring-up engineer checks; a red here voids every downstream claim."""
    issues = []
    try:
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
        m = mujoco.MjModel.from_xml_string(xml)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "issues": [f"model did not compile: {type(exc).__name__}: {exc}"[:160]],
                "inertia_valid": False, "joint_limits_present": False, "mass_finite": False}
    total_mass = float(m.body_mass.sum())
    mass_finite = math.isfinite(total_mass) and total_mass > 0
    if not mass_finite:
        issues.append("total mass is non-finite or non-positive")
    # inertia triangle inequality on each body's diagonal principal inertia (Drake-strict)
    inertia_valid = True
    for b in range(1, m.nbody):
        ix, iy, iz = (float(v) for v in m.body_inertia[b])
        if min(ix, iy, iz) < 0:
            inertia_valid = False; issues.append(f"body {b} has a negative principal inertia"); break
        if not (ix + iy >= iz - 1e-9 and iy + iz >= ix - 1e-9 and ix + iz >= iy - 1e-9):
            inertia_valid = False; issues.append(f"body {b} violates the inertia triangle inequality"); break
    # joint limits: every actuated (non-free) hinge/slide should be limited
    actuated = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (2, 3)]   # hinge=3, slide=2 in mjtJoint
    limited = [j for j in actuated if bool(m.jnt_limited[j])]
    joint_limits_present = (len(actuated) == 0) or (len(limited) >= max(1, int(0.8 * len(actuated))))
    if not joint_limits_present:
        issues.append(f"only {len(limited)}/{len(actuated)} actuated joints carry limits")
    ok = mass_finite and inertia_valid and joint_limits_present
    return {"ok": ok, "issues": issues, "inertia_valid": inertia_valid,
            "joint_limits_present": joint_limits_present, "mass_finite": mass_finite,
            "total_mass_kg": round(total_mass, 4), "n_bodies": int(m.nbody),
            "inertia_provenance": "compiler-derived from primitive geometry (CAD-derived inertia can be ~4× off on "
                                  "real hardware — measure on the built robot before trusting margins)"}


# ------------------------------------------------------------------ actuator fidelity level
def actuator_fidelity_level(gene) -> dict:
    """Declare the actuator model the verdict was earned under, and EARN the rung rather than assert it.

    L0/L1 are structural: no per-joint torque clamp → L0 (ideal); joints clamped to the BOM servo's datasheet
    limit → L1. L2 is *bench-identified actuator*, and it is now a computed verdict rather than a dead
    constant: ``sysid.calibration.l2_requirements`` checks six things — a fit is attached, the log it was
    fitted to was measured on physical hardware, the fit covers the machine, applying it measurably improves
    how the simulator TRACKS that log, the identified actuation delay is APPLIED and not merely reported, and
    the torque-speed envelope is identified rather than taken from the datasheet — and reports which ones are
    missing with the evidence for each.

    The tracking requirement is the one that is not about the experiment. The other five can all be satisfied
    by a fit whose parameters are absorbing an error they do not name: a robot built with 30% heavier links
    came back "identified" on every joint, with intervals excluding the truth, having made tracking marginally
    WORSE. Coverage counts APPLIED parameters and that fit applies none, so the rung is refused twice over.

    On this build the rung does NOT move. Three of the six block on our own best fit, and two of those three
    are ours rather than the customer's: there is nowhere in
    the compiled model to put an identified transport delay, and a safe excitation (bounded well under
    datasheet peak torque and no-load speed, because it runs on somebody's robot) cannot reach the saturation
    regime where the torque-speed knee lives. A fitted robot therefore reports **L1 with its joint dissipation
    and reflected inertia bench-identified** — which is more than L1 and less than L2, and says so — instead of
    being promoted to a rung whose label it does not satisfy.
    """
    actuated = gene.actuated_joints()
    clamped = [s for s in actuated if getattr(s, "actuator_torque_nm", None)]
    level = 1 if actuated and len(clamped) >= max(1, int(0.8 * len(actuated))) else 0
    try:
        from virturoid.services.sysid.calibration import l2_requirements
        l2 = l2_requirements(gene)
    except Exception as exc:  # noqa: BLE001
        l2 = {"earned": False, "requirements": [], "blocked_by": ["the L2 check could not run"],
              "verdict": f"L2 NOT evaluated: {type(exc).__name__}: {exc}"[:160],
              "bench_identified": {"parameters": [], "n_joints": 0}}
    if l2.get("earned") and level >= 1:
        level = 2
    bench = l2.get("bench_identified") or {}
    if level == 2:
        note = ("the actuator is bench-identified against a hardware log and every L2 requirement is met.")
    elif level == 1 and bench.get("n_joints"):
        note = ("the verdict was earned with joints clamped to their datasheet torque limits (L1), and "
                f"{', '.join(bench.get('parameters') or [])} are BENCH-IDENTIFIED on "
                f"{bench['n_joints']} joint(s) from a system-identification fit. The rung stays at L1 "
                f"because: {'; '.join(l2.get('blocked_by') or [])}.")
    elif level == 1:
        note = ("the verdict was earned with joints clamped to their datasheet torque limits (L1). URDF-only "
                "ideal actuators (L0) routinely fail to walk on real hardware; L2/L3 (bench/learned actuator "
                "ID) are the next fidelity rungs.")
    else:
        note = ("no per-joint torque clamp detected (L0 ideal) — margins below are advisory only; ground the "
                "gene (grounded_physics) to size real motors and earn an L1 verdict.")
    return {"level": level, "label": ACTUATOR_LEVELS[level],
            "clamped_joints": len(clamped), "actuated_joints": len(actuated),
            "bench_identified": bench,
            "l2": {k: v for k, v in l2.items() if k != "bench_identified"},
            "note": note}


# ------------------------------------------------------------------ per-joint margins (reuse bom_certificate)
def joint_margins(gene, *, steps: int = 600, n_seeds: int = 1) -> dict:
    """Per-joint torque/thermal/speed/power margins from the verified trajectory (reuses the executable-on-BOM
    grader). Returns the 6 datasheet gates + the weakest-joint headroom — margins, not a bare pass/fail."""
    try:
        from virturoid.services.bom_certificate import certify_policy_on_bom
        cert = certify_policy_on_bom(gene, policy=None, steps=steps, n_seeds=n_seeds)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"[:160]}
    gates = {g["name"]: {"passed": g["passed"], "measured": g["measured"], "limit": g["limit"]}
            for g in cert.get("gates", [])}
    headrooms = [j.get("headroom") for j in cert.get("joints", []) if j.get("headroom") is not None]
    return {"available": True, "pass": cert.get("pass"), "n_gates_pass": cert.get("n_gates_pass"),
            "weakest_joint_headroom": round(min(headrooms), 3) if headrooms else None,
            "gates": gates, "bom": cert.get("bom"), "summary": cert.get("summary"),
            "criterion": "safety-factor canon: 1.5–1.8 predictable duty, 2.5–3 mobile/variable; RMS<continuous is "
                         "the thermal gate"}


# ------------------------------------------------------------------ frozen-policy DR sweep
def _perturb_gene(gene, *, mass_scale=1.0, motor_scale=1.0, payload_kg=0.0):
    root = gene.root()
    root_name = root.name if root else None
    segs = []
    for s in gene.segments:
        m = float(getattr(s, "mass_kg", 0.2)) * mass_scale + (payload_kg if s.name == root_name else 0.0)
        kw = {"mass_kg": m}
        if getattr(s, "actuator_torque_nm", None):
            kw["actuator_torque_nm"] = float(s.actuator_torque_nm) * motor_scale
        segs.append(replace(s, **kw))
    return replace(gene, segments=segs)


def _credible_under(gene, gait_params) -> tuple:
    """Frozen-policy credibility of a (perturbed) body: the SAME controller re-run. Returns (credible, forward)."""
    from virturoid.services.task_matched_eval import robot_kind
    kind = robot_kind(gene)
    if kind == "legged" and gait_params:
        from virturoid.services.gait_search import evaluate_gait
        r = evaluate_gait(gene, gait_params, steps=500)
        fwd = float(r.get("forward", 0.0))
        cred = bool(r.get("survived") and fwd >= 0.25 and float(r.get("height_ratio", 1.0)) >= 0.55)
        return cred, fwd
    from virturoid.services.design_bench import _fitness_raw, _is_credible, _verdict_for_gene
    res = _verdict_for_gene(gene)
    return _is_credible(res), _fitness_raw(res, kind)


def _failure_boundary(gene, gait_params, axis: str, *, lo: float, hi: float, iters: int = 6):
    """Binary-search the value along one axis where the frozen policy STOPS being credible; report the last
    passing value (the robustness margin). ``lo`` passes (nominal side), ``hi`` is the stress extreme."""
    def cred(v):
        if axis == "payload_kg":
            g = _perturb_gene(gene, payload_kg=v)
        elif axis == "mass_scale":
            g = _perturb_gene(gene, mass_scale=v)
        elif axis == "motor_scale":
            g = _perturb_gene(gene, motor_scale=v)
        else:
            return True
        return _credible_under(g, gait_params)[0]
    if not cred(lo):
        return {"axis": axis, "last_passing": None, "note": "fails even at the nominal edge"}
    if cred(hi):
        return {"axis": axis, "last_passing": round(hi, 3), "note": "still passes at the stress extreme"}
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if cred(mid):
            a = mid
        else:
            b = mid
    return {"axis": axis, "last_passing": round(a, 3), "note": "first failure between last_passing and the extreme"}


def dr_sweep(gene, gait_params=None, *, draws: int = 16, seed: int = 0, with_boundaries: bool = True) -> dict:
    """Frozen-policy Monte-Carlo DR sweep over mass/actuator/payload. Pass-rate + Clopper-Pearson bound +
    per-parameter failure boundaries. Bounded (minutes on CPU). Friction/latency are model-level → Tier-2 scope."""
    rng = random.Random(seed)
    body_mass = sum(float(getattr(s, "mass_kg", 0.2)) for s in gene.segments) or 1.0
    passes = 0
    for _ in range(draws):
        ms = rng.uniform(*DR_RANGES["mass_scale"])
        mo = rng.uniform(*DR_RANGES["motor_scale"])
        pay = rng.uniform(*DR_RANGES["payload_frac"]) * body_mass
        cred, _ = _credible_under(_perturb_gene(gene, mass_scale=ms, motor_scale=mo, payload_kg=pay), gait_params)
        passes += int(cred)
    cp = clopper_pearson(draws - passes, draws)
    boundaries = {}
    if with_boundaries:
        boundaries = {
            "max_payload_kg": _failure_boundary(gene, gait_params, "payload_kg", lo=0.0, hi=0.6 * body_mass),
            "max_mass_scale": _failure_boundary(gene, gait_params, "mass_scale", lo=1.0, hi=2.0),
            "min_motor_scale": _failure_boundary(gene, gait_params, "motor_scale", lo=1.0, hi=0.4),
        }
    return {"draws": draws, "pass_rate": round(passes / draws, 4) if draws else None,
            "clopper_pearson_failure_bound": cp, "swept_ranges": DR_RANGES,
            "failure_boundaries": boundaries,
            "scope_note": ("swept: mass, actuator strength, payload — the sim-to-real DOMINANT factors (ETH 2025). "
                           "Friction µ and control latency are model-level perturbations reserved for Tier-2; they "
                           "are NOT swept here and the certificate says so (they are not silently assumed robust).")}


# ------------------------------------------------------------------ numerics invariance
def numerics_invariance(gene, gait_params=None) -> dict:
    """Re-verdict the frozen policy at a HALVED ROLLOUT BUDGET (500 -> 250 steps) as a cheap knife-edge alarm: a
    credible verdict that only exists at one episode length is a knife-edge result. 2 rollouts.

    This is deliberately NOT what the plan's "halved timestep" line describes, and it must not be read as one —
    a shorter rollout changes how long you watch, not how finely the integrator steps, so it cannot detect a
    solver exploit. It is the probe available without reaching into ``morph_policy`` internals.

    ``probed`` is the honest half of the contract. The re-run only exists on a legged body with gait params; on
    anything else the function used to compare a value with itself and report ``invariant: True``, i.e. a free
    pass earned by never running. Now it reports ``probed: False`` and ``invariant: None``.
    """
    base_c, base_f = _credible_under(gene, gait_params)
    from virturoid.services.task_matched_eval import robot_kind
    if robot_kind(gene) == "legged" and gait_params:
        from virturoid.services.gait_search import evaluate_gait
        r = evaluate_gait(gene, gait_params, steps=250)
        alt_c = bool(r.get("survived") and float(r.get("forward", 0)) >= 0.12)
        return {"probed": True, "nominal_credible": base_c, "coarse_budget_credible": alt_c,
                "invariant": bool(base_c == alt_c),
                "probe": "the SAME frozen controller re-run at 250 steps instead of 500 (a budget probe)",
                "note": "a credible verdict that flips under a shorter evaluation budget is a knife-edge result. "
                        "This does NOT test integrator/timestep invariance — the halved-timestep and "
                        "solver-iteration sweeps are the Tier-2 numerics refinement and have not been run."}
    return {"probed": False, "nominal_credible": base_c, "coarse_budget_credible": None, "invariant": None,
            "probe": None,
            "note": "not probed: the budget re-run needs a legged body with frozen gait params. Reported as "
                    "UNMEASURED rather than invariant — a check that never ran is not a check that passed."}


# ------------------------------------------------------------------ the certificate (NASA-STD-7009 schema)
#
# THE SCOPE / HONESTY BLOCK IS DERIVED. See the module docstring for what these four fields used to be and why
# that was the worst defect in the file. Everything below that is still a constant is named `_DECLARED_*` and is
# reported under `provenance: declared`, so a reader can tell an engineering assumption from a measurement.

# Failure categories that live outside rigid-body simulation entirely. No run can measure these, so they are
# DECLARED — but they are declared, and labelled as such, rather than presented as a finding.
_DECLARED_NOT_MODELED = ["backlash", "cable routing", "thermal transients", "battery voltage sag", "wear",
                         "assembly tolerance stack-ups", "firmware jitter", "ESD",
                         "friction µ (model-level perturbation; Tier-2, never swept here)",
                         "control latency (model-level perturbation; Tier-2, never swept here)"]

# The standing caveat that is true of every simulation verdict this system can produce.
_DECLARED_LIMITS = ("Physics-verified in simulation, not on hardware. Sim is optimistic by construction "
                    "(Simulation Optimization Bias > 0); every robustness claim is w.r.t. the DECLARED "
                    "perturbation distribution only. Contact/manipulation verdicts carry a lower ceiling "
                    "than locomotion. Whole failure categories (assembly, cabling, firmware, thermal "
                    "derating, ESD) live outside physics sim.")


def _scope_block(*, sanity: dict, act: dict, credible: bool, verdict_str: str,
                 robustness: dict | None, margins: dict | None, numerics: dict | None) -> tuple:
    """Build ``scope`` (valid_iff / not_modeled / claim_ceiling) and ``honest_limits`` FROM THIS RUN.

    ``robustness`` / ``margins`` / ``numerics`` are ``None`` when the corresponding stage did not run — which is
    the case the old constants got wrong: they described a swept envelope on certificates where nothing was
    swept. Returns ``(scope, honest_limits_text, measured_limits)``.
    """
    level = int(act.get("level", 0))
    n_clamped, n_actuated = int(act.get("clamped_joints", 0)), int(act.get("actuated_joints", 0))
    ranges = robustness.get("swept_ranges") if robustness else None
    pass_rate = robustness.get("pass_rate") if robustness else None
    draws = int(robustness.get("draws", 0)) if robustness else 0
    cp = (robustness or {}).get("clopper_pearson_failure_bound") or {}
    margins_ok = bool(margins and margins.get("available"))

    # ---- valid_iff: the conditions this certificate's claim is actually conditional on, each with its evidence.
    conds: list[dict] = [{"condition": "model_sanity is green",
                          "established": bool(sanity.get("ok")),
                          "evidence": (f"{sanity.get('n_bodies')} bodies, total mass "
                                       f"{sanity.get('total_mass_kg')} kg, inertia triangle inequality "
                                       f"{'holds' if sanity.get('inertia_valid') else 'VIOLATED'}"
                                       if sanity.get("ok") else f"issues: {sanity.get('issues')}")}]
    if level >= 1:
        conds.append({"condition": "the built joints stay inside the datasheet torque limits the verdict was "
                                   "earned under",
                      "established": True,
                      "evidence": f"{ACTUATOR_LEVELS[level]}; {n_clamped}/{n_actuated} joints clamped"})
    if robustness is not None:
        conds.append({"condition": "the real robot's mass, actuator strength and payload lie inside the SWEPT "
                                   "envelope",
                      "established": True,
                      "evidence": f"swept {ranges} over {draws} Monte-Carlo draws; pass_rate {pass_rate}"})
    else:
        conds.append({"condition": "the real robot matches the NOMINAL model — no perturbation envelope was "
                                   "swept on this certificate, so there is no envelope for it to lie inside",
                      "established": False,
                      "evidence": "run_dr was False: the robustness block is absent from this certificate"})
    if margins_ok:
        conds.append({"condition": "no joint is driven past the motor the BOM sized for it",
                      "established": bool(margins.get("pass")),
                      "evidence": f"weakest joint headroom {margins.get('weakest_joint_headroom')}; "
                                  f"{margins.get('n_gates_pass')} datasheet gate(s) passed"})
    valid_iff = " AND ".join(c["condition"] for c in conds)

    # ---- not_modeled: the declared categories PLUS what this particular run left unmeasured.
    unmeasured: list[str] = []
    if robustness is None:
        unmeasured.append("mass / actuator-strength / payload variation (no DR sweep ran on this certificate)")
    if not margins_ok:
        why = (margins or {}).get("error") or "not requested"
        unmeasured.append(f"per-joint torque / thermal margins ({why})")
    if level == 0:
        unmeasured.append("the actuator's real torque-speed limit (L0 ideal: no per-joint clamp was applied, so "
                          "the verdict assumes motors that cannot saturate)")
    if numerics is not None and not numerics.get("probed"):
        unmeasured.append("sensitivity to the evaluation budget (the numerics probe does not run on this body)")
    if numerics is None:
        unmeasured.append("sensitivity to the evaluation budget (the numerics probe did not run)")

    # ---- claim_ceiling: the strongest sentence THIS certificate is entitled to.
    if not sanity.get("ok"):
        ceiling = ("this certificate is VOID — model_sanity failed, so it claims nothing at all about this "
                   "robot beyond the sanity issues listed.")
    elif not credible:
        ceiling = (f"this certificate does NOT claim the robot works. The headline verdict is {verdict_str!r} "
                   f"and every field below describes a body that did not pass it.")
    elif robustness is None:
        ceiling = ("a single NOMINAL verdict under the stated assumptions. Nothing was perturbed, so this "
                   "certificate makes NO robustness claim — not 'robust', not 'will work when built'. Re-export "
                   "with dr_sweep=true to earn one.")
    elif pass_rate is not None and pass_rate >= 1.0:
        ceiling = (f"the failure modes we can model were swept ({draws} draws over {ranges}) and ALL passed; "
                   f"the true failure rate inside that envelope is at most {cp.get('hi')} at 95% confidence. "
                   f"NOT 'will work when built' — that awaits use_history.")
    else:
        ceiling = (f"the swept envelope ALREADY CONTAINS FAILURES: {cp.get('k_fail')}/{draws} draws over "
                   f"{ranges} were not credible (pass_rate {pass_rate}, true failure rate in "
                   f"[{cp.get('lo')}, {cp.get('hi')}] at 95%). This certificate claims a nominal verdict plus a "
                   f"measured failure rate — it does not claim robustness.")

    # ---- honest_limits: the declared caveat, then the limits THIS run measured about itself.
    measured: list[str] = []
    if not credible:
        measured.append(f"the headline verdict is {verdict_str!r} — not credible")
    if robustness is None:
        measured.append("no DR sweep ran (run_dr=False): the robustness block is ABSENT, which is not the same "
                        "as passed")
    else:
        measured.append(f"the DR sweep scores each draw with `survived and forward >= 0.25 and height_ratio >= "
                        f"0.55` at 500 steps — a WEAKER bar than the classify() verdict quoted above, so "
                        f"pass_rate {pass_rate} is not a re-run of the headline verdict")
    if numerics is not None and numerics.get("probed"):
        measured.append("the numerics check re-ran at a shorter ROLLOUT (250 vs 500 steps), not a halved "
                        "timestep — it cannot detect an integrator/solver exploit")
    else:
        measured.append("numerics sensitivity is UNMEASURED on this body (the budget probe needs a legged body "
                        "with frozen gait params)")
    if level == 0:
        measured.append("actuator fidelity is L0 (ideal, unclamped) — margins below are advisory only")
    if not margins_ok:
        measured.append("per-joint margins were not computed, so no torque/thermal claim is made")
    measured.append("use_history is N=0 and has no writer in this build, so predictivity_srcc reads null and "
                    "will keep reading null until real-build outcomes are recorded")

    honest_limits = _DECLARED_LIMITS + " MEASURED ON THIS RUN: " + "; ".join(measured) + "."
    scope = {
        "valid_iff": valid_iff,
        "valid_iff_conditions": conds,
        "not_modeled": _DECLARED_NOT_MODELED + unmeasured,
        "not_modeled_provenance": {
            "declared": _DECLARED_NOT_MODELED,
            "unmeasured_on_this_run": unmeasured,
            "note": "'declared' entries are failure categories outside rigid-body sim and are the same on every "
                    "certificate; 'unmeasured_on_this_run' is derived from which stages actually executed here",
        },
        "claim_ceiling": ceiling,
    }
    return scope, honest_limits, measured


def build_certificate_v2(gene, verdict: dict, *, gait_params=None, task: str = "", robot_id: str | None = None,
                         memory_dir: str = "build/memory", run_dr: bool = True, dr_draws: int = 12,
                         run_margins: bool = True, body_parity: dict | None = None) -> dict:
    """Assemble the tiered NASA-STD-7009-mapped certificate. Always computes the cheap Tier-1 (model_sanity,
    actuator level); the DR sweep + margins (bounded rollouts) are on by default but can be disabled for a fast
    export. VOID (``valid: False``) unless model_sanity is green — every downstream claim depends on it."""
    from virturoid.services.verdict_certificate import build_certificate
    base = build_certificate(gene, verdict, task=task, robot_id=robot_id, memory_dir=memory_dir,
                             body_parity=body_parity)
    sanity = model_sanity(gene)
    act = actuator_fidelity_level(gene)
    # RUN THE EVIDENCE FIRST, then describe it. The scope/honesty block is derived from which of these actually
    # executed and what they returned, so it cannot claim a swept envelope on a certificate that swept nothing.
    margins = joint_margins(gene) if (sanity["ok"] and run_margins) else None
    robustness = dr_sweep(gene, gait_params, draws=dr_draws) if (sanity["ok"] and run_dr) else None
    numerics = numerics_invariance(gene, gait_params) if (sanity["ok"] and run_dr) else None
    scope, honest_limits, measured_limits = _scope_block(
        sanity=sanity, act=act, credible=bool(base.get("credible")),
        verdict_str=str(base.get("verdict", "unverified")),
        robustness=robustness, margins=margins, numerics=numerics)
    cert = {
        "artifact": "virturoid_verification_certificate", "version": 2, "tier": 1,
        "valid": bool(sanity["ok"]),
        "identity": {"robot_id": robot_id, "species": getattr(gene, "species", None),
                     "robot_class": getattr(gene, "robot_class", None), "kind": base.get("kind"),
                     "n_segments": len(gene.segments), "dof": len(gene.actuated_joints())},
        "verdict": {k: base.get(k) for k in ("verdict", "credible", "checks", "gait_source", "verified_with",
                                             "deploy_is_measure", "body_parity")},
        "model_sanity": sanity,
        "actuator_fidelity_level": act,
        "flywheel_provenance": base.get("flywheel_provenance"),
        "scope": scope,
        "use_history": {"builds_attempted": 0, "predictivity_srcc": None,
                        "measured": False,
                        "note": "predictivity is unmeasured at N=0 real builds. Stated plainly: NO writer for "
                                "these two fields exists in this build, so they are a declared placeholder, not "
                                "a count of builds that happened to be zero. Each real build would close the "
                                "loop and the running rank-correlation would become this certificate's own "
                                "credibility flywheel."},
        "honest_limits": honest_limits,
        "honest_limits_measured": measured_limits,
        "honest_limits_declared": _DECLARED_LIMITS,
    }
    if not sanity["ok"]:
        cert["voided_reason"] = f"model_sanity failed: {sanity['issues']}"
        return cert
    if run_margins:
        cert["margins"] = margins
    if run_dr:
        cert["robustness"] = robustness
        cert["numerics"] = numerics
    return cert
