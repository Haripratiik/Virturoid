"""Diagnosis Artifact (breakthrough plan H1) — the structured, prompt-ready read of an evaluation that an LLM
design operator reasons over. The single highest-ROI component: Eureka's ablation showed removing this kind of
"reward reflection" costs -28.6% (docs/breakthrough_research_plan.md §3).

The rule (AIDE's summarization operator): the LLM never sees raw logs — it sees a *summary* of what happened,
with gate-by-gate pass/fail INCLUDING MARGINS (how far from each threshold), an actionable FAILURE MODE, and a
short list of next-actions. This module turns a rollout/eval result (recipe_rollout_morph or gait_diagnostics
for locomotion; a manipulation eval for grasp/pick) into that artifact.

Design notes grounded in live evidence:
* Direction awareness — a gait with real cadence that travels BACKWARD (the bilateral hexapod: cadence 10,
  forward -0.43) is a *different* failure from a shuffle, and gait_diagnostics buries it as "barely moves
  forward (-0.43)". The taxonomy names ``walks_backward`` explicitly and points at the G4 MJX<->CPU velocity-
  frame parity gap — so the Diagnostician operator (and a human) can act on it instead of mis-reading it.
* Task-general — locomotion and manipulation have distinct failure taxonomies (miss/slip/unreachable/drop),
  dispatched by task family, so the artifact serves any node in the design-search harness.

Pure-Python, dependency-free, deterministic; consumes a plain result dict (no MuJoCo), so it unit-tests fast.
"""

from __future__ import annotations

# Default honesty gates (the anti-Goodhart set: forward + real stepping + upright + survived). Task specs may
# override any threshold; ``survived`` is a hard bool gate.
LOCOMOTION_GATES = {"forward_m": 0.30, "cadence": 3.0, "upright": 0.60}
MANIPULATION_GATES = {"success_rate": 0.60}


def _num(*vals, default=0.0):
    """First non-None numeric among vals (lets one artifact read recipe_rollout_morph OR gait_diagnostics keys)."""
    for v in vals:
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float(default)


def _gate_row(name, value, threshold, *, hard_bool=None):
    """One gate line: value vs threshold with the signed margin (value - threshold) and pass flag."""
    if hard_bool is not None:
        return {"gate": name, "value": bool(hard_bool), "pass": bool(hard_bool)}
    margin = round(value - threshold, 4)
    return {"gate": name, "value": round(value, 4), "threshold": threshold,
            "margin": margin, "pass": bool(value >= threshold)}


def _locomotion_mode(fwd, cadence, upright, survived, *, late_speed=None, alternation=None, gates=LOCOMOTION_GATES):
    """Actionable locomotion failure mode from the metrics. Order matters: fatal modes first."""
    if not survived:
        return "fell", "the body falls/collapses before the episode ends — not a gait."
    if upright < gates["upright"]:
        return "leaning", f"trunk not upright (upright {upright:.2f} < {gates['upright']}) — unstable posture."
    # a REAL stepping rhythm pointed the wrong way — the hexapod case; distinct from a shuffle.
    if cadence >= gates["cadence"] and fwd < -0.05:
        return ("walks_backward",
                f"real gait (cadence {cadence:.1f}/s, upright) but travels BACKWARD (forward {fwd:+.2f} m). "
                "The gait works; its DIRECTION is reversed — flip the CPG calf_phase / the reward velocity-frame "
                "sign, and check the MJX<->CPU velocity-axis convention (parity gap G4). Do NOT redesign the body.")
    if cadence < gates["cadence"]:
        return ("shuffle",
                f"not stepping (cadence {cadence:.1f}/s < {gates['cadence']}) — dragging/sliding, forward "
                f"{fwd:+.2f} m. Needs a stepping rhythm (CPG prior / feet-air-time reward), not more forward reward.")
    if late_speed is not None and late_speed < 0.05 and fwd > 0.1:
        return ("lunge_stall",
                f"lunges then stalls (late speed {late_speed:.2f} m/s) — one-time push, not sustained locomotion.")
    if fwd < gates["forward_m"]:
        return ("weak_forward",
                f"steps upright but slow (forward {fwd:+.2f} m < {gates['forward_m']}). Refine the propulsion "
                "reward / gains; the gait is close.")
    return "walking", f"WALKING: cadence {cadence:.1f}/s, upright {upright:.2f}, forward {fwd:+.2f} m, survived."


def _manipulation_mode(sr, *, contacted=None, lifted=None, reached=None, dropped=None, gates=MANIPULATION_GATES):
    """Actionable manipulation failure mode (miss / unreachable / slip / drop / collision)."""
    if sr >= gates["success_rate"]:
        return "grasped", f"grasps reliably (success {sr:.0%} >= {gates['success_rate']:.0%})."
    if reached is False:
        return "unreachable", "target outside the reachable workspace — lengthen reach / reposition the scene."
    if contacted is False:
        return "miss", "gripper never contacts the object — approach/IK aim is off, not a grasp-force problem."
    if dropped:
        return "drop_in_transit", "grasps then drops mid-transport — grip force / contact-gating too weak to hold."
    if lifted is False:
        return "slip", "contacts but cannot lift — friction/closure insufficient; increase grip force or contact area."
    return "low_success", f"grasps sometimes (success {sr:.0%} < {gates['success_rate']:.0%}) — tune contact/approach."


def build_diagnosis_artifact(result: dict, *, task_type: str = "locomotion", gates: dict | None = None,
                             history: list | None = None) -> dict:
    """Turn an eval ``result`` into the structured diagnosis artifact.

    Returns ``{verdict, failure_mode, explanation, gate_report, metrics, trend, next_actions, summary_text}``.
    ``result`` may be a ``recipe_rollout_morph`` dict, a ``gait_diagnostics.diagnose_gait`` dict, or a
    manipulation eval dict (keys read defensively). ``history`` = prior artifacts' ``metrics`` for trend.
    """
    fam = "manipulation" if task_type in ("grasp", "grasp_lift", "pick_place", "pick_place_sort", "stack",
                                          "shelf", "push", "transport", "manipulation") else "locomotion"
    g = dict(MANIPULATION_GATES if fam == "manipulation" else LOCOMOTION_GATES)
    g.update(gates or {})

    if fam == "locomotion":
        fwd = _num(result.get("forward"), result.get("forward_m"))
        cadence = _num(result.get("cadence"), result.get("cadence_steps_per_s"))
        upright = _num(result.get("upright_frac"), result.get("upright_mean"), default=1.0)
        survived = bool(result.get("survived", not result.get("fell", False)))
        late_speed = result.get("late_speed")
        alternation = result.get("alternation")
        mode, explanation = _locomotion_mode(fwd, cadence, upright, survived, late_speed=late_speed,
                                             alternation=alternation, gates=g)
        gate_report = [_gate_row("forward_m", fwd, g["forward_m"]), _gate_row("cadence", cadence, g["cadence"]),
                       _gate_row("upright", upright, g["upright"]), _gate_row("survived", None, None, hard_bool=survived)]
        metrics = {"forward_m": round(fwd, 3), "cadence": round(cadence, 2), "upright": round(upright, 3),
                   "survived": survived}
        # TRIPOD-SUPPORT companion gate (WS3): a task may add ``support`` so lowering the upright bar for a
        # many-legged body moves the bar SIDEWAYS (a low tripod must still show real stepping support), never DOWN.
        # Absent from a result -> default 1.0 (older/quad rollouts pass vacuously; only tasks that opt in are gated).
        if "support" in g:
            support = _num(result.get("support_frac"), default=1.0)
            gate_report.append(_gate_row("support", support, g["support"]))
            metrics["support"] = round(support, 3)
        # COMMAND-TRACKING gate (WS8): a task may add ``track`` (mean exp(-err^2/sigma) of forward-speed tracking).
        # A constant gait tracks a varied command poorly -> low track_score -> fails; absent -> 1.0 (untracked tasks).
        if "track" in g:
            track = _num(result.get("track_score"), default=1.0)
            gate_report.append(_gate_row("track", track, g["track"]))
            metrics["track"] = round(track, 3)
            if track < g["track"]:                            # name it: real gait, but won't FOLLOW the command
                mode, explanation = ("poor_tracking",
                    f"does not track the commanded speed (track {track:.2f} < {g['track']}) — a fixed gait "
                    "can't slow/stop/reverse; needs a command-conditioned policy (WS8), not more forward reward.")
        if late_speed is not None:
            metrics["late_speed"] = round(_num(late_speed), 3)
    else:
        sr = _num(result.get("success_rate"), result.get("value"), result.get("metric"))
        mode, explanation = _manipulation_mode(
            sr, contacted=result.get("contacted"), lifted=result.get("lifted"),
            reached=result.get("reached"), dropped=result.get("dropped"), gates=g)
        gate_report = [_gate_row("success_rate", sr, g["success_rate"])]
        metrics = {"success_rate": round(sr, 3)}

    verdict = "pass" if all(row["pass"] for row in gate_report) else "fail"
    trend = _trend(metrics, history, fam)
    next_actions = _next_actions(mode, fam)
    summary_text = _summarize(verdict, mode, explanation, gate_report, trend)
    return {"verdict": verdict, "failure_mode": mode, "explanation": explanation, "gate_report": gate_report,
            "metrics": metrics, "trend": trend, "next_actions": next_actions, "summary_text": summary_text}


def fitness_from_artifact(artifact: dict) -> float:
    """The search's SELECTION signal (H2): a scalar from the gate margins, so the harness can rank + climb even
    BEFORE any candidate passes. Passing candidates score in [1, 2) (always above failing); failing candidates
    score in [0, 1) = mean per-gate progress toward threshold (a backward gait, value/threshold < 0, clamps to
    0 — correctly worst). Computed from the artifact, never from a raw number the LLM could game (Eureka's
    fitness-separate-from-reward discipline)."""
    parts = []
    for r in artifact.get("gate_report", []):
        if "threshold" in r and r.get("threshold"):
            parts.append(max(0.0, min(1.0, r["value"] / r["threshold"])))
        else:
            parts.append(1.0 if r.get("pass") else 0.0)
    base = (sum(parts) / len(parts)) if parts else 0.0
    return round((1.0 + base) if artifact.get("verdict") == "pass" else base, 4)


def _trend(metrics, history, fam):
    """Direction of the headline metric vs the previous artifact (so the Diagnostician sees if it's improving)."""
    if not history:
        return None
    key = "success_rate" if fam == "manipulation" else "forward_m"
    prev = history[-1].get(key) if isinstance(history[-1], dict) else None
    cur = metrics.get(key)
    if prev is None or cur is None:
        return None
    d = round(cur - prev, 4)
    return {"metric": key, "delta_vs_prev": d, "direction": "up" if d > 1e-6 else "down" if d < -1e-6 else "flat"}


def _next_actions(mode, fam):
    """A short, mode-specific menu of what a design operator should CONSIDER next (not a decision — a menu)."""
    return {
        "walking": ["accept + bank", "optionally push distance with a propulsion-reward tweak"],
        "walks_backward": ["flip CPG calf_phase / reward velocity sign (keep the body)", "run the MJX<->CPU parity probe (G4)"],
        "shuffle": ["enable/strengthen the CPG stepping prior", "add a feet-air-time reward term", "raise foot-clearance weight"],
        "leaning": ["stiffen posture gains / add an upright reward term", "check spawn height (standing_spawn_z)"],
        "lunge_stall": ["penalize energy/impulse spikes", "add a sustained-velocity (late-speed) reward term"],
        "weak_forward": ["increase propulsion-reward weight", "raise per-joint torque/gains via adaptive gains"],
        "poor_tracking": ["train a command-conditioned policy (mjx_morph_vel: v_x/w_z in obs + tracking reward)",
                          "verify at the frozen command schedule; a fixed gait cannot pass"],
        "fell": ["reduce control authority / add alive bonus", "check gains + spawn penetration", "simplify the body"],
        "grasped": ["accept + bank"],
        "miss": ["fix top-down IK aim / approach pose", "verify the object is in the reachable region"],
        "unreachable": ["lengthen reach (amend_gene)", "reposition the scene into the workspace"],
        "slip": ["increase grip force / contact area", "add contact-gating to the grasp residual"],
        "drop_in_transit": ["hold grip force through transport", "slow the transport trajectory"],
        "low_success": ["tune approach + contact", "train a learned grasp residual"],
    }.get(mode, ["diagnose further"])


def _summarize(verdict, mode, explanation, gate_report, trend):
    """The compact, prompt-ready block (AIDE Σ-operator style) — margins + mode + trend, never raw logs."""
    lines = [f"VERDICT: {verdict.upper()} — failure_mode={mode}", explanation]
    grs = []
    for r in gate_report:
        if "threshold" in r:
            mark = "OK" if r["pass"] else "MISS"
            grs.append(f"{r['gate']} {r['value']} vs {r['threshold']} ({r['margin']:+}) {mark}")
        else:
            grs.append(f"{r['gate']}={'OK' if r['pass'] else 'NO'}")
    lines.append("Gates: " + " | ".join(grs))
    if trend:
        lines.append(f"Trend: {trend['metric']} {trend['direction']} ({trend['delta_vs_prev']:+}) vs prev attempt.")
    return "\n".join(lines)
