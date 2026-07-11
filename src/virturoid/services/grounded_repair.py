"""Grounded iterative repair with error-TYPED feedback (master_plan_v6 §8.2.1 / WS-B.1).

Olausson (ICLR 2024): self-repair only beats plain resampling when the feedback is *external ground truth* — and
ours is exactly that (schema issues, compile errors, and the un-gameable physics verdict with its diagnostics).
The research also measured the routing that matters: **schema/compile errors repair well (~66–77%); semantic
verdict failures repair poorly (~45%) — so resample those instead of patching them.** This module encodes that:

  * ``classify`` reads the funnel's ``error_class`` (schema | compile | spawn | verdict);
  * ``diagnostics`` turns each failure into TYPED, physics-grounded feedback (the schema issue; the compile error;
    for a verdict failure, roll/pitch/support/cadence with a concrete fix hint — "high roll → widen stance / lower
    the CoM", not "try again");
  * ``ROUTING`` sends structural failures to *repair* and semantic failures to *resample*;
  * ``repair_loop`` drives an injected ``propose`` callable (the LLM designer in production; a stub in tests) for
    at most **2 rounds** and reports ``repair_iters`` — the number Design-Bench watches fall.

The determinism lives only here, in the grounding feedback. The design itself is always authored by ``propose``.
"""
from __future__ import annotations

MAX_ROUNDS = 2   # evidence: 2 rounds capture 76–95% of achievable repair gains (§8.1.6)

# route each error class to an action — the load-bearing §8.2.1 finding
ROUTING = {"schema": "repair", "compile": "repair", "spawn": "repair", "verdict": "resample", None: "done"}


def classify(row: dict) -> str | None:
    """The first failing funnel stage (from ``design_bench.evaluate_design``), or None if the design is credible."""
    return None if row.get("credible") else row.get("error_class")


def action_for(error_class: str | None) -> str:
    return ROUTING.get(error_class, "resample")


def diagnostics(gene, row: dict) -> dict:
    """Typed, physics-grounded repair feedback for a failed design. Never a design template — it reports what the
    grounding layer measured + a concrete, physics-justified fix direction the model can choose to apply."""
    ec = row.get("error_class")
    if ec == "schema":
        return {"error_class": "schema", "issue": row.get("schema_issue"),
                "fix": "make the kinematic tree valid: one root 'body' with no parent, every parent name present, "
                       "positive length/radius/mass, a known joint_type on each actuated part"}
    if ec == "compile":
        return {"error_class": "compile", "error": row.get("compile_error"),
                "fix": "the graph did not compile to a finite-mass MuJoCo model — check part sizes are in-band and "
                       "the geometry programs are well-formed (see get_design_schema.geometry_families)"}
    if ec == "spawn":
        return {"error_class": "spawn", "fix": "the body would not spawn stable — lower the CoM / widen the base"}
    # verdict (semantic) failure: surface the un-gameable diagnostics + a fix DIRECTION keyed to the failure mode
    v = str(row.get("verdict", ""))
    diag = {"error_class": "verdict", "verdict": v, "kind": row.get("kind"),
            "fitness_raw": row.get("fitness_raw")}
    # pull whatever the honesty engine measured (present on legged/mobile verdicts)
    res = row.get("verdict_detail") or {}
    for k in ("roll_max_deg", "pitch_max_deg", "support_frac", "cadence", "height_ratio",
              "wheel_ground_contact_frac", "wheel_spin_radps"):
        if k in res:
            diag[k] = res[k]
    diag["fix"] = _verdict_fix_hint(v, res)
    return diag


def _verdict_fix_hint(verdict: str, res: dict) -> str:
    """A physics-justified fix DIRECTION for a semantic verdict failure (the grounding note the LLM reasons over).
    Note: the routing still says *resample* — semantic failures repair poorly — but the hint conditions the fresh
    sample so the model doesn't repeat the same failure mode."""
    v = verdict.upper()
    if "LURCH" in v or float(res.get("roll_max_deg", 0) or 0) > 25 or float(res.get("pitch_max_deg", 0) or 0) > 25:
        return "unstable (rears/rocks): widen the stance (aim 'down_out'), lower the CoM (shorter/heavier torso), " \
               "and use 4 leg segments so a foot can plant"
    if "SLIDE" in v or float(res.get("support_frac", 1) or 1) < 0.3:
        return "feet don't plant (slide): give legs joint='revolute', segments>=4, and enough length to lift and " \
               "step (length/diameter >= ~2.5)"
    if "SHORT" in v or "FORWARD BUT" in v:
        return "moves but too little: increase leg length / stride reach; a credible walk needs real forward travel"
    if "STANDS" in v or "BIPED" in v:
        return "a dynamically-walking biped is a learned-control frontier — add a second leg pair (quadruped) for a " \
               "scripted-gait credible walk, or train a policy (train_held)"
    if "TIPPED" in v:
        return "tips while driving: widen the wheel base and lower the deck; keep wheel radius < chassis half-width"
    if "STUCK" in v or "SPINS IN PLACE" in v:
        return "no travel: ensure wheels contact the ground (aim the axle lateral, size the wheel to reach the floor)"
    return "did not earn a credible verdict — reconsider the limb count/proportions for this body's task"


def repair_context(gene, row: dict) -> dict:
    """The full grounding packet fed back for one failed attempt: the typed diagnostics + the routing action."""
    ec = classify(row)
    return {"error_class": ec, "action": action_for(ec), "diagnostics": diagnostics(gene, row)}


def repair_loop(prompt: str, propose, *, evaluate=None, constraints: dict | None = None,
                max_rounds: int = MAX_ROUNDS, verify: bool = True) -> dict:
    """Drive at most ``max_rounds`` grounded repairs/resamples of a design.

    ``propose(prompt, *, feedback, resample)`` authors (or repairs) a gene — the LLM in production, a stub in
    tests. ``evaluate`` defaults to ``design_bench.evaluate_design``. Returns the final credible gene (or the last
    attempt), ``repair_iters`` (rounds of feedback used), and the per-round history. Physics is the only judge.
    """
    from virturoid.services.design_bench import evaluate_design
    ev = evaluate or (lambda g: evaluate_design(g, constraints=constraints, verify=verify))
    history: list[dict] = []
    gene = propose(prompt, feedback=None, resample=False)
    row = ev(gene)
    history.append({"round": 0, "credible": bool(row.get("credible")), "error_class": row.get("error_class")})
    iters = 0
    while not row.get("credible") and iters < max_rounds:
        ctx = repair_context(gene, row)
        resample = ctx["action"] == "resample"
        gene = propose(prompt, feedback=ctx, resample=resample)
        row = ev(gene)
        iters += 1
        history.append({"round": iters, "credible": bool(row.get("credible")),
                        "error_class": row.get("error_class"), "action": ctx["action"]})
    return {"credible": bool(row.get("credible")), "gene": gene, "repair_iters": iters,
            "final_error_class": row.get("error_class"), "history": history}
