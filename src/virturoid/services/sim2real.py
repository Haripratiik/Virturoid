"""Sim2real reality-gap quantification (plan v3 M5) — a MEASURED transfer number without hardware.

Everything upstream is sim; contact-DR gives robustness but not a *number*. This module measures the reality
gap the honest way available in pure sim: run a policy on NOMINAL dynamics, then on a HELD-OUT domain-
randomization distribution deliberately chosen WIDER than the training DR (so it probes dynamics the policy was
NOT trained on), and report:

  * ``transfer_survival``  — fraction of held-out-DR trials the policy still WALKS (survives + steps + moves) —
    the MMRV-style robustness number (Muratore et al.'s "how well does a sim policy transfer" quantity, in sim).
  * ``forward_retention``  — held-out-DR mean forward / nominal forward (how much travel survives the gap).
  * ``reality_gap_m``      — nominal forward − held-out-DR mean forward (metres of travel lost to unmodeled dynamics).

This is the CPU dual-sim transfer report; hardware HIL is a separate, costed track (scripts/hil_plan.md). The
policy's banked deploy config (decimation / action-LPF / sphere-feet, meta[6..8]) is auto-adopted by
recipe_rollout_morph so nominal and held-out are both measured at deploy==train."""

from __future__ import annotations

# Held-out DR distribution: WIDER than the training DR (recipe_robustness defaults gain .15/mass .1/damping .2/
# friction .3) so it probes UNMODELED dynamics -- the honest reality-gap probe, not a re-test of the train dist.
HELD_OUT_DR = {"gain": 0.30, "mass": 0.20, "damping": 0.40, "friction": 0.50}


def sim2real_transfer_report(gene, policy=None, *, n: int = 12, steps: int = 900, seed: int = 0,
                             held_out: dict | None = None) -> dict:
    """Measure the sim2real reality gap for ``(gene, policy)``: nominal vs a held-out (wider) DR distribution.
    Returns nominal metrics, held-out-DR aggregates, and the three gap numbers. Deterministic for a fixed seed."""
    from virturoid.services.morph_policy import recipe_robustness, recipe_rollout_morph
    d = dict(HELD_OUT_DR)
    d.update(held_out or {})
    nom = recipe_rollout_morph(gene, policy, steps=steps, seed=seed)   # deploy config auto-adopted from policy meta
    nom_fwd = float(nom.get("forward", 0.0) or 0.0)
    dr = recipe_robustness(gene, policy, n=n, steps=steps, seed=seed,
                           gain=d["gain"], mass=d["mass"], damping=d["damping"], friction=d["friction"])
    dr_fwd = float(dr.get("mean_forward", 0.0) or 0.0)
    retention = (dr_fwd / nom_fwd) if abs(nom_fwd) > 1e-6 else 0.0
    return {
        "nominal": {"forward": round(nom_fwd, 3), "cadence": nom.get("cadence"),
                    "upright": nom.get("upright_frac"), "survived": bool(nom.get("survived"))},
        "held_out_dr": {"distribution": d, "n": dr["n"], "transfer_survival": dr["survival_rate"],
                        "mean_forward": dr["mean_forward"], "min_forward": dr["min_forward"],
                        "mean_cadence": dr["mean_cadence"]},
        "transfer_survival": dr["survival_rate"],                     # MMRV-style robustness (in sim)
        "forward_retention": round(max(0.0, retention), 3),          # fraction of nominal travel kept under the gap
        "reality_gap_m": round(nom_fwd - dr_fwd, 3),                 # metres of forward lost to unmodeled dynamics
        "note": "pure-sim reality-gap probe (held-out DR wider than training); hardware HIL is scripts/hil_plan.md",
    }
