"""Adopt a user's OWN control script / policy and let the sim UTILISE it, then IMPROVE it.

A customer who already has a controller (e.g. our exported CPG gait_controller.py + policy_params.json, or their
own parametric gait) should be able to drop it in and have Virturoid (1) RUN it on the body in real physics to see
how it does, and (2) make it BETTER -- warm-starting the gait search from the user's params so the sim exploits
their controller instead of cold-starting, then returning the improved params + a measured before/after.

This is the honest version of "our sim can use AND improve your policy": the imported controller is run in MuJoCo
(not trusted blindly), and the improvement is a real gait-search that must beat the imported gait on an un-gameable
credible-walk fitness -- if it can't, we say so and keep the user's controller.
"""
from __future__ import annotations

from virturoid.services.gait_search import _HI, _LO, PARAM_NAMES


def map_control_params(params: dict) -> dict:
    """Map an imported control program's params into the gait-search space (freq/hip_amp/knee_amp/kp/kd).
    Handles our exported CPG format (frequency_hz + a per-joint amplitude list/dict) and a freeform dict that
    already names gait-search keys. Anything missing falls back to a sensible mid gait so the search still runs.

    An imported ``duty`` is deliberately NOT carried across: the crawl controller derives its swing fraction
    structurally, so adopting the customer's number would only look like it was honored (task #265)."""
    out = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 120.0, "kd": 6.0}
    params = params or {}
    # explicit gait-search keys win outright
    for k in PARAM_NAMES:
        v = params.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    # our exported CPG format: frequency + per-joint amplitudes (hip is the smaller nonzero, knee the larger)
    freq = params.get("frequency_hz") or params.get("control_frequency_hz")
    if isinstance(freq, (int, float)) and freq > 0 and "freq" not in {k for k in PARAM_NAMES if k in params}:
        out["freq"] = float(freq)
    amp = params.get("amplitude")
    amps = list(amp.values()) if isinstance(amp, dict) else (amp if isinstance(amp, list) else [])
    nonzero = sorted(abs(float(a)) for a in amps if isinstance(a, (int, float)) and abs(float(a)) > 1e-6)
    if nonzero:
        out["hip_amp"] = float(nonzero[0])                      # the smaller cyclic amplitude = hip swing
        out["knee_amp"] = float(nonzero[-1])                    # the larger = knee flexion
    # clamp into the search bounds so an out-of-range imported value can't break the rollout
    return {k: float(min(max(out[k], _LO[k]), _HI[k])) for k in PARAM_NAMES}


def adopt_control_script(gene, params: dict, *, steps: int = 800, generations: int = 6, pop: int = 16,
                         seed: int = 0, workers: int = 1) -> dict:
    """UTILISE the imported controller on ``gene`` in real physics, then IMPROVE it (gait search warm-started from
    the imported params). Returns the measured before/after + the imported and improved params + whether the
    improvement is a CREDIBLE walk that actually beats the imported gait (else we keep the user's controller)."""
    from virturoid.services.gait_search import evaluate_gait, search_gait

    imported = map_control_params(params)
    used = evaluate_gait(gene, imported, steps=steps)           # run the USER's controller in our sim
    res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed,
                      workers=workers, warm_start=imported)
    improved_params = res.best_params
    improved = evaluate_gait(gene, improved_params, steps=steps)

    used_fwd, imp_fwd = float(used.get("forward", 0.0)), float(improved.get("forward", 0.0))
    # only claim an improvement if the tuned gait is a CREDIBLE walk AND travels further than the imported one
    beat = bool(improved.get("credible")) and imp_fwd > max(used_fwd, 0.0) + 1e-3
    return {
        "utilised": {"forward_m": round(used_fwd, 3), "credible": bool(used.get("credible")),
                     "height_ratio": round(float(used.get("height_ratio", 0.0)), 3),
                     "cadence": round(float(used.get("cadence", 0.0)), 2)},
        "improved": {"forward_m": round(imp_fwd, 3), "credible": bool(improved.get("credible")),
                     "height_ratio": round(float(improved.get("height_ratio", 0.0)), 3),
                     "cadence": round(float(improved.get("cadence", 0.0)), 2)},
        "imported_params": imported,
        "improved_params": {k: round(float(v), 3) for k, v in improved_params.items()},
        "beat_imported": beat,
        "forward_gain_x": (round(imp_fwd / used_fwd, 2) if used_fwd > 0.05 else None),
        "verdict": ("improved the user's controller (credible walk, further travel)" if beat
                    else "ran the user's controller; kept it (tuning did not credibly beat it this budget)"),
    }
