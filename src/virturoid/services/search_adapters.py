"""Search adapters — connect the design-search harness to REAL physics + an LLM-free proposer (night-shift N1
core). These turn the injectable ``propose``/``evaluate`` slots of design_search.run_design_search into concrete
CPU-MuJoCo operations, so the harness runs on real bodies with no LLM and no GPU.

The first adapter is the CPG-direction search — the concrete answer to tonight's live problem: instead of a
hand-run sweep to find the ``calf_phase`` that makes a body walk forward, the harness searches CPG gait params,
evaluates each with ``recipe_rollout_morph`` (pure CPG, zero residual, so it isolates the gait prior), and the
honesty gate selects the forward one. This is the lowest rung of the fidelity ladder for locomotion (seconds on
CPU) — the harness would then route the winner up to a GPU training burst.
"""

from __future__ import annotations

_CPG_KEYS = ("calf_phase", "freq", "thigh_amp", "calf_amp", "residual_scale")


def make_cpg_evaluate(gene, *, steps: int = 600, seed: int = 0, base_cpg: dict | None = None):
    """Return an ``evaluate(spec) -> result`` that applies a spec's CPG edit to ``gene`` and runs a pure-CPG
    (zero-residual) recipe rollout. The zero policy + compiled graph are built ONCE and reused across the search,
    so each eval is just a rollout (~1-2 s CPU). ``spec`` = ``{edit_kind:'cpg', params:{calf_phase,freq,...}}``.
    """
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import (CPG_DEFAULT, MorphPolicy, compiled_model,
                                                 recipe_rollout_morph, robot_mjcf)
    graph = encode_robot(compiled_model(robot_mjcf(gene)))
    zero = MorphPolicy(graph.feature_dim)
    zero.set_params(np.zeros(zero.n_params))                  # pure CPG: isolate the gait prior from any residual
    base = dict(base_cpg or CPG_DEFAULT)

    def evaluate(spec):
        cpg = dict(base)
        for k in _CPG_KEYS:
            if k in (spec.get("params") or {}):
                cpg[k] = float(spec["params"][k])
        r = recipe_rollout_morph(gene, zero, steps=steps, cpg=cpg, seed=seed)
        return {"forward": float(r.get("forward", 0.0)), "cadence": float(r.get("cadence", 0.0)),
                "upright_frac": r.get("upright_frac"), "survived": bool(r.get("survived")), "cpg": cpg}

    return evaluate


def cpg_grid_proposer(grid: list[dict] | None = None):
    """An LLM-FREE proposer (N1 core): sweep a coarse grid of CPG gait params, then stop. The harness selects the
    best by the honesty gate — so the forward-walking direction is found by search, not by hand. Default grid
    spans the gait-phase directions (calf_phase) × two frequencies that resolved the quad/hexapod cases."""
    if grid is None:
        grid = [{"calf_phase": cp, "freq": f}
                for cp in (0.0, 1.5708, 3.1416, -1.5708) for f in (1.5, 2.5)]
    state = {"i": 0}

    def propose(parent, history):
        if state["i"] >= len(grid):
            return None                                       # grid exhausted -> harness stops (proposer_exhausted)
        params = dict(grid[state["i"]])
        state["i"] += 1
        return {"edit_kind": "cpg", "params": params, "rationale": "grid sweep of CPG gait direction/frequency"}

    return propose
