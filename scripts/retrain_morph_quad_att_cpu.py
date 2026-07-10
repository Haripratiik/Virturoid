"""Regenerate the banked transferable MorphPolicy (models/morph_quad_att.npz) ON CPU via OpenAI-ES.

WHY: the anatomy differentiation rework changed body geometry, so the GPU-trained banked policy went stale —
it collapses the reworked steerable bodies (height_ratio ~0.07) and every topology scores -inf
(tests/test_topology_codesign.py). The MorphPolicy is small and ES-trainable on CPU (morph_trainer.train_morph_es),
and forward_score optimizes the SAME rollout_morph the topology search scores with (height-gated gait reward), so
a short CPU run recovers a non-collapsing shared policy without the GPU. Keep-best: only overwrites the banked npz
if the new policy makes >=1 topology viable (>-inf); the stale file is backed up first.

    PYTHONPATH=src python scripts/retrain_morph_quad_att_cpu.py [--gens 30 --pop 24 --steps 280 --workers 6]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="models/morph_quad_att.npz")
    args = ap.parse_args(argv)

    import numpy as np

    from virturoid.services.morph_policy import MorphPolicy, rollout_morph
    from virturoid.services.morph_trainer import train_morph_es
    from virturoid.services.steerable_body import steerable_quadruped
    from virturoid.services.topology_codesign import search_leg_count

    # Train ACROSS the test's body distribution (leg counts 3..6) so the ONE shared policy generalizes zero-shot.
    genes = [steerable_quadruped(species=f"evo_{n}leg", n_legs=n) for n in (4, 3, 5, 6)]
    fd = int(rollout_morph(genes[0], None, steps=1)["feature_dim"])

    # Fitness that TARGETS THE TEST'S GATE: score_topology rejects a body below height_ratio 0.3 (-inf), and the
    # default forward_score (a height-WEIGHTED gait reward) tolerates a crouch below it. So optimize height_ratio
    # under the SAME rollout_morph the test deploys through, + a little forward, so the policy learns to keep the
    # body UP (crossing the 0.3 gate) rather than fast-crouch-scooting. Non-finite blowups are strongly punished.
    def _height_fitness(policy, gs) -> float:
        vals = []
        for g in gs:
            r = rollout_morph(g, policy, steps=args.steps)
            if not r.get("finite", False):
                vals.append(-1.0)
                continue
            hr = float(r.get("height_ratio", 0.0))
            fwd = float(r.get("forward", 0.0))
            vals.append(min(hr, 0.6) + 0.25 * fwd)     # reward standing tall (saturating) + some forward travel
        return float(np.mean(vals))
    print(f"feature_dim={fd}  bodies={[g.species for g in genes]}  "
          f"gens={args.gens} pop={args.pop} steps={args.steps} workers={args.workers}", flush=True)

    out = Path(args.out)
    init = None
    if out.exists():
        try:
            p = MorphPolicy.from_npz(str(out))
            if p.feature_dim == fd:
                init = p
                print("warm-starting from the existing (stale) npz — ES elitism can only improve on it", flush=True)
            else:
                print(f"existing npz feature_dim {p.feature_dim} != {fd} (rework changed tokens); training FRESH", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"could not load existing npz ({e}); training fresh", flush=True)

    # NOTE: a custom fitness_fn forces serial evaluation in train_morph_es (the parallel pool is default-fitness
    # only), so keep the budget modest; the height objective converges fast because the bar (>=0.3) is low.
    best, hist = train_morph_es(genes, feature_dim=fd, generations=args.gens, pop=args.pop, steps=args.steps,
                                workers=args.workers, seed=0, init_policy=init, fitness_fn=_height_fitness)
    print("mean-fitness/gen:", [round(float(h), 3) for h in hist], flush=True)

    # Keep-best gate on the TEST's own metric: does at least one topology now score finite + upright?
    res = search_leg_count(best, n_range=range(3, 7), steps=400, baseline_n=4)
    mx = max(res["history"].values())
    print(f"post-train topology scores: {res['history']}  max={mx}", flush=True)
    if mx > float("-inf"):
        if out.exists():
            shutil.copyfile(str(out), str(out) + ".stale.bak")
        best.to_npz(str(out), score=float(mx))
        print(f"SAVED -> {out}  (>=1 topology viable; stale backed up to {out}.stale.bak)", flush=True)
        return 0
    print("new policy STILL collapses every topology (-inf); NOT saved — retrain needs a longer budget / recipe", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
