"""Validate a freshly GPU-trained quad locomotion policy and bank it as the species policy ONLY if it beats
the current banked policy on the demo's OWN episode metric (keep-best -> no regressions).

The recipe obs-normalizer is deterministic from the gene (it isn't saved in the trained npz), so we try banking
the new policy WITH it and RAW, deploy each through the demo's real `simulate_episode_for_viewer`, and keep
whichever the quad actually walks best with -- or revert to the existing policy if neither is better.

Usage:  PYTHONPATH=src python scripts/_bank_quad_walk.py [new_npz]   (default: models/quad_gpu_v1.npz)
"""
import sys
import traceback
from pathlib import Path

from virturoid.services.morphology_composer import compose_robot
from virturoid.services.morph_policy import MorphPolicy, recipe_obs_normalizer
from virturoid.services.viewer_sim import simulate_episode_for_viewer

NEW = sys.argv[1] if len(sys.argv) > 1 else "models/quad_gpu_v1.npz"
SPECIES = Path("models/learned_quadruped.composed.npz")
PKG = Path("build/ui_verify/build_a_quadruped_robot_that_walks").resolve()

g = compose_robot("build a quadruped robot that walks", llm=None)
norm = recipe_obs_normalizer(g)


def outcome(label):
    oc = simulate_episode_for_viewer(PKG, scene_index=0).get("outcome", {}) or {}
    fwd = float(oc.get("forward_m", 0) or 0)
    print(f"  {label:22} status={str(oc.get('status')):8} forward_m={fwd:+.3f} upright={oc.get('upright')}", flush=True)
    # honest score: upright first (a fall is never an improvement), then forward distance
    return (1 if oc.get("upright") else 0, fwd), oc


backup = SPECIES.read_bytes() if SPECIES.exists() else None
try:
    print("baseline (current banked policy):", flush=True)
    base_score, _ = outcome("EXISTING")

    new = MorphPolicy.from_npz(NEW)
    print(f"new policy: feature_dim={new.feature_dim}", flush=True)

    print("candidates (the new policy, two normalizations):", flush=True)
    new.to_npz(str(SPECIES), normalizer=norm)
    s_wn, oc_wn = outcome("NEW with-norm")
    MorphPolicy.from_npz(NEW).to_npz(str(SPECIES), normalizer=None)
    s_raw, oc_raw = outcome("NEW raw")

    cands = [("with-norm", s_wn, oc_wn, norm), ("raw", s_raw, oc_raw, None)]
    best = max(cands, key=lambda c: c[1])
    keep = best[1] > base_score and best[2].get("status") != "fell"
    if keep:
        MorphPolicy.from_npz(NEW).to_npz(str(SPECIES), normalizer=best[3])
        print(f"DECISION: BANK new ({best[0]})  score {best[1]} > baseline {base_score}", flush=True)
    else:
        if backup is not None:
            SPECIES.write_bytes(backup)
        print(f"DECISION: KEEP existing  (best new {best[1]} <= baseline {base_score})", flush=True)
    print("final state:", flush=True)
    outcome("FINAL")
except Exception:
    if backup is not None:
        SPECIES.write_bytes(backup)
    traceback.print_exc()
    print("ERROR -> reverted to existing banked policy", flush=True)
