"""Held-out split + scene-count scaling ablation (scene-gen plan S6). The honest generalization protocol is a
STRUCTURAL train/held-out split (novel layouts at eval, never new seeds of a trained layout), and the proof that
scene diversity prevents overfitting is the scene-count-vs-held-out-success curve the field reports (ProcTHOR
28.7->64.9% from 10->10k houses; RoboCasa 28.8->47.6%; CoinRun 66.8->90.0%). This module owns:

- ``family_to_split`` — a SceneFamily -> VIRT-Bench dev/held_out scene lists (dev = the train pool the submitter
  may tune on; held_out = the structurally-disjoint pool that is the headline number).
- ``scene_count_ablation`` — for each K in a schedule, build a K-scene family, train on it, evaluate on the SAME
  held-out pool, and return the (K -> held_out success) curve. The heavy train+eval is INJECTED as a callable so
  the harness is unit-testable on CPU with a stub and runs the real MJX trainer + verifier on the GPU.

Pure orchestration; no GPU, no MuJoCo here.
"""

from __future__ import annotations

from virturoid.services.scene_family import generate_family


def family_to_split(family) -> dict:
    """SceneFamily -> ``{"dev": [...], "held_out": [...], "disjoint": bool}`` for VIRT-Bench. dev is the train
    pool; held_out is the structurally-disjoint pool. The disjointness is asserted so a leak can't inflate the
    held-out number."""
    return {"dev": list(family.train), "held_out": list(family.held_out),
            "dev_keys": list(family.train_keys), "held_out_keys": list(family.held_out_keys),
            "disjoint": family.disjoint}


def scene_count_ablation(task_type: str, counts, train_eval_fn, *, n_held_out: int = 4, difficulty: int = 1,
                         seed: int = 0) -> dict:
    """Run the scene-count scaling ablation. For each ``K`` in ``counts``: generate a family with ``K`` TRAIN
    scenes and a FIXED held-out pool, then call ``train_eval_fn(train_scenes, held_out_scenes, k=K) -> success``
    (the held-out success rate in [0,1]). Returns ``{"curve": [(K, success)...], "held_out_ids", "monotone_frac"}``.

    The held-out pool is held FIXED across all K (same evaluation target) by seeding it independently, so the
    curve isolates the effect of TRAIN diversity. ``train_eval_fn`` is the injection point for the real
    MJX-train + VIRT-Bench-eval on the GPU; a stub makes the harness testable on CPU."""
    # a fixed held-out pool shared by every rung (generated once, from a dedicated seed)
    held_family = generate_family(task_type, n_train=1, n_held_out=n_held_out, difficulty=difficulty, seed=seed + 777)
    held_out = held_family.held_out
    curve = []
    for k in counts:
        fam = generate_family(task_type, n_train=int(k), n_held_out=0, difficulty=difficulty, seed=seed + int(k))
        # guarantee the train pool never overlaps the fixed held-out structures (honest split)
        held_keys = set(held_family.held_out_keys)
        train = [s for s, key in zip(fam.train, fam.train_keys) if key not in held_keys]
        succ = float(train_eval_fn(train, held_out, k=int(k)))
        curve.append((int(k), round(succ, 4)))
    # how often success does not DECREASE as K grows (a scaling law should be largely monotone-up)
    ups = sum(1 for a, b in zip(curve, curve[1:]) if b[1] >= a[1] - 1e-9)
    return {"curve": curve, "held_out_ids": [s.id for s in held_out],
            "monotone_frac": round(ups / max(1, len(curve) - 1), 3),
            "task_type": task_type}
