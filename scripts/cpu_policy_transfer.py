"""CPU params-mode flywheel demo (Theme 2 payoff): the next robot learns FASTER than the last — no GPU.

Trains a MorphPolicy on body A with ES, banks it, then trains a related body B two ways and compares:
  • COLD  — ES from random init;
  • WARM  — ES warm-started from A's banked policy (``transfer_train_morph`` recalls it automatically).
If the flywheel is real, WARM starts from a higher baseline and/or reaches a higher final fitness in the
same budget. Prints both fitness curves + the verdict (startup plan §13 memory, §32 training).

    PYTHONPATH=src python scripts/cpu_policy_transfer.py [--generations 8] [--pop 12] [--steps 200]
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _radial_legged(n_legs: int, species: str):
    from virturoid.services.morphology_builder import MorphologyBuilder
    b = MorphologyBuilder("legged", base_mount="free", species=species)
    b.base("torso", shape="box", length=0.08, radius=0.18, mass=3.0)
    for i in range(n_legs):
        a = 2 * math.pi * i / n_legs
        b.limb(f"leg{i}", [{"length": 0.12, "axis": (0, 1, 0), "torque": 14.0},
                           {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}],
               parent="torso", mount_offset=(0.16 * math.cos(a), 0.16 * math.sin(a), -0.04),
               mount_euler=(math.pi, 0, 0))
    return b.build()


def _curve(hist):
    return " -> ".join(f"{h:.3f}" for h in hist)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morph_trainer import train_morph_es
    from virturoid.services.policy_flywheel import _feature_dim_for, transfer_train_morph

    body_a = _radial_legged(4, "transfer.quad")     # the "prior" robot
    body_b = _radial_legged(6, "transfer.hexapod")  # the new, related robot
    fdim = _feature_dim_for(body_a)
    kw = dict(feature_dim=fdim, generations=args.generations, pop=args.pop, steps=args.steps)

    with tempfile.TemporaryDirectory() as tmp, MemoryDB(Path(tmp) / "mem.db") as db:
        print(f"[1/3] Training body A (quadruped) and banking it (feature_dim={fdim})...", flush=True)
        a = transfer_train_morph([body_a], db, params_path=Path(tmp) / "A.npz", seed=0, **kw)
        print(f"      A: warm_started={a['warm_started']} fitness {_curve(a['history'])}", flush=True)

        print("[2/3] COLD: training body B (hexapod) from scratch...", flush=True)
        _, cold_hist = train_morph_es([body_b], seed=1, **kw)
        print(f"      COLD fitness {_curve(cold_hist)}", flush=True)

        print("[3/3] WARM: training body B with the flywheel (recalls A, warm-starts)...", flush=True)
        warm = transfer_train_morph([body_b], db, params_path=Path(tmp) / "B.npz", seed=1, **kw)
        print(f"      WARM warm_started={warm['warm_started']} fitness {_curve(warm['history'])}", flush=True)

    cold_base, cold_final = cold_hist[0], cold_hist[-1]
    warm_base, warm_final = warm["baseline"], warm["final"]
    print("\n=== CPU params-mode flywheel ===")
    print(f"  body B baseline (gen 0):  cold {cold_base:+.3f}   warm {warm_base:+.3f}   "
          f"(warm starts {'higher' if warm_base > cold_base else 'lower/equal'})")
    print(f"  body B final   (gen {args.generations}):  cold {cold_final:+.3f}   warm {warm_final:+.3f}")
    faster = warm_base > cold_base
    better = warm_final >= cold_final
    print(f"  VERDICT: warm-start {'HELPED' if (faster or better) else 'did not help'} "
          f"(higher baseline={faster}, >= final={better})")
    return 0 if (faster or better) else 1


if __name__ == "__main__":
    raise SystemExit(main())
