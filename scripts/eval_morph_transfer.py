"""Cross-morphology transfer eval (Theme 1 payoff): load ONE attention MorphPolicy trained on one body
(``mjx_morph_attention.py --save``) and apply it ZERO-SHOT to DIFFERENT bodies (more/fewer legs). The
attention arch + fixed per-token feature_dim mean the same weights run on any token count — so a policy
trained on a quadruped also drives a hexapod / spider. CPU MuJoCo (no GPU needed for the eval).

    PYTHONPATH=src ~/rl/bin/python scripts/eval_morph_transfer.py --policy runs/morph_quad_att.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _radial_legged(n_legs: int):
    from virturoid.services.morphology_builder import MorphologyBuilder
    b = MorphologyBuilder("legged", base_mount="free", species=f"radial.{n_legs}")
    b.base("torso", shape="box", length=0.08, radius=0.18, mass=3.0)
    for i in range(n_legs):
        a = 2 * math.pi * i / n_legs
        b.limb(f"leg{i}", [{"length": 0.12, "axis": (0, 1, 0), "torque": 14.0},
                           {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}],
               parent="torso", mount_offset=(0.16 * math.cos(a), 0.16 * math.sin(a), -0.04),
               mount_euler=(math.pi, 0, 0))
    return b.build()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="npz saved by mjx_morph_attention.py --save")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args(argv)

    from virturoid.services.morph_policy import MorphPolicy, rollout_morph
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements

    policy = MorphPolicy.from_npz(args.policy)
    print(f"loaded policy feature_dim={policy.feature_dim} hidden={policy.hidden} n_params={policy.n_params}")
    quad = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt="q", robot_class="quadruped"))
    bodies = {"quadruped(trained, 4-leg)": quad, "hexapod(6-leg, zero-shot)": _radial_legged(6),
              "spider(8-leg, zero-shot)": _radial_legged(8), "biped(2-leg, zero-shot)": _radial_legged(2)}
    print(f"{'body':<28} {'tokens':>6} {'fwd':>8} {'disp':>8} {'upright':>8}")
    for name, g in bodies.items():
        r = rollout_morph(g, policy, steps=args.steps, alpha=0.6)
        print(f"{name:<28} {r['n_tokens']:>6} {r['forward']:>+8.3f} {r['base_displacement']:>8.3f} {str(r['upright']):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
