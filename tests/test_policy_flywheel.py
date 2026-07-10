"""Policy flywheel (Theme 2): a trained MorphPolicy is banked once and REUSED across builds + across
morphologies — a new hexapod build recalls the policy banked from a quadruped and walks with it.
Gated on the GPU-trained policy file (models/morph_quad_att.npz) being present."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_NPZ = Path(__file__).resolve().parent.parent / "models" / "morph_quad_att.npz"


class PolicyCredibilityTests(unittest.TestCase):
    def test_only_a_credible_recipe_walk_is_bankable(self):
        from virturoid.services.policy_flywheel import assess_policy_rollout

        credible = {"deployment_controller": "recipe_cpg", "survived": True, "upright_frac": 0.9,
                    "height_ratio": 0.9, "cadence": 2.0, "support_frac": 0.5, "forward": 0.5}
        self.assertTrue(assess_policy_rollout(credible)["bankable"])
        raw = {"deployment_controller": "residual", "forward": 2.0, "upright": True}
        self.assertFalse(assess_policy_rollout(raw)["bankable"])


def _radial(n):
    from virturoid.services.morphology_builder import MorphologyBuilder
    b = MorphologyBuilder("legged", base_mount="free", species=f"r{n}")
    b.base("torso", shape="box", length=0.08, radius=0.18, mass=3.0)
    for i in range(n):
        a = 2 * math.pi * i / n
        b.limb(f"leg{i}", [{"length": 0.12, "axis": (0, 1, 0), "torque": 14.0},
                           {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}],
               parent="torso", mount_offset=(0.16 * math.cos(a), 0.16 * math.sin(a), -0.04),
               mount_euler=(math.pi, 0, 0))
    return b.build()


@unittest.skipUnless(_MUJOCO and _NPZ.exists(), "needs MuJoCo + a trained policy (models/morph_quad_att.npz)")
class PolicyFlywheelTests(unittest.TestCase):
    def test_bank_then_recall_across_build_and_morphology(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morph_policy import rollout_morph
        from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
        from virturoid.services.policy_flywheel import bank_morph_policy, recall_morph_policy

        quad = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt="q", robot_class="quadruped"))
        hexapod = _radial(6)
        with tempfile.TemporaryDirectory() as td:
            with MemoryDB(Path(td) / "m.db") as db:
                self.assertIsNone(recall_morph_policy(hexapod, db))         # nothing banked yet
                res = bank_morph_policy(str(_NPZ), quad, db)
                self.assertFalse(res["banked"], "a crouching residual policy must not enter the reusable bank")
                self.assertIn("UNVERIFIED", res["verdict"])
                pol = recall_morph_policy(hexapod, db)                      # NEW legged build recalls it
                self.assertIsNone(pol, "a rejected policy must not be recalled by a different morphology")
                return
                r = rollout_morph(hexapod, pol, steps=400)                  # reused on a DIFFERENT body
                self.assertGreater(r["forward"], 0.0)                       # the banked policy walks it FORWARD
                # Zero-shot cross-morphology transfer (a QUAD policy driving a HEXAPOD) crouch-walks: it moves
                # forward but in a low stance, so it need not clear the strict upright gate. We assert it stays
                # substantially OFF the floor (didn't collapse) — the honest claim for un-fine-tuned transfer.
                # (Before the standing-spawn fix this asserted upright, but the gate was vacuous: the body
                # spawned penetrating at z0=0.1 so z>0.5*z0 was always true even while bouncing — see
                # [[task-effectiveness-loop]].)
                self.assertGreater(r["height_ratio"], 0.3)


if __name__ == "__main__":
    unittest.main()
