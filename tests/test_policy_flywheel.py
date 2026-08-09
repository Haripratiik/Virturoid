"""Policy flywheel (Theme 2): the GATE on the reusable policy bank — an UNVERIFIED artifact is refused entry,
and a refused artifact is not handed to a different morphology.

THE DOCSTRING HERE USED TO CLAIM THE OPPOSITE, and nothing tested it. It read "a trained MorphPolicy is banked
once and REUSED across builds + across morphologies — a new hexapod build recalls the policy banked from a
quadruped and walks with it", and the test below carried a bare ``return`` that made the last two assertions —
the rollout and the forward check, i.e. the ONLY lines that could have supported that sentence — unreachable.
What actually runs asserts the reverse: ``models/morph_quad_att.npz`` is a crouching residual policy, so
``bank_morph_policy`` REFUSES it and ``recall_morph_policy`` returns None. That refusal is the correct and
valuable behaviour; the sentence describing this file was simply describing a different one.

Cross-body recall IS tested, on the gait bank, in ``test_gait_flywheel.test_recall_by_morphology_embedding_
across_species`` and ``test_gait_recall_hygiene``. A live cross-morphology POLICY reuse test needs a bankable
(credible-recipe) artifact this checkout does not ship — see the note on the dead block below.
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
                pol = recall_morph_policy(hexapod, db)                      # a NEW legged build asks for it...
                self.assertIsNone(pol, "a rejected policy must not be recalled by a different morphology")
                # NOTHING FOLLOWED HERE BUT DEAD CODE. A bare `return` sat on this line with two assertions
                # after it — `rollout_morph(hexapod, pol)` and `forward > 0.0`, the module docstring's whole
                # claim — and `pol` is None on the line above, so they could never have run anyway. They were
                # deleted rather than left as decoration: an unreachable assertion is not weaker evidence than
                # a strong one, it is NO evidence, and it read as coverage for a claim nothing was checking.
                #
                # To restore a real cross-morphology POLICY reuse test, bank a CREDIBLE recipe-controlled
                # artifact (assess_policy_rollout requires deployment_controller == "recipe_cpg") and then
                # assert the recalled policy's SIGNED forward travel on the hexapod. That needs a bankable
                # artifact this checkout does not ship, so it is named here rather than faked.


if __name__ == "__main__":
    unittest.main()
