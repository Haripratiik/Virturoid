"""Geometric design critic — the computable VERIFY half of the render->verify->repair loop. It must catch
proportion/balance/stance defects (so an iterative loop or LLM-repair step has a signal), and never crash."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class AnatomyCriticTests(unittest.TestCase):
    def test_returns_structured_critique_with_measures(self):
        import os
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.services.anatomy_critic import critique_gene
        from virturoid.services.morphology_composer import compose_robot

        c = critique_gene(compose_robot("a humanoid robot"))
        self.assertIn("score", c)
        self.assertIn("issues", c)
        self.assertIn("aspect_w_h", c["measures"])
        self.assertGreater(c["measures"]["height_m"], 0.5)
        self.assertTrue(0.0 <= c["score"] <= 1.0)

    def test_flags_stance_defect_and_baked_rest_pose_fixes_it(self):
        # The critic must catch a proportion/stance defect when one exists (so an iterative loop / LLM-repair has a
        # signal). A quad with NO baked stance stands on straight-down legs = too tall/narrow (aspect < 0.8) and the
        # critic flags 'stance'. Baking the default crouch rest pose widens the stance and measurably clears it —
        # this also locks in that the offline quad's baked rest pose is a real proportion improvement, not cosmetic.
        import os
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.services.anatomy_critic import critique_gene
        from virturoid.services.morphology_composer import compose_robot
        unposed = compose_robot("a quadruped robot"); unposed.metadata = {}     # strip the baked crouch
        cu = critique_gene(unposed)
        self.assertIn("stance", {i["check"] for i in cu["issues"]},
                      f"critic must flag the straight-leg stance defect, got {cu['issues']}")
        posed = critique_gene(compose_robot("a quadruped robot"))               # default = baked crouch
        self.assertGreater(posed["score"], cu["score"])                        # the rest pose improves the body

    def test_never_crashes_on_an_invalid_gene(self):
        from virturoid.schemas.gene import GeneSegment, RobotGene
        from virturoid.services.anatomy_critic import critique_gene
        bad = RobotGene(id="x", species="x", robot_class="humanoid", base_mount="free",
                        end_effector_type="none", segments=[GeneSegment("a", parent="missing")])
        c = critique_gene(bad)                 # invalid -> graceful 'compile'/'import' issue, not an exception
        self.assertFalse(c["ok"])
        self.assertEqual(0.0, c["score"])


if __name__ == "__main__":
    unittest.main()
