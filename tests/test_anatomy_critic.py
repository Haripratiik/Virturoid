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

    def test_flags_stance_defect_and_the_shipped_quad_clears_it(self):
        # The critic must catch a proportion/stance defect when one exists (so an iterative loop / LLM-repair has a
        # signal). HISTORY: this test used to elicit the defect by stripping the baked crouch off the default quad
        # (straight-down legs read tall/narrow). The 2026-07 anatomy work moved the wide stance INTO the authored
        # geometry, so the default quad now clears the check even unposed (measured aspect ~1.25 — a real body
        # improvement, which went stale-proxy on this test). The capability is therefore pinned on a CONSTRUCTED
        # defective body: a stilt — one tall stack of short segments on a small base (W:H ~0.22) — the exact
        # too-tall/narrow shape the check exists to flag. NOTE the width measure uses geom_rbound (bounding
        # spheres), so long-limbed bodies self-widen; only a stack of SHORT segments is genuinely narrow to it.
        import os
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.schemas.gene import GeneSegment, RobotGene
        from virturoid.services.anatomy_critic import critique_gene
        from virturoid.services.morphology_composer import compose_robot
        segs = [GeneSegment("torso", parent=None, shape="box", length_m=0.1, radius_m=0.05, mass_kg=0.6)]
        prev = "torso"
        for i in range(5):                                     # children stack upward at the default mount_euler
            segs.append(GeneSegment(f"c{i}", parent=prev, shape="capsule", length_m=0.14, radius_m=0.025,
                                    mass_kg=0.12, joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-0.5,
                                    joint_upper=0.5, is_end_effector=(i == 4)))
            prev = f"c{i}"
        stilt = RobotGene(id="stilt", species="stilt", robot_class="quadruped", base_mount="free",
                          end_effector_type="none", segments=segs)
        cd = critique_gene(stilt)
        self.assertIn("stance", {i["check"] for i in cd["issues"]},
                      f"critic must flag the tall/narrow stilt body, got {cd['issues']}")
        self.assertLess(cd["score"], 1.0)
        posed = critique_gene(compose_robot("a quadruped robot"))              # the SHIPPED default quad
        self.assertNotIn("stance", {i["check"] for i in posed["issues"]},
                         "the default quad's authored wide stance must clear the check")
        self.assertGreater(posed["score"], cd["score"])                       # good body outscores the defect

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
