"""Seed gene contracts — the buildable trunk of the species tree.

Guards the gene SOURCE on CPU (no MuJoCo): each class's default gene is valid and is the structural kind
its class implies — in particular the quadruped gene is a real free-base LEGGED body (so a quadruped
request routes to the locomotion build + policy flywheel, not the manipulator path)."""

import unittest

from virturoid.fixtures.gene_library import (
    gene_for_class,
    humanoid_upper_body_gene,
    tabletop_arm_gene,
)


class GeneLibraryTests(unittest.TestCase):
    def test_quadruped_gene_is_a_valid_free_base_legged_body(self):
        from virturoid.services.task_matched_eval import robot_kind

        g = gene_for_class("quadruped")
        self.assertIsNotNone(g, "quadruped must have a buildable gene (else it falls through to manipulator)")
        self.assertEqual([], g.validate(), "quadruped gene must be schema-valid")
        self.assertEqual("quadruped", g.robot_class)
        self.assertEqual("free", g.base_mount)              # free-floating walker, not a bolted arm
        self.assertEqual("none", g.end_effector_type)       # no gripper/tool
        self.assertGreaterEqual(sum(1 for s in g.segments if s.joint_type == "revolute"), 8)
        self.assertEqual("legged", robot_kind(g))           # routes to locomotion eval + the policy flywheel

    def test_quadruped_gene_is_a_fresh_instance_each_call(self):
        # _maybe_gene_build may amend the returned gene; callers must not share mutable state.
        self.assertIsNot(gene_for_class("quadruped"), gene_for_class("quadruped"))

    def test_legged_gene_honours_requested_leg_count(self):
        # Generality: a hexapod/octopod request must build the RIGHT number of legs (3 DOF each),
        # not a 4-leg stand-in. The prompt is threaded through so the composer infers leg count.
        from virturoid.services.task_matched_eval import robot_kind

        def revolute(g):
            return sum(1 for s in g.segments if s.joint_type == "revolute")

        quad = gene_for_class("quadruped", prompt="a quadruped robot that walks")
        hexa = gene_for_class("quadruped", prompt="a hexapod robot that walks")
        octo = gene_for_class("quadruped", prompt="an octopod walking robot")
        self.assertEqual(12, revolute(quad))   # 4 legs x 3 DOF
        self.assertEqual(18, revolute(hexa))   # 6 legs x 3 DOF
        self.assertEqual(24, revolute(octo))   # 8 legs x 3 DOF
        for g in (quad, hexa, octo):
            self.assertEqual([], g.validate())
            self.assertEqual("legged", robot_kind(g))

    def test_legged_phrasings_route_to_a_legged_class(self):
        # Regression: legged-locomotion requests must NOT fall through to the manipulator path.
        from virturoid.services.morphology_selector import _explicit_class

        for p in ("build a hexapod robot", "a six-legged walking robot", "an octopod robot",
                  "a quadruped", "a walking robot", "a four-legged dog robot", "a spider robot",
                  "a robot that walks through a maze"):
            self.assertEqual("quadruped", _explicit_class(p), p)

    def test_navigation_intent_selects_the_perceptive_policy(self):
        # An obstacle/maze/avoid request trains a PERCEPTIVE sense-and-avoid policy; a plain walk does not.
        from virturoid.services.autonomous_build import _is_navigation_intent

        for p in ("a quadruped that navigates around obstacles", "a robot that walks through a maze",
                  "a legged robot that avoids walls", "walk to a goal avoiding obstacles"):
            self.assertTrue(_is_navigation_intent(p), p)
        for p in ("a quadruped that walks forward", "a hexapod that walks fast", "a robot arm that sorts blocks"):
            self.assertFalse(_is_navigation_intent(p), p)

    def test_known_classes_resolve_and_unknown_is_none(self):
        self.assertEqual("manipulator", gene_for_class("manipulator").robot_class)
        self.assertEqual("humanoid", gene_for_class("humanoid").robot_class)
        self.assertIsNone(gene_for_class("octopod"))        # no gene yet -> honest None (legacy path)

    def test_existing_seed_genes_unchanged_kind(self):
        from virturoid.services.task_matched_eval import robot_kind

        # The arm and the floor-mounted humanoid upper body remain manipulators (must keep the
        # pick-place path; only free-base bodies are legged).
        self.assertEqual("manipulator", robot_kind(tabletop_arm_gene()))
        self.assertEqual("manipulator", robot_kind(humanoid_upper_body_gene()))


if __name__ == "__main__":
    unittest.main()
