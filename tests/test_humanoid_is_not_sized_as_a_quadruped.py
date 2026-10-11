"""A humanoid's two arms are not two more legs.

``grounded_physics.physical_prior_for`` gave the QUADRUPED envelope ((12, 15) kg, "commercial mid-size quadruped
envelope") to any body ``_leg_branches`` counted four branches on, and that counter cannot tell an arm from a leg:
shoulder + elbow + a welded hand is the same chain shape as hip + knee + a welded foot. MEASURED through the MCP
server: an authored humanoid (torso 0.5, two 0.55 m arms, two 0.8 m legs) drew the critique finding "total mass
22.9 kg is outside the plausible humanoid band (12.0-15.0 kg)", and the composed humanoid was sized against the
band's 13.5 kg midpoint. The same counter split the support load over the arms, so once the hint stopped being a
quadruped's every arm joint of the composed humanoid was fitted with the catalog's 520 N.m drive.

The legs are now the chains the compiled body STANDS on (the ``appendage_map`` measurement ``body_kind`` uses),
intersected with the old candidates so nothing that was not a leg can become one.
"""

import copy
import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None

HUMANOID = {"robot_class": "humanoid", "name": "agent_humanoid", "parts": [
    {"name": "torso", "role": "body", "size": 0.50, "girth": 0.13},
    {"name": "head", "role": "head", "parent": "torso", "attach": "front_top", "aim": "up", "size": 0.18,
     "girth": 0.08},
    {"name": "arm", "role": "arm", "parent": "torso", "attach": "front_top", "aim": "down_out", "size": 0.55,
     "girth": 0.035, "segments": 3, "symmetry": "left_right", "joint": "revolute"},
    {"name": "leg", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down", "size": 0.80,
     "girth": 0.045, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]}

DOG = {"robot_class": "quadruped", "name": "agent_dog", "parts": [
    {"name": "torso", "role": "body", "size": 0.45, "girth": 0.10},
    {"name": "head", "role": "head", "parent": "torso", "attach": "front_top", "aim": "forward", "size": 0.12,
     "girth": 0.05},
    {"name": "leg_front", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out",
     "size": 0.40, "girth": 0.022, "segments": 4, "symmetry": "left_right", "joint": "revolute"},
    {"name": "leg_rear", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down_out",
     "size": 0.40, "girth": 0.022, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]}


def _build(graph):
    from virturoid.services.anatomy_compiler import build_from_anatomy
    return build_from_anatomy(copy.deepcopy(graph))


@unittest.skipUnless(_MUJOCO, "telling a leg from an arm is measured on the compiled body")
class PriorSelectionTests(unittest.TestCase):
    def test_the_old_counter_still_reads_four_so_this_test_means_something(self):
        from virturoid.services.grounded_physics import _leg_branches
        self.assertEqual(len(_leg_branches(_build(HUMANOID))), 4)

    def test_an_authored_humanoid_gets_no_quadruped_envelope(self):
        from virturoid.services.grounded_physics import physical_prior_for
        self.assertIsNone(physical_prior_for(_build(HUMANOID)))

    def test_the_composed_humanoid_gets_no_quadruped_envelope(self):
        """The offline anthropometric humanoid ("a humanoid robot") has the same two-arms-two-legs shape."""
        from virturoid.services.grounded_physics import _leg_branches, physical_prior_for
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot("a humanoid robot", ensure_walkable=False)
        self.assertEqual(len(_leg_branches(gene)), 4)
        self.assertIsNone(physical_prior_for(gene))

    def test_structure_decides_even_when_the_class_is_left_out(self):
        """``build_from_anatomy`` defaults a missing robot_class to "quadruped", so the class alone cannot be
        trusted to name a humanoid; the measurement does not read it."""
        from virturoid.services.grounded_physics import physical_prior_for
        unlabelled = {k: v for k, v in HUMANOID.items() if k != "robot_class"}
        gene = _build(unlabelled)
        self.assertEqual(gene.robot_class, "quadruped")
        self.assertIsNone(physical_prior_for(gene))

    def test_a_quadruped_keeps_its_envelope(self):
        from virturoid.services.agent_design_tools import _EXAMPLE_QUAD
        from virturoid.services.grounded_physics import physical_prior_for
        for name, graph in (("authored dog", DOG), ("schema example", _EXAMPLE_QUAD)):
            with self.subTest(body=name):
                prior = physical_prior_for(_build(graph))
                self.assertIsNotNone(prior)
                self.assertEqual(prior.id, "legged.quadruped")

    def test_the_critique_no_longer_quotes_a_quadruped_band(self):
        from virturoid.services.gene_validation import validate_gene_design
        from virturoid.services.grounded_physics import ground_gene
        gene = _build(HUMANOID)
        ground_gene(gene, material="aluminum", fill=0.25)           # what submit_design does
        flags = [f["detail"] for f in validate_gene_design(gene)["risk_flags"] if f["check"] == "mass_budget"]
        self.assertFalse([f for f in flags if "12.0-15.0" in f], flags)


@unittest.skipUnless(_MUJOCO, "telling a leg from an arm is measured on the compiled body")
class SupportSplitTests(unittest.TestCase):
    def test_the_arms_do_not_carry_the_body(self):
        from virturoid.services.grounded_physics import _carrying_branches
        names = sorted(s.name for s in _carrying_branches(_build(HUMANOID)))
        self.assertEqual(names, ["leg_l_0", "leg_r_0"])

    def test_an_arm_is_sized_as_an_arm(self):
        """Each leg holds half the robot; the arms hold only themselves. Under the old split the shoulder was
        sized for a quarter of the robot at full arm's reach, above the hip it hangs beside."""
        from virturoid.services.grounded_physics import ground_gene
        gene = _build(HUMANOID)
        ground_gene(gene, material="aluminum", fill=0.25)
        req = {s.name: float(s.torque_req_nm) for s in gene.segments if s.joint_type == "revolute"}
        self.assertLess(req["arm_l_0"], 0.1 * req["leg_l_0"], req)
        self.assertAlmostEqual(req["arm_l_0"], req["arm_r_0"], places=6)

    def test_a_quadruped_is_split_over_all_four_legs_as_before(self):
        from virturoid.services.grounded_physics import _carrying_branches, _leg_branches
        gene = _build(DOG)
        self.assertEqual([s.name for s in _carrying_branches(gene)], [s.name for s in _leg_branches(gene)])


class WithoutAMeasurementTests(unittest.TestCase):
    """No MuJoCo, or a body that will not compile: the DECLARED class decides, and only the prior changes."""

    def _patched(self):
        from unittest import mock

        from virturoid.services import grounded_physics as gp
        return mock.patch.object(gp, "_stands_on", lambda gene: None)

    def test_a_declared_humanoid_is_still_not_a_quadruped(self):
        from virturoid.services.grounded_physics import physical_prior_for
        with self._patched():
            self.assertIsNone(physical_prior_for(_build(HUMANOID)))

    def test_a_declared_quadruped_keeps_its_envelope(self):
        from virturoid.services.grounded_physics import physical_prior_for
        with self._patched():
            self.assertEqual(physical_prior_for(_build(DOG)).id, "legged.quadruped")

    def test_the_support_split_is_left_as_it_was(self):
        from virturoid.services.grounded_physics import _carrying_branches, _leg_branches
        with self._patched():
            gene = _build(HUMANOID)
            self.assertEqual([s.name for s in _carrying_branches(gene)], [s.name for s in _leg_branches(gene)])


if __name__ == "__main__":
    unittest.main()
