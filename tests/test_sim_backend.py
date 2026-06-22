"""Backend-agnostic boundary (roadmap ADR-2 / §9 / M-N9): the SimulationBackend interface + MuJoCo adapter.

Offline: compile is pure MJCF string-building, gene validation is pure, and the Protocol check is
structural — none of this needs MuJoCo installed."""

import unittest

from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.schemas.gene import RobotGene
from virturoid.services.sim_backend import (
    MujocoBackend,
    SimulationBackend,
    available_backends,
    get_backend,
)


class SimBackendTests(unittest.TestCase):
    def test_mujoco_adapter_conforms_to_the_interface(self):
        b = get_backend("mujoco")
        self.assertEqual(b.name, "mujoco")
        self.assertIsInstance(b, SimulationBackend)        # structural Protocol conformance
        self.assertIsInstance(MujocoBackend(), SimulationBackend)

    def test_unknown_backend_raises_with_a_helpful_message(self):
        with self.assertRaises(KeyError) as cm:
            get_backend("isaac")
        self.assertIn("isaac", str(cm.exception))

    def test_validate_robot_passes_a_valid_gene(self):
        out = get_backend("mujoco").validate_robot(tabletop_arm_gene())
        self.assertTrue(out.ok, out.issues)
        self.assertEqual(out.backend, "mujoco")

    def test_validate_robot_flags_an_invalid_gene(self):
        bad = RobotGene(id="bad", species="x", robot_class="manipulator", segments=[])
        out = get_backend("mujoco").validate_robot(bad)
        self.assertFalse(out.ok)
        self.assertTrue(out.issues)

    def test_compile_robot_returns_mjcf(self):
        xml = get_backend("mujoco").compile_robot(tabletop_arm_gene(), include_floor=False)
        self.assertIn("<mujoco", xml)
        self.assertIn("actuator", xml)

    def test_available_backends_lists_mujoco(self):
        ab = available_backends()
        self.assertIn("mujoco", ab)
        self.assertIsInstance(ab["mujoco"], bool)


if __name__ == "__main__":
    unittest.main()
