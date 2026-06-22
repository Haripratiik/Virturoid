"""Generality of control + dispatch: one gait drives ANY leg count, and evaluation routes by STRUCTURE
(so an LLM-named 'hexapod'/'spider'/'frog' is handled), not by a fixed class-name catalogue."""

import importlib.util
import math
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _radial_legged(n_legs: int):
    """A free-floating body with ``n_legs`` legs evenly spaced around the torso (legs point down)."""
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


class TopologyDispatchTests(unittest.TestCase):
    def test_robot_kind_by_structure(self):
        from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
        from virturoid.services.task_matched_eval import robot_kind

        legged = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt="x", robot_class="quadruped"))
        mobile = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt="x", robot_class="mobile_base"))
        arm = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt="grasp", robot_class="manipulator"))
        spray = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt="spray paint", robot_class="manipulator"))
        self.assertEqual(robot_kind(legged), "legged")
        self.assertEqual(robot_kind(mobile), "mobile")
        self.assertEqual(robot_kind(arm), "manipulator")
        self.assertEqual(robot_kind(spray), "spray")

    def test_hexapod_kind_is_legged_regardless_of_leg_count(self):
        from virturoid.services.task_matched_eval import robot_kind
        for n in (2, 4, 6, 8):
            self.assertEqual(robot_kind(_radial_legged(n)), "legged", f"{n}-leg body should be legged")


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class GeneralGaitTests(unittest.TestCase):
    def test_one_gait_walks_any_leg_count(self):
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.locomotion_controller import run_locomotion_episode

        for n in (4, 6, 8):
            mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(_radial_legged(n)))
            r = run_locomotion_episode(mj)
            self.assertGreater(r["distance_m"], 0.3, f"{n}-leg body should locomote, got {r}")
            self.assertTrue(r["upright"], f"{n}-leg body should stay upright, got {r}")

    def test_evaluate_robot_routes_a_hexapod_to_locomotion(self):
        from virturoid.services.task_matched_eval import evaluate_robot

        ev = evaluate_robot(_radial_legged(6))
        self.assertEqual(ev["task"], "locomotion")
        self.assertGreater(ev["value"], 0.3)


if __name__ == "__main__":
    unittest.main()
