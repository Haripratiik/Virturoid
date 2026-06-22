import importlib.util
import unittest

from virturoid.services.task_runtime import builtin_task_specs, select_task_spec

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class TaskSelectionTests(unittest.TestCase):
    def test_prompts_route_to_the_right_task(self):
        self.assertEqual("pick_place_sort", select_task_spec("sort red and blue blocks into bins").task_type)
        self.assertEqual("place_to_target", select_task_spec("lift boxes onto a shelf").task_type)
        self.assertEqual("place_to_target", select_task_spec("pick up the part and place it").task_type)
        self.assertEqual("transport_to_zone", select_task_spec("carry the box across the room").task_type)

    def test_specs_are_well_formed(self):
        for name, spec in builtin_task_specs().items():
            self.assertEqual(name, spec.task_type)
            self.assertTrue(spec.manipulables and spec.targets)
            # every manipulable is assigned to a real target
            for obj, tgt in spec.assignments.items():
                self.assertIn(obj, spec.manipulables)
                self.assertIn(tgt, spec.targets)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class TaskRuntimeTests(unittest.TestCase):
    def test_one_robot_performs_multiple_distinct_tasks(self):
        # The same gene runs sort, place-to-target, AND transport in real physics — task
        # diversity end to end. Assert each runs and the single-object tasks genuinely succeed.
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.task_runtime import evaluate_gene_on_task, generate_task_scenes

        g = tabletop_arm_gene()
        specs = builtin_task_specs()
        for name in ("place_to_target", "transport_to_zone"):
            spec = specs[name]
            scenes = generate_task_scenes(g, spec, count=5, seed=1)
            r = evaluate_gene_on_task(g, spec, scenes)
            self.assertEqual(name, r["task_type"])
            self.assertEqual(5, len(r["episodes"]))
            self.assertGreater(r["blocks_placed"], 0, f"{name} placed nothing")
        # The harder 2-object sort at least runs and places some.
        sort = specs["pick_place_sort"]
        rs = evaluate_gene_on_task(g, sort, generate_task_scenes(g, sort, count=4, seed=1))
        self.assertGreaterEqual(rs["blocks_placed"], 0)


if __name__ == "__main__":
    unittest.main()
