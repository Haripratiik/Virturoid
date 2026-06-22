import unittest

from virturoid.demo import build_demo_project


class DemoProjectTests(unittest.TestCase):
    def test_demo_project_has_full_loop_artifacts(self):
        project = build_demo_project()

        expected_keys = {
            "requirements",
            "components",
            "bom",
            "cad_models",
            "cad_assembly",
            "robot_genome",
            "task_graph",
            "scene_set",
            "policy",
            "evaluation_run",
            "export_bundle",
        }

        self.assertEqual(expected_keys, set(project.keys()))

    def test_demo_project_contains_failure_driven_loop(self):
        run = build_demo_project()["evaluation_run"]
        failures = run["failures"]

        self.assertEqual(1, len(failures))
        self.assertEqual("collision", failures[0]["failure_type"])
        self.assertIn("regression_scene_id", failures[0])


if __name__ == "__main__":
    unittest.main()

