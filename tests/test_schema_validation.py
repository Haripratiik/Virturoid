import unittest

from virturoid.schemas.components import BillOfMaterials, BomItem
from virturoid.schemas.scenes import SceneGraph
from virturoid.schemas.tasks import TaskGraph


class SchemaValidationTests(unittest.TestCase):
    def test_task_graph_requires_measurable_success_and_failure(self):
        task = TaskGraph(id="task_bad", name="Bad task", prompt="Do the thing", task_type="unknown")

        result = task.validate()

        self.assertFalse(result.ok)
        self.assertTrue(any(issue.field == "success_criteria" for issue in result.issues))
        self.assertTrue(any(issue.field == "failure_criteria" for issue in result.issues))

    def test_bom_rejects_invalid_quantity(self):
        bom = BillOfMaterials(
            id="bom_bad",
            robot_design_id="design_1",
            items=[BomItem(role="motor", component_id="cmp_1", quantity=0)],
        )

        result = bom.validate()

        self.assertFalse(result.ok)
        self.assertTrue(any(issue.code == "invalid_quantity" for issue in result.issues))

    def test_scene_requires_backend_and_spawn(self):
        scene = SceneGraph(id="scene_bad", name="Bad scene")

        result = scene.validate()

        self.assertFalse(result.ok)
        self.assertTrue(any(issue.field == "backend_targets" for issue in result.issues))
        self.assertTrue(any(issue.field == "robot_spawn_xyz_rpy" for issue in result.issues))


if __name__ == "__main__":
    unittest.main()

