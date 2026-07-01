"""Phase 1 of the robotics-native-AI plan: the RECALL organ.

Tips + lessons are un-dead-ended — indexed into the vector store keyed by the body's morphology embedding,
so knowledge written for one robot is retrievable for a NEAR-morphology robot (was write-only tips + EXACT
class+failure_code string lookups). Pure-Python, no MuJoCo.
"""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import (humanoid_upper_body_gene, quadruped_gene,
                                             tabletop_arm_gene)
from virturoid.services.design_critic import add_parallel_gripper
from virturoid.services.memory_db import MemoryDB
from virturoid.services.robotics_vector_memory import LESSON, TIP, RoboticsVectorMemory
from virturoid.services.species_discovery import auto_place_species


class KnowledgeRecallTests(unittest.TestCase):
    def _mem(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_index_tips_is_incremental(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
            db.add_species_tip(sp, "keep the wrist DOF wide for top-down grasps", audience="builder")
            vm = RoboticsVectorMemory(db)
            self.assertEqual(vm.index_tips(), 1)       # one tip embedded
            self.assertEqual(vm.index_tips(), 0)       # incremental: nothing new
            self.assertEqual(vm.count(TIP), 1)

    def test_tips_retrieve_by_morphology_not_exact_species(self):
        """The un-dead-ending: a tip written for the arm is recalled for a NEAR body (gripper-arm),
        ahead of a humanoid's tip — even though the query species string never matches the arm's."""
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            arm_sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
            hum_sp = auto_place_species(humanoid_upper_body_gene(), db)["species_pattern"]
            db.add_species_tip(arm_sp, "lengthen the upper arm when reach-limited", audience="builder")
            db.add_species_tip(hum_sp, "widen the stance for balance", audience="builder")
            vm = RoboticsVectorMemory(db)
            vm.index_tips()
            know = vm.recall_knowledge(add_parallel_gripper(tabletop_arm_gene()), "grasp", k=2)
            self.assertTrue(know["tips"])
            self.assertIn("upper arm", know["tips"][0]["meta"]["tip"])     # the arm tip ranks first
            self.assertEqual(know["tips"][0]["meta"]["species"], arm_sp)

    def test_lessons_retrieve_by_body_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            auto_place_species(tabletop_arm_gene(), db)          # a manipulator body exists for class-rep
            auto_place_species(humanoid_upper_body_gene(), db)
            db.record_lesson("manipulator", "reach_limited", "lengthen_reach", improvement=0.3,
                             task_type="grasp", root_cause="ee short by 0.12 m")
            db.record_lesson("humanoid", "unstable", "widen_base", improvement=0.2, task_type="walk")
            vm = RoboticsVectorMemory(db)
            self.assertEqual(vm.index_lessons(), 2)
            know = vm.recall_knowledge(add_parallel_gripper(tabletop_arm_gene()), "grasp",
                                       failure_code="reach_limited", k=2)
            self.assertTrue(know["lessons"])
            top = know["lessons"][0]["meta"]
            self.assertEqual(top["failure_code"], "reach_limited")         # the matching fix surfaces
            self.assertEqual(top["operator"], "lengthen_reach")
            self.assertIn("0.12", top["root_cause"])

    def test_index_lessons_skips_unproven(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            auto_place_species(tabletop_arm_gene(), db)
            db.record_lesson("manipulator", "proven", "op_a", improvement=0.4)
            db.record_lesson("manipulator", "unproven", "op_b", improvement=0.0)   # improvement not > 0
            vm = RoboticsVectorMemory(db)
            self.assertEqual(vm.index_lessons(), 1)               # only the proven lesson
            self.assertEqual(vm.count(LESSON), 1)

    def test_recall_knowledge_bundles_tips_lessons_skills(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            arm_sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
            db.add_species_tip(arm_sp, "keep the gripper light", audience="builder")
            db.record_lesson("manipulator", "reach_limited", "lengthen_reach", improvement=0.3)
            db.record_skill("grasp.arm", "manipulator", "grasp", success_rate=0.9, species=arm_sp)
            vm = RoboticsVectorMemory(db)
            # refresh=True (default) indexes everything on the fly -> a cold caller gets full recall
            know = vm.recall_knowledge(tabletop_arm_gene(), "grasp", failure_code="reach_limited")
            self.assertEqual(set(know), {"tips", "lessons", "skills"})
            self.assertTrue(know["tips"] and know["lessons"] and know["skills"])
            self.assertEqual(know["skills"][0]["obj_id"], "grasp.arm")

    def test_nearest_bodies(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            auto_place_species(tabletop_arm_gene(), db)
            auto_place_species(quadruped_gene(), db)
            vm = RoboticsVectorMemory(db)
            vm.index_species_bodies()
            hits = vm.nearest_bodies(add_parallel_gripper(tabletop_arm_gene()), k=2)
            self.assertEqual(hits[0]["meta"]["robot_class"], "manipulator")   # nearest body is the arm


if __name__ == "__main__":
    unittest.main()
