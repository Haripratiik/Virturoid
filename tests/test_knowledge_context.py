"""Phase 2 of the robotics-native-AI plan: RAG-inject the LLM roles (the AURA 38->99 move).

Verifies (a) the PRIOR-KNOWLEDGE block assembles from seeded memory, grounded in the body + retrieved
tips/lessons/skills, and (b) each LLM role actually prepends the block to the prompt it sends the model
(captured via a fake LLM). Pure-Python, no MuJoCo.
"""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.services.design_critic import add_parallel_gripper
from virturoid.services.knowledge_context import assemble_prior_knowledge, assemble_prompt_context
from virturoid.services.memory_db import MemoryDB


class _CaptureLLM:
    """A fake LLM that records every user prompt it's asked to complete (for injection assertions)."""
    name = "fake"

    def __init__(self, ret=None):
        self.ret = ret if ret is not None else {}
        self.users: list[str] = []

    def complete_json(self, system, user, schema, max_tokens=None):
        self.users.append(user)
        return self.ret


class AssembleTests(unittest.TestCase):
    def _mem(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def _seed(self, db):
        from virturoid.services.species_discovery import auto_place_species
        sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
        db.add_species_tip(sp, "keep the wrist DOF wide for top-down grasps", audience="builder")
        db.record_lesson("manipulator", "reach_limited", "lengthen_reach", improvement=0.3,
                         task_type="grasp", root_cause="ee short by 0.12 m")
        db.record_skill("grasp.arm", "manipulator", "grasp", success_rate=0.9, species=sp)
        return sp

    def test_block_is_grounded_in_body_and_retrieved_knowledge(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            self._seed(db)
            block = assemble_prior_knowledge(db, gene=add_parallel_gripper(tabletop_arm_gene()),
                                             task_type="grasp", failure_code="reach_limited")
            self.assertIn("PRIOR KNOWLEDGE", block)
            self.assertIn("Body:", block)                       # describe_robot summary
            self.assertIn("wrist DOF", block)                   # the tip
            self.assertIn("lengthen_reach", block)              # the lesson's fix
            self.assertIn("grasp", block)                       # the reusable skill

    def test_empty_memory_returns_empty_block(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            self.assertEqual(assemble_prior_knowledge(db, gene=tabletop_arm_gene(), task_type="grasp"), "")

    def test_prompt_context_recalls_prior_art(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            db.record_run("teach the arm to grasp a cube", "manipulator", "grasp", None, 0.9)
            block = assemble_prompt_context(db, prompt="grasp an object", task_type="grasp",
                                            robot_class="manipulator")
            self.assertIn("PRIOR ART", block)
            self.assertIn("grasp", block)

    def test_accepts_a_memory_dir_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryDB(Path(tmp) / "virturoid_memory.db") as db:
                self._seed(db)
            # a bare dir path (the caller's common case) opens the DB itself
            block = assemble_prior_knowledge(tmp, gene=tabletop_arm_gene(), task_type="grasp")
            self.assertIn("PRIOR KNOWLEDGE", block)


class RoleInjectionTests(unittest.TestCase):
    def test_reward_agent_prepends_prior_knowledge(self):
        from virturoid.services.reward_agent import translate_task_to_reward
        llm = _CaptureLLM({
            "task_type": "grasp",
            "sparse_success": {"name": "done", "expression": "object_in_zone == 1 and object_settled == 1"},
            "failure": [{"name": "timeout", "expression": "t > 10"}], "dense_terms": [],
        })
        translate_task_to_reward("grasp a cube", None, llm, prior_knowledge="PK-BLOCK-REWARD")
        self.assertTrue(any("PK-BLOCK-REWARD" in u for u in llm.users))

    def test_task_proposer_prepends_prior_knowledge(self):
        from virturoid.services.task_proposer import propose_task
        llm = _CaptureLLM({})          # invalid proposal -> falls back, but the prompt was still built
        propose_task("carry the box across the room", tabletop_arm_gene(), llm=llm,
                     prior_knowledge="PK-BLOCK-PLANNER")
        self.assertTrue(any("PK-BLOCK-PLANNER" in u for u in llm.users))

    def test_species_agent_prepends_prior_knowledge(self):
        from virturoid.services.species_agent import classify_or_create_species
        llm = _CaptureLLM({"decision": "existing", "species_pattern": "manipulator.arm",
                           "robot_class": "manipulator"})
        classify_or_create_species("a tabletop arm", None, [], ["manipulator"], llm,
                                   prior_knowledge="PK-BLOCK-DESIGNER")
        self.assertTrue(any("PK-BLOCK-DESIGNER" in u for u in llm.users))

    def test_gait_critic_prepends_prior_knowledge(self):
        from virturoid.services.gait_critic import critique_gait
        llm = _CaptureLLM({"weights": {}})
        critique_gait({"is_walking": False}, {}, llm, prior_knowledge="PK-BLOCK-GAIT")
        self.assertTrue(any("PK-BLOCK-GAIT" in u for u in llm.users))


if __name__ == "__main__":
    unittest.main()
