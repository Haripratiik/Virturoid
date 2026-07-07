"""T5 (total_generality_plan): the general TASK layer reaches the AGENT surface. Before this, TaskSpec/
verifier/executor existed across 12 files with ZERO agent references — evaluate_held only scored a fixed
morphology task. Now: list_skills (vocabulary), run_task (goal -> plan -> VERIFY vs morphology -> run ->
score), submit_task (agent authors the skill sequence). Headline win = honest INFEASIBILITY on a mismatch,
never a bogus run. Offline + NO_INTERNAL_LLM -> deterministic + LLM-free (heuristic planner)."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")  # offline (get_llm -> None); NO module-level
# NO_INTERNAL_LLM — it leaks process-wide during collection and breaks the llm-router tests
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class AgentTaskToolTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def _robot(self, prompt):
        return self._call("create_robot", {"prompt": prompt})["robot_id"]

    def test_list_skills_is_the_vocabulary(self):
        sk = self._call("list_skills")
        names = [s["skill"] for s in sk["skills"]]
        self.assertIn("locomote", names)
        self.assertIn("navigate", names)
        self.assertIn("traveled", sk["predicate_ops"])

    def test_run_task_general_goal_on_a_capable_body(self):
        rid = self._robot("a quadruped robot dog")
        r = self._call("run_task", {"robot_id": rid, "goal": "walk forward across the room"})
        self.assertTrue(r["feasible"])
        self.assertIn("locomote", r["planned_skills"])          # planned the right skill from the goal

    def test_run_task_is_honest_about_a_morphology_mismatch(self):
        rid = self._robot("a robot arm that sorts blocks")
        r = self._call("run_task", {"robot_id": rid, "goal": "navigate through a maze"})
        self.assertFalse(r["feasible"], "an arm cannot navigate a maze -> infeasible, not a bogus run")
        self.assertTrue(r["issues"])

    def test_submit_task_agent_authored_sequence(self):
        rid = self._robot("a quadruped robot dog")
        r = self._call("submit_task", {"robot_id": rid,
                                        "steps": [{"skill": "locomote", "params": {"prompt": "walk"}}],
                                        "goal": [{"op": "traveled", "args": [0.3]}]})
        self.assertTrue(r["ok"] and r["feasible"])
        self.assertIn("score", r)

    def test_submit_task_unknown_skill_teaches(self):
        rid = self._robot("a quadruped robot dog")
        r = self._call("submit_task", {"robot_id": rid, "steps": [{"skill": "teleport"}],
                                       "goal": [{"op": "traveled", "args": [1]}]})
        self.assertFalse(r["ok"])
        self.assertIn("teleport", r["error"])

    def test_run_task_is_in_the_mcp_view(self):
        from virturoid.services.agent_tools import tool_specs
        names = [t["name"] for t in tool_specs(view="mcp")]
        self.assertIn("run_task", names)
        self.assertLessEqual(len(names), 16)


if __name__ == "__main__":
    unittest.main()
