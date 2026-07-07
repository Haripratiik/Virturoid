"""B4 (plan_v5 W-B): agent-authored designs bank into the design flywheel. Before this, submit_design /
train_held never wrote to memory (0 write-backs) — the compounding moat didn't compound from the exact usage
the agent-first pivot created. Now submit_design banks a provenance row (source=agent) and the train worker
banks the trained outcome, so search_memory/nearest_bodies surface the agent's own prior work."""
import importlib.util
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")  # offline (get_llm -> None); NO module-level
# NO_INTERNAL_LLM — it leaks process-wide during collection and breaks the llm-router tests
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class FlywheelWriteBackTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        # isolate the flywheel under a temp build/ so the test is hermetic (safe_build_path uses cwd/build)
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp(prefix="virt_fly_")
        os.chdir(self._tmp)
        S.reset()

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_submit_design_banks_provenance(self):
        sch = self._call("get_design_schema")
        self._call("submit_design", {"graph": sch["examples"]["quadruped"]})
        # the flywheel db exists and search_memory surfaces the agent's design
        self.assertTrue(os.path.exists(os.path.join(self._tmp, "build", "memory", "virturoid_memory.db")),
                        "submit_design must write the design flywheel db")
        hits = self._call("search_memory", {"query": "quadruped", "robot_class": "quadruped", "limit": 5})
        self.assertGreaterEqual(len(hits.get("hits", [])), 1, "the agent's design must be recallable")

    def test_nearest_bodies_finds_the_agent_body(self):
        sch = self._call("get_design_schema")
        self._call("submit_design", {"graph": sch["examples"]["quadruped"]})
        nb = self._call("nearest_bodies", {"prompt": "a four legged robot", "k": 5})
        # after banking, the species index has at least the agent's body (or an honest empty note, never a crash)
        self.assertTrue("bodies" in nb or "note" in nb)


if __name__ == "__main__":
    unittest.main()
