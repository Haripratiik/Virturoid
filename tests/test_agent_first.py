"""Agent-first tools (docs/agent_first_plan.md P1: G-A + G-C) — the tools that let an EXTERNAL agent be the
whole brain against our substrate with ZERO internal LLM spend: it AUTHORS the anatomy graph itself
(submit_design) instead of prompting our generator, then drives the full loop on THE held gene
(evaluate_held / train_held / export_held) instead of recomposing. The headline test walks the entire
6-step canonical loop through ``call_tool`` JSON ONLY (no internal imports) — the proof an MCP agent can do
it. Offline (no local env) so it is deterministic AND provably LLM-free (get_llm returns None).
"""
import importlib.util
import os
import time
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class DesignSubmissionTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_schema_examples_actually_compile(self):
        # the language we TEACH the agent must be real: its worked examples must submit + hold successfully.
        sch = self._call("get_design_schema")
        for key in ("quadruped", "hexapod"):
            r = self._call("submit_design", {"graph": sch["examples"][key]})
            self.assertTrue(r.get("ok"), f"{key} example must compile: {r.get('error')}")
            self.assertIn("robot_id", r)

    def test_agent_authored_design_holds_and_discovers(self):
        # the agent submits ITS OWN graph (not a prompt); GEN-1 reads the leg count from the compiled body.
        sch = self._call("get_design_schema")
        r = self._call("submit_design", {"graph": sch["examples"]["hexapod"]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["appendages"]["legs"], 6)

    def test_broken_graph_teaches(self):
        r = self._call("submit_design", {"graph": {"robot_class": "quadruped", "parts": [{"name": "x", "role": "leg"}]}})
        self.assertFalse(r["ok"])
        self.assertIn("body", r["error"])                     # tells the agent it needs a body root

    def test_submit_scene_spec(self):
        objs = [{"name": "floor", "object_type": "floor", "category": "floor", "size_xyz": [4, 4, 0.05],
                 "pose_xyz_rpy": [0, 0, -0.025, 0, 0, 0]},
                {"name": "box", "category": "obstacle", "size_xyz": [0.4, 0.4, 0.4], "pose_xyz_rpy": [1, 0, 0.2, 0, 0, 0]}]
        r = self._call("submit_scene_spec", {"objects": objs, "task": "navigation"})
        self.assertTrue(r["ok"])
        self.assertIn("scene_id", r)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class HeldChainTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_evaluate_and_export_the_held_gene(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        ev = self._call("evaluate_held", {"robot_id": rid})
        self.assertEqual(ev["task"], "locomotion")
        ex = self._call("export_held", {"robot_id": rid, "formats": ["mjcf"]})
        self.assertTrue(ex["ok"])
        self.assertTrue(os.path.exists(ex["artifacts"]["mjcf"]), "MJCF must be a real file on disk")

    def test_train_held_job_completes_with_verdict(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        jid = self._call("train_held", {"robot_id": rid, "mode": "gait_search", "max_evals": 3})["job_id"]
        self.assertIsNotNone(jid)
        for _ in range(60):
            j = self._call("get_job", {"job_id": jid})
            if j["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(2)
        self.assertEqual(j["status"], "succeeded")
        self.assertEqual(j["result"]["mode"], "gait_search")
        self.assertIn("best", j["result"])


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class FullAgentLoopTests(unittest.TestCase):
    """The proof: an external agent completes the 6-step canonical loop through call_tool JSON ONLY, with the
    internal LLM OFF (no local env -> get_llm None). This is what a Claude Code / Codex MCP session does."""

    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def test_design_to_export_no_internal_llm(self):
        from virturoid.services.agent_tools import call_tool

        def T(name, args):
            r = call_tool(name, args)
            self.assertTrue(r["ok"], f"{name} dispatch failed: {r.get('error')}")
            return r["result"]

        # confirm the internal LLM is OFF, so this proves zero-our-tokens autonomy
        from virturoid.services.llm_client import get_llm
        self.assertIsNone(get_llm("morphology"), "test must run with the internal LLM off")

        schema = T("get_design_schema", {})                                    # 1. learn the language
        design = T("submit_design", {"graph": schema["examples"]["quadruped"]})  # 2. AUTHOR (design)
        self.assertTrue(design["ok"]); rid = design["robot_id"]
        T("simulate_gait", {"robot_id": rid, "steps": 400})                    # 3. simulate (compile+sim)
        ev = T("evaluate_held", {"robot_id": rid})                             # 4. evaluate/diagnose
        self.assertEqual(ev["task"], "locomotion")
        edit = T("edit_robot", {"robot_id": rid,                               # 5. edit (localized)
                                "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "girth", "factor": 1.15}}]})
        self.assertTrue(edit["ok"])
        exp = T("export_held", {"robot_id": rid, "formats": ["mjcf"]})         # 6. export
        self.assertTrue(exp["ok"] and os.path.exists(exp["artifacts"]["mjcf"]))


if __name__ == "__main__":
    unittest.main()
