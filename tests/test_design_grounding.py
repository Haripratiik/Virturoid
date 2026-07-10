"""Thesis A end-to-end in the DESIGN FLOW: retrieval = runtime grounding. An agent submits a design carrying a
verified shape; that shape enters the self-manufactured corpus; the NEXT get_design_schema RETRIEVES it as a
physics-verified exemplar; the agent GROUNDS its next design in that exemplar. All through the real MCP dispatch
(call_tool), isolated to a temp corpus, zero product LLM tokens.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_B123D = importlib.util.find_spec("build123d") is not None

# an octopus-ish design: a lofted mantle body + a tapered tentacle pair (the tentacle carries a real shape program)
_OCTO = {"robot_class": "legged", "name": "octo1", "parts": [
    {"name": "mantle", "role": "body", "size": 0.3, "girth": 0.16,
     "geometry": {"family": "loft", "sections": [[0.0, 0.16, 0.16], [0.45, 0.20, 0.20], [1.0, 0.07, 0.07]]}},
    {"name": "tentacle", "role": "tentacle", "parent": "mantle", "attach": "front", "aim": "down_out",
     "size": 0.5, "girth": 0.045, "symmetry": "left_right", "segments": 1,
     "geometry": {"family": "tapered", "length": 0.5, "r0": 0.045, "r1": 0.007}}]}


@unittest.skipUnless(_B123D, "grounding recall realizes shape programs with build123d")
class DesignGroundingLoopTests(unittest.TestCase):
    def test_retrieval_is_runtime_grounding_in_the_design_flow(self):
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        from virturoid.services.shape_flywheel import shape_verdict
        tmp = Path(tempfile.mkdtemp(prefix="grnd_")) / "mem.db"
        with mock.patch("virturoid.services.memory_db.DEFAULT_DB_PATH", tmp):
            S.reset()
            # (a) COLD: nothing banked yet -> the schema offers the static language but NO retrieved exemplars
            sch0 = call_tool("get_design_schema", {"roles": ["tentacle"]})["result"]
            self.assertTrue(sch0.get("ok"))
            self.assertNotIn("corpus_grounding", sch0)

            # (b) an agent SUBMITS a design carrying a verified tentacle shape -> its words enter the corpus
            r1 = call_tool("submit_design", {"graph": _OCTO})["result"]
            self.assertTrue(r1.get("ok"), r1)
            self.assertGreaterEqual(r1.get("corpus_shape_words", 0), 1, "the design's verified shapes must bank")

            # (c) WARM: the NEXT get_design_schema RETRIEVES the verified tentacle exemplar as grounding
            sch1 = call_tool("get_design_schema", {"roles": ["tentacle"]})["result"]
            self.assertIn("corpus_grounding", sch1, "retrieval = runtime grounding: the corpus word must surface")
            ex = sch1["corpus_grounding"]["shape_exemplars"]
            self.assertIn("tentacle", ex)
            self.assertTrue(shape_verdict(ex["tentacle"])["credible"], "the grounding is a physics-VERIFIED exemplar")

            # (d) the agent GROUNDS a new design in the recalled exemplar -> it compiles + passes the validity gate
            g2 = {"robot_class": "legged", "name": "octo2", "parts": [
                {"name": "mantle", "role": "body", "size": 0.32, "girth": 0.17,
                 "geometry": {"family": "loft", "sections": [[0.0, 0.17, 0.17], [0.5, 0.21, 0.21], [1.0, 0.08, 0.08]]}},
                {"name": "tentacle", "role": "tentacle", "parent": "mantle", "attach": "front", "aim": "down_out",
                 "size": 0.5, "girth": 0.045, "symmetry": "left_right", "segments": 1,
                 "geometry": ex["tentacle"]}]}                  # <- the RETRIEVED verified exemplar, adapted
            r2 = call_tool("submit_design", {"graph": g2})["result"]
            self.assertTrue(r2.get("ok"), f"a design grounded in the recalled exemplar must compile + validate: {r2}")


if __name__ == "__main__":
    unittest.main()
