"""Engineer mode (H5 + orchestration): retrieved morphology-keyed memory reaches the Proposer's prompt, and
the harness runs end-to-end for one body+task with an injected evaluator. Fake-LLM + seeded memory, no MuJoCo."""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.services.engineer_mode import run_engineer_search
from virturoid.services.memory_db import MemoryDB
from virturoid.services.species_discovery import auto_place_species


class _FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, schema, max_tokens=None):
        props = schema.get("properties", {})
        if "proposals" in props:
            self.calls.append(("proposer", user))
            return {"proposals": [{"edit_kind": "gains", "params": {"kp": 40.0}}]}
        if "viable" in props:
            return {"viable": True, "reason": "ok"}
        return {"edit_kind": "gains", "params": {"kp": 40.0}}


def _evaluate(spec):
    # a manipulation eval double: kp>=40 grasps, else slips
    sr = 0.8 if spec.get("params", {}).get("kp", 0) >= 40 else 0.2
    return {"success_rate": sr, "contacted": True, "lifted": sr > 0.5}


class EngineerModeTests(unittest.TestCase):
    def test_memory_block_reaches_the_proposer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryDB(Path(tmp) / "virturoid_memory.db") as db:
                sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
                db.add_species_tip(sp, "keep the wrist DOF wide for top-down grasps", audience="builder")
            llm = _FakeLLM()
            rep = run_engineer_search(task="grasp a cube", gene=tabletop_arm_gene(), evaluate=_evaluate,
                                      llm=llm, memory_dir=tmp, task_type="grasp", max_evals=3)
            self.assertTrue(rep.solved)                                  # kp=40 -> success 0.8 passes
            self.assertTrue(any("wrist DOF" in u for _, u in llm.calls))  # the seeded tip reached the prompt

    def test_offline_uses_heuristic_and_still_searches(self):
        # auto_llm=False forces the LLM-free path regardless of any VIRTUROID_LLM_BACKEND in the env (deterministic)
        rep = run_engineer_search(task="grasp", gene=tabletop_arm_gene(), evaluate=_evaluate, llm=None,
                                  auto_llm=False, task_type="grasp", max_evals=3,
                                  heuristic=lambda p, h: {"edit_kind": "gains", "params": {"kp": 45.0}})
        self.assertTrue(rep.solved)                                     # heuristic proposes kp=45 -> passes

    def test_auto_llm_wires_a_configured_backend(self):
        # brain-default-on (gap-audit G1): when llm is None + auto_llm, the harness resolves get_llm() and USES it.
        import unittest.mock as mock
        llm = _FakeLLM()
        with mock.patch("virturoid.services.llm_client.get_llm", return_value=llm):
            rep = run_engineer_search(task="grasp a cube", gene=tabletop_arm_gene(), evaluate=_evaluate,
                                      llm=None, auto_llm=True, task_type="grasp", max_evals=3)
        self.assertTrue(any(kind == "proposer" for kind, _ in llm.calls))   # the resolved backend drove the Proposer
        self.assertTrue(rep.solved)

    def test_no_memory_is_graceful(self):
        rep = run_engineer_search(task="grasp", gene=tabletop_arm_gene(), evaluate=_evaluate, llm=_FakeLLM(),
                                  task_type="grasp", max_evals=2)       # no memory_dir -> empty block, still runs
        self.assertIsNotNone(rep.best)


if __name__ == "__main__":
    unittest.main()
