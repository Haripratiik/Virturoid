"""W1: the Prompt Input Compiler (Phase 0) is WIRED into the package writer.

Locks the Input-plan Phase-0 acceptance criteria against regression:
  * every prompt build writes ``input/interpretation.json`` (vague AND constrained prompts);
  * the requested-vs-achieved spec-compliance report points back to the input evidence.

Offline + no internal LLM (AGENTS.md); needs MuJoCo only because the package writer compiles the gene.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo to compile the gene")
class InterpretationWiringTests(unittest.TestCase):
    def _build(self, prompt: str) -> Path:
        from virturoid.services.gene_build import build_gene_package
        from virturoid.services.morphology_composer import compose_robot

        out = Path(tempfile.mkdtemp(prefix="vpkg_wire_"))
        build_gene_package(compose_robot(prompt), prompt, out, scene_count=2)
        return out

    def test_vague_prompt_writes_interpretation(self):
        out = self._build("build me a robot that can walk")
        interp = out / "input" / "interpretation.json"
        self.assertTrue(interp.exists(), "every build must write input/interpretation.json")
        data = json.loads(interp.read_text(encoding="utf-8"))
        self.assertEqual(data["prompt"], "build me a robot that can walk")
        fields = {e["field_path"] for e in data["evidence"]}
        self.assertIn("task_summary", fields)
        # nothing quantitative was stated -> payload/reach must be DEFAULTED, not fabricated as explicit.
        by_field = {e["field_path"]: e["source_type"] for e in data["evidence"]}
        self.assertEqual(by_field.get("payload_kg"), "defaulted")

    def test_constrained_prompt_links_compliance_to_evidence(self):
        out = self._build("a 1.2 m tall humanoid that carries 3 kg")
        interp = json.loads((out / "input" / "interpretation.json").read_text(encoding="utf-8"))
        by_field = {e["field_path"]: e["source_type"] for e in interp["evidence"]}
        self.assertEqual(by_field.get("payload_kg"), "parsed")          # 3 kg was actually stated
        self.assertIn("target_height_m", by_field)                      # 1.2 m stature captured

        comp_path = out / "reports" / "spec_compliance.json"
        self.assertTrue(comp_path.exists())
        comp = json.loads(comp_path.read_text(encoding="utf-8"))
        constraints = {c["constraint"]: c for c in comp.get("constraints", [])}
        self.assertTrue(constraints, "a constrained prompt must produce compliance constraints")
        # acceptance: compliance points back to input evidence.
        for name in ("height_m", "payload_kg"):
            if name in constraints:
                self.assertIn("evidence", constraints[name])
                self.assertIn("source_type", constraints[name]["evidence"])


if __name__ == "__main__":
    unittest.main()
