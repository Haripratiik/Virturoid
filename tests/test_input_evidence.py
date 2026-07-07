"""Prompt Input Compiler (Phase 0) — merged, provenance-tracked interpretation of a build request.

Acceptance criteria from the Input Ingestion plan:
  * a vague prompt and a constrained prompt both produce one ``input/interpretation.json``;
  * every interpreted field carries a source type (explicit / parsed / inferred / defaulted);
  * parser disagreements (a self-weight budget read as a payload; a stature read as an arm reach) become
    INFERRED evidence with a conflict + a targeted intake question, not a silent wrong value;
  * a compliance report can be linked back to the input evidence.
"""

import json
import os
import tempfile
import unittest

from virturoid.schemas.input_bundle import InputSourceType
from virturoid.services.input_evidence import (
    interpret_prompt,
    link_compliance_to_evidence,
    write_interpretation,
)


class InterpretPromptTests(unittest.TestCase):
    def test_vague_prompt_defaults_and_validates(self):
        interp = interpret_prompt("build me a robot that can walk")
        self.assertTrue(interp.validate().ok)
        self.assertEqual(interp.field("payload_kg").source_type, InputSourceType.DEFAULTED)
        self.assertEqual(interp.field("reach_m").source_type, InputSourceType.DEFAULTED)
        self.assertFalse(interp.has_conflicts())
        self.assertEqual(interp.intake_questions, [])

    def test_constrained_prompt_parses_quantitative_fields(self):
        interp = interpret_prompt(
            "a 1.2 m tall humanoid under 15 kg that can carry 3 kg with a depth camera"
        )
        self.assertTrue(interp.validate().ok)
        self.assertEqual(interp.field("target_height_m").value, 1.2)
        self.assertEqual(interp.field("target_height_m").source_type, InputSourceType.PARSED)
        self.assertEqual(interp.field("weight_budget_kg").value, 15.0)
        self.assertEqual(interp.field("payload_kg").value, 3.0)
        self.assertEqual(interp.field("payload_kg").source_type, InputSourceType.PARSED)
        self.assertIn("depth camera", interp.field("pinned_parts").value)
        self.assertEqual(interp.field("sensor").source_type, InputSourceType.PARSED)

    def test_weight_budget_not_misreported_as_payload(self):
        # "under 12 kg" is a self-weight budget; there is NO carry payload -> conflict + question, not a silent 12.
        interp = interpret_prompt("a 1.2 m tall humanoid under 12 kg")
        payload = interp.field("payload_kg")
        self.assertEqual(payload.source_type, InputSourceType.INFERRED)
        self.assertTrue(payload.conflicts)
        self.assertTrue(any(q.field_path == "payload_kg" for q in interp.intake_questions))

    def test_stature_not_misreported_as_reach(self):
        interp = interpret_prompt("a 1.2 m tall humanoid under 12 kg")
        reach = interp.field("reach_m")
        self.assertEqual(reach.source_type, InputSourceType.INFERRED)
        self.assertTrue(reach.conflicts)
        self.assertTrue(any(q.field_path == "reach_m" for q in interp.intake_questions))

    def test_explicit_arguments_recorded_as_explicit(self):
        interp = interpret_prompt("a robot arm", payload_kg=3.0, reach_m=0.9, sensor="lidar")
        self.assertEqual(interp.field("payload_kg").source_type, InputSourceType.EXPLICIT)
        self.assertEqual(interp.field("payload_kg").value, 3.0)
        self.assertEqual(interp.field("reach_m").source_type, InputSourceType.EXPLICIT)
        self.assertEqual(interp.field("reach_m").value, 0.9)
        self.assertEqual(interp.field("sensor").source_type, InputSourceType.EXPLICIT)
        self.assertEqual(interp.field("sensor").value, "lidar")


class WriteAndLinkTests(unittest.TestCase):
    def test_both_ends_write_interpretation_json(self):
        for prompt in ("build me a walking robot", "a 1.4 m humanoid under 20 kg carrying 3 kg with rgbd"):
            with tempfile.TemporaryDirectory() as tmp:
                interp = interpret_prompt(prompt)
                path = write_interpretation(interp, tmp)
                self.assertEqual(path, os.path.join(tmp, "input", "interpretation.json"))
                self.assertTrue(os.path.exists(path))
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                self.assertEqual(data["prompt"], interp.prompt)
                self.assertTrue(data["evidence"])
                # provenance is preserved on disk as plain strings
                self.assertIn(data["evidence"][0]["source_type"],
                              {s.value for s in InputSourceType})

    def test_compliance_report_links_to_evidence(self):
        interp = interpret_prompt("a 1.2 m tall humanoid that can carry 3 kg")
        report = {
            "constraints": [
                {"constraint": "height_m", "requested": 1.2, "honored": True},
                {"constraint": "payload_kg", "requested": 3.0, "honored": True},
            ],
            "all_honored": True,
        }
        linked = link_compliance_to_evidence(interp, report)
        by = {c["constraint"]: c for c in linked["constraints"]}
        self.assertEqual(by["height_m"]["evidence"]["source_type"], "parsed")
        self.assertEqual(by["payload_kg"]["evidence"]["source_type"], "parsed")
        # original report is not mutated
        self.assertNotIn("evidence", report["constraints"][0])


if __name__ == "__main__":
    unittest.main()
