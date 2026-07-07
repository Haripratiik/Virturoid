"""Input Compiler schemas (Phase 0) — field-level provenance records validate and serialize."""

import json
import unittest

from virturoid.schemas.input_bundle import (
    InputArtifact,
    InputBundle,
    InputEvidence,
    InputInterpretation,
    InputSourceType,
    IntakeQuestion,
    ParseStatus,
)


class InputEvidenceTests(unittest.TestCase):
    def test_valid_evidence_ok(self):
        ev = InputEvidence("payload_kg", 3.0, InputSourceType.PARSED, "kg", confidence=0.9)
        self.assertTrue(ev.validate().ok)

    def test_confidence_must_be_in_unit_interval(self):
        self.assertFalse(InputEvidence("x", 1, confidence=1.5).validate().ok)
        self.assertFalse(InputEvidence("x", 1, confidence=-0.1).validate().ok)

    def test_field_path_required(self):
        self.assertFalse(InputEvidence("", 1).validate().ok)


class InputArtifactBundleTests(unittest.TestCase):
    def test_artifact_requires_type(self):
        art = InputArtifact(id="a1", artifact_type="")
        result = art.validate()
        self.assertFalse(result.ok)
        self.assertTrue(any(i.field == "artifact_type" for i in result.issues))

    def test_artifact_rejects_negative_size(self):
        art = InputArtifact(id="a1", artifact_type="urdf", size_bytes=-5)
        self.assertTrue(any(i.code == "invalid_size" for i in art.validate().issues))

    def test_bundle_requires_source_path_and_propagates_artifact_errors(self):
        bad = InputBundle(id="b1", source_path="", artifacts=[InputArtifact(id="a1", artifact_type="")])
        result = bad.validate()
        self.assertFalse(result.ok)
        self.assertTrue(any(i.field == "source_path" for i in result.issues))
        self.assertTrue(any("artifacts[0]" in (i.field or "") for i in result.issues))

    def test_good_bundle_ok(self):
        good = InputBundle(
            id="b1", source_path="/tmp/project",
            artifacts=[InputArtifact(id="a1", artifact_type="prompt", parse_status=ParseStatus.OK)],
        )
        self.assertTrue(good.validate().ok)


class InputInterpretationTests(unittest.TestCase):
    def _interp(self, evidence):
        return InputInterpretation(id="i1", prompt="build a robot", evidence=evidence)

    def test_requires_prompt_and_evidence(self):
        self.assertFalse(InputInterpretation(id="i1", prompt="", evidence=[]).validate().ok)
        self.assertFalse(self._interp([]).validate().ok)  # empty evidence

    def test_duplicate_evidence_field_flagged(self):
        interp = self._interp([
            InputEvidence("payload_kg", 1.0, InputSourceType.PARSED),
            InputEvidence("payload_kg", 2.0, InputSourceType.DEFAULTED),
        ])
        result = interp.validate()
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "duplicate_evidence" for i in result.issues))

    def test_accessors_and_serialization(self):
        interp = self._interp([
            InputEvidence("payload_kg", 3.0, InputSourceType.EXPLICIT, "kg", confidence=1.0),
            InputEvidence("reach_m", 0.65, InputSourceType.DEFAULTED, "m", conflicts=["ambiguous length"]),
        ])
        interp.intake_questions.append(IntakeQuestion(id="q1", field_path="reach_m", question="What reach?"))
        self.assertTrue(interp.validate().ok)
        self.assertEqual(interp.field("payload_kg").source_type, InputSourceType.EXPLICIT)
        self.assertIsNone(interp.field("nonexistent"))
        self.assertEqual(interp.confidence_map()["payload_kg"], 1.0)
        self.assertIn("payload_kg", interp.by_source()["explicit"])
        self.assertTrue(interp.has_conflicts())
        # to_dict() is JSON-serializable and lowers enums to their string value.
        blob = json.dumps(interp.to_dict())
        loaded = json.loads(blob)
        self.assertEqual(loaded["evidence"][0]["source_type"], "explicit")


if __name__ == "__main__":
    unittest.main()
