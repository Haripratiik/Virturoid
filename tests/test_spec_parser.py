"""§4.8E — deterministic parsing + honoring of quantitative prompt constraints (height / weight budget /
payload / pinned parts), and the requested-vs-achieved compliance report. The audit found these were largely
ignored (height not parsed offline; no weight-budget or pinned-part parsing)."""

import unittest

from virturoid.services.spec_parser import (
    measure_gene_height_m, parse_payload_kg, parse_pinned_parts, parse_prompt_spec,
    parse_target_height_m, parse_weight_budget_kg, spec_compliance_report,
)


class SpecParserTests(unittest.TestCase):
    def test_height_parsed_both_orders_and_units(self):
        self.assertAlmostEqual(parse_target_height_m("a 1.2 m tall humanoid"), 1.2)
        self.assertAlmostEqual(parse_target_height_m("a robot that stands 40 cm high"), 0.40)
        self.assertAlmostEqual(parse_target_height_m("height of 1.5 meters"), 1.5)
        # an arm REACH or a payload must NOT be misread as a height (no stature cue)
        self.assertIsNone(parse_target_height_m("an arm with 0.9 m reach"))
        self.assertIsNone(parse_target_height_m("carry a 2 kg box"))

    def test_weight_budget_distinct_from_payload(self):
        self.assertAlmostEqual(parse_weight_budget_kg("a humanoid under 12 kg"), 12.0)
        self.assertAlmostEqual(parse_weight_budget_kg("no heavier than 5 kg"), 5.0)
        self.assertIsNone(parse_weight_budget_kg("carry 5 kg"))            # a payload, not a self-weight budget
        self.assertAlmostEqual(parse_payload_kg("carry 5 kg"), 5.0)
        self.assertAlmostEqual(parse_payload_kg("lift a 250 g tool"), 0.25)

    def test_pinned_parts(self):
        got = parse_pinned_parts("a rover with a lidar and a parallel gripper")
        self.assertIn("lidar", got)
        self.assertIn("parallel gripper", got)
        self.assertEqual(parse_pinned_parts("a plain wooden block"), [])

    def test_parse_prompt_spec_combines_all(self):
        s = parse_prompt_spec("a 1.4 m tall humanoid under 15 kg that can carry 3 kg with a depth camera")
        self.assertAlmostEqual(s.target_height_m, 1.4)
        self.assertAlmostEqual(s.weight_budget_kg, 15.0)
        self.assertAlmostEqual(s.payload_kg, 3.0)
        self.assertIn("depth camera", s.pinned_parts)
        self.assertTrue(s.any())


class HeightHonoredTests(unittest.TestCase):
    def test_humanoid_height_scales_with_request_offline(self):
        from virturoid.services.morphology_composer import compose_robot
        short = measure_gene_height_m(compose_robot("a 1.0 m tall humanoid", llm=None))
        tall = measure_gene_height_m(compose_robot("a 2.0 m tall humanoid", llm=None))
        self.assertIsNotNone(short)
        self.assertIsNotNone(tall)
        self.assertLess(abs(short - 1.0), 0.15)        # built ~1.0 m
        self.assertLess(abs(tall - 2.0), 0.25)         # built ~2.0 m
        self.assertGreater(tall, short * 1.6)          # a 2x-taller request => a clearly taller robot

    def test_default_humanoid_unchanged(self):
        # no height in the prompt -> the head-canon default stature; not perturbed by the new path
        from virturoid.services.morphology_composer import compose_robot
        h = measure_gene_height_m(compose_robot("a humanoid robot", llm=None))
        self.assertIsNotNone(h)
        self.assertGreater(h, 1.4)
        self.assertLess(h, 2.0)

    def test_compliance_report_requested_vs_achieved(self):
        from virturoid.services.morphology_composer import compose_robot
        prompt = "a 1.2 m tall humanoid under 60 kg"
        spec = parse_prompt_spec(prompt)
        g = compose_robot(prompt, llm=None)
        rep = spec_compliance_report(g, spec)
        by = {c["constraint"]: c for c in rep["constraints"]}
        self.assertIn("height_m", by)
        self.assertTrue(by["height_m"]["honored"])             # built to the requested height (within tolerance)
        self.assertIn("self_weight_kg", by)                    # weight budget reported (achieved vs requested_max)


if __name__ == "__main__":
    unittest.main()
