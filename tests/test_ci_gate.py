"""The CI gate must not be able to empty itself.

``scripts/ci_gate.py`` is the list of tests CI enforces before a merge. Its predecessor was a list of paths in
a YAML file, and the failure mode that list had is the one these tests close: a path list cannot tell you it
has stopped matching anything. Rename ``tests/test_export_gate.py`` and the YAML keeps a green check while
gating on nothing.

So three things are asserted here, and this file is itself in the FAST tier so the gate protects its own
definition:

  * every enforced path exists -- a rename fails the suite, not just CI;
  * every gate carries a ``failure`` naming a real incident, because a gate that cannot name one is noise and
    should be deleted deliberately rather than kept out of superstition;
  * the audit that decides "green" actually rejects the two things it exists to reject -- a file that
    contributed no executed test, and a test skipped because the environment could not run it.

The last one matters most. Measured 2026-08-07 against a runner with the Menagerie cache hidden, pytest
reported "14 passed, 83 skipped" -- a green run -- for a tier whose entire purpose is the corpus.
"""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE = runpy.run_path(str(REPO_ROOT / "scripts" / "ci_gate.py"))


def _junit(cases: str) -> str:
    return f'<?xml version="1.0"?><testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>'


class CiGateDefinitionTests(unittest.TestCase):
    def test_every_enforced_test_file_still_exists(self):
        gone = GATE["missing"](GATE["ALL"])
        self.assertEqual(gone, [], f"scripts/ci_gate.py enforces files that no longer exist: {gone}. "
                                   "A rename must update the gate, or the gate stops gating.")

    def test_every_gate_names_the_incident_it_exists_to_catch(self):
        for g in GATE["ALL"]:
            self.assertGreater(len(g.failure), 60,
                               f"{g.path} has no substantive justification; a gate that cannot name a real "
                               f"failure is noise")

    def test_the_tiers_do_not_overlap_and_all_is_their_union(self):
        fast = {g.path for g in GATE["FAST"]}
        corpus = {g.path for g in GATE["CORPUS"]}
        self.assertEqual(fast & corpus, set(), "a file in both tiers would be run twice and floored twice")
        self.assertEqual(fast | corpus, {g.path for g in GATE["ALL"]})

    def test_the_uncovered_count_is_real_and_reported(self):
        """The gate's own honesty claim: it says how much of the suite it does NOT cover."""
        uncovered = GATE["uncovered_files"]()
        all_files = sorted(f"tests/{p.name}" for p in (REPO_ROOT / "tests").glob("test_*.py"))
        self.assertEqual(len(uncovered) + len(GATE["ALL"]), len(all_files))
        self.assertGreater(len(uncovered), 0, "if this ever hits zero, say so loudly -- it would be news")


class CiGateAuditTests(unittest.TestCase):
    """The audit is what makes the tier's green mean something. These are its two refusals."""

    def setUp(self):
        self.gate = GATE["Gate"]("tests/test_moat_panel.py", failure="x" * 80)
        self.gates = (self.gate,)

    def _audit(self, xml: str, floor: int = 1):
        import tempfile
        p = Path(tempfile.mkdtemp()) / "r.xml"
        p.write_text(xml, encoding="utf-8")
        return GATE["_audit"](p, self.gates, floor)

    def test_a_clean_run_produces_no_problems(self):
        cases = '<testcase classname="tests.test_moat_panel" name="a" time="0.1" />' * 3
        self.assertEqual(self._audit(_junit(cases), floor=3), [])

    def test_a_gate_that_ran_nothing_is_reported_as_empty_not_as_passing(self):
        cases = '<testcase classname="tests.test_something_else" name="a" time="0.1" />'
        problems = self._audit(_junit(cases))
        self.assertTrue(any("ZERO executed tests" in p for p in problems), problems)

    def test_an_environment_skip_fails_the_tier_that_claimed_to_cover_it(self):
        """The 'green on a subset' refusal. A skipped corpus test is not a passing corpus test."""
        cases = ('<testcase classname="tests.test_moat_panel" name="a">'
                 '<skipped message="aloha/aloha.xml is not in the local Menagerie cache" /></testcase>')
        problems = self._audit(_junit(cases))
        self.assertTrue(any("ENVIRONMENT could not run them" in p for p in problems), problems)

    def test_a_deliberate_opt_out_skip_is_not_treated_as_an_environment_failure(self):
        """The rule must not fire on a test that CHOSE to skip -- otherwise it gets disabled as noisy."""
        cases = ('<testcase classname="tests.test_moat_panel" name="a">'
                 '<skipped message="this build did not bake a source mesh for torso_link" /></testcase>'
                 '<testcase classname="tests.test_moat_panel" name="b" time="0.1" />')
        problems = self._audit(_junit(cases), floor=1)
        self.assertEqual(problems, [])

    def test_falling_under_the_recorded_floor_is_a_failure(self):
        cases = '<testcase classname="tests.test_moat_panel" name="a" time="0.1" />'
        problems = self._audit(_junit(cases), floor=50)
        self.assertTrue(any("recorded floor" in p for p in problems), problems)

    def test_the_file_attribute_form_is_matched_as_well_as_the_classname_form(self):
        """pytest emits one or the other depending on version and test style. Matching only whole strings
        reported every gate empty -- observed while building this script."""
        cases = '<testcase file="tests/test_moat_panel.py" classname="TopLevel" name="a" time="0.1" />'
        self.assertEqual(self._audit(_junit(cases)), [])


if __name__ == "__main__":
    unittest.main()
