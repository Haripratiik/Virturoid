"""Shape-word flywheel (Thesis A × B): a GEOMETRY-verified shape program is banked as a reusable corpus word,
keyed by role + morphology, verdict+config pinned; a future body recalls it. No LLM — the geometry verdict is
the gate. build123d-gated (the realizer is the judge). AGENTS.md offline.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_B123D = importlib.util.find_spec("build123d") is not None

# a real tentacle shape word (tapered core) — the exact program the octopus/mantle grammar would author
_TENTACLE = {"family": "tapered", "length": 0.5, "r0": 0.045, "r1": 0.007}
_MANTLE = {"family": "loft", "sections": [[0.0, 0.16, 0.16], [0.45, 0.20, 0.20], [1.0, 0.07, 0.07]]}


class RoleNormalizerTests(unittest.TestCase):
    """The corpus keys on WHAT a part is, not where it sits — no build123d needed."""

    def test_segment_name_resolves_to_functional_role(self):
        from virturoid.services.shape_flywheel import _segment_role
        for name, role in [("tentacle1_l_0", "tentacle"), ("leg2_r_1", "leg"), ("upper_arm_l", "arm"),
                           ("mantle", "mantle"), ("torso", "torso"), ("head", "head"), ("tail3", "tail")]:
            self.assertEqual(_segment_role(name), role, name)


@unittest.skipUnless(_B123D, "the shape verdict realizes the program with build123d")
class ShapeFlywheelTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="shapefw_")) / "m.db")

    def test_verdict_is_ungameable_and_config_pinned(self):
        from virturoid.services.shape_flywheel import shape_verdict
        good = shape_verdict(_TENTACLE)
        self.assertTrue(good["credible"], f"a real tentacle program must verify: {good}")
        self.assertGreater(good["volume_cm3"], 0.0)
        # the gate REJECTS when the solid is below the volume floor (proves it can say no, not just yes)
        self.assertFalse(shape_verdict(_TENTACLE, min_volume_cm3=1e6)["credible"])

    def test_bank_only_verified_then_recall_the_program(self):
        from virturoid.services.shape_flywheel import _realizer_config, bank_shape, recall_shape, shape_verdict
        db = self._db()
        self.assertIsNone(recall_shape(db, "arm"), "nothing banked yet")
        sid = bank_shape(db, "arm", _TENTACLE)
        self.assertTrue(sid and sid.startswith("shape::arm::"))
        got = recall_shape(db, "arm")
        self.assertIsNotNone(got, "a banked shape word must be recallable")
        self.assertAlmostEqual(got["r0"], _TENTACLE["r0"], places=4)   # the actual program came back
        self.assertTrue(shape_verdict(got)["credible"], "the recalled word still realizes a valid solid")
        # verdict + realizer config are PINNED with the write (reproducible corpus row)
        row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (sid,)).fetchone()
        bc = json.loads(row["base_config"])
        self.assertEqual(bc["sim_config"]["realizer"], _realizer_config()["realizer"])
        self.assertIn("verdict", bc)

    def test_cross_body_recall_by_morphology_embedding(self):
        # the tokenized moat: bank a mantle word on octopus A, recall it for a structurally-similar octopus B
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.shape_flywheel import bank_shape, recall_shape
        db = self._db()
        gA = compose_robot("an octopus robot")
        self.assertTrue(bank_shape(db, "body", _MANTLE, gene=gA))
        gB = compose_robot("an octopus robot")               # a structurally-similar body
        got = recall_shape(db, "body", gene=gB)
        self.assertIsNotNone(got, "a structurally-similar body recalls the banked shape word for the role")
        self.assertEqual(got.get("family"), "loft")

    def test_bank_body_shapes_self_manufactures_the_corpus(self):
        # the ordinary build loop grows the corpus: one octopus contributes a mantle word AND a tentacle word
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.shape_flywheel import bank_body_shapes, recall_shape
        db = self._db()
        g = compose_robot("an octopus robot")
        banked = bank_body_shapes(db, g)
        self.assertGreaterEqual(len(banked), 2, f"a body with many roles banks >=2 shape words: {banked}")
        self.assertEqual(len({s for s in banked}), len(banked), "one representative word per role (deduped)")
        self.assertIsNotNone(recall_shape(db, "mantle", gene=g), "the mantle word is recallable")
        self.assertIsNotNone(recall_shape(db, "tentacle", gene=g), "the tentacle word is recallable")

    def test_grounding_tools_round_trip_and_gate(self):
        # the agent-facing surface: bank an authored word, recall it as grounding, reject a degenerate one — all
        # against an ISOLATED corpus (patch the default DB so the real memory bank is untouched by the test)
        from virturoid.services import shape_flywheel as SF
        from virturoid.services.memory_db import MemoryDB
        tmp = Path(tempfile.mkdtemp(prefix="shapetool_")) / "m.db"
        with mock.patch.object(SF, "_default_db", lambda: MemoryDB(tmp)):
            miss = SF._recall_shape_word({"role": "wing"})
            self.assertFalse(miss["found"], "nothing banked yet -> honest miss")
            ok = SF._bank_shape_word({"role": "wing", "shape_program": _TENTACLE})
            self.assertTrue(ok["banked"] and ok["verdict"]["credible"], ok)
            hit = SF._recall_shape_word({"role": "wing"})
            self.assertTrue(hit["found"], "a banked word grounds the next design")
            self.assertAlmostEqual(hit["shape_program"]["r0"], _TENTACLE["r0"], places=4)
            bad = SF._bank_shape_word({"role": "wing", "shape_program": {"family": "loft", "sections": []}})
            self.assertFalse(bad["banked"], f"a degenerate program is rejected with feedback: {bad}")


class ShapeToolRegistrationTests(unittest.TestCase):
    def test_grounding_tools_are_registered_on_the_agent_surface(self):
        from virturoid.services.agent_tools import TOOLS
        self.assertIn("recall_shape_word", TOOLS)
        self.assertIn("bank_shape_word", TOOLS)


if __name__ == "__main__":
    unittest.main()
