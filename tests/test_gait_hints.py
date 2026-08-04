"""The flywheel-as-HINTS (not copy-paste): hints are auto-mined from banked credible walks (data-derived, no
hardcoding), and warm-start a fresh per-body adaptation — so two different bodies get two different fitted gaits
from the same hints. The moat is transferable principles + adaptation, never a pasted policy.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _bank(db, cls, gene_id, params, sr, forward=None):
    from virturoid.services.gait_flywheel import LOCOMOTION
    bc = {"gait_params": params, "controller": "crawl_gait"}
    if forward is not None:
        bc["forward_m"] = float(forward)      # the OUTCOME a mined region must be tested against
    db.record_skill(f"gait::{cls}::{gene_id}", cls, LOCOMOTION, success_rate=sr, base_config=bc)


def _bank_vec(db, gene, params, *, forward=1.0):
    """Bank a CREDIBLE gait THROUGH the real flywheel (bank_gait) so it is indexed into the robotics vector
    memory by THIS body's morphology embedding — the path a future body borrows from by structural similarity."""
    import types
    from virturoid.services.gait_flywheel import bank_gait
    r = types.SimpleNamespace(best_survived=True, best_forward=float(forward), best_credible=True,
                              best_params=dict(params), best_height_ratio=0.8)
    return bank_gait(db, gene, r)


class GaitHintsMiningTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="hints_")) / "m.db")

    def test_hints_are_mined_from_data_not_hardcoded(self):
        from virturoid.services.gait_hints import mine_gait_hints
        db = self._db()
        # thin corpus -> honest "not enough data" (the default prior, no invented region)
        cold = mine_gait_hints(db, robot_class="quadruped")
        self.assertEqual(cold["n"], 0)
        self.assertIn("not enough", cold["note"])
        # bank a few CREDIBLE walks whose freq clusters ~2.0 (deliberately NOT the 1.5 default) -> the mined region
        # must follow the DATA to ~2.0, proving it isn't a hardcoded constant
        for i, f in enumerate((1.9, 2.0, 2.1)):
            _bank(db, "quadruped", f"b{i}", {"freq": f, "hip_amp": 0.8, "knee_amp": 1.1, "duty": 0.3,
                                             "kp": 30.0, "kd": 1.4}, 0.9)
        h = mine_gait_hints(db, robot_class="quadruped")
        self.assertEqual(h["n"], 3)
        self.assertAlmostEqual(h["prior"]["freq"], 2.0, delta=0.15)   # region tracked the data, not the 1.5 default
        # ...but 3 walks with no recorded travel CANNOT establish a param region, so none is CLAIMED (#266).
        # The prior stays available as an explicitly un-tuned warm-start seed; a hint is a claim about evidence.
        self.assertFalse([x for x in h["hints"] if x.get("kind") == "param_region"], h["hints"])
        # the relational hint is DISCOVERED (all 3 have knee_amp > hip_amp)
        self.assertTrue(any(hint.get("kind") == "relation" for hint in h["hints"]))

    def test_new_body_borrows_hints_from_its_VECTOR_nearest_robot(self):
        """THE moat the user asked for: a brand-new body, with NOTHING banked under its own class string, still
        borrows gait hints from the robot it is SHAPED like in the robotics vector space — and the nearer body's
        params pull the prior harder (similarity-weighted). Proves real embedding transfer, not a class match."""
        from virturoid.services.gait_hints import mine_gait_hints
        from virturoid.services.morphology_composer import compose_robot
        db = self._db()
        hexa = compose_robot("a six-legged hexapod robot")
        quad = compose_robot("a small quadruped robot dog")
        # bank a credible gait for each REAL body -> each indexed by its OWN morphology embedding.
        # the hexapod's walk clusters freq ~2.2; the quad's ~1.2 (deliberately far apart so weighting is visible).
        self.assertIsNotNone(_bank_vec(db, hexa, {"freq": 2.2, "hip_amp": 0.7, "knee_amp": 1.2, "duty": 0.3,
                                                  "kp": 30.0, "kd": 1.4}))
        self.assertIsNotNone(_bank_vec(db, quad, {"freq": 1.2, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25,
                                                  "kp": 32.0, "kd": 1.5}))
        # a FRESH hexapod (never banked) asks for hints -> it must SOURCE them from the vector index, and the
        # hexapod neighbor (nearer in morphology) must dominate the quad -> prior.freq lands nearer 2.2 than 1.2.
        newbie = compose_robot("a fresh six-legged walking robot")
        h = mine_gait_hints(db, gene=newbie)
        self.assertEqual(h["source"], "vector_nearest")             # borrowed via the embedding, NOT a class string
        self.assertGreaterEqual(h["n"], 1)
        self.assertGreater(h["prior"]["freq"], 1.7,                 # pulled toward the SHAPE-similar hexapod (2.2)
                           f"expected the morphology-nearest (hexapod) walk to dominate, got {h['prior']}")


class MinedRegionsMustBeEarnedTests(unittest.TestCase):
    """#266 — a ``param_region`` hint is shown to the operator as EVIDENCE, so it must survive three gates.

    Before this, ``mine_gait_hints`` emitted one for EVERY key in ``_PARAM_KEYS`` with >=2 observations, in
    identical confident wording, with no dispersion test, no association test and no comparison against the
    prior the values were drawn from. Measured on the real 90-walk bank it announced "nearby walkers cluster
    kp near 56.61 (67 robots)" over an advertised cluster of [20.0, 240.0] — a band that SPANS AND EXCEEDS the
    whole [24, 240] search range. The removed ``duty`` was the extreme case, not the mechanism.
    """
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="hints_gate_")) / "m.db")

    @staticmethod
    def _regions(h):
        return {x["param"]: x for x in h["hints"] if x.get("kind") == "param_region"}

    def test_no_region_hint_for_a_parameter_that_spans_its_whole_search_range(self):
        """THE defect, in one assertion: kp swept across the FULL [24, 240] search range and uncorrelated with
        travel is not a cluster — it is the search range itself, restated as advice."""
        from virturoid.services.gait_hints import mine_gait_hints
        db = self._db()
        n = 40
        for i in range(n):
            kp = 24.0 + (240.0 - 24.0) * i / (n - 1)              # the ENTIRE search range, end to end
            fwd = 0.6 + 1.4 * ((i * 7) % n) / (n - 1)             # travel: unrelated to kp
            _bank(db, "quadruped", f"span{i}", {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0,
                                                "kp": kp, "kd": 1.0 + 13.0 * ((i * 3) % n) / (n - 1)},
                  0.9, forward=fwd)
        h = mine_gait_hints(db, robot_class="quadruped")
        self.assertEqual(h["n"], n)
        self.assertNotIn("kp", self._regions(h),
                         f"kp spans 100% of its search range and predicts nothing — no region to claim: {h['hints']}")
        # and the operator is TOLD why there is no hint, rather than shown silence
        self.assertTrue(any(s.get("param") == "kp" for s in h.get("suppressed", [])), h.get("suppressed"))

    def test_a_real_parameter_survives_the_gates_and_a_null_one_does_not(self):
        """The generator must DISCRIMINATE. Same bank, same sample size, same selection machinery: ``freq`` has a
        genuine operating band (walkers inside it travel far, outside it barely move); ``kd`` is swept across its
        range with no bearing on travel. One hint must survive, the other must not."""
        from virturoid.services.gait_hints import mine_gait_hints
        db = self._db()
        n = 40
        for i in range(n):
            if i % 2 == 0:                                        # REAL signal: freq in a tight band -> walks far
                freq, fwd = 1.45 + 0.30 * (i / (n - 1)), 2.0 + 0.5 * (i / (n - 1))
            else:                                                 # far from the band -> barely moves
                freq, fwd = 0.8 + 2.4 * (i / (n - 1)), 0.4 + 0.2 * (i / (n - 1))
            kd = 1.0 + 13.0 * ((i * 11) % n) / (n - 1)            # NULL: swept, unrelated to travel
            _bank(db, "quadruped", f"mix{i}", {"freq": freq, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0,
                                               "kd": kd}, 0.9, forward=fwd)
        h = mine_gait_hints(db, robot_class="quadruped")
        reg = self._regions(h)
        self.assertIn("freq", reg, f"a parameter with a real, outcome-linked band must still be minable: {h}")
        self.assertNotIn("kd", reg, f"kd predicts nothing about travel — it must not be advised on: {h['hints']}")
        # and it RECOVERS the planted band [1.45, 1.75] rather than merely existing — a gate that only ever
        # says "no" would pass every negative test in this class while being useless
        self.assertGreaterEqual(reg["freq"]["range"][0], 1.40, reg["freq"])
        self.assertLessEqual(reg["freq"]["range"][1], 1.80, reg["freq"])
        # the surviving hint is FALSIFIABLE (the `relation` template's shape): it states a proportion and the
        # odds the same pattern arises from selection alone, not a bare "walkers cluster X near Y"
        note = reg["freq"]["note"]
        self.assertIn("%", note, note)
        self.assertLessEqual(reg["freq"]["spread_vs_prior"], 0.5, reg["freq"])
        self.assertLess(reg["freq"]["p_association"], 0.05, reg["freq"])
        self.assertLess(reg["freq"]["p_vs_winner_null"], 0.05, reg["freq"])

    def test_a_tight_cluster_with_no_outcome_link_is_still_refused(self):
        """GATE 3, the subtle one. Banked values are CEM WINNERS, and elite selection shrinks a coordinate's
        variance whether or not it has signal — fastest where it has NONE. So "tight" is not evidence on its own:
        a band far tighter than the sampling prior, but no tighter than random same-size groups of the banked
        winners and unlinked to travel, must be refused. (Measured: the removed ``duty`` sat at 13% of a uniform
        draw — TIGHTER than every real parameter — precisely because nothing pushed back on it.)"""
        from virturoid.services.gait_hints import mine_gait_hints
        db = self._db()
        n = 40
        for i in range(n):
            kd = 1.40 + 0.20 * (i / (n - 1))                      # 1.5% of the [1, 14] range: a very tight band
            _bank(db, "quadruped", f"tight{i}", {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0,
                                                 "kd": kd}, 0.9,
                  forward=0.5 + 1.5 * ((i * 13) % n) / (n - 1))   # travel: unrelated to where in the band it sat
        h = mine_gait_hints(db, robot_class="quadruped")
        self.assertNotIn("kd", self._regions(h),
                         f"tight is not the same as earned — nothing links kd to travel: {h['hints']}")


@unittest.skipUnless(_MUJOCO, "adaptation runs a short gait search in MuJoCo")
class GaitHintsAdaptationTests(unittest.TestCase):
    def test_two_bodies_get_two_gaits_from_the_same_hints(self):
        from virturoid.services.gait_hints import adapt_gait_from_hints
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="hints2_")) / "m.db")
        for i, f in enumerate((1.4, 1.5, 1.6)):
            _bank(db, "quadruped", f"b{i}", {"freq": f, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25,
                                             "kp": 32.0, "kd": 1.5}, 0.9)
        a = adapt_gait_from_hints(compose_robot("a quadruped robot dog that walks"), db,
                                  generations=3, pop=6, steps=400, deploy_steps=600)
        b = adapt_gait_from_hints(compose_robot("a large quadruped robot"), db,
                                  generations=3, pop=6, steps=400, deploy_steps=600)
        self.assertEqual(a["source"], "hint_guided_adaptation")
        self.assertGreater(a["adapted_from_prior_by"], 0.0)          # it MOVED from the prior -> adapted, not copied
        self.assertNotEqual(a["params"], b["params"])                # different bodies -> different fitted gaits


if __name__ == "__main__":
    unittest.main()
