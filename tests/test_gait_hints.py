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
        # bank CREDIBLE walks whose freq clusters ~2.0 (deliberately NOT the 1.5 default) -> the mined region
        # must follow the DATA to ~2.0, proving it isn't a hardcoded constant
        freqs = (1.9, 2.0, 2.1, 1.95, 2.05, 1.92, 2.08, 2.0, 1.98, 2.02)
        for i, f in enumerate(freqs):
            _bank(db, "quadruped", f"b{i}", {"freq": f, "hip_amp": 0.8, "knee_amp": 1.1, "duty": 0.3,
                                             "kp": 30.0, "kd": 1.4}, 0.9)
        h = mine_gait_hints(db, robot_class="quadruped")
        self.assertEqual(h["n"], len(freqs))
        self.assertAlmostEqual(h["prior"]["freq"], 2.0, delta=0.15)   # region tracked the data, not the 1.5 default
        # ...but walks with no recorded travel CANNOT establish a param region, so none is CLAIMED (#266).
        # The prior stays available as an explicitly un-tuned warm-start seed; a hint is a claim about evidence.
        self.assertFalse([x for x in h["hints"] if x.get("kind") == "param_region"], h["hints"])
        # the relational hint is DISCOVERED (all of them have knee_amp > hip_amp), and its support is counted in
        # DISTINCT BODIES — these rows carry no inline body, so each counts as one of its own (#274)
        rel = [x for x in h["hints"] if x.get("kind") == "relation"]
        self.assertTrue(rel, h["hints"])
        self.assertEqual(rel[0]["support"], len(freqs))
        self.assertEqual((h["n_bodies"], h["n_rows_without_body"]), (len(freqs), len(freqs)))

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


class PseudoReplicationCannotCarryAGateTests(unittest.TestCase):
    """#274 — the gates count DISTINCT BODIES, so a bank that re-fits one machine cannot manufacture evidence.

    All three gates are counting arguments over observations assumed independent, and banked rows are not: the
    bank is keyed by ``structural_gait_key`` (exact kinematics), so re-fitting ONE robot at a dozen slightly
    different masses writes a dozen rows that any honest reading of "how many robots is this advice based on"
    calls one. MEASURED on the live 101-row bank: 28 distinct bodies, largest supplying 24 rows; on the 55 that
    also survive the fragility gate, 21 bodies with one supplying 18 — and there ``freq`` cleared the association
    gate at rank-corr -0.39, p=0.0075, while one row per body inverted it to +0.12, the wrong sign entirely.

    This pins the property DIRECTLY rather than hoping the real bank keeps it, and it pins it BOTH ways: the same
    planted numbers must be refused when they come from one body and accepted when they come from many. A gate
    that only ever says "no" would satisfy the negative half while being useless.
    """
    N = 40

    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="hints_pseudo_")) / "m.db")

    @classmethod
    def _planted(cls, i: int) -> tuple[float, float]:
        """A STRONG, genuine freq<->travel association: inside a tight band the robot walks far, outside it the
        robot barely moves. Identical numbers are fed to both arms, so the ONLY thing that differs is how many
        robots produced them."""
        t = i / (cls.N - 1)
        return (1.45 + 0.30 * t, 2.0 + 0.5 * t) if i % 2 == 0 else (0.8 + 2.4 * t, 0.4 + 0.2 * t)

    @staticmethod
    def _params(freq: float) -> dict:
        return {"freq": freq, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5}

    def test_N_rows_from_ONE_body_cannot_pass_a_gate_while_the_same_numbers_from_N_bodies_can(self):
        from virturoid.services.gait_flywheel import bank_gait
        from virturoid.services.gait_hints import _MIN_REGION_SUPPORT, mine_gait_hints
        from virturoid.services.heldout_set import body_key
        from virturoid.services.morphology_composer import compose_robot
        import copy
        import types

        # ---- ARM A: N banked rows, ONE robot. Mass is in ``structural_gait_key`` (so each variant banks its own
        # row, exactly as a re-fit of the same machine does) and NOT in ``body_key`` (so they are one body).
        db = self._db()
        base = compose_robot("a small quadruped robot dog")
        keys = set()
        for i in range(self.N):
            g = copy.deepcopy(base)
            for s in g.segments:
                s.mass_kg = round(float(s.mass_kg) * (1.0 + 0.013 * i), 6)
            g.id = f"{base.id}_m{i}"
            freq, fwd = self._planted(i)
            keys.add(body_key(g))
            self.assertIsNotNone(
                bank_gait(db, g, types.SimpleNamespace(best_survived=True, best_credible=True,
                                                       best_forward=fwd, best_height_ratio=0.8,
                                                       best_params=self._params(freq))),
                f"row {i} must actually bank, or the arm proves nothing")
        self.assertEqual(len(keys), 1, "the N variants must be ONE body by the structural key, else no pseudo-"
                                       "replication is being simulated")
        one = mine_gait_hints(db, robot_class="quadruped", min_success=0.0)
        self.assertEqual(one["n"], self.N, one["note"])            # N ROWS reached the miner...
        self.assertEqual(one["n_bodies"], 1, one["note"])          # ...from ONE robot
        self.assertEqual(one["n_rows_without_body"], 0)            # every row carried its body inline

        # THE REGRESSION: not one gate may pass. No param_region, and no relational rule either — a proportion
        # over a single observation is not a proportion.
        self.assertEqual([x for x in one["hints"] if x.get("kind") in ("param_region", "relation")], [],
                         f"a corpus of {self.N} rows from ONE body passed a gate: {one['hints']}")
        self.assertEqual(one["bounds"], {}, one["bounds"])
        freq_sup = [s for s in one["suppressed"] if s["param"] == "freq"]
        self.assertTrue(freq_sup, one["suppressed"])
        # and the operator is told the REAL sample size, not the row count that flattered it
        self.assertEqual((freq_sup[0]["support"], freq_sup[0]["rows"]), (1, self.N), freq_sup[0])
        self.assertIn("distinct bodies", freq_sup[0]["reason"])
        self.assertIn("DISTINCT bodies", one["note"])
        db.close()

        # ---- ARM B (the control): the SAME planted numbers, one per distinct bank identity. The signal is real,
        # so it must survive — otherwise the negative half above is satisfied by a gate that refuses everything.
        db2 = self._db()
        for i in range(self.N):
            freq, fwd = self._planted(i)
            _bank(db2, "quadruped", f"body{i}", self._params(freq), 0.9, forward=fwd)
        many = mine_gait_hints(db2, robot_class="quadruped", min_success=0.0)
        self.assertEqual((many["n"], many["n_bodies"]), (self.N, self.N), many["note"])
        regions = {x["param"]: x for x in many["hints"] if x.get("kind") == "param_region"}
        self.assertIn("freq", regions, f"the same association from {self.N} bodies must still be minable: {many}")
        self.assertEqual((regions["freq"]["support"], regions["freq"]["rows"]), (self.N, self.N))
        self.assertGreaterEqual(regions["freq"]["range"][0], 1.40, regions["freq"])
        self.assertLessEqual(regions["freq"]["range"][1], 1.80, regions["freq"])
        self.assertGreaterEqual(self.N, _MIN_REGION_SUPPORT)
        db2.close()

    def test_the_representative_of_a_body_is_its_MEDIAN_walk_deterministically(self):
        """The dedup rule itself. ``best`` re-introduces the very bias being removed — a body searched 24 times
        wins on max-of-n effort, measured at rank-corr +0.44 between a family's row count and its
        representative's distance on the live bank (median: +0.10) — and ``random`` is not reproducible. So the
        shipped rule is the median-PERFORMING ROW: a real (value, distance) pair that was actually run, never a
        synthesised row of independent per-parameter medians."""
        from virturoid.services.gait_hints import _DEDUP_RULE, representative_rows
        self.assertEqual(_DEDUP_RULE, "median")
        bodies = ["a", "a", "a", "b", "a", "b"]
        fwd = [0.1, 9.0, 0.5, 2.0, 0.3, 7.0]
        keep = representative_rows(bodies, fwd)
        self.assertEqual(len(keep), 2)                             # one row per distinct body
        # a: lower median of {0.1, 0.3, 0.5, 9.0} -> 0.3;  b: lower median of {2.0, 7.0} -> 2.0
        self.assertEqual(sorted(fwd[i] for i in keep), [0.3, 2.0])
        self.assertNotIn(9.0, [fwd[i] for i in keep], "the body's BEST row must not represent it")
        # deterministic, and independent of the order the rows arrive in
        self.assertEqual(keep, representative_rows(bodies, fwd))
        rb, rf = list(reversed(bodies)), list(reversed(fwd))
        self.assertEqual(sorted(rf[i] for i in representative_rows(rb, rf)), [0.3, 2.0])
        # ties cannot make it wobble, and a single-row body represents itself
        self.assertEqual(representative_rows(["x", "x", "x"], [1.0, 1.0, 1.0]), [1])
        self.assertEqual(representative_rows(["x"], [4.0]), [0])
        self.assertEqual(representative_rows([], []), [])

    def test_a_region_that_depends_on_WHICH_walk_represents_a_body_is_refused(self):
        """Collapsing to one row per body raises a question row-counting never had to answer, and on the live
        bank the answer decides the result: of the 21 gate-surviving bodies, ``median`` passed knee_amp/kp/kd,
        ``best`` passed nothing, and ``random`` passed ``freq`` under one seed of four. No parameter passed under
        more than one selection — noise at n=21, not four findings. So the gate requires the verdict to hold
        under EVERY selection in its panel.

        Constructed directly: each body banks one walk that lands on a tight planted band and one that does not,
        with the band member always the further-travelling of the two. ``best`` therefore sees a clean signal and
        ``median``/``random`` see a coin flip — the verdict is selection-dependent, so no region may be claimed.
        """
        from virturoid.services.gait_hints import _region_evidence, representative_rows
        n = 24
        vals, fwd, bodies = [], [], []
        for i in range(n):
            t = i / (n - 1)
            # an in-band walk, plus a far-flung one whose value and travel are scrambled INDEPENDENTLY of each
            # other, so the second walk carries no band of its own — otherwise the arm plants a second signal
            v2, f2 = ((i * 7) % n) / (n - 1), ((i * 11) % n) / (n - 1)
            vals += [1.50 + 0.05 * t, 0.85 + 2.30 * v2]
            fwd += [2.0 + 0.5 * t, 0.30 + 0.2 * f2]                # the in-band walk always travels further
            bodies += [f"body{i}", f"body{i}"]
        # the panel disagrees by construction: `best` takes the in-band walk from every body...
        best = representative_rows(bodies, fwd, rule="best")
        self.assertTrue(_region_evidence([vals[i] for i in best], [fwd[i] for i in best], 0.8, 3.2)["ok"])
        # ...`median` (lower median of two) takes the OTHER one, which carries no band at all
        med = representative_rows(bodies, fwd)
        self.assertFalse(_region_evidence([vals[i] for i in med], [fwd[i] for i in med], 0.8, 3.2)["ok"])
        # so the gate, asked with the bodies, must refuse and SAY it was the choice that decided it
        ev = _region_evidence(vals, fwd, 0.8, 3.2, bodies=bodies)
        self.assertFalse(ev["ok"], ev)
        self.assertEqual((ev["support"], ev["rows"]), (n, 2 * n))
        self.assertIn("flips with which of a body's banked walks", ev["why"])
        self.assertNotEqual(ev["selections_ok"], "5/5", ev)
        # and a corpus where every body has ONE walk has no such choice to make -> the panel cannot veto it
        solo = _region_evidence(vals[::2], fwd[::2], 0.8, 3.2, bodies=[f"b{i}" for i in range(n)])
        self.assertTrue(solo["ok"], solo)
        self.assertIsNone(solo["selections_ok"], "a one-row-per-body corpus has nothing to choose between")

    def test_the_gate_reports_the_same_reading_on_rows_that_it_refuses_on_bodies(self):
        """The audit's side-by-side arm, pinned: ``bodies=None`` is the OLD row-counting behaviour and it is what
        the live bank's ``freq`` cleared the association gate on. Same inputs, one argument apart."""
        from virturoid.services.gait_hints import _region_evidence
        vals = [self._planted(i)[0] for i in range(self.N)]
        fwd = [self._planted(i)[1] for i in range(self.N)]
        rows_arm = _region_evidence(vals, fwd, 0.8, 3.2)
        body_arm = _region_evidence(vals, fwd, 0.8, 3.2, bodies=["one"] * self.N)
        self.assertTrue(rows_arm["ok"] and rows_arm["support"] == self.N, rows_arm)
        self.assertFalse(body_arm["ok"], body_arm)
        self.assertEqual((body_arm["support"], body_arm["rows"]), (1, self.N))
        self.assertFalse(rows_arm["deduped_by_body"])
        self.assertTrue(body_arm["deduped_by_body"])


class AuditArmTests(unittest.TestCase):
    """``scripts/audit_gait_bank.gates`` — the three dedup arms an audit reports side by side.

    Body dedup answers "how many ROBOTS is this claim made of". It does NOT answer "how many CONTROLLERS",
    and on a corpus grown by warm-starting every fit from the same bank those are different questions: distinct
    bodies converging on one operating point are one observation of that point counted many times, and the
    replication lands on all five parameters at once. ``gait_hints._DEDUP_RULE`` records that check dissolving
    ``knee_amp`` on the live bank; it was done by hand there, and this pins it as an arm the audit always runs.
    """
    def _audit(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import audit_gait_bank as A
        return A

    def _row(self, i, params, forward):
        return {"skill_id": f"gait::quadruped::b{i}", "params": params, "gene": None,
                "base_config": {"gait_params": params, "forward_m": forward}}

    def test_the_op_point_arm_sees_replication_the_body_arm_cannot(self):
        A = self._audit()
        # Twenty DISTINCT bodies -- so the body arm collapses nothing -- that between them shipped only two
        # operating points. Every parameter reads as a razor-tight "region" that is really n=2.
        pts = [{"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5},
               {"freq": 1.52, "hip_amp": 0.91, "knee_amp": 1.01, "kp": 33.0, "kd": 1.55}]
        rows = [self._row(i, dict(pts[i % 2]), 2.0 + 0.01 * i) for i in range(20)]
        out = A.gates(rows, None)
        by_arm = {k.split(", ", 1)[1]: v for k, v in out.items()}
        self.assertEqual(set(by_arm), {"rows as observations", "ONE ROW PER DISTINCT BODY",
                                       "ONE ROW PER DISTINCT OP-POINT"})
        self.assertEqual(by_arm["rows as observations"]["freq"]["support"], 20)
        self.assertEqual(by_arm["ONE ROW PER DISTINCT BODY"]["freq"]["support"], 20)
        self.assertEqual(by_arm["ONE ROW PER DISTINCT OP-POINT"]["freq"]["support"], 2)
        for key in ("freq", "kp", "kd"):
            ev = by_arm["ONE ROW PER DISTINCT OP-POINT"][key]
            self.assertFalse(ev["ok"], ev)
            self.assertIn("too few", ev["why"])

    def test_the_selection_panel_tally_is_reported_not_just_the_verdict(self):
        """``ok`` and ``selections_ok`` answer different questions, so the audit prints both. A one-row-per-group
        corpus never exercises the panel and must report null rather than a manufactured 5/5."""
        A = self._audit()
        rows = [self._row(i, {"freq": 1.4 + 0.02 * i, "hip_amp": 0.9, "knee_amp": 1.0,
                              "kp": 30.0 + i, "kd": 1.5}, 2.0 + 0.01 * i) for i in range(12)]
        ev = A.gates(rows, None)["whole bank, ONE ROW PER DISTINCT BODY"]["freq"]
        self.assertIn("selections_ok", ev)
        self.assertIsNone(ev["selections_ok"])
        self.assertIn("center", ev)
        self.assertIn("band", ev)


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
