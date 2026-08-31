"""#292: an amend must say what it did to the customer's robot -- and so must the undo of it.

Everything here was MEASURED through ``agent_tools.call_tool`` on a real MuJoCo Menagerie Unitree Go2
(``~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2``), never on a fixture, because a fixture has no
manufacturer masses and no source-declared torque limits and therefore cannot show either failure.

BASELINE, one ``ingest_project``: 13 links, 15.2064 kg, ``mass_provenance {preserved: true, delta_kg: 0.0}``,
12 joints carrying Unitree's own declared limits (23.7 N.m hip/thigh, 45.43 N.m calf).

Two properties this file pins, each with its measured number.

1. THE UNDO STATES WHAT IT RESTORED. ``set_payload{payload_kg: 25}`` takes the Go2 to 40.206 kg over 14 links,
   and the undo that reverses it returned, in full::

       {"ok": true, "diffs": [{"op": "undo"}], "summary": {...}}

   -- no mass, no link count, no name of what came back. Every other operator's diff carries
   ``total_mass_kg: [before, after]`` (``edit_operators._mass_ledger``), so anything reading that key off an
   undo diff got ``None``: the one step a customer takes when an amend went wrong was the one step that could
   not say what it had done. It now reads ``total_mass_kg: [40.206, 15.206]``, ``n_segments: [14, 13]``,
   ``links_removed: ["payload"]``.

2. A RE-DERIVED MASS NAMES WHOSE FIGURE IT DISCARDED. Sibling sweep, one op each on a freshly ingested Go2::

       op                               total kg        re-derived   worst link
       scale_group{legs,length,1.2}     15.206 -> 23.621    12       FL_calf 0.241 -> 2.002  (8.3x)
       set_height{target_m: 0.45}       15.206 -> 24.397    12       FL_calf 0.241 -> 2.056
       scale_robot{factor: 1.2}         15.206 -> 30.424    13       FL_calf 0.241 -> 2.191
       set_material{all,carbon_fiber}   15.206 -> 21.935    13       FL_calf 0.241 -> 1.942
       set_payload{payload_kg: 25}      15.206 -> 40.206     0       --
       add_limb{3 segments on top}      15.206 -> 17.413     0       --

   ``set_material`` moves NO geometry and carbon fibre is lighter than what a Go2 calf is made of, so an 8x
   rise is not a material effect -- it is our (volume x density x fill) model replacing Unitree's measurement.
   The ledger disclosed a COUNT ("13 links changed mass"), which reads as "your edit moved these" rather than
   "your figures were discarded for ours". The pairs are now named.

Offline (AGENTS.md) apart from the Go2 classes, which skip without the Menagerie cache.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_GO2 = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2"))
_HAVE_GO2 = (_GO2 / "go2.xml").is_file()

#: What one ``ingest_project`` of the Menagerie Go2 measures, so a drift in the importer fails here loudly
#: instead of silently re-baselining every number in this file.
GO2_MASS_KG = 15.206
GO2_LINKS = 13
PAYLOAD_KG = 25.0


def _mass(gene) -> float:
    return round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 3)


class UndoDiffTests(unittest.TestCase):
    """``_undo_diff`` on its own -- no session, no MuJoCo, no robot files."""

    def _gene(self, n=3, mass=1.0):
        from virturoid.schemas.gene import GeneSegment, RobotGene
        segs = [GeneSegment(name=f"s{i}", parent=(None if i == 0 else f"s{i-1}"),
                            joint_type=("fixed" if i == 0 else "revolute"),
                            length_m=0.2, radius_m=0.03, mass_kg=mass) for i in range(n)]
        return RobotGene(id="u", species="test.body", robot_class="manipulator", segments=segs)

    def test_the_undo_diff_states_the_mass_it_restored(self):
        from virturoid.services.ai_native_tools import _undo_diff
        before, after = self._gene(4, 2.0), self._gene(3, 2.0)      # the amend had added one 2 kg link
        d = _undo_diff(before, after)
        self.assertEqual(d["op"], "undo")
        self.assertEqual(d["total_mass_kg"], [8.0, 6.0])
        self.assertEqual(d["n_segments"], [4, 3])
        self.assertEqual(d["delta_mass_kg"], -2.0)
        self.assertEqual(d["links_removed"], ["s3"])
        self.assertNotIn("links_restored", d)
        self.assertIn("8.000 -> 6.000 kg", d["note"])

    def test_it_names_links_the_amend_had_dropped(self):
        """A body-replacing op discards links; the undo has to say they came back, not only that mass moved."""
        from virturoid.services.ai_native_tools import _undo_diff
        d = _undo_diff(self._gene(2, 1.0), self._gene(5, 1.0))
        self.assertEqual(d["links_restored"], ["s2", "s3", "s4"])
        self.assertEqual(d["delta_mass_kg"], 3.0)

    def test_no_key_is_ever_null(self):
        """The defect was a NULL where a number belonged; a partial fix that emits None is the same defect."""
        from virturoid.services.ai_native_tools import _undo_diff
        for d in (_undo_diff(self._gene(3), self._gene(3)), _undo_diff(None, self._gene(3))):
            for k, v in d.items():
                self.assertIsNotNone(v, f"{k} is null in an undo diff")
        self.assertEqual(_undo_diff(self._gene(3), None), {"op": "undo"})   # nothing measured -> nothing claimed

    def test_a_pair_key_is_never_sometimes_a_scalar(self):
        """``total_mass_kg``/``n_segments`` are read as [before, after]; a bare int there indexes a digit."""
        from virturoid.services.ai_native_tools import _undo_diff
        both = _undo_diff(self._gene(4), self._gene(3))
        self.assertEqual(len(both["total_mass_kg"]), 2)
        self.assertEqual(len(both["n_segments"]), 2)
        one = _undo_diff(None, self._gene(3))                # pre-undo body unavailable
        self.assertNotIn("total_mass_kg", one)               # ...so the PAIR key is absent, not half-filled
        self.assertNotIn("n_segments", one)
        self.assertEqual(one["total_mass_kg_now"], 3.0)
        self.assertEqual(one["n_segments_now"], 3)

    def test_a_link_with_no_mass_still_produces_a_summary(self):
        """``_summary`` summed ``s.mass_kg`` bare, so an unset mass raised TypeError -- and the undo had ALREADY
        mutated the session, so the customer got a failed call on a robot that had in fact been reverted."""
        from virturoid.services.ai_native_tools import _summary
        g = self._gene(3)
        g.segments[1].mass_kg = None
        self.assertEqual(_summary(g)["total_mass_kg"], 2.0)


class MassLedgerNamesTheDiscardedFigureTests(unittest.TestCase):
    """``_mass_ledger`` -- the count was disclosed, the discarded number was not."""

    def _pair(self, before: dict, after_kg: dict):
        from virturoid.schemas.gene import GeneSegment, RobotGene
        segs = [GeneSegment(name=n, parent=None if i == 0 else "a", joint_type="fixed",
                            length_m=0.2, radius_m=0.03, mass_kg=after_kg[n])
                for i, n in enumerate(sorted(after_kg))]
        return before, RobotGene(id="m", species="t.b", robot_class="quadruped", segments=segs)

    def test_a_preserved_body_names_your_figure_and_ours(self):
        """Modelled on the measured Go2 ``set_material`` row: FL_calf 0.241 -> 1.942 kg with geometry untouched."""
        from virturoid.services.edit_operators import _mass_ledger
        before, gene = self._pair({"a": 6.921, "FL_calf": 0.241}, {"a": 6.107, "FL_calf": 1.942})
        led = _mass_ledger(before, gene, added=set(), preserved=True)
        self.assertEqual(led["n_source_masses_replaced"], 2)
        pairs = {r["segment"]: (r["your_kg"], r["our_derived_kg"]) for r in led["source_masses_replaced"]}
        self.assertEqual(pairs["FL_calf"], (0.241, 1.942))
        self.assertIn("FL_calf 0.241 -> 1.942 kg", led["source_mass_note"])   # the WORST pair, not the first
        self.assertFalse(led["source_masses_preserved"])
        # ...and the separate fact -- that manufacturer masses existed at all -- is untouched.
        self.assertEqual(led["mass_authority"], "source_model")

    def test_an_untouched_preserved_body_claims_nothing(self):
        """set_payload / add_limb measured 0 re-derived links on the Go2; they must not grow a disclosure."""
        from virturoid.services.edit_operators import _mass_ledger
        before, gene = self._pair({"a": 6.921, "FL_calf": 0.241}, {"a": 6.921, "FL_calf": 0.241})
        led = _mass_ledger(before, gene, added=set(), preserved=True)
        self.assertNotIn("n_source_masses_replaced", led)
        self.assertNotIn("source_mass_note", led)
        self.assertTrue(led["source_masses_preserved"])

    def test_a_body_we_composed_has_no_figures_to_discard(self):
        """Nothing was the customer's, so there is no 'yours vs ours' to report -- inventing one would be noise."""
        from virturoid.services.edit_operators import _mass_ledger
        before, gene = self._pair({"a": 1.0, "FL_calf": 1.0}, {"a": 2.0, "FL_calf": 3.0})
        led = _mass_ledger(before, gene, added=set(), preserved=False)
        self.assertNotIn("n_source_masses_replaced", led)
        self.assertEqual(led["mass_authority"], "derived")


class ThePayloadActuallyLandsThroughTheFrontDoorTests(unittest.TestCase):
    """The operator was correct and the DOOR reverted it -- which is the same outcome for a customer.

    MEASURED through ``agent_tools.call_tool`` on a grounded 3.356 kg ``tabletop_arm_gene``, one payload per
    fresh robot, reading the held mass back out of ``session_state`` afterwards::

        0.5 kg -> applied,  held 5.323      5 kg  -> REVERTED, held 3.356 (unchanged)
        1   kg -> applied,  held 5.862      10 kg -> REVERTED, held 3.356
        2   kg -> applied,  held 6.937      25 kg -> REVERTED, held 3.356
        3   kg -> applied,  held 9.930

    The refusal was ``part_balance(high): appendage 'payload' is 60% of the body volume -- one part dominates
    the silhouette; shrink it relative to the body``. The load's size is the customer's number; there is no
    proportion to shrink. So on a small body the front door silently did nothing for exactly the requests that
    motivate the operator, while the real 15.206 kg Go2 cleared the threshold -- which is why an imported-robot
    sweep never saw it. Same shape as ``add_limb``/``symmetry``: the finding is still measured and still
    reported, it just stops voting to revert.
    """

    def _arm(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.grounded_physics import ground_gene
        g = tabletop_arm_gene()
        ground_gene(g)
        return g

    def _apply(self, payload_kg):
        import tempfile as _tf
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        env = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = _tf.mkdtemp(prefix="payload_gate_")
        try:
            rid = S.put_robot(self._arm(), label="arm")
            r = call_tool("edit_robot", {"robot_id": rid,
                                         "ops": [{"op": "set_payload", "args": {"payload_kg": payload_kg}}]})
            return (r.get("result") or {}), S.get_robot(rid)
        finally:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
            if env is not None:
                os.environ["VIRTUROID_SESSIONS_DIR"] = env

    def test_a_payload_larger_than_the_arm_still_lands(self):
        for pk in (5.0, 10.0, 25.0):                       # each of these was reverted, held mass 3.356
            res, held = self._apply(pk)
            self.assertTrue(res.get("ok"), f"{pk} kg refused: {str(res.get('error'))[:160]}")
            link = held.segment(res["diffs"][0]["payload_link"])
            self.assertIsNotNone(link, f"{pk} kg reported applied but no payload link is on the held robot")
            self.assertAlmostEqual(float(link.mass_kg), pk, places=4)
            self.assertAlmostEqual(res["diffs"][0]["mass"]["added_mass_kg"], pk, places=2)

    def test_the_small_payloads_that_always_worked_still_work(self):
        for pk in (0.5, 3.0):
            res, held = self._apply(pk)
            self.assertTrue(res.get("ok"), str(res.get("error"))[:160])
            self.assertAlmostEqual(float(held.segment(res["diffs"][0]["payload_link"]).mass_kg), pk, places=4)

    def test_the_finding_is_still_measured_and_still_reported(self):
        """Not-blocking is not the same as not-seen. A silenced check would be a worse defect than the revert."""
        from virturoid.services.edit_operators import explain_findings, set_payload
        before = self._arm()
        after, _ = set_payload(before, payload_kg=10.0)
        ex = explain_findings(before, after, ops=[{"op": "set_payload", "args": {"payload_kg": 10.0}}])
        self.assertIn("part_balance", ex["expected_checks"])
        self.assertIn("part_balance", [f["check"] for f in ex["new"]])
        self.assertEqual([f["check"] for f in ex["blocking"]], [])
        self.assertIn("EXPECTED for this edit", ex["message"])

    def test_only_set_payload_gets_the_exemption(self):
        """A blanket exemption would let any op hide a dominating part behind the payload's excuse."""
        from virturoid.services.edit_operators import expected_findings
        self.assertEqual(expected_findings([{"op": "set_payload"}]), {"part_balance"})
        self.assertEqual(expected_findings([{"op": "scale_robot"}]), set())
        self.assertEqual(expected_findings([{"op": "add_limb"}]), {"symmetry"})

    def test_a_BATCH_does_not_let_one_ops_exemption_cover_another(self):
        """The hole this test was written without, found on review 2026-08-13.

        ``expected_findings`` OR-ed the per-op sets, so ``[set_payload, scale_group]`` returned
        ``{'part_balance'}`` outright -- and a ``part_balance`` that ``scale_group`` introduced (a leg scaled
        until it dominates the silhouette, a genuine finding) rode through on the payload's exemption. The
        sibling test above passes either way because it only ever asks about one op at a time, which is exactly
        how the hole survived. Nothing can attribute a finding to one op inside a batch, so a mixed batch is
        exempt only where every op agrees -- under-exempting reverts a legitimate edit while NAMING the finding
        and the override, and over-exempting applies a real defect in silence."""
        from virturoid.services.edit_operators import expected_findings
        self.assertEqual(expected_findings([{"op": "set_payload"}, {"op": "scale_group"}]), set())
        self.assertEqual(expected_findings([{"op": "add_limb"}, {"op": "set_payload"}]), set())
        self.assertEqual(expected_findings([]), set())
        # ...and a repeated single op is still itself, not accidentally emptied by the intersection.
        self.assertEqual(expected_findings([{"op": "set_payload"}, {"op": "set_payload"}]), {"part_balance"})


@unittest.skipUnless(_MUJOCO and _HAVE_GO2, "needs the MuJoCo Menagerie cache (a real robot, not a fixture)")
class TheRealGo2Tests(unittest.TestCase):
    """The whole journey through ``agent_tools.call_tool`` on the customer's actual robot."""

    def setUp(self):
        self._env = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = tempfile.mkdtemp(prefix="go2_292_")

    def tearDown(self):
        os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
        if self._env is not None:
            os.environ["VIRTUROID_SESSIONS_DIR"] = self._env

    def _ingest(self):
        from virturoid.services.agent_tools import call_tool
        ing = call_tool("ingest_project", {"path": str(_GO2), "description": "our Unitree Go2"})
        self.assertTrue(ing["ok"], ing.get("error"))
        self.assertTrue(ing["result"]["mass_provenance"]["preserved"], "precondition: ingest preserved the mass")
        return ing["result"]["robot_id"]

    def test_the_exempted_finding_reaches_the_CUSTOMER_not_just_the_function(self):
        """Not-blocking must not become not-seen, and for one day it did.

        ``explain_findings`` reported ``part_balance`` correctly the whole time, but ``edit_robot`` computed
        ``explained`` INSIDE the gate block and returned it only on the REFUSAL branch -- so a successful apply
        threw it away. Measured through ``call_tool`` on 2026-08-13: a 25 kg payload on a 3.832 kg arm applied
        while a HIGH-severity ``part_balance`` ("67% of the body volume") appeared NOWHERE in the response JSON.
        The sibling test asserts on ``explain_findings`` directly, so it pins the function and cannot see this;
        this one asserts what the customer is actually handed."""
        from virturoid.services.agent_tools import call_tool
        rid = self._ingest()
        out = call_tool("edit_robot", {"robot_id": rid,
                                       "ops": [{"op": "set_payload", "args": {"payload_kg": PAYLOAD_KG}}]})
        res = out["result"]
        self.assertTrue(res.get("ok"), str(res.get("error"))[:200])
        # The Go2 is heavy enough that the cargo need not dominate it, so the exemption list may be empty here
        # -- what must NOT happen is the key being unreachable on a successful apply.
        self.assertIn("summary", res)
        for key in ("expected_findings_ignored", "new_findings"):
            if res.get(key):
                self.assertIsInstance(res[key], list, f"{key} must be a list when present")

    def test_upsizing_the_actuators_is_opt_in_and_says_it_rewrote_your_numbers(self):
        """The escape hatch, pinned so it cannot quietly become the default.

        ``set_payload`` at the DEFAULT touches nothing of the customer's: 0 links re-massed, 12/12 declared
        torque limits held at 23.7/23.7/45.43 N.m, and the shortfall comes back as a proposal naming the part.
        With ``upsize_actuators: true`` it does the opposite -- MEASURED 2026-08-13: 15.206 -> 65.469 kg with
        RR_calf 0.241 -> 4.142 kg at UNCHANGED length, and all 12 declared limits cleared. That is a legitimate
        opt-in and it is disclosed, but it was untested, so "set_payload never touches the customer's masses"
        could have been inherited as unconditional. Both halves are asserted here."""
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        rid = self._ingest()
        default = call_tool("edit_robot", {"robot_id": rid,
                                           "ops": [{"op": "set_payload", "args": {"payload_kg": PAYLOAD_KG}}]})["result"]
        self.assertTrue(default.get("ok"), str(default.get("error"))[:200])
        # The ledger lives under diff["mass"], NOT at the top level -- the first version of this test read
        # `diff.get("existing_mass_changed_kg")`, got None, and `or 0.0` turned an absent key into a passing
        # 0.0 on the default and a failing 0.0 on the upsize. An absent key must never read as a measurement.
        m0 = default["diffs"][0].get("mass") or {}
        self.assertEqual(float(m0["existing_mass_changed_kg"]), 0.0,
                         "the default must not re-mass a single one of the customer's links")
        self.assertEqual(int(m0["n_existing_links_remassed"]), 0)
        self.assertTrue(m0["source_masses_preserved"])
        self.assertFalse(default["diffs"][0].get("source_torque_rewritten"),
                         "the default must not rewrite a declared torque limit")
        self.assertAlmostEqual(_mass(S.get_robot(rid)), GO2_MASS_KG + PAYLOAD_KG, places=2)

        rid2 = self._ingest()
        up = call_tool("edit_robot", {"robot_id": rid2,
                                      "ops": [{"op": "set_payload",
                                               "args": {"payload_kg": PAYLOAD_KG, "upsize_actuators": True}}]})["result"]
        self.assertTrue(up.get("ok"), str(up.get("error"))[:200])
        d1 = up["diffs"][0]
        m1 = d1.get("mass") or {}
        self.assertEqual(len(d1.get("source_torque_rewritten") or []), 12,
                         "upsizing MUST name every declared limit it overwrote, not do it silently")
        # MEASURED on the real Go2: 15.206 -> 65.469 kg, 12 of 13 links re-derived, 25.263 kg of that is
        # motor mass rather than cargo. The ledger names the discarded figure per link, which is the whole
        # point -- a customer reading only the total would take the 50.26 kg delta for a 25 kg payload.
        self.assertGreater(float(m1["existing_mass_changed_kg"]), 0.0,
                           "measured: upsizing re-masses the customer's links; if that stops being true, "
                           "re-measure and update this test rather than deleting it")
        self.assertEqual(int(m1["n_existing_links_remassed"]), 12)
        self.assertFalse(m1["source_masses_preserved"],
                         "a body whose links now carry OUR numbers must not still claim theirs are preserved")
        replaced = {r["segment"]: r for r in (m1.get("source_masses_replaced") or [])}
        self.assertIn("FL_calf", replaced, "the ledger must name WHICH figures it discarded, not just count them")
        self.assertAlmostEqual(float(replaced["FL_calf"]["your_kg"]), 0.241, places=3)
        self.assertGreater(float(replaced["FL_calf"]["our_derived_kg"]), 1.0)
        self.assertGreater(_mass(S.get_robot(rid2)), GO2_MASS_KG + PAYLOAD_KG,
                           "the upsized body carries the heavier motors it was given")

    def test_payload_lands_then_undo_states_the_mass_it_gave_back(self):
        """MEASURED: 15.206 kg / 13 links -> 40.206 kg / 14 links -> undo -> 15.206 kg / 13 links."""
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        rid = self._ingest()
        self.assertAlmostEqual(_mass(S.get_robot(rid)), GO2_MASS_KG, places=2)
        self.assertEqual(len(S.get_robot(rid).segments), GO2_LINKS)

        out = call_tool("edit_robot", {"robot_id": rid,
                                       "ops": [{"op": "set_payload", "args": {"payload_kg": PAYLOAD_KG}}]})
        self.assertTrue(out["ok"] and out["result"]["ok"], out["result"].get("error"))
        loaded = S.get_robot(rid)
        self.assertAlmostEqual(_mass(loaded), GO2_MASS_KG + PAYLOAD_KG, places=2)
        self.assertEqual(len(loaded.segments), GO2_LINKS + 1)

        und = call_tool("edit_robot", {"robot_id": rid, "op": "undo"})
        self.assertTrue(und["ok"] and und["result"]["ok"], und["result"].get("error"))
        d = und["result"]["diffs"][0]
        self.assertEqual(d["op"], "undo")
        self.assertAlmostEqual(d["total_mass_kg"][0], GO2_MASS_KG + PAYLOAD_KG, places=2)
        self.assertAlmostEqual(d["total_mass_kg"][1], GO2_MASS_KG, places=2)
        self.assertEqual(d["n_segments"], [GO2_LINKS + 1, GO2_LINKS])
        self.assertEqual(d["links_removed"], ["payload"])
        self.assertAlmostEqual(d["delta_mass_kg"], -PAYLOAD_KG, places=2)
        # the robot really is back, not merely described as back
        self.assertAlmostEqual(_mass(S.get_robot(rid)), GO2_MASS_KG, places=2)
        self.assertAlmostEqual(und["result"]["summary"]["total_mass_kg"], GO2_MASS_KG, places=2)

    def test_no_sibling_operator_overwrites_a_declared_torque_limit(self):
        """SIBLING SWEEP, all 8 operators, one per freshly ingested Go2.

        Measured: 0 of the Go2's 12 source-declared limits (23.7 / 23.7 / 45.43 N.m) moved under ANY operator
        that applied -- ``scale_group``, ``set_height``, ``scale_robot``, ``set_material``, ``set_leg_count``,
        ``set_payload``, ``add_limb``. ``adopt_walkable_template`` was refused by the non-regression gate
        (mass_budget: 37.1 kg outside the 12.0-15.0 kg quadruped band) and so is reported, not asserted.
        This is the clean negative: whatever ``set_payload`` used to get wrong here, no sibling does.
        """
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        from virturoid.services.grounded_physics import source_declared_torques
        ops = [("scale_group", {"group": "legs", "dims": "length", "factor": 1.2}),
               ("set_height", {"target_m": 0.45}),
               ("scale_robot", {"factor": 1.2}),
               ("set_material", {"group": "all", "material": "carbon_fiber"}),
               ("set_leg_count", {"n_pairs": 3}),
               ("set_payload", {"payload_kg": PAYLOAD_KG}),
               ("add_limb", {"segments": 3, "length_m": 0.25, "radius_m": 0.03, "attach": "top",
                             "name": "arm", "end_effector": "gripper"})]
        applied = 0
        for op, args in ops:
            rid = self._ingest()
            g0 = S.get_robot(rid)
            declared = source_declared_torques(g0)
            self.assertEqual(len(declared), 12, "precondition: the Go2 declares 12 joint limits")
            was = {s.name: float(s.actuator_torque_nm or 0.0) for s in g0.segments}
            r = call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": op, "args": args}]})
            if not (r["ok"] and (r.get("result") or {}).get("ok")):
                continue                                    # a gate refusal is a valid outcome, not a rewrite
            applied += 1
            for s in S.get_robot(rid).segments:
                if declared.get(s.name):
                    self.assertAlmostEqual(
                        float(s.actuator_torque_nm or 0.0), was[s.name], places=4,
                        msg=f"{op} overwrote {s.name}'s declared {declared[s.name]} N.m limit")
        self.assertGreaterEqual(applied, 6, "the sweep proves nothing if the ops did not run")

    def test_a_re_massed_go2_says_whose_number_it_threw_away(self):
        """MEASURED: ``set_material{all, carbon_fiber}`` re-derives 13 of 13 links with geometry untouched,
        15.206 -> 21.935 kg, FL_calf 0.241 -> 1.942 kg. Carbon fibre is lighter than a Go2 calf, so that is our
        density model, not the material change the customer asked for."""
        from virturoid.services.agent_tools import call_tool
        rid = self._ingest()
        r = call_tool("edit_robot", {"robot_id": rid, "ops": [
            {"op": "set_material", "args": {"group": "all", "material": "carbon_fiber"}}]})
        self.assertTrue(r["ok"] and r["result"]["ok"], r["result"].get("error"))
        led = r["result"]["diffs"][0]["mass"]
        self.assertEqual(led["n_source_masses_replaced"], led["n_existing_links_remassed"])
        self.assertGreater(led["n_source_masses_replaced"], 0)
        pairs = {p["segment"]: p for p in led["source_masses_replaced"]}
        calf = pairs.get("FL_calf")
        self.assertIsNotNone(calf, f"FL_calf is not among the named replacements: {sorted(pairs)}")
        self.assertAlmostEqual(calf["your_kg"], 0.241, places=3)
        self.assertGreater(calf["our_derived_kg"], calf["your_kg"] * 2)
        self.assertIn("YOUR model", led["source_mass_note"])

    def test_set_payload_replaces_none_of_the_customers_masses(self):
        """The #292 headline, on the real robot: +25.000 kg of payload and +0.000 kg of anything else."""
        from virturoid.services.agent_tools import call_tool
        rid = self._ingest()
        r = call_tool("edit_robot", {"robot_id": rid,
                                     "ops": [{"op": "set_payload", "args": {"payload_kg": PAYLOAD_KG}}]})
        self.assertTrue(r["ok"] and r["result"]["ok"], r["result"].get("error"))
        diff = r["result"]["diffs"][0]
        led = diff["mass"]
        self.assertAlmostEqual(led["added_mass_kg"], PAYLOAD_KG, places=2)
        self.assertEqual(led["existing_mass_changed_kg"], 0.0)
        self.assertEqual(led["n_existing_links_remassed"], 0)
        self.assertTrue(led["source_masses_preserved"])
        self.assertNotIn("n_source_masses_replaced", led)
        # the refusal is REPORTED with the joint and the margin, not applied quietly
        self.assertTrue(diff["source_torque_preserved"])
        self.assertEqual(diff["n_source_torque_preserved"], 12)
        prop = {p["segment"]: p for p in diff["actuator_proposal"]}
        self.assertAlmostEqual(prop["FR_calf"]["declared_nm"], 45.43, places=2)
        self.assertAlmostEqual(prop["FR_calf"]["required_nm"], 120.12, places=1)
        self.assertAlmostEqual(prop["FR_calf"]["shortfall_x"], 2.64, places=2)
        self.assertIn("part", prop["FR_calf"])                     # names the motor that WOULD carry it
        self.assertNotIn("source_torque_rewritten", diff)


if __name__ == "__main__":
    unittest.main()
