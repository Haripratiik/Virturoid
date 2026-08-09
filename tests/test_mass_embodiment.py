"""TASK #211 — the robot we simulate weighs what the robot you build weighs.

Two halves of one defect, and they pointed in OPPOSITE directions, which is why neither showed up as an
obvious number:

  * THE SIM WAS LIGHT. ``ground_gene`` gives a link its structure and the motor that drives it, and stops
    there. The battery, the compute, the cameras, the LiDAR and the wheels are all in the bill of materials
    with datasheet masses, and no link carried any of them. Measured at HEAD on this checkout, per body:

        authored hexapod   15.043 kg simulated, 2.851 kg of named electronics never on it (+19%)
        authored horse     15.951 kg simulated, 2.851 kg missing
        two-legged robot   46.365 kg simulated, 3.724 kg missing

  * THE PARTS LIST WAS HEAVY, by more. ``bom_builder`` grouped finished link masses into its "material"
    lines while ALSO listing every motor as an actuator line, so the same hardware was billed twice:

        body                     simulated      parts list      truth
        authored horse             15.951         29.212        18.849
        authored hexapod           15.043         25.664        17.894
        imported Unitree G1        33.341         94.176        44.866   (its own motors, billed again)

    The G1 row is the one that matters most: its 33.341 kg is the CUSTOMER'S OWN, preserved verbatim by the
    ingest path, and the parts list still claimed 94 kg for it.

So: ``grounded_physics.embody_component_masses`` bolts the parts list's own components onto the links that
physically carry them, ``bom_builder`` counts each part once, and the two numbers meet. The tests below are
that equality, its per-link placement, its idempotence, and the one case where it must NOT fire — an imported
robot, whose masses are the customer's and already include their own electronics.

WHY THIS WAS PARKED, AND WHY IT SHIPS NOW. The 2026-07-21 attempt was reverted because a heavier body stopped
walking. That measurement was taken at ONE hand-tuned operating point shared by every body; a body that gets
heavier and fails at another robot's operating point is telling you it needs its own, not that its mass is
wrong. Re-measured with ``gait_flywheel.fit_gait_for_body`` giving each body its own point, the verdicts hold
(see the task #211 report). Nothing here tunes a morphology to flatter a controller.
"""

import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_COMPONENT_CATEGORIES = ("camera", "lidar", "imu", "compute", "power", "wheel", "drive_motor", "gripper",
                         "thermal", "gps", "microphone", "force_torque", "rotor", "esc", "flight_controller")


def _grounded(prompt: str, task: str = ""):
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot(prompt, ensure_walkable=False)
    report = ground_and_repair(gene, task=task)
    return gene, report


def _total(gene) -> float:
    return round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 3)


@unittest.skipUnless(_MUJOCO, "composing + grounding a body needs MuJoCo")
class MassEmbodimentTests(unittest.TestCase):
    def test_simulated_mass_equals_the_bill_of_materials(self):
        """The headline. Same robot, one mass — not a 1.7x split between the sim and the spec sheet."""
        from virturoid.services.bom_builder import build_bom
        for prompt, task in (("a six-legged walking robot", "walk forward"),
                             ("a robot horse", "walk forward"),
                             ("a 6-axis robot arm", "pick and place")):
            with self.subTest(prompt=prompt):
                gene, _ = _grounded(prompt, task)
                bom = build_bom(gene, task=task)
                sim, listed = _total(gene), float(bom["totals"]["mass_kg"])
                # The tolerance is ROUNDING ONLY, and it has to scale with the part count: link masses are
                # kept to 3 dp and materials lines to 4, so a 26-link hexapod can differ from its own parts
                # list by ~1 g. Measured across these three bodies the real gap is 0.000-0.001 kg.
                self.assertAlmostEqual(sim, listed, delta=max(0.02, 0.002 * sim),
                                       msg=f"{prompt}: simulated {sim} kg vs parts list {listed} kg")

    def test_a_motor_is_not_billed_twice(self):
        """The material lines are RAW STOCK. They used to be finished link masses, motors included, while the
        actuator lines listed those same motors — so the totals were the actuators plus the actuators."""
        from virturoid.services.bom_builder import build_bom
        from virturoid.services.grounded_physics import structural_link_masses
        gene, _ = _grounded("a six-legged walking robot", "walk forward")
        bom = build_bom(gene, task="walk forward")
        material = sum(float(ln["mass_kg"]) for ln in bom["lines"] if ln["category"] == "material")
        actuators = sum(float(ln["mass_kg"]) for ln in bom["lines"] if ln["category"] == "actuator")
        self.assertGreater(actuators, 0.0)
        self.assertAlmostEqual(material, sum(structural_link_masses(gene).values()), delta=0.01)
        # structure alone is strictly lighter than the finished body: the motors and the electronics are in it
        self.assertLess(material, _total(gene) - actuators + 1e-6)

    def test_the_body_carries_the_battery_and_the_compute(self):
        """The point of the whole change: named parts, on links, with their datasheet masses."""
        from virturoid.services.grounded_physics import _embodiment_record
        gene, report = _grounded("a six-legged walking robot", "walk forward")
        emb = report.get("component_embodiment") or {}
        self.assertTrue(emb.get("applied"), emb.get("reason"))
        self.assertGreater(float(emb.get("components_kg") or 0.0), 0.5)
        categories = {p["category"] for p in emb.get("parts", [])}
        self.assertIn("power", categories)                     # a battery is the single heaviest omission
        self.assertIn("compute", categories)
        rec = _embodiment_record(gene)
        self.assertTrue(rec["component_kg"])
        # every kilogram claimed is on a link that exists
        names = {s.name for s in gene.segments}
        self.assertTrue(set(rec["component_kg"]) <= names)

    def test_the_battery_rides_on_the_trunk_not_on_a_foot(self):
        """Placement is inertia, not bookkeeping: 2.6 kg on the trunk and 2.6 kg on a shin are different
        robots, and a mass-embodiment pass that spreads parts evenly would be inventing a third one."""
        gene, report = _grounded("a robot horse", "walk forward")
        root = gene.root().name
        hosts = {p["part"]: p["link"] for p in (report.get("component_embodiment") or {}).get("parts", [])}
        power = [link for part, link in hosts.items() if "battery" in part.lower() or "lipo" in part.lower()]
        for link in power:
            self.assertEqual(link, root, f"the pack was mounted on {link}, not the trunk")

    def test_embodiment_is_idempotent_across_re_grounding(self):
        """ground -> embody -> ground -> embody must converge. A ratchet here would inflate a body every time
        an edit touched it, which is exactly the failure mode the torque-requirement pin exists to stop."""
        from virturoid.services.bom_builder import build_bom
        from virturoid.services.gene_build import ground_and_repair
        gene, _ = _grounded("a robot cheetah", "walk forward")
        first, first_bom = _total(gene), build_bom(gene, task="walk forward")["totals"]["mass_kg"]
        for _ in range(2):
            ground_and_repair(gene, task="walk forward")
        self.assertAlmostEqual(_total(gene), first, delta=1e-6)
        self.assertAlmostEqual(build_bom(gene, task="walk forward")["totals"]["mass_kg"], first_bom, delta=1e-6)

    def test_a_task_less_re_ground_ships_the_same_robot(self):
        """The export door re-grounds WITHOUT the prompt and then compares physical fingerprints to decide
        whether it is shipping the body the verdict was signed on. The sensor suite is task-adaptive, so a
        task-less re-ground that re-picked the hardware would swap a LiDAR for a webcam and read as 'not the
        same robot' on a robot nobody edited."""
        from virturoid.services.gene_build import ground_and_repair
        from virturoid.services.grounded_physics import fingerprint_delta, physical_fingerprint
        gene, _ = _grounded("a four-wheeled warehouse rover", "navigate a warehouse and avoid obstacles")
        held = physical_fingerprint(gene)
        ground_and_repair(gene)                                   # the export door, no task in hand
        delta = fingerprint_delta(held, physical_fingerprint(gene))
        self.assertTrue(delta["same"], delta)
        self.assertEqual(delta["delta_mass_kg"], 0.0)

    def test_the_body_carries_the_motor_the_parts_list_orders(self):
        """Sizing picks a motor per joint; the BOM then standardises onto fewer part numbers, and a roll-up
        only ever goes UP. The robot a customer procures is the standardised one, so that is the mass the
        simulated link must carry — otherwise 'sim mass == BOM mass' is true only to within the rounding."""
        from virturoid.services.bom_builder import build_bom
        from virturoid.services.component_catalog import resolve_part
        from virturoid.services.grounded_physics import _embodiment_record
        gene, _ = _grounded("a robot horse", "walk forward")
        bom = build_bom(gene, task="walk forward")
        embodied = _embodiment_record(gene)["actuator_kg"]
        for joint, part_name in (bom.get("actuator_map") or {}).items():
            if joint not in embodied:
                continue
            part = resolve_part(str(part_name))
            self.assertIsNotNone(part, part_name)
            self.assertAlmostEqual(float(embodied[joint]), float(part.mass_kg), delta=1e-4,
                                   msg=f"{joint} carries a different motor than the BOM orders")

    def test_the_balance_of_system_estimate_is_itemised_where_a_part_number_exists(self):
        """The class band tops a too-light body up to a plausible finished mass with an estimate of 'battery,
        compute, wiring, fasteners, covers'. Once the battery and the compute are real parts on real links,
        estimating them as well would bill them twice — so the estimate keeps only the residue, and the
        finished mass does not move."""
        gene, report = _grounded("a 6-axis robot arm", "pick and place")
        prior = (gene.metadata or {}).get("physical_prior") or {}
        if not prior:
            self.skipTest("this body grounds above its class floor, so it carries no estimate to itemise")
        itemised = float(prior.get("itemised_component_mass_kg") or 0.0)
        self.assertGreater(itemised, 0.0)
        # the estimate gave way to the itemised parts (by at least their mass), and the FINISHED mass held:
        # the band decides how heavy this class of machine is, not how much of that is guesswork.
        before_estimate = float(report["balance_of_system_mass_kg"])
        self.assertLessEqual(float(prior["balance_of_system_mass_kg"]), before_estimate - itemised + 1e-6)
        lo, hi = prior["mass_band_kg"]
        self.assertGreaterEqual(_total(gene) + 1e-6, lo)
        self.assertLessEqual(_total(gene) - 1e-6, hi)

    def test_the_estimate_is_a_bom_line_so_the_parts_list_adds_up(self):
        """A parts list that silently omits 11 kg of harness and covers cannot equal the body it describes."""
        from virturoid.services.bom_builder import build_bom
        gene, _ = _grounded("a 6-axis robot arm", "pick and place")
        bom = build_bom(gene, task="pick and place")
        bos = [ln for ln in bom["lines"] if ln["category"] == "balance_of_system"]
        if not (gene.metadata or {}).get("physical_prior"):
            self.skipTest("no estimate on this body")
        self.assertEqual(len(bos), 1)
        self.assertNotIn("$", bos[0]["detail"])                       # no invented price
        self.assertEqual(bos[0]["unit_price_usd"], 0.0)
        self.assertIn("NOT a part number", bos[0]["detail"])


@unittest.skipUnless(_MUJOCO, "composing + grounding a body needs MuJoCo")
class ImportedRobotKeepsItsOwnMassTests(unittest.TestCase):
    """Embodiment applies to bodies WE compose. A customer's model already includes their battery."""

    def _imported_stub(self):
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot("a quadruped robot dog", ensure_walkable=False)
        gene.metadata = {**(gene.metadata or {}), "mass_source": "source_model"}
        return gene

    def test_embodiment_declines_and_changes_nothing(self):
        from virturoid.services.grounded_physics import embody_component_masses
        gene = self._imported_stub()
        before = _total(gene)
        out = embody_component_masses(gene)
        self.assertFalse(out["applied"])
        self.assertTrue(out.get("mass_preserved"))
        self.assertEqual(out["delta_kg"], 0.0)
        self.assertEqual(_total(gene), before)
        self.assertIn("CUSTOMER'S OWN", out["reason"])

    def test_grounding_an_imported_body_still_leaves_its_mass_alone(self):
        from virturoid.services.gene_build import ground_and_repair
        gene = self._imported_stub()
        before = _total(gene)
        report = ground_and_repair(gene, task="walk forward")
        self.assertTrue(report.get("mass_preserved"))
        self.assertAlmostEqual(_total(gene), before, delta=1e-9)
        self.assertFalse((report.get("component_embodiment") or {}).get("applied"))

    def test_the_parts_list_does_not_bill_their_motors_a_second_time(self):
        """A Unitree G1 we preserve at 33.341 kg shipped a 94.176 kg parts list, because our catalog
        equivalents for motors already inside its link masses were added on top."""
        from virturoid.services.bom_builder import build_bom
        gene = self._imported_stub()
        bom = build_bom(gene, task="walk forward")
        actuator_lines = [ln for ln in bom["lines"] if ln["category"] == "actuator"]
        self.assertTrue(actuator_lines)
        self.assertTrue(all(ln["in_mass_total"] is False for ln in actuator_lines))
        self.assertTrue(all("EQUIVALENT" in ln["detail"] for ln in actuator_lines))
        listed = float(bom["totals"]["mass_kg"])
        self.assertLess(listed, _total(gene) + sum(float(ln["mass_kg"]) for ln in actuator_lines))
        # and the totals SAY which robot they are describing
        self.assertIn("proposes ADDING", bom["totals"]["mass_note"])


@unittest.skipUnless(_MUJOCO, "composing + grounding a body needs MuJoCo")
class FidelityReportTests(unittest.TestCase):
    def test_it_compares_the_body_against_its_own_parts_list_not_against_itself(self):
        """This used to ground a COPY of the gene and compare the two, which on the current build path is the
        same call twice — it reported ratio 1.00 'faithful' while the body carried no battery at all."""
        from virturoid.services.fidelity_report import bom_sim_fidelity
        gene, _ = _grounded("a six-legged walking robot", "walk forward")
        rep = bom_sim_fidelity(gene)
        self.assertIsNotNone(rep["bom_mass_kg"])
        self.assertAlmostEqual(float(rep["bom_mass_kg"]), float(rep["sim_mass_kg"]), delta=0.02)
        self.assertTrue(rep["faithful"], rep["flags"])

    def test_it_flags_a_body_that_was_grounded_but_never_embodied(self):
        """``ground_gene`` alone leaves the sensors, compute and battery off the body — the exact state this
        whole change exists to end, and the state the old report certified as 'faithful'."""
        from virturoid.services.fidelity_report import bom_sim_fidelity
        from virturoid.services.grounded_physics import ground_gene
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot("a six-legged walking robot", ensure_walkable=False)
        ground_gene(gene)                                    # no embodiment pass
        rep = bom_sim_fidelity(gene)
        self.assertFalse(rep["faithful"])
        self.assertTrue(any("disagree" in f for f in rep["flags"]), rep["flags"])
        self.assertGreater(float(rep["bom_mass_kg"]), float(rep["sim_mass_kg"]))


class LedgerIsNotSharedWithACopyTests(unittest.TestCase):
    """The ledger has to survive being COPIED, because everything downstream of it reads a copy.

    ``RobotGene.from_dict(gene.to_dict())`` is this repo's copy idiom -- three call sites use it precisely
    because ``ground_gene`` mutates -- and ``to_dict`` only copied the TOP level of ``metadata``. So the copy
    and the original shared the one dict at ``metadata['embodied_mass']``, and grounding the copy rewrote the
    original's ledger in place.

    MEASURED on ``submit_design`` of the taught quadruped: ``gene_validation.validate_gene_design`` grounds
    such a copy AFTER ``session_state.put_robot`` has written the session file, and the held gene came out of
    it with ``balance_of_system_kg == {}`` while its torso still carried the 3.675 kg of battery and wiring
    that entry accounts for. ``structural_link_masses`` then reports that 3.675 kg as aluminium to buy, and the
    BOM bills the same hardware twice -- the exact failure the ledger exists to prevent. It surfaced as a
    session round-trip failure (``test_agent_first`` SessionPersistenceTests) because the DISK copy, written
    before the aliasing write, was the correct one.

    Needs no MuJoCo: a hand-built quadruped is enough to make the balance-of-system estimate fire.
    """

    @staticmethod
    def _quadruped():
        from virturoid.schemas.gene import GeneSegment, RobotGene
        # Sized to land UNDER the quadruped band (12-15 kg) on its own structure + motors -- that is the case
        # in which the balance-of-system estimate is drawn at all, and therefore the case that has a ledger
        # entry to lose. Measured on this fixture: 12.043 kg total, 3.011 kg of it estimated.
        segs = [GeneSegment(name="torso", parent=None, shape="box", length_m=0.4, radius_m=0.05, mass_kg=1.0)]
        for i, leg in enumerate(("fl", "fr", "bl", "br")):
            segs.append(GeneSegment(name=f"{leg}_hip", parent="torso", shape="capsule", length_m=0.16,
                                    radius_m=0.02, mass_kg=0.3, joint_type="revolute", joint_axis=(0, 1, 0)))
            segs.append(GeneSegment(name=f"{leg}_shin", parent=f"{leg}_hip", shape="capsule", length_m=0.18,
                                    radius_m=0.02, mass_kg=0.3, joint_type="revolute", joint_axis=(0, 1, 0)))
            # a WELDED leaf: `_leg_branches` only counts a chain as a leg when it ends in a foot, which is
            # what makes `physical_prior_for` resolve the quadruped band and draw a balance-of-system estimate
            segs.append(GeneSegment(name=f"{leg}_foot", parent=f"{leg}_shin", shape="sphere", length_m=0.05,
                                    radius_m=0.02, mass_kg=0.05, joint_type="fixed",
                                    is_end_effector=(i == 0)))
        return RobotGene(id="alias_quad", species="alias_quad", robot_class="quadruped",
                         base_mount="free", end_effector_type="none", segments=segs)

    def test_grounding_a_copy_cannot_rewrite_the_originals_ledger(self):
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.grounded_physics import _embodiment_record, ground_gene, structural_link_masses
        gene = self._quadruped()
        ground_gene(gene)
        held = {k: dict(v) for k, v in _embodiment_record(gene).items() if isinstance(v, dict)}
        self.assertTrue(held["actuator_kg"], "the fixture must fold at least one motor in, or this proves nothing")
        self.assertTrue(held["balance_of_system_kg"], "the fixture must draw a balance-of-system estimate")
        masses = {s.name: s.mass_kg for s in gene.segments}
        structure = structural_link_masses(gene)

        # The copy idiom, grounded at a DIFFERENT material so its ledger genuinely lands somewhere else --
        # otherwise a converging re-ground could write the same numbers back and hide the sharing.
        copy = RobotGene.from_dict(gene.to_dict())
        ground_gene(copy, material="steel")
        copy_rec = {k: dict(v) for k, v in _embodiment_record(copy).items() if isinstance(v, dict)}
        self.assertNotEqual(copy_rec, held, "the copy must ground to a different ledger for this to prove anything")

        after = {k: v for k, v in _embodiment_record(gene).items() if isinstance(v, dict)}
        self.assertEqual(after, held, "grounding a COPY rewrote the original gene's embodiment ledger")
        self.assertEqual({s.name: s.mass_kg for s in gene.segments}, masses)
        # the ledger and the body still agree, so no folded-in part is about to be re-billed as raw stock
        self.assertEqual(structural_link_masses(gene), structure)

        # ...and the sharp version of the same property, insensitive to what any re-ground happens to compute:
        # a write through the copy's own ledger may not reach the original's.
        _embodiment_record(copy)["balance_of_system_kg"]["torso"] = 42.0
        _embodiment_record(copy)["actuator_kg"].clear()
        self.assertEqual({k: v for k, v in _embodiment_record(gene).items() if isinstance(v, dict)}, held,
                         "the copy is still writing through into the original's metadata")

    def test_a_snapshot_of_a_gene_does_not_move_under_the_gene(self):
        """``to_dict`` is what the session writer persists and what the undo stack keeps as history. Both are
        snapshots, and a snapshot that keeps changing is not one."""
        from virturoid.services.grounded_physics import _embodiment_record, ground_gene
        gene = self._quadruped()
        ground_gene(gene)
        snap = gene.to_dict()
        frozen = {k: dict(v) for k, v in snap["metadata"]["embodied_mass"].items()}
        _embodiment_record(gene)["balance_of_system_kg"].clear()          # any later in-place write
        _embodiment_record(gene)["actuator_kg"]["torso"] = 99.0
        self.assertEqual({k: dict(v) for k, v in snap["metadata"]["embodied_mass"].items()}, frozen)
