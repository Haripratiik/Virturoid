"""Part B: the ingestion agent — a user's EXISTING robot (CAD/BOM/URDF + NLP) becomes ONE editable robot.

Two layers: (1) nlp_properties turns a description into typed, grammar-paired materials + payload + DOF; (2)
ingest_project orchestrates classify -> import -> NLP -> apply -> hold. Offline; the URDF path needs MuJoCo
(it compiles the model), the description-only path does not.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_URDF = """<?xml version="1.0"?>
<robot name="arm2">
  <link name="base_link"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="upper"><inertial><mass value="0.6"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.04 0.04 0.3"/></geometry></collision></link>
  <link name="forearm"><inertial><mass value="0.4"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.03 0.03 0.25"/></geometry></collision></link>
  <joint name="pan" type="revolute"><parent link="base_link"/><child link="upper"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="forearm"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/></joint>
</robot>"""


class NlpPropertiesTests(unittest.TestCase):
    def _x(self, text):
        from virturoid.services.nlp_properties import extract_properties
        return extract_properties(text)

    def test_materials_paired_to_the_right_part(self):
        p = self._x("aluminum body, carbon-fiber legs")
        got = {m["group"]: m["material"] for m in p.materials}
        self.assertEqual(got.get("torso"), "aluminum")          # "aluminum body" -> torso, NOT nearest-word confusion
        self.assertEqual(got.get("legs"), "carbon_fiber")

    def test_copula_and_multipart(self):
        got = {m["group"]: m["material"] for m in self._x("the legs are titanium and the arms are steel").materials}
        self.assertEqual(got.get("legs"), "titanium")
        self.assertEqual(got.get("arms"), "steel")

    def test_lone_material_is_whole_robot(self):
        p = self._x("it is made of aluminum")
        self.assertEqual(p.materials[0]["group"], "all")
        self.assertEqual(p.materials[0]["material"], "aluminum")

    def test_payload_units_and_dof(self):
        self.assertEqual(self._x("carries a 5 kg payload").payload_kg, 5.0)
        self.assertEqual(self._x("can lift 10 lbs").payload_kg, round(10 * 0.45359237, 2))
        self.assertEqual(self._x("a 6-DOF arm").dof, 6)

    def test_ops_are_editor_ready(self):
        ops = self._x("carbon-fiber legs that carry 4 kg").ops
        names = [o["op"] for o in ops]
        self.assertIn("set_material", names)
        self.assertIn("set_payload", names)


class PayloadIsNotTheRobotsOwnMassTests(unittest.TestCase):
    """A stated MASS is not a payload.

    The parser scored every mass figure as (has-a-cue-word-nearby, kilograms) and took the ``max``, with the
    comment "among equals, the larger load (the binding requirement)". Both numbers in a sentence fall inside the
    same +/-30 / 18-char cue window, so the EXPLICITLY LABELLED payload lost to the bigger bare number. Measured
    at HEAD (2026-08-01):

        "Our Unitree G1 humanoid, 1.3 m tall, 35 kg, 29 joints."  -> set_payload 35.0 kg
        "A telescoping arm, 24 kg, 2 kg payload."                 -> set_payload 24.0 kg  (the 2 kg lost)
        "Our Boston Dynamics Spot quadruped, 32 kg, 12 joints."   -> set_payload 32.0 kg

    ``set_payload`` is not cosmetic -- it raises joint torques, re-grounds to bigger motors and adds mass -- so a
    misparse silently RE-SPECS the customer's robot, and then emits its own fabricated "N joint(s) exceed the
    strongest catalog motor for this payload" warning off the number it invented.

    Three rules pinned here: a labelled payload always beats a bare mass; a bare mass is never silently adopted;
    genuine ambiguity is SURFACED in ``notes`` rather than guessed. The sibling parser in ``input_evidence``
    already works this way (a self-weight budget becomes a conflict + an intake question, not a silent value).
    """

    def setUp(self):
        os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="ingest_pay_")

    def _x(self, text):
        from virturoid.services.nlp_properties import extract_properties
        return extract_properties(text)

    def test_a_labelled_payload_beats_a_bare_mass(self):
        p = self._x("A telescoping arm, 24 kg, 2 kg payload.")
        self.assertEqual(p.payload_kg, 2.0, "the labelled 2 kg payload lost to the arm's own 24 kg mass")
        self.assertIn("payload", (p.payload_evidence or ""))

    def test_the_robots_own_mass_is_never_applied_as_a_payload(self):
        for text in ("Our Unitree G1 humanoid, 1.3 m tall, 35 kg, 29 joints.",
                     "Our Boston Dynamics Spot quadruped, 32 kg, 12 joints."):
            with self.subTest(text=text):
                p = self._x(text)
                self.assertIsNone(p.payload_kg, f"read the robot's own mass as a payload: {p.payload_evidence!r}")
                self.assertNotIn("set_payload", [o["op"] for o in p.ops],
                                 "an unlabelled mass must not silently upsize the customer's actuators")

    def test_an_unlabelled_mass_is_surfaced_not_guessed(self):
        p = self._x("Our Unitree G1 humanoid, 1.3 m tall, 35 kg, 29 joints.")
        blob = " ".join(p.notes).lower()
        self.assertTrue(p.notes, "an ambiguous mass must be surfaced, not silently dropped either")
        self.assertIn("35", blob)
        self.assertIn("payload", blob)

    def test_an_explicit_weight_word_marks_the_robots_own_mass(self):
        for text in ("the robot weighs 12 kg", "total mass 35 kg", "dry weight of 18 kg"):
            with self.subTest(text=text):
                self.assertIsNone(self._x(text).payload_kg)

    def test_a_carry_verb_binds_forward_to_the_number_it_governs(self):
        # "carries" governs the 5 kg AFTER it, never the 40 kg base it is predicated of
        p = self._x("a 40 kg base that carries 5 kg")
        self.assertEqual(p.payload_kg, 5.0)

    def test_among_genuine_payloads_the_larger_still_wins(self):
        # the original intent, kept: when BOTH numbers are really payloads, the binding requirement is the bigger
        self.assertEqual(self._x("carries 5 kg, rated for a 10 kg payload").payload_kg, 10.0)

    def test_the_labelled_forms_that_already_worked_still_work(self):
        self.assertEqual(self._x("carries a 5 kg payload").payload_kg, 5.0)
        self.assertEqual(self._x("can lift 10 lbs").payload_kg, round(10 * 0.45359237, 2))
        self.assertEqual(self._x("payload: 10 kg").payload_kg, 10.0)
        self.assertEqual(self._x("a payload of 3 kg").payload_kg, 3.0)
        self.assertEqual(self._x("aluminum body, carbon-fiber legs, 6 kg payload").payload_kg, 6.0)
        self.assertEqual(self._x("lifts 2 kg, 2-DOF").payload_kg, 2.0)
        self.assertEqual(self._x("10 kg load capacity").payload_kg, 10.0)

    def test_an_ingest_does_not_apply_the_robots_own_mass(self):
        from virturoid.services.input_training_tools import _ingest_project
        r = _ingest_project({"description": "Our Spot-like quadruped, 32 kg, aluminum body, 12 joints."})
        self.assertIsNone(r["payload_kg"], "the ingest applied the robot's own 32 kg as a payload")
        self.assertTrue(any("payload" in n.lower() for n in r["notes"]),
                        f"the unread mass must reach the customer-visible notes: {r['notes']}")


class IngestProjectTests(unittest.TestCase):
    def setUp(self):
        os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="ingest_sess_")

    def test_description_only_composes_and_applies(self):
        # no files, just words -> still yields ONE editable held robot with the stated materials + payload applied
        from virturoid.services.input_training_tools import _ingest_project
        r = _ingest_project({"description": "a quadruped dog with an aluminum body, carbon-fiber legs, carries 6 kg"})
        self.assertTrue(r["robot_id"], r.get("warnings"))
        groups = {m["group"]: m["material"] for m in r["materials_applied"]}
        self.assertEqual(groups.get("legs"), "carbon_fiber")
        self.assertEqual(groups.get("torso"), "aluminum")
        self.assertEqual(r["payload_kg"], 6.0)
        # the held robot is real + editable
        from virturoid.services import session_state as S
        self.assertIsNotNone(S.get_robot(r["robot_id"]))

    def test_no_fabrication_skips_missing_parts(self):
        # a wheeled base has no "wings" -> the op is SKIPPED + noted, never fabricated, ingest still succeeds
        from virturoid.services.input_training_tools import _ingest_project
        r = _ingest_project({"description": "a small wheeled rover, steel wheels, carbon-fiber wings, lifts 2 kg"})
        self.assertTrue(r["robot_id"])
        skipped_groups = {s["args"].get("group") for s in r["skipped_ops"]}
        applied_groups = {m["group"] for m in r["materials_applied"]}
        # wings can't exist on a rover -> either skipped or simply not applied; must NOT be fabricated
        self.assertNotIn("wings", applied_groups)
        self.assertNotIn("head", applied_groups)

    def test_description_only_is_not_reported_as_a_substitution(self):
        # composing from words is exactly what a description-only ingest ASKED for -> no alarm, but the lane must
        # still name what the held body actually is, never an import lane.
        from virturoid.services.input_training_tools import _ingest_project
        r = _ingest_project({"description": "a quadruped dog with an aluminum body"})
        self.assertTrue(r["robot_id"])
        self.assertTrue(r.get("ok"))
        self.assertNotIn("substituted", r)
        self.assertEqual(r["lane_used"], "composed_from_description")
        self.assertNotIn("SUBSTITUTED", r["summary"])

    def test_composed_fallback_on_a_real_project_is_a_hard_failure(self):
        """The Spot symptom. A customer hands us FILES; nothing importable comes out of them; we compose a robot
        from their sentence instead. That must be impossible to mistake for their robot:

            before -> ok=True  "Ingested -> robot_5eecc737: lane=faithful, 0 material(s) applied,
                                payload=None kg, 0 warning(s). Ready to edit/verify."
                      with the only trace of the swap buried in notes[].
        """
        from virturoid.services.input_training_tools import _ingest_project
        d = tempfile.mkdtemp(prefix="proj_")
        # a URDF that names links no <link> declares -> no importer can build a body from it
        Path(d, "robot.urdf").write_text(
            '<?xml version="1.0"?><robot name="broken"><link name="a"/>'
            '<joint name="j" type="revolute"><parent link="nope"/><child link="alsonope"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>',
            encoding="utf-8")
        r = _ingest_project({"project_path": d, "description": "a quadruped robot dog"})

        self.assertTrue(r["robot_id"])                       # the composed body is still usable...
        self.assertIs(r.get("ok"), False)                    # ...but the ingest did NOT succeed
        self.assertTrue(r.get("substituted"))
        self.assertNotEqual(r.get("lane_used"), "faithful")  # never "faithful" for a body we generated
        self.assertTrue(r["warnings"], "a substitution can never report 0 warning(s)")
        self.assertIn("SUBSTITUTED", r["warnings"][0])
        self.assertIn("SUBSTITUTED", r["summary"])           # survives into the customer-facing line
        self.assertIn("NOT YOUR ROBOT", r["summary"])
        self.assertIn("how_to_fix", r["substitution"])
        self.assertIn(r["substitution"]["reason"], r["summary"])

    @unittest.skipUnless(_MUJOCO, "needs MuJoCo: the bug depended on the faithful lane compiling the scene")
    def test_a_floor_only_scene_never_reports_a_faithful_lane(self):
        """The exact mechanism behind `lane=faithful` on a generated Spot: boston_dynamics_spot/scene.xml is a
        floor plus an <include>. import_model COMPILES it (a plane is a valid model) so the faithful lane went
        green, while import_robot correctly refused it (no <body>) so the held robot was composed -- and the
        summary reported the compiled scene's lane for the generated body."""
        from virturoid.services.input_training_tools import _ingest_project
        d = tempfile.mkdtemp(prefix="scene_")
        Path(d, "scene.xml").write_text(
            '<mujoco model="scene"><worldbody><geom name="floor" size="0 0 .05" type="plane"/>'
            "</worldbody></mujoco>", encoding="utf-8")
        r = _ingest_project({"project_path": d, "description": "a quadruped robot dog"})
        self.assertNotEqual(r.get("lane_used"), "faithful")
        self.assertIs(r.get("ok"), False)
        self.assertTrue(r.get("substituted"))
        self.assertNotIn("Ready to edit/verify", r["summary"])

    def test_a_usd_only_project_names_the_format_and_the_conversion(self):
        """USD/SDF ingest was advertised (`_EXT_MAP` classified .usd as a robot_model; the blocker string read
        "(URDF/MJCF/SDF/USD)") and does not exist. The customer most likely to arrive with a .usd is an Isaac
        engineer -- our stated target user -- and what they used to get was "none of the 1 model file(s) in the
        project could be imported", which names neither the reason nor the one step that works."""
        from virturoid.services.input_training_tools import _ingest_project
        d = tempfile.mkdtemp(prefix="usdproj_")
        Path(d, "spot.usd").write_text("#usda 1.0\n", encoding="utf-8")
        r = _ingest_project({"project_path": d, "description": "a four legged robot dog"})
        self.assertIs(r.get("ok"), False)
        self.assertTrue(r.get("substituted"))
        self.assertEqual(r["substitution"]["unsupported_model_files"], ["spot.usd"])
        self.assertIn("cannot read", r["substitution"]["reason"])
        self.assertIn("URDF", r["substitution"]["reason"])
        self.assertNotIn("no robot description file", r["substitution"]["reason"])
        fix = r["substitution"]["how_to_fix"]
        self.assertIn("no USD importer", fix)          # the truth
        self.assertIn("Isaac", fix)                    # ...and the step that works, not a bare "unsupported"

    @unittest.skipUnless(_MUJOCO, "URDF import needs MuJoCo to compile the model")
    def test_folder_with_urdf_and_description(self):
        from virturoid.services import session_state as S
        from virturoid.services.input_training_tools import _ingest_project
        d = tempfile.mkdtemp(prefix="proj_")
        Path(d, "robot.urdf").write_text(_URDF, encoding="utf-8")
        r = _ingest_project({"project_path": d,
                             "description": "aluminum base, carbon-fiber forearm, lifts 2 kg, 2-DOF"})
        self.assertTrue(r["robot_id"], r.get("warnings"))
        self.assertIn("import", r, "the URDF model must be imported into an editable gene")
        self.assertEqual(r["payload_kg"], 2.0)
        self.assertIsNotNone(S.get_robot(r["robot_id"]))
        # at least one stated material landed on a part that exists (base_link=torso and/or forearm=arms)
        self.assertGreaterEqual(len(r["materials_applied"]), 1, r)


if __name__ == "__main__":
    unittest.main()
