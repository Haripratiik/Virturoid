"""General anatomy compiler: the LLM describes a creature's anatomy as a structured graph; ONE compiler
realizes connected, credible geometry — no per-species code. Locks in the dog-from-anatomy fix."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None

from virturoid.services.anatomy_compiler import (
    _AIM_VEC,
    build_from_anatomy,
    generic_creature_gene,
)
from virturoid.services.morphology_composer import compose_robot

DOG = {
    "robot_class": "quadruped", "name": "dog",
    "parts": [
        {"name": "torso", "role": "body", "size": 0.6, "girth": 0.2},
        {"name": "neck", "role": "neck", "parent": "torso", "attach": "front_top", "aim": "forward_up",
         "size": 0.14, "girth": 0.07},
        {"name": "head", "role": "head", "parent": "neck", "attach": "tip", "aim": "forward", "size": 0.12},
        {"name": "snout", "role": "snout", "parent": "head", "attach": "tip", "aim": "forward", "size": 0.07},
        {"name": "tail", "role": "tail", "parent": "torso", "attach": "rear_top", "aim": "back_up", "size": 0.18},
        {"name": "front_leg", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down",
         "size": 0.3, "segments": 3, "symmetry": "left_right"},
        {"name": "front_foot", "role": "foot", "parent": "front_leg", "attach": "tip", "aim": "forward",
         "size": 0.06, "symmetry": "left_right"},
        {"name": "hind_leg", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down",
         "size": 0.3, "segments": 3, "symmetry": "left_right"},
    ],
}


class AnatomyCompilerTests(unittest.TestCase):
    def test_dog_anatomy_builds_a_connected_valid_gene(self):
        g = build_from_anatomy(DOG)
        self.assertEqual(g.validate(), [])                       # a real, connected kinematic tree
        names = {s.name for s in g.segments}
        self.assertIn("torso", names)
        self.assertTrue(any("head" in n for n in names) and any("snout" in n for n in names))
        self.assertTrue(any("tail" in n for n in names))
        # 4 legs (front+hind, mirrored) each 3 segments -> plenty of actuated DOF, not a flat disc.
        self.assertGreaterEqual(len(g.actuated_joints()), 8)
        self.assertEqual(g.robot_class, "quadruped")

    def test_oversized_head_is_capped_to_the_torso(self):
        # R3 proportion band: an LLM graph proposing a head as big as the whole robot must be capped by the
        # compiler so it reads as a sensor head, not a "second body" (the #1 toy tell the styling pass exposed).
        graph = {
            "robot_class": "quadruped", "name": "bighead",
            "parts": [
                {"name": "torso", "role": "body", "size": 0.6, "girth": 0.22},
                {"name": "neck", "role": "neck", "parent": "torso", "attach": "front_top", "aim": "forward_up",
                 "size": 0.12, "girth": 0.30},                  # neck girth absurdly torso-fat -> capped
                {"name": "head", "role": "sensor_head", "parent": "neck", "attach": "tip", "aim": "forward",
                 "size": 0.9, "girth": 0.4},                    # head as long as the whole robot -> capped
                {"name": "front_leg", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down",
                 "size": 0.3, "segments": 3, "symmetry": "left_right"},
                {"name": "hind_leg", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down",
                 "size": 0.3, "segments": 3, "symmetry": "left_right"},
            ],
        }
        g = build_from_anatomy(graph)
        torso = next(s for s in g.segments if s.name == "torso")
        head = next(s for s in g.segments if "head" in s.name)
        neck = next(s for s in g.segments if "neck" in s.name)
        ext = head.length_m + 2.0 * head.radius_m                # capsule visible extent
        self.assertLessEqual(ext, 1.06 * torso.length_m,
                             f"head extent {ext:.3f} must not rival torso length {torso.length_m:.3f}")
        self.assertLess(neck.radius_m, torso.radius_m, "a torso-fat neck must be slimmed below the trunk")

    def test_morphology_gate_flags_a_head_that_rivals_the_trunk(self):
        # The gate MEASURES head/torso extent on every quadruped, and flags a head that rivals the trunk (the
        # backstop for the compiler cap above; drives repair once the gate-last/repair loop lands).
        from virturoid.services.morphology_priors import validate_morphology
        g = build_from_anatomy(DOG)
        rep0 = validate_morphology(g)
        self.assertIn("head_torso_ext", rep0.measured)           # the band is measured on a normal dog
        self.assertLessEqual(rep0.measured["head_torso_ext"], 1.05)
        # Inflate the head past the trunk and re-check: the gate must catch it.
        head = next(s for s in g.segments if "head" in s.name.lower())
        torso = next(s for s in g.segments if s.parent is None)
        head.length_m = 1.4 * float(torso.length_m)
        head.radius_m = 0.5 * float(torso.length_m)
        rep1 = validate_morphology(g)
        self.assertGreater(rep1.measured["head_torso_ext"], 1.05)
        self.assertTrue(any("head extent" in i for i in rep1.issues), rep1.issues)

    def test_leg_clamp_stays_strictly_inside_the_gate_after_thicken(self):
        # D3a: the leg-slenderness clamp must land aspect strictly >= 2.2 (the gate fails on `< 2.2`), not exactly
        # at the 2.2 boundary where round() can nudge it under (the T4 seed3 experiment caught a clamped leg
        # failing its OWN gate by 4e-4). Thicken a quadruped hard -> legs hit the cap -> the gate must still pass.
        from virturoid.services.design_critic import thicken_links
        from virturoid.services.morphology_priors import validate_morphology
        g = generic_creature_gene("a quadruped robot dog", "quadruped")
        fat = thicken_links(g, factor=3.0)                   # aggressive thicken drives legs onto the cap
        rep = validate_morphology(fat)
        self.assertGreaterEqual(rep.measured.get("min_leg_aspect", 9.9), 2.2,
                                f"clamped legs must clear the >=2.2 gate, got {rep.measured}")
        self.assertFalse(any("aspect" in i for i in rep.issues), rep.issues)

    def test_symmetric_child_attaches_to_correct_side_of_symmetric_parent(self):
        # The foot (symmetric) parents the leg (symmetric) — each side's foot must attach to THAT side's leg
        # leaf, not an un-split name. If the leaf registry is wrong, validate() reports an unknown parent.
        g = build_from_anatomy(DOG)
        self.assertEqual(g.validate(), [])
        seg = {s.name: s for s in g.segments}
        feet = [s for s in g.segments if "front_foot" in s.name]
        self.assertTrue(feet)
        for f in feet:
            self.assertIn(f.parent, seg)                         # parent resolves (no 'unknown parent')

    def test_unknown_role_is_rejected_instead_of_silently_becoming_a_limb(self):
        with self.assertRaisesRegex(ValueError, "unsupported anatomy role"):
            build_from_anatomy({"robot_class": "legged", "parts": [
                {"name": "core", "role": "body", "size": 0.3, "girth": 0.12},
                {"name": "frobnicator", "role": "zzz_unknown", "parent": "core", "attach": "front", "size": 0.1}]})

    def test_round_body_and_tentacles_keep_their_own_body_plan(self):
        g = build_from_anatomy({"robot_class": "legged", "parts": [
            {"name": "mantle", "role": "body", "aspect": "round", "size": 0.42, "girth": 0.32},
            {"name": "arm", "role": "tentacle", "parent": "mantle", "attach": "front_bottom",
             "aim": "down_out", "size": 0.30, "girth": 0.035, "segments": 4,
             "joint": "revolute", "curl": 1.2, "symmetry": "left_right"},
        ]})
        self.assertEqual(g.validate(), [])
        self.assertEqual(g.root().shape, "sphere")
        arms = [s for s in g.segments if s.name.startswith("arm_")]
        self.assertEqual(len(arms), 8)
        self.assertFalse(any(s.geometry and s.geometry.get("family") == "extrude" for s in arms))
        self.assertTrue(any(abs(v) > 0 for k, v in g.metadata["rest_pose"].items() if k.startswith("arm_")))

    def _legged(self, name, **bodyfields):
        parts = [{"name": "body", "role": "body", "size": 0.3, "girth": 0.3, **bodyfields}]
        for nm, anchor in (("l1", "front_bottom"), ("l2", "front_mid_bottom"),
                           ("l3", "rear_mid_bottom"), ("l4", "rear_bottom")):
            parts.append({"name": nm, "role": "leg", "parent": "body", "attach": anchor, "aim": "down_out",
                          "size": 0.2, "girth": 0.02, "segments": 3, "symmetry": "left_right", "joint": "revolute"})
        return {"robot_class": "quadruped", "name": name, "parts": parts}

    def test_body_aspect_makes_a_wide_low_torso_not_a_capped_barrel(self):
        # 'wide' must let the torso be broader than the 32%-of-length cap that keeps a normal animal sleek.
        long_root = next(s for s in build_from_anatomy(self._legged("a")).segments if s.parent is None)
        wide_root = next(s for s in build_from_anatomy(self._legged("b", aspect="wide")).segments if s.parent is None)
        flat_root = next(s for s in build_from_anatomy(self._legged("c", aspect="flat")).segments if s.parent is None)
        self.assertGreater(wide_root.radius_m, long_root.radius_m * 1.5)   # cap lifted -> genuinely wider
        self.assertLess(flat_root.length_m, wide_root.length_m)            # flat is shallower (low disc)

    def test_body_and_extruded_limb_dimensions_reach_the_physics_proxy(self):
        g = build_from_anatomy(self._legged("proxy"))
        root = g.root()
        self.assertEqual(root.shape, "box")
        self.assertIsNotNone(root.cross_section)
        self.assertGreater(root.cross_section[0], root.cross_section[1], "torso retains its longitudinal envelope")
        shaped = build_from_anatomy({"robot_class": "legged", "parts": [
            {"name": "body", "role": "body", "size": 0.4, "girth": 0.12},
            {"name": "beam", "role": "arm", "parent": "body", "attach": "front_top", "aim": "forward",
             "size": 0.2, "girth": 0.04, "joint": "revolute"},
        ]})
        beam = shaped.segment("beam")
        self.assertEqual(beam.shape, "box", "extruded beam geometry must not compile as a generic capsule")
        self.assertIsNotNone(beam.cross_section)
        self.assertGreater(beam.cross_section[0], beam.cross_section[1])

    def test_intermediate_anchors_spread_four_leg_pairs_along_the_body(self):
        g = build_from_anatomy(self._legged("spreadtest"))
        firsts = {s.name: s for s in g.segments if s.name.endswith(("l1_l_0", "l2_l_0", "l3_l_0", "l4_l_0"))}
        xs = sorted(round(s.mount_offset[0], 3) for s in firsts.values())
        self.assertEqual(len(g.validate()), 0)
        self.assertEqual(len(set(xs)), 4)                                 # four DISTINCT longitudinal hips
        self.assertTrue(xs[0] < 0 < xs[-1])                              # spans rear(-) to front(+)

    def test_curl_bends_a_multi_segment_chain_at_rest(self):
        graph = {"robot_class": "quadruped", "parts": [
            {"name": "body", "role": "body", "size": 0.4, "girth": 0.12},
            {"name": "tail", "role": "tail", "parent": "body", "attach": "rear_top", "aim": "back_up",
             "size": 0.4, "girth": 0.04, "segments": 6, "joint": "revolute", "curl": 2.4}]}
        pose = build_from_anatomy(graph).metadata.get("rest_pose", {})
        bent = [v for k, v in pose.items() if k.startswith("tail_") and abs(v) > 1e-6]
        self.assertGreaterEqual(len(bent), 4)                            # internal joints carry the arch
        self.assertAlmostEqual(sum(bent), 2.4, places=5)                 # total bend == requested curl

    def test_diagonal_lateral_aims_exist(self):
        for tok in ("forward_out", "back_out", "forward_down_out", "back_down_out", "back_down"):
            self.assertIn(tok, _AIM_VEC)

    def test_head_levels_off_an_angled_neck(self):
        # The head (child of an up-angled neck) must re-aim to its OWN world direction, not blindly continue the
        # neck. Before the fix a non-body part ignored aim and inherited the parent's direction (euler 0,0,0).
        g = build_from_anatomy(DOG)
        head = next(s for s in g.segments if s.name == "head")
        self.assertNotEqual(tuple(round(e, 4) for e in head.mount_euler), (0.0, 0.0, 0.0))

    def test_generic_creature_fallback_is_class_general_not_per_species(self):
        # The offline fallback must NOT be a per-animal catalog: a creature nobody enumerated ("axolotl",
        # "wyvern") still gets a connected, standing GENERIC QUADRUPED — same body for every animal name; the
        # LLM supplies species specificity. No "dog"/"lizard"/"axolotl" branch anywhere.
        def legs(g):
            return sum(1 for s in g.segments if s.name.startswith("leg") and s.name.endswith("_0"))
        for prompt in ("an axolotl robot", "a wyvern robot", "a dog robot", "a lizard robot"):
            g = generic_creature_gene(prompt, "quadruped")
            self.assertIsNotNone(g, prompt)
            self.assertEqual(g.validate(), [], prompt)
            self.assertEqual(legs(g), 4, prompt)                         # one generic quadruped for all animals
        # FUNCTIONAL / CLASS walker prompts defer to the gait-tuned parametric template (it owns leg counts)
        self.assertIsNone(generic_creature_gene("a quadruped walking robot", "quadruped"))
        self.assertIsNone(generic_creature_gene("a six-legged hexapod walking robot", "legged"))
        # non-creature / handled-elsewhere classes opt out
        self.assertIsNone(generic_creature_gene("a 6-dof arm", "manipulator"))

    def test_many_legged_fallbacks_are_not_mislabeled_quadrupeds(self):
        from virturoid.services.anatomy_compiler import _generic_legged_graph

        self.assertEqual("quadruped", _generic_legged_graph(n_pairs=2)["robot_class"])
        self.assertEqual("legged", _generic_legged_graph(n_pairs=3)["robot_class"])
        self.assertEqual("legged", _generic_legged_graph(n_pairs=4)["robot_class"])

    def test_offline_tentacled_prompt_is_not_a_dog_with_eight_feet(self):
        g = generic_creature_gene("an octopus robot with eight tentacles", "legged", n_legs=8)
        self.assertIsNotNone(g)
        self.assertEqual(g.root().shape, "sphere")
        tentacles = [s for s in g.segments if s.name.startswith("tentacle")]
        self.assertEqual(len(tentacles), 32)  # eight mirrored tentacles, four links each
        self.assertFalse(any((s.geometry or {}).get("family") == "extrude" for s in tentacles))
        self.assertEqual(sum(s.joint_type == "revolute" for s in tentacles), 32)
        self.assertIsNone(generic_creature_gene("a humanoid", "humanoid"))

    def test_actuators_are_sized_to_load_not_a_flat_constant(self):
        # Each joint's actuator_torque_nm must reflect the load it bears — NOT one constant everywhere (which
        # would make the BOM pick the same motor for every joint). A weight-bearing leg joint > a tail joint.
        g = build_from_anatomy(DOG)
        tq = {s.name: s.actuator_torque_nm for s in g.actuated_joints()}
        self.assertGreater(len({round(v, 1) for v in tq.values()}), 1)        # differentiated, not flat
        legs = [v for n, v in tq.items() if "leg" in n]
        tails = [v for n, v in tq.items() if "tail" in n]
        self.assertTrue(legs and tails and max(legs) > max(tails))            # a leg bears more than the tail

    def test_detail_and_thickness_give_cad_control(self):
        # The agents can SHAPE a part: 'vented' detail -> chamfer + lightening cutouts; thickness -> a beefier
        # load-bearing limb; and the default body is a multi-feature COMPOUND (shell + deck), not one block.
        g = build_from_anatomy({"robot_class": "quadruped", "parts": [
            {"name": "body", "role": "body", "size": 0.5, "girth": 0.16, "detail": "vented"},
            {"name": "leg", "role": "leg", "parent": "body", "attach": "front_bottom", "aim": "down",
             "size": 0.3, "girth": 0.05, "segments": 3, "symmetry": "left_right", "joint": "revolute",
             "thickness": 1.8, "detail": "rugged"}]})
        body = next(s for s in g.segments if s.parent is None)
        self.assertEqual(body.geometry.get("family"), "compound")          # multi-feature body, not a block
        main = body.geometry["parts"][0]
        self.assertIn("cutouts", main)
        self.assertIn("chamfer", main)                                     # vented detail reached the shell
        leg = next(s for s in g.segments if s.name == "leg_l_0")
        # G6 fidelity band: thickness still thickens the limb, but only WITHIN the slenderness band — a walking
        # leg may never return to the 1:1 sausage stubs the e2e fidelity review rejected.
        g0 = build_from_anatomy({"robot_class": "quadruped", "parts": [
            {"name": "body", "role": "body", "size": 0.5, "girth": 0.16},
            {"name": "leg", "role": "leg", "parent": "body", "attach": "front_bottom", "aim": "down",
             "size": 0.3, "girth": 0.012, "segments": 3, "symmetry": "left_right", "joint": "revolute"}]})
        leg0 = next(s for s in g0.segments if s.name == "leg_l_0")
        self.assertGreater(leg.radius_m, leg0.radius_m)                     # the knob still has effect...
        self.assertLessEqual(leg.radius_m, 0.055 * 0.3 * 1.01)              # ...but capped at the band
        self.assertIn("chamfer", leg.geometry)                             # rugged -> machined bevels

    @unittest.skipUnless(importlib.util.find_spec("build123d"), "needs build123d")
    def test_cad_compound_chamfer_cutouts_realize(self):
        from virturoid.services.cad_geometry import realize_shape
        comp = realize_shape({"family": "compound", "parts": [
            {"family": "box", "length": 0.2, "radius": 0.08},
            {"family": "sphere", "radius": 0.05, "at": (0, 0, 0.12)},
            {"family": "cylinder", "length": 0.3, "radius": 0.02, "at": (0.05, 0, 0), "subtract": True}]})
        self.assertTrue(comp.is_valid and comp.volume > 0)                 # boolean union/subtract worked
        cham = realize_shape({"family": "box", "length": 0.2, "radius": 0.08, "chamfer": 0.015})
        self.assertTrue(cham.is_valid and cham.volume > 0)

    def test_walking_legs_get_real_flat_feet(self):
        # Each walking leg must terminate in a flat foot PAD (an extruded plate), not a downward-poking stub —
        # so the robot has real feet. DOG has 4 three-segment legs -> 4 foot pads.
        g = build_from_anatomy(DOG)
        feet = [s for s in g.segments
                if s.geometry and s.geometry.get("family") == "extrude" and s.joint_type is None]
        self.assertGreaterEqual(len(feet), 4)

    @unittest.skipUnless(_MUJOCO, "needs MuJoCo")
    def test_feet_rest_on_the_floor_not_penetrating(self):
        # standing_spawn_z grounds the DISPLAYED mesh so the lowest visual point sits just above the floor
        # (the foot-penetration bug was the mesh hanging below the primitive collider).
        import mujoco

        from virturoid.services.gene_compiler import (
            _lowest_world_z,
            gene_to_meshed_mjcf,
            standing_spawn_z,
        )
        g = build_from_anatomy(DOG)
        sz = standing_spawn_z(g)
        m = mujoco.MjModel.from_xml_string(gene_to_meshed_mjcf(g, spawn_z=sz, include_floor=False))
        d = mujoco.MjData(m)
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        self.assertGreater(_lowest_world_z(m, d), -0.005)         # on/above the floor, not buried

    def test_compose_robot_uses_anatomy_path_for_a_creature(self):
        # End-to-end: an LLM that emits an anatomy graph drives the general compiler (design_source 'anatomy').
        class _AnatomyLLM:
            def __init__(self, graph):
                self.graph = graph

            def complete_json(self, system, user, schema, max_tokens=None, reasoning_effort=None):
                return self.graph
        g = compose_robot("a dog like robot", llm=_AnatomyLLM(DOG))
        self.assertEqual(g.design_source, "anatomy")
        self.assertEqual(g.validate(), [])


if __name__ == "__main__":
    unittest.main()
