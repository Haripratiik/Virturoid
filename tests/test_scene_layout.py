"""General procedural scene-layout engine: ONE solver realizes ANY room graph (no per-environment code). A prompt
must produce the right rooms, and the realized floor plan must be dimensionally real, connected via doorways, and
A*-navigable — for environments that were NEVER hardcoded (office, clinic, store), proving it is general."""

import unittest

import numpy as np

from virturoid.services.scene_layout import (
    propose_room_graph, realize_scene_spec, generate_scene, ROOM_TYPES, ENV_TEMPLATES)
from virturoid.services.scene_validity import validate_scene_physical


class SceneLayoutTests(unittest.TestCase):
    def test_proposer_maps_prompts_to_rooms(self):
        self.assertIn("warehouse", propose_room_graph("a warehouse for box picking"))
        self.assertIn("bedroom", propose_room_graph("a house for a roomba to vacuum"))
        # an unknown prompt still yields a realizable multi-room space, never empty
        self.assertTrue(propose_room_graph("some strange place", rng=np.random.default_rng(0)))

    def test_generates_valid_navigable_for_many_environments(self):
        # house/warehouse/office/clinic/store — office/clinic/store were never hardcoded, only DATA
        for prompt in ["a house for a roomba", "a warehouse for picking boxes", "an office for a delivery robot",
                       "a clinic for a robot to navigate", "a store for a shelf-scanning robot"]:
            ok = sum(generate_scene(prompt, seed=s)[1]["valid"] for s in range(5))
            self.assertGreaterEqual(ok, 4, f"{prompt}: should reliably generate valid navigable scenes")

    def test_scene_is_dimensionally_real_and_connected(self):
        s, rep = generate_scene("a house for a roomba", seed=2)
        self.assertTrue(rep["valid"])
        walls = [o for o in s.objects if o.object_type == "wall"]
        furn = [o for o in s.objects if o.object_type == "obstacle"]
        self.assertGreaterEqual(len(walls), 6)                    # perimeter + interior partitions
        self.assertTrue(all(w.size_xyz[2] >= 2.0 for w in walls))  # real ~2.44 m walls
        self.assertTrue(furn)                                     # furnished
        self.assertTrue(any(o.object_type == "floor" for o in s.objects))
        # the realized plan is A*-navigable spawn -> goal
        self.assertTrue(validate_scene_physical(s, robot_radius=0.18, run_settle=False)["ok"])

    def test_adding_an_environment_is_data_not_code(self):
        # a brand-new environment defined purely by a room list realizes through the SAME engine
        rooms = ["lobby", "office", "storage", "bathroom", "office"]
        self.assertTrue(all(r in ROOM_TYPES for r in rooms))
        s = realize_scene_spec(rooms, env="research_lab", seed=0)
        self.assertGreaterEqual(s.variation_parameters["n_rooms"], 3)
        self.assertIn("research_lab", s.name)

    def test_warehouse_routes_over_house_substring(self):
        # "warehouse" contains "house" — the proposer must not misroute it to a home
        self.assertEqual(propose_room_graph("a warehouse for pallets"), ENV_TEMPLATES["warehouse"])


if __name__ == "__main__":
    unittest.main()
