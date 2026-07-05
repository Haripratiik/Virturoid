"""S3 structural scene families: a family must contain STRUCTURALLY-distinct scenes (different object counts /
layouts / topologies — not seed-jitter of one template), the held-out pool must be structurally DISJOINT from
train (the honest generalization split), every scene must carry realistic dimensions + compile in MuJoCo, and the
whole thing must be deterministic."""

import unittest

from virturoid.services.scene_family import generate_family, StructureKey


def _compile(scene, materials=True):
    import mujoco
    from virturoid.services.mujoco_exporter import _scene_objects_xml
    body = "\n".join(_scene_objects_xml(scene.objects))
    assets = ('<asset>' + "".join(
        f'<material name="{m}" rgba="0.6 0.6 0.6 1"/>' for m in
        ("mat_gray", "mat_red", "mat_blue")) + '</asset>')
    xml = f'<mujoco>{assets}<worldbody>{body}</worldbody></mujoco>'
    return mujoco.MjModel.from_xml_string(xml)


class SceneFamilyTests(unittest.TestCase):
    def test_manip_family_is_structurally_diverse(self):
        fam = generate_family("pick_place_sort", n_train=8, n_held_out=3, seed=1)
        self.assertEqual(len(fam.train), 8)
        self.assertEqual(fam.n_distinct_train, 8)                  # all 8 are DIFFERENT structures, not jitters
        # object counts actually vary across the family (the thing that was constant before)
        counts = {s.variation_parameters["n_objects"] for s in fam.train}
        layouts = {s.variation_parameters["layout"] for s in fam.train}
        self.assertGreater(len(counts) + len(layouts), 3, "family barely varies structurally")

    def test_held_out_is_disjoint_from_train(self):
        fam = generate_family("pick_place_sort", n_train=8, n_held_out=3, seed=2)
        self.assertTrue(fam.disjoint, "held-out structures leaked into train")
        self.assertEqual(len(set(fam.train_keys) & set(fam.held_out_keys)), 0)
        self.assertEqual(len(fam.held_out), 3)

    def test_nav_family_varies_topology(self):
        fam = generate_family("navigation", n_train=6, n_held_out=2, seed=3)
        topos = {s.variation_parameters["topology"] for s in fam.train}
        self.assertGreaterEqual(len(topos), 2, "nav family should span multiple corridor topologies")
        # corridor width is realistic (>= ADA 0.915 m), not the old 0.52 m
        for s in fam.train:
            self.assertGreaterEqual(s.variation_parameters["corridor_width_m"], 0.9)

    def test_scenes_have_real_dimensions(self):
        fam = generate_family("navigation", n_train=3, n_held_out=1, seed=4)
        for scene in fam.train:
            walls = [o for o in scene.objects if o.object_type == "wall"]
            self.assertTrue(walls)
            for w in walls:
                self.assertIsNotNone(w.size_xyz)
                self.assertGreaterEqual(w.size_xyz[2], 2.0)        # walls are ~2.4 m tall (z), not 0.32 m

    def test_families_compile_in_mujoco(self):
        try:
            import mujoco  # noqa: F401
        except Exception:  # noqa: BLE001
            self.skipTest("mujoco not installed")
        for task in ("pick_place_sort", "navigation"):
            fam = generate_family(task, n_train=3, n_held_out=1, seed=5)
            for scene in fam.train + fam.held_out:
                _compile(scene)                                    # raises if any scene is malformed

    def test_determinism(self):
        a = generate_family("stack", n_train=5, n_held_out=2, seed=7)
        b = generate_family("stack", n_train=5, n_held_out=2, seed=7)
        self.assertEqual(a.train_keys, b.train_keys)
        self.assertEqual(a.held_out_keys, b.held_out_keys)


if __name__ == "__main__":
    unittest.main()
