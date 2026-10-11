"""The head proportion cap reads the trunk's LENGTH, not its height.

``anatomy_compiler.build_from_anatomy`` caps a head at half the trunk "length", and the number it used was the
second value ``_role_geometry('body', ...)`` returns, which for the root body is the torso's HEIGHT (the trunk
is extruded along +z = height, with its front-back extent in the profile). MEASURED through the MCP server
(get_design_schema -> submit_design -> render_view): an authored quadruped with torso 0.45 / girth 0.10 and a
head of 0.12 / girth 0.05 at front_top compiled a head 0.057 m long (half of the torso's 0.115 m height),
mounted 0.184 m forward inside the 0.45 m torso box, so 1.7 cm of it showed past the trunk. The composer's own
graphs were capped the same way: the offline dog asks for a 0.139 m head and got 0.075 m.

The morphology gate's head check (``morphology_priors.validate_morphology``) measured the head against the same
height, so once the cap was corrected it would have rejected every anatomy quadruped with a properly sized head
-- these tests pin the two together.
"""

import copy
import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None

AUTHORED_DOG = {"robot_class": "quadruped", "name": "agent_dog", "parts": [
    {"name": "torso", "role": "body", "size": 0.45, "girth": 0.10},
    {"name": "head", "role": "head", "parent": "torso", "attach": "front_top", "aim": "forward", "size": 0.12,
     "girth": 0.05},
    {"name": "leg_front", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out",
     "size": 0.40, "girth": 0.022, "segments": 4, "symmetry": "left_right", "joint": "revolute"},
    {"name": "leg_rear", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down_out",
     "size": 0.40, "girth": 0.022, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]}


def _seg(gene, name):
    return next(s for s in gene.segments if s.name == name)


class HeadCapTests(unittest.TestCase):
    def test_an_authored_head_keeps_its_length(self):
        """0.12 m is well inside half of a 0.45 m trunk, so nothing may shorten it."""
        from virturoid.services.anatomy_compiler import build_from_anatomy
        g = build_from_anatomy(copy.deepcopy(AUTHORED_DOG))
        torso, head = _seg(g, "torso"), _seg(g, "head")
        self.assertAlmostEqual(2.0 * torso.cross_section[0], 0.45, places=6)     # the trunk's real length
        self.assertLess(torso.length_m, 0.12, "the trunk's length_m is its HEIGHT; this test needs it short")
        self.assertAlmostEqual(head.length_m, 0.12, places=6)
        # The girth cap is untouched and like for like: the head's half-width (0.92 x girth) against the trunk's
        # realized half-width, so the 0.05 the agent authored is held to 0.55 x 0.05 m.
        self.assertAlmostEqual(head.radius_m, round(0.92 * 0.55 * 0.05, 4), places=4)

    def test_the_cap_still_binds_on_the_trunk_length(self):
        from virturoid.services.anatomy_compiler import build_from_anatomy
        graph = copy.deepcopy(AUTHORED_DOG)
        graph["parts"][1]["size"] = 0.40                       # a head nearly as long as the trunk
        head = _seg(build_from_anatomy(graph), "head")
        self.assertAlmostEqual(head.length_m, 0.5 * 0.45, places=6)

    def test_the_offline_dog_gets_the_head_its_graph_asks_for(self):
        """The generic legged graph asks for a head 0.24 x the trunk length; the cap used to cut it to half the
        torso's height (0.075 m on the 0.58 m dog)."""
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a robot dog", ensure_walkable=False)
        torso = g.root()
        head = next(s for s in g.segments if s.name == "head")
        trunk_len = 2.0 * float(torso.cross_section[0])
        self.assertAlmostEqual(head.length_m, round(0.24 * trunk_len, 4), places=3)
        self.assertGreater(head.length_m, 0.5 * float(torso.length_m) + 0.03)   # past the old height cap

    def test_the_morphology_gate_measures_the_head_against_the_same_length(self):
        """With the cap corrected, a gate still measuring against the torso height reads the dog's head as 1.37x
        its torso and rejects it (on the LLM path a rejected anatomy is replaced by a generic body)."""
        from virturoid.services.agent_design_tools import _EXAMPLE_QUAD
        from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.morphology_priors import validate_morphology
        bodies = {"offline dog": compose_robot("a robot dog", ensure_walkable=False),
                  "schema example": build_from_anatomy(copy.deepcopy(_EXAMPLE_QUAD)),
                  "authored dog": build_from_anatomy(copy.deepcopy(AUTHORED_DOG)),
                  "walkable template": build_from_anatomy(_generic_legged_graph(n_pairs=2, fan=True, girth=0.18))}
        for name, gene in bodies.items():
            with self.subTest(body=name):
                rep = validate_morphology(gene)
                self.assertLessEqual(rep.measured["head_torso_ext"], 0.5, rep.measured)
                self.assertFalse([i for i in rep.issues if "head extent" in i], rep.issues)


@unittest.skipUnless(_MUJOCO, "measuring the compiled body needs MuJoCo")
class HeadIsVisibleTests(unittest.TestCase):
    def test_the_authored_head_reaches_past_the_front_of_the_trunk(self):
        """What the render shows: the head's front face against the torso's, in the compiled model at rest."""
        import mujoco
        import numpy as np

        from virturoid.services.anatomy_compiler import build_from_anatomy
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        g = build_from_anatomy(copy.deepcopy(AUTHORED_DOG))
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, spawn_z=standing_spawn_z(g)))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)

        def front_x(body):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            aabb = m.geom_aabb.reshape(m.ngeom, 6)
            xs = [float((d.geom_xpos[i] + (aabb[i, :3] + signs * aabb[i, 3:]) @ d.geom_xmat[i].reshape(3, 3).T)
                        [:, 0].max()) for i in range(m.ngeom) if int(m.geom_bodyid[i]) == bid]
            return max(xs)
        past = front_x("head") - front_x("torso")
        # 0.12 m head mounted 0.82 x 0.225 m forward: 0.0795 m of it past the trunk (it was 0.017 m)
        self.assertGreater(past, 0.07, f"only {past:.3f} m of the head shows past the trunk")


if __name__ == "__main__":
    unittest.main()
