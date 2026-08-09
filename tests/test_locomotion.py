"""Legged locomotion (§11/§32.1): a composed quadruped stands + trots forward, evaluated by distance."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class LocomotionTests(unittest.TestCase):
    def test_backward_or_sideways_progress_never_certifies_a_forward_walk(self):
        from virturoid.services.locomotion_controller import locomotion_status

        self.assertEqual(locomotion_status(-0.6, True), "stalled")
        self.assertEqual(locomotion_status(0.0, True), "stalled")
        self.assertEqual(locomotion_status(0.11, True), "walked")
        self.assertEqual(locomotion_status(1.0, False), "fell")

    def test_composed_quadruped_walks_forward(self):
        import mujoco
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.locomotion_controller import run_locomotion_episode

        g = compose_robot("a quadruped walking robot")
        self.assertEqual(g.robot_class, "quadruped")
        self.assertEqual(g.base_mount, "free")                    # floating base (so it can move)
        # Real-quadruped accuracy (Go1-class): 4 legs x (abduction + thigh + shin + foot) = 16 segments,
        # 3 actuated joints/leg (foot welded) = 12 DOF — matching a real quadruped baseline, not 8.
        self.assertEqual(sum(1 for s in g.segments if "leg" in s.name), 16)
        # LEG DOF only -- the body also articulates a neck and tail, which are not part of the Go1-class
        # leg-DOF baseline this asserts (measured total: 14 = 12 leg + neck + tail).
        self.assertEqual(len([s for s in g.actuated_joints() if s.name.startswith("leg")]), 12)
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        r = run_locomotion_episode(mj)
        self.assertTrue(r["upright"])                             # legs point down -> it stands
        # `distance_m` IS NOT A WALK. It is norm(p1 - p0) — the UNSIGNED planar travel that
        # `locomotion_status`'s own docstring, and `test_backward_or_sideways_progress_never_certifies_a_
        # forward_walk` twenty lines above, both say cannot certify a forward walk. This line used to read
        # `assertGreater(r["distance_m"], 0.1)` inside a test named `..._walks_forward`, and MEASURED
        # 2026-08-09 the body it was passing on travels forward_m -0.793 with distance_m 0.793 and
        # status 'stalled': the bare trot drives the AUTHORED quadruped BACKWARD, and the assertion could
        # not tell. So this line now claims only what it can see — the trot machinery drives the body —
        # and the FORWARD claim is made below, on signed travel, where it is true.
        self.assertGreater(r["distance_m"], 0.1, "the bare trot must actually drive the body somewhere")
        # The PRODUCT verdict for a quadruped is the GAIT-AWARE evaluate_robot (trot, else the statically-stable
        # crawl), which is what verify_robot ships. B1 scales the fanned walkable template to the body's size for
        # per-prompt differentiation; that wide-stance body DRIFTS under the bare trot (direction is the learned
        # policy's job, as the distance check already concedes) but WALKS under the crawl -- exactly why the crawl
        # gait exists ([[walking-breakthrough-abduction]]). Assert the walk the product actually earns, gait-aware.
        #
        # ASK FOR THE WALKABLE BODY (#285). ``compose_robot`` used to run the walkability gate inside itself, so
        # a bare compose silently returned the fanned template and this line measured THAT. It no longer does:
        # composing returns the AUTHORED body, and the gate -- unchanged -- runs for a caller who asks for it
        # (``ensure_walkable=True``) or for ``create_robot``, which grounds and fits an operating point first and
        # then decides. Measured on the authored body at the shipped freq 1.5 / kp 32 it travels -0.17 m and rolls
        # over, and with an operating point of its own it is a CREDIBLE WALK -- which is precisely why the
        # decision moved to a caller that has one. This assertion is about the WALKABLE body, so it asks for one.
        from virturoid.services.task_matched_eval import evaluate_robot
        w = compose_robot("a quadruped walking robot", ensure_walkable=True)
        ev = evaluate_robot(w)
        # THE FORWARD CLAIM, and the one this test is named for. `evaluate_robot`'s locomotion branch scores
        # `max(0.0, forward_m) if upright else 0.0` — SIGNED forward, clamped, upright-gated — so the metric
        # name is asserted too: if it ever reverts to unsigned planar travel this line must fail, not pass.
        self.assertEqual(ev["metric"], "forward_m",
                         "the product's locomotion score must be SIGNED forward, never unsigned distance")
        self.assertGreaterEqual(float(ev.get("value", 0.0)), 0.5,
                                "the composed quadruped must walk under the product's gait-aware verdict")

    def test_leg_count_is_parametric(self):
        # "build whatever I want from scratch": a hexapod gets 6 legs (leg count is a PARAMETER), not the
        # 4-leg quadruped template — while the quadruped stays exactly 4 legs / 8 DOF.
        from virturoid.services.morphology_composer import compose_robot
        # Count LEG DOF, not total DOF: the composer now also articulates a neck and tail on quadrupeds (richer
        # anatomy), so a whole-body count measures "how much extra anatomy exists", not the parametric property
        # this test is about. Measured: quad = 12 leg joints + neck + tail = 14 actuated.
        def leg_dof(g):
            return len([s for s in g.actuated_joints() if s.name.startswith("leg")])
        h = compose_robot("a six-legged hexapod walking robot")
        leg_groups = {s.name.split("_")[0] for s in h.segments if s.name.startswith("leg")}
        self.assertEqual(len(leg_groups), 6)                  # 6 distinct legs
        self.assertEqual(leg_dof(h), 18)                      # 6 legs x 3 actuated joints (abduction+hip+knee)
        q = compose_robot("a quadruped walking robot")
        self.assertEqual(leg_dof(q), 12)                      # quadruped = 4 legs x 3 DOF (neck/tail excluded)

    def test_free_bodies_spawn_standing_not_ejected(self):
        # The spawn-height fix: free bodies spawn on their feet, not penetrating the floor. With the old fixed
        # 0.1 m spawn a humanoid launched to z~2.1 under the contact solver; standing_spawn_z keeps it put.
        import mujoco
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        from virturoid.services.morph_graph import encode_robot
        for prompt in ("a quadruped walking robot", "a humanoid robot"):
            g = compose_robot(prompt)
            xml = compile_gene_to_mjcf(g, include_floor=True, spawn_z=standing_spawn_z(g))
            mj = mujoco.MjModel.from_xml_string(xml); bq = encode_robot(mj).base_qadr
            d = mujoco.MjData(mj); mujoco.mj_forward(mj, d)
            # not penetrating the floor — measured from each geom's TRUE oriented AABB. (geom_rbound is a
            # bounding SPHERE: for a long flat foot box it dips ~half the foot length below the sole even when
            # the foot rests flat, so it false-positives once the feet are correctly grounded.)
            import numpy as np
            lowest = 1e9
            for i in range(mj.ngeom):
                if mj.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                c, h = mj.geom_aabb[i, :3], mj.geom_aabb[i, 3:]
                R = d.geom_xmat[i].reshape(3, 3)
                lowest = min(lowest, float(d.geom_xpos[i, 2] + R[2] @ c - np.abs(R[2]) @ h))
            self.assertGreater(lowest, -0.03)
            z0 = float(d.qpos[bq + 2])
            for _ in range(150):
                mujoco.mj_step(mj, d)
            self.assertLess(float(d.qpos[bq + 2]), z0 + 0.5)                              # not ejected upward
            q = d.qpos[bq + 3:bq + 7]
            self.assertGreater(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), 0.5)                  # stays upright

    def test_legs_point_down(self):
        # the composition fix: quadruped legs are oriented downward (else the body can't stand)
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        hips = [s for s in g.segments if s.name.endswith("_0") and "leg" in s.name]
        self.assertTrue(hips and all(abs(h.mount_euler[0]) > 1.0 for h in hips))   # rotated to point down


if __name__ == "__main__":
    unittest.main()
