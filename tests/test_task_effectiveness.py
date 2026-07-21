"""Task EFFECTIVENESS regression locks (the scene-creation + training goal): a robot must reliably COMPLETE a
user task, not just attempt it. These pin the fixes that took the representative task suite from 4/6 to 6/6 and
lifted per-task success rates — locomotion (route to a walkable body + learned MorphPolicy), maze (rover-sized
cells + turn-in-place), pick_place (graspable-zone scenes + auto reach-mode), spray (firm joint-PD coverage),
and navigation (distance-scaled horizon). See [[task-effectiveness-loop]]."""

import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class RoutingTests(unittest.TestCase):
    def test_locomotion_intent_routes_to_a_walkable_body(self):
        # "walk forward" used to build the STUNTED generic anatomy quad (huge body, stub legs — unwalkable).
        # Locomotion INTENT must route to a body that ACTUALLY WALKS.
        # This asserted species == "quadruped.composed" as a proxy for "walkable". That proxy went stale: the
        # anatomy route now produces a credible walker too (measured: quadruped.anatomy, 20 segments,
        # CREDIBLE WALK 0.501 m), so the string check failed while the capability it stands for was fine.
        # Assert the CAPABILITY instead — that is what the test is named for, and it cannot go stale.
        from virturoid.services.morph_policy import crawl_gait_rollout
        from virturoid.services.morphology_composer import compose_robot
        for prompt in ("walk forward across the floor", "make a robot walk forward", "a robot that crawls"):
            g = compose_robot(prompt, llm=None)
            self.assertEqual(g.robot_class, "quadruped", prompt)
            # Abduction authority, checked frame-ROBUSTLY: the anatomy compiler expresses each leg's axes in that
            # segment's LOCAL frame, so an angled leg's abduction axis reads e.g. (0.8, 0, -0.5) rather than the
            # exact world (1,0,0) this used to demand. Same physical joint; assert the dominant lateral (roll)
            # component instead of one hard-coded vector.
            axes = [tuple(float(a) for a in (s.joint_axis or (0, 0, 0))) for s in g.actuated_joints()]
            self.assertTrue(any(abs(a[0]) >= 0.5 for a in axes),
                            f"{prompt}: a walkable quad needs hip-abduction (roll) authority, axes={axes}")
            r = crawl_gait_rollout(g, steps=600)
            self.assertGreater(float(r.get("forward", 0.0)), 0.15,
                               f"{prompt}: locomotion intent must yield a body that travels, got {r.get('forward')}")

    def test_animal_prompts_still_get_the_anatomy_creature(self):
        # the routing widening must NOT hijack animal prompts — "a dog" is still a credible anatomy creature.
        from virturoid.services.morphology_composer import compose_robot
        self.assertEqual(compose_robot("a dog", llm=None).species, "quadruped.anatomy")

    def test_diverse_phrasings_route_to_the_right_skill(self):
        # the offline classifier must not silently MISROUTE a task (a "trot" request used to build a
        # manipulator and run pick_place — succeeding at the WRONG task). Verbs across all skill families
        # reach the correct skill: locomotion, navigation, maze, spray, manipulation.
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_proposer import propose_task
        cases = [("trot ahead", "locomote"), ("traverse the floor", "locomote"),
                 ("walk across the room", "locomote"), ("go to the waypoint", "navigate"),
                 ("reach the destination", "navigate"), ("escape the maze", "solve_maze"),
                 ("coat the surface", "spray_coverage"), ("varnish the table", "spray_coverage"),
                 ("sort the cubes into bins", "pick_place"), ("grab the box and lift it", "grasp_lift")]
        for prompt, expected_skill in cases:
            g = compose_robot(prompt, llm=None)
            skills = [s.skill for s in propose_task(prompt, g, llm=None).steps]
            self.assertIn(expected_skill, skills, f"{prompt!r} misrouted to {skills}")


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class SkillEffectivenessTests(unittest.TestCase):
    def _arm(self, prompt="a tabletop robot arm"):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot(prompt, llm=None)

    def test_maze_solves_for_the_rover(self):
        # rover-sized cells (room to turn) + turn-in-place control + scaled horizon -> the rover reaches the
        # goal (it used to stall at 3/17 waypoints). cell is widened to the rover's turning footprint.
        from virturoid.services.maze import cell_for_rover, solve_maze
        from virturoid.services.morphology_composer import compose_robot
        rover = compose_robot("solve the maze to reach the exit", llm=None)
        self.assertGreater(cell_for_rover(rover), 0.85)                      # widened beyond the old fixed cell
        self.assertTrue(solve_maze(rover, n=3, seed=0)["solved"])

    def test_navigation_reaches_a_far_goal(self):
        # the step budget scales with distance, so a far (2.2 m) goal is reachable, not just near ones.
        from virturoid.services.capability_registry import _run_navigate
        out = _run_navigate(self._arm("a wheeled rover"), {"goal": (2.0, -1.0)})
        self.assertTrue(out.success)

    def test_pick_place_is_robust_and_auto_picks_reach_mode(self):
        # graspable-zone scenes (objects only where the arm can pick them up) + per-arm reach-mode auto-select:
        # a 6-DOF arm uses joint-PD, a redundant arm uses task-space -> both place reliably (was 0.33).
        from virturoid.services.task_runtime import (evaluate_gene_on_task, generate_task_scenes,
                                                     select_task_spec)
        spec = select_task_spec("sort blocks into bins")
        for prompt in ("a tabletop robot arm",
                       "a Franka-style 7-dof arm that sorts red and blue blocks into bins"):
            g = self._arm(prompt)
            scenes = generate_task_scenes(g, spec, count=5)
            r = evaluate_gene_on_task(g, spec, scenes)
            self.assertGreaterEqual(r["success_rate"], 0.7, f"{prompt}: {r['success_rate']}")

    def test_graspable_scenes_stay_inside_the_reliable_workspace(self):
        # scenes must place objects only where a grasp actually succeeds (<= ~0.72 of max reach), not anywhere
        # IK can merely touch — the root cause of the pick_place missed_grasp failures.
        from virturoid.services.gene_build import robot_reachable_points
        g = self._arm()
        reach = sum(s.length_m for s in g.actuated_joints()) + 0.05
        pts = robot_reachable_points(g)
        self.assertTrue(pts)
        self.assertLessEqual(max((x * x + y * y) ** 0.5 for x, y in pts), 0.75 * reach)

    def test_spray_covers_the_whole_panel(self):
        # firm joint-PD reaches every panel cell (soft gains + task-space left ~37% uncoated).
        from virturoid.services.spray_coverage import evaluate_spray_coverage
        self.assertGreaterEqual(evaluate_spray_coverage(self._arm("a spray painting robot"), grid=4)["coverage"], 0.9)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class LearnedLocomotionTests(unittest.TestCase):
    def test_learned_policy_beats_the_scripted_trot_and_is_banked(self):
        # the scripted trot drives the composed quad BACKWARD; a short learned MorphPolicy walks it forward,
        # and locomotion_episode returns whichever travelled further forward (so it never regresses). Banking
        # makes the second build reuse it. Small budget to stay fast.
        import tempfile

        from virturoid.services.learn_locomotion import (banked_policy_for, learn_locomotion,
                                                         locomotion_episode)
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("walk forward", llm=None)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(banked_policy_for(g, models_dir=d))            # nothing banked yet
            learn_locomotion(g, generations=10, pop=10, steps=200, models_dir=d)
            self.assertIsNotNone(banked_policy_for(g, models_dir=d))         # now banked (flywheel reuse)
            r = locomotion_episode(g, horizon=1500, models_dir=d)
            self.assertIn(r["source"], ("learned", "scripted"))
            self.assertIn("forward_m", r)
            # The loop returns whichever travelled FURTHER forward; when it picks the LEARNED policy, that policy
            # must show positive forward displacement. (A regression that collapses the recipe to a stand-still or
            # backward lunge would otherwise pass silently — gait reward is clipped non-negative, so a non-moving
            # policy still scores >1.0; only a forward-displacement check catches it.)
            if r["source"] == "learned":
                self.assertGreater(r["forward_m"], 0.0, f"learned replay must move forward, got {r['forward_m']}")

    def test_trained_recipe_moves_the_quad_forward(self):
        # The core "Learn to move" demo claim, directly asserted: a short on-device train produces POSITIVE
        # forward displacement (deterministic at this budget: measured ~0.33 m, threshold 0.05 leaves ~6x
        # headroom). Without this, a locomotion collapse to ~0 m passes green because the clipped gait reward
        # stays >1.0 even for a body that doesn't move.
        import tempfile

        from virturoid.services.learn_locomotion import learn_locomotion
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("walk forward", llm=None)
        with tempfile.TemporaryDirectory() as d:
            res = learn_locomotion(g, generations=8, pop=10, steps=200, models_dir=d, seeds=2)
            self.assertGreater(res["forward_m"], 0.05,
                               f"trained recipe must walk the quad FORWARD, got {res['forward_m']}")


if __name__ == "__main__":
    unittest.main()
