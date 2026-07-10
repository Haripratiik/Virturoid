"""Thread C — semantic MOMENT search: the sim EMITS events, so 'when did the gripper fail' is a predicate query
over a ~10^2-event log, not an LLM-per-frame scan. Ground truth is un-gameable (planted transitions we know the
answer to). MuJoCo-gated; offline; zero LLM calls.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "moment extraction replays a compiled MuJoCo episode")
class EpisodeMomentsTests(unittest.TestCase):
    def _quad(self):
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped robot dog")
        m = mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(g, include_floor=True, spawn_z=standing_spawn_z(g)))
        return g, m

    def test_real_walk_emits_contact_events_and_out_of_vocab_escalates(self):
        from virturoid.services.episode_moments import (CONTACT_BEGIN, FOOT_DOWN, ask_episode,
                                                        extract_events, summarize_episode)
        from virturoid.services.morph_policy import crawl_gait_rollout
        g, m = self._quad()
        r = crawl_gait_rollout(g, steps=1200, record_qpos=True, frame_every=5)
        frames = r.get("qpos_frames") or []
        self.assertTrue(frames, "the rollout must record qpos frames")
        ev = extract_events(m, frames, dt=float(m.opt.timestep), frame_every=5)
        self.assertGreater(len(ev), 5, "a real walk emits many transition events (a temporal sentence)")
        self.assertTrue([e for e in ev if e.type in (FOOT_DOWN, CONTACT_BEGIN)], "a walking quad hits the ground")
        summ = summarize_episode(ev)
        self.assertEqual(summ["n_events"], len(ev))
        # a question OUTSIDE the closed vocabulary escalates (matched=False), never crashes / never scans frames
        self.assertFalse(ask_episode(ev, "what colour is the robot?")["matched"])

    def test_when_did_it_fall_is_exact(self):
        # UN-GAMEABLE: synthesize a trace whose base z collapses at a KNOWN frame; the answer must be that frame.
        import mujoco

        from virturoid.services.episode_moments import FELL, ask_episode, extract_events
        _, m = self._quad()
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        q = d.qpos.copy()
        frames = []
        for i in range(40):
            qi = q.copy()
            qi[2] = float(q[2]) * (1.0 if i < 20 else 0.2)    # base height collapses to 20% at frame 20
            frames.append(qi)
        ev = extract_events(m, frames, dt=0.01, frame_every=1)
        falls = [e for e in ev if e.type == FELL]
        self.assertEqual(len(falls), 1, "exactly one fall TRANSITION (edge, not per-frame)")
        self.assertEqual(falls[0].step, 20, "fall detected at the collapse frame")
        ans = ask_episode(ev, "when did the robot fall over?")
        self.assertTrue(ans["matched"] and ans["found"])
        self.assertEqual(ans["when_step"], 20)
        self.assertEqual(ans["event"]["type"], FELL)

    def test_when_did_the_gripper_fail(self):
        # the headline query. A minimal gripper+payload scene; the payload leaves the gripper at a KNOWN frame.
        import mujoco

        from virturoid.services.episode_moments import GRASP_ATTACH, GRASP_DETACH, ask_episode, extract_events
        xml = """<mujoco><worldbody>
          <geom name='floor' type='plane' size='3 3 0.1'/>
          <body name='gripper_hand' pos='0 0 0.3'><geom name='grip' type='box' size='0.05 0.05 0.05'/></body>
          <body name='box_payload' pos='0 0 0.3'><freejoint/><geom name='cube' type='box' size='0.03 0.03 0.03'/></body>
        </worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        frames = []
        for i in range(30):
            # box overlaps the gripper (contact) until frame 18, then is carried away (no contact) = a DROP
            x = 0.0 if i < 18 else 0.6
            frames.append([x, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
        ev = extract_events(m, frames, dt=0.02, frame_every=1)
        self.assertTrue([e for e in ev if e.type == GRASP_ATTACH], "the gripper first grasps the payload")
        detach = [e for e in ev if e.type == GRASP_DETACH]
        self.assertEqual(len(detach), 1, "one grasp failure (detach) transition")
        self.assertEqual(detach[0].step, 18, "the failure is at the frame the payload left the gripper")
        ans = ask_episode(ev, "when did the gripper fail?")
        self.assertTrue(ans["matched"] and ans["found"])
        self.assertEqual(ans["when_step"], 18)
        self.assertEqual(ans["event"]["type"], GRASP_DETACH)
        # zero LLM calls were made — this is pure predicate matching over the event log


    def test_ask_episode_tool_end_to_end(self):
        # the product surface: create a robot, ask temporal questions through the MCP tool registry
        from virturoid.services.agent_tools import call_tool, tool_specs
        self.assertIn("ask_episode", [t["name"] for t in tool_specs()], "ask_episode must be a registered tool")
        rid = call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]["robot_id"]
        step = call_tool("ask_episode", {"robot_id": rid, "question": "when did it first step?"})["result"]
        self.assertTrue(step["matched"] and step["found"], "a walking quad has a first-step moment")
        self.assertGreaterEqual(step["episode"]["n_events"], 5)
        # in-vocab but absent (a quad has no gripper) -> matched, honestly not found; out-of-vocab -> escalate
        self.assertFalse(call_tool("ask_episode",
                                   {"robot_id": rid, "question": "what colour is it?"})["result"]["matched"])


if __name__ == "__main__":
    unittest.main()
