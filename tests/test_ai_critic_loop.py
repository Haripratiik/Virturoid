"""The AI-assisted REWARD loop is WIRED end-to-end (gait_critic was orphaned before): train -> diagnose ->
LLM critic redesigns the reward weights -> retrain with them. These tests mock the GPU train+diagnose and the
LLM (no GPU/network needed) and assert the orchestration: the critic's redesigned weights actually reach the
NEXT training round, and offline (no LLM) it degrades gracefully to a single default-weight run."""

import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class _Gene:
    robot_class = "quadruped"
    id = "test"


class _FakeLLM:
    """Stands in for the reward-engineer LLM: always proposes a slide-fix (raise slip_w + alt_w + swing_w)."""
    def complete_json(self, system, user, schema, max_tokens=700):
        return {"failure_diagnosis": "upright slide, cadence~0", "key_change": "raise slip_w + alt_w",
                "rationale": "penalize planted-foot drift, force alternation",
                "weights": {"slip_w": 5.0, "alt_w": 4.0, "swing_w": 3.0}}


class AiCriticLoopTests(unittest.TestCase):
    def test_critic_redesigned_weights_reach_the_next_training_round(self):
        from virturoid.services.assisted_trainer import ai_critic_gait_loop
        calls = []

        def fake_train_diagnose(gene, weights, **kw):
            calls.append(dict(weights))
            walking = len(calls) >= 2                       # round 1 slides; round 2 (after the critic) walks
            cad = 6.0 if walking else 0.1
            return f"npz{len(calls)}", {"cadence": cad, "walking": walking, "gait_quality": 0.5 if walking else 0.05,
                                        "summary": "walk" if walking else "slide"}

        res = ai_critic_gait_loop(_Gene(), llm=_FakeLLM(), rounds=3, _train_diagnose=fake_train_diagnose)
        self.assertTrue(res["walking"], "the loop should converge once the critic fixes the reward")
        self.assertEqual(len(calls), 2, "should stop as soon as it walks")
        # round 1 used trainer defaults; round 2 used the critic's REDESIGNED weights (the wiring under test)
        self.assertGreater(calls[1]["slip_w"], calls[0]["slip_w"], "critic's higher slip_w must reach round 2")
        self.assertEqual(calls[1]["alt_w"], 4.0, "critic's alt_w must reach round 2")

    def test_offline_no_llm_runs_once_with_default_weights(self):
        from virturoid.services.assisted_trainer import ai_critic_gait_loop
        calls = []

        def fake_train_diagnose(gene, weights, **kw):
            calls.append(dict(weights))
            return "npz", {"cadence": 0.1, "walking": False, "gait_quality": 0.05, "summary": "slide"}

        res = ai_critic_gait_loop(_Gene(), llm=None, rounds=3, _train_diagnose=fake_train_diagnose)
        self.assertEqual(len(calls), 1, "offline (no LLM) must not loop — one default-weight run, graceful")
        self.assertFalse(res["walking"])
        self.assertEqual(res["rounds"], 1)


if __name__ == "__main__":
    unittest.main()
