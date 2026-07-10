"""Design-intent compiler: the LLM picks the body PLAN; deterministic builders synthesize credible geometry.

Locks in the fix for gpt-5.2 raw-geometry unreliability — every body the intent path produces is connected,
role-shaped, and class-correct BY CONSTRUCTION, regardless of how the LLM phrases the plan."""

import importlib.util
import unittest

from virturoid.services.intent_morphology import _clean_appendages, build_from_intent, propose_intent
from virturoid.services.morphology_composer import _morphology_quality_issues, compose_robot

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class _Req:
    reach_m = 0.6
    payload_kg = 0.25
    environment = "floor"


class _IntentLLM:
    """Returns a fixed body-plan intent (what gpt-5.2 emits for INTENT_SCHEMA)."""
    def __init__(self, intent):
        self.intent, self.calls = intent, []

    def complete_json(self, system, user, schema, max_tokens=None, reasoning_effort=None):
        self.calls.append(user)
        return self.intent


class _Plan:
    def __init__(self, rc):
        self.robot_class, self.morphology = rc, ""


class IntentMorphologyTests(unittest.TestCase):
    def test_build_humanoid_intent_is_anthropometric_and_connected(self):
        g = build_from_intent({"robot_class": "humanoid", "scale_m": 1.7, "archetype": "none"},
                              "a humanoid robot", _Req())
        self.assertEqual(g.validate(), [])
        self.assertEqual(g.design_source, "intent")
        self.assertTrue(g.species.startswith("humanoid"))
        self.assertEqual(len(g.actuated_joints()), 8)          # the credible 8-DOF humanoid, never a noodle

    def test_build_legged_honors_llm_leg_count(self):
        # A crab/octopod must get the LLM's 8 legs (not over-fit to a 4-leg dog).
        g = build_from_intent({"robot_class": "legged", "scale_m": 0.4, "n_legs": 8, "archetype": "none"},
                              "a crab robot", _Req())
        self.assertEqual(g.validate(), [])
        legs = [s for s in g.segments if "leg" in s.name]
        self.assertEqual(len({s.name.split("_")[0] for s in legs}), 8)   # 8 DISTINCT legs (leg0..leg7), not >=0
        self.assertEqual(len(g.actuated_joints()), 24)         # 8 legs x 3 actuated joints

    def test_build_manipulator_intent(self):
        g = build_from_intent({"robot_class": "manipulator", "scale_m": 0.6, "arm_dof": 6}, "a robot arm", _Req())
        self.assertEqual(g.validate(), [])
        self.assertEqual(g.robot_class, "manipulator")

    def test_spider_archetype_dispatch(self):
        g = build_from_intent({"robot_class": "legged", "scale_m": 0.3, "archetype": "spider"},
                              "a spider robot", _Req())
        self.assertEqual(g.validate(), [])
        self.assertEqual(g.design_source, "intent")
        self.assertIn("arachnid", g.species.lower())            # the dedicated spider archetype builder

    def test_propose_intent_normalizes_sparse_llm_output(self):
        # Even a minimal LLM answer becomes a complete, valid intent (defaults fill the rest).
        llm = _IntentLLM({"robot_class": "quadruped"})
        intent = propose_intent("a dog robot", _Plan("quadruped"), _Req(), llm)
        self.assertEqual(intent["robot_class"], "quadruped")
        self.assertEqual(intent["source"], "llm")
        self.assertEqual(len(llm.calls), 1)

    def test_compose_robot_uses_intent_path_with_llm(self):
        # End-to-end: compose_robot with an LLM goes through the intent compiler, NOT raw geometry.
        llm = _IntentLLM({"robot_class": "humanoid", "scale_m": 1.7, "archetype": "none", "has_head": True})
        g = compose_robot("a humanoid robot", llm=llm)
        self.assertEqual(g.design_source, "intent")
        self.assertEqual(g.validate(), [])
        self.assertEqual(len(g.actuated_joints()), 8)

    def test_compose_robot_offline_unchanged(self):
        # No LLM -> deterministic keyword builder (the anthropometric humanoid), behavior preserved.
        g = compose_robot("a humanoid robot", llm=None)
        self.assertEqual(g.species, "humanoid.anthropometric")
        self.assertEqual(g.validate(), [])

    def test_uncertain_prompt_provenance_survives_composition(self):
        g = compose_robot("build a blorptron", llm=None)
        plan = g.metadata["build_plan"]
        self.assertFalse(plan["buildable"])
        self.assertEqual(plan["routing_confidence"], "uncertain")
        self.assertTrue(any("clarification" in note.lower() for note in g.composition_notes))

    def test_appendages_attach_flush_and_connected(self):
        # A scorpion = 8 legs + 2 front claws + a 5-segment tail. Every extra part attaches FLUSH (no floats),
        # the body validates, and the appendage DOF is added on top of the legs (8*3 + 2 claws + 5 tail = 31).
        scorp = build_from_intent({"robot_class": "legged", "scale_m": 0.5, "archetype": "none", "n_legs": 8,
                                   "appendages": [{"kind": "claw", "count": 2, "mount": "front", "scale": 1.0},
                                                  {"kind": "tail", "count": 5, "mount": "rear", "scale": 1.0}]},
                                  "a scorpion robot", _Req())
        self.assertEqual(scorp.validate(), [])
        self.assertFalse([i for i in _morphology_quality_issues(scorp) if "floats" in i])  # all attached
        self.assertTrue(any("claw" in s.name for s in scorp.segments))
        self.assertTrue(any("tail" in s.name for s in scorp.segments))
        self.assertEqual(len(scorp.actuated_joints()), 8 * 3 + 2 + 5)

    def test_clean_appendages_clamps_and_filters(self):
        # Unknown kinds dropped; counts clamped; groups capped — a hallucinated list can't explode the body.
        out = _clean_appendages([{"kind": "claw", "count": 99}, {"kind": "laser_cannon", "count": 2},
                                 {"kind": "tail", "count": 5, "mount": "rear"}])
        kinds = [a["kind"] for a in out]
        self.assertIn("claw", kinds)
        self.assertIn("tail", kinds)
        self.assertNotIn("laser_cannon", kinds)                # not a known appendage -> dropped
        self.assertEqual(next(a for a in out if a["kind"] == "claw")["count"], 4)   # 99 clamped to 4

    def test_intent_path_repairs_once_on_validation_flag(self):
        # When the built body trips a fatal/high validation flag, compose_robot re-prompts the plan ONCE
        # (feeding the issues back) and takes the sound body. Forced flag-then-clean to test the mechanism.
        from unittest import mock

        from virturoid.services import morphology_composer as mc
        # Use a MANIPULATOR (not a creature) so the anatomy-graph path is skipped and this isolates the
        # intent-compiler repair loop (creatures try the anatomy designer first, which is a separate path).
        llm = _IntentLLM({"robot_class": "manipulator", "scale_m": 0.6, "archetype": "none", "arm_dof": 4})
        seq = {"n": 0}

        def fake_issues(gene, req):
            seq["n"] += 1
            return ["forced: statically unstable rest pose"] if seq["n"] == 1 else []

        with mock.patch.object(mc, "_intent_validation_issues", side_effect=fake_issues):
            g = mc.compose_robot("a tabletop robot arm to pick and place", llm=llm)
        self.assertEqual(g.design_source, "intent")
        # LLM-driven plan classification + initial intent + exactly one repair re-prompt = 3 calls.
        self.assertEqual(len(llm.calls), 3)
        self.assertTrue(any("FAILED validation" in c for c in llm.calls[1:]))  # issues fed back


if __name__ == "__main__":
    unittest.main()
