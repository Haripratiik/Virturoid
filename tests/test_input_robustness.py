"""Tester-mode robustness: garbage / empty / non-numeric inputs must yield a clear TEACHING error, never a raw
Python exception string and never a FALSE SUCCESS. Found by adversarially driving the real tool surface
(call_tool) with empty goals, non-numeric edit factors, absurd/injection prompts (2026-07-09).

These reject paths fail fast (before any MuJoCo), so the suite runs without the physics engine.
"""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


def _min_gene():
    from virturoid.schemas.gene import GeneSegment, RobotGene
    return RobotGene(id="t", species="test", robot_class="quadruped", base_mount="free", segments=[
        GeneSegment(name="torso", parent=None, shape="box", length_m=0.3, radius_m=0.1, mass_kg=2.0),
        GeneSegment(name="leg1_l_0", parent="torso", shape="capsule", length_m=0.2, radius_m=0.03, mass_kg=0.3,
                    joint_type="revolute", joint_axis=(1, 0, 0), joint_lower=-1.0, joint_upper=1.0,
                    actuator_torque_nm=8.0)])


class RunTaskGoalTests(unittest.TestCase):
    def _run(self, goal):
        from virturoid.services import session_state as S
        from virturoid.services.agent_design_tools import run_task
        rid = S.put_robot(_min_gene(), prompt="a quadruped robot dog", label="rob")
        return run_task({"robot_id": rid, "goal": goal})

    def test_empty_goal_is_rejected_not_falsely_succeeded(self):
        for goal in ("", "   ", "\t\n"):
            r = self._run(goal)
            self.assertFalse(r.get("ok"), f"an explicit empty goal {goal!r} must be rejected, not a false success")
            self.assertIn("empty", r.get("error", "").lower())


class EditNumericCoercionTests(unittest.TestCase):
    """A non-numeric edit arg must raise a TEACHING EditError, not a raw ValueError."""

    def test_non_numeric_args_teach(self):
        from virturoid.services import edit_operators as EO
        cases = [
            (EO.scale_group, {"factor": "big"}, "factor"),
            (EO.scale_robot, {"factor": "huge"}, "factor"),
            (EO.set_height, {"target_m": "tall"}, "target_m"),
            (EO.set_leg_count, {"n_pairs": "lots"}, "n_pairs"),
            (EO.set_payload, {"payload_kg": "heavy"}, "payload_kg"),
        ]
        for fn, kw, name in cases:
            with self.assertRaises(EO.EditError) as cm:
                fn(_min_gene(), **kw)
            self.assertIn(f"{name} must be a number", str(cm.exception))

    def test_valid_numbers_still_pass_the_coercion(self):
        # a numeric string or float is accepted (coercion is lenient, only garbage is rejected)
        from virturoid.services import edit_operators as EO
        with self.assertRaises(EO.EditError) as cm:
            EO.scale_group(_min_gene(), factor=99.0)                # numeric -> passes coercion, fails RANGE (0.2-5.0)
        self.assertIn("out of the safe range", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
