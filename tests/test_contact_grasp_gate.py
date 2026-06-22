"""Honest contact-grasp certification + the fail-CLOSED readiness gate.

Two coupled invariants:
  1) A REAL (non-pinned) contact grasp+lift, certified by grasp_eval.evaluate_grasp_lift, lets a gripper
     manipulator read physics_evaluated=ATTAINED — a genuine physics capability, not the idealized pin.
  2) The gate fails CLOSED: an idealized pin, a MISSING grasp_model disclosure, or any unknown value reads
     BELOW_GATE (an allowlist of known-real grasp models, not a blocklist of one bad string — the audit's
     fail-OPEN hole where a dropped disclosure silently certified).
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.readiness_ledger import ATTAINED, BELOW_GATE
from virturoid.services.readiness_ledger import _probe_physics

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _probe(reports: dict):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td); (p / "reports").mkdir()
        for name, data in reports.items():
            (p / "reports" / name).write_text(json.dumps(data), encoding="utf-8")
        return _probe_physics(p, None)


class GraspHonestyGateTests(unittest.TestCase):
    def test_real_contact_grasp_attains(self):
        rec = _probe({"grasp_evaluation_report.json":
                      {"task_type": "grasp_lift", "grasp_model": "contact", "success_rate": 1.0, "mean_lift_m": 0.11}})
        self.assertEqual(rec.status, ATTAINED)
        self.assertIn("contact", rec.detail.lower())

    def test_idealized_pin_is_below_gate(self):
        rec = _probe({"gene_evaluation_report.json":
                      {"task_type": "pick_place_sort", "grasp_model": "idealized_pin", "success_rate": 1.0}})
        self.assertEqual(rec.status, BELOW_GATE)
        self.assertIn("idealized", rec.detail.lower())

    def test_missing_grasp_model_on_a_manipulation_report_fails_closed(self):
        # THE HOLE: a manipulation report that DROPS its grasp_model disclosure must NOT silently certify.
        rec = _probe({"gene_evaluation_report.json": {"task_type": "pick_place_sort", "success_rate": 1.0}})
        self.assertEqual(rec.status, BELOW_GATE, "a manipulation report with no grasp_model must fail closed")

    def test_unknown_grasp_model_fails_closed(self):
        rec = _probe({"gene_evaluation_report.json":
                      {"task_type": "pick_place", "grasp_model": "magic", "success_rate": 1.0}})
        self.assertEqual(rec.status, BELOW_GATE)

    def test_contact_grasp_below_threshold_is_below_gate(self):
        rec = _probe({"grasp_evaluation_report.json":
                      {"task_type": "grasp_lift", "grasp_model": "contact", "success_rate": 0.2}})
        self.assertEqual(rec.status, BELOW_GATE)

    def test_locomotion_without_grasp_model_still_attains(self):
        # the allowlist must NOT regress non-manipulation reports that legitimately omit grasp_model
        rec = _probe({"gene_evaluation_report.json":
                      {"task_type": "locomotion", "forward_m": 0.5, "upright": True, "status": "walked"}})
        self.assertEqual(rec.status, ATTAINED)

    def test_locomotion_below_walk_threshold_fails_closed(self):
        # a 0.10-0.15 m drift (below the shared _WALK_MIN_FORWARD_M) or an upright SLIDE must NOT certify — the gate
        # was lenient at 0.1 m and is now unified with the build's honest 'walked' label (status-gated).
        drift = _probe({"gene_evaluation_report.json":
                        {"task_type": "locomotion", "forward_m": 0.12, "upright": True, "status": "upright"}})
        self.assertEqual(drift.status, BELOW_GATE, "a 0.12 m drift must not read as a walk")
        slide = _probe({"gene_evaluation_report.json":   # honest status overrides a high distance with no real gait
                        {"task_type": "locomotion", "forward_m": 0.6, "upright": True, "status": "upright"}})
        self.assertEqual(slide.status, BELOW_GATE, "status='upright' (a slide) must not certify even at 0.6 m")


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ContactGraspEndToEndTests(unittest.TestCase):
    def test_composed_sorting_arm_does_a_real_grasp(self):
        from virturoid.services.grasp_eval import evaluate_grasp_lift
        from virturoid.services.robot_factory import build_robot
        gene = build_robot("a tabletop arm that sorts red and blue blocks into matching bins")["gene"]
        r = evaluate_grasp_lift(gene)
        self.assertGreaterEqual(r["success_rate"], 0.66, f"real contact grasp must succeed, got {r['success_rate']}")
        self.assertGreater(r["mean_lift"], 0.05, "the grasp must actually LIFT the block by friction")

    def test_real_contact_grasp_transport_places_a_block(self):
        # The real (no-pin) grasp -> lift -> transport -> place MECHANISM: at a frontally-reachable position the
        # composed gripper grasps the block by CONTACT FRICTION, carries it to the bin, and releases it (placed).
        # Guards run_contact_pick_place_episode (build/_grasp_transport_derisk.py de-risked 4/4). NOTE: this is the
        # frontal/dexterous workspace — the full 4-object spread-out SORT exceeds the arm's top-down reach (see the
        # honest gap), so this mechanism is validated but not yet the default sort path.
        from virturoid.schemas.scenes import SceneObject
        from virturoid.services.pick_place_controller import run_contact_pick_place_episode
        from virturoid.services.robot_factory import build_robot
        gene = build_robot("a tabletop arm that sorts red and blue blocks into matching bins")["gene"]
        objs = [SceneObject("red_block", "cube", (0.40, 0.0, 0.02, 0, 0, 0), mass_kg=0.05,
                            material="red_block", friction=1.0, scale=1.0),
                SceneObject("red_bin", "container", (0.40, 0.18, 0.0, 0, 0, 0), material="red_bin")]
        r = run_contact_pick_place_episode(gene, objs, {"red_block": "red_bin"})
        self.assertEqual(r["grasp_model"], "contact")
        self.assertEqual(r["placed_count"], 1, f"real contact grasp+transport must place the block, got {r}")

    def test_contact_grasp_is_robust_to_randomized_object_mass_and_friction(self):
        # SIM2REAL (manipulation): the friction grasp must hold across VARIED object dynamics, not one tuned object.
        # Light/low-friction .. heavy/high-friction objects all get grasped + placed (the manipulation analog of the
        # locomotion domain-randomization robustness eval).
        from virturoid.schemas.scenes import SceneObject
        from virturoid.services.pick_place_controller import run_contact_pick_place_episode
        from virturoid.services.robot_factory import build_robot
        gene = build_robot("a tabletop arm that picks up a box and places it on a target")["gene"]
        placed = 0
        for m, fr in [(0.03, 0.6), (0.06, 1.0), (0.09, 1.4)]:
            objs = [SceneObject("box", "cube", (0.40, 0.0, 0.02, 0, 0, 0), mass_kg=m, material="gray_block",
                                friction=fr, scale=1.0),
                    SceneObject("target_zone", "container", (0.40, 0.18, 0.0, 0, 0, 0), material="gray_bin")]
            placed += int(run_contact_pick_place_episode(gene, objs, {"box": "target_zone"})["placed_count"] > 0)
        self.assertGreaterEqual(placed, 2, f"contact grasp must be robust across object mass/friction, placed {placed}/3")

    def test_visible_arm_tasks_run_the_real_contact_grasp(self):
        # The visible arm task — single-object pick-place AND the multi-object sort — must run the REAL contact
        # grasp (no pin) in its dexterous workspace, so the watched artifact == the certified one. The per-block
        # approach + dexterous-zone scene posing make the sort place every block by friction.
        from virturoid.services.robot_factory import build_robot
        from virturoid.services.task_executor import evaluate_task
        for prompt in ("a tabletop arm that picks up a box and places it on a target",
                       "a tabletop arm that sorts red and blue blocks into matching bins"):
            gene = build_robot(prompt)["gene"]
            r = evaluate_task(prompt, gene, record=True)
            self.assertEqual(r.get("grasp_model"), "contact", f"{prompt!r} must run the real contact grasp")
            self.assertTrue(r.get("success"), f"{prompt!r} should place its blocks via the real grasp")
            self.assertIsNotNone(r.get("view"), "the real grasp must be recorded for the viewport")

    def test_build_certifies_a_real_grasp_through_the_ledger(self):
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.services.gene_build import build_gene_package
        from virturoid.services.readiness_ledger import build_product_readiness_ledger
        from virturoid.services.robot_factory import build_robot
        gene = build_robot("a tabletop arm that sorts red and blue blocks into matching bins")["gene"]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pkg"
            build_gene_package(gene, "a tabletop arm that sorts red and blue blocks into matching bins", out, scene_count=2)
            gr = out / "reports" / "grasp_evaluation_report.json"
            self.assertTrue(gr.exists(), "a gripper build must write a contact grasp_evaluation_report")
            self.assertEqual(json.loads(gr.read_text())["grasp_model"], "contact")
            led = build_product_readiness_ledger(out, robot_class="manipulator", gene=gene)
            self.assertEqual(led.by_stage["physics_evaluated"].status, ATTAINED,
                             "a real certified contact grasp must read physics_evaluated=ATTAINED")


if __name__ == "__main__":
    unittest.main()
