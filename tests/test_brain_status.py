"""P5 — make the moat VISIBLE: moat_status now carries a `brain` section that reports the P1-P3 robotics-AI
layers (transfer ledger, episodes, gated-metric state, per-kind provenance deltas) read straight from their
tables, so every number a user sees traces to a ledger row rather than a placeholder."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class BrainStatusTests(unittest.TestCase):
    def test_brain_section_reflects_the_ledgers(self):
        from virturoid.services.flywheel_status import moat_status
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        from virturoid.services.transfer_ledger import record_transfer_trial
        mem = Path(tempfile.mkdtemp(prefix="brain_"))
        a = compose_robot("a small quadruped robot dog")
        b = compose_robot("a six-legged hexapod robot")
        gp = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
        with MemoryDB(mem / "virturoid_memory.db") as db:
            record_transfer_trial(db, src_gene=a, dst_gene=b, gait_params=gp,
                                  result={"survived": True, "credible": True, "forward": 0.8})
            record_transfer_trial(db, src_gene=b, dst_gene=a, gait_params=gp,
                                  result={"survived": True, "credible": False, "forward": 0.05})
            RoboticsVectorMemory(db).index_episode("e1", {"forward_m": 0.8, "cadence": 2.0}, {"status": "walked"})
            vm = RoboticsVectorMemory(db)
            vm.record_provenance("gene", "hint-win", kind="gait_hint_deploy", delta=0.2)
            vm.record_provenance("gene", "hint-loss", kind="gait_hint_deploy", delta=-0.1)
            db.record_run(prompt="p", robot_class="quadruped", task_type="locomotion",
                          converged_design={"x": 1}, success_rate=0.7)

        st = moat_status(mem)
        self.assertIn("brain", st)
        self.assertEqual(st["brain"]["transfer_ledger"]["trials"], 2)
        self.assertEqual(st["brain"]["transfer_ledger"]["credible"], 1)
        self.assertEqual(st["brain"]["episodes"], 1)
        hint = st["brain"]["provenance_by_kind"]["gait_hint_deploy"]
        self.assertEqual((hint["wins"], hint["losses"]), (1, 1))
        self.assertEqual(0.5, hint["hit_rate"])
        self.assertEqual(0.5, st["warm_start"]["hint_reuse"]["hit_rate"])
        self.assertFalse(st["brain"]["embedding"]["metric_proven"])          # nothing proven -> honest baseline
        self.assertEqual(st["brain"]["embedding"]["active"], "baseline_29d")
        self.assertIn("verified transfer trials", st["summary"])             # user-facing headline traces to ledger

    def test_empty_memory_is_honest_not_a_crash(self):
        from virturoid.services.flywheel_status import moat_status
        st = moat_status(Path(tempfile.mkdtemp(prefix="empty_")))
        self.assertEqual(st["brain"]["transfer_ledger"]["trials"], 0)
        self.assertEqual(st["brain"]["embedding"]["active"], "baseline_29d")


if __name__ == "__main__":
    unittest.main()
