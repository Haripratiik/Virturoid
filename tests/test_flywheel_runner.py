"""Flywheel runner (Theme 2): many build cycles against ONE shared memory must demonstrably compound.

Uses an injected fake ``build_fn`` that banks a design + a skill into the shared memory each cycle, so
the runner's per-cycle snapshots reflect real DB growth without the slow physics/training loop. The point
under test is the moat claim: banked assets grow and later cycles see prior warm-start assets."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from virturoid.services.flywheel_runner import run_flywheel
from virturoid.services.memory_db import MemoryDB


def _fake_build_fn(prompt, out_dir, *, target_success_rate, train, memory_dir):
    """Stand-in for autonomous_build: banks one design + one skill into the shared memory, like a real
    successful build would, and returns a minimal report with the attributes the runner reads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        n = int(db.stats().get("runs", 0))   # monotonic cycle index from prior runs in the shared DB
        success = 0.8 + 0.02 * n
        db.record_run(prompt, "manipulator", "pick_place_sort",
                      {"forearm_mm": 330.0 + n}, success, species=f"arm.v{n}",
                      succeeded=True, design_source="as_built")
        db.record_skill(f"pick_place_sort__arm.v{n}", "manipulator", "pick_place_sort",
                        success_rate=success, species=f"arm.v{n}")
    return SimpleNamespace(robot_class="manipulator", task_type="pick_place_sort",
                           species=f"arm.v{n}", final_success_rate=success,
                           succeeded=True, memory_reused=n > 0)


class FlywheelRunnerTests(unittest.TestCase):
    def test_memory_compounds_across_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prompts = ["sort red blocks", "sort blue blocks", "sort green blocks"]
            result = run_flywheel(prompts, memory_dir=tmp / "memory", out_root=tmp / "runs",
                                  build_fn=_fake_build_fn)

            self.assertEqual(len(result["cycles"]), 3)
            # First cycle had nothing banked; later cycles see prior assets to warm-start from.
            self.assertFalse(result["cycles"][0]["warm_start_available"])
            self.assertTrue(result["cycles"][1]["warm_start_available"])
            self.assertTrue(result["cycles"][2]["warm_start_available"])
            # Banked memory strictly grew (the moat is real, not a no-op).
            self.assertGreater(result["cycles"][2]["banked_after"]["designs"],
                               result["cycles"][0]["banked_after"]["designs"])
            self.assertEqual(result["totals"]["designs_banked"], 3)
            self.assertEqual(result["totals"]["skills_banked"], 3)
            self.assertEqual(result["totals"]["warm_start_cycles"], 2)
            self.assertTrue(result["compounding"])

    def test_single_cycle_is_not_claimed_to_compound(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = run_flywheel(["one build"], memory_dir=tmp / "memory",
                                  out_root=tmp / "runs", build_fn=_fake_build_fn)
            self.assertEqual(len(result["cycles"]), 1)
            self.assertFalse(result["compounding"])  # needs >=2 cycles to claim compounding

    def test_each_cycle_builds_into_its_own_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run_flywheel(["a", "b"], memory_dir=tmp / "memory", out_root=tmp / "runs",
                         build_fn=_fake_build_fn)
            self.assertTrue((tmp / "runs" / "cycle_0").exists())
            self.assertTrue((tmp / "runs" / "cycle_1").exists())


if __name__ == "__main__":
    unittest.main()
