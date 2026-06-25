import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class AutonomousBuildTests(unittest.TestCase):
    def test_autonomous_build_produces_a_working_validated_robot(self):
        from virturoid.services.autonomous_build import autonomous_build
        from virturoid.services.memory_store import save_build_record
        from virturoid.services.package_validator import validate_mvp_export_package

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "robot"
            memory_dir = Path(tmpdir) / "memory"
            # Seed prior memory so the "memory is not None" reuse branch executes during
            # this single build (regression: that branch once crashed on build.robot_class).
            save_build_record("manipulator", "pick_place_sort",
                              {"forearm_mm": 330.0, "wrist_mm": 150.0, "actuator_torque_nm": 12.0},
                              0.5, "seed prior build", memory_dir)
            report = autonomous_build(
                "Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
                out,
                target_success_rate=0.8,
                memory_dir=memory_dir,
            )
            self.assertTrue(report.memory_reused)  # the reuse branch ran without crashing

            self.assertTrue(report.validate().ok, report.validate().issues)
            self.assertEqual("manipulator", report.robot_class)
            self.assertTrue(report.feasible)
            # The software autonomously reached a working robot for the task.
            self.assertTrue(report.succeeded)
            self.assertGreaterEqual(report.final_success_rate, report.target_success_rate)
            # It improved over the as-built robot via autonomous co-design.
            self.assertGreaterEqual(report.final_success_rate, report.initial_success_rate)

            # The converged, validated robot is exported as a buildable package.
            self.assertTrue(validate_mvp_export_package(out).ok)
            self.assertTrue((out / "robot" / "robot_genome.json").exists())
            self.assertTrue((out / "robot" / "robot.urdf").exists())
            self.assertTrue((out / "simulation" / "mujoco" / "compiled_scene_index.json").exists())
            self.assertTrue((out / "reports" / "autonomy_report.json").exists())

            # Decisions trace the autonomous loop (build -> evaluate -> improve).
            stages = {d.stage for d in report.decisions}
            self.assertIn("build_and_validate", stages)
            self.assertIn("evaluate", stages)
            self.assertIn("memory_warm_start", stages)  # the flywheel actually warm-starts the body

            # The build is remembered for reuse.
            mem = list((Path(tmpdir) / "memory").glob("*.json"))
            self.assertTrue(mem)
            record = json.loads(mem[0].read_text(encoding="utf-8"))
            self.assertIn("converged_design", record)

            # Compute provenance proves real physics actually ran (not a fast stub).
            self.assertTrue(report.compute.get("physics_executed"))
            self.assertGreater(report.compute.get("physics_steps", 0), 1000)
            self.assertGreater(report.compute.get("wall_time_seconds", 0), 0)
            self.assertTrue((out / "reports" / "compute_provenance.json").exists())

            # The robot's intelligent control program is exported.
            control = out / "software" / "control_program.json"
            self.assertTrue(control.exists())
            program = json.loads(control.read_text(encoding="utf-8"))
            self.assertEqual("pick_place_sort", program["task"])
            self.assertIn("phases", program)
            self.assertEqual(report.converged_design, program["converged_design"])
            self.assertTrue((out / "reports" / "memory_effectiveness_report.json").exists())
            self.assertIn("memory_effectiveness_report", report.exported_artifacts)

    def test_memory_warm_start_skips_codesign_when_prior_design_already_solves_it(self):
        """The moat (§13/§26): a second build of a known task reuses the prior converged body and
        SKIPS the expensive co-design search — measurably cheaper, not just 'memory_reused=True'."""
        from virturoid.services.autonomous_build import autonomous_build
        from virturoid.services.memory_store import save_build_record

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "robot"
            memory_dir = Path(tmpdir) / "memory"
            # Seed a strong prior converged body (from an earlier real run) that already meets target.
            save_build_record("manipulator", "pick_place_sort",
                              {"forearm_mm": 362.3, "wrist_mm": 120.0, "actuator_torque_nm": 13.7},
                              0.9, "prior converged build", memory_dir)
            report = autonomous_build(
                "Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
                out, target_success_rate=0.8, memory_dir=memory_dir)
            stages = [d.stage for d in report.decisions]
            self.assertIn("memory_warm_start", stages)
            warm = next(d for d in report.decisions if d.stage == "memory_warm_start")
            self.assertGreaterEqual(warm.success_after, 0.8)        # the reused body already solves it
            self.assertNotIn("co_design_hardware", stages)          # so the expensive search is skipped
            self.assertTrue(report.succeeded)
            self.assertGreaterEqual(report.final_success_rate, 0.8)
            mem_report = json.loads((out / "reports" / "memory_effectiveness_report.json").read_text(encoding="utf-8"))
            self.assertTrue(mem_report["memory_reused"])
            self.assertTrue(mem_report["warm_start_attempted"])
            self.assertTrue(mem_report["co_design_skipped_after_memory"])
            self.assertFalse(mem_report["co_design_search_ran"])
            self.assertGreaterEqual(mem_report["warm_start_success_rate"], 0.8)
            self.assertGreaterEqual(mem_report["evidence"]["training_rows_at_target"], 1)


if __name__ == "__main__":
    unittest.main()
