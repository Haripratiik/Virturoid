"""I1: enterprise robot import report (Input Ingestion plan, Phase 1).

Drops a real URDF file and checks the durable report runs BOTH lanes (faithful native + inferred RobotGene),
classifies warnings into fixable typed records, scores readiness, and writes input/import_report.json.
Offline; needs MuJoCo (both lanes compile the model).
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

# A base + two links; the elbow is CONTINUOUS (no limits) -> a fixable "missing_joint_limits" warning.
_URDF = """<?xml version="1.0"?>
<robot name="arm2">
  <link name="base_link"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="upper"><inertial><mass value="0.6"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.04 0.04 0.3"/></geometry></collision></link>
  <link name="forearm"><inertial><mass value="0.4"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.03 0.03 0.25"/></geometry></collision></link>
  <joint name="pan" type="revolute"><parent link="base_link"/><child link="upper"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/></joint>
  <joint name="elbow" type="continuous"><parent link="upper"/><child link="forearm"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/></joint>
</robot>"""


class WarningClassificationTests(unittest.TestCase):
    def test_every_known_warning_maps_to_a_fix(self):
        from virturoid.services.robot_import_report import classify_warning
        cases = {
            "joint on 'x' has no limits (continuous); set joint_lower/upper": ("missing_joint_limits", "warning"),
            "body 'x' has zero/negative mass — inertia missing": ("missing_mass", "warning"),
            "2 root bodies attach to the world (a, b)": ("multiple_roots", "error"),
            "body 'x' has 3 joints; a gene segment models one": ("multi_joint_body_lossy", "warning"),
        }
        for msg, (code, sev) in cases.items():
            w = classify_warning(msg)
            self.assertEqual(w.code, code)
            self.assertEqual(w.severity, sev)
            self.assertTrue(w.fix_action, "every warning must carry a concrete fix action")

    def test_unknown_warning_still_gets_a_fix(self):
        from virturoid.services.robot_import_report import classify_warning
        w = classify_warning("something unusual happened")
        self.assertEqual(w.code, "import_warning")
        self.assertTrue(w.fix_action)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo to compile both import lanes")
class ImportReportTests(unittest.TestCase):
    def _write_urdf(self) -> str:
        d = tempfile.mkdtemp(prefix="import_")
        p = os.path.join(d, "arm2.urdf")
        Path(p).write_text(_URDF, encoding="utf-8")
        return p

    def test_both_lanes_run_and_report_is_durable(self):
        from virturoid.services.robot_import_report import build_import_report, write_import_report

        path = self._write_urdf()
        report = build_import_report(path, robot_id="arm2")
        # faithful native lane compiles as-is; inferred RobotGene lane also produced.
        self.assertIsNotNone(report.faithful_lane)
        self.assertTrue(report.faithful_lane.ok, report.faithful_lane.summary)
        self.assertEqual(report.source_format, "urdf")
        self.assertGreaterEqual(report.actuated_joint_count, 1)
        self.assertEqual(report.first_runnable_sim, "faithful_native_mjcf")

        # the continuous elbow surfaces as a fixable missing-limits warning.
        codes = {w.code for w in report.warnings}
        self.assertIn("missing_joint_limits", codes)
        for w in report.warnings:
            self.assertTrue(w.fix_action)

        # readiness scored on independent axes.
        for axis in ("model_parse_score", "kinematic_confidence", "training_readiness_score"):
            self.assertIn(axis, report.confidence)
        self.assertEqual(report.confidence["model_parse_score"], 1.0)

        # durable: written next to the package as input/import_report.json.
        pkg = tempfile.mkdtemp(prefix="pkg_")
        out = write_import_report(report, pkg)
        self.assertTrue(os.path.exists(out))
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual(data["source_format"], "urdf")
        self.assertIn("faithful_lane", data)
        self.assertIn("gene_lane", data)
        self.assertTrue(all("fix_action" in w for w in data["warnings"]))

    def test_report_validates(self):
        from virturoid.services.robot_import_report import build_import_report
        report = build_import_report(self._write_urdf(), robot_id="arm2")
        self.assertTrue(report.validate().ok, [i.message for i in report.validate().issues])


if __name__ == "__main__":
    unittest.main()
