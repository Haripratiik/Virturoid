"""Whole-project artifact classifier + Project Graph (Input Ingestion Plan, Phase 2).

Acceptance: dropping a folder with a URDF + meshes + config + ROS + logs + BOM produces a classified
Project Graph with a first runnable sim target and no blockers; junk/VCS files are skipped; an unrecognized
file is surfaced (not silently dropped); a project without any robot description reports a blocker.
"""

import os
import tempfile
import unittest
import zipfile

from virturoid.schemas.input_bundle import ParseStatus
from virturoid.services.input_classifier import (
    classify_name,
    project_graph_summary,
    scan_folder,
    scan_zip,
)


def _write(root: str, rel: str, content: bytes = b"x") -> None:
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)


class ClassifyNameTests(unittest.TestCase):
    def test_extension_and_filename_cues(self):
        self.assertEqual(classify_name("robot.urdf")[:2], ("robot_model", "model"))
        self.assertEqual(classify_name("meshes/base.stl")[:2], ("mesh", "mesh"))
        self.assertEqual(classify_name("cad/arm.step")[:2], ("cad", "cad"))
        self.assertEqual(classify_name("ros/package.xml")[:2], ("ros_package", "ros"))
        self.assertEqual(classify_name("bringup.launch.py")[:2], ("ros_launch", "ros"))
        self.assertEqual(classify_name("robot_ros2_control.xacro")[:2], ("ros_control", "ros"))
        self.assertEqual(classify_name("parts_bom.csv")[:2], ("bom", "bom"))
        self.assertEqual(classify_name("policy.onnx")[:2], ("policy", "policy"))
        self.assertEqual(classify_name("data/run.mcap")[:2], ("log", "log"))
        self.assertFalse(classify_name("weird.foo")[2])  # unrecognized


class ScanFolderTests(unittest.TestCase):
    def _project(self, root: str) -> None:
        _write(root, "robot.urdf")
        _write(root, "meshes/base.stl")
        _write(root, "cad/arm.step")
        _write(root, "ros/package.xml")
        _write(root, "ros/bringup.launch.py")
        _write(root, "ros/robot_ros2_control.xacro")
        _write(root, "controllers/policy.py")
        _write(root, "controllers/policy.onnx")
        _write(root, "data/run.mcap")
        _write(root, "data/demo.hdf5")
        _write(root, "bom/parts_bom.csv")
        _write(root, "docs/README.md")
        _write(root, "config.yaml")
        _write(root, "weird.foo")           # unrecognized -> surfaced, not dropped
        _write(root, ".git/config")          # skipped (ignored dir)
        _write(root, ".DS_Store")            # skipped (junk file)

    def test_full_project_graph(self):
        with tempfile.TemporaryDirectory() as root:
            self._project(root)
            bundle = scan_folder(root, bundle_id="b_test")
            self.assertTrue(bundle.validate().ok)
            refs = {a.extracted_refs[0] for a in bundle.artifacts}
            self.assertIn("robot.urdf", refs)
            self.assertNotIn(".DS_Store", refs)                 # junk skipped
            self.assertFalse(any(r.startswith(".git") for r in refs))  # VCS dir skipped

            summary = project_graph_summary(bundle)
            self.assertEqual(summary["first_runnable_sim_target"], "robot.urdf")
            self.assertEqual(summary["blockers"], [])
            self.assertTrue(summary["has_ros"] and summary["has_bom"] and summary["has_logs"])
            self.assertTrue(summary["has_cad"] and summary["has_policies"])
            self.assertEqual(summary["unrecognized"], 1)        # only weird.foo

            urdf = next(a for a in bundle.artifacts if a.extracted_refs[0] == "robot.urdf")
            self.assertEqual(len(urdf.checksum), 64)            # real sha256 checksum
            self.assertEqual(urdf.parse_status, ParseStatus.OK)

    def test_project_without_robot_model_reports_blocker(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, "notes.md")
            _write(root, "parts_bom.csv")
            summary = project_graph_summary(scan_folder(root))
            self.assertIsNone(summary["first_runnable_sim_target"])
            self.assertTrue(summary["blockers"])


class ScanZipTests(unittest.TestCase):
    def test_zip_entries_classified_without_extraction(self):
        with tempfile.TemporaryDirectory() as root:
            zip_path = os.path.join(root, "project.zip")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("robot.urdf", "<robot/>")
                archive.writestr("meshes/base.stl", b"solid")
                archive.writestr("__MACOSX/junk", b"junk")   # skipped
            bundle = scan_zip(zip_path)
            refs = {a.extracted_refs[0] for a in bundle.artifacts}
            self.assertIn("robot.urdf", refs)
            self.assertIn("meshes/base.stl", refs)
            self.assertNotIn("__MACOSX/junk", refs)
            self.assertEqual(project_graph_summary(bundle)["first_runnable_sim_target"], "robot.urdf")


if __name__ == "__main__":
    unittest.main()
