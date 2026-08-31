"""Whole-project artifact classifier + Project Graph (Input Ingestion Plan, Phase 2).

Acceptance: dropping a folder with a URDF + meshes + config + ROS + logs + BOM produces a classified
Project Graph with a first runnable sim target and no blockers; junk/VCS files are skipped; an unrecognized
file is surfaced (not silently dropped); a project without any robot description reports a blocker.

Plus (2026-07-31) the MODEL PICKER: a real robot package ships several .xml files and only one of them is the
robot. Picking ``robot_models[0]`` -- directory-walk order, i.e. alphabetical -- got 11 of 20 real MuJoCo
Menagerie robots wrong, handing the customer a bare gripper, a scene with a table welded on, or an include
fragment that cannot load at all. The ScanFolderTests fixture below cannot catch that: it holds exactly ONE
robot.urdf, so every ordering is correct. ModelPickerTests uses realistic multi-file model directories.
"""

import os
import tempfile
import unittest
import zipfile

from virturoid.schemas.input_bundle import ParseStatus
from virturoid.services.input_classifier import (
    classify_name,
    project_graph_summary,
    rank_model_candidates,
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


class UnsupportedModelFormatTests(unittest.TestCase):
    """USD and SDF were ADVERTISED and never implemented.

    ``_EXT_MAP`` classified ``.usd/.usda/.usdc/.usdz`` as ``robot_model`` and ``.sdf`` as ``world_or_model``,
    i.e. straight into the sim-target candidate set, and the "no model found" blocker read "(URDF/MJCF/SDF/USD)".
    There is no USD reader and no SDF reader in this repo -- ``model_import.import_model`` takes .xml/.mjcf/.urdf
    and nothing else -- so a customer dropping an Isaac project had their .usd NOMINATED as the first runnable
    sim target, watched it fail to import, and got a GENERATED robot with the reason "none of the 1 model
    file(s) could be imported". USD is the format NVIDIA trained the industry on, so that is the likeliest
    first impression a prospective customer forms.
    """

    def test_usd_and_sdf_are_recognized_but_never_nominated(self):
        for name in ("robot.usd", "robot.usda", "robot.usdc", "robot.usdz", "world.sdf"):
            artifact_type, category, recognized = classify_name(name)
            self.assertEqual((artifact_type, category), ("robot_model_unsupported", "model"), name)
            self.assertTrue(recognized, f"{name} is a robot description, not junk — it must not read unrecognized")

    def test_a_usd_only_project_says_so_and_says_what_to_do(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, "robot.usd")
            _write(root, "meshes/base.stl")
            summary = project_graph_summary(scan_folder(root))
            self.assertIsNone(summary["first_runnable_sim_target"])   # never hand over a file we cannot read
            self.assertEqual(summary["unsupported_models"], ["robot.usd"])
            blob = " ".join(summary["blockers"])
            self.assertIn("URDF", blob)                               # what we DO read
            self.assertIn("robot.usd", blob)                          # the file, named
            self.assertIn("no USD importer", blob)                    # the truth, not "no robot description"
            self.assertIn("Isaac", blob)                              # ...and the step that works

    def test_a_usd_beside_a_urdf_does_not_displace_the_urdf(self):
        with tempfile.TemporaryDirectory() as root:
            _write(root, "spot.usd")                                  # exact package-name stem: outscored the urdf
            _write(root, "spot_description.urdf")
            summary = project_graph_summary(scan_folder(root))
            self.assertEqual(summary["first_runnable_sim_target"], "spot_description.urdf")
            self.assertEqual(summary["unsupported_models"], ["spot.usd"])   # surfaced, not silently dropped
            self.assertEqual(summary["blockers"], [])                       # a readable model exists

    def test_import_model_names_the_format_and_the_conversion(self):
        import importlib.util
        if importlib.util.find_spec("mujoco") is None:
            self.skipTest("import_model needs MuJoCo")
        from virturoid.services.model_import import import_model
        with tempfile.TemporaryDirectory() as root:
            _write(root, "robot.usda", b"#usda 1.0\n")
            out = import_model(os.path.join(root, "robot.usda"))
            self.assertFalse(out["ok"])
            self.assertEqual(out["format"], "OpenUSD")
            self.assertIn("no USD importer", out["note"])
            self.assertIn("URDF", out["note"])


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


# --- the model picker: which of a package's many .xml files is THE ROBOT? -------------------------------
# Shapes taken from the real files that were mis-picked (boston_dynamics_spot/, shadow_hand/).
_SCENE_XML = """<mujoco model="spot scene">
  <include file="spot.xml"/>
  <statistic center="0.15 0.1 0.38" extent=".8"/>
  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane"/>
    <body name="table" pos="0 -1 .24"><geom type="box" size=".6 .5 .24" mass="46.7"/></body>
  </worldbody>
</mujoco>
"""
_ROBOT_XML = """<mujoco model="spot">
  <compiler angle="radian"/>
  <worldbody>
    <body name="body"><freejoint/><geom type="box" size="0.3 0.15 0.08" mass="10"/>
      <body name="fl_hip"><joint name="fl_hx" type="hinge" axis="1 0 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1"/></body>
    </body>
  </worldbody>
  <actuator><motor joint="fl_hx" gear="20"/></actuator>
</mujoco>
"""
_KEYFRAMES_XML = """<mujoco>
  <keyframe><key name="stand" qpos="0 0 0.5 1 0 0 0 0"/></keyframe>
</mujoco>
"""


def _mjcf(name: str, bodies: int) -> str:
    inner = "".join(f'<body name="b{i}"><joint name="j{i}" type="hinge"/>'
                    f'<geom type="box" size=".1 .1 .1"/></body>' for i in range(bodies))
    return f'<mujoco model="{name}"><worldbody>{inner}</worldbody></mujoco>'


class ModelPickerTests(unittest.TestCase):
    def _package(self, root: str) -> str:
        """A REALISTIC model directory: a scene wrapper, the robot, and an include fragment."""
        pkg = os.path.join(root, "boston_dynamics_spot")
        _write(pkg, "scene.xml", _SCENE_XML.encode())
        _write(pkg, "spot.xml", _ROBOT_XML.encode())
        _write(pkg, "keyframes.xml", _KEYFRAMES_XML.encode())
        return pkg

    def test_picks_the_robot_not_the_scene_or_the_fragment(self):
        with tempfile.TemporaryDirectory() as root:
            pkg = self._package(root)
            summary = project_graph_summary(scan_folder(pkg))
            # alphabetical order here is keyframes.xml < scene.xml < spot.xml -- the pick must not be alphabet
            self.assertEqual(summary["first_runnable_sim_target"], "spot.xml")
            self.assertEqual(summary["blockers"], [])
            # an <include> fragment declares no <body>; robot_import REFUSES it, so it is not a candidate at all
            self.assertNotIn("keyframes.xml", summary["robot_models"])
            # the scene is still listed (a customer may want it) but never leads
            self.assertIn("scene.xml", summary["robot_models"])
            self.assertGreater(summary["robot_models"].index("scene.xml"), 0)
            # and the choice is auditable
            top = summary["model_ranking"][0]
            self.assertEqual(top["ref"], "spot.xml")
            self.assertTrue(any("boston_dynamics_spot" in r for r in top["reasons"]), top["reasons"])

    def test_scene_is_penalised_for_wrapping_the_robot_and_adding_props(self):
        with tempfile.TemporaryDirectory() as root:
            pkg = self._package(root)
            ranked = {c["ref"]: c for c in project_graph_summary(scan_folder(pkg))["model_ranking"]}
            reasons = " ".join(ranked["scene.xml"]["reasons"])
            self.assertIn("include", reasons)
            self.assertIn("floor/props", reasons)
            self.assertLess(ranked["scene.xml"]["score"], ranked["spot.xml"]["score"])

    def test_a_folder_of_only_include_fragments_reports_a_blocker(self):
        # honest: we will not nominate a file we already know the importer refuses
        with tempfile.TemporaryDirectory() as root:
            _write(root, "keyframes.xml", _KEYFRAMES_XML.encode())
            summary = project_graph_summary(scan_folder(root))
            self.assertIsNone(summary["first_runnable_sim_target"])
            self.assertTrue(summary["blockers"])

    def test_compiled_body_and_actuator_counts_break_a_name_tie(self):
        """franka_emika_panda's shape: two importable models, neither named after the folder. hand.xml is the
        bare gripper (3 bodies / 2 DOF) and panda.xml is the arm (12 / 8) -- only a real compile separates them,
        so the picker must use it. Uses a stub probe so the assertion stays hermetic."""
        with tempfile.TemporaryDirectory() as root:
            pkg = os.path.join(root, "customer_project")
            _write(pkg, "alpha.xml", _mjcf("alpha", 2).encode())
            _write(pkg, "omega.xml", _mjcf("omega", 4).encode())
            refs = ["alpha.xml", "omega.xml"]

            static = rank_model_candidates(refs, root_folder=pkg)
            self.assertEqual(static[0]["ref"], "alpha.xml")      # no evidence -> alphabetical, as before

            counts = {"alpha.xml": {"ok": True, "bodies": 3, "actuators": 2},
                      "omega.xml": {"ok": True, "bodies": 12, "actuators": 8}}
            ranked = rank_model_candidates(
                refs, root_folder=pkg,
                compile_probe=lambda p: counts[os.path.basename(p)])
            self.assertEqual(ranked[0]["ref"], "omega.xml")
            self.assertEqual(ranked[0]["actuators"], 8)

    def test_a_model_that_does_not_compile_loses_to_one_that_does(self):
        with tempfile.TemporaryDirectory() as root:
            pkg = os.path.join(root, "customer_project")
            _write(pkg, "alpha.xml", _mjcf("alpha", 2).encode())
            _write(pkg, "omega.xml", _mjcf("omega", 2).encode())
            probe = {"alpha.xml": {"ok": False, "bodies": 0, "actuators": 0},
                     "omega.xml": {"ok": True, "bodies": 6, "actuators": 5}}
            ranked = rank_model_candidates(["alpha.xml", "omega.xml"], root_folder=pkg,
                                           compile_probe=lambda p: probe[os.path.basename(p)])
            self.assertEqual(ranked[0]["ref"], "omega.xml")

    def test_a_urdf_carrying_a_mujoco_extension_block_is_not_read_as_a_fragment(self):
        """A URDF may legally carry <mujoco><compiler/></mujoco> (model_import injects one and explicitly
        respects a customer-provided one) and declares <link>, never <body> — so "has <mujoco>, has no <body>"
        would throw the customer's own robot away. The ROOT element decides."""
        with tempfile.TemporaryDirectory() as root:
            pkg = os.path.join(root, "cust")
            _write(pkg, "arm.urdf",
                   ('<?xml version="1.0"?><robot name="arm"><mujoco><compiler fusestatic="false"/></mujoco>'
                    '<link name="base_link"/><link name="upper"/>'
                    '<joint name="pan" type="revolute"><parent link="base_link"/><child link="upper"/>'
                    "</joint></robot>").encode())
            _write(pkg, "keyframes.xml", _KEYFRAMES_XML.encode())
            summary = project_graph_summary(scan_folder(pkg))
            self.assertEqual(summary["first_runnable_sim_target"], "arm.urdf")

    def test_assembly_root_beats_its_own_part_files(self):
        """flexiv_rizon4's shape: the robot <include>s assets/linkN/linkN.xml. Those parts are valid MJCF with
        bodies AND their folder name matches their filename exactly, so naive name+include evidence picks ONE
        LINK of the arm. Direction decides it: including files BELOW you makes you the assembly root."""
        with tempfile.TemporaryDirectory() as root:
            pkg = os.path.join(root, "flexiv_rizon4")
            _write(pkg, "flexiv_rizon4.xml",
                   ('<mujoco model="rizon4"><include file="assets/link0/link0.xml"/>'
                    '<include file="assets/link1/link1.xml"/><worldbody><body name="base"/></worldbody>'
                    "</mujoco>").encode())
            _write(pkg, "assets/link0/link0.xml", _mjcf("link0", 1).encode())
            _write(pkg, "assets/link1/link1.xml", _mjcf("link1", 1).encode())
            summary = project_graph_summary(scan_folder(pkg))
            self.assertEqual(summary["first_runnable_sim_target"], "flexiv_rizon4.xml")


_MENAGERIE = os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie")


@unittest.skipUnless(os.path.isdir(_MENAGERIE), "MuJoCo Menagerie corpus not cached on this machine")
class MenagerieCorpusTests(unittest.TestCase):
    """The real corpus, when it happens to be on the machine. Skipped cleanly so CI stays hermetic.

    Every one of these picked the WRONG file before the evidence-based picker: 9 handed the customer a
    different robot entirely (a bare gripper, a generic arm, a template quadcopter) and the rest a scene with
    non-robot props welded on as robot links.
    """

    CASES = {
        "franka_emika_panda": "panda.xml",        # was hand.xml       -- the bare gripper, 3 bodies / 2 DOF
        "hello_robot_stretch_3": "stretch.xml",   # was scene.xml      -- robot + a 1.2 m, 46.7 kg table
        "boston_dynamics_spot": "spot.xml",       # was scene.xml      -- Spot replaced by a generated quadruped
        "universal_robots_ur5e": "ur5e.xml",      # was scene.xml      -- replaced by a generic arm
        "shadow_hand": "right_hand.xml",          # was keyframes.xml  -- replaced by a generic arm
        "skydio_x2": "x2.xml",                    # was scene.xml      -- replaced by a template quadcopter
        "pal_talos": "talos.xml",                 # was scene_motor.xml
        "booster_t1": "t1.xml",                   # was scene.xml
        "stanford_tidybot": "tidybot.xml",        # was base.xml       -- the mobile base only, no arm
        "trossen_vx300s": "vx300s.xml",           # was scene.xml
        "wonik_allegro": "right_hand.xml",        # was left_hand.xml
        "unitree_go2": "go2.xml",                 # already right -- must stay right
        "unitree_g1": "g1.xml",
        "anybotics_anymal_c": "anymal_c.xml",
        "ufactory_lite6": "lite6.xml",
    }

    def test_real_packages_pick_the_customers_own_robot(self):
        wrong = []
        for pkg, want in self.CASES.items():
            root = os.path.join(_MENAGERIE, pkg)
            if not os.path.isdir(root):
                continue
            got = project_graph_summary(scan_folder(root))["first_runnable_sim_target"]
            if got != want:
                wrong.append(f"{pkg}: picked {got}, want {want}")
        self.assertEqual(wrong, [])


if __name__ == "__main__":
    unittest.main()
