"""§24 / §30: the exported ROS2 package is structurally real and RUNS the exported controller."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.trained_policy import ActionDimension, TrainedPolicy
from virturoid.services.controller_exporter import export_controller_bundle
from virturoid.services.ros2_exporter import export_ros2_package, maybe_export_ros2_package


def _fake_package(root: Path):
    """Minimal package on disk: a robot_genome + a tiny scene set (enough for the ROS2 export)."""
    (root / "robot").mkdir(parents=True, exist_ok=True)
    (root / "simulation").mkdir(parents=True, exist_ok=True)
    genome = {"id": "test_arm", "joints": [{"name": "base_yaw"}, {"name": "shoulder_pitch"},
                                           {"name": "elbow_pitch"}]}
    (root / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
    (root / "robot" / "bill_of_materials.json").write_text(json.dumps(
        {"actuator_map": {"base_yaw": "Dynamixel XM430-W350-T", "shoulder_pitch": "T-Motor AK80-9",
                          "elbow_pitch": "Dynamixel XM430-W350-T"}}), encoding="utf-8")
    (root / "simulation" / "scene_set.json").write_text(json.dumps(
        {"scenes": [{"objects": [{"pose_xyz_rpy": [0.42, -0.05, 0.05, 0, 0, 0]}]}]}), encoding="utf-8")


def _policy() -> TrainedPolicy:
    joints = ["base_yaw", "shoulder_pitch", "elbow_pitch"]
    return TrainedPolicy(
        id="p1", name="reach", robot_genome_id="test_arm", joint_names=joints,
        action_dimensions=[ActionDimension(j, -1.0, 1.0) for j in joints],
        control_frequency_hz=20.0, weights=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.2], [0.0, -0.1, 0.1]],
        input_features=["x", "y", "bias"],
        safety_clamps={"joint_position_limits": [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]})


def _run_bundled_test(root: Path, fn: str) -> None:
    """Import and run a function from the package's generated regression test (no ROS2 needed)."""
    spec = importlib.util.spec_from_file_location("ros2_regr", root / "test" / "test_task_regression.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    getattr(mod, fn)()


class Ros2ExportTests(unittest.TestCase):
    def test_structure_without_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _fake_package(root)
            pkg = export_ros2_package(root)
            for rel in ("package.xml", "setup.py", "launch/evaluate.launch.py", "config/robot.yaml",
                        "virturoid_robot/evaluation_node.py", "test/test_task_regression.py"):
                self.assertTrue((pkg / rel).exists(), rel)
            cfg = json.loads((pkg / "config" / "robot.yaml").read_text())
            self.assertEqual(cfg["joints"], ["base_yaw", "shoulder_pitch", "elbow_pitch"])
            self.assertFalse(cfg["has_controller"])
            # node + test compile as valid Python
            compile((pkg / "virturoid_robot" / "evaluation_node.py").read_text(), "node.py", "exec")
            _run_bundled_test(pkg, "test_config_has_joints")

    def test_embeds_and_runs_the_exported_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _fake_package(root)
            export_controller_bundle(root, _policy())          # write software/controller/*
            pkg = maybe_export_ros2_package(root)
            self.assertIsNotNone(pkg)
            cfg = json.loads((pkg / "config" / "robot.yaml").read_text())
            self.assertTrue(cfg["has_controller"])
            self.assertEqual(cfg["target_positions"], [[0.42, -0.05]])   # pulled from the scene set
            self.assertTrue((pkg / "virturoid_robot" / "controller.py").exists())
            self.assertTrue((pkg / "virturoid_robot" / "policy_params.json").exists())
            # the bundled regression test runs the REAL controller and checks clamped joint targets
            _run_bundled_test(pkg, "test_controller_runs_if_present")

    def test_emits_ros2_control_hardware_interface_and_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _fake_package(root)
            pkg = export_ros2_package(root)
            rc = (pkg / "config" / "ros2_control.yaml").read_text()
            self.assertIn("joint_trajectory_controller", rc)
            for j in ("base_yaw", "shoulder_pitch", "elbow_pitch"):
                self.assertIn(j, rc)
            hw = (pkg / "config" / "hardware_interface.yaml").read_text()
            self.assertIn("Dynamixel XM430-W350-T", hw)              # joint -> the REAL BOM actuator
            self.assertIn("T-Motor AK80-9", hw)
            sf_path = pkg / "virturoid_robot" / "safety_filter.py"   # bundled CBF-lite gate
            self.assertTrue(sf_path.exists())
            spec = importlib.util.spec_from_file_location("vq_safety", sf_path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            out, viol = mod.SafetyFilter(lower=[-1.0], upper=[1.0], vel_limit=100.0).clamp([5.0], q=[0.0], dt=0.01)
            self.assertEqual(viol, 1)
            self.assertLessEqual(out[0], 1.0)

    def test_maybe_export_is_safe_without_genome(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(maybe_export_ros2_package(Path(tmp)))    # no genome -> no crash, no package


def _yaml_actuators(text: str) -> dict:
    """{joint: actuator-string-or-None} parsed back out of the written hardware_interface.yaml."""
    out, cur, in_joints = {}, None, False
    for raw in text.splitlines():
        if raw.startswith("joints:"):
            in_joints = True
            continue
        if raw and not raw.startswith((" ", "#")):
            in_joints = False
        if not in_joints or raw.startswith("#"):
            continue
        if raw.startswith("  ") and not raw.startswith("    ") and raw.rstrip().endswith(":"):
            cur = raw.strip()[:-1]
        elif cur and raw.strip().startswith("actuator:"):
            val = raw.split("actuator:", 1)[1].strip()
            out[cur] = None if val == "null" else val.strip('"')
    return out


class HardwareInterfaceNamesRealMotorsTests(unittest.TestCase):
    """The hardware interface must name the part THIS robot's own BOM assigns to each joint's segment.

    Before this test: `bom_builder` keys `actuator_map` by SEGMENT (`leg1_l_0`) while the export looked up the
    URDF/genome JOINT name (`leg1_l_0_joint`), so the intersection was empty and every row emitted the fallback
    `GENERIC position actuator (set from the BOM)` -- measured 0/14 on a composed quadruped, through BOTH the
    package-build door and the agent-facing `export_held` door. The old fixture in this file could not see it
    because it keys its BOM by joint name.
    """

    def _build(self, prompt="a quadruped robot dog that walks"):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot(prompt)

    def _check_every_row(self, pkg: Path, bom: dict, genome: dict):
        text = (pkg / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
        self.assertNotIn("GENERIC position actuator", text)   # never an instruction in place of a failure
        rows = _yaml_actuators(text)
        seg_of = {j["name"]: j["child_link"] for j in genome["joints"]}
        amap = bom["actuator_map"]
        self.assertGreater(len(rows), 0)
        for joint, part in rows.items():
            seg = seg_of[joint]
            self.assertEqual(part, amap[seg],
                             f"{joint} (drives {seg}) names {part!r}; this robot's BOM assigns {amap[seg]!r}")
        return rows, text

    def test_package_build_door_names_the_bom_part_for_every_joint(self):
        from virturoid.services.gene_build import _emit_bom, _write_genome_and_urdf
        gene = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_genome_and_urdf(gene, root, task="walk")   # runs the ROS2 export INSIDE, before the BOM
            _emit_bom(gene, root, task="walk")
            bom = json.loads((root / "robot" / "bill_of_materials.json").read_text(encoding="utf-8"))
            genome = json.loads((root / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
            pkg = root / "export" / "ros2" / "virturoid_robot"
            rows, _ = self._check_every_row(pkg, bom, genome)
            self.assertEqual(len(rows), len(genome["joints"]))

    def test_a_rebuild_into_the_same_directory_does_not_ship_the_previous_robots_motors(self):
        """THE REGRESSION, pinned. `autonomous_build` re-runs `build_gene_package` into the SAME output_dir
        after a redesign. The actuator map used to be computed only `if not bill_of_materials.json.exists()`,
        so the second build found the FIRST robot's parts list and mapped every joint from it: measured
        14 of 14 rows naming a motor that is not the one the package's own bill_of_materials.json assigns
        (`leg1_l_0_joint` -> "Dynamixel XM540-W270-T" while the shipped BOM says "Harmonic Drive FHA-40C-160"),
        with GENERIC count 0, no `unresolved_joints:` block, and a header claiming 14 of 14 mapped.

        The pin is DISAGREEMENT with the package's own shipped parts list -- not "the map is non-empty", which
        the stale map also satisfied. It fails if the injection is made conditional again by any route.
        """
        from virturoid.services.edit_operators import scale_robot
        from virturoid.services.gene_build import _emit_bom, _write_genome_and_urdf
        first = self._build()
        second, _ = scale_robot(first, factor=2.6)     # same topology, motors an order of magnitude apart
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_genome_and_urdf(first, root, task="walk")
            _emit_bom(first, root, task="walk")
            bom_first = json.loads(
                (root / "robot" / "bill_of_materials.json").read_text(encoding="utf-8"))["actuator_map"]

            _write_genome_and_urdf(second, root, task="walk")      # the REBUILD, into the reused directory
            _emit_bom(second, root, task="walk")

            bom = json.loads((root / "robot" / "bill_of_materials.json").read_text(encoding="utf-8"))
            genome = json.loads((root / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
            pkg = root / "export" / "ros2" / "virturoid_robot"
            rows, text = self._check_every_row(pkg, bom, genome)   # every row == the SHIPPED parts list
            self.assertEqual(len(rows), len(genome["joints"]))
            # the two builds must actually disagree, or this test could pass on a stale map by coincidence
            differing = [s for s in bom["actuator_map"] if bom_first.get(s) != bom["actuator_map"][s]]
            self.assertGreater(len(differing), 0,
                               "the rescaled robot selected the same motors -- this case cannot see staleness")
            self.assertNotIn("unresolved_joints:", text)           # a real map, not gaps papering over it

    def test_agent_export_held_door_names_the_bom_part_for_every_joint(self):
        """The agent-facing door, pinned on ITS OWN artifacts rather than on the gene_build injection.

        Reverting `export_held` alone used to leave all 8 tests green, because the handed-over actuator map
        carried the YAML no matter what `export_held` wrote. So this asserts what export_held is actually
        responsible for: the parts list lands INSIDE the package at `robot/bill_of_materials.json`, agreeing
        with the `bom.json` it returns -- and a disk-only consumer (a re-export with NO map handed in)
        reproduces the same rows from it. Drop that write and the re-export can no longer be checked against
        the package's own list.
        """
        from virturoid.services import agent_design_tools as adt
        from virturoid.services import session_state as S
        gene = self._build()
        rid = S.put_robot(gene, label="ros2-hw-test")
        res = adt.export_held({"robot_id": rid, "formats": ["urdf", "ros2", "bom"]})
        self.assertTrue(res.get("ok"), res)
        pkg = Path(res["artifacts"]["ros2"])
        root = pkg.parent.parent.parent
        bom = json.loads(Path(res["artifacts"]["bom"]).read_text(encoding="utf-8"))
        genome = json.loads((root / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
        rows, _ = self._check_every_row(pkg, bom, genome)

        # (a) the package carries the parts list where its own consumers look for it, and it is the SAME list
        in_pkg = root / "robot" / "bill_of_materials.json"
        self.assertTrue(in_pkg.exists(), "export_held shipped no bill of materials inside the package")
        self.assertEqual(json.loads(in_pkg.read_text(encoding="utf-8"))["actuator_map"], bom["actuator_map"])
        # (b) a consumer reading ONLY the package off disk reaches the same rows (no handed-in map)
        redone = export_ros2_package(root, package_name="disk_only")
        self.assertEqual(_yaml_actuators((redone / "config" / "hardware_interface.yaml").read_text()), rows)

    def test_an_unresolvable_joint_is_a_named_gap_not_an_instruction(self):
        """A part that genuinely cannot be resolved must READ as a gap. The old fallback string was phrased as
        a to-do, so a total failure looked like remaining setup."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            genome = {"id": "g", "joints": [{"name": "a_joint", "child_link": "a"},
                                            {"name": "b_joint", "child_link": "b"}]}
            (root / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
            (root / "robot" / "bill_of_materials.json").write_text(
                json.dumps({"actuator_map": {"a": "T-Motor AK80-9"}}), encoding="utf-8")
            pkg = export_ros2_package(root)
            text = (pkg / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertNotIn("GENERIC position actuator", text)
            self.assertIn("no BOM entry for joint b_joint", text)
            self.assertIn("unresolved_joints:", text)
            self.assertEqual(_yaml_actuators(text), {"a_joint": "T-Motor AK80-9", "b_joint": None})
            readme = (pkg / "README.md").read_text(encoding="utf-8")
            self.assertIn("1 of 2 commandable joints", readme)   # the README states the real hit rate

    def test_a_known_segment_absent_from_the_parts_list_is_a_gap_not_another_part(self):
        """THE MECHANISM that turns a gap into a guess. `_resolve_actuator` tried (segment, joint, stem) in
        order, so a joint whose driven segment IS known but is simply missing from the parts list fell through
        to the joint name and the `<segment>_joint` stem -- and named whatever those happened to hit. That is
        how a stale/partial map produces confident wrong rows instead of honest nulls.

        Here `b_joint` drives segment `b`, which the parts list does not carry; the list DOES carry entries
        under both fall-through keys, naming parts that belong to nothing on this robot.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            genome = {"id": "g", "joints": [{"name": "a_joint", "child_link": "a"},
                                            {"name": "b_joint", "child_link": "b"}]}
            (root / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
            (root / "robot" / "bill_of_materials.json").write_text(json.dumps({"actuator_map": {
                "a": "T-Motor AK80-9",
                "b_joint": "Unitree M107 (B2/H1-class)",   # the joint-name fall-through
                "b": None,                                  # present but assigns nothing -> still a gap
            }}), encoding="utf-8")
            text = (export_ros2_package(root) / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            rows = _yaml_actuators(text)
            self.assertIsNone(rows["b_joint"], "a known-but-unlisted segment must be a null, not a guess")
            self.assertNotIn("Unitree M107", text)          # the unrelated part is never named
            self.assertIn("looked for the part assigned to b", text)   # names the SEGMENT it looked for
            self.assertEqual(rows["a_joint"], "T-Motor AK80-9")

    # ---- the OTHER namespace: bom_builder.build_bom_from_genome keys actuator_map by JOINT NAME ----------
    #
    # `build_bom` keys by SEGMENT; `build_bom_from_genome` (bom_builder.py:1333) keys by JOINT NAME, and
    # `emit_genome_bom` writes THAT dict to the same `robot/bill_of_materials.json`. Four production sites read
    # it back off disk with no handed-in map (mvp.py:571, package_writer_registry.py:230, agent.py:200,
    # autonomous_build.py:507). A rule that tries only the segment key resolves NONE of it.

    _TEMPLATE_ARM_GENOME = {
        "id": "template_arm", "species": "manipulator", "robot_class": "manipulator",
        "links": ["base_link", "upper_link", "forearm_link", "wrist_link"],
        "joints": [
            {"name": "base_yaw", "joint_type": "revolute", "parent_link": "base_link",
             "child_link": "upper_link", "limit": {"effort": 40.0, "lower": -3.14, "upper": 3.14}},
            {"name": "shoulder_pitch", "joint_type": "revolute", "parent_link": "upper_link",
             "child_link": "forearm_link", "limit": {"effort": 40.0, "lower": -1.57, "upper": 1.57}},
            {"name": "elbow_pitch", "joint_type": "revolute", "parent_link": "forearm_link",
             "child_link": "wrist_link", "limit": {"effort": 18.0, "lower": -1.57, "upper": 1.57}}]}

    def test_the_template_manipulator_package_names_its_real_parts(self):
        """THE REGRESSION, pinned on the producer that actually feeds this path.

        Measured on this exact genome with the strict-segment rule in place: the BOM assigned
        `base_yaw`/`shoulder_pitch`/`elbow_pitch` real motors and all three rows came out `actuator: null`,
        under shipped text reading "3 joint(s) are NOT mapped: this package's bill of materials names no part
        for them" -- FALSE about the file sitting beside it. The genome records `child_link`, so resolution
        stopped at the segment key and never tried the joint name the parts list is keyed by.
        """
        from virturoid.services.bom_builder import emit_genome_bom
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            (root / "robot" / "robot_genome.json").write_text(
                json.dumps(self._TEMPLATE_ARM_GENOME), encoding="utf-8")
            bom = emit_genome_bom(root, task="pick and place")
            amap = bom["actuator_map"]
            self.assertEqual(set(amap), {"base_yaw", "shoulder_pitch", "elbow_pitch"},
                             "this producer keys by JOINT NAME -- if that changed, this test is measuring nothing")
            text = (export_ros2_package(root, package_name="tmpl")
                    / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            rows = _yaml_actuators(text)
            self.assertEqual(rows, amap)                       # every row == the package's own parts list
            self.assertNotIn(None, set(rows.values()))
            self.assertNotIn("unresolved_joints:", text)
            self.assertNotIn("names no part for them", text)   # never a false claim about the file beside it
            self.assertIn("PARTS SOURCE: read from robot/bill_of_materials.json", text)
            self.assertIn("3 of 3 commandable joints", text)

    def test_the_mobile_base_package_names_its_wheel_motors(self):
        """The other genome-keyed production path (`package_writer_registry._write_mobile_base_package`).

        Two defects met here. The export resolved nothing because the map is joint-keyed; and the map was
        EMPTY in the first place because `build_bom_from_genome` filtered joints to revolute/prismatic while a
        differential-drive base declares both wheels `continuous` -- so the parts list shipped `dof: 0` and no
        actuator at all for the two motors the robot is built around.
        """
        from virturoid.services.bom_builder import emit_genome_bom
        from virturoid.mobile_base import build_mobile_base_project
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mb"
            d = Path(build_mobile_base_project("a warehouse mobile base that navigates aisles", out)["output_dir"])
            bom = emit_genome_bom(d, task="navigate")
            genome = json.loads((d / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
            wheels = [j["name"] for j in genome["joints"]]
            self.assertEqual(bom["dof"], len(wheels), "the wheels must be counted as driven joints")
            self.assertEqual(set(bom["actuator_map"]), set(wheels))
            text = (export_ros2_package(d, package_name="mb")
                    / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            rows = _yaml_actuators(text)
            self.assertEqual(rows, bom["actuator_map"])
            self.assertNotIn(None, set(rows.values()))
            self.assertNotIn("unresolved_joints:", text)

    def test_a_joint_keyed_map_never_lets_a_joint_borrow_a_same_named_segments_part(self):
        """THE MIRROR of the original defect, constructed.

        Original: a known-but-unlisted SEGMENT fell through to the joint name and borrowed an unrelated part.
        Mirror: in a JOINT-keyed list, a joint with no entry of its own falls through to the segment it drives
        -- and here that segment is named `elbow`, which is ALSO the name of a different joint. The part under
        `elbow` belongs to joint `elbow`; joint `wrist` taking it would be exactly as wrong, in the opposite
        direction. The last two assertions SHOW that: forced into the segment namespace, `wrist` does borrow it.
        """
        from virturoid.services.ros2_exporter import _resolve_actuator, _map_key_namespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            genome = {"id": "g", "links": ["base", "elbow", "tip", "upper"],
                      "joints": [{"name": "wrist", "child_link": "elbow"},     # drives the segment named `elbow`
                                 {"name": "elbow", "child_link": "tip"},       # a DIFFERENT thing, same string
                                 {"name": "shoulder", "child_link": "upper"}]}
            (root / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
            amap = {"elbow": "Unitree M107 (B2/H1-class)", "shoulder": "T-Motor AK80-9"}   # `wrist` is absent
            (root / "robot" / "bill_of_materials.json").write_text(
                json.dumps({"actuator_map": amap}), encoding="utf-8")
            text = (export_ros2_package(root, package_name="mirror")
                    / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            rows = _yaml_actuators(text)
            self.assertEqual(rows, {"wrist": None, "elbow": "Unitree M107 (B2/H1-class)",
                                    "shoulder": "T-Motor AK80-9"})
            self.assertIn("looked for the part assigned to wrist", text)   # its OWN name, not the segment's
            self.assertNotIn("looked for the part assigned to elbow", text)
            # what the blocked fall-through would have done, shown rather than asserted about
            seg_of = {j["name"]: j["child_link"] for j in genome["joints"]}
            self.assertEqual(_map_key_namespace(amap, list(seg_of), seg_of, genome["links"])["namespace"],
                             "joint")
            self.assertEqual(_resolve_actuator("wrist", amap, seg_of, "segment")[0],
                             "Unitree M107 (B2/H1-class)",
                             "the mirror defect must be REACHABLE, or this test proves nothing")

    def test_the_key_namespace_is_decided_from_the_body_not_assumed(self):
        """The determination itself, pinned on one fixture of each namespace -- with the mutation that shows
        no CONSTANT answer can pass. Force the namespace to `joint` and the segment-keyed fixture resolves
        nothing; force it to `segment` and the joint-keyed one resolves nothing. Only a determination read off
        the body clears both.
        """
        from virturoid.services.ros2_exporter import _map_key_namespace, _resolve_actuator
        seg_seg_of = {"a_joint": "a", "b_joint": "b"}                       # build_bom shape
        seg_map = {"a": "T-Motor AK80-9", "b": "Dynamixel XM430-W350-T"}
        jnt_seg_of = {"base_yaw": "upper_link", "shoulder_pitch": "forearm_link"}   # build_bom_from_genome shape
        jnt_map = {"base_yaw": "T-Motor AK10-9", "shoulder_pitch": "Unitree GO-M8010-6"}
        cases = (("segment", seg_map, seg_seg_of, ["a", "b"]),
                 ("joint", jnt_map, jnt_seg_of, ["upper_link", "forearm_link"]))
        for expected, amap, seg_of, links in cases:
            ns = _map_key_namespace(amap, list(seg_of), seg_of, links)
            self.assertEqual(ns["namespace"], expected, ns)
            self.assertEqual({j: _resolve_actuator(j, amap, seg_of, ns["namespace"])[0] for j in seg_of},
                             {j: amap[seg_of[j] if expected == "segment" else j] for j in seg_of})
        # THE MUTATION: a determination hard-coded to either single answer blinds the other fixture entirely.
        for forced, (_e, amap, seg_of, _l) in (("joint", cases[0]), ("segment", cases[1])):
            resolved = {j: _resolve_actuator(j, amap, seg_of, forced)[0] for j in seg_of}
            self.assertEqual(set(resolved.values()), {None},
                             f"forcing the namespace to {forced!r} must lose every row of the other fixture")

    def test_a_parts_list_whose_namespace_cannot_be_decided_is_disclosed_not_guessed_at(self):
        """A map that matches NEITHER name set (a different robot's parts list, the case the PARTS SOURCE
        header exists for) has no decidable namespace. It resolves nothing -- and says so, instead of printing
        the gap wording that would claim the list names no part when it may well name several."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            (root / "robot" / "robot_genome.json").write_text(json.dumps(
                {"id": "g", "links": ["a", "b"], "joints": [{"name": "a_joint", "child_link": "a"},
                                                            {"name": "b_joint", "child_link": "b"}]}),
                encoding="utf-8")
            (root / "robot" / "bill_of_materials.json").write_text(json.dumps({"actuator_map": {
                "leg0_0": "Harmonic Drive FHA-40C-160", "leg1_0": "Dynamixel XM540-W270-T"}}), encoding="utf-8")
            text = (export_ros2_package(root, package_name="undet")
                    / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertEqual(_yaml_actuators(text), {"a_joint": None, "b_joint": None})
            self.assertNotIn("Harmonic Drive", text)              # nothing borrowed from a list we cannot read
            self.assertIn("KEY NAMESPACE: UNDETERMINED", text)
            self.assertNotIn("names no part for them", text)      # the claim it is NOT entitled to make

    def test_the_agent_bom_landing_spot_is_one_of_the_searched_paths(self):
        """`_read_actuator_map`'s search list is production behaviour, not decoration: `export_held` lands the
        parts list at `bom.json`, and reading only the two `bill_of_materials.json` paths made the map EMPTY
        for any consumer opening that package off disk. Reverting the added paths turns every row null here."""
        from virturoid.services.ros2_exporter import _read_actuator_map
        for rel in ("bom.json", "reports/bom.json", "bom/bom.json"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "robot").mkdir(parents=True)
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / "robot" / "robot_genome.json").write_text(
                    json.dumps({"id": "g", "joints": [{"name": "a_joint", "child_link": "a"}]}), encoding="utf-8")
                (root / rel).write_text(json.dumps({"actuator_map": {"a": "T-Motor AK80-9"}}), encoding="utf-8")
                amap, src = _read_actuator_map(root)
                self.assertEqual(amap, {"a": "T-Motor AK80-9"}, rel)
                self.assertIn(rel, src, "the provenance string must name the file it actually read")
                text = (export_ros2_package(root) / "config" / "hardware_interface.yaml").read_text()
                self.assertEqual(_yaml_actuators(text), {"a_joint": "T-Motor AK80-9"}, rel)

    def test_the_header_names_the_parts_list_it_read(self):
        """The header may not assert provenance the code has not checked. It used to say the rows came from
        "the bill of materials" full stop -- which on a rebuild into a reused directory was a DIFFERENT
        robot's. Now it prints the source, including when there is none."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            (root / "robot" / "robot_genome.json").write_text(
                json.dumps({"id": "g", "joints": [{"name": "a_joint", "child_link": "a"}]}), encoding="utf-8")
            # no parts list anywhere: the header must say so, and every row must be a gap
            text = (export_ros2_package(root) / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertIn("PARTS SOURCE: no bill of materials was found", text)
            self.assertEqual(_yaml_actuators(text), {"a_joint": None})
            # read off disk: the header names the file
            (root / "robot" / "bill_of_materials.json").write_text(
                json.dumps({"actuator_map": {"a": "T-Motor AK80-9"}}), encoding="utf-8")
            text = (export_ros2_package(root) / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertIn("PARTS SOURCE: read from robot/bill_of_materials.json", text)
            # handed in by the build: the header says it came from the gene being exported, not from disk
            text = (export_ros2_package(root, package_name="handed",
                                        actuator_map={"a": "Dynamixel XM430-W350-T"},
                                        actuator_map_source="computed for this build by bom_builder.build_bom")
                    / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertIn("PARTS SOURCE: computed for this build by bom_builder.build_bom", text)
            self.assertEqual(_yaml_actuators(text), {"a_joint": "Dynamixel XM430-W350-T"})

    def test_an_empty_handed_in_map_never_falls_back_to_a_file_on_disk(self):
        """An EMPTY map handed in means "this build has no parts list" -- it must NOT be read as "nothing was
        handed in, go look on disk", because what is on disk may be a previous build's robot. Honest nulls."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            (root / "robot" / "robot_genome.json").write_text(
                json.dumps({"id": "g", "joints": [{"name": "a_joint", "child_link": "a"}]}), encoding="utf-8")
            (root / "robot" / "bill_of_materials.json").write_text(
                json.dumps({"actuator_map": {"a": "STALE MOTOR FROM A PREVIOUS BUILD"}}), encoding="utf-8")
            text = (export_ros2_package(root, package_name="empty_handed", actuator_map={},
                                        actuator_map_source="NOT AVAILABLE - could not be computed")
                    / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertNotIn("STALE MOTOR", text)
            self.assertEqual(_yaml_actuators(text), {"a_joint": None})

    def test_a_coupled_joint_stays_absent_from_the_commandable_set(self):
        """A mimic joint has no motor: it must not appear with an actuator OR as a named gap -- it belongs under
        `coupled_joints:`, exempt by design."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "robot").mkdir(parents=True)
            genome = {"id": "g", "joints": [{"name": "drive_joint", "child_link": "drive"},
                                            {"name": "follow_joint", "child_link": "follow"}]}
            (root / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
            (root / "robot" / "bill_of_materials.json").write_text(
                json.dumps({"actuator_map": {"drive": "Dynamixel XM430-W350-T"}}), encoding="utf-8")
            (root / "robot" / "robot.urdf").write_text(
                '<robot name="r"><link name="base"/><link name="drive"/><link name="follow"/>'
                '<joint name="drive_joint" type="revolute"><parent link="base"/><child link="drive"/></joint>'
                '<joint name="follow_joint" type="revolute"><parent link="base"/><child link="follow"/>'
                '<mimic joint="drive_joint" multiplier="-1" offset="0"/></joint></robot>', encoding="utf-8")
            pkg = export_ros2_package(root)
            text = (pkg / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
            self.assertEqual(_yaml_actuators(text), {"drive_joint": "Dynamixel XM430-W350-T"})
            self.assertNotIn("no BOM entry for joint follow_joint", text)   # exempt, not a gap
            self.assertIn("coupled_joints:", text)
            self.assertIn("follow_joint", text.split("coupled_joints:")[1])


if __name__ == "__main__":
    unittest.main()
