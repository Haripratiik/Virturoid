"""A SENSOR THE CUSTOMER'S MODEL DOES NOT DECLARE MUST NOT BE REPORTED AS PRESENT, AND MUST NOT BE BILLED.

MEASURED DEFECT (2026-08-10, real MuJoCo Menagerie Go2 through ``agent_tools.call_tool``). Their file compiles
to ``ncam 0, nsensor 0``. On the robot imported from it we answered:

    verify_robot.vision -> {"camera_part": "Intel RealSense D435i", "fovy_deg": 58.0, "sees": true}
    bom.json            -> one Intel RealSense D435i, $334, inside the headline price

Neither surface was buggy alone: ``bom_builder._sensor_suite`` maps ``robot_class`` to parts, which is a fine
DEFAULT for a robot we are designing. Nothing asked whether this robot was ours to design, so the class default
was applied to a machine the customer already owns -- ``sensor_geometry`` emitted a functional
``<camera name="robot_cam">`` from it, ``camera_perception`` rendered through that camera, and ``sees: true``
became a real measurement of a camera we had invented.

So EVERY TEST HERE READS TWO SURFACES AND ASSERTS THEY AGREE. A one-sided test is what let this ship: each side
passed its own tests the whole time. The control cases are as load-bearing as the defect case -- a robot we
COMPOSED must keep its camera (it is a proposal by construction, the machine does not exist yet), and a part the
customer PINNED is theirs to assert on their own machine.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")
_SENSOR_CATS = ("camera", "lidar", "imu", "force_torque", "gps", "thermal", "microphone", "encoder")


def _menagerie(pkg: str, model: str) -> str:
    p = os.path.join(_MENAGERIE, pkg, model)
    if not os.path.exists(p):
        raise unittest.SkipTest(f"MuJoCo Menagerie not cached at {p}")
    return p


def _go2():
    """The customer's own Go2, imported exactly as the product imports it."""
    from virturoid.services.robot_import import import_robot
    g = import_robot(_menagerie("unitree_go2", "go2.xml"))["gene"]
    assert g is not None, "go2 did not import"
    return g


def _composed():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a quadruped dog robot that navigates a warehouse", llm=None)


def _sensor_lines(bom):
    return [ln for ln in bom["lines"] if ln["category"] in _SENSOR_CATS]


@unittest.skipUnless(_MUJOCO, "the customer's model has to be compiled to be read")
class TheModelIsTheGroundTruth(unittest.TestCase):
    def test_go2_really_does_declare_no_sensing(self):
        """The premise every other test rests on, measured rather than assumed."""
        import mujoco
        m = mujoco.MjModel.from_xml_path(_menagerie("unitree_go2", "go2.xml"))
        self.assertEqual((int(m.ncam), int(m.nsensor)), (0, 0))

    def test_import_records_the_inventory_verbatim(self):
        """UNKNOWN and NONE are different answers, so the empty inventory has to be RECORDED, not inferred from
        its own absence. Everything downstream refuses differently for the two cases."""
        inv = (_go2().metadata or {}).get("source_sensors")
        self.assertIsNotNone(inv, "import must record what the customer's model declares")
        self.assertEqual((inv["ncam"], inv["nsensor"]), (0, 0))
        self.assertEqual(inv["cameras"], [])


@unittest.skipUnless(_MUJOCO, "import + verify need MuJoCo")
class TheTwoSurfacesAgree(unittest.TestCase):
    """The defect itself: `verify_robot.vision` and `bom.json`, read together, against ``ncam 0``."""

    def test_verify_reports_no_camera_and_says_why(self):
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        rid = S.put_robot(_go2(), prompt="customer go2", label="customer go2")
        try:
            env = call_tool("verify_robot", {"robot_id": rid, "mode": "full"})
        finally:
            S.forget_robot(rid)
        self.assertTrue(env.get("ok"), env)
        vision = env["result"].get("vision")
        self.assertIsNotNone(vision, "the refusal IS the answer and must ship; silence is what let the claim back in")
        self.assertFalse(vision["has_camera"])
        self.assertFalse(vision["sees"], "'sees' is a claim about a sensor this model does not declare")
        self.assertNotIn("camera_part", vision, "no part number for hardware the customer does not have")
        # ...and it is legible as a REFUSAL, quoting the number it rests on — the same shape as
        # controller_provenance.what_we_took_from_your_model, which enumerates what was read verbatim.
        self.assertEqual(vision["your_model_declares"], {"cameras": 0, "sensors": 0})
        self.assertIn("ncam 0", vision["we_did_not_add_one"])

    def test_the_bom_does_not_bill_the_camera_into_the_machine(self):
        from virturoid.services.bom_builder import build_bom
        bom = build_bom(_go2(), task="")
        lines = _sensor_lines(bom)
        self.assertTrue(lines, "the RECOMMENDATION is the useful part; the line stays, it is the CLAIM that goes")
        for ln in lines:
            self.assertTrue(ln["proposed"], f"{ln['part']} is not declared by the model and must be marked proposed")
            self.assertIn("PROPOSED ADDITION", ln["detail"])
        totals = bom["totals"]
        billed = round(sum(ln["price_usd"] for ln in lines), 2)
        self.assertEqual(totals["proposed_additions_usd"], billed)
        self.assertAlmostEqual(totals["as_imported_price_usd"] + billed, totals["price_usd"], places=2)
        self.assertGreater(billed, 0.0, "this Go2 is exactly the case where money was invented")

    def test_the_bom_says_which_sensors_were_read_and_which_were_assumed(self):
        """The deliverable table, on the robot itself: every perception line is an inference from robot_class,
        and the BOM has to set that against what the model declares instead of leaving them indistinguishable."""
        from virturoid.services.bom_builder import build_bom
        prov = build_bom(_go2(), task="")["sensor_provenance"]
        self.assertEqual(prov["read_from_your_model"]["ncam"], 0)
        self.assertTrue(prov["perception_lines"])
        for row in prov["perception_lines"]:
            self.assertFalse(row["declared_by_your_model"])
            self.assertIn("robot_class", row["why"])

    def test_no_camera_is_synthesized_into_the_customers_simulated_body(self):
        """THE MECHANISM, not just the report. `sensor_geometry` emitted a functional <camera name="robot_cam">
        from the assumed suite; that camera is what `sees: true` was measured through. If it is still in the
        compiled model, every reporting fix above is one refactor away from being undone."""
        import mujoco
        from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
        gene = _go2()
        m = mujoco.MjModel.from_xml_string(
            gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        self.assertNotIn("robot_cam", cams,
                         "we mounted a camera on the customer's machine and then measured what it saw")

    def test_the_generated_fusion_stack_does_not_imply_the_topics_exist(self):
        """Same defect one layer along: a runnable ROS2 EKF whose IMU input is a part the robot does not carry.
        MEASURED before the fix: sensors wired to /imu/data_raw + /camera/color/image_raw, and `missing` -- this
        module's own honesty channel -- read []."""
        from virturoid.services.sensor_fusion_compiler import compile_sensor_fusion
        fus = compile_sensor_fusion(_go2())
        self.assertTrue(fus["sensors"], "the recommended stack still ships")
        self.assertTrue(all(s["proposed"] for s in fus["sensors"]))
        self.assertTrue(any("HARDWARE NOT ON YOUR ROBOT" in m for m in fus["missing"]),
                        "an unpopulated topic fails at deploy exactly like an unobservable state")
        yaml = fus["_files_content"]["config/sensors.yaml"]
        self.assertIn("proposed: true", yaml, "the file the engineer opens must carry the disclosure itself")
        self.assertNotIn("Each entry is a real part on a real link", yaml)


@unittest.skipUnless(_MUJOCO, "composing + compiling need MuJoCo")
class TheRefusalIsNarrow(unittest.TestCase):
    """A refusal that also silences the legitimate cases is a second defect, not a fix."""

    def test_a_robot_we_designed_keeps_its_camera(self):
        from virturoid.services.bom_builder import build_bom
        from virturoid.services.camera_perception import robot_camera_context
        gene = _composed()
        part, src = robot_camera_context(gene, task="navigate a warehouse")
        self.assertIsNotNone(part, "this robot does not exist yet; its sensor suite IS the design we propose")
        self.assertEqual(src, "our_design")
        bom = build_bom(gene, task="navigate a warehouse")
        cams = [ln for ln in bom["lines"] if ln["category"] == "camera"]
        self.assertTrue(cams)
        self.assertFalse(any(ln["proposed"] for ln in cams),
                         "on a design of ours the whole list is a proposal; there is no fitted machine to split from")
        self.assertNotIn("proposed_additions_usd", bom["totals"])

    def test_a_robot_we_designed_still_gets_a_functional_camera_in_sim(self):
        import mujoco
        from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
        gene = _composed()
        m = mujoco.MjModel.from_xml_string(
            gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        self.assertIn("robot_cam", cams, "the refusal must not reach robots we designed")

    def test_a_camera_the_customer_pins_is_theirs_to_assert(self):
        """A pin is the customer's own instruction -- "this camera is / will be on it" -- and outranks our
        refusal. Without this the fix would make an imported robot permanently unable to have perception."""
        import copy

        from virturoid.services.bom_builder import build_bom
        from virturoid.services.camera_perception import robot_camera_context
        gene = copy.deepcopy(_go2())
        md = dict(gene.metadata or {})
        md["pinned_parts"] = {**(md.get("pinned_parts") or {}), "camera": "Intel RealSense D435i"}
        gene.metadata = md
        part, src = robot_camera_context(gene)
        self.assertEqual(getattr(part, "name", None), "Intel RealSense D435i")
        self.assertEqual(src, "pinned")
        cams = [ln for ln in build_bom(gene)["lines"] if ln["category"] == "camera"]
        self.assertTrue(cams and not any(ln["proposed"] for ln in cams),
                        "the customer said this part is on the robot; we do not overrule them")


@unittest.skipUnless(_MUJOCO, "reading declared instruments needs MuJoCo")
class TheTableIsRightInBothDirections(unittest.TestCase):
    """Quoting an IMU as an ADDITION to a robot whose file declares gyro + accelerometer is the same failure to
    read their model, only cheaper. A ``<sensor>`` IS declared instrumentation (unlike a ``<camera>``, which is
    a render viewpoint), so where the model evidences the instrument the line becomes an EQUIVALENT."""

    def _declares_an_imu(self):
        from virturoid.services.robot_import import import_robot
        from virturoid.services.sensor_provenance import category_declared
        for pkg, model in (("google_barkour_v0", "barkour_v0.xml"), ("booster_t1", "t1.xml"),
                           ("berkeley_humanoid", "berkeley_humanoid.xml")):
            path = os.path.join(_MENAGERIE, pkg, model)
            if not os.path.exists(path):
                continue
            gene = import_robot(path)["gene"]
            if category_declared(gene, "imu")[0]:
                return pkg, gene
        raise unittest.SkipTest("no cached Menagerie package declares a gyro+accelerometer")

    def test_a_declared_imu_is_an_equivalent_not_a_purchase(self):
        from virturoid.services.bom_builder import build_bom
        pkg, gene = self._declares_an_imu()
        imus = [ln for ln in build_bom(gene)["lines"] if ln["category"] == "imu"]
        self.assertTrue(imus, pkg)
        for ln in imus:
            self.assertFalse(ln["proposed"], f"{pkg} declares an IMU; billing one as an addition misreads it")
            self.assertIn("EQUIVALENT", ln["detail"])
        # ...and the camera on the SAME robot is still a proposal, so this is a per-instrument read, not a
        # blanket "it declares something, trust the whole suite".
        cams = [ln for ln in build_bom(gene)["lines"] if ln["category"] == "camera"]
        self.assertTrue(all(ln["proposed"] for ln in cams), "an MJCF <camera> is not evidence of a camera part")

    def test_the_fusion_stack_agrees_with_the_bom_line_for_line(self):
        """The two surfaces that disagreed are the point of this whole file; a fix that makes a NEW pair
        disagree is not a fix. The invoice's `proposed` and the launch file's `proposed` must be one answer."""
        from virturoid.services.bom_builder import build_bom
        from virturoid.services.sensor_fusion_compiler import compile_sensor_fusion
        _pkg, gene = self._declares_an_imu()
        bom_flag = {ln["part"]: ln["proposed"] for ln in build_bom(gene)["lines"]
                    if ln["category"] in _SENSOR_CATS}
        for s in compile_sensor_fusion(gene)["sensors"]:
            if s["part"] in bom_flag:
                self.assertEqual(s["proposed"], bom_flag[s["part"]], f"{s['part']} disagrees across surfaces")


@unittest.skipUnless(_MUJOCO, "reading a declared camera needs MuJoCo")
class ADeclaredCameraIsQuotedNotPricedAsAPart(unittest.TestCase):
    """``ncam > 0`` must not license a part number either. 19 of the 63 Menagerie packages declare a camera and
    most are tracking or cinematic viewpoints (``tracking@trunk`` on the Go1) -- turning one into a $334
    RealSense would be the same invention wearing a better disguise."""

    def test_go1s_tracking_camera_does_not_become_a_purchase(self):
        from virturoid.services.robot_import import import_robot
        from virturoid.services.sensor_provenance import camera_is_ours_to_add, declared_cameras
        gene = import_robot(_menagerie("unitree_go1", "go1.xml"))["gene"]
        inv = (gene.metadata or {}).get("source_sensors") or {}
        if int(inv.get("ncam", 0)) < 1:
            self.skipTest("this go1 revision declares no camera; the quoting rule is exercised by the Go2 case")
        allowed, why = camera_is_ours_to_add(gene)
        self.assertFalse(allowed, "an MJCF <camera> is a render viewpoint, not declared hardware")
        # the refusal QUOTES what their file says rather than converting it into a part
        self.assertIn(str(declared_cameras(gene)[0]["name"]), why)
        self.assertIn("render viewpoint", why)


if __name__ == "__main__":
    unittest.main()
