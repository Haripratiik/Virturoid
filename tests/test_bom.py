"""Bill of materials: every generated robot becomes a real, buildable parts list — each joint matched to a
real actuator by torque, the class's sensor suite (a humanoid's camera eyes, an arm's F/T sensor), materials,
compute and power, with rolled-up totals."""

import unittest

from virturoid.services.anatomy_compiler import build_from_anatomy
from virturoid.services.bom_builder import build_bom, format_bom_markdown
from virturoid.services.component_catalog import select_actuator

QUAD = {"robot_class": "quadruped", "name": "q", "parts": [
    {"name": "torso", "role": "body", "size": 0.5, "girth": 0.16},
    {"name": "neck", "role": "neck", "parent": "torso", "attach": "front_top", "aim": "forward_up",
     "size": 0.1, "girth": 0.05},
    {"name": "head", "role": "head", "parent": "neck", "attach": "tip", "aim": "forward", "size": 0.12, "girth": 0.07},
    {"name": "front_leg", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down",
     "size": 0.28, "girth": 0.05, "segments": 3, "symmetry": "left_right", "joint": "revolute"},
    {"name": "hind_leg", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down",
     "size": 0.3, "girth": 0.055, "segments": 3, "symmetry": "left_right", "joint": "revolute"},
]}


class BomTests(unittest.TestCase):
    def test_select_actuator_meets_torque_and_is_monotonic(self):
        # the chosen motor's PEAK torque clears the demand (with margin), and a bigger demand never gets a
        # weaker motor — the selection is a real engineering sizing, not arbitrary.
        self.assertGreaterEqual(select_actuator(3.0).peak_torque_nm, 3.0)
        self.assertGreaterEqual(select_actuator(80.0).peak_torque_nm,
                                select_actuator(2.0).peak_torque_nm)
        self.assertGreaterEqual(select_actuator(200.0).peak_torque_nm, 100.0)   # reaches the high end

    def test_every_joint_maps_to_a_real_actuator(self):
        g = build_from_anatomy(QUAD)
        bom = build_bom(g)
        joints = {s.name for s in g.actuated_joints()}
        self.assertEqual(set(bom["actuator_map"]), joints)         # one real actuator per actuated joint
        self.assertEqual(bom["dof"], len(joints))
        self.assertTrue(any(ln["category"] == "actuator" for ln in bom["lines"]))
        self.assertTrue(any(ln["category"] == "material" for ln in bom["lines"]))   # link material listed

    def test_humanoid_gets_camera_eyes_imu_compute_power(self):
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        bom = build_bom(build_anthropometric_humanoid())
        cats = {ln["category"] for ln in bom["lines"]}
        cams = [ln for ln in bom["lines"] if ln["category"] == "camera"]
        self.assertTrue(cams and cams[0]["qty"] == 2)              # two camera EYES
        for need in ("imu", "compute", "power"):
            self.assertIn(need, cats)

    def test_arm_camera_always_and_force_torque_is_task_driven(self):
        # An arm always gets a camera; the $4,500 6-axis F/T sensor is TASK-ADAPTIVE — a sorter doesn't need it,
        # a delicate contact/insertion task does. This is the point of task-driven sensor selection.
        from virturoid.services.morphology_composer import compose_robot
        arm = compose_robot("a tabletop robot arm", llm=None)
        sort = {ln["category"] for ln in build_bom(arm, task="sort blocks into colored bins")["lines"]}
        self.assertIn("camera", sort)
        self.assertNotIn("force_torque", sort)
        contact = {ln["category"] for ln in build_bom(arm, task="insert pegs with delicate force control")["lines"]}
        self.assertIn("force_torque", contact)

    def test_sensor_suite_is_task_adaptive(self):
        # Same body, different jobs -> different perception: navigation adds LiDAR, inspection adds thermal.
        g = build_from_anatomy(QUAD)
        nav = {ln["category"] for ln in build_bom(g, task="map and patrol a warehouse")["lines"]}
        insp = {ln["category"] for ln in build_bom(g, task="inspect pipes for heat leaks")["lines"]}
        self.assertIn("lidar", nav)
        self.assertIn("thermal", insp)
        self.assertNotIn("thermal", nav)

    def test_materials_are_per_part_and_task_adaptive(self):
        # intelligent per-part materials: shell body, metal contact parts, a STRONG skeleton chosen by task
        # (steel for heavy work, carbon-fibre for flight) — not one material everywhere.
        from virturoid.services.bom_builder import ensure_materials, refine_materials_for_task
        g = build_from_anatomy(QUAD)
        ensure_materials(g)
        mats = {s.material for s in g.segments}
        self.assertIn("shell", mats)
        self.assertIn("metal", mats)
        heavy = refine_materials_for_task(build_from_anatomy(QUAD), "haul heavy payloads")
        self.assertTrue(any(s.material == "steel" for s in heavy.segments))
        fly = refine_materials_for_task(build_from_anatomy(QUAD), "agile flying-leap robot")
        self.assertTrue(any(s.material == "carbon_fiber" for s in fly.segments))

    def test_bom_lists_each_material_used(self):
        bom = build_bom(build_from_anatomy(QUAD))
        mat_lines = [ln for ln in bom["lines"] if ln["category"] == "material"]
        self.assertGreaterEqual(len(mat_lines), 2)             # at least shell + a structural metal

    def test_sensors_render_on_the_right_segment(self):
        # the head carries a camera; a navigation task adds a LiDAR puck — placed as visual geoms.
        from virturoid.services.sensor_geometry import sensor_geoms_for_gene
        g = build_from_anatomy(QUAD)
        base = sensor_geoms_for_gene(g)
        head = next(s.name for s in g.segments if "head" in s.name.lower())
        self.assertIn(head, base)
        self.assertIn("cam", base[head])                       # camera housing on the head
        nav = sensor_geoms_for_gene(g, task="map and navigate a warehouse")
        self.assertTrue(any("lidar" in xml for xml in nav.values()))

    def test_finalize_for_task_sizes_skeleton_and_material_together(self):
        # The compose-time finalizer makes the DESIGN adapt to the load: heavy -> thicker steel skeleton,
        # flight -> slimmer carbon-fibre. So design, physics and BOM all agree on one robot.
        from virturoid.services.bom_builder import finalize_for_task
        leg = lambda g: next(s for s in g.segments if s.name == "front_leg_l_0")
        base = finalize_for_task(build_from_anatomy(QUAD), "")
        heavy = finalize_for_task(build_from_anatomy(QUAD), "haul very heavy reinforced payloads")
        light = finalize_for_task(build_from_anatomy(QUAD), "lightweight agile racing flyer")
        self.assertGreater(leg(heavy).radius_m, leg(base).radius_m)
        self.assertLess(leg(light).radius_m, leg(base).radius_m)
        self.assertEqual(leg(heavy).material, "steel")
        self.assertEqual(leg(light).material, "carbon_fiber")

    def test_material_choice_changes_the_robots_mass_physics(self):
        # The task-chosen SKELETON material affects PHYSICS, not just colour: a heavy steel frame makes the
        # robot genuinely heavier in sim; a carbon-fibre one lighter. A default (aluminium) robot is unchanged.
        from virturoid.services.bom_builder import finalize_for_task
        tot = lambda g: sum(s.mass_kg for s in g.segments)
        base = tot(finalize_for_task(build_from_anatomy(QUAD), ""))
        heavy = tot(finalize_for_task(build_from_anatomy(QUAD), "haul very heavy reinforced payloads"))
        agile = tot(finalize_for_task(build_from_anatomy(QUAD), "lightweight agile racing flyer"))
        self.assertGreater(heavy, base * 1.3)              # thicker + steel -> much heavier
        self.assertLess(agile, base)                       # slimmer + carbon -> lighter

    def test_compose_robot_returns_a_finalized_gene(self):
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot", llm=None)
        self.assertTrue(all(s.material for s in g.segments))   # every part tagged with a material (finalized)

    def test_actuators_scale_with_task_load(self):
        # a heavy-duty task drives BIGGER motors: the joint torque demand (and the selected actuator) goes up.
        from virturoid.services.bom_builder import finalize_for_task
        from virturoid.services.component_catalog import select_actuator
        leg = lambda g: next(s for s in g.segments if s.name == "front_leg_l_0")
        base = leg(finalize_for_task(build_from_anatomy(QUAD), ""))
        heavy = leg(finalize_for_task(build_from_anatomy(QUAD), "carry super heavy reinforced payloads"))
        self.assertGreater(heavy.actuator_torque_nm, base.actuator_torque_nm * 2)
        self.assertGreaterEqual(select_actuator(heavy.actuator_torque_nm).peak_torque_nm,
                                select_actuator(base.actuator_torque_nm).peak_torque_nm)

    def test_lidar_scales_with_robot_size(self):
        # a bigger robot needs a longer-range LiDAR; the part selection scales with size.
        from virturoid.services.bom_builder import _pick_lidar
        self.assertEqual(_pick_lidar(3.0), "Slamtec RPLIDAR A2M12")     # tiny -> 2D puck
        self.assertEqual(_pick_lidar(40.0), "Ouster OS1-32")           # large -> long-range 3D
        self.assertNotEqual(_pick_lidar(80.0), _pick_lidar(3.0))       # huge != tiny

    def test_totals_and_markdown(self):
        bom = build_bom(build_from_anatomy(QUAD))
        t = bom["totals"]
        for k in ("mass_kg", "price_usd", "est_power_w", "actuators", "line_items"):
            self.assertIn(k, t)
            self.assertGreater(t[k], 0)
        md = format_bom_markdown(bom)
        self.assertIn("Bill of Materials", md)
        self.assertIn("| Part |", md)


if __name__ == "__main__":
    unittest.main()
