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

    def test_lidar_range_follows_the_job_not_the_robots_mass(self):
        # A LiDAR's price is its RANGE, and range is a task requirement. Sizing it by mass alone put an $8,000
        # Ouster OS1-32 on a 60 kg humanoid and an $18,000 OS2-128 on a 108 kg one, when every real machine of
        # that class ships a ~$749 Livox Mid-360. So: long range is bought when the JOB is long range.
        from virturoid.services.bom_builder import _pick_lidar
        self.assertEqual(_pick_lidar(3.0, "patrol a room"), "Slamtec RPLIDAR A2M12")   # tiny indoor -> 2D puck
        self.assertEqual(_pick_lidar(40.0, "patrol a warehouse"), "Livox Mid-360")     # big INDOOR -> class 3D
        self.assertEqual(_pick_lidar(40.0, "survey an outdoor site"), "Ouster OS1-32")  # the job needs range
        self.assertEqual(_pick_lidar(80.0, "drive an outdoor haul road"), "Ouster OS2-128")   # big + far

    def test_power_source_is_decided_by_the_robot_kind(self):
        # A LEGGED robot walks away from the socket: a 4.2 kg enclosed wall PSU is not a candidate part for it at
        # any wattage. A fixed-base arm legitimately runs off the wall. The prompt can override either default.
        quad = build_from_anatomy(QUAD)
        walking = [ln for ln in build_bom(quad, task="patrol a warehouse")["lines"] if ln["category"] == "power"]
        self.assertTrue(walking, "a power source is always specified")
        self.assertTrue(all("LiPo" in ln["part"] or "Li-ion" in ln["part"] for ln in walking),
                        f"a walking robot must carry a battery, got {[ln['part'] for ln in walking]}")
        self.assertTrue(all("runtime" in ln["detail"] for ln in walking),
                        "a battery must state the runtime it was sized for")
        # ... unless the customer explicitly asks for a tethered bench rig
        tethered = [ln for ln in build_bom(quad, task="step in place on a lab bench")["lines"]
                    if ln["category"] == "power"]
        self.assertTrue(all("Mean Well" in ln["part"] for ln in tethered),
                        f"an explicit bench/tethered prompt pins wall power, got {[ln['part'] for ln in tethered]}")
        from virturoid.services.morphology_composer import compose_robot
        arm = compose_robot("a tabletop robot arm", llm=None)
        fixed = [ln for ln in build_bom(arm, task="sort blocks into bins")["lines"] if ln["category"] == "power"]
        self.assertTrue(all("Mean Well" in ln["part"] for ln in fixed),
                        f"a fixed-base arm runs off a socketed supply, got {[ln['part'] for ln in fixed]}")

    def test_one_power_number_everywhere(self):
        # The package used to carry THREE different power figures: totals said 800.6 W, the PSU line said
        # "1500 W >= 1133 W draw" (1133 being a different draw x the 1.4 headroom, labelled as the draw), and the
        # part was 1500 W. There is now one budget; the totals, the budget block and the power line all quote it.
        for task in ("patrol a warehouse", "step in place on a lab bench"):
            bom = build_bom(build_from_anatomy(QUAD), task=task)
            draw = bom["power_budget"]["total_draw_w"]
            self.assertEqual(bom["totals"]["est_power_w"], draw)
            self.assertAlmostEqual(draw, bom["power_budget"]["actuators_w"]
                                   + bom["power_budget"]["electronics_w"], delta=0.2)   # each rounded to 0.1 W
            power = [ln for ln in bom["lines"] if ln["category"] == "power"]
            self.assertTrue(power)
            for ln in power:
                self.assertIn(f"{draw:.0f} W budgeted draw", ln["detail"],
                              f"the power line must quote the ONE budget, got: {ln['detail']}")

    def test_actuator_draw_tracks_the_load_not_the_motor_bought(self):
        # The old bus term was `rated_torque x max_speed x 0.3` over the SELECTED motors -- a corner power no
        # motor reaches, and a figure that DOUBLED when a joint was handed a motor two rungs too big. It reported
        # 1409 W for a 14 kg quadruped whose real-world equivalent draws ~150-250 W walking.
        bom = build_bom(build_from_anatomy(QUAD), task="patrol a warehouse")
        pb = bom["power_budget"]
        self.assertGreater(pb["actuators_w"], 0.0)
        self.assertEqual(pb["driven_axes"], bom["dof"])
        self.assertLess(pb["total_draw_w"], 600.0, "a small quadruped does not draw a kilowatt to walk")

    def test_customer_material_wins_and_a_refusal_says_why(self):
        # "…carries a 5 kg payload, aluminium frame" used to ship 12 Steel 4140 links, because the task heuristic
        # matched "payload" and nothing read the word "aluminium".
        asked = build_from_anatomy(QUAD)
        bom = build_bom(asked, task="haul a heavy payload, aluminium frame")
        self.assertEqual(bom["material_policy"]["requested"], "aluminum")
        self.assertTrue(bom["material_policy"]["honoured"])
        mats = {ln["part"] for ln in bom["lines"] if ln["category"] == "material"}
        self.assertIn("Aluminum 6061-T6", mats)
        self.assertNotIn("Steel 4140", mats)
        # and a material the load path cannot use is SUBSTITUTED WITH THE REASON, never silently
        pla = build_bom(build_from_anatomy(QUAD), task="a 3d-printed PLA frame that hauls heavy crates")
        self.assertFalse(pla["material_policy"]["honoured"])
        self.assertEqual(pla["material_policy"]["requested"], "pla")
        self.assertIn("PLA", pla["material_policy"]["reason"])
        self.assertIn("Nm", pla["material_policy"]["reason"])       # the reason carries the load it must carry

    def test_actuator_skus_are_standardised_under_a_stated_policy(self):
        # A generated 16-joint humanoid shipped SIX motor SKUs from four vendors, chosen with no policy at all.
        g = build_from_anatomy(QUAD)
        bom = build_bom(g, task="patrol a warehouse")
        pol = bom["actuator_policy"]
        self.assertIn("rule", pol)
        self.assertLessEqual(pol["skus"], pol["sized_skus"])         # standardising never ADDS part numbers
        skus = {ln["part"] for ln in bom["lines"] if ln["category"] == "actuator"}
        self.assertEqual(len(skus), pol["skus"])
        # every joint still gets a motor that meets its requirement (roll-ups only ever go up)
        from virturoid.services.component_catalog import resolve_part
        for s in g.actuated_joints():
            part = resolve_part(bom["actuator_map"][s.name])
            req = getattr(s, "torque_req_nm", None) or s.actuator_torque_nm or 6.0
            self.assertGreaterEqual(part.peak_torque_nm, float(req))
        # a SKU kept beyond the target count must say WHY it could not be folded
        for kept in pol.get("kept_separate", []):
            self.assertTrue(kept["reason"])

    def test_cost_drivers_are_reported(self):
        bom = build_bom(build_from_anatomy(QUAD), task="patrol a warehouse")
        cd = bom["cost_drivers"]
        self.assertAlmostEqual(sum(cd["by_category"].values()), bom["totals"]["price_usd"], places=1)
        self.assertTrue(cd["top_lines"] and cd["basis"])

    def test_compute_scales_with_load_not_just_class(self):
        # The brain is sized by COMPUTE LOAD (DOF + vision/SLAM + whole-body), not the robot class alone.
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        tier = {"Raspberry Pi 5 (8GB)": 0, "NVIDIA Jetson Orin Nano 8GB": 1, "NVIDIA Jetson AGX Orin 64GB": 2}
        arm = compose_robot("a tabletop robot arm", llm=None)
        plain = [ln for ln in build_bom(arm, task="hold a fixed pose")["lines"] if ln["category"] == "compute"][0]
        visiony = [ln for ln in build_bom(arm, task="use a camera to detect and sort blocks")["lines"]
                   if ln["category"] == "compute"][0]
        self.assertLessEqual(tier[plain["part"]], tier[visiony["part"]])      # vision needs >= the no-vision board
        hum = [ln for ln in build_bom(build_anthropometric_humanoid(), task="walk and balance")["lines"]
               if ln["category"] == "compute"][0]
        self.assertEqual(hum["part"], "NVIDIA Jetson AGX Orin 64GB")          # whole-body -> top board

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
