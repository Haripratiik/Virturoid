"""W2/I4/T-E: the input-ingestion + training-improvement capabilities are callable by the agent.

The agent is the brain (Claude/Codex via MCP), so these must dispatch through the shared registry's call_tool
and appear in tool_specs. Offline; MuJoCo only for import_robot_model. Uses call_tool exactly as the agent does.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

from virturoid.services.agent_tools import call_tool, tool_specs  # noqa: E402

_NEW_TOOLS = {"interpret_prompt", "inspect_project_bundle", "import_robot_model",
              "plan_training", "check_perception_leakage", "amplify_demonstrations", "data_dividends"}

#: real vendor models, not fixtures. Fixtures have lied here before: a hand-written URDF has tidy primitive
#: links whose volume-derived mass happens to land near its stated mass, so the double-count this file guards
#: against does not show up. Only a real Go2/Panda/G1/UR5e exposes it.
_MENAGERIE = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
_REAL_ROBOTS = {"go2": ("unitree_go2", "go2.xml"), "panda": ("franka_emika_panda", "panda.xml"),
                "g1": ("unitree_g1", "g1.xml"), "ur5e": ("universal_robots_ur5e", "ur5e.xml")}
_HAVE_MENAGERIE = all((_MENAGERIE / pkg / xml).is_file() for pkg, xml in _REAL_ROBOTS.values())


def _source_mass_kg(model_path: Path) -> float:
    """The manufacturer's own total: every body mass in the customer's model, straight from MuJoCo."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(model_path))
    return float(sum(m.body_mass[i] for i in range(1, m.nbody)))


def _ok(name, args=None):
    r = call_tool(name, args or {})
    assert r.get("ok"), f"{name} failed: {r.get('error')}"
    return r["result"]


class ToolRegistrationTests(unittest.TestCase):
    def test_all_new_tools_are_discoverable(self):
        names = {t["name"] for t in tool_specs()}
        self.assertTrue(_NEW_TOOLS <= names, _NEW_TOOLS - names)

    def test_interpret_prompt_tool(self):
        out = _ok("interpret_prompt", {"prompt": "a 1.2 m humanoid that carries 3 kg"})
        by_field = {e["field_path"]: e["source_type"] for e in out["evidence"]}
        self.assertEqual(by_field.get("payload_kg"), "parsed")

    def test_plan_training_tool(self):
        out = _ok("plan_training", {"task": "sort blocks by color into two bins"})
        self.assertEqual(out["task_family"], "sort")
        self.assertIn("demonstration_amplifier", out["teacher_sources"])
        self.assertEqual(out["trainer_ladder"][0], "reuse_evaluate")

    def test_check_perception_leakage_blocks_privileged_policy(self):
        out = _ok("check_perception_leakage", {
            "policy_observation_keys": ["rgb", "object_pose"],   # object_pose = privileged truth in the policy
            "required_modalities": ["rgb"], "deploy_modalities": ["rgb"]})
        self.assertFalse(out["leakage"]["ok"])
        codes = {e["code"] for e in out["leakage"]["errors"]}
        self.assertIn("privileged_state_in_policy", codes)

    def test_check_perception_leakage_passes_clean_contract(self):
        out = _ok("check_perception_leakage", {
            "policy_observation_keys": ["rgb", "joint_state"],
            "required_modalities": ["rgb"], "deploy_modalities": ["rgb"], "randomization_logged": True})
        self.assertTrue(out["leakage"]["ok"])

    def test_inspect_project_bundle_tool(self):
        d = tempfile.mkdtemp(prefix="proj_")
        Path(d, "robot.urdf").write_text("<robot name='x'/>", encoding="utf-8")
        Path(d, "controllers.yaml").write_text("k: 1", encoding="utf-8")
        Path(d, "notes.md").write_text("# hi", encoding="utf-8")
        out = _ok("inspect_project_bundle", {"path": d})
        self.assertEqual(out["total_files"], 3)
        self.assertTrue(out["robot_models"])                    # the urdf is the first runnable sim target
        self.assertEqual(out["first_runnable_sim_target"], "robot.urdf")

    def test_data_dividends_tool(self):
        mem = tempfile.mkdtemp(prefix="mem_")
        out = _ok("data_dividends", {"memory_dir": mem})
        self.assertEqual(out["total_dividends"], 0)             # empty ledger reads as honest-empty

    def test_bad_args_return_structured_error(self):
        r = call_tool("interpret_prompt", {})                   # missing prompt
        self.assertTrue(r.get("ok"))                            # call_tool succeeds; handler reports the error
        self.assertIn("error", r["result"])


@unittest.skipUnless(_MUJOCO, "needs MuJoCo for the physics-grounded amplifier")
class FlywheelClosureTests(unittest.TestCase):
    def test_amplify_banks_a_data_dividend(self):
        # one call closes the flywheel: amplify demos -> auto-bank the demonstration_dataset dividend.
        mem = tempfile.mkdtemp(prefix="moat_")
        out = _ok("amplify_demonstrations",
                  {"prompt": "a quadruped robot dog", "n_variants": 3, "bank_dividend": True, "memory_dir": mem})
        self.assertGreater(out["kept"], 0)
        self.assertIn("data_dividend", out)
        self.assertTrue(out["data_dividend"]["reusable_by_default"])
        ledger = _ok("data_dividends", {"memory_dir": mem})
        self.assertEqual(ledger["total_dividends"], 1)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo to compile the import lanes")
class ImportModelToolTests(unittest.TestCase):
    def test_import_robot_model_tool(self):
        urdf = """<?xml version="1.0"?>
        <robot name="arm2">
          <link name="base_link"><inertial><mass value="1.0"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
          <link name="upper"><inertial><mass value="0.6"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision><geometry><box size="0.04 0.04 0.3"/></geometry></collision></link>
          <joint name="pan" type="revolute"><parent link="base_link"/><child link="upper"/>
            <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/></joint>
        </robot>"""
        d = tempfile.mkdtemp(prefix="import_")
        p = os.path.join(d, "arm2.urdf")
        Path(p).write_text(urdf, encoding="utf-8")
        out = _ok("import_robot_model", {"path": p, "robot_id": "arm2"})
        self.assertEqual(out["source_format"], "urdf")
        self.assertTrue(out["faithful_lane"]["ok"])
        self.assertIn("training_readiness_score", out["confidence"])


class MassProvenanceUnitTests(unittest.TestCase):
    """B1: `metadata['mass_source']` must describe the masses the gene is ACTUALLY carrying.

    It is not a label — `gene_build.grounding_config` reads it and sets `preserve_mass=True` for
    "source_model", so every later re-ground refuses to touch whatever masses it finds. A wrong value is
    therefore self-sealing: it freezes our estimate in place as the manufacturer's own measurement.
    """

    @staticmethod
    def _gene(masses, mass_source="source_model"):
        segs = [SimpleNamespace(name=n, mass_kg=m, radius_m=0.05, length_m=0.2) for n, m in masses.items()]
        return SimpleNamespace(segments=segs, metadata={"mass_source": mass_source,
                                                        "imported_from": "mjcf_or_urdf"})

    def test_preserved_masses_keep_the_source_model_claim(self):
        from virturoid.services.input_training_tools import _link_shape, _reconcile_mass_provenance
        g = self._gene({"base": 6.921, "thigh": 1.152})
        before = _link_shape(g)
        res = {"warnings": [], "materials_applied": [], "skipped_ops": []}
        prov = _reconcile_mass_provenance(g, before, res)
        self.assertTrue(prov["preserved"])
        self.assertEqual(g.metadata["mass_source"], "source_model")
        self.assertEqual(res["warnings"], [])

    def test_replaced_masses_lose_the_source_model_claim_and_warn(self):
        from virturoid.services.input_training_tools import _link_shape, _reconcile_mass_provenance
        g = self._gene({"base": 6.921, "thigh": 1.152})
        before = _link_shape(g)
        g.segments[0].mass_kg = 12.4                       # a re-ground replaced the manufacturer's number
        res = {"warnings": [], "materials_applied": [], "skipped_ops": []}
        prov = _reconcile_mass_provenance(g, before, res)
        self.assertFalse(prov["preserved"])
        self.assertEqual(prov["corrected_to"], "virturoid_estimate")
        self.assertEqual(g.metadata["mass_source"], "virturoid_estimate")
        self.assertEqual(g.metadata["mass_source_replaced"]["was"], "source_model")
        self.assertTrue(any("REPLACED" in w for w in res["warnings"]))

    def test_a_material_the_customer_never_asked_for_is_flagged_not_implied(self):
        from virturoid.services.input_training_tools import _link_shape, _reconcile_mass_provenance
        g = self._gene({"base": 6.921})
        before = _link_shape(g)
        g.metadata["grounding"] = {"material": "aluminum", "fill": 0.3}
        res = {"warnings": [], "materials_applied": [],
               "skipped_ops": [{"reason": "this robot has no 'torso' part"}]}
        prov = _reconcile_mass_provenance(g, before, res)
        self.assertEqual(prov["material_masses_derived_at"], "aluminum")
        self.assertFalse(prov["material_requested_by_customer"])
        self.assertTrue(any("NOT a material you specified" in w for w in res["warnings"]))

    def test_a_material_the_customer_did_ask_for_is_not_flagged(self):
        from virturoid.services.input_training_tools import _link_shape, _reconcile_mass_provenance
        g = self._gene({"base": 6.921})
        before = _link_shape(g)
        g.metadata["grounding"] = {"material": "abs_plastic", "fill": 0.25}
        res = {"warnings": [], "materials_applied": [{"group": "all", "material": "abs_plastic"}],
               "skipped_ops": []}
        prov = _reconcile_mass_provenance(g, before, res)
        self.assertTrue(prov["material_requested_by_customer"])
        self.assertEqual(res["warnings"], [])


class ReadmeFoldSanitizerTests(unittest.TestCase):
    def test_a_citation_url_is_not_a_material_request(self):
        # the real Menagerie G1 README cites `url={https://arxiv.org/abs/2502.08844}`; the `/abs/` in it parsed
        # as ABS PLASTIC and silently re-derived all 30 link masses of an imported humanoid.
        from virturoid.services.input_training_tools import _strip_urls
        from virturoid.services.nlp_properties import extract_properties
        line = "  url={https://arxiv.org/abs/2502.08844},"
        self.assertTrue(extract_properties(line).ops, "precondition: the raw citation does parse as a material")
        self.assertEqual(extract_properties(_strip_urls(line)).ops, [])

    def test_a_real_spec_sentence_survives_sanitising(self):
        from virturoid.services.input_training_tools import _strip_urls
        from virturoid.services.nlp_properties import extract_properties
        txt = _strip_urls("Aluminum body, carbon-fiber legs, 5 kg payload. See https://example.com/datasheet")
        self.assertIn("Aluminum body", txt)
        self.assertTrue([o for o in extract_properties(txt).ops if o["op"] == "set_material"])


@unittest.skipUnless(_MUJOCO, "needs MuJoCo to compile a real vendor model")
@unittest.skipUnless(_HAVE_MENAGERIE, "needs the MuJoCo Menagerie cache (real robots, not fixtures)")
class RealRobotMassPreservationTests(unittest.TestCase):
    """B1, measured end-to-end through the agent's own entry point on four REAL vendor robots.

    Before the fix, ingest called `ground_gene(gene)` bare — defaults material="aluminum", fill=0.3,
    preserve_mass=False — so the manufacturer's per-link masses were discarded and re-derived as primitive
    volume + one of OUR catalog motors, double-counting motors the real robot already contains:

        Go2 15.206 -> 29.031 kg | Panda 17.452 -> 50.425 | G1 33.341 -> 42.339 | UR5e 20.995 -> 25.772

    and `mass_source` still read "source_model", which `grounding_config` turns into preserve_mass=True —
    locking our estimate in as the customer's own number on every door downstream.
    """

    def test_ingest_preserves_every_manufacturer_mass(self):
        env = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = tempfile.mkdtemp(prefix="b1_sess_")
        try:
            from virturoid.services import session_state as S
            for name, (pkg, xml) in _REAL_ROBOTS.items():
                with self.subTest(robot=name):
                    src = _source_mass_kg(_MENAGERIE / pkg / xml)
                    out = _ok("ingest_project", {"path": str(_MENAGERIE / pkg)})
                    gene = S.get_robot(out["robot_id"])
                    self.assertIsNotNone(gene, f"{name}: ingest held no robot")
                    held = sum(float(s.mass_kg or 0.0) for s in gene.segments)
                    self.assertAlmostEqual(held, src, places=2,
                                           msg=f"{name}: {src:.3f} kg of manufacturer mass became {held:.3f}")
                    # the claim and the number agree, and no material was invented to explain them
                    self.assertEqual(gene.metadata.get("mass_source"), "source_model")
                    self.assertIsNone(gene.metadata.get("grounding"),
                                      f"{name}: stamped a material the customer never chose")
                    self.assertTrue(out["mass_provenance"]["preserved"])
        finally:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
            if env is not None:
                os.environ["VIRTUROID_SESSIONS_DIR"] = env

    def test_a_skipped_material_request_does_not_repaint_the_whole_robot(self):
        # the Go2 has no 'torso' part, so "aluminium body" is SKIPPED — and must not be applied globally anyway.
        env = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = tempfile.mkdtemp(prefix="b1_mat_")
        try:
            from virturoid.services import session_state as S
            pkg, xml = _REAL_ROBOTS["go2"]
            src = _source_mass_kg(_MENAGERIE / pkg / xml)
            out = _ok("ingest_project", {"path": str(_MENAGERIE / pkg), "description": "aluminum body"})
            self.assertTrue(out["skipped_ops"], "precondition: the aluminium request is skipped on a Go2")
            gene = S.get_robot(out["robot_id"])
            self.assertAlmostEqual(sum(float(s.mass_kg or 0.0) for s in gene.segments), src, places=2)
            self.assertIsNone(gene.metadata.get("grounding"))
        finally:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
            if env is not None:
                os.environ["VIRTUROID_SESSIONS_DIR"] = env


if __name__ == "__main__":
    unittest.main()
