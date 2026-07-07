"""W2/I4/T-E: the input-ingestion + training-improvement capabilities are callable by the agent.

The agent is the brain (Claude/Codex via MCP), so these must dispatch through the shared registry's call_tool
and appear in tool_specs. Offline; MuJoCo only for import_robot_model. Uses call_tool exactly as the agent does.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

from virturoid.services.agent_tools import call_tool, tool_specs  # noqa: E402

_NEW_TOOLS = {"interpret_prompt", "inspect_project_bundle", "import_robot_model",
              "plan_training", "check_perception_leakage", "amplify_demonstrations", "data_dividends"}


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


if __name__ == "__main__":
    unittest.main()
