import unittest
from unittest import mock

from virturoid.services.design_agent import DESIGN_SCHEMA, propose_arm_design
from virturoid.services.llm_client import ClaudeLLM, LocalLLM, MockLLM, get_llm, unwrap_llm
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.task_builder import build_task_graph


def _req_task():
    req = build_requirements_from_prompt("Build a tabletop arm that sorts red and blue blocks.")
    return req, build_task_graph(req)


class LLMRouterTests(unittest.TestCase):
    def test_default_backend_is_off(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("VIRTUROID_LLM_BACKEND", None)
            self.assertIsNone(get_llm("designer"))

    def test_env_selects_backend(self):
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "mock"}):
            self.assertIsInstance(unwrap_llm(get_llm("designer")), MockLLM)
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "claude", "VIRTUROID_CLAUDE_MODEL": "claude-opus-4-8"}):
            llm = get_llm("designer")
            self.assertIsInstance(unwrap_llm(llm), ClaudeLLM)
            self.assertEqual("claude-opus-4-8", llm.model)
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "local", "VIRTUROID_LOCAL_LLM_URL": "http://x/v1", "VIRTUROID_LOCAL_LLM_MODEL": "nvidia/nemotron"}):
            llm = get_llm("part_recommender")
            self.assertIsInstance(unwrap_llm(llm), LocalLLM)
            self.assertEqual("nvidia/nemotron", llm.model)

    def test_env_selects_openai_backend(self):
        from virturoid.services.llm_client import OpenAILLM

        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "openai"}):
            llm = get_llm("designer")
            self.assertIsInstance(unwrap_llm(llm), OpenAILLM)
            self.assertEqual("gpt-4o-mini", llm.model)
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "chatgpt", "VIRTUROID_OPENAI_MODEL": "gpt-4.1-mini"}):
            self.assertEqual("gpt-4.1-mini", get_llm("failure_analyst").model)

    def test_openai_per_role_split(self):
        # Reasoning roles get the strong model; high-volume roles get the fast model.
        with mock.patch.dict("os.environ", {
            "VIRTUROID_LLM_BACKEND": "openai",
            "VIRTUROID_OPENAI_MODEL": "gpt-4.1",
            "VIRTUROID_OPENAI_FAST_MODEL": "gpt-4o-mini",
        }):
            self.assertEqual("gpt-4.1", get_llm("designer").model)
            self.assertEqual("gpt-4.1", get_llm("task_graph").model)
            self.assertEqual("gpt-4.1", get_llm("transfer").model)
            self.assertEqual("gpt-4o-mini", get_llm("part_recommender").model)
            self.assertEqual("gpt-4o-mini", get_llm("scene").model)
            self.assertEqual("gpt-4o-mini", get_llm("species").model)

    def test_fast_roles_default_to_cheap_model(self):
        # High-volume roles default to the cheap gpt-4.1-mini; body/plan DESIGN (morphology, planner) uses
        # the STRONG model for fidelity (build model -> reasoning model), NOT the cheap fast model.
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "openai", "VIRTUROID_OPENAI_MODEL": "gpt-4o"}):
            self.assertEqual("gpt-4o", get_llm("designer").model)            # reasoning -> main model
            self.assertEqual("gpt-4.1-mini", get_llm("part_recommender").model)
            self.assertEqual("gpt-4o", get_llm("morphology").model)          # design hot-path -> STRONG model
            self.assertEqual("gpt-4o", get_llm("planner").model)

    def test_fast_override_and_build_model(self):
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "openai", "VIRTUROID_OPENAI_MODEL": "gpt-4o",
                                            "VIRTUROID_OPENAI_FAST_MODEL": "gpt-4o-mini",
                                            "VIRTUROID_OPENAI_BUILD_MODEL": "gpt-4.1"}):
            self.assertEqual("gpt-4o-mini", get_llm("part_recommender").model)  # fast override honored
            self.assertEqual("gpt-4.1", get_llm("morphology").model)            # build model wins for design
            self.assertEqual("gpt-4.1", get_llm("planner").model)

    def test_mock_returns_fixed_json(self):
        out = MockLLM(fixed={"a": 1}).complete_json("s", "u", {"type": "object"})
        self.assertEqual({"a": 1}, out)


class LocalEnvFileTests(unittest.TestCase):
    """The project-local .env loader fills missing keys but never overrides real env."""

    def _load(self, tmp, env):
        import os

        from virturoid.services import llm_client

        # clear=True drops the conftest VIRTUROID_NO_LOCAL_ENV flag; cwd patched to
        # tmp so only the temp .env is read (not the real project .env at cwd).
        with mock.patch.object(llm_client, "_REPO_ROOT", tmp), \
             mock.patch.object(llm_client, "_loaded_local_env", False), \
             mock.patch("pathlib.Path.cwd", return_value=tmp), \
             mock.patch.dict("os.environ", env, clear=True):
            llm_client._load_local_env()
            return dict(os.environ)

    def test_loads_values_from_env_file(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / ".env").write_text(
                '# comment\nVIRTUROID_LLM_BACKEND=openai\nOPENAI_API_KEY="sk-abc"\n\nVIRTUROID_OPENAI_MODEL=gpt-4o-mini\n',
                encoding="utf-8",
            )
            result = self._load(tmp, {})
            self.assertEqual("openai", result.get("VIRTUROID_LLM_BACKEND"))
            self.assertEqual("sk-abc", result.get("OPENAI_API_KEY"))  # quotes stripped
            self.assertEqual("gpt-4o-mini", result.get("VIRTUROID_OPENAI_MODEL"))

    def test_real_env_takes_precedence(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / ".env").write_text("VIRTUROID_LLM_BACKEND=openai\n", encoding="utf-8")
            result = self._load(tmp, {"VIRTUROID_LLM_BACKEND": "claude"})
            self.assertEqual("claude", result.get("VIRTUROID_LLM_BACKEND"))


class DesignAgentTests(unittest.TestCase):
    def test_no_backend_returns_none(self):
        req, task = _req_task()
        self.assertIsNone(propose_arm_design("x", req, task, None))

    def test_self_repair_recovers_from_infeasible_proposal(self):
        req, task = _req_task()
        calls = {"n": 0}

        def responder(system, user, schema):
            self.assertEqual(schema, DESIGN_SCHEMA)  # agent passes its schema to the model
            calls["n"] += 1
            if calls["n"] == 1:
                return {"forearm_mm": 205, "wrist_mm": 85, "actuator_torque_nm": 12, "rationale": "too short"}
            return {"forearm_mm": 340, "wrist_mm": 150, "actuator_torque_nm": 14, "rationale": "covers workspace"}

        out = propose_arm_design("sort blocks", req, task, MockLLM(responder=responder))
        self.assertTrue(out["feasible"])
        self.assertEqual(2, out["attempts"])
        self.assertAlmostEqual(340.0, out["design"]["forearm_mm"])

    def test_persistently_infeasible_proposal_is_rejected(self):
        req, task = _req_task()
        # Always too short -> never passes the reach gate.
        bad = MockLLM(fixed={"forearm_mm": 205, "wrist_mm": 85, "actuator_torque_nm": 12})
        out = propose_arm_design("sort blocks", req, task, bad, max_repairs=1)
        self.assertFalse(out["feasible"])
        self.assertIsNone(out["design"])
        self.assertTrue(out["issues"])

    def test_out_of_range_values_are_flagged(self):
        req, task = _req_task()
        bad = MockLLM(fixed={"forearm_mm": 9999, "wrist_mm": 150, "actuator_torque_nm": 14})
        out = propose_arm_design("sort blocks", req, task, bad, max_repairs=0)
        self.assertFalse(out["feasible"])


class PartRecommenderTests(unittest.TestCase):
    def _setup(self):
        from virturoid.fixtures.components import extended_component_library

        req = build_requirements_from_prompt(
            "Build a tabletop arm that sorts blocks with an rgbd camera.", sensor="rgbd_camera"
        )
        return req, extended_component_library()

    def test_no_backend_returns_none(self):
        from virturoid.services.part_agent import recommend_parts

        req, cat = self._setup()
        self.assertIsNone(recommend_parts("x", req, cat, None))

    def test_valid_bom_passes(self):
        from virturoid.services.part_agent import recommend_parts

        req, cat = self._setup()
        good = MockLLM(fixed={
            "joint_actuator": "cmp_actuator_vx12", "wrist_sensor": "cmp_camera_rgbd_mini",
            "gripper": "cmp_gripper_pg2", "main_compute": "cmp_compute_edge1",
        })
        out = recommend_parts("sort blocks", req, cat, good, max_repairs=0)
        self.assertTrue(out["feasible"])
        self.assertEqual("cmp_actuator_vx12", out["selection"]["joint_actuator"])

    def test_wrong_sensor_type_is_rejected(self):
        from virturoid.services.part_agent import recommend_parts

        req, cat = self._setup()
        bad = MockLLM(fixed={
            "joint_actuator": "cmp_actuator_vx12", "wrist_sensor": "cmp_camera_rgb_mini",  # rgb, not rgbd
            "gripper": "cmp_gripper_pg2", "main_compute": "cmp_compute_edge1",
        })
        out = recommend_parts("sort blocks", req, cat, bad, max_repairs=0)
        self.assertFalse(out["feasible"])

    def test_unknown_or_miscategorized_ids_are_rejected(self):
        from virturoid.services.part_agent import recommend_parts

        req, cat = self._setup()
        bad = MockLLM(fixed={
            "joint_actuator": "cmp_camera_rgbd_mini",  # a sensor in the actuator slot
            "wrist_sensor": "does_not_exist",
            "gripper": "cmp_gripper_pg2", "main_compute": "cmp_compute_edge1",
        })
        out = recommend_parts("sort blocks", req, cat, bad, max_repairs=0)
        self.assertFalse(out["feasible"])
        self.assertTrue(out["issues"])


class FailureAnalystTests(unittest.TestCase):
    CLUSTERS = [{"label": "missed_grasp", "count": 4}, {"label": "wrong_bin", "count": 1}]

    def test_no_backend_or_no_clusters_returns_none(self):
        from virturoid.services.failure_agent import analyze_failures

        self.assertIsNone(analyze_failures(self.CLUSTERS, None))
        self.assertIsNone(analyze_failures([], MockLLM(fixed={})))

    def test_valid_diagnosis_passes(self):
        from virturoid.services.failure_agent import analyze_failures

        good = MockLLM(fixed={
            "primary_failure": "missed_grasp",
            "root_cause": "blocks spawn at the reach limit",
            "recommended_action": "co_design_hardware",
            "detail": "lengthen the arm",
        })
        out = analyze_failures(self.CLUSTERS, good, max_repairs=0)
        self.assertTrue(out["valid"])
        self.assertEqual("co_design_hardware", out["diagnosis"]["recommended_action"])

    def test_unobserved_failure_or_bad_action_is_rejected(self):
        from virturoid.services.failure_agent import analyze_failures

        # primary_failure not in the observed clusters
        out = analyze_failures(self.CLUSTERS, MockLLM(fixed={
            "primary_failure": "exploded", "root_cause": "x", "recommended_action": "co_design_hardware",
        }), max_repairs=0)
        self.assertFalse(out["valid"])
        # non-executable action
        out2 = analyze_failures(self.CLUSTERS, MockLLM(fixed={
            "primary_failure": "missed_grasp", "root_cause": "x", "recommended_action": "buy_a_bigger_robot",
        }), max_repairs=0)
        self.assertFalse(out2["valid"])


class LocalLLMOllamaTests(unittest.TestCase):
    """LocalLLM must work against Ollama, whose /v1 rejects json_schema response_format."""

    def _fake_response(self, content):
        class _Resp:
            def raise_for_status(self_inner):
                return None

            def json(self_inner):
                return {"choices": [{"message": {"content": content}}]}

        return _Resp()

    def test_defaults_to_json_object_mode_and_injects_schema(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return self._fake_response('{"forearm_mm": 340}')

        fake_requests = mock.Mock()
        fake_requests.post.side_effect = fake_post
        with mock.patch.dict("sys.modules", {"requests": fake_requests}):
            out = LocalLLM("http://pc:11434/v1", "nemotron-3-nano").complete_json(
                "sys", "user", DESIGN_SCHEMA
            )
        self.assertEqual({"forearm_mm": 340}, out)
        # Ollama-safe: never json_schema by default; uses json_object.
        self.assertEqual({"type": "json_object"}, captured["json"]["response_format"])
        # Schema is injected into the system prompt so the shape survives json_object mode.
        self.assertIn("JSON Schema", captured["json"]["messages"][0]["content"])

    def test_json_schema_mode_for_vllm(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return self._fake_response('{"ok": true}')

        fake_requests = mock.Mock()
        fake_requests.post.side_effect = fake_post
        with mock.patch.dict("sys.modules", {"requests": fake_requests}):
            LocalLLM("http://x/v1", "m", format_mode="json_schema").complete_json("s", "u", {"type": "object"})
        self.assertEqual("json_schema", captured["json"]["response_format"]["type"])

    def test_parses_fenced_and_prose_wrapped_json(self):
        from virturoid.services.llm_client import _parse_json_object

        self.assertEqual({"a": 1}, _parse_json_object('```json\n{"a": 1}\n```'))
        self.assertEqual({"a": 1}, _parse_json_object('Here you go: {"a": 1} hope that helps'))
        self.assertEqual({"a": 1}, _parse_json_object('{"a": 1}'))
        with self.assertRaises(ValueError):
            _parse_json_object("no json at all")

    def test_env_selects_local_format_mode(self):
        with mock.patch.dict("os.environ", {
            "VIRTUROID_LLM_BACKEND": "local",
            "VIRTUROID_LOCAL_LLM_URL": "http://pc:11434/v1",
            "VIRTUROID_LOCAL_LLM_MODEL": "nemotron-3-nano",
            "VIRTUROID_LOCAL_LLM_FORMAT": "json_schema",
        }):
            llm = get_llm("part_recommender")
            self.assertIsInstance(unwrap_llm(llm), LocalLLM)
            self.assertEqual("json_schema", llm.format_mode)


class OpenAILLMTests(unittest.TestCase):
    """OpenAILLM uses json_object mode, injects the schema, and parses defensively."""

    def test_calls_chat_completions_with_json_object(self):
        from virturoid.services.llm_client import OpenAILLM

        captured = {}

        class _Msg:
            content = '{"forearm_mm": 330}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        def create(**kwargs):
            captured.update(kwargs)
            return _Resp()

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = create
        fake_openai = mock.Mock()
        fake_openai.OpenAI.return_value = fake_client
        with mock.patch.dict("sys.modules", {"openai": fake_openai}):
            out = OpenAILLM("gpt-4o-mini").complete_json("sys", "user", DESIGN_SCHEMA)
        self.assertEqual({"forearm_mm": 330}, out)
        self.assertEqual({"type": "json_object"}, captured["response_format"])
        self.assertIn("JSON Schema", captured["messages"][0]["content"])

    def test_reasoning_model_honors_budget_and_effort(self):
        # A reasoning model uses max_completion_tokens (proportional, not a flat 8000) + reasoning_effort.
        from virturoid.services.llm_client import OpenAILLM

        captured = {}

        class _Msg:
            content = '{"x": 1}'

        class _Resp:
            choices = [type("C", (), {"message": _Msg()})()]

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = lambda **kw: (captured.update(kw) or _Resp())
        fake_openai = mock.Mock()
        fake_openai.OpenAI.return_value = fake_client
        with mock.patch.dict("sys.modules", {"openai": fake_openai}):
            OpenAILLM("gpt-5.2").complete_json("s", "u", {"type": "object"}, max_tokens=600,
                                               reasoning_effort="low")
        self.assertEqual(2400, captured["max_completion_tokens"])     # 600*4, not the old 8000 floor
        self.assertEqual("low", captured["reasoning_effort"])
        self.assertNotIn("temperature", captured)                     # reasoning models reject temperature

    def test_daily_rate_limit_falls_back_to_cheaper_model(self):
        # The 50-req/day tier cap on the strong model must NOT crash the build into the keyword path: it
        # retries the SAME prompt on the cheaper high-throughput model, keeping LLM-designed bodies.
        from virturoid.services.llm_client import OpenAILLM

        calls = []

        class _Msg:
            content = '{"ok": true}'

        class _Resp:
            choices = [type("C", (), {"message": _Msg()})()]

        def create(**kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gpt-5.2":
                raise RuntimeError("Error code: 429 - rate limit reached ... requests per day (RPD): Limit 50")
            return _Resp()

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = create
        fake_openai = mock.Mock()
        fake_openai.OpenAI.return_value = fake_client
        with mock.patch.dict("sys.modules", {"openai": fake_openai}):
            out = OpenAILLM("gpt-5.2", fallback_model="gpt-4.1-mini").complete_json("s", "u", {"type": "object"})
        self.assertEqual({"ok": True}, out)
        self.assertEqual(calls, ["gpt-5.2", "gpt-4.1-mini"])           # tried primary once, then fell back

    def test_get_llm_wires_fast_fallback_for_build_roles_only(self):
        with mock.patch.dict("os.environ", {"VIRTUROID_LLM_BACKEND": "openai",
                                            "VIRTUROID_OPENAI_BUILD_MODEL": "gpt-5.2",
                                            "VIRTUROID_OPENAI_FAST_MODEL": "gpt-4.1-mini"}):
            build = get_llm("morphology")
            self.assertEqual("gpt-5.2", build.model)
            self.assertEqual("gpt-4.1-mini", build._fallback_model)        # strong build model -> fast fallback
            self.assertIsNone(get_llm("part_recommender")._fallback_model)  # already fast -> no self-fallback


class TaskAgentTests(unittest.TestCase):
    def _req(self):
        return build_requirements_from_prompt("Sort red and blue blocks into matching bins.")

    _GOOD = {
        "name": "Sort colored blocks", "task_type": "pick_place_sort",
        "objects": ["red_block", "blue_block", "red_bin", "blue_bin"],
        "success_criteria": [{"name": "red_sorted", "expression": "red_block inside red_bin"}],
        "failure_criteria": [
            {"name": "dropped", "expression": "any block outside workspace"},
            {"name": "timeout", "expression": "episode_time > 60"},
        ],
        "metrics": ["success_rate", "drop_rate"], "required_skills": ["reach", "grasp"],
        "safety_constraints": ["no_joint_limit_violation"],
    }

    def test_no_backend_returns_none(self):
        from virturoid.services.task_agent import synthesize_task_graph

        self.assertIsNone(synthesize_task_graph("x", self._req(), None))

    def test_valid_graph_converts_to_validatable_taskgraph(self):
        from virturoid.services.task_agent import synthesize_task_graph, to_task_graph

        out = synthesize_task_graph("sort blocks", self._req(), MockLLM(fixed=self._GOOD), max_repairs=0)
        self.assertTrue(out["valid"])
        graph = to_task_graph(out, self._req())
        self.assertTrue(graph.validate().ok)
        self.assertEqual("pick_place_sort", graph.task_type)

    def test_missing_timeout_is_rejected(self):
        from virturoid.services.task_agent import synthesize_task_graph

        no_timeout = dict(self._GOOD, failure_criteria=[{"name": "dropped", "expression": "block outside workspace"}])
        out = synthesize_task_graph("sort blocks", self._req(), MockLLM(fixed=no_timeout), max_repairs=0)
        self.assertFalse(out["valid"])
        self.assertTrue(any("timeout" in i for i in out["issues"]))

    def test_missing_success_metric_is_rejected(self):
        from virturoid.services.task_agent import synthesize_task_graph

        no_metric = dict(self._GOOD, metrics=["drop_rate"])
        out = synthesize_task_graph("sort blocks", self._req(), MockLLM(fixed=no_metric), max_repairs=0)
        self.assertFalse(out["valid"])

    def test_self_repair_recovers(self):
        from virturoid.services.task_agent import synthesize_task_graph

        calls = {"n": 0}

        def responder(system, user, schema):
            calls["n"] += 1
            if calls["n"] == 1:
                return dict(self._GOOD, metrics=["drop_rate"])  # missing success_rate
            return self._GOOD

        out = synthesize_task_graph("sort blocks", self._req(), MockLLM(responder=responder))
        self.assertTrue(out["valid"])
        self.assertEqual(2, out["attempts"])


class SceneAgentTests(unittest.TestCase):
    def _req(self):
        return build_requirements_from_prompt("Sort blocks into bins.")

    # base 0.36 +/- 0.04 = [0.32, 0.40], inside the [0.30, 0.42] annulus.
    _GOOD = {
        "scene_count": 12, "base_radius_m": 0.36, "position_spread_m": 0.04,
        "mass_range_kg": [0.04, 0.07], "friction_range": [0.8, 1.3],
        "min_separation_m": 0.05, "materials": ["plastic", "wood"],
    }

    def test_no_backend_returns_none(self):
        from virturoid.services.scene_agent import propose_scene_envelope

        self.assertIsNone(propose_scene_envelope("x", self._req(), None))

    def test_feasible_envelope_passes(self):
        from virturoid.services.scene_agent import propose_scene_envelope

        out = propose_scene_envelope("sort", self._req(), MockLLM(fixed=self._GOOD), max_repairs=0)
        self.assertTrue(out["feasible"])
        self.assertEqual(12, out["envelope"]["scene_count"])

    def test_out_of_reach_envelope_is_rejected(self):
        from virturoid.services.scene_agent import propose_scene_envelope

        # base + spread = 0.60 > reach max -> unreachable
        bad = dict(self._GOOD, base_radius_m=0.5, position_spread_m=0.1)
        out = propose_scene_envelope("sort", self._req(), MockLLM(fixed=bad), max_repairs=0)
        self.assertFalse(out["feasible"])
        self.assertTrue(any("reach" in i for i in out["issues"]))

    def test_interpenetration_and_overmass_rejected(self):
        from virturoid.services.scene_agent import propose_scene_envelope

        bad = dict(self._GOOD, min_separation_m=0.01, mass_range_kg=[0.1, 2.0])
        out = propose_scene_envelope("sort", self._req(), MockLLM(fixed=bad), max_repairs=0)
        self.assertFalse(out["feasible"])
        self.assertTrue(any("separation" in i for i in out["issues"]))
        self.assertTrue(any("ceiling" in i for i in out["issues"]))


class RobotParserAgentTests(unittest.TestCase):
    _GOOD = {
        "name": "arm3", "links": ["base", "upper", "forearm", "gripper"],
        "joints": [
            {"name": "j1", "joint_type": "revolute", "parent_link": "base", "child_link": "upper"},
            {"name": "j2", "joint_type": "revolute", "parent_link": "upper", "child_link": "forearm"},
            {"name": "j3", "joint_type": "revolute", "parent_link": "forearm", "child_link": "gripper"},
        ],
        "end_effectors": ["gripper"], "warnings": [],
    }

    def test_no_backend_returns_none(self):
        from virturoid.services.robot_parser_agent import parse_robot_description

        self.assertIsNone(parse_robot_description("<robot/>", "urdf", None))

    def test_valid_tree_converts_to_validatable_genome(self):
        from virturoid.services.robot_parser_agent import parse_robot_description, to_robot_genome

        out = parse_robot_description("<robot>...</robot>", "urdf", MockLLM(fixed=self._GOOD), max_repairs=0)
        self.assertTrue(out["valid"])
        genome = to_robot_genome(out)
        self.assertTrue(genome.validate().ok)

    def test_unknown_link_and_two_roots_rejected(self):
        from virturoid.services.robot_parser_agent import parse_robot_description

        bad = dict(self._GOOD, joints=[
            {"name": "j1", "joint_type": "revolute", "parent_link": "base", "child_link": "upper"},
            {"name": "j2", "joint_type": "revolute", "parent_link": "ghost", "child_link": "forearm"},
        ])
        out = parse_robot_description("x", "urdf", MockLLM(fixed=bad), max_repairs=0)
        self.assertFalse(out["valid"])

    def test_cycle_is_rejected(self):
        from virturoid.services.robot_parser_agent import parse_robot_description

        cyclic = {
            "name": "c", "links": ["a", "b"],
            "joints": [
                {"name": "j1", "joint_type": "revolute", "parent_link": "a", "child_link": "b"},
                {"name": "j2", "joint_type": "revolute", "parent_link": "b", "child_link": "a"},
            ],
            "end_effectors": ["b"],
        }
        out = parse_robot_description("x", "urdf", MockLLM(fixed=cyclic), max_repairs=0)
        self.assertFalse(out["valid"])


class TransferAgentTests(unittest.TestCase):
    def _req(self):
        return build_requirements_from_prompt("Sort blocks into bins.")

    _CANDIDATES = [
        {"key": "manipulator__pick_place_sort", "robot_class": "manipulator", "task_type": "pick_place_sort",
         "converged_design": {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, "success_rate": 1.0},
        {"key": "manipulator__tiny", "robot_class": "manipulator", "task_type": "pick_place_sort",
         "converged_design": {"forearm_mm": 120, "wrist_mm": 60, "actuator_torque_nm": 8}, "success_rate": 0.5},
    ]

    def test_no_backend_returns_none(self):
        from virturoid.services.transfer_agent import recommend_transfer

        self.assertIsNone(recommend_transfer("x", self._req(), self._CANDIDATES, None))

    def test_no_candidates_means_no_reuse(self):
        from virturoid.services.transfer_agent import recommend_transfer

        out = recommend_transfer("x", self._req(), [], MockLLM(fixed={"reuse": True}))
        self.assertTrue(out["valid"])
        self.assertFalse(out["decision"]["reuse"])

    def test_feasible_transfer_accepted(self):
        from virturoid.services.transfer_agent import recommend_transfer

        good = MockLLM(fixed={"reuse": True, "source_key": "manipulator__pick_place_sort",
                              "assets": ["design", "policy"], "rationale": "same task, reaches"})
        out = recommend_transfer("sort", self._req(), self._CANDIDATES, good, max_repairs=0)
        self.assertTrue(out["valid"])
        self.assertEqual("manipulator__pick_place_sort", out["decision"]["source_key"])

    def test_unreachable_candidate_rejected(self):
        from virturoid.services.transfer_agent import recommend_transfer

        bad = MockLLM(fixed={"reuse": True, "source_key": "manipulator__tiny", "assets": ["design"]})
        out = recommend_transfer("sort", self._req(), self._CANDIDATES, bad, max_repairs=0)
        self.assertFalse(out["valid"])
        self.assertTrue(any("cover" in i for i in out["issues"]))

    def test_hallucinated_key_rejected(self):
        from virturoid.services.transfer_agent import recommend_transfer

        bad = MockLLM(fixed={"reuse": True, "source_key": "does_not_exist", "assets": ["design"]})
        out = recommend_transfer("sort", self._req(), self._CANDIDATES, bad, max_repairs=0)
        self.assertFalse(out["valid"])


class FlywheelDatasetTests(unittest.TestCase):
    def test_append_design_record_writes_jsonl_row(self):
        import json
        import tempfile
        from pathlib import Path

        from virturoid.services.memory_store import append_design_record

        with tempfile.TemporaryDirectory() as tmp:
            append_design_record(
                "sort blocks", "manipulator", "pick_place_sort",
                {"forearm_mm": 330, "wrist_mm": 150, "actuator_torque_nm": 14},
                0.91, "physics_codesign", memory_dir=Path(tmp),
            )
            append_design_record(
                "sort blocks", "manipulator", "pick_place_sort",
                {"forearm_mm": 340, "wrist_mm": 150, "actuator_torque_nm": 12},
                1.0, "ai_designer+codesign", memory_dir=Path(tmp),
            )
            rows = [json.loads(line) for line in (Path(tmp) / "design_dataset.jsonl").read_text().splitlines() if line.strip()]
            self.assertEqual(2, len(rows))
            self.assertEqual("ai_designer+codesign", rows[1]["source"])
            self.assertEqual(1.0, rows[1]["success_rate"])
            self.assertIn("design", rows[0])


class GatedBuildHelperTests(unittest.TestCase):
    """The autonomous-build agent hooks must be no-ops without a backend and advisory with one."""

    def test_helpers_are_noops_without_backend(self):
        from virturoid.services.autonomous_build import _maybe_analyze_failures, _maybe_recommend_parts

        import os

        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("VIRTUROID_LLM_BACKEND", None)
            self.assertIsNone(_maybe_recommend_parts("sort blocks", lambda *a, **k: None))

            class C:
                failure_label = "missed_grasp"
                count = 3

            self.assertIsNone(_maybe_analyze_failures([C()], lambda *a, **k: None))

    def test_failure_analyst_helper_reads_clusters(self):
        from virturoid.services.autonomous_build import _maybe_analyze_failures
        from virturoid.services import llm_client

        class C:
            def __init__(self, label, count):
                self.failure_label = label
                self.count = count

        diagnosis = {
            "primary_failure": "missed_grasp", "root_cause": "blocks at reach limit",
            "recommended_action": "co_design_hardware", "detail": "lengthen arm",
        }
        with mock.patch.object(llm_client, "get_llm", return_value=MockLLM(fixed=diagnosis)):
            out = _maybe_analyze_failures([C("missed_grasp", 4)], lambda *a, **k: None)
        self.assertIsNotNone(out)
        self.assertTrue(out["valid"])
        self.assertEqual("co_design_hardware", out["diagnosis"]["recommended_action"])


if __name__ == "__main__":
    unittest.main()
