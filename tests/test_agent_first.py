"""Agent-first tools (docs/agent_first_plan.md P1: G-A + G-C) — the tools that let an EXTERNAL agent be the
whole brain against our substrate with ZERO internal LLM spend: it AUTHORS the anatomy graph itself
(submit_design) instead of prompting our generator, then drives the full loop on THE held gene
(evaluate_held / train_held / export_held) instead of recomposing. The headline test walks the entire
6-step canonical loop through ``call_tool`` JSON ONLY (no internal imports) — the proof an MCP agent can do
it. Offline (no local env) so it is deterministic AND provably LLM-free (get_llm returns None).
"""
import importlib.util
import os
import time
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class DesignSubmissionTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_schema_examples_actually_compile(self):
        # the language we TEACH the agent must be real: its worked examples must submit + hold successfully.
        sch = self._call("get_design_schema")
        for key in ("quadruped", "hexapod"):
            r = self._call("submit_design", {"graph": sch["examples"][key]})
            self.assertTrue(r.get("ok"), f"{key} example must compile: {r.get('error')}")
            self.assertIn("robot_id", r)

    def test_agent_authored_design_holds_and_discovers(self):
        # the agent submits ITS OWN graph (not a prompt); GEN-1 reads the leg count from the compiled body.
        sch = self._call("get_design_schema")
        r = self._call("submit_design", {"graph": sch["examples"]["hexapod"]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["appendages"]["legs"], 6)

    def test_broken_graph_teaches(self):
        r = self._call("submit_design", {"graph": {"robot_class": "quadruped", "parts": [{"name": "x", "role": "leg"}]}})
        self.assertFalse(r["ok"])
        self.assertIn("body", r["error"])                     # tells the agent it needs a body root

    def test_schema_has_numeric_grounding(self):
        # T8: the schema teaches realistic dimension bands so the agent authors credible sizes.
        sch = self._call("get_design_schema")
        self.assertIn("typical_dimensions_m", sch)
        self.assertIn("wheel", sch["typical_dimensions_m"])
        self.assertIn("proportion_rules", sch)

    def test_disproportionate_design_is_flagged_not_silently_held(self):
        # T8: a wheel larger than the chassis (the exact 'box with oversized wheels' failure) warns the agent.
        bad = {"robot_class": "mobile_base", "name": "bad", "parts": [
            {"name": "c", "role": "body", "size": 0.4, "girth": 0.05, "aspect": "wide"},
            {"name": "w", "role": "wheel", "parent": "c", "attach": "front_bottom", "size": 0.3, "girth": 0.08,
             "symmetry": "left_right"}]}
        r = self._call("submit_design", {"graph": bad})
        self.assertTrue(r["ok"])                                # non-blocking — held, but flagged
        self.assertTrue(any("dwarf" in w for w in r.get("proportion_warnings", [])))
        # the taught examples are proportionate -> no warnings
        for ex in ("quadruped", "rover"):
            g = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"][ex]})
            self.assertFalse(g.get("proportion_warnings"), f"{ex} example should be proportionate")

    def test_absurd_scale_is_gated(self):
        # M16: a 30 m leg used to hold a 19.7 m / 130 kg "robot" silently -> now a teaching error, before compile.
        g = {"robot_class": "quadruped", "name": "absurd", "parts": [
            {"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
            {"name": "leg1", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out",
             "size": 30.0, "girth": 0.02, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]}
        r = self._call("submit_design", {"graph": g})
        self.assertFalse(r["ok"])
        self.assertIn("band", r["error"])                     # names the buildable band, teaches the fix
        # a sane design is NOT a false positive (the taught examples still hold)
        self.assertTrue(self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["ok"])

    def test_submit_scene_spec(self):
        objs = [{"name": "floor", "object_type": "floor", "category": "floor", "size_xyz": [4, 4, 0.05],
                 "pose_xyz_rpy": [0, 0, -0.025, 0, 0, 0]},
                {"name": "box", "category": "obstacle", "size_xyz": [0.4, 0.4, 0.4], "pose_xyz_rpy": [1, 0, 0.2, 0, 0, 0]}]
        r = self._call("submit_scene_spec", {"objects": objs, "task": "navigation"})
        self.assertTrue(r["ok"])
        self.assertIn("scene_id", r)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class HeldChainTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_evaluate_and_export_the_held_gene(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        ev = self._call("evaluate_held", {"robot_id": rid})
        self.assertEqual(ev["task"], "locomotion")
        ex = self._call("export_held", {"robot_id": rid, "formats": ["mjcf"]})
        self.assertTrue(ex["ok"])
        self.assertTrue(os.path.exists(ex["artifacts"]["mjcf"]), "MJCF must be a real file on disk")

    def test_export_the_full_buildable_bundle(self):
        # B3: urdf/ros2/bom/spec are real files, not just sim (the buildable-robot story).
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        ex = self._call("export_held", {"robot_id": rid, "formats": ["mjcf", "urdf", "ros2", "bom", "spec"], "task": "walk"})
        self.assertTrue(ex["ok"])
        for fmt in ("mjcf", "urdf", "bom", "spec"):
            self.assertIn(fmt, ex["artifacts"], f"{fmt} must be produced")
            self.assertTrue(os.path.exists(ex["artifacts"][fmt]), f"{fmt} must be a real file: {ex['artifacts'].get(fmt)}")
        self.assertTrue(os.path.isdir(ex["artifacts"]["ros2"]), "ros2 must be an installable package dir")

    def test_export_rejects_unknown_format(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        ex = self._call("export_held", {"robot_id": rid, "formats": ["stl_but_typo"]})
        self.assertFalse(ex["ok"])
        self.assertIn("unknown format", ex["error"])

    def test_train_held_job_completes_with_verdict(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        jid = self._call("train_held", {"robot_id": rid, "mode": "gait_search", "max_evals": 3})["job_id"]
        self.assertIsNotNone(jid)
        for _ in range(60):
            j = self._call("get_job", {"job_id": jid})
            if j["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(2)
        self.assertEqual(j["status"], "succeeded")
        self.assertEqual(j["result"]["mode"], "gait_search")
        self.assertIn("best", j["result"])


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class FullAgentLoopTests(unittest.TestCase):
    """The proof: an external agent completes the 6-step canonical loop through call_tool JSON ONLY, with the
    internal LLM OFF (no local env -> get_llm None). This is what a Claude Code / Codex MCP session does."""

    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def test_design_to_export_no_internal_llm(self):
        from virturoid.services.agent_tools import call_tool

        def T(name, args):
            r = call_tool(name, args)
            self.assertTrue(r["ok"], f"{name} dispatch failed: {r.get('error')}")
            return r["result"]

        # confirm the internal LLM is OFF, so this proves zero-our-tokens autonomy
        from virturoid.services.llm_client import get_llm
        self.assertIsNone(get_llm("morphology"), "test must run with the internal LLM off")

        schema = T("get_design_schema", {})                                    # 1. learn the language
        design = T("submit_design", {"graph": schema["examples"]["quadruped"]})  # 2. AUTHOR (design)
        self.assertTrue(design["ok"]); rid = design["robot_id"]
        T("simulate_gait", {"robot_id": rid, "steps": 400})                    # 3. simulate (compile+sim)
        ev = T("evaluate_held", {"robot_id": rid})                             # 4. evaluate/diagnose
        self.assertEqual(ev["task"], "locomotion")
        edit = T("edit_robot", {"robot_id": rid,                               # 5. edit (localized)
                                "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "girth", "factor": 1.15}}]})
        self.assertTrue(edit["ok"])
        exp = T("export_held", {"robot_id": rid, "formats": ["mjcf"]})         # 6. export
        self.assertTrue(exp["ok"] and os.path.exists(exp["artifacts"]["mjcf"]))


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class SessionPersistenceTests(unittest.TestCase):
    """G-B: file-backed sessions so a stdio MCP client (its OWN process) and the app viewer share state.
    T3 in the plan FAILED here before this layer existed."""

    def setUp(self):
        import tempfile
        from virturoid.services import session_state as S
        self._tmp = tempfile.mkdtemp(prefix="virt_sess_")
        self._prev = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = self._tmp
        S.reset(wipe_disk=True)

    def tearDown(self):
        import shutil
        from virturoid.services import session_state as S
        S.reset(wipe_disk=True)
        if self._prev is None:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
        else:
            os.environ["VIRTUROID_SESSIONS_DIR"] = self._prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_reload_after_cache_drop_is_a_faithful_gene(self):
        # a second process has NO in-memory cache -> get_robot must reload the exact gene from disk.
        from virturoid.services import session_state as S
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        before = S.get_robot(rid).to_dict()
        S._ROBOTS.clear()                                          # simulate a fresh process (empty cache)
        after = S.get_robot(rid)
        self.assertIsNotNone(after, "a fresh process must see the robot via the shared session file")
        import json                                                # normalize tuple-vs-list (JSON has no tuples)
        norm = lambda d: json.loads(json.dumps(d, default=str))
        self.assertEqual(norm(after.to_dict()), norm(before), "disk round-trip must preserve the gene exactly")
        self.assertIn(rid, [r["robot_id"] for r in S.list_robots()])

    def test_edit_in_one_context_visible_after_cache_drop(self):
        from virturoid.services import session_state as S
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        h0 = self._call("get_robot", {"robot_id": rid})["standing_height_m"]
        self._call("edit_robot", {"robot_id": rid,
                                  "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": 1.3}}]})
        S._ROBOTS.clear()                                          # viewer polls from a cold cache
        h1 = self._call("get_robot", {"robot_id": rid})["standing_height_m"]
        self.assertGreater(h1, h0, "the app viewer must observe the agent's edit across the cache boundary")

    def test_real_subprocess_writes_parent_reads(self):
        # the actual T3 acceptance: a SEPARATE OS process authors a design; this process reads it back.
        import subprocess, sys
        child = (
            "import os;"
            "from virturoid.services.agent_tools import call_tool;"
            "s=call_tool('get_design_schema',{})['result']['examples']['quadruped'];"
            "r=call_tool('submit_design',{'graph':s})['result'];"
            "print('RID:'+r['robot_id'])"
        )
        env = dict(os.environ, VIRTUROID_SESSIONS_DIR=self._tmp, VIRTUROID_NO_LOCAL_ENV="1",
                   PYTHONPATH=os.pathsep.join(sys.path))
        out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(out.returncode, 0, f"child failed: {out.stderr[-800:]}")
        rid = next(l for l in out.stdout.splitlines() if l.startswith("RID:")).split("RID:")[1].strip()
        from virturoid.services import session_state as S
        self.assertIsNotNone(S.get_robot(rid), "parent process must see the child process's robot (T3 flipped green)")


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ZeroTokenSwitchTests(unittest.TestCase):
    """G-D: the provable zero-internal-token switch + spend ledger. The pitch is 'we spend no LLM tokens';
    this MEASURES it rather than asserting it."""

    def setUp(self):
        from virturoid.services import session_state as S
        from virturoid.services.llm_client import reset_spend_ledger
        S.reset()
        reset_spend_ledger()
        self._prev = os.environ.get("VIRTUROID_NO_INTERNAL_LLM")
        os.environ["VIRTUROID_NO_INTERNAL_LLM"] = "1"

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("VIRTUROID_NO_INTERNAL_LLM", None)
        else:
            os.environ["VIRTUROID_NO_INTERNAL_LLM"] = self._prev

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_switch_blocks_every_role_and_records_it(self):
        from virturoid.services.llm_client import get_llm
        for role in ("morphology", "scene", "planner", "assistant", "failure_analyst", "designer"):
            self.assertIsNone(get_llm(role), f"{role} must get NO backend under the switch")
        led = self._call("llm_spend")
        self.assertTrue(led["no_internal_llm"])
        self.assertEqual(led["totals"]["internal_calls"], 0)
        self.assertGreaterEqual(led["totals"]["blocked"], 6)      # all six denials recorded

    def test_full_loop_is_provably_zero_spend(self):
        # design -> simulate -> evaluate -> edit -> export, then the ledger must read zero internal calls.
        sch = self._call("get_design_schema")
        rid = self._call("submit_design", {"graph": sch["examples"]["quadruped"]})["robot_id"]
        self._call("simulate_gait", {"robot_id": rid, "steps": 300})
        self._call("evaluate_held", {"robot_id": rid})
        self._call("edit_robot", {"robot_id": rid,
                                  "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "girth", "factor": 1.1}}]})
        self._call("export_held", {"robot_id": rid, "formats": ["mjcf"]})
        led = self._call("llm_spend")
        self.assertTrue(led["zero_internal_spend"], f"loop must spend zero internal tokens; ledger={led['totals']}")
        self.assertEqual(led["totals"]["internal_calls"], 0)


class ToolConsolidationTests(unittest.TestCase):
    """G-G: the MCP surface is a lean, workflow-shaped <=15-tool view; folded tools stay callable by name."""

    def test_mcp_view_is_at_most_15_and_workflow_shaped(self):
        from virturoid.services.agent_tools import tool_specs, TOOLS
        view = tool_specs(view="mcp")
        self.assertLessEqual(len(view), 16, "MCP menu must fit the cross-client budget (well under Cursor's ~40)")
        names = [t["name"] for t in view]
        for essential in ("submit_design", "get_robot", "edit_robot", "verify_robot", "export_held",
                          "create_scene", "get_job", "llm_spend"):
            self.assertIn(essential, names, f"{essential} must be in the MCP view")
        for n in names:                                          # every advertised tool really dispatches
            self.assertIn(n, TOOLS)
        # the full registry stays larger — folded tools remain callable, just not advertised
        self.assertGreater(len(tool_specs()), len(view))

    def test_mcp_server_lists_the_consolidated_view(self):
        from virturoid.mcp_server import _handle
        listed = _handle("tools/list", {})["tools"]
        self.assertLessEqual(len(listed), 16)
        self.assertTrue(all("inputSchema" in t for t in listed))


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class FoldedToolTests(unittest.TestCase):
    """G-G: edit_robot folds undo + the op catalog; verify_robot folds simulate_gait via mode."""

    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_edit_robot_lists_ops_and_undoes(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        cat = self._call("edit_robot", {"robot_id": rid, "op": "list"})
        self.assertTrue(cat["ok"]); self.assertIn("operators", cat)
        h0 = self._call("get_robot", {"robot_id": rid})["standing_height_m"]
        self._call("edit_robot", {"robot_id": rid,
                                  "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": 1.3}}]})
        self.assertGreater(self._call("get_robot", {"robot_id": rid})["standing_height_m"], h0)
        undo = self._call("edit_robot", {"robot_id": rid, "op": "undo"})   # folded undo_robot
        self.assertTrue(undo["ok"])
        self.assertAlmostEqual(self._call("get_robot", {"robot_id": rid})["standing_height_m"], h0, places=3)

    def test_verify_robot_quick_mode(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        q = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(q["mode"], "quick")
        self.assertIn("verdict", q)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ViewerRoutesTests(unittest.TestCase):
    """G-B viewer: the running app exposes /api/sessions so its webapp live-follows the agent's held robot,
    served over REAL HTTP (the agent writes to the shared session dir; the app reads it back)."""

    def setUp(self):
        import tempfile, threading
        from pathlib import Path
        from virturoid.services import session_state as S
        from virturoid.ui_server import create_server
        self._tmp = tempfile.mkdtemp(prefix="virt_view_")
        self._prev = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = self._tmp
        S.reset(wipe_disk=True)
        self._srv = create_server("127.0.0.1", 0, Path(self._tmp))
        self._port = self._srv.server_address[1]
        self._th = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._th.start()

    def tearDown(self):
        import shutil
        from virturoid.services import session_state as S
        self._srv.shutdown(); self._srv.server_close()
        S.reset(wipe_disk=True)
        if self._prev is None:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
        else:
            os.environ["VIRTUROID_SESSIONS_DIR"] = self._prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _get(self, path):
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self._port}{path}", timeout=30) as r:
            return r.status, r.headers.get_content_type(), r.read()

    def test_sessions_list_and_detail_and_render_over_http(self):
        import json
        from virturoid.services.agent_tools import call_tool
        sch = call_tool("get_design_schema", {})["result"]
        rid = call_tool("submit_design", {"graph": sch["examples"]["quadruped"]})["result"]["robot_id"]

        st, _, body = self._get("/api/sessions")               # the app lists what the agent is holding
        self.assertEqual(st, 200)
        self.assertIn(rid, [r["robot_id"] for r in json.loads(body)["robots"]])

        st, _, body = self._get(f"/api/sessions/{rid}")        # detail + a render URL
        self.assertEqual(st, 200)
        detail = json.loads(body)
        self.assertEqual(detail["summary"]["robot_class"], "quadruped")
        self.assertTrue(detail["render_url"], "detail must include a render URL for the viewer")

        st, ctype, img = self._get(detail["render_url"])       # the render itself is served
        self.assertEqual(st, 200)
        self.assertEqual(ctype, "image/png")
        self.assertGreater(len(img), 1000)

    def test_sessions_viewer_page_serves(self):
        # C1-C3: the 'watch the agent build' viewer page is served and references the live APIs it polls.
        st, ctype, body = self._get("/sessions")
        self.assertEqual(st, 200)
        self.assertIn("text/html", ctype)
        html = body.decode("utf-8", "ignore")
        self.assertIn("Agent Sessions", html)
        self.assertIn("/api/sessions", html)                    # it polls the session list
        self.assertIn("render_url", html)                       # and shows each robot's render


if __name__ == "__main__":
    unittest.main()
