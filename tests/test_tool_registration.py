"""Three finished tools were reachable by nobody -- and we advertised one of them to the customer's model.

`probe_robot`, `scope_amend` and `assert_design` shipped fully implemented (`ai_native_tools.py:1095/1123/1146`)
over engines that were themselves complete and tested (`robot_probe.probe`, `change_impact.scope`,
`design_assertions.check`). They appeared in NO registry, so `call_tool`, MCP `tools/list` and MCP `tools/call`
all answered `unknown tool`. Meanwhile `get_design_schema` -- the grounding pack we hand the customer's own LLM
-- told it, of the joint-axis field that is the easiest thing in the language to get wrong: "Check it with
probe_robot rather than reasoning about frames." An agent following our own instructions called a tool that was
not on the wire.

EVERY test here goes through `agent_tools.call_tool`, the dispatcher an agent actually reaches the platform
through. That is the whole point: a test that imported `robot_probe.probe` directly would have PASSED at HEAD,
while the tool was 0% agent-reachable, and would therefore have proved nothing. `tests/test_robot_probe.py`,
`test_change_impact.py` and `test_design_assertions.py` already cover the engines; what was missing was any test
that the agent can get to them.
"""
from __future__ import annotations

import importlib.util
import os
import re

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_THREE = ("probe_robot", "scope_amend", "assert_design")


@pytest.fixture(scope="module")
def held_dog():
    """A held robot_id -- the handle all three tools are addressed by.

    Seeded straight into the session rather than through `call_tool("create_robot")` because that path grounds
    and gait-fits the body and measures ~93 s; the SUBJECT here is the dispatcher, not the composer. The tools
    under test are still reached only through `call_tool`."""
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    return S.put_robot(compose_robot("a four legged robot dog", llm=None), prompt="a four legged robot dog")


# ---------------------------------------------------------------- reachable through the dispatcher

@pytest.mark.skipif(not _MUJOCO, reason="probing needs MuJoCo")
def test_probe_robot_dispatches_and_returns_real_measurements(held_dog):
    """Not just `ok` -- numbers. An entry that dispatched and returned an empty envelope would satisfy
    "registered" while leaving the agent exactly as unable to measure the robot as before."""
    from virturoid.services.agent_tools import call_tool
    env = call_tool("probe_robot", {"robot_id": held_dog, "fields": ["mass", "torque", "reach"]})
    assert env["ok"] is True and env["tool"] == "probe_robot", env
    r = env["result"]
    assert r["ok"] is True and r["robot_id"] == held_dog
    assert r["mass"]["total_kg"] > 0.0, r["mass"]
    assert r["reach"]["distance_m"] > 0.0, r["reach"]
    assert r["torque"], "no joints measured"
    # the torque margin is the answer the plan's Stage-0 demo asks for ("what is the margin on my knee joint?")
    for name, row in r["torque"].items():
        assert row["static_hold_nm"] >= 0.0, f"{name}: {row}"
        assert row["distal_mass_kg"] > 0.0, f"{name}: {row}"


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_scope_amend_dispatches_and_names_what_the_edit_invalidates(held_dog):
    from virturoid.services.agent_tools import call_tool
    env = call_tool("scope_amend", {"robot_id": held_dog, "ops": [
        {"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": 1.2}}]})
    assert env["ok"] is True and env["tool"] == "scope_amend", env
    r = env["result"]
    assert r["ok"] is True and r["robot_id"] == held_dog
    for check in ("gait", "torque", "mass", "stability"):
        assert check in r["invalidates"], f"lengthening the legs must invalidate {check}: {r['invalidates']}"
    assert r["invalidates_meaning"], "a recheck list without meanings is a tag dump"


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_scope_amend_is_a_DRY_RUN_and_edits_nothing(held_dog):
    """The one property that makes it worth having: it answers before it commits. Also exercises the single-op
    shortcut form (`op`/`args` instead of `ops`), which the schema advertises."""
    from virturoid.services.agent_tools import call_tool
    before = call_tool("get_robot", {"robot_id": held_dog})["result"]
    env = call_tool("scope_amend", {"robot_id": held_dog, "op": "set_height", "args": {"target_m": 0.9}})
    assert env["ok"] is True and env["result"]["ok"] is True, env
    after = call_tool("get_robot", {"robot_id": held_dog})["result"]
    assert after["standing_height_m"] == before["standing_height_m"], "a dry run must not move the robot"
    assert after["total_mass_kg"] == before["total_mass_kg"]
    assert after["n_segments"] == before["n_segments"]
    assert after["undo_depth"] == before["undo_depth"], "a dry run must not land an undo step"


@pytest.mark.skipif(not _MUJOCO, reason="checking assertions needs MuJoCo")
def test_assert_design_dispatches_and_actually_discriminates(held_dog):
    """A checker that passes everything is worse than none. One true claim and one false one, so the test fails
    if `assert_design` degrades into a rubber stamp."""
    from virturoid.services.agent_tools import call_tool
    parts = call_tool("probe_robot", {"robot_id": held_dog, "fields": ["parts"]})["result"]["parts"]
    root = "torso" if "torso" in parts else sorted(parts)[0]
    far = max(parts, key=lambda n: sum(abs(v) for v in parts[n]["centre"]))
    env = call_tool("assert_design", {"robot_id": held_dog, "assertions": [
        {"kind": "expect_clearance", "a": root, "min_m": 0.001},          # TRUE: the torso is off the floor
        {"kind": "expect_within", "a": root, "b": far, "max_m": 1e-4},    # FALSE: they are not 0.1 mm apart
    ]})
    assert env["ok"] is True and env["tool"] == "assert_design", env
    r = env["result"]
    assert r["ok"] is True and r["n"] == 2
    assert r["failed"] == 1, r["results"]
    assert r["results"][0]["passed"] is True, r["results"][0]
    assert r["results"][1]["passed"] is False, r["results"][1]


def test_assert_design_list_form_returns_the_vocabulary():
    """The `kind:'list'` shortcut takes no robot -- an agent has to be able to learn the vocabulary before it
    holds anything."""
    from virturoid.services.agent_tools import call_tool
    env = call_tool("assert_design", {"kind": "list"})
    assert env["ok"] is True, env
    assert env["result"]["ok"] is True
    assert "expect_contact" in env["result"]["assertions"]


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_a_bad_assertion_teaches_instead_of_crashing(held_dog):
    from virturoid.services.agent_tools import call_tool
    env = call_tool("assert_design", {"robot_id": held_dog,
                                      "assertions": [{"kind": "expect_teleport", "a": "torso"}]})
    r = env["result"]
    assert isinstance(r, dict), env                              # the dispatcher did not blow up (no traceback) ...
    assert r["ok"] is False and "unknown kind" in r["error"]     # ... and the tool refused, naming the fix
    assert env["ok"] is False, "a refused call must not be wrapped in a success envelope"
    assert "unknown kind" in env["error"], env                   # the reason reaches the envelope the client reads
    assert "expect_contact" in r["assertions"], "a refusal must hand back the vocabulary"


@pytest.mark.parametrize("tool", _THREE)
def test_an_unknown_robot_is_an_honest_failure_not_a_crash(tool):
    """Registration must not turn a bad robot_id into a stack trace on the agent's side."""
    from virturoid.services.agent_tools import call_tool
    args = {"robot_id": "no_such_robot_xyz"}
    if tool == "scope_amend":
        args["ops"] = [{"op": "set_height", "args": {"target_m": 0.5}}]
    env = call_tool(tool, args)
    assert isinstance(env.get("result"), dict), env               # structured refusal, not a stack trace
    assert env["result"]["ok"] is False
    assert "no robot" in env["result"]["error"], env["result"]
    # ... and the envelope says so too. mcp_server sets isError from env["ok"], so ok=True here told the client
    # that "no robot 'no_such_robot_xyz'" was a successful call.
    assert env["ok"] is False, env


# ---------------------------------------------------------------- on the MCP wire, not just in the registry

def test_the_three_are_registered_with_well_formed_schemas():
    from virturoid.services.agent_tools import TOOLS
    for name in _THREE:
        assert name in TOOLS, f"{name} is implemented but registered nowhere -- it is 0% agent-reachable"
        spec = TOOLS[name]
        assert callable(spec["handler"])
        params = spec["parameters"]
        assert params["type"] == "object" and "robot_id" in params["properties"], params
        assert spec["description"].strip()


@pytest.mark.skipif(not _MUJOCO, reason="the MCP registry import needs MuJoCo")
def test_the_three_dispatch_over_MCP_tools_call(held_dog):
    """Being in the internal registry is not the same as being on the wire. `tools/call` is the wire."""
    from virturoid.mcp_server import _handle
    for name, args in (("probe_robot", {"robot_id": held_dog, "fields": ["mass"]}),
                       ("scope_amend", {"robot_id": held_dog, "ops": [{"op": "set_height", "args": {"target_m": 0.5}}]}),
                       ("assert_design", {"kind": "list"})):
        res = _handle("tools/call", {"name": name, "arguments": args})
        assert res["isError"] is False, (name, res)
        assert res["structuredContent"].get("ok") is True, (name, res["structuredContent"])


@pytest.mark.skipif(not _MUJOCO, reason="the MCP registry import needs MuJoCo")
def test_the_three_are_discoverable_in_the_tools_list_payload():
    """`MCP_TOOL_VIEW` is at its documented cross-client cap (`MCP_TOOL_VIEW_MAX`), so these are advertised on
    the anchor tool they belong to -- the same treatment the ingest importers and the advanced authoring
    compilers get. A client reading tools/list therefore SEES all three names and can call them."""
    from virturoid.mcp_server import _handle
    from virturoid.services.agent_tools import MCP_TOOL_VIEW_MAX
    listed = _handle("tools/list", {})["tools"]
    assert len(listed) <= MCP_TOOL_VIEW_MAX, "the lean menu must not grow past its budget"
    blob = " ".join(t["description"] for t in listed)
    for name in _THREE:
        assert name in blob, f"{name} dispatches but no tools/list entry names it -- undiscoverable"


# ---------------------------------------------------------------- the guard that stops this recurring

# snake_case vocabulary that shares a VERB PREFIX with a registered tool but is a FIELD NAME, not a callable.
# Anything else matching a tool verb is treated as an advertised tool and must dispatch.
_NOT_A_TOOL = frozenset({"part_fields"})


def _not_callable_vocabulary(tools: dict) -> frozenset[str]:
    """`_NOT_A_TOOL` plus the EDIT OPERATOR names, which are a separate namespace an agent passes INSIDE
    ``edit_robot {ops:[{op}]}`` and never calls. Derived from ``edit_operators.OPERATORS`` rather than listed,
    for the same reason `edit_ops`' description is: `adopt_walkable_template` shares the `adopt_` verb with
    `adopt_control_script`, so restating it here would just move the drift one file over."""
    try:
        from virturoid.services.edit_operators import OPERATORS
    except Exception:  # noqa: BLE001
        return _NOT_A_TOOL
    return _NOT_A_TOOL | frozenset(op for op in OPERATORS if op not in tools)

# The verbs tool names are built from. STATIC ON PURPOSE, and unioned with whatever the live registry uses.
# Deriving this set only from `TOOLS` makes the check go blind at exactly the moment it matters: unregister the
# last tool starting with `probe_` and `probe` leaves the prefix set, so `probe_robot` in the schema stops
# looking like a tool name and the guard passes on the very bug it exists to catch (verified: with the three
# popped, the derived-only version of this test was the ONLY one of 14 that still passed).
_TOOL_VERBS = frozenset({
    "adapt", "amplify", "adopt", "ask", "assert", "bank", "build", "capabilities", "check", "classify", "create",
    "critique", "data", "describe", "design", "diagnose", "edit", "evaluate", "export", "flywheel", "generate",
    "get", "import", "ingest", "inspect", "interpret", "learn", "list", "llm", "nearest", "part", "pin", "plan",
    "probe", "recall", "render", "run", "sandbox", "scope", "search", "simulate", "start", "submit", "train",
    "undo", "verify",
})


def test_no_agent_facing_text_names_a_tool_that_does_not_exist():
    """THE regression test for the actual defect, which was not "a tool is missing" but "we INSTRUCT the
    customer's model to call a tool that is missing".

    Scans everything we put in front of an LLM -- the MCP `initialize` instructions, the workflow prompts, the
    design schema, and every registered tool's own description -- for identifiers built on the same verb prefixes
    the tool registry uses (`probe_`, `scope_`, `assert_`, `submit_`, `get_`, ...). Each must be a tool that
    actually dispatches. At HEAD this failed on `probe_robot`, named in `get_design_schema`'s `axis` field.

    If this fails on a new schema KEY rather than a tool, add it to `_NOT_A_TOOL`; if it fails on a real tool,
    register it rather than deleting the sentence."""
    from virturoid.services.agent_tools import TOOLS

    blobs = _agent_facing_blobs()                           # the SAME corpus the parameter guard below scans,
    verbs = _TOOL_VERBS | {n.split("_")[0] for n in TOOLS}  # so neither can go blind on text the other reads
    not_tools = _not_callable_vocabulary(TOOLS)
    bad: dict[str, set[str]] = {}
    for src, txt in blobs.items():
        # `(?!\s*=)` skips KEYWORD ARGUMENTS. Prose like ``train_backend="gpu"`` documents a PARAMETER, not a
        # tool call, but `train_` is a live tool verb (train_held, train_reward) so a bare prefix match flags it.
        # A denylist entry would only defer this: every future kwarg sharing a tool verb would trip the same wire.
        # An identifier written bare ANYWHERE still gets flagged -- that is the case this guard exists for.
        for ident in set(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b(?!\s*=)", txt)):
            if ident in TOOLS or ident in not_tools:
                continue
            if ident.split("_")[0] in verbs:
                bad.setdefault(ident, set()).add(src)
    assert not bad, ("agent-facing text names tools that do not dispatch: "
                     + "; ".join(f"{k} (in {sorted(v)})" for k, v in sorted(bad.items())))


def test_the_design_schema_still_points_at_probe_robot():
    """The specific sentence that was false. Kept as its own test so that if someone ever deletes the tool
    again, the failure names the customer-facing instruction rather than an abstract registry count."""
    from virturoid.services.agent_design_tools import AGENT_DESIGN_TOOLS
    from virturoid.services.agent_tools import TOOLS
    axis = AGENT_DESIGN_TOOLS["get_design_schema"]["handler"]({})["part_fields"]["axis"]
    assert "probe_robot" in axis
    assert "probe_robot" in TOOLS, "the schema tells the customer's LLM to call probe_robot; it must exist"


# ---------------------------------------------------------------- the same guard, one level down: PARAMETERS
#
# The tool-level guard above says the NAME dispatches. It says nothing about the ARGUMENTS, and that is where
# the same defect had simply moved: `verify_robot` advertised `inputSchema {robot_id}` while both the MCP
# `initialize` instructions and the `design_robot_workflow` prompt told agents to send `mode:'quick'`. A strict
# MCP client validates `tools/call` arguments against the advertised schema and DROPS the call, so the server
# was instructing clients to make a request it had declared invalid — the tool "existed" and was still
# unusable as documented. `train_reward {robot_id, objective}` was the same lie with no parameter at all
# behind it (the handler takes `task`). Two directions are checked, because both were wrong:
#
#   (a) PROSE -> SCHEMA: every parameter we NAME in agent-facing text must be declared. (advertised-but-absent)
#   (b) HANDLER -> SCHEMA: every parameter a handler HONOURS must be declared, or an agent cannot reach it.
#
# Both scanners are factored out of their tests so `test_..._has_teeth` can run them against a deliberately
# broken registry and prove they fail — a guard nobody has ever seen fail is not evidence.

# `tool {a, b}` / `tool {{a:'x'}}` (prompt templates double the braces) and `tool (a:'x')`. Body stops at the
# next brace so a NESTED literal — `edit_scene {{ops:[{{op:'swap_theme'}}]}}` — contributes `ops` and not the
# operator's own `op`/`args`, which belong to the ops-item schema rather than the tool's.
_BRACE_CALL = re.compile(r"\b([a-z][a-z0-9_]*)\s*\{\{?([^{}]*)")
_PAREN_CALL = re.compile(r"\b([a-z][a-z0-9_]*)\s*\(([^()]*)")
_KEYED = re.compile(r"\b([a-z][a-z0-9_]*)\s*:")
_BARE_ITEM = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*$")


def _prose_named_params(blobs: dict[str, str], tools: dict) -> dict[tuple[str, str], set[str]]:
    """``{(tool, param): {sources}}`` for every parameter named in prose that the tool does not declare."""
    bad: dict[tuple[str, str], set[str]] = {}
    for src, txt in blobs.items():
        mentions: list[tuple[str, str]] = []
        for tool, body in _BRACE_CALL.findall(txt):
            if tool not in tools:
                continue
            keys = set(_KEYED.findall(body))
            keys |= {m.group(1) for m in (_BARE_ITEM.match(c) for c in body.split(",")) if m}
            mentions += [(tool, k) for k in keys]
        for tool, body in _PAREN_CALL.findall(txt):
            # parens carry ordinary prose ("create_robot (from a prompt)"), so only `key:` counts here
            if tool in tools:
                mentions += [(tool, k) for k in _KEYED.findall(body)]
        for tool, key in mentions:
            if key not in ((tools[tool].get("parameters") or {}).get("properties") or {}):
                bad.setdefault((tool, key), set()).add(src)
    return bad


def _agent_facing_blobs() -> dict[str, str]:
    """Everything we put in front of an LLM: the MCP handshake, the workflow prompts, every tool description."""
    from virturoid.mcp_server import _PROMPTS, _handle
    from virturoid.services.agent_design_tools import AGENT_DESIGN_TOOLS
    from virturoid.services.agent_tools import TOOLS

    schema = dict(AGENT_DESIGN_TOOLS["get_design_schema"]["handler"]({}))
    schema.pop("corpus_grounding", None)                    # retrieved from the corpus; not agent-facing prose
    blobs = {"mcp initialize instructions": _handle("initialize", {})["instructions"],
             "get_design_schema": str(schema)}
    for n, p in _PROMPTS.items():
        blobs[f"mcp prompt {n!r}"] = p["description"] + " " + p["text"]
    for n, t in TOOLS.items():
        blobs[f"description of {n!r}"] = t["description"]
    return blobs


def test_no_agent_facing_text_names_a_parameter_the_schema_does_not_declare():
    """(a) advertised-but-absent PARAMETERS. At HEAD this failed on `verify_robot {mode:'quick'}` (named in the
    initialize instructions AND the design_robot_workflow prompt, absent from a `{robot_id}`-only schema) and on
    `train_reward {robot_id, objective}` (no such parameter at all — the handler takes `task`).

    Fix by making the schema true, or by correcting the sentence when the parameter really does not exist."""
    from virturoid.services.agent_tools import TOOLS
    bad = _prose_named_params(_agent_facing_blobs(), TOOLS)
    assert not bad, ("agent-facing text names parameters a strict MCP client would reject: "
                     + "; ".join(f"{t}.{k} (in {sorted(v)})" for (t, k), v in sorted(bad.items())))


def _unadvertised_params(tools: dict) -> dict[str, list[str]]:
    """``{tool: [keys the handler honours but the schema hides]}``."""
    from virturoid.services.agent_tools import _NOT_ADVERTISED, _accepted_params
    out: dict[str, list[str]] = {}
    for name, spec in tools.items():
        props = set(((spec.get("parameters") or {}).get("properties") or {}))
        hidden = _accepted_params(spec.get("handler")) .keys() - props - set(_NOT_ADVERTISED.get(name, ()))
        if hidden:
            out[name] = sorted(hidden)
    return out


def test_every_parameter_a_handler_honours_is_advertised():
    """(b) the direction prose alone cannot see. A parameter the handler reads out of ``args`` but the schema
    never declares is unreachable from a strict client even if no sentence ever mentions it — which is how
    `verify_robot`'s `steps` and `scene_id` sat hidden with no prose to betray them. Measured at HEAD: 21 tools
    hid 39 parameters between them; `agent_tools._publish_accepted_params` now derives them from the handlers,
    so this stays at zero without anyone maintaining a list."""
    from virturoid.services.agent_tools import TOOLS
    hidden = _unadvertised_params(TOOLS)
    assert not hidden, ("handlers honour parameters their schema hides (agents cannot reach them): "
                        + "; ".join(f"{t}: {ks}" for t, ks in sorted(hidden.items())))


def _forwards_args(handler) -> bool:
    """True if the handler hands its whole args dict to another call (``probe(gene, args)``).

    Such a tool's parameters are read one frame down, where the AST scan cannot see them, so the check below
    has to excuse it. Detected rather than allowlisted: an allowlist would quietly excuse the next tool that
    stops forwarding, which is exactly when its dead parameters would start mattering."""
    import ast
    import inspect
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    except Exception:  # noqa: BLE001
        return True                                          # unreadable source: do not accuse it
    fn = tree.body[0]
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.args.args:
        return True
    argname = fn.args.args[0].arg
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            passed = list(node.args) + [k.value for k in node.keywords]
            if any(isinstance(a, ast.Name) and a.id == argname for a in passed):
                return True
    return False


def test_no_tool_advertises_a_parameter_its_handler_ignores():
    """The MIRROR of the guard above, and a lie in the other direction: a schema property nothing reads promises
    the agent a lever that moves nothing, and it fails SILENTLY — the call succeeds and the argument is dropped.
    Found at HEAD on `evaluate_held.task`: `evaluate_held` scores the morphology-implied task and takes the
    prompt from the held robot's metadata, so `task:'transport'` changed nothing and said nothing.

    Handlers that forward their whole args dict one frame down are excused (detected, not allowlisted)."""
    from virturoid.services.agent_tools import TOOLS, _accepted_params
    dead: dict[str, list[str]] = {}
    for name, spec in TOOLS.items():
        handler = spec.get("handler")
        if _forwards_args(handler):
            continue
        unread = sorted(set((spec.get("parameters") or {}).get("properties") or {}) - set(_accepted_params(handler)))
        if unread:
            dead[name] = unread
    assert not dead, ("tools advertise parameters no handler reads (they are silently dropped): "
                      + "; ".join(f"{t}: {ks}" for t, ks in sorted(dead.items())))


def test_the_curated_parameter_docs_describe_parameters_that_still_exist():
    """`_PARAM_DOCS` is the one hand-written part of the repair — it supplies enum/units/meaning where the
    derived shape is too thin. An entry naming a key no handler reads any more is the drift this whole file
    exists to prevent, so it fails here rather than shipping a documented parameter that does nothing."""
    from virturoid.services.agent_tools import TOOLS, _accepted_params, _PARAM_DOCS
    stale = [f"{t}.{k}" for t, doc in _PARAM_DOCS.items() for k in doc
             if k not in _accepted_params((TOOLS.get(t) or {}).get("handler"))]
    assert not stale, f"_PARAM_DOCS documents parameters no handler reads: {stale}"


def test_verify_robot_advertises_the_mode_the_server_tells_agents_to_send():
    """THE specific call that was rejected. Named on its own so the failure reads as the customer-facing
    contract it is, not as an abstract count."""
    from virturoid.mcp_server import _PROMPTS, _handle
    from virturoid.services.agent_tools import TOOLS
    props = TOOLS["verify_robot"]["parameters"]["properties"]
    for key in ("mode", "steps", "scene_id"):
        assert key in props, f"verify_robot honours {key!r} but does not advertise it"
    assert props["mode"].get("enum") == ["quick", "full"], props["mode"]
    taught = _handle("initialize", {})["instructions"] + _PROMPTS["design_robot_workflow"]["text"]
    assert "mode:'quick'" in taught, "the instructions that motivated this must still teach the parameter"


def test_edit_ops_advertises_every_operator_the_registry_can_apply():
    """The description named 4 of 8 (scale_group/set_height/set_material/set_leg_count), so scale_robot,
    set_payload, add_limb and adopt_walkable_template were undiscoverable — including the only op that can grow
    a new limb. It is derived from ``OPERATORS`` now; this fails if anyone restates it by hand again."""
    from virturoid.services.agent_tools import TOOLS
    from virturoid.services.edit_operators import OPERATORS
    desc = TOOLS["edit_ops"]["description"]
    missing = [op for op in OPERATORS if op not in desc]
    assert not missing, f"edit_ops does not name {missing}; derive the description from OPERATORS"
    enum = TOOLS["edit_robot"]["parameters"]["properties"]["ops"]["items"]["properties"]["op"]["enum"]
    assert set(OPERATORS) <= set(enum), f"edit_robot ops enum is missing {sorted(set(OPERATORS) - set(enum))}"
    for verb in ("undo", "list"):                    # the documented single-verb shortcut must stay valid
        assert verb in enum, f"the ops enum would reject the documented {{op:'{verb}'}} call"


def test_export_held_advertises_every_format_it_will_actually_write():
    """The same drift, one tool over: the description listed 8 of the 9 `_EXPORT_FORMATS`, and the one it left
    out was `certificate` — the verification certificate. `export_held` REJECTS an unlisted format outright
    ("unknown format(s) ..."), so a format missing from the description and absent from the schema enum was a
    real export the customer's agent had no way to ask for."""
    from virturoid.services.agent_design_tools import _EXPORT_FORMATS
    from virturoid.services.agent_tools import TOOLS
    spec = TOOLS["export_held"]
    missing = [f for f in _EXPORT_FORMATS if f not in spec["description"]]
    assert not missing, f"export_held does not name {missing}; derive the list from _EXPORT_FORMATS"
    enum = spec["parameters"]["properties"]["formats"]["items"]["enum"]
    assert list(enum) == list(_EXPORT_FORMATS), f"formats enum {enum} != registry {list(_EXPORT_FORMATS)}"


def test_the_parameter_guards_have_teeth():
    """A guard nobody has watched fail is decoration. Both scanners are run here against a DELIBERATELY broken
    registry — a schema with the parameter removed, and a handler that reads a key it never declares — and must
    flag exactly that. If either returns clean, the tests above are passing vacuously."""
    from virturoid.services.agent_tools import TOOLS

    # (a) prose names a parameter the schema does not declare -> flagged
    broken = {"verify_robot": {**TOOLS["verify_robot"],
                               "parameters": {"type": "object", "required": ["robot_id"],
                                              "properties": {"robot_id": {"type": "string"}}}}}
    blob = {"a fake instruction": "verify_robot {robot_id, mode:'quick'} then export_held {robot_id}."}
    bad = _prose_named_params(blob, broken)
    assert ("verify_robot", "mode") in bad, f"the prose guard missed a removed parameter: {bad}"
    # ... and stays quiet on the real, repaired registry
    assert not _prose_named_params(blob, TOOLS), "flagged a parameter that IS advertised"

    # the hardening this file already carries: `train_backend="gpu"` is a KEYWORD ARGUMENT in prose, not a tool
    kwarg_blob = {"kwarg prose": "Set train_backend=\"gpu\" and the same expression steers MJX PPO."}
    verbs = _TOOL_VERBS | {n.split("_")[0] for n in TOOLS}
    not_tools = _not_callable_vocabulary(TOOLS)
    flagged = {i for i in re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b(?!\s*=)", kwarg_blob["kwarg prose"])
               if i not in TOOLS and i not in not_tools and i.split("_")[0] in verbs}
    assert not flagged, f"a documented keyword argument was mistaken for a tool: {flagged}"

    # (b) a handler honouring an undeclared key -> flagged
    def _handler(args: dict) -> dict:
        return {"ok": True, "n": int(args.get("undeclared_knob", 3))}

    fake = {"fake_tool": {"description": "x", "handler": _handler,
                          "parameters": {"type": "object", "properties": {}}}}
    assert _unadvertised_params(fake) == {"fake_tool": ["undeclared_knob"]}, _unadvertised_params(fake)
