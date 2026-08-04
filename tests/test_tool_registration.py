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
    assert env["ok"] is True, env                                # the dispatcher did not blow up ...
    r = env["result"]
    assert r["ok"] is False and "unknown kind" in r["error"]     # ... and the tool refused, naming the fix
    assert "expect_contact" in r["assertions"], "a refusal must hand back the vocabulary"


@pytest.mark.parametrize("tool", _THREE)
def test_an_unknown_robot_is_an_honest_failure_not_a_crash(tool):
    """Registration must not turn a bad robot_id into a stack trace on the agent's side."""
    from virturoid.services.agent_tools import call_tool
    args = {"robot_id": "no_such_robot_xyz"}
    if tool == "scope_amend":
        args["ops"] = [{"op": "set_height", "args": {"target_m": 0.5}}]
    env = call_tool(tool, args)
    assert env["ok"] is True, env
    assert env["result"]["ok"] is False
    assert "no robot" in env["result"]["error"], env["result"]


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
    """`MCP_TOOL_VIEW` is at its documented cross-client cap of 15 (test_agent_first asserts it), so these are
    advertised on the anchor tool they belong to -- the same treatment the ingest importers and the advanced
    authoring compilers get. A client reading tools/list therefore SEES all three names and can call them."""
    from virturoid.mcp_server import _handle
    listed = _handle("tools/list", {})["tools"]
    assert len(listed) <= 15, "the lean menu must not grow past its budget"
    blob = " ".join(t["description"] for t in listed)
    for name in _THREE:
        assert name in blob, f"{name} dispatches but no tools/list entry names it -- undiscoverable"


# ---------------------------------------------------------------- the guard that stops this recurring

# snake_case vocabulary that shares a VERB PREFIX with a registered tool but is a FIELD NAME, not a callable.
# Anything else matching a tool verb is treated as an advertised tool and must dispatch.
_NOT_A_TOOL = frozenset({"part_fields"})

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

    verbs = _TOOL_VERBS | {n.split("_")[0] for n in TOOLS}
    bad: dict[str, set[str]] = {}
    for src, txt in blobs.items():
        # `(?!\s*=)` skips KEYWORD ARGUMENTS. Prose like ``train_backend="gpu"`` documents a PARAMETER, not a
        # tool call, but `train_` is a live tool verb (train_held, train_reward) so a bare prefix match flags it.
        # A denylist entry would only defer this: every future kwarg sharing a tool verb would trip the same wire.
        # An identifier written bare ANYWHERE still gets flagged -- that is the case this guard exists for.
        for ident in set(re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b(?!\s*=)", txt)):
            if ident in TOOLS or ident in _NOT_A_TOOL:
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
