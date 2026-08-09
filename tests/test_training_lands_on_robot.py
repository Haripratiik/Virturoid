"""The training loop was OPEN: nothing we trained ever reached the robot.

Four engineers used the product on a real MuJoCo-Menagerie Unitree Go2. One trained it twice, by two different
paths, and the product re-measured the UNTRAINED controller both times and said so in its own provenance field:

    | verify_robot full, real Go2 | forward_m | gait_source     |
    | before training             | 0.917     | flywheel_hint   |
    | after train_reward (227 s)  | 1.102     | flywheel_hint   |

The cause was one missing verb. ``reward_loop._train_reward`` read the gene with ``session_state.get_robot`` and
never wrote one back; neither did ``gait_hints._adapt_gait`` nor ``input_training_tools._learn_gait``. Every
``put_robot`` caller in the repo was a design/ingest/edit path — no training tool mutated the held robot, and no
tool anywhere accepted gait parameters as INPUT, so an engineer who read the trained numbers had no way to
attach them either.

EVERY test here goes through ``agent_tools.call_tool`` or the MCP ``_handle``, because that is the surface the
engineers used. A test that called ``learn_gait_flywheel`` directly would have PASSED at HEAD while the product
discarded its result — which is exactly how this survived to a customer-facing demo.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_GOOD = {"freq": 2.0, "hip_amp": 0.7, "knee_amp": 1.1, "kp": 90.0, "kd": 4.0}


@pytest.fixture()
def held_dog():
    """A held robot_id. Seeded into the session directly — the SUBJECT is the write side, not the composer."""
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    return S.put_robot(compose_robot("a four legged robot dog", llm=None), prompt="a four legged robot dog")


# ---------------------------------------------------------------- the parameters must survive the round trip

@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_applying_a_trained_gait_changes_the_held_gene(held_dog):
    """The whole bug in one assertion: after training, the held gene must carry the trained parameters.

    Read back through ``session_state.get_robot`` — the exact call every verdict path makes — because that is
    what was byte-identical before and after training.
    """
    from virturoid.services import session_state as S
    from virturoid.services.trained_controller import apply_trained_gait
    assert not (getattr(S.get_robot(held_dog), "metadata", {}) or {}).get("gait_params")
    rep = apply_trained_gait(held_dog, _GOOD, door="learn_gait", credible=True, verdict="CREDIBLE WALK")
    assert rep["applied"] is True, rep
    md = getattr(S.get_robot(held_dog), "metadata", {}) or {}
    assert md["gait_params"] == _GOOD
    assert md["gait_provenance"]["door"] == "learn_gait"
    assert rep["gait_source_after"] == "tuned_for_this_body::learn_gait"


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_the_landing_is_persisted_not_just_mutated_in_memory(held_dog):
    """``get_robot`` hands back the LIVE gene, so mutating it in place looks like it worked — until another
    process (the MCP server vs. the running viewer) reloads from ``build/sessions`` and gets the old controller.
    The commit has to reach the session file."""
    import json
    from virturoid.services import session_state as S
    from virturoid.services.trained_controller import apply_trained_gait
    apply_trained_gait(held_dog, _GOOD, door="train_reward", credible=True)
    disk = json.loads(S._robot_path(held_dog).read_text(encoding="utf-8"))
    assert disk["gene"]["metadata"]["gait_params"] == _GOOD


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_training_is_an_edit_so_undo_restores_the_previous_controller(held_dog):
    """Applied through ``commit_robot``, so the controller a training run replaced is on the SAME undo stack
    ``edit_robot {ops:[{op:'undo'}]}`` pops. A training tool that could not be taken back would be a worse
    contract than the one that discarded results."""
    from virturoid.services.agent_tools import call_tool
    from virturoid.services import session_state as S
    from virturoid.services.trained_controller import apply_trained_gait
    apply_trained_gait(held_dog, _GOOD, door="adapt_gait", credible=True)
    env = call_tool("edit_robot", {"robot_id": held_dog, "ops": [{"op": "undo"}]})
    assert env["ok"], env
    assert not (getattr(S.get_robot(held_dog), "metadata", {}) or {}).get("gait_params")


# ---------------------------------------------------------------- the contract: auto / always / never

@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_auto_refuses_a_non_credible_run_and_says_why(held_dog):
    """A run that did not produce a walk does not get to overwrite a controller that might — and the refusal
    carries the verdict AND the way to land it anyway, so it is a decision the engineer can act on rather than
    a silent no-op (which is what the open loop looked like)."""
    from virturoid.services import session_state as S
    from virturoid.services.trained_controller import apply_trained_gait
    rep = apply_trained_gait(held_dog, _GOOD, door="train_reward", credible=False, verdict="CROUCH")
    assert rep["applied"] is False
    assert "CROUCH" in rep["reason"] and "always" in rep["reason"]
    assert rep["apply_with"]["tool"] == "apply_gait"
    assert not (getattr(S.get_robot(held_dog), "metadata", {}) or {}).get("gait_params")


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_never_is_the_artifact_only_contract_and_always_overrides_loudly(held_dog):
    from virturoid.services import session_state as S
    from virturoid.services.trained_controller import apply_trained_gait
    dry = apply_trained_gait(held_dog, _GOOD, door="learn_gait", apply="never", credible=True)
    assert dry["applied"] is False and dry["params"] == _GOOD
    assert not (getattr(S.get_robot(held_dog), "metadata", {}) or {}).get("gait_params")
    forced = apply_trained_gait(held_dog, _GOOD, door="learn_gait", apply="always", credible=False,
                                verdict="FELL")
    assert forced["applied"] is True and "FELL" in forced["override"]


def test_partial_or_unreachable_parameters_are_refused_not_repaired():
    """A four-key dict silently inherits the fifth from whatever the body carried, which is a controller nobody
    chose; a value outside the search box is one no search can reproduce. Both are rejected, with the reason."""
    from virturoid.services.trained_controller import normalize_gait_params
    assert normalize_gait_params({k: v for k, v in _GOOD.items() if k != "kd"})[1].startswith("missing")
    assert "outside the searchable range" in normalize_gait_params({**_GOOD, "freq": 99.0})[1]
    assert "not finite" in normalize_gait_params({**_GOOD, "kp": float("nan")})[1]
    assert normalize_gait_params(_GOOD) == (_GOOD, "")


# ---------------------------------------------------------------- the verdict has to SAY it measured the trained one

def test_gait_source_names_the_door_and_still_counts_as_fitted_to_this_body():
    """``tuned_for_this_body`` stood for four different origins. The door rides as a ``::`` suffix — and the
    imported-body honesty check must match on the base, or a controller we JUST trained for a customer's robot
    would be reframed as 'NO LOCOMOTION VERDICT — we do not have your robot's controller'."""
    from virturoid.services.ai_native_tools import _is_fitted_to_this_body
    assert _is_fitted_to_this_body("tuned_for_this_body::train_reward")
    assert _is_fitted_to_this_body("learned_policy")
    assert not _is_fitted_to_this_body("flywheel_hint")
    assert not _is_fitted_to_this_body("default_crawl")


@pytest.mark.skipif(not _MUJOCO, reason="the verdict path needs MuJoCo")
def test_verify_deploys_the_landed_parameters_and_reports_their_provenance(held_dog):
    """End to end on the surface the engineers used: land a controller, then read the verdict back through
    ``call_tool('verify_robot')`` and check it names what it ran.

    The landed parameters are the SHIPPED DEFAULT, so the deploy-select safety net (which re-runs the default
    alongside any non-shipped gait and keeps whichever is credible, further) cannot prefer the other arm — the
    two arms are the same gait. That pins the assertion to the landing rather than to whether an arbitrary
    operating point happens to beat the default on this stub body."""
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    from virturoid.services.trained_controller import apply_trained_gait
    apply_trained_gait(held_dog, dict(_DEFAULT_GAIT), door="learn_gait", credible=True,
                       evidence={"forward_m": 1.23})
    res = call_tool("verify_robot", {"robot_id": held_dog, "mode": "quick"})["result"]
    if res.get("kind") != "legged":                       # a body that routed elsewhere says nothing about this
        pytest.skip(f"verify routed to {res.get('kind')}")
    assert res["gait_source"] == "tuned_for_this_body::learn_gait", res["gait_source"]
    assert res["gait_provenance"]["door"] == "learn_gait"
    assert res["gait_provenance"]["forward_m"] == 1.23


@pytest.mark.skipif(not _MUJOCO, reason="the verdict path needs MuJoCo")
def test_a_landed_gait_worse_than_the_default_cannot_make_the_robot_verify_worse(held_dog):
    """The reason apply-by-default is safe. Verify's deploy-select re-runs the shipped default alongside any
    non-shipped gait and keeps whichever is credible (tie-break: further), so landing a controller can only
    ever help — and when the default wins, the verdict says ``default_crawl`` rather than crediting a gait it
    did not run.

    This is also the regression guard for a defect introduced WITH the door suffix: the guard that protects a
    body's own op-point from being wiped read ``gait_source != "tuned_for_this_body"`` exactly, so a trained
    body fell through it, ``gait_params`` was emptied, and the deploy-select block below (``if gait_params:``)
    was skipped entirely — the safety net silently absent on precisely the bodies that had just been trained."""
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.trained_controller import apply_trained_gait
    apply_trained_gait(held_dog, _GOOD, door="train_reward", apply="always", credible=False, verdict="FELL")
    res = call_tool("verify_robot", {"robot_id": held_dog, "mode": "quick"})["result"]
    if res.get("kind") != "legged":
        pytest.skip(f"verify routed to {res.get('kind')}")
    assert res["gait_source"] in ("default_crawl", "tuned_for_this_body::train_reward"), res["gait_source"]
    if res["gait_source"] == "default_crawl":             # the net fired: the trained arm lost, honestly named
        assert "gait_provenance" not in res, "a verdict produced by the default must not carry a training badge"


# ---------------------------------------------------------------- discoverability: the two fitters were on no wire

_FITTERS = ("learn_gait", "adapt_gait", "apply_gait")


def test_the_gait_fitters_are_registered_and_carry_an_apply_contract():
    from virturoid.services.agent_tools import TOOLS
    for name in _FITTERS:
        assert name in TOOLS, f"{name} is not in the registry"
        props = TOOLS[name]["parameters"]["properties"]
        assert "robot_id" in props
        assert "apply" in props, f"{name} accepts `apply` but does not advertise it"


@pytest.mark.skipif(not _MUJOCO, reason="the MCP registry import needs MuJoCo")
def test_the_gait_fitters_are_discoverable_in_the_tools_list_payload():
    """They dispatched fine through ``call_tool`` and were absent from ``tools/list`` AND from the server
    ``instructions``, so a customer's agent could only find them by grepping our source. ``MCP_TOOL_VIEW`` is at
    its documented cap, so they are advertised on the anchor tool — the same treatment the ingest
    importers and the authoring compilers get, and the same standard ``test_tool_registration`` holds them to.

    The budget is read from ``MCP_TOOL_VIEW_MAX`` rather than repeated as a literal here: it is a share of a
    CLIENT-side limit that gets re-argued when a capability earns a slot, and a second copy of the number in a
    test about gait discoverability turns that decision into an unrelated red."""
    from virturoid.mcp_server import _handle
    from virturoid.services.agent_tools import MCP_TOOL_VIEW_MAX
    listed = _handle("tools/list", {})["tools"]
    assert len(listed) <= MCP_TOOL_VIEW_MAX, "the lean menu must not grow past its budget"
    blob = " ".join(t["description"] for t in listed)
    for name in _FITTERS:
        assert name in blob, f"{name} dispatches but no tools/list entry names it -- undiscoverable"


def test_the_server_instructions_teach_the_fitters_and_where_results_land():
    from virturoid.mcp_server import _handle
    ins = _handle("initialize", {})["instructions"]
    for name in _FITTERS:
        assert name in ins, f"the instructions never mention {name}"
    assert "tuned_for_this_body::" in ins, "the instructions must say what verify reports after training"


@pytest.mark.skipif(not _MUJOCO, reason="the MCP registry import needs MuJoCo")
def test_apply_gait_dispatches_over_the_wire(held_dog):
    """Being in the internal registry is not the same as being on the wire. ``tools/call`` is the wire."""
    from virturoid.mcp_server import _handle
    res = _handle("tools/call", {"name": "apply_gait", "arguments": {"robot_id": held_dog, "params": _GOOD}})
    assert res["isError"] is False, res
    assert res["structuredContent"]["applied"] is True


# ---------------------------------------------------------------- the reward-as-code claim, made true

def test_train_reward_advertises_a_reward_parameter_that_shows_the_dsl():
    """The server instructions told every connected agent that ``train_reward`` is where 'you author a
    reward-as-code objective'. The tool took ``task`` (English) only, had no ``reward`` parameter, and
    ``reward_dsl.REWARD_FEATURES`` was exposed by NO tool — an agent could not have written a legal expression
    if it wanted to. The vocabulary is DERIVED from ``reward_dsl`` so it cannot drift back out of date."""
    from virturoid.services.agent_tools import TOOLS
    from virturoid.services.reward_dsl import REWARD_FEATURES
    prop = TOOLS["train_reward"]["parameters"]["properties"]["reward"]
    for feat in REWARD_FEATURES:
        assert feat in prop["description"], f"the reward vocabulary omits {feat}"


def test_the_instructions_claim_of_authoring_is_backed_by_a_real_parameter():
    """The guard that stops the claim drifting away from the tool again: if the instructions promise authoring,
    the schema must accept it."""
    from virturoid.mcp_server import _handle
    from virturoid.services.agent_tools import TOOLS
    ins = _handle("initialize", {})["instructions"]
    if "author the objective" in ins or "reward-as-code" in ins:
        assert "reward" in TOOLS["train_reward"]["parameters"]["properties"]


def test_an_authored_reward_compiles_through_the_same_sandbox_as_an_llms():
    from virturoid.services.reward_loop import compile_authored_rewards
    cands, why = compile_authored_rewards("max(0.0, forward_vel)*alive + 0.3*upright - 0.2*slip")
    assert why == "" and len(cands) == 1 and cands[0].compiled is not None
    assert cands[0].compiled({"forward_vel": 1.0, "alive": 1.0, "upright": 1.0, "slip": 0.0}) == pytest.approx(1.3)
    multi, why = compile_authored_rewards(["forward_vel", "upright"])
    assert why == "" and len(multi) == 2


def test_a_bad_authored_reward_is_refused_with_the_vocabulary_attached():
    """"Authored by the customer" buys no extra trust — and a rejection that does not name the legal vocabulary
    leaves the agent guessing at a closed allowlist."""
    from virturoid.services.reward_loop import compile_authored_rewards
    for bad in ("__import__('os').system('x')", "velocity * 2", "forward_vel ** 40"):
        cands, why = compile_authored_rewards(bad)
        assert cands == [] and why
        assert "forward_vel" in why and "did not compile" in why


@pytest.mark.skipif(not _MUJOCO, reason="the dispatcher needs MuJoCo")
def test_train_reward_rejects_a_bad_reward_before_running_any_physics(held_dog):
    import time
    from virturoid.services.agent_tools import call_tool
    t0 = time.monotonic()
    env = call_tool("train_reward", {"robot_id": held_dog, "reward": "os.system('x')"})
    assert env["ok"] is False and "did not compile" in env["error"]
    assert time.monotonic() - t0 < 5.0, "a syntax refusal must not cost a physics search"


# ---------------------------------------------------------------- the door 248afca MISSED: train_held
#
# 248afca closed this exact defect for ``train_reward``, ``learn_gait`` and ``adapt_gait``. It missed
# ``train_held`` — the tool an engineer reaches for first, the only training tool in ``MCP_TOOL_VIEW``, and the
# only one named in the MCP server ``instructions``. Re-measured on a real Menagerie Go2 through ``call_tool``:
#
#     train_held      15.0 s   status "succeeded"
#     verify_robot    0.36 m   gait_source "flywheel_hint"                       <- BYTE-IDENTICAL
#     apply_gait      0.01 s   (the same numbers train_held had just returned)
#     verify_robot    0.466 m  gait_source "tuned_for_this_body::train_held..."  <- it landed
#
# So the tests below go through ``call_tool`` and the JOB, not through ``apply_trained_gait``: a test that
# called the write side directly passed at HEAD while the product discarded the result.

_LEARNED = {"params": dict(_GOOD), "forward_m": 1.4, "default_forward_m": 0.2, "beats_default": True,
            "survived": True, "n_evals": 2, "stopped_reason": "credible_walk", "reused_prior": False,
            "banked_skill": None, "height_ratio": 0.9, "robustness_rel": 0.2}


def _stub_flywheel(monkeypatch, tmp_path, **over):
    """Replace the physics search with a canned result: the SUBJECT is where the result goes, not the search."""
    learned = {**_LEARNED, **over}
    monkeypatch.setattr("virturoid.services.gait_flywheel.learn_gait_flywheel",
                        lambda *a, **k: dict(learned))
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)
    return learned


def _await_job(job_id: str, timeout: float = 60.0) -> dict:
    import time
    from virturoid.services import job_registry as J
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view = J.get(job_id)
        if view and view["status"] in J.TERMINAL_STATUSES:
            return view
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_train_held_commits_its_controller_to_the_held_robot(held_dog, tmp_path, monkeypatch):
    """The whole finding in one test, on the engineer's surface: train through ``call_tool``, then read the
    held gene back. Before this, the gene was byte-identical here."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    _stub_flywheel(monkeypatch, tmp_path)
    before = dict((getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params") or {})
    env = call_tool("train_held", {"robot_id": held_dog, "max_evals": 2})
    assert env["ok"], env
    view = _await_job(env["result"]["job_id"])
    rep = view["result"]["applied_to_robot"]
    assert rep["applied"] is True, rep
    assert rep["gait_source_after"] == "tuned_for_this_body::train_held"
    md = getattr(S.get_robot(held_dog), "metadata", None) or {}
    assert md["gait_params"] == _GOOD and md["gait_params"] != before
    assert md["gait_provenance"]["door"] == "train_held"
    assert view["status"] == "succeeded"


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_train_held_is_an_edit_so_undo_gives_the_previous_controller_back(held_dog, tmp_path, monkeypatch):
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    _stub_flywheel(monkeypatch, tmp_path)
    _await_job(call_tool("train_held", {"robot_id": held_dog})["result"]["job_id"])
    assert (getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params") == _GOOD
    assert call_tool("edit_robot", {"robot_id": held_dog, "ops": [{"op": "undo"}]})["ok"]
    assert not (getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params")


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_a_train_held_run_that_changed_nothing_does_not_finish_succeeded(held_dog, tmp_path, monkeypatch):
    """``status: succeeded`` on top of a byte-identical robot is the defect, not just a symptom of it. A
    non-credible run keeps the old controller (correct) — and must SAY so at the job level, where the engineer
    was reading a green chip."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    _stub_flywheel(monkeypatch, tmp_path, survived=False, beats_default=False, stopped_reason="CROUCH")
    view = _await_job(call_tool("train_held", {"robot_id": held_dog})["result"]["job_id"])
    rep = view["result"]["applied_to_robot"]
    assert rep["applied"] is False and "CROUCH" in rep["reason"]
    assert rep["apply_with"]["tool"] == "apply_gait"          # ...and how to land it anyway
    assert view["status"] == "no_output", "a train job that landed nothing must not read SUCCEEDED"
    assert not (getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params")


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_train_held_honours_never_and_always_like_every_other_door(held_dog, tmp_path, monkeypatch):
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    _stub_flywheel(monkeypatch, tmp_path, survived=False, beats_default=False, stopped_reason="FELL")
    dry = _await_job(call_tool("train_held", {"robot_id": held_dog, "apply": "never"})["result"]["job_id"])
    assert dry["result"]["applied_to_robot"]["params"] == _GOOD
    assert dry["status"] == "succeeded", "apply='never' asked for an artifact and got one — that is a success"
    assert not (getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params")
    forced = _await_job(call_tool("train_held", {"robot_id": held_dog, "apply": "always"})["result"]["job_id"])
    assert forced["result"]["applied_to_robot"]["applied"] is True
    assert "FELL" in forced["result"]["applied_to_robot"]["override"]
    assert (getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params") == _GOOD


def test_a_train_job_whose_worker_refused_is_not_a_success():
    """``run_train_gene_job`` returns ``{"error": ...}`` for a robot that is not held; the job used to store
    that dict and finish SUCCEEDED, so a client checking the status could not see the refusal at all."""
    from virturoid.services import job_registry as J
    job = {"kind": "train_gene", "error": None, "result": {"error": "no robot 'nope' held"}}
    assert J._terminal_status(job) == J.FAILED
    assert job["error"]


@pytest.mark.skipif(not _MUJOCO, reason="the GPU arm still composes a gene")
def test_the_gpu_arm_banks_its_policy_or_says_it_did_not(held_dog, tmp_path, monkeypatch):
    """The gpu_rl arm's artifact is a neural policy, not five scalars, so it lands in the POLICY bank — which
    ``verify_robot`` consults. Nothing in ANY agent path called ``bank_morph_policy`` (the repo's only caller
    was ``desktop.py``), so a GPU run wrote an .npz into a build dir no verdict path reads and reported
    ``trained: true``. Same defect, one artifact type over."""
    from virturoid.services.agent_design_tools import run_train_gene_job
    monkeypatch.setattr("virturoid.services.gpu_trainer.gpu_available", lambda **k: True)
    monkeypatch.setattr("virturoid.services.gpu_trainer.train_gene_on_gpu",
                        lambda *a, **k: str(tmp_path / "policy.npz"))
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)
    monkeypatch.setattr("virturoid.services.policy_flywheel.bank_morph_policy",
                        lambda *a, **k: {"banked": True, "skill_id": "morph_legged_locomotion",
                                         "verdict": "CREDIBLE WALK", "forward": 1.1})
    out = run_train_gene_job({"robot_id": held_dog, "mode": "gpu_rl"})
    assert out["applied_to_robot"]["applied"] is True
    assert out["applied_to_robot"]["skill_id"] == "morph_legged_locomotion"

    monkeypatch.setattr("virturoid.services.policy_flywheel.bank_morph_policy",
                        lambda *a, **k: {"banked": False, "skill_id": None, "verdict": "FELL", "forward": 0.1})
    refused = run_train_gene_job({"robot_id": held_dog, "mode": "gpu_rl"})
    assert refused["trained"] is True                          # the artifact exists...
    assert refused["applied_to_robot"]["applied"] is False     # ...and the report does not pretend it deployed
    assert "FELL" in refused["applied_to_robot"]["reason"]


# ---------------------------------------------------------------- the sweep: every controller door, one rule

#: tool -> the function that must carry the apply contract. A door that fits or improves a controller for a
#: HELD robot and does not appear here is the next ``train_held``: two of three were wired in 248afca and the
#: third was not noticed for two weeks. ``design_search`` is deliberately absent — it searches a body composed
#: from a prompt and holds nothing — and is covered by the honesty test below instead.
_CONTROLLER_DOORS = {
    "train_reward": "virturoid.services.reward_loop._land_on_robot",
    "learn_gait": "virturoid.services.input_training_tools._learn_gait",
    "adapt_gait": "virturoid.services.gait_hints._adapt_gait",
    "apply_gait": "virturoid.services.gait_hints._apply_gait",
    "train_held": "virturoid.services.agent_design_tools.run_train_gene_job",
    "adopt_control_script": "virturoid.services.input_training_tools._adopt_control_script",
}


@pytest.mark.parametrize("tool,target", sorted(_CONTROLLER_DOORS.items()))
def test_every_controller_door_is_registered_and_wired_to_the_apply_contract(tool, target):
    import importlib
    import inspect
    from virturoid.services.agent_tools import TOOLS
    assert tool in TOOLS, f"{tool} is not registered — an unreachable door is the same as a missing one"
    assert "apply" in TOOLS[tool]["parameters"]["properties"], f"{tool} does not advertise its apply contract"
    mod, fn = target.rsplit(".", 1)
    src = inspect.getsource(getattr(importlib.import_module(mod), fn))
    assert "apply_trained_gait" in src, f"{target} produces a controller and never lands one"


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_adopt_control_script_lands_the_improvement_it_claims(held_dog, monkeypatch):
    """Found by the same sweep. It fits a controller to the held body WARM-STARTED FROM THE CUSTOMER'S OWN
    parameters, said "improved the user's controller", and wrote nothing — so the next verify_robot re-measured
    what the robot had before. The gate is its own honest bar (``beat_imported``): an improvement that failed it
    must never overwrite what the customer shipped."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    canned = {"utilised": {"forward_m": 0.34, "credible": False},
              "improved": {"forward_m": 0.62, "credible": True},
              "imported_params": dict(_GOOD), "improved_params": dict(_GOOD), "beat_imported": True,
              "verdict": "improved the user's controller (credible walk, further travel)"}
    monkeypatch.setattr("virturoid.services.control_adopter.adopt_control_script", lambda *a, **k: dict(canned))
    res = call_tool("adopt_control_script", {"robot_id": held_dog, "params": {"freq": 1.0}})["result"]
    assert res["applied_to_robot"]["applied"] is True
    assert (getattr(S.get_robot(held_dog), "metadata", None) or {}).get("gait_params") == _GOOD

    monkeypatch.setattr("virturoid.services.control_adopter.adopt_control_script",
                        lambda *a, **k: {**canned, "beat_imported": False,
                                         "verdict": "ran the user's controller; kept it"})
    kept = call_tool("adopt_control_script", {"robot_id": held_dog, "params": {"freq": 1.0}})["result"]
    assert kept["applied_to_robot"]["applied"] is False and "kept it" in kept["applied_to_robot"]["reason"]


def test_a_controller_search_with_no_robot_to_land_on_says_so():
    """Rule 3 of the sweep: a tool that produces something and cannot apply it must SAY so, not report bare
    success. ``design_search`` composes a body from a prompt and holds nothing, which is a fine contract — but
    only when stated. Asserted on the source because running it costs a real physics search."""
    import inspect
    from virturoid.services.agent_tools import _design_search
    src = inspect.getsource(_design_search)
    assert "applied_to_robot" in src and "applied to nothing" in src


# =========================================================================================================
# THE SECOND SWEEP (2026-08-09). The first found four doors that discarded a controller. This one asks the
# adjacent question — does any tool's TEXT promise an effect its code does not produce? — and found two.

_CANNED_REWARD_RUN = {
    "ok": True, "reward_source": "templates", "reward_name": "template_0", "reward_expr": "forward_vel",
    "n_candidates": 1, "n_gamed": 0, "ranked": [], "verdict": "CREDIBLE WALK", "credible": True,
    "forward_m": 1.2, "height_ratio": 0.9, "gait_params": dict(_GOOD), "reflection": {},
    "iterations_run": 1, "iteration_log": [], "reward_hints_recalled": 0, "seeded_from_flywheel": False,
}


def _stub_reward_run(monkeypatch, npz, **over):
    """Replace the whole reward loop with a canned CREDIBLE result that also 'trained' a GPU policy.

    The subject is the SECOND artifact's channel, not the reward search — which costs hundreds of rollouts and
    a GPU box. ``run_intelligent_reward_loop`` is called as a module global by ``_train_reward``, so this is
    still the real tool, entered through ``call_tool``.
    """
    out = {**_CANNED_REWARD_RUN, **over,
           "gpu_training": {"attempted": True, "trained": bool(npz), "policy": npz, "backend": "mjx_ppo_gpu"}}
    monkeypatch.setattr("virturoid.services.reward_loop.run_intelligent_reward_loop", lambda *a, **k: dict(out))
    return out


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_train_reward_gpu_arm_banks_its_policy_or_says_it_did_not(held_dog, tmp_path, monkeypatch):
    """``train_reward`` told the agent its GPU ``.npz`` "deploys through the POLICY bank (verify recalls it)".

    It did not. No agent-reachable path anywhere called ``bank_morph_policy`` — the repo's only caller was
    ``desktop.py:480`` — so ``recall_morph_policy`` could never find the file and the GPU training the customer
    paid minutes for reached nothing. The sentence was the whole implementation. It now goes through
    ``policy_flywheel.land_gpu_policy``, the same helper ``train_held``'s GPU arm uses, and the bank's own
    credible-rollout screen on THIS body is the gate.
    """
    from virturoid.services.agent_tools import call_tool
    _stub_reward_run(monkeypatch, str(tmp_path / "reward_policy.npz"))
    monkeypatch.setattr("virturoid.services.policy_flywheel.bank_morph_policy",
                        lambda *a, **k: {"banked": True, "skill_id": "morph_legged_locomotion",
                                         "verdict": "CREDIBLE WALK", "forward": 1.4})
    res = call_tool("train_reward", {"robot_id": held_dog, "train_backend": "gpu"})["result"]
    pol = res["applied_to_robot"]["policy"]
    assert pol["applied"] is True and pol["skill_id"] == "morph_legged_locomotion"
    assert pol["channel"] == "policy_bank" and pol["door"] == "train_reward"
    assert pol["gait_source_after"] == "learned_policy"

    monkeypatch.setattr("virturoid.services.policy_flywheel.bank_morph_policy",
                        lambda *a, **k: {"banked": False, "skill_id": None, "verdict": "FELL", "forward": 0.1})
    refused = call_tool("train_reward", {"robot_id": held_dog, "train_backend": "gpu"})["result"]
    assert refused["gpu_training"]["trained"] is True         # the artifact exists...
    assert refused["applied_to_robot"]["policy"]["applied"] is False   # ...and nothing pretends it deployed
    assert "FELL" in refused["applied_to_robot"]["policy"]["reason"]


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_the_policy_channel_honours_never_and_cannot_be_forced_by_always(held_dog, tmp_path, monkeypatch):
    """The two apply modes the policy channel does NOT treat like the gait channel, both said out loud.

    ``never`` must touch neither the robot nor the bank. ``always`` must NOT override the bank's screen: a
    MorphPolicy is keyed by morphology KIND, so admitting an uncredible one would hand it to every other legged
    build on recall. ``train_reward``'s description promises 'always commits it regardless' for the gait; the
    refusal has to say why that promise stops at this channel instead of quietly not applying.
    """
    from virturoid.services.agent_tools import call_tool
    _stub_reward_run(monkeypatch, str(tmp_path / "reward_policy.npz"))
    banked = []
    monkeypatch.setattr("virturoid.services.policy_flywheel.bank_morph_policy",
                        lambda *a, **k: banked.append(a) or {"banked": False, "verdict": "CROUCH",
                                                             "skill_id": None, "forward": 0.0})
    dry = call_tool("train_reward", {"robot_id": held_dog, "train_backend": "gpu", "apply": "never"})["result"]
    assert dry["applied_to_robot"]["policy"]["applied"] is False
    assert "apply='never'" in dry["applied_to_robot"]["policy"]["reason"]
    assert banked == [], "apply='never' must not even offer the policy to the bank"

    forced = call_tool("train_reward", {"robot_id": held_dog, "train_backend": "gpu", "apply": "always"})["result"]
    reason = forced["applied_to_robot"]["policy"]["reason"]
    assert forced["applied_to_robot"]["policy"]["applied"] is False and "CROUCH" in reason
    assert "does not override" in reason and "recalled by every other legged build" in reason


#: The GPU doors and the function that must route their neural artifact into the one channel that deploys it.
#: Both were written independently and both shipped the same hole; a third GPU arm that lands its ``.npz``
#: anywhere else — or nowhere, with a sentence — is the next one of these.
_POLICY_DOORS = {
    "train_reward": "virturoid.services.reward_loop._land_on_robot",
    "train_held": "virturoid.services.agent_design_tools._land_gpu_policy",
}


@pytest.mark.parametrize("tool,target", sorted(_POLICY_DOORS.items()))
def test_every_gpu_door_routes_its_policy_through_the_bank_that_deploys_it(tool, target):
    import importlib
    import inspect
    from virturoid.services.agent_tools import TOOLS
    assert tool in TOOLS
    mod, fn = target.rsplit(".", 1)
    src = inspect.getsource(getattr(importlib.import_module(mod), fn))
    assert "land_gpu_policy" in src, f"{target} produces a neural policy and never lands one"


def test_the_only_caller_of_the_policy_bank_is_no_longer_the_desktop_app():
    """The shape of the original defect, as a standing assertion. ``bank_morph_policy`` existed, was tested, and
    had exactly one caller in the whole repo — ``desktop.py`` — which no agent, MCP client or tool can reach. A
    write path with no agent-reachable caller is indistinguishable from no write path at all."""
    import inspect
    from virturoid.services import policy_flywheel as PF
    assert "bank_morph_policy" in inspect.getsource(PF.land_gpu_policy)


# ---------------------------------------------------------------- a tool's NAME is a claim too

def test_import_onnx_policy_says_plainly_that_it_does_not_deploy(tmp_path):
    """``import_onnx_policy`` never claimed to deploy — the word "import" did, and "bring your own trained
    policy" is a stated priority use case. Nothing in the repo can make an imported ONNX policy a held robot's
    controller: the bank takes MorphPolicy ``.npz``, and an ``.onnx`` declares tensor names/shapes only — not
    the observation layout, the joint order, or whether the outputs are torques or position targets. So the
    tool says so, in the result and in its description, and names the doors that DO deploy.
    """
    from virturoid.services.agent_tools import TOOLS, call_tool
    p = tmp_path / "customer_policy.onnx"
    p.write_bytes(b"not-a-real-onnx-graph")                   # the disclosure must not depend on a valid model
    res = call_tool("import_onnx_policy", {"path": str(p)})["result"]
    assert res["deployed"] is False
    dep = res["deployment"]
    assert dep["deployable"] is False
    assert "does not" in dep["what_this_tool_does"]
    assert "our generic scripted controller" in dep["verify_robot_still_measures"]
    routes = " ".join(f"{r['if']} {r['do']}" for r in dep["instead"])
    for door in ("train_held", "train_reward", "adopt_control_script", "import_controller_interface"):
        assert door in routes, f"{door} not named as what the customer should do instead"
    desc = TOOLS["import_onnx_policy"]["description"]
    assert "DOES NOT DEPLOY" in desc and "INSPECT and VALIDATE" in desc


@pytest.mark.skipif(not _MUJOCO, reason="counting actuators compiles the gene")
def test_import_onnx_policy_validates_the_action_width_against_the_customers_own_robot(held_dog, tmp_path):
    """What we CAN do for a customer who brings their own policy, made concrete: ``robot_id`` turns "the action
    is dimension-correct" from a statement about an abstract vector into one about THEIR robot's actuator
    count, and a contradicting ``action_dim`` is refused rather than silently preferred."""
    from virturoid.services.agent_tools import call_tool
    p = tmp_path / "customer_policy.onnx"
    p.write_bytes(b"not-a-real-onnx-graph")
    res = call_tool("import_onnx_policy", {"path": str(p), "robot_id": held_dog})["result"]
    assert res["action_dim_checked_against"]["robot_id"] == held_dog
    assert res["action_dim_checked_against"]["n_actuators"] > 0
    clash = call_tool("import_onnx_policy", {"path": str(p), "robot_id": held_dog, "action_dim": 999})
    assert "contradicts" in (clash["result"].get("error") or clash.get("error") or "")
    missing = call_tool("import_onnx_policy", {"path": str(p), "robot_id": "no_such_robot"})
    assert "no held robot" in (missing["result"].get("error") or missing.get("error") or "")


def test_no_tool_RESULT_offers_a_route_to_a_tool_that_does_not_exist():
    """``test_tool_registration.test_no_agent_facing_text_names_a_tool_that_does_not_exist`` already checks the
    MCP handshake, the prompts, the design schema and every tool DESCRIPTION. Its corpus stops there — and the
    defect had moved one step downstream, into what a tool RETURNS. ``verify_robot`` on an imported body told
    the customer to use ``import_control_script``, which is not a registered tool and never was (the real one is
    ``adopt_control_script``); the same route also promised that importing an ONNX policy and re-verifying would
    make the verdict describe their controller, which no code anywhere can do. An agent reads a result far more
    often than it re-reads a description, so results are held to the same bar here.

    Same identifier rule as the registry guard, including its ``(?!\\s*=)`` skip so ``mode='gpu_rl'`` reads as
    the keyword argument it is. STRING LITERALS ONLY: a source scan would also flag the comment that explains
    this very defect.
    """
    import ast
    import inspect
    import re
    import textwrap
    from virturoid.services import ai_native_tools as AIT
    from virturoid.services.agent_tools import TOOLS
    from virturoid.services.policy_native_adapter import DEPLOYMENT

    def _literals(fn) -> str:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        doc = ast.get_docstring(tree.body[0])
        return " ".join(n.value for n in ast.walk(tree)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value is not doc)

    def _values(obj) -> str:
        """The prose a customer READS. Field NAMES are a separate namespace (``verify_robot_still_measures`` is
        a key, not a route), so only values are scanned — the same distinction ``_NOT_A_TOOL`` draws."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return " ".join(_values(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return " ".join(_values(v) for v in obj)
        return ""

    verbs = {"import", "adopt", "train", "apply", "adapt", "verify", "export", "render", "probe", "sandbox",
             "learn", "generate", "ingest", "inspect", "amplify", "calibrate", "measure"}
    ident = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b(?!\s*=)")
    results = {"verify_robot: the imported-body routes": _literals(AIT._reframe_for_imported_body),
               "import_onnx_policy: the deployment disclosure": _values(DEPLOYMENT)}
    bad: dict[str, set[str]] = {}
    for where, blob in results.items():
        for name in set(ident.findall(blob)):
            if name not in TOOLS and name.split("_")[0] in verbs:
                bad.setdefault(name, set()).add(where)
    assert not bad, ("a tool RESULT routes the customer to something that does not dispatch: "
                     + "; ".join(f"{k} (in {sorted(v)})" for k, v in sorted(bad.items())))


@pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")
def test_landing_over_a_controller_that_was_already_fitted_says_what_it_replaced(held_dog):
    """The module claimed a landing "can never make a robot verify WORSE than it did before". Measured through
    ``call_tool`` on an authored horse, it can: verify 3.305 m CREDIBLE WALK -> train_held (auto, gated on
    beats_default over a 600-step deploy horizon) -> verify 2.107 m FELL by YAW-DRIFT -> undo -> 3.305 m again.
    Verify's deploy-select guards a landing against the SHIPPED DEFAULT, not against whatever the body was
    already carrying. So when there IS a previous fitted controller, the report has to name it."""
    from virturoid.services.trained_controller import apply_trained_gait
    first = apply_trained_gait(held_dog, _GOOD, door="learn_gait", credible=True)
    assert "replaced" not in first, "nothing was replaced on a body with no controller of its own"
    other = {**_GOOD, "freq": 1.8}
    second = apply_trained_gait(held_dog, other, door="train_held", credible=True)
    assert second["applied"] is True
    assert second["previous_params"] == _GOOD
    assert "undo" in second["replaced"] and "SHIPPED DEFAULT" in second["replaced"]
