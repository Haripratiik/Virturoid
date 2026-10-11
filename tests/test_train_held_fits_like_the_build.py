"""``train_held`` fits a gait the way the product's build path does.

THE DEFECT, measured 2026-10-10 through the MCP server (``ingest_project`` on the Menagerie Unitree Go2, then
``train_held`` with mode 'gait_search', apply 'auto'): the default ``max_evals=8`` searched 8 configs in ~10 s and
``max_evals=96`` searched 96 in ~60 s, both ending LURCHES with nothing applied and the job 'no_output', while
``gait_flywheel.fit_gait_for_body`` on the same twin adopts a CREDIBLE WALK. The door scored and judged every
candidate over 600 steps, and ``gait_quality.classify`` asks a walk for 0.3 m, so nothing slower than 0.25 m/s
could ever be certified there; the Go2 walks at ~0.13 m/s.

AFTER, same door, same body: the job runs the product's per-body fit and lands what it adopts (measured through
MCP: 49 candidates over 1 seed attempt, 61 s of fitting, CREDIBLE WALK +1.542 m, and the next verify_robot
reports ``tuned_for_this_body::train_held`` with the rate held). These tests hold the contract the fix created,
cheaply, and one of them runs that Go2 end to end where the Menagerie is installed:

  * the door runs the fit, at the fit's own judging horizon, on a COPY of the held gene, with both budgets;
  * ``max_evals`` caps the WHOLE run and a run it stops says it did not finish (real physics, 3 candidates);
  * a body the leg crawl never drives is not searched and nothing lands (real physics, zero rollouts);
  * a body whose shipped default already walks is not re-fitted, and the report says so;
  * get_job events carry the fit's own up-front cost statement and a heartbeat while it runs;
  * a cancel requested while the fit runs lands nothing;
  * and the defect itself, end to end: the ingested Go2 gets its walk through the agent's doors (real physics,
    about three minutes, so it runs only where the Menagerie is installed).
"""
from __future__ import annotations

import importlib.util
import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composing and fitting a body needs MuJoCo")

_GOOD = {"freq": 2.0, "hip_amp": 0.7, "knee_amp": 1.1, "kp": 90.0, "kd": 4.0}
_GO2_DIR = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2"))

#: A fit that searched and adopted nothing. The door's reads of a real disclosure are covered by
#: ``test_training_lands_on_robot``'s canned fit; this one only has to be a well-formed decline.
_DECLINE = {"ok": True, "searched": True, "adopted": False, "params": None, "searched_params": None,
            "n_evals": 2, "n_rollouts": 3, "seed_attempts": 1, "horizon_steps": 6000,
            "default_verdict": "FELL by ROLL-OVER", "default_forward_m": 0.1, "verdict": "FELL by ROLL-OVER",
            "reason": "searched 2 gaits (3 physics rollouts) for this body over 1 seed attempt(s); none was still "
                      "walking at the 6000-step horizon"}


@pytest.fixture()
def fresh_bank(tmp_path, monkeypatch):
    """Point the door's memory (and its job workspace) at an empty directory.

    The fit RECALLS a structural prior from the bank before it searches, so a row another test banked for a
    similar body would steer these searches and turn a measured outcome into an order-dependent one."""
    monkeypatch.setattr("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: tmp_path)
    return tmp_path


def _hold(prompt: str) -> str:
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    return S.put_robot(compose_robot(prompt, llm=None), prompt=prompt)


def _gait_params(rid: str) -> dict:
    from virturoid.services import session_state as S
    return dict((getattr(S.get_robot(rid), "metadata", None) or {}).get("gait_params") or {})


def _start(args: dict) -> tuple[dict, str]:
    from virturoid.services.agent_tools import call_tool
    env = call_tool("train_held", args)
    assert env["ok"], env
    return env["result"], env["result"]["job_id"]


def _finish(job_id: str, timeout: float = 300.0) -> tuple[dict, list[dict]]:
    from virturoid.services import job_registry as J
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        view, events = J.events_since(job_id, 0)
        if view["status"] in J.TERMINAL_STATUSES:
            return view, events
        time.sleep(0.1)
    raise AssertionError(f"train_held job {job_id} never finished")


def test_the_door_runs_the_products_fit_on_a_copy_with_both_budgets(fresh_bank, monkeypatch):
    """The wiring, with the physics stubbed: WHICH fit runs, on WHAT, under which bounds.

    On a COPY because ``fit_gait_for_body`` writes its result onto the gene it is handed and
    ``session_state.get_robot`` hands back the live held gene: fitting that in place would change the customer's
    robot with no undo entry and before any verdict was read. The stub writes ``gait_params`` onto whatever it
    is given, exactly as the real fit does on adoption, and the held robot must not see it.
    """
    from virturoid.services import build_progress as P
    from virturoid.services import session_state as S
    from virturoid.services.gait_flywheel import _SETTLE_STEPS
    rid = _hold("a four legged robot dog")
    held = S.get_robot(rid)
    seen: dict = {}

    def fake_fit(gene, **kw):
        rep = P.current()
        seen.update(gene=gene, kw=kw, evals_budget=getattr(rep, "evals_budget", None),
                    budget_s=getattr(rep, "budget_s", None))
        gene.metadata = {**(gene.metadata or {}), "gait_params": dict(_GOOD)}
        return dict(_DECLINE)

    monkeypatch.setattr("virturoid.services.gait_flywheel.fit_gait_for_body", fake_fit)
    call, jid = _start({"robot_id": rid, "max_evals": 5, "budget_s": 7, "seed": 3})
    view, _ = _finish(jid)
    assert call["max_evals"] == 5 and call["budget_s"] == 7.0, call
    assert seen, "the door never ran the per-body fit"
    assert seen["gene"] is not held and seen["gene"] is not S.get_robot(rid), "the fit ran on the held gene itself"
    kw = seen["kw"]
    assert kw.get("deploy_steps", _SETTLE_STEPS) == _SETTLE_STEPS == kw.get("search_steps", _SETTLE_STEPS), (
        "the door must not shorten the horizon the fit searches and judges at")
    assert kw.get("seed") == 3 and kw.get("bank") is True and kw.get("db") is not None, kw
    assert seen["evals_budget"] == 5, "max_evals must reach the fit as the whole run's evaluation ledger"
    assert seen["budget_s"] == 7.0, "budget_s must reach the fit as its clock"
    assert not _gait_params(rid), "the fit's write to its own copy leaked onto the held robot"
    assert view["status"] == "no_output" and view["result"]["applied_to_robot"]["applied"] is False


def test_malformed_budgets_are_refused_before_a_job_exists(fresh_bank):
    """A cap the door cannot read must not quietly become the full search (minutes of physics nobody asked
    for) or no search at all; and it must fail NOW, at the call, not minutes later as a failed job."""
    from virturoid.services import job_registry as J
    from virturoid.services.agent_tools import call_tool
    rid = _hold("a four legged robot dog")
    before = len(J.list_jobs())
    for bad in ({"max_evals": "lots"}, {"budget_s": "soon"}, {"budget_s": float("inf")}, {"seed": "x"}):
        env = call_tool("train_held", {"robot_id": rid, **bad})
        assert env["ok"] is False, (bad, env)
        assert any(k in env["error"] for k in bad), (bad, env["error"])
    assert len(J.list_jobs()) == before, "a refused call must not have started a job"


def test_max_evals_caps_the_whole_run_and_a_capped_run_says_it_did_not_finish(fresh_bank):
    """Real physics, three candidates. ``max_evals`` used to be the CEM's own count for a 600-step search; it is
    now the cap on gait candidates for the WHOLE run (warm pass, cold re-run and every seed restart together),
    and a search it stopped is a search that did not finish, never a finding about the body.

    MEASURED on this body (composed "a four legged robot dog", empty bank, seed 0): the shipped default rolls
    over at step ~430 of 6000, so the fit searches; three candidates do not find a walk, and the run stops at
    the cap with 7 physics rollouts in about 5 s.
    """
    rid = _hold("a four legged robot dog")
    call, jid = _start({"robot_id": rid, "max_evals": 3})
    view, _ = _finish(jid)
    r = view["result"]
    a = r["applied_to_robot"]
    assert call["max_evals"] == 3
    assert r["judged_at_steps"] == 6000, "candidates are judged at the horizon a walk is judged at"
    assert not str(r["default_verdict"]).startswith("CREDIBLE"), "this body's default must need a search"
    assert r["fit"]["searched"] is True and 1 <= r["n_evals"] <= 3, r["n_evals"]
    assert r["credible"] is False, "measured: three candidates do not find this body's walk"
    assert r["fit"]["stopped_by_eval_budget"] is True
    assert "max_evals=3" in a["reason"] and "NOTHING HERE SAYS THIS BODY CANNOT WALK" in a["reason"], a["reason"]
    assert a["applied"] is False and a["apply_with"]["tool"] == "apply_gait"
    assert view["status"] == "no_output"
    assert not _gait_params(rid)


def test_a_body_the_leg_crawl_never_drives_is_not_searched_and_lands_nothing(fresh_bank):
    """A body that cannot walk must report honestly and land nothing.

    MEASURED BEFORE through MCP on a composed "a 6-DOF robot arm": the old door searched 8 leg-crawl configs
    on the ARM in ~5 s and answered "did not produce a credible walk (SLIDE ...). Re-run with more budget", a
    decline-shaped sentence about a body no leg gait ever drives, inviting the agent to spend more. The fit
    asks first whether the controller it searches ever reaches this body, and refuses without a rollout.
    """
    rid = _hold("a 6-DOF robot arm")
    _, jid = _start({"robot_id": rid})
    view, _ = _finish(jid)
    r = view["result"]
    a = r["applied_to_robot"]
    assert r["n_evals"] == 0 and r["fit"].get("not_applicable") is True, r["fit"]
    assert a["applied"] is False and a["params"] is None
    assert "NO SEARCH WAS RUN" in a["reason"] and "did not produce a credible walk" not in a["reason"], a["reason"]
    assert "apply_with" not in a, "there are no parameters, so there is nothing to force onto the robot"
    assert view["status"] == "no_output"
    assert not _gait_params(rid)


def test_a_body_whose_default_already_walks_is_not_refit_and_says_so(fresh_bank):
    """The product's fit never churns a body that already walks, and neither does the door.

    The schema's quadruped example measures CREDIBLE WALK +7.246 m on the shipped default at the 6000-step
    horizon, so the fit stops after that one rollout. The report must say THAT, not "did not produce a
    credible walk", which would read as a failure on a robot that walks.
    """
    from virturoid.services.agent_design_tools import get_design_schema, submit_design
    rid = submit_design({"graph": get_design_schema({})["examples"]["quadruped"]})["robot_id"]
    _, jid = _start({"robot_id": rid, "max_evals": 3})
    view, _ = _finish(jid)
    r = view["result"]
    a = r["applied_to_robot"]
    assert str(r["default_verdict"]).startswith("CREDIBLE"), r["default_verdict"]
    assert r["n_evals"] == 0 and r["fit"]["searched"] is False
    assert a["applied"] is False and "already a CREDIBLE WALK" in a["reason"], a["reason"]
    assert "did not produce a credible walk" not in a["reason"]
    assert r["not_banked_reason"] and "already walks" in r["not_banked_reason"]
    assert not _gait_params(rid)


def test_get_job_events_carry_the_fits_cost_statement_and_a_heartbeat(fresh_bank, monkeypatch):
    """The fit takes as long as the body needs (a minute or more on the Go2), and a long job that says nothing is
    indistinguishable from a hang. Its own up-front statement of what one rollout costs on THIS body must reach
    the agent while it is still useful, and the clock must visibly move until the fit returns."""
    from virturoid.services import build_progress as P
    monkeypatch.setattr("virturoid.services.agent_design_tools._TRAIN_HEARTBEAT_S", 0.5)
    note = "fitting an operating point to this body: one 6000-step rollout costs 0.7s here (canned)"

    def slow_fit(gene, **kw):
        P.say(note)
        time.sleep(2.5)
        return dict(_DECLINE)

    monkeypatch.setattr("virturoid.services.gait_flywheel.fit_gait_for_body", slow_fit)
    rid = _hold("a four legged robot dog")
    _, jid = _start({"robot_id": rid, "budget_s": 60})
    _, events = _finish(jid)
    msgs = [e["message"] for e in events]
    done = next(i for i, m in enumerate(msgs) if m.startswith("fit finished"))
    assert note in msgs[:done], f"the fit's cost statement must arrive before it finishes: {msgs}"
    beats = [m for m in msgs[:done] if m.startswith("still fitting")]
    assert beats, f"no heartbeat while the fit ran: {msgs}"
    assert "60 s search budget" in beats[-1], beats[-1]


def test_a_cancel_requested_while_the_fit_runs_lands_nothing(fresh_bank, monkeypatch):
    """Cancellation is cooperative at the job's emits, and the fit cannot be interrupted mid-rollout. So the
    cancel takes effect at the first emit after the fit returns, BEFORE anything lands: here the fit comes back
    with a credible walk that ``apply='auto'`` would have committed, and the robot must not receive it."""
    from virturoid.services import job_registry as J
    started = threading.Event()
    adopted = {**_DECLINE, "adopted": True, "params": dict(_GOOD), "searched_params": dict(_GOOD),
               "confirm_verdict": "CREDIBLE WALK", "confirm_forward_m": 1.4, "beats_default": True,
               "verdict": "CREDIBLE WALK", "reason": "adopted one that is still walking at step 6000"}

    def slow_walk(gene, **kw):
        started.set()
        time.sleep(1.5)
        return dict(adopted)

    monkeypatch.setattr("virturoid.services.gait_flywheel.fit_gait_for_body", slow_walk)
    rid = _hold("a four legged robot dog")
    _, jid = _start({"robot_id": rid})
    assert started.wait(60), "the fit never started"
    assert J.request_cancel(jid)
    view, _ = _finish(jid)
    assert view["status"] == "cancelled", view["status"]
    assert not _gait_params(rid), "a cancelled job landed its controller"


@pytest.mark.skipif(not (_GO2_DIR / "go2.xml").is_file(), reason=f"needs the MuJoCo Menagerie Go2 at {_GO2_DIR}")
def test_an_ingested_go2_gets_the_walk_its_physics_supports(fresh_bank):
    """THE DEFECT ITSELF, end to end through the agent's doors: ingest the customer's Go2, train it, verify it.

    Before 2026-10-10 this ended LURCHES with nothing landed, at max_evals 8 and 96 alike. It is real physics
    with the clock switched off (``budget_s=0``): on a loaded machine the 180 s default can stop the search before
    the candidate it adopts (its fit took 61 s through MCP, and three times that beside other runs), and a test
    that depends on the machine's speed is a test of the machine. The search is seeded, so what it adopts is not.
    MEASURED through MCP: 49 candidates, CREDIBLE WALK +1.542 m, landed as ``tuned_for_this_body::train_held``.
    Since the crawl's swing lifts the Go2's feet instead of pressing them (2026-10-11) its shipped default walks
    it 5.66 m, so the fit searches nothing; the door then lands that measured point as the robot's own, because
    verify will not certify our generic default on an imported body. Without that landing this ended no_output
    with verify still saying NO LOCOMOTION VERDICT. The distance bound is loose on purpose: it rules out a
    degenerate pass at the 0.3 m a credible walk needs, not a particular operating point.
    """
    from virturoid.services.agent_tools import call_tool
    env = call_tool("ingest_project", {"project_path": str(_GO2_DIR), "description": "a Unitree Go2 quadruped"})
    assert env["ok"], env
    rid = env["result"]["robot_id"]
    _, jid = _start({"robot_id": rid, "budget_s": 0})
    view, _ = _finish(jid, timeout=1200.0)
    r = view["result"]
    assert view["status"] == "succeeded", (view["status"], r.get("credible_reason"))
    assert r["credible"] is True and r["applied_to_robot"]["applied"] is True, r["applied_to_robot"]
    assert r["forward_m"] >= 1.0, r["forward_m"]
    ver = call_tool("verify_robot", {"robot_id": rid})
    assert ver["ok"], ver
    v = ver["result"]
    assert str(v["verdict"]).startswith("CREDIBLE WALK"), v["verdict"]
    assert v["gait_source"] == "tuned_for_this_body::train_held", v["gait_source"]
