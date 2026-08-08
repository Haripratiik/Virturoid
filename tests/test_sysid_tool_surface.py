"""The sim-to-real gap had no door, and a capability nobody can reach is a capability nobody has.

`services/sysid/` (3,156 lines) and `services/sim2real.py` shipped complete, tested and 0% agent-reachable. Four
engineers used the product, searched `list_tools` (67 tools), `capabilities`, the README and the Studio, and
found nothing; `grep -rn "import sim2real" src/` returned nothing; the single README mention sat under
`## Roadmap`, telling a prospect the feature did not exist. The only way in was `import
virturoid.services.sysid`, and their verdict once they got there was "the gap report is the best artifact in
the product".

So EVERY test in this file goes through `agent_tools.call_tool`, and `test_this_file_never_imports_the_package`
enforces it by reading this module's own source. That is not decoration: a test that imported `sysid.measure_gap`
directly would have PASSED at HEAD, with the tool 0% reachable, and would therefore have proved nothing.
`tests/test_sysid_stage1.py` and `test_sysid_stage2.py` already cover the estimator; what was missing was any
test that a customer can get to it.

Two properties beyond "it dispatches", because both were real hazards:

  * **Our own published log schema had to be a schema we accept.** The experiment plan advertises `t_s` /
    `q_cmd_rad` / `q_meas_rad` / `tau_meas_nm`; the estimator reads `t` / `q_cmd` / `q_meas` / `tau_meas`. An
    engineer who followed our documentation to the letter got "log is missing required field 'q_cmd'" back
    from their own correctly-formatted file.
  * **A synthetic result must never read as a measurement.** The no-hardware path deliberately produces a log
    indistinguishable in SHAPE from a bench log -- that is the point of the harness and it is also the risk.
    Provenance is therefore carried in a field, not inferred, and it is what the actuator-fidelity rung turns on.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

#: The journey, in order. Ordered on purpose: an agent reading `tools/list` should get a sequence, not a bag.
_JOURNEY = ("plan_bench_experiment", "measure_sim_to_real_gap", "fit_actuators", "calibration_status")
_NO_HARDWARE = ("simulate_bench_log", "sim_to_real_transfer")
_ALL = _JOURNEY + _NO_HARDWARE


def _call(name, args=None):
    from virturoid.services.agent_tools import call_tool
    return call_tool(name, args or {})


def _result(name, args=None):
    env = _call(name, args)
    assert env["ok"] is True, env
    return env["result"]


@pytest.fixture(scope="module")
def held_arm():
    """A held robot, obtained the way a customer does: through the tool surface.

    A 4-DOF arm rather than a quadruped purely for cost -- the whole journey runs in ~9 s on it against ~75 s
    on a 12-joint body, and the SUBJECT here is the door, not the estimator."""
    schema = _result("get_design_schema")
    return _result("submit_design", {"graph": schema["examples"]["scara_arm"]})["robot_id"]


@pytest.fixture(scope="module")
def synthetic(held_arm):
    """The no-hardware path's own output: named paths, plus the log parsed back for the schema tests."""
    r = _result("simulate_bench_log", {"robot_id": held_arm, "budget_s": 10.0, "delay_ms": 0.0,
                                       "out_dir": "sysid_tests"})
    return r


# ---------------------------------------------------------------- reachable at all

def test_the_journey_is_registered_with_well_formed_schemas():
    from virturoid.services.agent_tools import TOOLS
    for name in _ALL:
        assert name in TOOLS, f"{name} is implemented but registered nowhere -- it is 0% agent-reachable"
        spec = TOOLS[name]
        assert callable(spec["handler"])
        params = spec["parameters"]
        assert params["type"] == "object" and "robot_id" in params["properties"], (name, params)
        assert spec["description"].strip()


@pytest.mark.skipif(not _MUJOCO, reason="the MCP registry import needs MuJoCo")
def test_the_journey_is_discoverable_in_the_tools_list_payload():
    """`MCP_TOOL_VIEW` is at its documented cross-client cap of 15 (test_agent_first asserts it), so these are
    advertised on `verify_robot` -- the honesty anchor, since "is this verdict true of MY machine?" is the next
    question anyone with a physical robot asks. A client reading tools/list must SEE every name."""
    from virturoid.mcp_server import _handle
    listed = _handle("tools/list", {})["tools"]
    assert len(listed) <= 15, "the lean menu must not grow past its budget"
    blob = " ".join(t["description"] for t in listed)
    for name in _ALL:
        assert name in blob, f"{name} dispatches but no tools/list entry names it -- undiscoverable"


@pytest.mark.skipif(not _MUJOCO, reason="the MCP registry import needs MuJoCo")
def test_the_journey_dispatches_over_MCP_tools_call(held_arm):
    """Being in the internal registry is not the same as being on the wire. `tools/call` is the wire."""
    from virturoid.mcp_server import _handle
    res = _handle("tools/call", {"name": "plan_bench_experiment",
                                 "arguments": {"robot_id": held_arm, "budget_s": 10.0}})
    assert res["isError"] is False, res
    assert res["structuredContent"].get("ok") is True


def test_capabilities_answers_the_question_an_engineer_actually_asked():
    """The engineer who could not find this searched `capabilities` among other things. `capability_summary`
    enumerates TASKS a body is scored on and system identification is not one, so the answer was silence."""
    cap = _result("capabilities")
    s2r = cap.get("sim_to_real")
    assert s2r, "capabilities says nothing about the sim-to-real gap"
    assert [s["tool"] for s in s2r["journey"]][:1] == ["plan_bench_experiment"]
    assert "simulate_bench_log" in s2r["needs"], "the no-hardware path must be named where it is looked for"


# ---------------------------------------------------------------- the experiment, and its own log schema

@pytest.mark.skipif(not _MUJOCO, reason="authoring an experiment needs MuJoCo")
def test_the_plan_is_bounded_by_this_robots_own_hardware(held_arm):
    """Not just `ok` -- the numbers that make it safe to run on somebody's machine."""
    r = _result("plan_bench_experiment", {"robot_id": held_arm, "budget_s": 10.0, "out_dir": "sysid_tests"})
    assert Path(r["plan_path"]).exists()
    assert r["budget"]["n_excitable"] >= 1 and r["budget"]["duration_s"] > 0
    for j in r["joints"]:
        lo, hi = j["safe_band_rad"]
        dlo, dhi = j["declared_limits_rad"]
        assert dlo <= lo < hi <= dhi, f"{j['joint']} is commanded outside its declared limits: {j}"
        assert j["datasheet_peak_torque_nm"] > 0 and j["actuator_part"], j
        if j["excitable"]:
            assert lo <= j["q_start_rad"] - j["amplitude_rad"], j
            assert j["q_start_rad"] + j["amplitude_rad"] <= hi, j
        else:
            assert j["not_excitable_because"], "a refusal without a reason is not a refusal"
    assert r["safety_envelope"]["reviewed_by_a_human"] is False
    assert "measure_sim_to_real_gap" in r["next"]["if_you_have_the_robot"]
    assert "simulate_bench_log" in r["next"]["if_you_do_not"]


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_log_schema_we_publish_is_a_schema_we_accept(held_arm, synthetic):
    """THE defect this file exists for, one level down from "the tool is missing".

    `plan_bench_experiment` tells the engineer to send back `t_s` / `q_cmd_rad` / `q_meas_rad` / `tau_meas_nm`.
    The estimator reads `t` / `q_cmd` / `q_meas` / `tau_meas`. Handing back a file written to our own published
    schema therefore failed with "log is missing required field 'q_cmd'" -- a documentation contract the
    product did not honour. Both spellings must work, and this asserts it in the DOCUMENTED spelling."""
    plan = _result("plan_bench_experiment", {"robot_id": held_arm, "budget_s": 10.0, "out_dir": "sysid_tests"})
    published = plan["log_format"]["required"]
    assert published == ["t_s", "q_cmd_rad", "q_meas_rad", "measured_on"], published

    raw = json.loads(Path(synthetic["log_path"]).read_text(encoding="utf-8"))
    as_published = {"joints": raw["joints"], "t_s": raw["t"], "q_cmd_rad": raw["q_cmd"],
                    "q_meas_rad": raw["q_meas"], "tau_meas_nm": raw["tau_meas"],
                    "control_hz": raw["control_hz"], "measured_on": "sim2sim"}
    r = _result("measure_sim_to_real_gap", {"robot_id": held_arm, "log": as_published,
                                            "plan_path": synthetic["plan_path"]})
    assert r["joints"], "a log written to our own published schema produced no per-joint table"
    assert r["attribution"]["available"] is True


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_wide_csv_telemetry_is_accepted(held_arm, synthetic, tmp_path):
    """Real telemetry arrives as CSV, one column per signal, far more often than as JSON."""
    raw = json.loads(Path(synthetic["log_path"]).read_text(encoding="utf-8"))
    js = raw["joints"]
    path = tmp_path / "telemetry.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s"] + [f"{j}.{f}" for j in js for f in ("q_cmd_rad", "q_meas_rad", "tau_meas_nm")])
        for i, t in enumerate(raw["t"]):
            row = [t]
            for k in range(len(js)):
                row += [raw["q_cmd"][i][k], raw["q_meas"][i][k], raw["tau_meas"][i][k]]
            w.writerow(row)
    r = _result("measure_sim_to_real_gap", {"robot_id": held_arm, "log_path": str(path),
                                            "plan_path": synthetic["plan_path"]})
    assert set(r["joints"]) == set(js), r["joints"]
    assert r["attribution"]["available"] is True, "a CSV carrying torque must still attribute a parameter"


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_a_csv_without_the_columns_teaches_instead_of_crashing(held_arm, tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("time,value\n0.0,1.0\n0.01,1.1\n", encoding="utf-8")
    env = _call("measure_sim_to_real_gap", {"robot_id": held_arm, "log_path": str(path)})
    assert env["ok"] is False, "a refused call must not be wrapped in a success envelope"
    r = env["result"]
    assert "expected" in r and "example_header" in r, r
    assert "q_cmd_rad" in json.dumps(r), "a refusal must hand back the field names it wanted"


# ---------------------------------------------------------------- the no-hardware path, and its labelling

@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_synthetic_path_returns_named_paths_and_never_a_tuple(synthetic):
    """`synthetic_hardware_log` returns a bare 2-tuple; an engineer guessed the order wrong and burned a 143 s
    run to find out. Nothing crosses this boundary without a key on it."""
    for key in ("log_path", "plan_path", "injected", "log_summary", "provenance"):
        assert key in synthetic, f"{key} missing: {sorted(synthetic)}"
    assert Path(synthetic["log_path"]).exists() and Path(synthetic["plan_path"]).exists()
    assert synthetic["log_path"] != synthetic["plan_path"]
    assert synthetic["log_summary"]["carries_measured_torque"] is True
    assert synthetic["injected"]["delay_ms"] == 0.0 and "delay_control_ticks" in synthetic["injected"]


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_delay_is_stated_in_milliseconds_and_converted_for_you(held_arm, synthetic):
    """`delay_ticks` is a count of control ticks, which is a unit only this package thinks in."""
    r = _result("simulate_bench_log", {"robot_id": held_arm, "budget_s": 10.0, "delay_ms": 20.0,
                                       "out_dir": "sysid_tests"})
    assert r["injected"]["delay_ms"] == pytest.approx(20.0, abs=5.0)
    assert r["injected"]["delay_control_ticks"] == 2, r["injected"]
    # ...and a second configuration must not land on the first one's path. It did: this test overwrote the
    # module fixture's 0 ms log, so a later fit measured a 20 ms-delay run while believing otherwise and its
    # withheld verdict read as a regression in the estimator. The filename carries the configuration now.
    assert r["log_path"] != synthetic["log_path"], "two injections wrote to the same file"
    assert Path(synthetic["log_path"]).exists(), "the earlier run's log was clobbered"


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_a_synthetic_result_can_never_read_as_a_measurement(held_arm, synthetic):
    """The one mistake this surface must not permit. Provenance is carried in a field and rides downstream."""
    assert synthetic["synthetic"] is True
    assert "NOT YOUR ROBOT" in synthetic["about"].upper()
    assert synthetic["provenance"]["is_a_measurement_of_your_hardware"] is False
    assert synthetic["provenance"]["raises_the_bench_identified_rung"] is False

    gap = _result("measure_sim_to_real_gap", {"robot_id": held_arm, "log_path": synthetic["log_path"],
                                              "plan_path": synthetic["plan_path"]})
    assert gap["provenance"]["class"] == "sim2sim"
    assert gap["provenance"]["is_a_measurement_of_your_hardware"] is False
    assert gap["headline"].startswith("NOT A MEASUREMENT OF YOUR ROBOT")


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_an_unstated_provenance_is_treated_as_not_hardware(held_arm, synthetic):
    """The direction that cannot over-claim. A customer exporter writes whatever it writes."""
    raw = json.loads(Path(synthetic["log_path"]).read_text(encoding="utf-8"))
    raw.pop("measured_on", None)
    raw.pop("provenance", None)
    r = _result("measure_sim_to_real_gap", {"robot_id": held_arm, "log": raw,
                                            "plan_path": synthetic["plan_path"]})
    assert r["provenance"]["class"] == "unstated"
    assert r["provenance"]["is_a_measurement_of_your_hardware"] is False
    assert "measured_on" in r["provenance"]["headline"], "it must say how to correct the record"


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_declaring_hardware_is_what_changes_the_verdict(held_arm, synthetic):
    """`measured_on` is not a formality: it is the field the bench-identified rung turns on, and it is
    load-bearing in BOTH directions -- the same bytes read as a measurement or not depending only on it."""
    raw = json.loads(Path(synthetic["log_path"]).read_text(encoding="utf-8"))
    r = _result("measure_sim_to_real_gap", {"robot_id": held_arm, "log": raw, "measured_on": "hardware",
                                            "plan_path": synthetic["plan_path"]})
    assert r["provenance"]["class"] == "hardware"
    assert r["provenance"]["raises_the_bench_identified_rung"] is True
    assert r["headline"].startswith("MEASURED ON HARDWARE")
    env = _call("measure_sim_to_real_gap", {"robot_id": held_arm, "log": raw, "measured_on": "wishful"})
    assert env["ok"] is False and "accepted" in env["result"], env


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_an_uninformative_experiment_produces_no_confident_numbers(held_arm):
    """`hold_only` perturbs the robot exactly as usual and never moves it: a real gap exists and NOTHING in the
    data can locate it. A tool that names parameters here is broken in the way that matters."""
    h = _result("simulate_bench_log", {"robot_id": held_arm, "budget_s": 10.0, "hold_only": True,
                                       "out_dir": "sysid_tests_hold"})
    assert "hold_only" in h["log_summary"].get("measured_on", "") or h["injected"]["hold_only"] is True
    gap = _result("measure_sim_to_real_gap", {"robot_id": held_arm, "log_path": h["log_path"],
                                              "plan_path": h["plan_path"]})
    assert gap["implicated_parameters"] == [], gap["implicated_parameters"]
    assert gap["joints_with_an_identified_gap"] == [], gap["joints_with_an_identified_gap"]


# ---------------------------------------------------------------- the fit, applied to the robot you hold

@pytest.mark.skipif(not _MUJOCO, reason="fitting needs MuJoCo")
def test_the_fit_reaches_the_model_and_comes_back_out(held_arm, synthetic):
    """A fit that stays in a report changes nothing, and a change a customer cannot undo is a change they
    cannot accept. Both halves, through the door, on the robot the session holds."""
    before = _result("calibration_status", {"robot_id": held_arm, "revert": True})
    assert before["calibrated"] is False, "revert must leave the arm uncalibrated whatever ran before"

    fit = _result("fit_actuators", {"robot_id": held_arm, "log_path": synthetic["log_path"],
                                    "plan_path": synthetic["plan_path"], "apply": True})
    gate = fit["application_gate"]
    assert gate["passed"] is True, f"a clean recovery must clear the tracking gate: {gate}"
    assert fit["calibration"]["applied"] is True and fit["calibration"]["n_joints_changed"] >= 1
    assert fit["identified_pairs"] >= 1
    for cell in fit["pinned"]:
        lo, hi = cell["interval"]
        assert lo <= cell["value"] <= hi, cell            # a value outside its own interval is not a posterior
        assert cell["unit"] and cell["also_absorbs"], cell

    after = _result("calibration_status", {"robot_id": held_arm})
    assert after["calibrated"] is True and after["report"]["n_changes"] >= 1
    assert after["actuator_ladder"]["earned"] is False, "a sim2sim log must not earn the bench rung"
    assert "measured on PHYSICAL HARDWARE" in json.dumps(after["actuator_ladder"]["blocked_by"])

    back = _result("calibration_status", {"robot_id": held_arm, "revert": True})
    assert back["revert"]["reverted"] is True and back["calibrated"] is False


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_the_fit_is_not_applied_unless_it_is_asked_for(held_arm, synthetic):
    """`apply` defaults to false, and the assertion is on the MODEL rather than on whether a record exists: a
    fit the tracking gate withholds still ATTACHES, deliberately, so the reason it changed nothing is on the
    robot instead of in a log line somebody has to go and find."""
    _result("calibration_status", {"robot_id": held_arm, "revert": True})
    r = _result("fit_actuators", {"robot_id": held_arm, "log_path": synthetic["log_path"],
                                  "plan_path": synthetic["plan_path"]})
    assert r["calibration"]["applied"] is False
    status = _result("calibration_status", {"robot_id": held_arm})
    assert status["calibrated"] is False, "a fit that was never applied must leave no record at all"
    assert status["report"].get("applied_to_model") in (None, False)


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("tool", _ALL)
def test_an_unknown_robot_is_an_honest_failure_not_a_crash(tool):
    args = {"robot_id": "no_such_robot_xyz"}
    if tool == "sim_to_real_transfer":
        args["policy_path"] = "nope.npz"
    env = _call(tool, args)
    assert isinstance(env.get("result"), dict), env
    assert env["result"]["ok"] is False and "no robot" in env["result"]["error"], env["result"]
    assert env["ok"] is False, "mcp_server sets isError from the envelope; a refusal must not read as success"


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_asking_for_a_gap_with_no_log_points_at_the_no_hardware_path(held_arm):
    """The likeliest first mistake, from the likeliest first user: someone with a robot and no telemetry."""
    env = _call("measure_sim_to_real_gap", {"robot_id": held_arm})
    assert env["ok"] is False
    r = env["result"]
    assert "log_path" in r["error"]
    assert "simulate_bench_log" in r["no_hardware"], r
    assert "plan_bench_experiment" in r["schema"], r


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_transfer_refuses_rather_than_measuring_a_random_policy(held_arm):
    """`sim2real_transfer_report` accepts `policy=None`, and the rollout underneath treats that as a RANDOM
    policy rather than a zero baseline -- so the honest-looking retention number would be a number for noise."""
    env = _call("sim_to_real_transfer", {"robot_id": held_arm})
    assert env["ok"] is False
    r = env["result"]
    assert "policy_path" in r["error"] and "RANDOM" in r["why"]
    assert "train_held" in r["how"], "a refusal must name where a policy comes from"


# ---------------------------------------------------------------- the discipline this file is written under

def test_this_file_never_imports_the_package_it_tests():
    """A test that reached `sysid` directly would have passed at HEAD, with the capability 0% reachable, and
    proved nothing. Every assertion above goes through the dispatcher a customer's agent actually uses."""
    src = Path(__file__).read_text(encoding="utf-8")
    offenders = [ln.strip() for ln in src.splitlines()
                 if ln.strip().startswith(("import ", "from ")) and "virturoid.services.sysid" in ln]
    assert not offenders, f"the door has to be the only way in: {offenders}"
