"""Stage 1 of the calibration wedge: the excitation, the gap number, and the identifiability statement.

The reality gap ranks #1 of 10 in the only relevant developer survey and no product serves it. Before this
package there was no system identification anywhere in the repo -- no parameter fitting, no log replay, no
trajectory matching -- and `docs/comprehensive_roadmap.md:148` had named the file for four months without it
being written.

WE OWN NO HARDWARE. Every test below is sim2sim: a second MuJoCo model with known perturbations stands in for
the robot. That validates the estimator and the plumbing; it cannot validate the physics, because MuJoCo's own
modelling error cancels when both sides are MuJoCo. See ``WHAT_SIM2SIM_DOES_NOT_PROVE``.

The tests are deliberately weighted toward what the tool REFUSES to say. Recovering a perturbation is the easy
half; declining to name a parameter the data could not constrain is the half that decides whether an engineer
can trust the output, and it is the half a calibration tool usually gets wrong.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="system identification needs MuJoCo")

INJECTED = {"frictionloss": 0.08, "damping": 0.6, "armature": 0.03}
DELAY_TICKS = 2


@pytest.fixture(scope="module")
def dog():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four legged robot dog", llm=None)


@pytest.fixture(scope="module")
def plan(dog):
    from virturoid.services.sysid import build_excitation
    return build_excitation(dog)


@pytest.fixture(scope="module")
def recovery(dog, plan):
    from virturoid.services.sysid import recovery_table
    return recovery_table(dog, perturbation=INJECTED, delay_ticks=DELAY_TICKS, plan=plan)


# --------------------------------------------------------------------------------------------------------
# The bench rig. If inverse dynamics is not exact on our own rollout, every residual downstream is slop.
# --------------------------------------------------------------------------------------------------------

def test_inverse_dynamics_reproduces_the_torque_that_produced_the_motion(dog):
    """The load-bearing identity: replaying a rollout's own (q, qd, qacc) through ``mj_inverse`` must return the
    torque that was actually applied. Everything this package attributes to a parameter is the DEPARTURE from
    this identity, so if the identity itself carried error we would be regressing numerical noise onto physical
    units and reporting the result in N.m."""
    import mujoco
    import numpy as np

    from virturoid.services.sysid.bench_rig import bench_gains, bench_model, start_pose

    model, _ = bench_model(dog)
    kp, kd, _ = bench_gains(model)
    nv = model.nv
    q0 = start_pose(model, dog)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:nv] = q0
    mujoco.mj_forward(model, data)

    inv = mujoco.MjData(model)
    worst = 0.0
    applied = np.zeros(nv)
    for k in range(400):
        if k % 5 == 0:
            applied = np.clip(kp * (q0 + 0.05 - data.qpos[:nv]) - kd * data.qvel[:nv],
                              -np.abs(model.actuator_forcerange[:, 1]).max(),
                              np.abs(model.actuator_forcerange[:, 1]).max())
        for a in range(model.nu):
            data.ctrl[a] = applied[int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])])]
        q, qd = data.qpos[:nv].copy(), data.qvel[:nv].copy()
        mujoco.mj_step(model, data)
        inv.qpos[:nv] = q; inv.qvel[:nv] = qd; inv.qacc[:nv] = data.qacc[:nv]
        mujoco.mj_inverse(model, inv)
        worst = max(worst, float(np.abs(applied - inv.qfrc_inverse[:nv]).max()))
    assert worst < 1e-9, f"inverse dynamics disagrees with the applied torque by {worst} N.m"


def test_the_bench_welds_the_base_and_says_so(dog):
    """A quadruped is measured on a stand, not on the floor -- which is the right experiment for actuator
    dynamics and the wrong one for contact. The report must carry that, or a reader will assume a number
    measured with the feet in the air also covers the feet on the ground."""
    from virturoid.services.sysid.bench_rig import bench_model

    assert dog.base_mount == "free", "fixture no longer floating-base; this test would be vacuous"
    model, rig = bench_model(dog)
    assert model.nq == len(dog.actuated_joints()), "free base was not welded"
    assert rig["welded_free_base"] is True
    assert rig["contact_gap_measured"] is False


def test_pd_replay_applies_the_torque_immediately_at_zero_delay(dog):
    """delay_ticks=0 must mean no delay. This off-by-one cancels between our own log and our own replay, so the
    sim2sim gate would stay green while every real log picked up a phantom control tick of latency."""
    import numpy as np

    from virturoid.services.sysid.bench_rig import bench_gains, bench_model, pd_replay, start_pose

    model, _ = bench_model(dog)
    kp, kd, _ = bench_gains(model)
    q0 = start_pose(model, dog)
    cmd = np.tile(q0 + 0.05, (40, 1))
    _, _, _, tau = pd_replay(model, cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=5, delay_ticks=0)
    assert np.abs(tau[0]).max() > 0.0, "first sample applied zero torque -> one tick of phantom delay"

    _, _, _, tau1 = pd_replay(model, cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=5, delay_ticks=1)
    assert np.abs(tau1[0]).max() == 0.0, "delay_ticks=1 did not delay"


# --------------------------------------------------------------------------------------------------------
# The excitation: short, and inside the hardware's envelope.
# --------------------------------------------------------------------------------------------------------

def test_excitation_is_short_enough_that_someone_would_actually_run_it(plan):
    assert plan["budget"]["duration_s"] <= plan["budget"]["requested_s"] + 1e-6
    assert plan["budget"]["duration_s"] > 0.0
    assert plan["budget"]["over_budget"] is False


def test_every_commanded_position_stays_inside_the_declared_joint_limits(dog, plan):
    """Checked against the GENE's declared limits, on the realized command array -- not against the plan's own
    summary of itself, which would be a tautology."""
    import numpy as np

    from virturoid.services.sysid.bench_rig import joint_dof_map, bench_model
    from virturoid.services.sysid import excitation_command_series

    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    _, cmd = excitation_command_series(dog, plan)
    for seg in dog.actuated_joints():
        adr = dofs.get(seg.name)
        if adr is None or seg.joint_lower is None or seg.joint_upper is None:
            continue
        col = cmd[:, adr]
        assert col.min() >= float(seg.joint_lower), f"{seg.name} commanded below its declared lower limit"
        assert col.max() <= float(seg.joint_upper), f"{seg.name} commanded above its declared upper limit"
        assert np.isfinite(col).all()


def test_no_joint_is_driven_past_the_torque_its_own_datasheet_motor_can_deliver(dog, plan):
    """The peak torque the excitation actually produces, measured on the rollout, against the peak torque of the
    real actuator ``select_actuator`` sized for that joint. A plan that respected its own arithmetic but not the
    physics would pass a limits check and still cook a motor."""
    import numpy as np

    from virturoid.services.sysid.bench_rig import (
        bench_gains, bench_model, joint_dof_map, pd_replay, start_pose,
    )
    from virturoid.services.sysid import excitation_command_series

    model, _ = bench_model(dog)
    kp, kd, _ = bench_gains(model)
    dofs = joint_dof_map(model, dog)
    _, cmd = excitation_command_series(dog, plan)
    q0 = start_pose(model, dog)
    for j in plan["joints"]:
        q0[int(j["dof"])] = float(j["q_start_rad"])
    ctrl_every = max(1, int(round((1.0 / model.opt.timestep) / plan["controller"]["control_hz"])))
    _, _, _, tau = pd_replay(model, cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every)
    for j in plan["joints"]:
        peak = float(np.abs(tau[:, int(j["dof"])]).max())
        assert peak <= float(j["datasheet_peak_torque_nm"]) + 1e-6, (
            f"{j['joint']} hit {peak:.3f} N.m against a datasheet peak of {j['datasheet_peak_torque_nm']} N.m")


def test_a_joint_whose_rest_pose_sits_outside_the_backoff_band_is_repositioned_not_dropped(dog, plan):
    """Regression. Clipping the rest pose to the EDGE of the safe band leaves zero headroom, zero amplitude, and
    the joint is dropped -- which silently removed all four knees from the quadruped's experiment. Their rest
    angle is 0.0 and their range is [-0.2, 2.4], so a 15% back-off puts the band at [0.19, 2.01] and 0.0 lands
    outside it. Those are the WIDEST-travelling joints on the machine and the best excitation candidates, and
    the report called them "not excitable: amplitude" -- a statement about this code, phrased as a fact about
    the robot."""
    wide = [s for s in dog.actuated_joints()
            if s.joint_lower is not None and s.joint_upper is not None
            and (s.joint_upper - s.joint_lower) > 2.0]
    assert wide, "fixture has no wide-range joint; this test would be vacuous"
    rows = {j["joint"]: j for j in plan["joints"]}
    for seg in wide:
        j = rows[seg.name]
        assert j["excitable"], f"{seg.name} has {seg.joint_upper - seg.joint_lower:.1f} rad of travel and was dropped"
        assert j["amplitude_rad"] > 0.0
        lo_b, hi_b = j["safe_band_rad"]
        assert lo_b < j["q_start_rad"] < hi_b, f"{seg.name} start pose pinned to a band edge"
        if j["moved_off_rest_pose"]:
            assert j["q_start_rad"] != j["rest_pose_rad"]


def test_a_joint_that_cannot_be_moved_safely_says_which_constraint_binds(dog):
    """Silence is the failure mode that matters: a joint quietly dropped from the experiment turns into a joint
    with no gap number, which reads as a joint with no gap."""
    from virturoid.services.sysid import build_excitation

    tiny = build_excitation(dog, budget_s=30.0)
    for j in tiny["joints"]:
        if not j["excitable"]:
            assert j["not_excitable_because"], f"{j['joint']} dropped without a reason"
            assert j["binding_constraint"] in ("gravity", "peak_torque", "no_load_speed",
                                               "amplitude", "bandwidth_cap")
        if j.get("chirp_bandwidth_limited"):
            assert j["chirp_bandwidth_limited_because"]


# --------------------------------------------------------------------------------------------------------
# The gap number.
# --------------------------------------------------------------------------------------------------------

def test_the_gap_is_reported_per_joint_in_physical_units_not_as_a_score(recovery):
    gap = recovery["gap"]
    assert gap["ok"] and gap["joints"]
    for name, row in gap["joints"].items():
        for field in ("position_rms_rad", "position_p95_rad", "position_max_rad",
                      "output_phase_lag_ms", "torque_rms_nm", "torque_bias_nm"):
            assert field in row, f"{name} missing {field}"
        assert row["position_rms_rad"] >= 0.0
    assert gap["worst_joints_by_position_rad"], "no ranking -> the report never names a joint"
    assert "score" not in gap and "fidelity" not in gap


def test_a_larger_perturbation_produces_a_larger_gap(dog, plan):
    """Monotonicity is the cheapest evidence that the number tracks reality rather than the harness. A gap that
    did not grow with the error would be measuring the excitation, not the robot."""
    import numpy as np

    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    rms = []
    for scale in (0.25, 1.0, 3.0):
        _, log = synthetic_hardware_log(
            dog, perturbation={k: v * scale for k, v in INJECTED.items()},
            delay_ticks=0, plan=plan)
        gap = measure_gap(dog, log, plan=plan, measure_noise_floor=False, delay_max_ticks=0)
        rms.append(float(np.mean([r["position_rms_rad"] for r in gap["joints"].values()])))
    assert rms[0] < rms[1] < rms[2], f"gap did not grow with the perturbation: {rms}"


def test_a_log_without_measured_torque_refuses_to_name_any_parameter(dog, plan):
    """Trajectory error alone cannot attribute a gap. The tool must say so rather than fall back on the
    trajectory and label the guess a parameter."""
    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, log = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan)
    log.pop("tau_meas")
    gap = measure_gap(dog, log, plan=plan, delay_max_ticks=1)
    assert gap["ok"]
    assert gap["joints"], "trajectory gap should still be reported"
    assert gap["attribution"]["available"] is False
    assert "tau_meas" in gap["attribution"]["why"]
    assert "implicated_parameters" not in gap


def test_a_partial_log_is_rejected_rather_than_zero_filled(dog, plan):
    """Inverse dynamics on joint J depends on where every other joint was. Zero-filling an unlogged joint would
    produce a confident residual attributable to nothing at all."""
    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, log = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan)
    dropped = log["joints"][0]
    log["joints"] = log["joints"][1:]
    for key in ("q_cmd", "q_meas", "qd_meas", "tau_meas"):
        log[key] = [row[1:] for row in log[key]]
    gap = measure_gap(dog, log, plan=plan)
    assert gap["ok"] is False
    assert dropped in gap["missing_joints"]


# --------------------------------------------------------------------------------------------------------
# Recovery: the sim2sim gate.
# --------------------------------------------------------------------------------------------------------

def test_the_injected_perturbation_comes_back(recovery):
    """Recovery accuracy, measured. The tolerances are the ones the estimator actually achieves, not aspirational
    ones: frictionloss returns low because MuJoCo models it as a solver constraint with a stick regime and the
    sensitivity is linearized about the current value, which under-reads a perturbation that nearly doubles it."""
    tol = {"frictionloss": 0.35, "damping": 0.15, "armature": 0.25}
    for name, row in recovery["parameters"].items():
        err = abs(row["recovered_median"] - row["injected"]) / row["injected"]
        assert err <= tol[name], (
            f"{name}: injected {row['injected']} recovered {row['recovered_median']} ({err:.1%} error)")
        assert (row["recovered_median"] > 0) == (row["injected"] > 0), f"{name} recovered with the wrong sign"


def test_the_actuation_delay_is_recovered_exactly(recovery):
    """Whole control ticks, recovered by re-simulating the declared loop. Exact is the right bar here: the delay
    lives on a grid, so anything other than the injected tick is a different answer, not a noisy one."""
    d = recovery["delay"]
    assert d["identified"] is True, d
    assert d["error_ms"] == 0.0, f"injected {d['injected_ms']} ms, recovered {d['recovered_ms']} ms"
    assert d["at_grid_edge"] is False, "best delay sat at the edge of the search grid; widen it"


def test_latency_is_not_claimed_when_a_dynamics_error_could_explain_it_instead(dog, plan):
    """Measured failure, guarded: a delay-only sweep on an UNCORRECTED model picks delay=0 when friction and
    inertia errors dominate the trajectory. The tool must decline on that pass and say why, and only claim the
    delay once the identified parameter deltas have removed the competing explanation."""
    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, log = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=DELAY_TICKS, plan=plan)
    gap = measure_gap(dog, log, plan=plan)
    lat = gap["latency"]
    assert "after_parameter_correction" in lat, "no corrected-model pass was run"
    assert lat["identified"] is True
    assert lat["source"] == "after_parameter_correction", (
        "the uncorrected sweep should not have been able to claim this delay")
    assert lat["after_parameter_correction"]["delay_ms"] == DELAY_TICKS * 1000.0 / plan["controller"]["control_hz"]


def test_the_estimator_reports_its_own_floor_and_the_floor_is_not_zero(recovery):
    """Feeding the simulator's own acceleration recovers a perturbation EXACTLY; differentiating logged velocity
    does not. That difference is measurement error, it is what a real log will have, and it has to be published
    as a floor rather than absorbed into the estimates."""
    rows = recovery["gap"]["attribution"]["per_joint"]
    floors = [v["noise_floor"] for r in rows.values() for v in r["parameters"].values()]
    assert floors and max(floors) > 0.0, "noise floor never measured -> estimates carry no lower bound"
    assert "differentiating logged velocity" in recovery["gap"]["attribution"]["noise_floor_source"]


def test_sim2sim_is_labelled_and_its_limits_are_stated(recovery):
    assert recovery["provenance"] == "sim2sim"
    text = recovery["what_this_does_not_prove"]
    assert "does NOT validate the PHYSICS" in text
    assert "UPPER BOUND" in text


# --------------------------------------------------------------------------------------------------------
# Identifiability: what the tool REFUSES to say.
# --------------------------------------------------------------------------------------------------------

def test_an_uninformative_experiment_identifies_nothing_and_names_the_missing_excitation(dog, plan):
    """THE test. The robot is perturbed exactly as in the recovery gate -- a real gap exists -- but it is
    commanded to hold still, so nothing in the data can locate it. Every parameter must come back unidentified
    with a physical reason and a fix. Returning three confident numbers here is the failure this whole package
    exists to prevent."""
    from virturoid.services.sysid import measure_gap
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    _, log = synthetic_hardware_log(dog, perturbation=INJECTED, delay_ticks=0, plan=plan, hold_only=True)
    gap = measure_gap(dog, log, plan=plan, delay_max_ticks=1)
    assert gap["implicated_parameters"] == [], (
        f"named parameters from a robot that never moved: {gap['implicated_parameters']}")

    rows = gap["attribution"]["per_joint"]
    assert rows, "no joints were analysed at all"
    for name, rep in rows.items():
        # A constant torque OFFSET is the one term a hold CAN identify -- ``EXCITATION_REQUIREMENTS['offset']``
        # says so outright ("needs only samples"), and it is excluded from ``implicated_parameters`` for
        # exactly that reason. What must never survive a hold-only log is a PHYSICAL parameter, and an
        # earlier version of this assertion caught the offset too, so it passed only while the bench gain was
        # too low to hold the distal joints against gravity in the first place.
        assert [p for p in rep["identified"] if p != "offset"] == [], (
            f"{name} claimed {rep['identified']} from a hold-only log")
        for pname in ("frictionloss", "damping", "armature"):
            p = rep["parameters"][pname]
            assert p["reasons_not_identified"], f"{name}/{pname} declined without a reason"
            joined = " ".join(p["reasons_not_identified"])
            assert "not excited" in joined, f"{name}/{pname} gave a statistical reason, not a physical one"
            assert p["excitation_metric"] and p["excitation_needed"] > 0


def test_coulomb_friction_is_declared_unidentifiable_without_velocity_sign_reversals():
    """The textbook degeneracy, and the one an engineer is most likely to walk into: with one-directional motion
    ``sign(qd)`` is a constant and is exactly collinear with the gravity offset. No sample count fixes it. The
    report must name the confound rather than split the two arbitrarily."""
    import numpy as np

    from virturoid.services.sysid.identifiability import identifiability_report

    n = 600
    rng = np.random.default_rng(0)
    qd = np.linspace(0.4, 1.2, n)                       # never reverses
    X = np.column_stack([np.ones(n), np.sign(qd), qd, rng.normal(size=n)])
    y = 0.3 + 0.08 * np.sign(qd) + 0.6 * qd + 0.01 * rng.normal(size=n)
    rep = identifiability_report("one_way", X, y, ["frictionloss", "damping", "armature"],
                                 excitation_stats={"velocity_sign_reversals": 0, "speed_rms_radps": 0.8,
                                                   "accel_rms_radps2": 1.0, "n_samples": n})
    fric = rep["parameters"]["frictionloss"]
    assert fric["identified"] is False
    assert "not excited" in " ".join(fric["reasons_not_identified"])
    assert fric["suggested_experiment"], "declined without telling the engineer what to add"
    assert rep["parameters"]["damping"]["identified"] is True, "damping is fine here and should still be claimed"


def test_two_parameters_with_the_same_signature_are_flagged_as_confounded():
    """When two columns carry the same shape, the split between them is arbitrary. VIF must catch it and the
    report must name the partner -- 'damping is confounded with armature' is actionable, a number is not."""
    import numpy as np

    from virturoid.services.sysid.identifiability import identifiability_report

    n = 800
    rng = np.random.default_rng(1)
    qd = np.sin(np.linspace(0, 20, n))
    twin = qd + 1e-4 * rng.normal(size=n)               # a near-duplicate column
    X = np.column_stack([np.ones(n), np.sign(qd), qd, twin])
    y = 0.5 * qd + 0.5 * twin + 0.02 * rng.normal(size=n)
    rep = identifiability_report("twins", X, y, ["frictionloss", "damping", "armature"],
                                 excitation_stats={"velocity_sign_reversals": 20, "speed_rms_radps": 0.7,
                                                   "accel_rms_radps2": 2.0, "n_samples": n})
    damp, arm = rep["parameters"]["damping"], rep["parameters"]["armature"]
    assert not (damp["identified"] and arm["identified"]), "both halves of a confounded pair were claimed"
    flagged = [p for p in (damp, arm) if p["confounded_with"]]
    assert flagged, "collinearity was not reported"
    assert flagged[0]["vif"] > 10.0


def test_standard_errors_are_deflated_for_autocorrelation():
    """25,000 rows of a 2 Hz sinusoid are not 25,000 independent observations. Without the deflation the error
    bars claim orders of magnitude more precision than the experiment contains, and every parameter passes."""
    import numpy as np

    from virturoid.services.sysid.identifiability import _effective_n

    n = 4000
    rng = np.random.default_rng(2)
    white = rng.normal(size=n)
    smooth = np.convolve(white, np.ones(50) / 50.0, mode="same")     # strongly autocorrelated
    assert _effective_n(smooth) < 0.25 * n, "autocorrelated residual was not deflated"
    assert _effective_n(white) > 0.8 * n, "white residual was deflated anyway"


def test_the_intercept_does_not_report_infinite_collinearity_with_itself():
    """A constant column has an undefined VIF, not an infinite one. Reporting inf failed the intercept's
    separability gate on every joint forever -- a bug that reads exactly like a finding."""
    import numpy as np

    from virturoid.services.sysid.identifiability import identifiability_report

    n = 500
    rng = np.random.default_rng(3)
    qd = np.sin(np.linspace(0, 30, n))
    X = np.column_stack([np.ones(n), np.sign(qd), qd, np.cos(np.linspace(0, 30, n))])
    y = 1.0 + 0.05 * np.sign(qd) + 0.4 * qd + 0.01 * rng.normal(size=n)
    rep = identifiability_report("offsetting", X, y, ["frictionloss", "damping", "armature"],
                                 excitation_stats={"velocity_sign_reversals": 30, "speed_rms_radps": 0.7,
                                                   "accel_rms_radps2": 1.5, "n_samples": n})
    off = rep["parameters"]["offset"]
    assert off["vif"] is None, f"intercept reported a VIF of {off['vif']}"
    assert not any("not separable" in r for r in off["reasons_not_identified"])
    assert off["identified"] is True, "a real 1.0 N.m offset should be identified"


def test_only_identified_parameters_are_applied_to_the_corrected_model(dog, plan):
    """The corrected model used for the latency re-search must carry ONLY parameters that passed the gates.
    Applying an unidentified estimate would smuggle the over-claim back in through the side door."""
    from virturoid.services.sysid.gap_report import _corrected_model
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map

    model, _ = bench_model(dog)
    dofs = joint_dof_map(model, dog)
    name = next(iter(dofs))
    rows = {name: {"identified": ["damping"],
                   "parameters": {"damping": {"estimate": 0.5}, "frictionloss": {"estimate": 99.0},
                                  "armature": {"estimate": 99.0}}}}
    fixed = _corrected_model(model, rows, dofs)
    adr = dofs[name]
    assert fixed.dof_damping[adr] == pytest.approx(model.dof_damping[adr] + 0.5)
    assert fixed.dof_frictionloss[adr] == pytest.approx(model.dof_frictionloss[adr])
    assert fixed.dof_armature[adr] == pytest.approx(model.dof_armature[adr])


def test_the_bench_model_is_not_the_shared_compile_cache(dog):
    """``bench_model`` must hand back a private MjModel: the sensitivity columns mutate dof_* in place, and one
    perturbed model leaking into ``morph_policy.compiled_model``'s cache would silently change the dynamics of
    every unrelated rollout in the process."""
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    from virturoid.services.sysid.bench_rig import bench_model

    shared = compiled_model(robot_mjcf(dog))
    before = float(shared.dof_damping[0])
    model, _ = bench_model(dog)
    model.dof_damping[:] = 999.0
    assert float(compiled_model(robot_mjcf(dog)).dof_damping[0]) == before
