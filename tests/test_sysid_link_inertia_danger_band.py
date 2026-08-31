"""The DANGER BAND of the link-inertia check: correct calibrations that only just clear the tracking gate.

``tests/test_sysid_stage2.py`` already pins the link-inertia check on its catches and on five correctly
specified fits. It passed while the check was refusing correct calibrations, and the reason is the whole
point of this file: **every one of those five improved tracking by 3.34x or more.** The check's failure mode
lives between 1.5x and about 1.9x, which is where a real customer's fit is most likely to land, and nothing
sampled there.

WHAT WAS MEASURED (composed 14-DOF dog, 35 s excitation, delay 0; bit-identical at n_boot 64 and 96), with the
first shipped version of the check:

    armature-only +0.008   improvement_x 4.009   inertia rival 0.436   applied
    armature-only +0.009   improvement_x 1.602   inertia rival 1.108   REFUSED as link_inertia   <- WRONG
    armature-only +0.010   improvement_x 1.752   inertia rival 1.024   REFUSED as link_inertia   <- WRONG
    armature-only +0.011   improvement_x 1.925   inertia rival 0.938   applied, 6% from refusal

+0.010 against a prior of 0.010 is a reflected inertia twice the prior's -- an ordinary calibration. The
customer was told to go and re-derive CAD inertia tensors that are correct.

THE CAUSE is arithmetic, not physics. ``explains_x = fit_rms / rival_rms``, and

    explains_x = (before/rival) / (before/after) = rival_improvement / FIT_improvement

so the fit's own strength is in the denominator. Across the correctly-specified population the fit's strength
ranges from 1.5x to 20.9x on these two bodies, and the SAME rival therefore scores 0.099 against a strong
correct fit and 1.108 against a weak one. The ranking is monotone in fit strength, not in whether anything is
wrong. ``fit.LINK_INERTIA_RIVAL_IMPROVEMENT_X`` is the repair: the numerator alone, which has no such confound.

AND THE MECHANISM THE FILE GAVE FOR WHY THIS COULD NOT HAPPEN WAS FALSE, in four places at once: an inertia
scalar's term is proportional to each joint's own rotational inertia, said to "span ~40x between the trunk and
a distal leg joint". MEASURED via ``mj_fullM`` it is 1.547x on the dog and 1.528x on the hexapod -- and the
shipped ``rotational_inertia_span_x`` field was printing 1.55 one line under the comment claiming 40. A span
that narrow is exactly why a scalar CAN nearly imitate a uniform armature offset.

AND THE REPAIR'S OWN MARGIN WAS A TWO-BODY SAMPLE (added 2026-08-13, same day). The threshold was sized as the
geometric midpoint of a correct ceiling of 1.930 and a catch floor of 19.604 -- both measured on the dog and
the hexapod, whose rotational-inertia spans are near-twins (1.547x / 1.528x). A composed CAT (span 2.628x)
moved BOTH edges, in opposite directions, in one sitting:

    armature-only +0.003            improvement 1.542   explains_x 1.989   rival_improvement_x  3.089
    armature +45 x each joint's R_a improvement 2.285   explains_x 1.610   rival_improvement_x  3.679  ceiling
    inertia_scale x30               improvement 1.500   explains_x 9.646   rival_improvement_x 14.469  floor

so the band read 3.93x rather than 10.2x, and the line moved to sqrt(3.679 x 14.469) = 7.3. No measured
verdict changed. **AND 3.93x IS ALSO SUPERSEDED (2026-08-13, same day): a composed CENTIPEDE reads 4.037 on
the correct side -- it is in the shipped per-body table in ``what_this_gate_does_not_catch`` -- so the ceiling
in force is 4.037 and the empty band is 14.469 / 4.037 = 3.58x.** 7.3 is kept (no measured verdict differs
from the current midpoint of 7.643) and is now 1.81x above the ceiling and 1.98x below the floor rather than
centred. Every restatement of this band has shrunk it, each time by adding one more body: the edges are
running maxima over what has been sampled, not bounds.
The cat is in this file as a THIRD BODY for exactly that reason -- and its ``armature +0.003``
(``explains_x`` 1.989) is a stronger reproduction of the original defect than either dog case, since the old
single-condition rule would have refused an ordinary calibration by a factor of two rather than by 10%.

These tests are slow (one full fit per case, on three composed bodies). They are the acceptance evidence for
the repair, so they run the real thing rather than a stub.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="system identification needs MuJoCo")

BUDGET_S = 35.0

#: The two reproduced false refusals, plus the points either side of them that were APPLIED. Keeping the
#: neighbours matters: a check that refuses a strip of the correct population is different from one that
#: refuses all of it, and the strip is what nobody sampled.
_DOG_BAND = [
    ("armature +0.009", {"armature": 0.009}),      # was REFUSED: improvement 1.602, old statistic 1.108
    ("armature +0.010", {"armature": 0.010}),      # was REFUSED: improvement 1.752, old statistic 1.024
    ("armature +0.0105", {"armature": 0.0105}),    # improvement 1.843, old statistic 0.976 -- 2% from refusal
    ("armature +0.011", {"armature": 0.011}),      # improvement 1.925, old statistic 0.938 -- 6% from refusal
    ("frictionloss +0.030", {"frictionloss": 0.030}),           # improvement 1.535, rival 0.982
    ("mixed fl.005 d.05 arm.009", {"frictionloss": 0.005, "damping": 0.05, "armature": 0.009}),  # 2.030
    # OUTSIDE the band on purpose: the strong end of the same family. It is the case the original nine were
    # made of (improvement 17.910), and putting it beside +0.009 is what shows the confound -- the old
    # statistic moves 0.099 -> 1.108 across these two while the new one moves 1.774 -> 1.777.
    ("armature +0.030", {"armature": 0.030}),
]

#: The SECOND body. A number from the composed dog is not general -- this package has been burned by exactly
#: that before (docs/calibration_wedge_under_delay.md section 12).
_HEXA_BAND = [
    ("armature +0.003", {"armature": 0.003}),        # improvement 1.607, rival_improvement_x 1.208
    ("armature +0.0035", {"armature": 0.0035}),      # 1.870 / 1.216
    ("armature +0.004", {"armature": 0.004}),        # 2.139 / 1.226
    ("armature +0.0045", {"armature": 0.0045}),      # 2.399 / 1.241
    ("frictionloss +0.015", {"frictionloss": 0.015}),  # 1.728 / 0.993
    ("default x0.1", {"frictionloss": 0.008, "damping": 0.06, "armature": 0.003}),  # 2.619 / 1.047
]

#: The THIRD body, added 2026-08-13, and it is not a third confirmation -- it is what made the numbers in
#: ``fit.LINK_INERTIA_RIVAL_IMPROVEMENT_X`` move. The dog and the hexapod have nearly the same
#: rotational-inertia span (1.547x / 1.528x), so together they were ONE sample of the thing that decides how
#: degenerate the rival is. The cat spans 2.628x and its whole correct population sits about twice as high:
#:
#:   * ``armature +0.003`` reads ``explains_x`` **1.989** -- under the single-condition rule this file exists
#:     to test, an ordinary calibration would be refused by a FACTOR OF TWO, a stronger reproduction of the
#:     original defect than either dog case (1.108 and 1.024).
#:   * ``armature proportional k=45`` is the ADVERSARIAL correct calibration: a per-joint armature offset built
#:     equal to 45 x each joint's own ``R_a``, i.e. the one correct fit whose signature the inertia rival's
#:     one-parameter family contains EXACTLY. It reads rival_improvement_x **3.679**, which is the measured
#:     ceiling of the correct population across all three bodies and what the threshold is now sized against.
_CAT_BAND = [
    ("armature +0.003", {"armature": 0.003}),        # improvement 1.542, explains_x 1.989, rival_imp 3.089
    ("armature +0.0045", {"armature": 0.0045}),      # 2.229 / 1.386 / 3.083
    ("armature +0.006", {"armature": 0.006}),        # 2.840 / 1.126 / 3.181
    ("frictionloss +0.020", {"frictionloss": 0.020}),  # 2.108 / 0.475 / 1.003
]

#: The two adversarial correct calibrations on the cat, as multiples of each joint's OWN rotational inertia.
#: ``synthetic_hardware_log``'s ``perturbation`` dict is one scalar per parameter for every joint, so these
#: cannot be expressed through it and the harness builds the plant itself. k=30 is the largest ``explains_x``
#: measured anywhere in the correct population (2.228); k=45 is the largest ``rival_improvement_x`` (3.679).
_CAT_PROPORTIONAL = [("armature +30 x R_a", 30.0), ("armature +45 x R_a", 45.0)]


def _fit(gene, plan, perturbation):
    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log
    _, lg = synthetic_hardware_log(gene, perturbation=perturbation, delay_ticks=0, plan=plan)
    return fit_parameters(gene, lg, plan=plan, n_boot=64, measure_delay=False)


def _rotational_diag(model, dofs, q0):
    """``R_a`` per joint: diag(M) at 2x body inertia minus diag(M) at 1x. Exact, because the dependence is
    linear -- the same measurement ``fit.link_inertia_signature`` makes."""
    import copy

    import mujoco
    import numpy as np

    def diag(m):
        d = mujoco.MjData(m)
        d.qpos[:m.nv] = q0
        mujoco.mj_forward(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, full, d.qM)
        return np.diag(full).copy()

    twice = copy.deepcopy(model)
    twice.body_inertia[1:] = np.asarray(twice.body_inertia[1:], dtype=float) * 2.0
    d1, d2 = diag(model), diag(twice)
    return {n: float(d2[a] - d1[a]) for n, a in dofs.items() if float(d2[a] - d1[a]) > 1e-12}


def _fit_proportional_armature(gene, plan, k):
    """A CORRECT calibration whose armature offset is ``k`` x each joint's own rotational inertia.

    This is the exactly-degenerate case, and it is the reason it is in this file: an inertia scale of (1+k)
    adds the SAME per-joint quantity to diag(M) that this armature offset does, so the rival's one-parameter
    family contains the data-generating process even though nothing about the plant's CAD is wrong. If any
    correct fit is going to be refused, it is this one -- and it is where the correct population's ceiling was
    found (3.679 at k=45 on the cat).
    """
    import numpy as np

    from virturoid.services.sysid import fit_parameters
    from virturoid.services.sysid.bench_rig import (
        bench_gains, bench_model, joint_dof_map, pd_replay, start_pose,
    )
    from virturoid.services.sysid.excitation import excitation_command_series

    model, _ = bench_model(gene)
    kp, kd, _ = bench_gains(model)
    dofs = joint_dof_map(model, gene)
    rots = _rotational_diag(model, dofs, start_pose(model, gene))

    import copy
    hw = copy.deepcopy(model)
    for n, r in rots.items():
        hw.dof_armature[dofs[n]] = float(model.dof_armature[dofs[n]]) + float(k) * r

    _, q_cmd = excitation_command_series(gene, plan)
    q0 = start_pose(model, gene)
    for j in plan["joints"]:
        q0[int(j["dof"])] = float(j["q_start_rad"])
    ctrl_every = max(1, int(round((1.0 / float(model.opt.timestep))
                                  / float(plan["controller"]["control_hz"]))))
    t, q, qd, tau = pd_replay(hw, q_cmd, kp=kp, kd=kd, q_start=q0, ctrl_every=ctrl_every, delay_ticks=0)
    names = list(dofs)
    cols = [dofs[n] for n in names]
    lg = {"joints": names, "control_hz": float(plan["controller"]["control_hz"]),
          "log_hz": 1.0 / float(model.opt.timestep), "t": t.tolist(),
          "q_cmd": np.asarray(q_cmd)[:, cols].tolist(), "q_meas": q[:, cols].tolist(),
          "qd_meas": qd[:, cols].tolist(), "tau_meas": tau[:, cols].tolist(),
          "measured_on": "sim2sim",
          "provenance": f"SYNTHETIC - armature offset of {k:g} x each joint's own rotational inertia, i.e. a "
                        f"CORRECT reflected-inertia calibration that is exactly degenerate with a link-inertia "
                        f"scale of {1.0 + k:g}."}
    return fit_parameters(gene, lg, plan=plan, n_boot=64, measure_delay=False)


@pytest.fixture(scope="module")
def dog_band():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation
    g = compose_robot("a four legged robot dog", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    return {label: _fit(g, p, pert) for label, pert in _DOG_BAND}


@pytest.fixture(scope="module")
def hexa_band():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation
    g = compose_robot("a six legged walking robot", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    return {label: _fit(g, p, pert) for label, pert in _HEXA_BAND}


@pytest.fixture(scope="module")
def cat_band():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation
    g = compose_robot("a small robot cat", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    out = {label: _fit(g, p, pert) for label, pert in _CAT_BAND}
    out.update({label: _fit_proportional_armature(g, p, k) for label, k in _CAT_PROPORTIONAL})
    return out


def _row(fit):
    app, li = fit["application"], fit["link_inertia"]
    riv = li.get("rival") or {}
    return {"improvement_x": app.get("improvement_x"), "refused_by": app.get("refused_by"),
            "explains_x": riv.get("explains_x"), "rival_improvement_x": riv.get("rival_improvement_x"),
            "implied_s": li.get("implied_inertia_scale_s"), "span_x": li.get("rotational_inertia_span_x")}


def _assert_band_is_clean(band, body):
    from virturoid.services.sysid.fit import LINK_INERTIA_RIVAL_IMPROVEMENT_X

    table = {label: _row(fit) for label, fit in band.items()}
    in_band = {k: v for k, v in table.items()
               if v["improvement_x"] is not None and 1.5 <= v["improvement_x"] <= 3.0}
    assert len(in_band) >= 4, (
        f"the {body} sweep no longer LANDS in the 1.5x-3.0x band -- this test proves nothing unless the cases "
        f"are actually in the region that was never sampled. Retune the magnitudes; do not delete. {table}")
    for label, r in table.items():
        assert r["refused_by"] != "link_inertia", (
            f"{body}/{label} is a CORRECTLY SPECIFIED calibration and the link-inertia check refused it: "
            f"{r}. This is the defect, not a regression of a nicety -- the customer is sent to re-derive CAD "
            f"inertia tensors that are right. Threshold in force: "
            f"{LINK_INERTIA_RIVAL_IMPROVEMENT_X}x on rival_improvement_x. Table: {table}")
        assert fit_passed(r), f"{body}/{label} lost its PASS for some other reason: {r}. Table: {table}"
    return table


def fit_passed(r):
    return r["refused_by"] is None


def test_no_correct_calibration_in_the_danger_band_is_refused_on_the_dog(dog_band):
    """THE ACCEPTANCE TEST. Seven correct calibrations on the composed dog, four of them inside 1.5x-3.0x.

    Two of these (+0.009, +0.010) were refused by the shipped check and are the reproduction of the defect.
    """
    table = _assert_band_is_clean(dog_band, "dog")

    # ...and the OLD statistic still fires on them, which is what makes this a test of the REPAIR rather than
    # a test that the cases drifted somewhere harmless. If ``explains_x`` stops exceeding 1.0 here, the
    # separation being asserted is no longer the one that was broken.
    old_would_refuse = [k for k, v in table.items()
                        if v["explains_x"] is not None and v["explains_x"] >= 1.0]
    assert old_would_refuse, (
        f"none of these cases trips the OLD single-condition rule any more, so this file has stopped testing "
        f"the defect. Measured: armature +0.009 -> 1.108 and +0.010 -> 1.024. Table: {table}")


def test_no_correct_calibration_in_the_danger_band_is_refused_on_the_cat(cat_band):
    """THE THIRD BODY, and the one that moved the threshold. 14 joints, rotational-inertia span 2.628x.

    This is not a third confirmation of the dog's numbers. The cat's whole correct population sits about twice
    as high on the separating statistic, and ``armature +0.003`` reads ``explains_x`` 1.989 -- under the
    single-condition rule this file exists to test, an ordinary reflected-inertia calibration would be refused
    by a FACTOR OF TWO, where the dog's worst case was 1.108.
    """
    from virturoid.services.sysid.fit import LINK_INERTIA_RIVAL_IMPROVEMENT_X

    table = _assert_band_is_clean(cat_band, "cat")

    worst_old = max((v["explains_x"] for v in table.values() if v["explains_x"] is not None), default=0.0)
    assert worst_old >= 1.9, (
        f"the cat cases no longer reproduce the defect at strength. Measured: armature +0.003 -> explains_x "
        f"1.989 and armature +30 x R_a -> 2.228, i.e. the OLD rule refused an ordinary calibration by a factor "
        f"of two. If this drops under 1.9 the cat has stopped being the strongest reproduction there is. "
        f"Table: {table}")

    # ...and the CEILING that sized the threshold. The adversarial proportional case is the largest
    # rival_improvement_x ever measured on a correct fit, on any body: 3.679.
    ceiling = max((v["rival_improvement_x"] for v in table.values()
                   if v["rival_improvement_x"] is not None), default=0.0)
    assert 3.0 <= ceiling < LINK_INERTIA_RIVAL_IMPROVEMENT_X, (
        f"the cat's correct-population ceiling reads {ceiling} (measured 3.679 at armature +45 x R_a). It is "
        f"what LINK_INERTIA_RIVAL_IMPROVEMENT_X = {LINK_INERTIA_RIVAL_IMPROVEMENT_X} is sized against, so if it "
        f"has moved the threshold needs re-deciding, not this assertion relaxing. Table: {table}")


def test_the_cats_correct_ceiling_is_twice_the_dogs_so_one_body_could_not_have_sized_the_threshold(
        dog_band, cat_band):
    """WHY THREE BODIES. The dog and the hexapod span 1.547x and 1.528x -- together they are ONE sample of the
    quantity that decides how degenerate this rival is. The threshold was sized on them and read 1.930 as the
    correct ceiling; on a body that is not a near-twin the ceiling is 3.679, nearly double.

    This test fails if the two bodies ever converge, because then the file has stopped demonstrating that a
    single body's ceiling is not the population's.
    """
    dog = max((_row(f)["rival_improvement_x"] or 0.0) for f in dog_band.values())
    cat = max((_row(f)["rival_improvement_x"] or 0.0) for f in cat_band.values())
    assert cat > dog * 1.5, (
        f"the cat's correct ceiling ({cat}) is no longer well above the dog's ({dog}); measured 3.679 vs 1.804. "
        f"The point of this file is that a threshold sized on one body's ceiling is not sized on the "
        f"population's, and that point rests on these two differing")


def test_a_real_link_inertia_error_on_the_cat_is_still_refused():
    """The catch on the third body, and the one that moved the CATCH FLOOR down.

    ``inertia_scale x40`` reads rival_improvement_x 29.16 here. The neighbouring x30 is the lowest catch ever
    measured (14.469, at improvement_x 1.500 -- right on the tracking gate), which is why the threshold is 7.3
    and not the 8.5 that the retired floor of 19.604 would have implied.
    """
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation, fit_parameters
    from virturoid.services.sysid.fit import LINK_INERTIA_RIVAL_IMPROVEMENT_X
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    g = compose_robot("a small robot cat", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    _, lg = synthetic_hardware_log(g, perturbation={}, delay_ticks=0, plan=p, inertia_scale=40.0)
    fit = fit_parameters(g, lg, plan=p, n_boot=64, measure_delay=False)
    r = _row(fit)
    assert r["refused_by"] == "link_inertia", f"the cat's catch was lost: {r}"
    assert r["rival_improvement_x"] >= LINK_INERTIA_RIVAL_IMPROVEMENT_X, r
    # The margin. Measured 29.16 against a threshold of 7.3 and a correct ceiling on this body of 3.679. The
    # bound is deliberately below the neighbouring x30's 14.469, which is the true floor of the catch
    # population -- if THIS drifts under it, the two populations on this body are merging.
    assert r["rival_improvement_x"] >= 14.0, (
        f"the cat catch's margin collapsed to {r['rival_improvement_x']} (measured 29.16). The lowest catch "
        f"measured anywhere is 14.469 (cat inertia_scale x30) and the correct population tops out at 3.679; a "
        f"catch below 14 means the gap this threshold sits in is closing")


def test_no_correct_calibration_in_the_danger_band_is_refused_on_the_hexapod(hexa_band):
    """THE SAME BAND ON A SECOND BODY, 18 joints. The rotational-inertia span that decides how degenerate the
    rival is reads 1.53 here against 1.55 on the dog, so the two bodies are NOT independent evidence about
    wide-span robots -- see the residual risk named in ``what_this_gate_does_not_catch``."""
    _assert_band_is_clean(hexa_band, "hexapod")


def test_the_separating_statistic_is_flat_across_the_correct_population(dog_band):
    """WHY the repair is a repair and not a bigger number: the statistic the check now gates on does not move
    with the fit's strength, and the one it used to gate on does.

    Over the dog band the fit's improvement_x spans roughly 1.6x to 4x while ``rival_improvement_x`` stays
    inside a narrow strip. That is the whole claim -- a fit-strength-free statistic can carry a threshold, a
    fit-strength-coupled one cannot.
    """
    from virturoid.services.sysid.fit import LINK_INERTIA_RIVAL_IMPROVEMENT_X

    # Scoped to the ARMATURE-ONLY family on purpose: it is the one whose fit strength varies most while the
    # thing being measured (a uniform addition to diag(M)) stays the same shape, so it is where a
    # fit-strength confound shows up cleanest. Mixed families move the rival's target as well as the fit's.
    rows = [(_row(f)["improvement_x"], _row(f)["rival_improvement_x"], _row(f)["explains_x"], label)
            for label, f in dog_band.items() if label.startswith("armature")]
    rows = [r for r in rows if r[1] is not None]
    assert len(rows) >= 4, f"too few rivals ran to say anything: {rows}"
    imps = [r[0] for r in rows]
    rivs = [r[1] for r in rows]
    assert max(imps) / min(imps) >= 1.8, (
        f"the fit strengths in this band no longer vary enough to expose the confound: {rows}")
    assert max(rivs) / min(rivs) <= 1.6, (
        f"rival_improvement_x is NOT flat across the correct population ({min(rivs)}-{max(rivs)}), which is the "
        f"property the threshold rests on: {rows}")
    assert max(rivs) < LINK_INERTIA_RIVAL_IMPROVEMENT_X, (
        f"a correct calibration reached the refusal threshold on the separating statistic "
        f"({max(rivs)} against {LINK_INERTIA_RIVAL_IMPROVEMENT_X}): {rows}")


def test_a_real_link_inertia_error_in_the_same_band_is_still_refused():
    """THE OTHER HALF, and it has to be in the SAME BAND or it proves nothing about the trade.

    ``inertia_scale x20`` improves tracking 1.620x -- inside the very band the false refusals live in -- and it
    is a genuine misspecification. It stays REFUSED, at a rival_improvement_x an order of magnitude above the
    correct population's. If it ever stops being refused, the repair has bought safety with a hole.
    """
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation, fit_parameters
    from virturoid.services.sysid.fit import LINK_INERTIA_RIVAL_IMPROVEMENT_X
    from virturoid.services.sysid.synthetic_hardware import synthetic_hardware_log

    g = compose_robot("a four legged robot dog", llm=None)
    p = build_excitation(g, budget_s=BUDGET_S)
    _, lg = synthetic_hardware_log(g, perturbation={}, delay_ticks=0, plan=p, inertia_scale=20.0)
    fit = fit_parameters(g, lg, plan=p, n_boot=64, measure_delay=False)
    r = _row(fit)
    assert 1.5 <= r["improvement_x"] <= 3.0, (
        f"this catch is no longer IN the danger band ({r['improvement_x']}x, measured 1.620), so it stops "
        f"being evidence that the repair kept the catch where the false refusals were: {r}")
    assert r["refused_by"] == "link_inertia", f"the catch was lost: {r}"
    assert r["rival_improvement_x"] >= LINK_INERTIA_RIVAL_IMPROVEMENT_X, r
    # The MARGIN, not the boolean. Measured 20.0 against a threshold of 6.0 and a correct-population max of
    # 1.81 -- if this drifts toward the line the two populations are merging and the check needs re-sizing.
    assert r["rival_improvement_x"] >= 12.0, (
        f"the catch's margin on the separating statistic collapsed to {r['rival_improvement_x']} (measured "
        f"20.0). The correct population tops out at 1.81; a catch at 12 or below means the gap is closing")


def test_the_mechanism_claim_matches_the_measured_span_everywhere_it_appears():
    """A SHIPPED FIELD CONTRADICTING ITS OWN DOCSTRING ONE LINE AWAY is the defect this file has already had
    to fix twice, and the link-inertia check shipped with it: four comments said the rotational inertias span
    "~40x" while ``rotational_inertia_span_x`` printed 1.55.

    This test measures the span itself and asserts the source no longer carries the false number.
    """
    import copy
    import inspect

    import mujoco
    import numpy as np

    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import fit as fit_module
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map, start_pose

    for prompt, expected in (("a four legged robot dog", 1.547), ("a six legged walking robot", 1.528)):
        gene = compose_robot(prompt, llm=None)
        model, _ = bench_model(gene)
        dofs = joint_dof_map(model, gene)
        q0 = start_pose(model, gene)

        def diag(m, q0=q0):
            d = mujoco.MjData(m)
            d.qpos[:m.nv] = q0
            mujoco.mj_forward(m, d)
            full = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, full, d.qM)
            return np.diag(full).copy()

        twice = copy.deepcopy(model)
        twice.body_inertia[1:] = np.asarray(twice.body_inertia[1:], dtype=float) * 2.0
        d1, d2 = diag(model), diag(twice)
        rots = [float(d2[a] - d1[a]) for a in dofs.values() if float(d2[a] - d1[a]) > 1e-12]
        span = max(rots) / min(rots)
        assert abs(span - expected) < 0.05, (
            f"{prompt}: the rotational-inertia span is {span:.4f}, not the {expected} this package's "
            f"docstrings now quote. Re-measure the docstrings; do not relax this")
        assert span < 3.0, (
            f"{prompt}: span {span:.4f}. Every claim that a single inertia scalar CANNOT imitate a uniform "
            f"armature offset rests on this being large. It is not")

    # The false claim may only survive as a QUOTED RETRACTION -- this file's convention is to keep the wrong
    # sentence next to the measurement that killed it. What must never come back is an ASSERTED "40x".
    lines = inspect.getsource(fit_module).splitlines()
    markers = ("retract", "used to", "it read", "false", "corrected", "was wrong", "measured via")
    for i, line in enumerate(lines):
        if "40x" not in line:
            continue
        near = " ".join(lines[max(0, i - 12):i + 6]).lower()
        assert any(m in near for m in markers), (
            f"fit.py line {i + 1} states the ~40x rotational-inertia span as fact:\n  {line.strip()}\n"
            f"It is false (measured 1.547x on the dog, 1.528x on the hexapod) and it is the claim that "
            f"justified shipping a check which refuses correct calibrations. It may appear only inside an "
            f"explicit retraction.")

    # ...and it must never reach the CUSTOMER at all. The shipped dict is not a place for a retraction.
    g = compose_robot("a four legged robot dog", llm=None)
    m, _ = bench_model(g)
    d = joint_dof_map(m, g)
    joints = {n: {"parameters": {"armature": {"delta": 0.002}}} for n in d}
    sig = fit_module.link_inertia_signature(m, d, joints, start_pose(m, g))
    blob = " ".join(str(v) for v in sig.values()).lower()
    assert "40x" not in blob, f"the shipped signature still tells the customer the span is ~40x: {sig}"
    assert sig["rotational_inertia_span_x"] is not None and sig["rotational_inertia_span_x"] < 2.0, sig


def test_span_does_not_predict_how_sharp_this_check_is_and_the_prediction_is_retracted(cat_band):
    """THE PREDICTION THAT REPLACED THE FALSE "~40x" IS ITSELF REFUTED, and this is the measurement.

    When the span was corrected from ~40x to 1.55x, the file kept a consolation claim: the span is still worth
    reporting because "a body with a genuinely wide one would make this check SHARPER" -- flagged honestly as a
    prediction rather than a measurement. It is now measured, and it does not order the bodies:

        dog       span  1.547   correct population 0.98 - 1.93
        cat       span  2.628   correct population 3.08 - 3.68   <- WIDER span, WORSE separation
        horse     span  4.040   correct population 1.53 - 1.55
        humanoid  span 14.656   correct population 1.19 - 1.23

    The horse and the humanoid fit the prediction. The cat, whose span sits between the dog's and the horse's,
    reads nearly double either. A non-monotone relation cannot carry an extrapolation, so the prediction is
    retracted in place and NOT replaced with another one.

    The live half of this test is the CAT vs the HORSE: a wider span with a narrower correct population.
    """
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import build_excitation

    horse = compose_robot("a robot horse", llm=None)
    p = build_excitation(horse, budget_s=BUDGET_S)
    horse_row = _row(_fit(horse, p, {"armature": 0.009}))

    cat_rows = [_row(f) for f in cat_band.values()]
    cat_ceiling = max(r["rival_improvement_x"] or 0.0 for r in cat_rows)
    cat_span = next(r["span_x"] for r in cat_rows if r["span_x"])

    assert horse_row["span_x"] > cat_span, (
        f"the horse's rotational-inertia span ({horse_row['span_x']}) is no longer wider than the cat's "
        f"({cat_span}); measured 4.04 vs 2.63. The refutation rests on the WIDER-span body separating BETTER")
    assert cat_ceiling > (horse_row["rival_improvement_x"] or 0.0) * 1.5, (
        f"the cat ({cat_span} span, ceiling {cat_ceiling}) no longer separates markedly WORSE than the horse "
        f"({horse_row['span_x']} span, {horse_row['rival_improvement_x']}). Measured 3.679 vs 1.526. If these "
        f"ever line up with span, re-open the prediction with a measurement -- do not restore it by argument")


def test_no_source_or_output_predicts_that_a_wider_span_makes_this_check_sharper():
    """The retraction has to hold in the SHIPPED DICT as well as in the comments, which is the failure mode
    this file already caught once: a field printing 1.55 one line under a comment claiming 40."""
    import inspect

    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.sysid import fit as fit_module
    from virturoid.services.sysid.bench_rig import bench_model, joint_dof_map, start_pose

    lines = inspect.getsource(fit_module).splitlines()
    markers = ("retract", "used to", "it read", "false", "corrected", "was wrong", "measured", "does not",
               "not a bound", "prediction, not")
    for i, line in enumerate(lines):
        low = line.lower()
        if "sharper" not in low:
            continue
        near = " ".join(lines[max(0, i - 14):i + 8]).lower()
        assert any(m in near for m in markers), (
            f"fit.py line {i + 1} predicts a wider span makes this check sharper, as fact:\n  {line.strip()}\n"
            f"MEASURED: the cat (span 2.63) separates WORSE than the horse (4.04) and the humanoid (14.66). It "
            f"may appear only inside an explicit retraction.")

    g = compose_robot("a four legged robot dog", llm=None)
    m, _ = bench_model(g)
    d = joint_dof_map(m, g)
    joints = {n: {"parameters": {"armature": {"delta": 0.002}}} for n in d}
    sig = fit_module.link_inertia_signature(m, d, joints, start_pose(m, g))
    reading = str(sig.get("rotational_inertia_span_reading", "")).lower()
    assert "sharper" not in reading, (
        f"the shipped signature still tells the customer a wide span makes this check sharper: {reading}")
    assert "does not" in reading or "not used as a bound" in reading, (
        f"the shipped signature no longer says the span is NOT a bound, which is the whole correction: "
        f"{reading}")


def test_the_disclosure_states_the_band_it_sampled_and_does_not_overclaim():
    """The shipped disclosure said "ZERO correctly-specified fits refused" on a sample of nine that never
    entered the region where the check fails. A universal claim from an unsampled distribution is the same
    defect as a wrong number, and it is the one that made this ship."""
    from virturoid.services.sysid.fit import application_gate

    field = application_gate({"improvement_x": 2.0}, 12)["what_this_gate_does_not_catch"]
    low = field.lower()
    # The old universal claim may survive ONLY as a quoted retraction. Anywhere else it is the defect again.
    claim = "zero correctly-specified fits refused"
    at = -1
    while True:
        at = low.find(claim, at + 1)
        if at < 0:
            break
        assert "used to read" in low[max(0, at - 40):at], (
            f"the disclosure asserts an unconditional zero-false-refusal claim at offset {at}: {field}")
    assert claim in low, (
        f"the disclosure no longer RETRACTS the claim it used to make. The retraction is the part a customer "
        f"who read the old output needs: {field}")
    assert "1.5x" in low and "3.0x" in low, (
        f"the disclosure does not state the improvement_x band the false-refusal claim was sampled over: "
        f"{field}")
    for phrase in ("hexapod", "armature-only", "imported", "cat"):
        assert phrase in low, (
            f"the disclosure stopped naming part of what was sampled or what remains at risk ({phrase}): "
            f"{field}")

    # THE THREE NUMBERS THAT SIZE THE CHECK, and the retracted versions of them. A customer who read the old
    # output was told a 10.2x band, then a 3.93x one; the measured band is 3.58x (ceiling 4.037 on the
    # composed centipede against a catch floor of 14.469), and every generation has to stay visible or the
    # correction is invisible to exactly the reader it is for.
    for token in ("3.679", "4.037", "7.3", "14.469"):
        assert token in field, (
            f"the disclosure no longer states the measured correct ceiling / threshold / catch floor "
            f"({token}): {field}")
    for retracted in ("1.930", "6.0", "19.604"):
        assert retracted in field, (
            f"the disclosure stopped RETRACTING the superseded figure {retracted}, which is the part a "
            f"customer who read the old output needs: {field}")

    # ...and the span extrapolation, which was flagged in this very field as "a prediction, not a
    # measurement" and has since been measured and refuted.
    assert "span" in low and "retract" in low, (
        f"the disclosure no longer retracts the claim that a wider rotational-inertia span would make this "
        f"check sharper. Measured on five composed bodies, span does not order them: {field}")

    # The fourth refusal has to be disclosed too: it is the only one that names nothing, and a customer
    # reading a PASS is entitled to know the log itself was checked.
    assert "implausible_log" in low, (
        f"the disclosure does not mention the log-plausibility refusal at all: {field}")
