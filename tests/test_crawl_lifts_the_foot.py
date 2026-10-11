"""THE CRAWL'S SWING PRESSED THE FOOT INTO THE FLOOR ON EVERY BODY WHOSE KNEE BENDS THE OTHER WAY.

During swing ``crawl_gait_rollout`` added ``+knee_amp`` to the lift joint and never asked which way that turns
the leg. The stride direction was measured (``hip_sign``); the lift direction was assumed, and the assumption
holds only on the composed walkers. MEASURED 2026-10-11, foot height for +-0.3 rad on the lift joint at rest: on
the anatomy-compiled walkers (the offline dog, cat, horse, crab and ant, the design schema's quadruped and
hexapod) and the Menagerie quadrupeds (Go2, Go1, A1, Spot, Barkour vB) a positive angle LOWERS the foot by 2 to
4 cm; ANYmal C, knees pointing opposite ways front and rear, lowers two feet and raises two.

What it cost, each body at its own fitted operating point (``fit_gait_for_body``, 150 s budget, no bank):

    body              lift as shipped                          lift measured
    offline dog       no walk found in 360 gaits               CREDIBLE WALK  7.13 m
    offline cat       no walk found                            CREDIBLE WALK 10.47 m
    offline horse     no walk found                            CREDIBLE WALK  9.01 m
    offline crab/ant  CREDIBLE 2.84 m at 20.3 lifts/s (chatter) CREDIBLE WALK  3.55 m at 7.9 lifts/s
    schema hexapod    -2.50 m (backwards), no walk found       CREDIBLE WALK  1.67 m
    schema quadruped  CREDIBLE  6.61 m                         CREDIBLE WALK  8.83 m (default point)
    composed hexapod  CREDIBLE  7.88 m                         identical, byte for byte (its knee lifts on +)

The composed walkers keep their gait exactly: their sign is +1, which is what the law always assumed.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="the crawl gait needs MuJoCo")

_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")


def _legs(gene):
    """(model, data, [(leg, lift_tok, sign)]) exactly as the crawl builds them."""
    import mujoco
    import numpy as np

    from virturoid.services import morph_policy as MP
    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.morph_graph import encode_robot
    model = MP.compiled_model(MP.robot_mjcf(gene), solver_iterations=20)
    data = mujoco.MjData(model); MP._reset_to_rest(model, data); mujoco.mj_forward(model, data)
    graph = encode_robot(model)
    qadr, act_u = np.asarray(graph.qadr, int), np.asarray(graph.act_u, int)
    out = []
    for leg in build_appendage_map(model).legs:
        if not leg.stride_tokens:
            continue
        lift, _ = MP.lift_joint_for_leg(model, data, qadr, act_u, leg, leg.stride_tokens[0])
        out.append((leg, lift, MP.lift_sign_for_leg(model, data, qadr, leg, lift)))
    return model, data, qadr, out


def _foot_rise(model, data, qadr, leg, tok, angle):
    import mujoco

    from virturoid.services.morph_policy import _foot_geom_id
    fg = _foot_geom_id(model, data, int(leg.tip_body))

    def z():
        return float(data.geom_xpos[fg][2]) if fg is not None else float(data.xpos[int(leg.tip_body)][2])
    qa = int(qadr[tok]); q0 = float(data.qpos[qa]); z0 = z()
    data.qpos[qa] = q0 + angle; mujoco.mj_forward(model, data); rise = z() - z0
    data.qpos[qa] = q0; mujoco.mj_forward(model, data)
    return rise


def _schema(name):
    from virturoid.services.agent_design_tools import get_design_schema
    from virturoid.services.anatomy_compiler import build_from_anatomy
    return build_from_anatomy(get_design_schema({})["examples"][name])


@pytest.mark.parametrize("label, make, sign", [
    ("composed quadruped", lambda: __import__("virturoid.services.morphology_composer", fromlist=["x"])
     .compose_robot("a quadruped robot"), 1.0),
    ("offline dog", lambda: __import__("virturoid.services.morphology_composer", fromlist=["x"])
     .compose_robot("a robot dog"), -1.0),
    ("schema hexapod", lambda: _schema("hexapod"), -1.0),
])
def test_the_swing_raises_every_foot(label, make, sign):
    model, data, qadr, legs = _legs(make())
    assert legs, label
    for leg, tok, s in legs:
        assert s == sign, (label, s)
        assert _foot_rise(model, data, qadr, leg, tok, s * 0.3) > 0.005, label


def test_real_quadrupeds_are_measured_not_assumed():
    from virturoid.services.robot_import import import_robot
    cases = {("unitree_go2", "go2.xml"): {-1.0: 4}, ("anybotics_anymal_c", "anymal_c.xml"): {-1.0: 2, 1.0: 2}}
    ran = 0
    for (pkg, model_file), want in cases.items():
        path = os.path.join(_MENAGERIE, pkg, model_file)
        if not os.path.exists(path):
            continue
        model, data, qadr, legs = _legs(import_robot(path)["gene"])
        got = {}
        for leg, tok, s in legs:
            got[s] = got.get(s, 0) + 1
            assert _foot_rise(model, data, qadr, leg, tok, s * 0.3) > 0.005, pkg
        assert got == want, (pkg, got)
        ran += 1
    if not ran:
        pytest.skip("MuJoCo Menagerie not cached")


def test_the_exported_program_lifts_the_way_the_rollout_did():
    """deploy == measure: the frozen program carries each leg's sign and the shipped controller applies it."""
    from virturoid.services.gene_build import _CRAWL_CONTROLLER_SOURCE
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.morphology_composer import compose_robot
    plan = crawl_gait_rollout(compose_robot("a robot dog"), steps=200, return_control_plan=True)["control_plan"]
    assert [leg["lift_sign"] for leg in plan["legs"]] == [-1.0] * len(plan["legs"])
    ns: dict = {}
    exec(compile(_CRAWL_CONTROLLER_SOURCE, "gait_controller.py", "exec"), ns)
    ctrl = ns["GaitController"](plan)
    leg = plan["legs"][0]
    t = ((0.5 * plan["lift_duty"] - leg["phase"]) % 1.0) / plan["frequency_hz"]   # mid-swing for this leg
    target = ctrl.infer(t)
    knee = plan["joint_names"][leg["knee_index"]]
    assert target[knee] < plan["default_pose"][leg["knee_index"]], "the shipped controller must bend this knee DOWN"
    # A package exported before the sign existed has no "lift_sign" and keeps reproducing what it was measured on.
    legacy = ns["GaitController"]({**plan, "legs": [{k: v for k, v in lg.items() if k != "lift_sign"}
                                                     for lg in plan["legs"]]})
    assert legacy.infer(t)[knee] > plan["default_pose"][leg["knee_index"]]


def test_the_schema_hexapod_walks_forward_at_its_own_point():
    """Its leg wave cannot be reversed (three stations at the tripod duty), so a pressed swing had nothing to
    hide behind: shipped, it went 2.5 m backwards and no point was found. This is the point the fitter found
    once the swing lifted."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morph_policy import crawl_gait_rollout
    gene = _schema("hexapod")
    ground_and_repair(gene, task="schema:hexapod")
    # FULL PRECISION ON PURPOSE: rounded to four decimals this point reads SLIDE, because the direction probe's
    # 350-step trials then pick a different rung of its frequency ladder. The fitted point is the fitted point.
    r = crawl_gait_rollout(gene, steps=6000, freq=1.2445606807375278, hip_amp=1.3663290380532422,
                           knee_amp=1.3364688559455762, kp=174.19081563782308, kd=8.359480798570237)
    assert classify(r).startswith("CREDIBLE"), (classify(r), r["forward"])
    assert r["forward"] > 1.0
