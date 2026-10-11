"""The step counter measures FOOT CLEARANCE: one foot per leg, its lowest point, from where it stands.

MEASURED 2026-10-10/11, on composed bodies and on imported Menagerie twins against their own native models. The
counter behind ``cadence``/``support_frac`` (and so behind ``gait_quality.classify``'s SLIDE gate) used to take
every geom whose CENTRE sat within 5 cm of the lowest geom centre at the SPAWN pose, and call a foot lifted once
that centre rose 2 cm above its spawn height. Three things were wrong with that, and each is pinned below:

  * DECORATION. A composed quadruped carries three geoms per leg there, two of them visual-only (a shin collar
    and a foot boot): on the canonical template 8 of the 12 "feet" were visual. Collars ride up whenever a knee
    bends, so the template driven with its knees LOCKED read CREDIBLE WALK at 6000 steps, 13.42 lifts/s.
  * CENTRES. An imported twin draws each link as one capsule from joint to tip, so its foot geom is the whole
    shank and the centre sits mid-shin. The A1 twin at its native's operating point read SLIDE (0.08 lifts/s,
    support 0.003) while, by its floor contacts, every one of its legs stepped.
  * THE SPAWN REFERENCE. The native Go2 spawned 32.2 mm higher, same controller, went from 20.33 lifts/s and
    support 0.843 to 8.75 and 0.366: the reading depended on where the body was dropped from.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
needs_mujoco = pytest.mark.skipif(not _MUJOCO, reason="mujoco is not installed")
_MENAGERIE = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
_GO2_DIR = _MENAGERIE / "unitree_go2"
_A1_XML = _MENAGERIE / "unitree_a1" / "a1.xml"
needs_go2 = pytest.mark.skipif(not (_MUJOCO and (_GO2_DIR / "go2.xml").is_file()),
                               reason="unitree_go2 is not cached locally (robot_descriptions)")
needs_a1 = pytest.mark.skipif(not (_MUJOCO and _A1_XML.is_file()),
                              reason="unitree_a1 is not cached locally (robot_descriptions)")

#: the operating point the product's own search found for the native Go2 (seed 0)
_GO2_OP = {"freq": 1.412, "hip_amp": 0.447, "knee_amp": 1.18, "kp": 66.513, "kd": 9.083}
#: ...and for the native A1 (seed 101), which its twin is driven at below
_A1_OP = {"freq": 1.9818, "hip_amp": 0.4, "knee_amp": 0.7517, "kp": 96.0976, "kd": 8.8052}


def _capsule_quadruped(trunk_z: float = 0.36) -> str:
    """Four one-joint legs, each ONE tilted capsule from hip to foot (as an imported twin draws a link), carrying
    a visual collar mid-shin and a visual boot that hangs 1 cm BELOW the capsule's end. Spawned with the feet 3 cm
    above the floor. The old rule's foot set on this body is the four boots and nothing else."""
    legs, motors = [], []
    for i, (x, y) in enumerate(((0.18, 0.12), (0.18, -0.12), (-0.18, 0.12), (-0.18, -0.12))):
        legs.append(f'''
      <body name="leg{i}" pos="{x} {y} -0.04">
        <joint name="hip{i}" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
        <geom name="shank{i}" type="capsule" fromto="0 0 0 0.05 0 -0.26" size="0.03" mass="0.3"/>
        <geom name="collar{i}" type="sphere" pos="0.025 0 -0.13" size="0.035" contype="0" conaffinity="0" mass="0"/>
        <geom name="boot{i}" type="box" pos="0.05 0 -0.27" size="0.04 0.04 0.03" contype="0" conaffinity="0" mass="0"/>
      </body>''')
        motors.append(f'<motor joint="hip{i}" ctrlrange="-20 20" forcerange="-20 20"/>')
    return f'''<mujoco model="capsule_quadruped">
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="trunk" pos="0 0 {trunk_z}">
      <freejoint/>
      <geom name="trunk_geom" type="box" size="0.2 0.1 0.04" mass="5"/>{"".join(legs)}
    </body>
  </worldbody>
  <actuator>{"".join(motors)}</actuator>
</mujoco>'''


def _feet(xml):
    import mujoco

    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import _FootClearance, _reset_to_rest

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    _reset_to_rest(m, d)
    mujoco.mj_forward(m, d)
    return m, d, _FootClearance(m, d, build_appendage_map(m).legs, encode_robot(m).act_u)


def _template():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a robot dog", llm=None, ensure_walkable=True)


@needs_mujoco
def test_the_foot_is_the_far_end_of_the_shank_and_never_a_visual_geom():
    import mujoco
    import numpy as np

    from virturoid.services.morph_policy import _lowest_point_plan, _lowest_points

    m, d, feet = _feet(_capsule_quadruped())
    assert feet.n == 4, f"one foot per leg, got {feet.n}"
    assert feet.floor == 0.0
    # tilt the whole body so the capsules lean, then compare with the capsule's own geometry: its lowest point is
    # the lower end-cap, at (lower end) - radius, and the boot that hangs below it must not be read instead
    d.qpos[3:7] = [np.cos(0.2), np.sin(0.2), 0.0, 0.0]
    mujoco.mj_forward(m, d)
    got = feet.heights(d)
    for i in range(4):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"shank{i}")
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
        ends = (d.geom_xpos[g] + ax, d.geom_xpos[g] - ax)
        want = min(float(e[2]) for e in ends) - float(m.geom_size[g][0])
        assert got[i] == pytest.approx(want, abs=1e-9), f"leg {i}: read {got[i]:.4f}, its shank ends at {want:.4f}"
        boot = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"boot{i}")])
        assert _lowest_points(m, d, boot, _lowest_point_plan(m, boot))[0] < got[i] - 0.005, (
            "premise: the visual boot reaches lower than the shank")


@needs_mujoco
def test_every_primitive_is_measured_at_its_lowest_surface_point():
    """Against dense surface samples, at an arbitrary tilt. A mesh is bounded by its box, so it may only read LOW."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_policy import _lowest_point_plan, _lowest_points

    xml = '''<mujoco><asset><mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.08 0  0.02 0.03 0.12"/></asset>
      <worldbody><body pos="0 0 1" euler="0.7 -0.4 1.1"><freejoint/>
        <geom type="sphere" size="0.05"/>
        <geom type="capsule" size="0.03 0.1" pos="0.3 0 0" euler="0.3 0.5 0"/>
        <geom type="cylinder" size="0.04 0.07" pos="0.6 0 0" euler="1.0 0.2 0"/>
        <geom type="ellipsoid" size="0.05 0.08 0.03" pos="0.9 0 0" euler="0.4 0.9 0.1"/>
        <geom type="box" size="0.05 0.03 0.07" pos="1.2 0 0" euler="0.2 0.3 0.6"/>
        <geom type="mesh" mesh="tet" pos="1.5 0 0"/>
      </body></worldbody></mujoco>'''
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    idx = np.arange(m.ngeom)
    got = _lowest_points(m, d, idx, _lowest_point_plan(m, idx))
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for g in range(m.ngeom):
        R, c, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        t = int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            a, n = int(m.mesh_vertadr[m.geom_dataid[g]]), int(m.mesh_vertnum[m.geom_dataid[g]])
            true_low = float(np.min((m.mesh_vert[a:a + n] @ R.T + c)[:, 2]))
            assert got[g] <= true_low + 1e-9, "a mesh's box bound read ABOVE its real surface"
            continue
        if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            pts = u * s[0]
        elif t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            pts = u * s[0] + np.outer(np.sign(u[:, 2]), [0, 0, s[1]])
        elif t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            ang = rng.uniform(0, 2 * np.pi, 200000)
            pts = np.stack([s[0] * np.cos(ang), s[0] * np.sin(ang), rng.choice([-s[1], s[1]], 200000)], axis=1)
        elif t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            pts = u * s
        else:
            pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
        sampled = float(np.min((pts @ R.T + c)[:, 2]))
        assert got[g] <= sampled + 1e-9 and got[g] >= sampled - 2e-4, (t, got[g], sampled)


@needs_mujoco
def test_a_foot_is_measured_from_where_it_stands_not_where_it_spawned():
    """Synthetic heights on the capsule body: spawned 3 cm up, standing 1.5 cm INTO a soft floor (the Menagerie
    feet stand 1.4 cm into it while walking). A foot that rises 2.5 cm from there is one lift; a foot that rises
    1.5 cm is none, though it reaches the floor surface. Where it spawned, and how far the floor lets it sink,
    change nothing."""
    import numpy as np

    _, _, feet = _feet(_capsule_quadruped())
    h = np.full((600, 4), -0.015)
    h[0] = 0.03                                               # the spawn row
    h[100:150, 0] = -0.015 + 0.025                            # clears 2 cm above where it stands
    h[300:350, 1] = -0.015 + 0.015                            # never does
    assert feet.count(h) == (1, 50)
    spawned_at_stance = h.copy()
    spawned_at_stance[0] = -0.015
    assert feet.count(spawned_at_stance) == (1, 50), "the spawn height moved the count"
    stiffer_floor = h + 0.012                                 # the same feet on a floor they sink 3 mm into
    assert feet.count(stiffer_floor) == (1, 50), "how far a foot sinks moved the count"


@needs_mujoco
def test_decoration_is_never_a_foot_on_the_canonical_template():
    """One foot per leg, and no visual geom among the geoms a foot is read from. The old rule's set on this very
    body is recomputed here only to show what it was: 12 geoms, 8 of them visual collars and boots."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_policy import robot_mjcf

    m, d, feet = _feet(robot_mjcf(_template()))
    assert feet.n == 4, f"one foot per leg, got {feet.n}"
    visual = [int(g) for g in feet._idx if not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in visual]
    assert not visual, f"visual geoms read as feet: {names}"
    gz = np.asarray(d.geom_xpos[:, 2])
    body = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) != 0]
    old = [g for g in body if gz[g] < min(gz[b] for b in body) + 0.05]
    old_visual = [g for g in old if not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))]
    assert (len(old), len(old_visual)) == (12, 8), "premise: the old rule's set on this body (12 geoms, 8 visual)"


@needs_mujoco
def test_a_knees_locked_template_does_not_read_as_a_walk():
    """The defect's headline case, at the settling horizon the product decides at: the canonical template driven
    with ``knee_amp=0``. The old rule read CREDIBLE WALK here, 13.42 lifts/s and support 0.995, almost all of it
    collars and boots riding up as the hips swung. MEASURED with contact ground truth on this run: every foot does
    leave the floor (duty 0.47-0.52), but its flights barely clear it (median flight peak 0.3-0.6 cm, p90 under
    2 cm), so the gate's own words, "feet barely lift", are the true description. The counter reads 0.75 lifts/s
    and support 0.097.

    NOT A CLAIM ABOUT LOCKED KNEES IN GENERAL. On bodies whose hip stride swings the foot through a high arc a
    stiff-legged gait is a real walk, and the counter says so: the composed mouse with ``knee_amp=0`` carries
    92-104% of its travel through the air on every leg and reads CREDIBLE WALK (6.92 lifts/s, support 0.668)."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morph_policy import crawl_gait_rollout

    g = _template()
    r = crawl_gait_rollout(g, steps=6000, record_qpos=True, knee_amp=0.0)
    assert r["n_feet"] == 4, "one foot per leg"
    assert classify(r).startswith("SLIDE"), classify(r)
    assert r["cadence"] < 1.0 and r["support_frac"] < 0.25, (r["cadence"], r["support_frac"])


@needs_mujoco
def test_the_recipe_and_the_crawl_count_the_same_feet():
    """Both verdict paths read one foot definition, and the recipe's upright bar follows it. The bar is per LEG
    count by design (``upright_height_ratio``: 0.70 for a quadruped, 0.55 for a hexapod); with geoms for feet the
    template's recipe rollouts counted 16 "feet" at its zero pose and judged a quadruped by the octopod's 0.50."""
    from virturoid.services.morph_policy import crawl_gait_rollout, recipe_rollout_morph
    from virturoid.services.morphology_composer import compose_robot

    for gene, legs, tau in ((_template(), 4, 0.70), (compose_robot("a hexapod robot", llm=None), 6, 0.55)):
        rec = recipe_rollout_morph(gene, None, steps=60, seed=0)
        crawl = crawl_gait_rollout(gene, steps=60)
        assert rec["n_feet"] == crawl["n_feet"] == legs, (rec["n_feet"], crawl["n_feet"], legs)
        assert rec["upright_tau"] == pytest.approx(tau)


@needs_mujoco
def test_the_exported_program_is_judged_by_the_crawls_feet():
    """``gene_build._run_exported_controller`` verifies the frozen crawl program a package ships; it must count
    feet the way the crawl that produced the program counted them."""
    from virturoid.services import gene_build as GB
    from virturoid.services.morph_policy import crawl_gait_rollout

    g = _template()
    r = crawl_gait_rollout(g, steps=400, return_control_plan=True)
    plan = r["control_plan"]
    assert plan["exportable"]
    shipped = GB._run_exported_controller(g, plan, GB._CRAWL_CONTROLLER_SOURCE, steps=400)
    assert shipped["n_feet"] == r["n_feet"] == 4


def _native_go2(dz: float = 0.0) -> str:
    """The source Go2 as one string (as tests/test_imported_twin_moves_like_its_source.py builds it), its rest pose
    raised by ``dz``."""
    x = (_GO2_DIR / "go2.xml").read_text(encoding="utf-8")
    x = x.replace('meshdir="assets"', f'meshdir="{(_GO2_DIR / "assets").as_posix()}"')
    x = x.replace('<motor ctrlrange="-23.7 23.7"/>', '<motor ctrlrange="-23.7 23.7" forcerange="-23.7 23.7"/>')
    x = x.replace('<motor ctrlrange="-45.43 45.43"/>', '<motor ctrlrange="-45.43 45.43" forcerange="-45.43 45.43"/>')
    x = x.replace("<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1"/>', 1)
    home = [float(v) for v in re.search(r'<key name="home" qpos="([^"]+)"', x).group(1).split()]
    home[2] += dz
    return x.replace("</keyframe>", f'  <key name="rest" qpos="{" ".join(repr(v) for v in home)}"/>\n  </keyframe>')


@needs_go2
def test_spawning_higher_does_not_change_what_the_feet_did():
    """Same robot, same controller, spawned 32.2 mm higher. It walks either way and the counter must say so in the
    same numbers. MEASURED: old rule 20.33 -> 8.75 lifts/s and support 0.843 -> 0.366; this counter 6.33 -> 6.75
    and 0.791 -> 0.795 (the two rollouts are different trajectories, so the numbers are close, not equal)."""
    from virturoid.services.gait_search import evaluate_gait

    low = evaluate_gait(_native_go2(), _GO2_OP, steps=6000)
    high = evaluate_gait(_native_go2(0.0322), _GO2_OP, steps=6000)
    assert low["verdict"] == "CREDIBLE WALK" and high["verdict"] == "CREDIBLE WALK", (low["verdict"], high["verdict"])
    assert abs(low["support_frac"] - high["support_frac"]) < 0.05, (low["support_frac"], high["support_frac"])
    assert abs(low["cadence"] - high["cadence"]) < 1.0, (low["cadence"], high["cadence"])


@needs_go2
def test_a_go2_whose_feet_never_leave_the_floor_reads_slide():
    """The SLIDE gate keeps its meaning on a body with no decoration at all. With the knee pulse off, the native
    Go2's feet never leave the floor (contact on every step of every leg, measured) and nothing may count as a
    step."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morph_policy import crawl_gait_rollout

    r = crawl_gait_rollout(_native_go2(), steps=6000, record_qpos=True, **dict(_GO2_OP, knee_amp=0.0))
    assert classify(r).startswith("SLIDE"), classify(r)
    assert r["cadence"] == 0.0 and r["n_feet"] == 4, (r["cadence"], r["n_feet"])


@pytest.fixture(scope="module")
def a1_twin(tmp_path_factory):
    """The imported A1, every anchored build path kept out of the checkout."""
    from virturoid.services import install_paths
    from virturoid.services.robot_import import import_robot

    root = tmp_path_factory.mktemp("a1_twin")
    saved = install_paths.anchored
    install_paths.anchored = lambda rel: root / rel
    try:
        gene = import_robot(str(_A1_XML), robot_id="a1_twin_counter")["gene"]
        assert gene is not None
        return gene
    finally:
        install_paths.anchored = saved


@needs_a1
def test_an_imported_twins_real_steps_count(a1_twin):
    """The A1 twin at the operating point its native walks at. By its floor contacts every leg steps: airborne
    29-65% of the time, its flights carrying 63-93% of the body's travel at 1.65-2.0 flights/s that clear 1 cm.
    The old rule read 0.08 lifts/s and support 0.003 and called it SLIDE, because the "feet" it watched were the
    shank capsules' mid-shin centres and the twin's visual meshes. MEASURED with this counter: CREDIBLE WALK,
    6.83 lifts/s, support 0.950."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morph_policy import crawl_gait_rollout

    r = crawl_gait_rollout(a1_twin, steps=6000, record_qpos=True, **_A1_OP)
    assert r["n_feet"] == 4, "one foot per leg"
    assert r["cadence"] >= 1.0 and r["support_frac"] >= 0.25, (r["cadence"], r["support_frac"])
    assert classify(r) == "CREDIBLE WALK", classify(r)
