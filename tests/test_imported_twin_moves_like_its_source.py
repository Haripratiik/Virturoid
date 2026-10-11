"""An imported twin must stand, step and be judged like the robot it was imported from.

Measured on the Menagerie Unitree Go2, 2026-10-10. The product's own crawl gait search (3 seeds, 6000 steps)
found a CREDIBLE WALK for the source model on its first seed, +2.36 m. Its twin had never walked on any seed,
and four separate defects stood between them. All four are fixed and pinned here, each against the SOURCE model
rather than against a number we chose:

  * MASS DISTRIBUTION. The compiler took every link's inertia from the primitive collider drawn for it, and the
    trunk's capsule runs from the trunk origin to its farthest hip. That put 6.92 kg of a 15.2 kg robot 0.188 m
    forward of where the source has it, and the whole robot's centre of mass 5.7 cm forward and 2 cm left.
  * FOOT PLACEMENT. A leaf link's length was read off every geom it owns, visual mesh included, so the calf ran
    0.284 m toward the mesh centroid instead of 0.213 m to the foot: every foot 6.6 cm ahead of the source's.
  * THE BASE FRAME. The importer turns the root so the trunk is drawn along the gene's local +z, and the compiled
    base body kept that rotation. A twin standing level read as lying on its side to everything that reads the
    base orientation (64 places): the crawl's uprightness term averaged -0.085 against the source's +0.986, so
    every gait the search found for it was judged CROUCH.
  * THE CONTACT MODEL. The source's feet are condim 6, priority 1, friction 0.8 0.02 0.01, with a soft solimp.
    Under one fixed controller the twin travelled 0.750 m with the compiler's contact defaults on its feet and
    2.347 m with the source's, against 2.353 m for the source.

Fixing the first three made the twin walk (CREDIBLE WALK +0.99 m); the base frame alone did not (CROUCH, height
ratio 0.70). The fourth closed most of the remaining distance.

Two collider fixes then made its contacts the source's too. Every link's radius came from its visual mesh's
bounding box (thigh 0.051 m around a source thigh 0.017 m thick), and the trunk was a capsule from its origin to
the front-left hip that overhung the source's nose by 0.12 m; the gait the twin's seed-0 search ended on rested
on that overhang for a quarter of the run, where the source falls. Judged gait by gait against the source on a
fixed sample of 252 controllers, the twin's verdict agreed on 183 before and 221 after.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_GO2_DIR = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2"))
_HAVE_GO2 = _MUJOCO and (_GO2_DIR / "go2.xml").is_file()
needs_go2 = pytest.mark.skipif(not _HAVE_GO2, reason="unitree_go2 is not cached locally (robot_descriptions)")

_LEGS = ("FL", "FR", "RL", "RR")


def _native_xml() -> str:
    """The source Go2 as one string: its own physics, a floor, a ``rest`` key, and its declared torque limits.

    The crawl clamps torque at ``forcerange`` and falls back to 10 N.m without one; the Go2 states its limits as
    ``ctrlrange`` on pure motors, so they are restated as ``forcerange`` here, which changes nothing else.
    """
    x = (_GO2_DIR / "go2.xml").read_text(encoding="utf-8")
    x = x.replace('meshdir="assets"', f'meshdir="{(_GO2_DIR / "assets").as_posix()}"')
    x = x.replace('<motor ctrlrange="-23.7 23.7"/>', '<motor ctrlrange="-23.7 23.7" forcerange="-23.7 23.7"/>')
    x = x.replace('<motor ctrlrange="-45.43 45.43"/>', '<motor ctrlrange="-45.43 45.43" forcerange="-45.43 45.43"/>')
    x = x.replace("<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="5 5 0.1"/>', 1)
    home = re.search(r'<key name="home" qpos="([^"]+)"', x).group(1)
    return x.replace("</keyframe>", f'  <key name="rest" qpos="{home}"/>\n  </keyframe>')


@pytest.fixture(scope="module")
def go2(tmp_path_factory):
    """``(gene, twin_xml, native_xml)``, imported once, with every anchored build path kept out of the checkout."""
    from virturoid.services import install_paths
    from virturoid.services.morph_policy import robot_mjcf
    from virturoid.services.robot_import import import_robot

    root = tmp_path_factory.mktemp("go2_twin")
    saved = install_paths.anchored
    install_paths.anchored = lambda rel: root / rel
    try:
        gene = import_robot(str(_GO2_DIR / "go2.xml"), robot_id="go2_twin_fidelity")["gene"]
        assert gene is not None
        return gene, robot_mjcf(gene), _native_xml()
    finally:
        install_paths.anchored = saved


def _at_rest(xml):
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "rest"))
    mujoco.mj_forward(m, d)
    return m, d


def _body(m, name):
    import mujoco

    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


@needs_go2
def test_every_link_keeps_the_sources_centre_of_mass(go2):
    """Per link, relative to the base, so the twin's own spawn height does not enter the comparison."""
    import numpy as np

    _, twin, native = go2
    mt, dt = _at_rest(twin)
    mn, dn = _at_rest(native)
    worst, where = 0.0, None
    for b in range(1, mn.nbody):
        name = mn.body(b).name
        bt = _body(mt, name)
        if bt < 0:
            continue
        err = float(np.linalg.norm((dt.xipos[bt] - dt.xpos[1]) - (dn.xipos[b] - dn.xpos[1])))
        if err > worst:
            worst, where = err, name
    assert worst < 1e-4, f"{where}'s centre of mass is {worst * 1000:.1f} mm from the source's"
    whole = float(np.linalg.norm((dt.subtree_com[1] - dt.xpos[1]) - (dn.subtree_com[1] - dn.xpos[1])))
    assert whole < 1e-4, f"the whole robot's centre of mass is {whole * 1000:.1f} mm from the source's"


@needs_go2
def test_the_feet_land_where_the_source_plants_them(go2):
    """The bottom of each calf collider against the bottom of the source's foot sphere, measured from the hip."""
    import mujoco
    import numpy as np

    _, twin, native = go2
    mt, dt = _at_rest(twin)
    mn, dn = _at_rest(native)
    for leg in _LEGS:
        hip_t = dt.xanchor[mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")]
        hip_n = dn.xanchor[mujoco.mj_name2id(mn, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_thigh_joint")]
        g = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_calf_geom")
        ax = dt.geom_xmat[g].reshape(3, 3)[:, 2] * mt.geom_size[g][1]
        tip = min((dt.geom_xpos[g] + ax, dt.geom_xpos[g] - ax), key=lambda p: p[2]) - [0, 0, mt.geom_size[g][0]]
        foot = mujoco.mj_name2id(mn, mujoco.mjtObj.mjOBJ_GEOM, leg)
        sole = dn.geom_xpos[foot] - [0, 0, mn.geom_size[foot][0]]
        off = float(np.linalg.norm((tip - hip_t) - (sole - hip_n)))
        assert off < 0.01, f"{leg} foot lands {off * 100:.1f} cm from where the source plants it"


@needs_go2
def test_the_floating_base_frame_is_the_sources(go2):
    """Standing at rest, the twin's base reads level, as the source's does, while every geom stays put."""
    import numpy as np

    _, twin, native = go2
    mt, dt = _at_rest(twin)
    mn, dn = _at_rest(native)
    assert np.allclose(dt.xmat[1], dn.xmat[1], atol=1e-9), "the twin's base frame is rotated against the source's"
    q = dt.qpos[3:7]
    assert 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2) > 0.999, "the crawl's uprightness term reads a level twin as tipped"


@needs_go2
def test_the_feet_carry_the_sources_contact_model(go2):
    import mujoco
    import numpy as np

    _, twin, native = go2
    mt, mn = mujoco.MjModel.from_xml_string(twin), mujoco.MjModel.from_xml_string(native)
    for leg in _LEGS:
        t = mujoco.mj_name2id(mt, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_calf_geom")
        n = mujoco.mj_name2id(mn, mujoco.mjtObj.mjOBJ_GEOM, leg)
        assert int(mt.geom_condim[t]) == int(mn.geom_condim[n]) == 6
        assert int(mt.geom_priority[t]) == int(mn.geom_priority[n])
        for field in ("geom_friction", "geom_solimp", "geom_solref", "geom_margin"):
            assert np.allclose(getattr(mt, field)[t], getattr(mn, field)[n], atol=1e-6), (leg, field)


@needs_go2
def test_resizing_a_link_retires_only_its_own_inertial(go2):
    """A centre of mass recorded for a 0.213 m calf says nothing about a longer one, so that link falls back."""
    import copy

    import mujoco
    import numpy as np

    from virturoid.services.morph_policy import robot_mjcf

    gene = copy.deepcopy(go2[0])
    calf = gene.segment("FL_calf")
    calf.length_m *= 1.2
    m = mujoco.MjModel.from_xml_string(robot_mjcf(gene))
    resized, kept = _body(m, "FL_calf"), _body(m, "FR_calf")
    assert np.allclose(m.body_ipos[resized], [0.0, 0.0, calf.length_m / 2.0], atol=1e-6), (
        "a resized link kept the centre of mass recorded for the link it used to be")
    row = gene.metadata["source_inertials"]["FR_calf"]
    assert np.allclose(m.body_ipos[kept], row["pos"], atol=1e-6), "an untouched link lost its recorded inertial"


def _trunk_collision(m, d):
    """The trunk's collision geoms in the BASE frame at rest: ``(type, size, centre, base-frame AABB, contact)``."""
    import numpy as np

    b = _body(m, "base")
    Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    rows = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != b or not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g])):
            continue
        R = Rb.T @ d.geom_xmat[g].reshape(3, 3)
        centre = Rb.T @ (d.geom_xpos[g] - pb)
        corners = centre + (m.geom_aabb[g][:3] + signs * m.geom_aabb[g][3:]) @ R.T
        contact = (int(m.geom_condim[g]), *np.round(m.geom_friction[g], 6), *np.round(m.geom_solref[g], 6),
                   *np.round(m.geom_solimp[g], 6), round(float(m.geom_margin[g]), 6))
        rows.append((int(m.geom_type[g]), tuple(np.round(m.geom_size[g], 6)), centre,
                     np.concatenate([corners.min(axis=0), corners.max(axis=0)]), contact))
    return sorted(rows, key=lambda r: (r[0], r[1]))


@needs_go2
def test_the_trunk_collides_where_the_sources_trunk_does(go2):
    """The trunk's collision volume is the source's: a centred box, a head cylinder and a chin sphere.

    A segment's collider runs from its origin along its local +z, which suits a limb and not a trunk, whose origin
    is its centre. Drawn that way, the Go2's trunk was a capsule from its origin to the front-left hip, 0.376 m
    long with a 0.098 m radius: in the base frame x -0.098..+0.464 and y -0.098..+0.186, against the source's
    x -0.188..+0.340 and y -0.050..+0.050. Inertia is carried separately, so this only ever mattered on contact,
    and it did: under the gait the twin's own seed-0 search ended on, the trunk was on the floor for 1475 of 6000
    steps, propped on the overhang.
    """
    import numpy as np

    _, twin, native = go2
    mt, dt = _at_rest(twin)
    mn, dn = _at_rest(native)
    got, want = _trunk_collision(mt, dt), _trunk_collision(mn, dn)
    assert [(r[0], r[1]) for r in got] == [(r[0], r[1]) for r in want], (
        f"the twin's trunk is drawn as {[(r[0], r[1]) for r in got]}, the source's is "
        f"{[(r[0], r[1]) for r in want]} (type, size)")
    for (t, size, c_got, box_got, k_got), (_, _, c_want, box_want, k_want) in zip(got, want):
        assert np.allclose(c_got, c_want, atol=1e-5), f"geom {t} {size}: centre {c_got} against {c_want}"
        assert np.allclose(box_got, box_want, atol=1e-5), f"geom {t} {size}: extent {box_got} against {box_want}"
        assert k_got == k_want, f"geom {t} {size}: contact model {k_got} against the source's {k_want}"


@needs_go2
def test_the_physics_only_trunk_keeps_its_extents_without_a_cylinder(go2):
    """MJX has no cylinder collision, so the GPU model draws the head as the box with the cylinder's extents."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(go2[0], physics_only=True))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rows = _trunk_collision(m, d)
    box, sphere = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_SPHERE)
    kinds = sorted(int(r[0]) for r in rows)
    assert kinds == sorted([box, box, sphere]), f"the physics-only trunk is drawn with geom types {kinds}"
    head = next(r for r in rows if np.allclose(r[1], [0.05, 0.05, 0.045], atol=1e-6))
    assert np.allclose(head[3], [0.235, -0.05, -0.035, 0.335, 0.05, 0.055], atol=1e-5), head[3]


@needs_go2
def test_resizing_the_trunk_retires_its_source_collision(go2):
    """The record describes the trunk it was read from, so a longer trunk is drawn from its own dimensions."""
    import copy

    import mujoco
    import numpy as np

    from virturoid.services.morph_policy import robot_mjcf

    gene = copy.deepcopy(go2[0])
    root = gene.root()
    root.length_m *= 1.2
    m = mujoco.MjModel.from_xml_string(robot_mjcf(gene))
    b = _body(m, "base")
    solid = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b
             and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))]
    assert len(solid) == 1, f"a resized trunk kept {len(solid)} collision geoms recorded for the trunk it was"
    g = solid[0]
    assert int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert np.allclose(m.geom_size[g][:2], [root.radius_m, root.length_m / 2.0], atol=1e-5)


def test_a_composed_body_compiles_as_it_did():
    """The inertial and contact records are written only by the importer, and the composer leaves a dog's root
    unrotated, so a composed dog meets none of the three changes."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morphology_composer import _compose_robot_impl

    dog = _compose_robot_impl("a quadruped robot dog", llm=None)
    xml = compile_gene_to_mjcf(dog)
    assert "<inertial" not in xml and "<frame" not in xml and "solmix=" not in xml


def test_the_frame_wrap_moves_no_geom_on_the_composed_body_it_reaches(monkeypatch):
    """The composer rotates one free root, a snake's head laid along +x. The wrap may change what its base frame
    reads, from pitched 90 degrees (gimbal lock for any heading taken from it) to level, and nothing else."""
    import mujoco
    import numpy as np

    from virturoid.services import gene_compiler as gc
    from virturoid.services.morphology_composer import _compose_robot_impl

    snake = _compose_robot_impl("a snake robot", llm=None)
    assert gc._framed_root(snake, snake.root()) is not None, "premise: the composed snake's head is rotated"

    def at_rest():
        m = mujoco.MjModel.from_xml_string(gc.compile_gene_to_mjcf(snake))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        return d

    wrapped = at_rest()
    monkeypatch.setattr(gc, "_framed_root", lambda gene, seg: None)
    unwrapped = at_rest()
    assert np.allclose(wrapped.geom_xpos, unwrapped.geom_xpos, atol=1e-4), "the wrap moved a geom"
    assert np.allclose(wrapped.geom_xmat, unwrapped.geom_xmat, atol=1e-4), "the wrap turned a geom"
    level = lambda q: 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
    assert level(wrapped.qpos) > 0.999 and abs(level(unwrapped.qpos)) < 1e-3


def test_the_frame_wrap_moves_nothing_placed_on_the_root(monkeypatch):
    """Sensor geoms and loop-closure anchors on a framed root are written in its SEGMENT frame, so every reader
    that turns world directions into that frame has to use the segment's rotation, not the compiled body's. One
    did not: an arm meant for the top of an imported Go2 grew straight out of its nose until
    ``edit_operators._parent_frame`` composed the frame back in. This pins the other two such readers."""
    import mujoco
    import numpy as np

    from virturoid.services import gene_compiler as gc
    from virturoid.services.bom_builder import _sensor_suite
    from virturoid.services.morphology_composer import _compose_robot_impl
    from virturoid.services.sensor_geometry import plan_sensor_geoms

    snake = _compose_robot_impl("a snake robot", llm=None)
    root = snake.root()
    assert gc._framed_root(snake, root) is not None, "premise: the composed snake's head is rotated"
    child = next(s.name for s in snake.segments if s.parent == root.name)
    grandchild = next(s.name for s in snake.segments if s.parent == child)     # a loop may not join parent+child
    snake.loop_closures = [{"a": root.name, "b": grandchild, "anchor": [0.01, 0.02, 0.03]}]
    suite = _sensor_suite("quadruped", None, "", 10.0)

    def placed():
        geoms = plan_sensor_geoms(snake, suite)
        assert geoms.get(root.name), "premise: some sensor is mounted on the root"
        m = mujoco.MjModel.from_xml_string(gc.compile_gene_to_mjcf(snake, sensor_geoms=geoms))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root.name)
        anchor = d.xpos[a] + d.xmat[a].reshape(3, 3) @ m.eq_data[0][:3]
        on_root = np.array([d.geom_xpos[g] for g in range(m.ngeom) if int(m.geom_bodyid[g]) == a])
        return anchor, on_root

    wrapped = placed()
    monkeypatch.setattr(gc, "_framed_root", lambda gene, seg: None)
    unwrapped = placed()
    assert np.allclose(wrapped[0], unwrapped[0], atol=1e-6), "the wrap moved a loop-closure anchor"
    assert wrapped[1].shape == unwrapped[1].shape and np.allclose(wrapped[1], unwrapped[1], atol=1e-4), (
        "the wrap moved a sensor or a geom on the root")


@needs_go2
def test_the_twin_walks_like_its_source_under_one_controller(go2):
    """The claim itself, as a comparison: the same gait on both bodies, judged by the same verdict.

    The operating point is the one the product's own search found for the SOURCE (seed 0), not one tuned for
    the twin, so passing means the twin reproduces the source rather than that a gait exists for it.

    That search ran while the crawl's swing pressed the Go2's feet down instead of lifting them (see
    ``morph_policy.lift_sign_for_leg``). The point is still a walk for both once the swing lifts: measured
    2026-10-11, source +2.997 m, twin +3.032 m with the source's ``impratio`` (``robot_import`` now carries it;
    without it the twin went +2.293 m, 23% short).
    """
    from virturoid.services.gait_search import evaluate_gait

    _, twin, native = go2
    p = {"freq": 1.412, "hip_amp": 0.447, "knee_amp": 1.18, "kp": 66.513, "kd": 9.083}
    src = evaluate_gait(native, p, steps=6000)
    got = evaluate_gait(twin, p, steps=6000)
    assert src["verdict"] == "CREDIBLE WALK", f"precondition: the source walks at this point ({src['verdict']})"
    assert got["verdict"] == "CREDIBLE WALK", f"the twin did not walk where its source does: {got['verdict']}"
    assert abs(got["forward"] - src["forward"]) < 0.15 * src["forward"], (
        f"the twin travelled {got['forward']:+.3f} m where its source travelled {src['forward']:+.3f} m")


#: What the product's search (6000 steps, stop at the first credible) finds for the SOURCE on seeds 0-2 now that
#: the swing lifts the feet: +11.64, +5.93 and +19.08 m, against +2.36 m on its best seed before.
_SOURCE_SEARCH_POINTS = (
    {"freq": 2.075438132656036, "hip_amp": 0.9136711625948919, "knee_amp": 1.1601056626108206,
     "kp": 137.66460632626413, "kd": 5.75907453722639},
    {"freq": 2.2073505152388715, "hip_amp": 1.1759449894628184, "knee_amp": 1.0826092690458469,
     "kp": 61.62950949336451, "kd": 10.442406566687632},
    {"freq": 2.6864995232223374, "hip_amp": 0.8605087198613482, "knee_amp": 1.1934516466819154,
     "kp": 147.18537616907307, "kd": 5.7000757816218295},
)


@needs_go2
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_twin_is_judged_like_its_source_at_the_gaits_its_source_now_finds(go2, seed):
    """The verdict agrees at every gait the search finds for the source. The DISTANCE does not, yet: at these
    faster gaits the twin went +15.64 / +9.06 / +14.16 m against the source's +11.64 / +5.93 / +19.08 m (with the
    source's impratio; its integrator and start height were measured and change nothing). That gap is open and is
    stated here rather than hidden behind a tolerance wide enough to swallow it."""
    from virturoid.services.gait_search import evaluate_gait

    _, twin, native = go2
    p = _SOURCE_SEARCH_POINTS[seed]
    src = evaluate_gait(native, p, steps=6000)
    got = evaluate_gait(twin, p, steps=6000)
    assert src["verdict"] == "CREDIBLE WALK", f"precondition: the source walks at this point ({src['verdict']})"
    assert got["verdict"] == src["verdict"], f"the twin was judged {got['verdict']!r} where its source walks"
