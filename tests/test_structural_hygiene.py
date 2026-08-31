"""WS-I — clipping/penetration budgets as a standing CI measurement (master_plan_v6 WS-I)."""
from __future__ import annotations

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import structural_hygiene as SH
from virturoid.services.design_battery import battery, prompt_id
from virturoid.services.design_cassette import DesignCassette


def _clean_quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a sturdy four-legged walking robot", ensure_walkable=True)


def _diverging_gene():
    """A body that provably blows the simulation up during the settle, with no control applied.

    The mechanism is a REAL one a customer file can carry, not a synthetic poke: a closed-loop machine whose
    ``<equality><connect>`` is given a hand-written solver reference. MuJoCo reads a NEGATIVE ``solref`` as a
    direct (stiffness, damping) pair, so ``[-1e9, -1e6]`` is an absurdly stiff constraint — and the import path
    carries ``solref`` straight off the source (Cassie ships its own). Two segments mounted 2 m apart and then
    welded together by that constraint diverge in a handful of steps.
    """
    segments = [
        GeneSegment(name="root", parent=None, shape="box", length_m=0.4, radius_m=0.2, mass_kg=1e4),
        GeneSegment(name="a", parent="root", shape="capsule", length_m=0.5, radius_m=0.05, mass_kg=1e4,
                    joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-3.0, joint_upper=3.0),
        GeneSegment(name="b", parent="root", shape="capsule", length_m=0.5, radius_m=0.05, mass_kg=1e-3,
                    joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-3.0, joint_upper=3.0,
                    mount_offset=(2.0, 0.0, 0.0), is_end_effector=True),
    ]
    gene = RobotGene(id="diverging", species="test.diverging", robot_class="manipulator", base_mount="free",
                     end_effector_type="none", segments=segments,
                     loop_closures=[{"a": "a", "b": "b", "solref": [-1e9, -1e6]}])
    assert gene.validate() == []          # a SCHEMA-valid body: nothing here is caught by validation
    return gene


def test_penetration_report_on_a_real_body_is_within_budget():
    r = SH.penetration_report(_clean_quad())
    assert r["ok"] and r["finite"] and r["within_budget"]
    assert r["max_penetration_m"] <= r["budget_m"]


def test_connectivity_passes_a_valid_tree_and_catches_a_floating_piece():
    assert SH.connectivity_report(_clean_quad())["ok"]
    # a body with TWO roots -> the second root's subtree is unreachable from root() -> a disconnected piece
    twin = RobotGene(id="twin", species="t", robot_class="quadruped", base_mount="free", end_effector_type="none",
                     segments=[GeneSegment(name="a", parent=None, is_end_effector=True),
                               GeneSegment(name="floating", parent=None)])
    con = SH.connectivity_report(twin)
    assert not con["ok"] and "floating" in con["disconnected_pieces"]


def test_a_body_that_explodes_is_never_reported_finite():
    """THE TRAP, pinned. ``mj_checkAcc`` reacts to a blown-up ``qacc`` by raising ``mjWARN_BADQACC`` and then
    calling ``mj_resetData`` — so ``isfinite(d.qpos)`` read AFTER the settle is True for exactly the failure it
    exists to detect, and ``penetration_report`` used to answer ``finite=True`` on an exploded model. The test
    measures all three legs so it cannot be simplified back into the hole:

      1. the model really does diverge (MuJoCo's own warning counter says so),
      2. the naive post-settle finiteness check really is vacuous here (it reads True),
      3. ``penetration_report`` reads the warning counters and says ``finite=False`` anyway.

    Leg 2 is what makes this a regression test rather than a tautology: if a future MuJoCo stops auto-resetting,
    leg 2 fails loudly instead of the suite quietly asserting nothing.
    """
    import mujoco
    import numpy as np
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z

    gene = _diverging_gene()
    settle = 60
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    assert m.neq >= 1, "the exploding mechanism is the equality constraint; it must reach the model"
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(settle):
        mujoco.mj_step(m, d)

    # 1. it diverged.
    assert int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0, \
        "premise broken: this body no longer explodes, so the test proves nothing — pick a harsher one"
    # 2. and the state-based check cannot see it (MuJoCo already reset the data out from under it).
    assert bool(np.all(np.isfinite(d.qpos))), \
        "MuJoCo no longer hides the divergence behind mj_resetData; re-derive this test against the new behaviour"

    # 3. the report is not fooled.
    rep = SH.penetration_report(gene, settle_steps=settle)
    assert rep["ok"] is True                                  # it compiled; the failure is dynamic, not structural
    assert rep["finite"] is False, f"an exploded model was reported finite: {rep}"
    assert rep["within_budget"] is False, f"an exploded model passed the budget: {rep}"
    assert rep["mujoco_warnings"].get("BADQACC", 0) > 0, rep
    assert 0 <= rep["first_bad_step"] < settle, rep
    assert "diverged" in rep["finite_reason"]

    # and a combined verdict built on it cannot come out clean.
    assert SH.hygiene_report(gene)["clean"] is False


def test_hygiene_report_combines_the_checks():
    rep = SH.hygiene_report(_clean_quad())
    assert set(rep) == {"clean", "penetration", "connectivity"}
    assert isinstance(rep["clean"], bool)


def test_battery_penetration_does_not_regress():
    """Standing gate: measured baseline is 19/20 cassette bodies within the 2 cm budget, worst ~0.075 m (one
    offline-composed body — 'lynx__construction' — self-clips; a known offline-heuristic defect flagged
    separately). A regression (fewer clean bodies, or a worse peak) blocks."""
    cas = DesignCassette()
    n = within = 0
    worst = 0.0
    for rec in battery():
        g = cas.get_gene(prompt_id(rec))
        if g is None:
            continue
        n += 1
        r = SH.penetration_report(g)
        assert r["ok"], f"{prompt_id(rec)} failed to compile for the hygiene check"
        within += int(r["within_budget"])
        worst = max(worst, r["max_penetration_m"])
    assert n >= 20
    assert within >= 19, f"penetration regression: only {within}/{n} bodies within budget (baseline 19)"
    assert worst <= 0.10, f"peak self-penetration {worst:.3f} m exceeds the 0.10 m ceiling"
