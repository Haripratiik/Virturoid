"""WS-I — clipping/penetration budgets as a standing CI measurement (master_plan_v6 WS-I)."""
from __future__ import annotations

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import structural_hygiene as SH
from virturoid.services.design_battery import battery, prompt_id
from virturoid.services.design_cassette import DesignCassette


def _clean_quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a sturdy four-legged walking robot", ensure_walkable=True)


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
