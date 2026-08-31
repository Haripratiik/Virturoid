import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")


def test_current_dog_has_aligned_contact_visuals_and_grounded_spawn():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.visual_physics_gate import audit_gene
    report = audit_gene(compose_robot("a robot dog", llm=None, ensure_walkable=True))
    assert report.ok, report.to_dict()
    assert report.support_gap_m is not None and report.support_gap_m <= 0.01


def test_legacy_oversized_foot_visual_fails_the_gate():
    import re
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.visual_physics_gate import audit_mjcf
    gene = compose_robot("a robot dog", llm=None, ensure_walkable=True)
    xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene, clearance=0.002, meshed=False))

    def oversize(match):
        return match.group(0).replace(f'size="{match.group(1)}"', f'size="{float(match.group(1)) * 1.35:.5f}"')

    legacy = re.sub(r'<geom name="[^"]+_boot"[^>]*size="([0-9.]+)"[^>]*/>', oversize, xml)
    report = audit_mjcf(legacy)
    assert not report.ok
    assert any(issue.code == "contact_visual_outside_collider" for issue in report.issues)


def test_airborne_spawn_fails_the_gate():
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.visual_physics_gate import audit_mjcf
    gene = compose_robot("a robot dog", llm=None, ensure_walkable=True)
    spawn = standing_spawn_z(gene, clearance=0.002, meshed=False)
    report = audit_mjcf(compile_gene_to_mjcf(gene, spawn_z=spawn + 0.04))
    assert any(issue.code == "airborne_spawn" for issue in report.issues)
