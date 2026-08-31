"""Two looks defects, both found by asking why every archetype renders crude EXCEPT the arm.

1. A KEY-NAME MISMATCH cut the general body path off from the detailed solids. ``cad_geometry.build_anatomy``
   ships multi-feature original solids (a 5-section tapered limb spindle, a heel->toe sole, a jaw->crown skull),
   but ``realize_shape`` only reached them via ``family="role"`` -- which ONLY the hard-coded morphology_composer
   recipes emit. Bodies from ``anatomy_compiler``, i.e. the GENERAL LLM anatomy-graph path that handles dogs,
   horses, geckos and every novel creature, tag segments ``semantic_role`` instead, and nothing mapped the two
   vocabularies. Measured before the fix: dog 0/20 segments reached the rich solids, hexapod 18/26. Exactly
   backwards from the strategy -- the path meant to handle ANY creature got the crudest geometry.

2. THE RENDER DREW A DIFFERENT ROBOT THAN THE ONE SIMULATED. Geometry is authored at compose time, but the link
   keeps changing after: grounding re-sizes it, the structural pass thickens it, the housing-fit pass grows it to
   carry its motor. None of that propagated to the VISUAL spec, so a leg whose real link radius is 27.3 mm was
   drawn at 12.7 mm -- 2.15x thinner than the simulated robot, next to a full-size datasheet motor can. That is
   both the "fat drums on pencil sticks" toy cue and a truthfulness bug.

These tests pin both, plus the invariant that made the fix safe to ship: it is ADDITIVE, so mass and colliders
are untouched.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composition needs MuJoCo")

# Roles that only exist as detailed solids -- reaching them is the whole point of the mapping.
_DETAILED = {"quad_hip", "quad_thigh", "quad_calf", "foot_pad", "sensor_head", "upper_arm", "forearm",
             "hand_plate", "quad_trunk", "torso_shell", "mount_base", "actuator_segment"}


def _compose(prompt: str):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt)


def _detailed_role(geom) -> str:
    """The role this segment will actually be BUILT from, by either vocabulary. The hard-coded recipes emit
    ``family="role"`` directly; the general anatomy path stamps ``anatomy_role``. Both must land on a solid."""
    if not isinstance(geom, dict):
        return ""
    if str(geom.get("family") or "").lower() == "role":
        r = str(geom.get("role") or "")
        return r if r in _DETAILED else ""
    r = str(geom.get("anatomy_role") or "")
    return r if r in _DETAILED else ""


@pytest.mark.parametrize(("prompt", "min_roles"), [("a robot dog", 12), ("a hexapod robot", 12),
                                                   ("a robot horse", 12)])
def test_the_general_body_path_reaches_the_detailed_solids(prompt, min_roles):
    """Regression for the vocabulary split: legs/feet/heads must reach a detailed solid, not fall through to a
    4-point rectangle extruded into a bar."""
    gene = _compose(prompt)
    roles = [_detailed_role(getattr(s, "geometry", None)) for s in gene.segments]
    detailed = [r for r in roles if r]
    assert len(detailed) >= min_roles, (
        f"{prompt!r}: only {len(detailed)}/{len(gene.segments)} segments reach a detailed solid "
        f"(roles seen: {sorted(set(r for r in roles if r))}). The general path is back on bare extrusions.")
    # Only bodies that actually declare feet get checked -- a hexapod's coxa/femur/tibia leg has no separate
    # foot segment, and inventing one here would be testing the composer's topology, not this fix.
    semantic = [str((getattr(s, "geometry", None) or {}).get("semantic_role") or "").lower()
                for s in gene.segments]
    if any(r in ("foot", "paw", "hoof") for r in semantic):
        assert "foot_pad" in roles, (
            f"{prompt!r}: {sum(r in ('foot', 'paw', 'hoof') for r in semantic)} foot segments are still generic "
            "bars, not contact pads")


def test_the_general_anatomy_path_specifically_gets_the_detailed_solids():
    """The narrower half of the regression. A dog goes through anatomy_compiler -- the GENERAL path -- and used
    to reach 0/20 detailed solids while the hard-coded hexapod recipe reached 18/26. Pin the general one."""
    gene = _compose("a robot dog")
    stamped = [str((getattr(s, "geometry", None) or {}).get("anatomy_role") or "") for s in gene.segments]
    assert len([r for r in stamped if r in _DETAILED]) >= 12, (
        f"the general anatomy path stamped only {sorted(set(r for r in stamped if r))} -- it is back on "
        "bare extrusions while the hard-coded recipes keep the good geometry.")


def test_anatomy_roles_are_additive_and_do_not_move_a_gram():
    """The mapping must not change physics. ``family`` is untouched, so the collider proxy is identical and mass
    is byte-identical -- that is what made this safe to ship without re-verifying every walk."""
    from virturoid.services.anatomy_compiler import _anatomy_role_for

    gene = _compose("a robot dog")
    for s in gene.segments:
        geo = getattr(s, "geometry", None)
        if not isinstance(geo, dict) or not geo.get("anatomy_role"):
            continue
        assert geo.get("family") != "role", (
            f"{s.name}: anatomy_role overwrote family -- that WOULD change the collider and the mass.")

    # A leg's segments are distinguished BY POSITION, the way a real leg is built.
    assert _anatomy_role_for("leg", 0, 4) == "quad_hip"
    assert _anatomy_role_for("leg", 1, 4) == "quad_thigh"
    assert _anatomy_role_for("leg", 3, 4) == "quad_calf"
    assert _anatomy_role_for("leg", 0, 1) == "quad_thigh"        # a one-segment leg is a limb, not a hip block
    assert _anatomy_role_for("foot", 0, 1) == "foot_pad"
    assert _anatomy_role_for("torso", 0, 1) is None              # no better solid -> leave the authored spec


@pytest.mark.parametrize("prompt", ["a robot dog", "a hexapod robot"])
def test_the_render_draws_the_link_that_is_actually_simulated(prompt):
    """The visual spec is reconciled to the link's real radius at bake time, so the picture cannot claim a
    pencil-thin leg while the physics runs a thick one."""
    from virturoid.services.cad_geometry import _visual_matches_link

    gene = _compose(prompt)
    worst, worst_name = 1.0, ""
    for s in gene.segments:
        geo = getattr(s, "geometry", None)
        r = float(getattr(s, "radius_m", 0.0) or 0.0)
        if not isinstance(geo, dict) or r <= 0.0:
            continue
        fixed = _visual_matches_link(geo, s)
        if fixed.get("anatomy_role") or str(fixed.get("family") or "").lower() == "role":
            drawn = float(fixed.get("radius") or 0.0)
        else:
            prof = fixed.get("profile")
            if not isinstance(prof, (list, tuple)) or not prof:
                continue
            drawn = max(max(abs(float(p[0])), abs(float(p[1]))) for p in prof if len(p) >= 2)
        if drawn <= 0:
            continue
        if drawn / r < worst:
            worst, worst_name = drawn / r, s.name
    assert worst >= 0.95, (
        f"{prompt!r}: {worst_name} is DRAWN at {worst:.2f}x its simulated radius. The render is showing a "
        "different robot than the one being verified.")


def test_reconciliation_only_grows_and_never_erases_an_authored_flourish():
    from virturoid.services.cad_geometry import _visual_matches_link

    class _Seg:
        name, radius_m, length_m = "l", 0.030, 0.20

    # A visual already RICHER than the link is left alone (we never shrink what the model asked for).
    rich = {"family": "extrude", "profile": [[0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]],
            "fillet": 0.004}
    assert _visual_matches_link(rich, _Seg())["profile"] == rich["profile"]

    # A stale THIN visual is grown to the link, with its edge treatment kept proportionate.
    thin = {"family": "extrude", "profile": [[0.01, 0.01], [-0.01, 0.01], [-0.01, -0.01], [0.01, -0.01]],
            "fillet": 0.002, "detail": "keep-me"}
    out = _visual_matches_link(thin, _Seg())
    assert max(abs(float(p[0])) for p in out["profile"]) == pytest.approx(0.030)
    assert out["fillet"] == pytest.approx(0.006)                 # 3x profile growth -> 3x fillet
    assert out["detail"] == "keep-me" and out["family"] == "extrude"
    assert thin["fillet"] == 0.002, "reconciliation mutated the caller's spec in place"
