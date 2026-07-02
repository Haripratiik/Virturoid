"""Sphere-feet + feet-only collision transform (breakthrough plan v2 T1.4 / gap N12).

The MJX<->CPU deploy gap for vigorous learned gaits is worst at the FOOT contact: a capsule foot gives a
2-point manifold in MJX (ncon=2) but is resolved by MPR in C MuJoCo -> the manifolds differ, and the policy
overfits the MJX one. MuJoCo-Playground's fix (stream 1) is SPHERE feet with FEET-ONLY collision: a sphere-plane
contact is an IDENTICAL 1-point manifold in both engines, so the mismatch is removed AT THE SOURCE, and feet-only
masking drops the spurious body/limb contacts that also differ.

This is a post-compile ``mujoco.MjModel`` transform (applied identically on the training compile AND the deploy
compile, so train==deploy): (1) find the foot geoms (lowest body geoms at the default pose), (2) turn each into a
SPHERE keeping its radius, (3) set FEET-ONLY collision (only the foot spheres + the floor collide). Pure model
arrays -> works before ``mjx.put_model`` on the box and after ``compiled_model`` on CPU. Idempotent; returns the
foot geom indices. The tiny foot-height change from re-centering is absorbed by recomputing the standing spawn.
"""

from __future__ import annotations


def _world_bottom(model, data, gi) -> float:
    """The lowest WORLD-z point of geom ``gi`` at the current pose — an ACCURATE per-type extent, not the bounding
    sphere. ``geom_rbound`` (the bounding-sphere radius) badly overestimates a flat box's downward reach (it's the
    half-DIAGONAL), which mis-ranks a high wide torso as low as the feet. We instead project the geom's true local
    half-extents onto world-z: ``z_center - sum_k |R[2,k]| * halfext_k``."""
    import mujoco
    import numpy as np
    G = mujoco.mjtGeom
    t = int(model.geom_type[gi])
    s = np.asarray(model.geom_size[gi], dtype=float)
    zc = float(data.geom_xpos[gi, 2])
    az = np.abs(np.asarray(data.geom_xmat[gi], dtype=float).reshape(3, 3)[2, :])  # |world-z . local axes|
    if t == int(G.mjGEOM_SPHERE):
        h = np.array([s[0], s[0], s[0]])
    elif t in (int(G.mjGEOM_CAPSULE), int(G.mjGEOM_CYLINDER)):
        cap = s[0] if t == int(G.mjGEOM_CAPSULE) else 0.0      # capsule adds a radius cap on the z ends
        h = np.array([s[0], s[0], s[1] + cap])                 # radius in x,y; halflen(+cap) in z
    elif t in (int(G.mjGEOM_BOX), int(G.mjGEOM_ELLIPSOID)):
        h = np.array([s[0], s[1], s[2]])
    else:                                                      # plane/hfield/mesh: fall back to the bounding radius
        h = np.array([float(model.geom_rbound[gi])] * 3)
    return zc - float(az @ h)


def find_foot_geoms(model, *, band: float = 0.15) -> list:
    """The foot geoms = the ground contact of each leg. Identified PER-LEG rather than by a magic height band,
    which is what makes this robust across morphologies:
      * skip VISUAL-only geoms (contype==conaffinity==0) — the fidelity layer adds non-colliding detail geoms
        (cylinders, shells) at foot height that never touch the floor; converting those would be wrong.
      * a foot is the lowest colliding geom on a LEAF body (a body with no child bodies). The calf/shin (whose
        body has the foot as a CHILD) is structural, not a contact, so it is excluded automatically — no need to
        tune a band tight enough to separate foot-pad from calf (they can be <2cm apart).
      * rank each geom by its accurate LOWEST extent (``_world_bottom``, true per-type half-extents projected on
        world-z), NOT the bounding sphere — a high wide torso box has a huge bounding radius and would otherwise
        rank as low as the feet.
      * finally keep only leaf-body candidates within ``band`` (m) of the LOWEST one, so a leaf that is up high
        (a head, a raised arm) is not mistaken for a foot.
    Falls back to the plain lowest-colliding-within-band set if the body has no leaf-body contacts (degenerate)."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    has_child = [False] * model.nbody
    for b in range(1, model.nbody):
        p = int(model.body_parentid[b])
        if p >= 0:
            has_child[p] = True
    colliding = [gi for gi in range(model.ngeom)
                 if int(model.geom_bodyid[gi]) != 0                         # exclude worldbody/floor
                 and (int(model.geom_contype[gi]) or int(model.geom_conaffinity[gi]))]  # exclude visual-only geoms
    if not colliding:
        return []
    bottom = {gi: _world_bottom(model, data, gi) for gi in colliding}
    by_leaf: dict = {}
    for gi in colliding:
        b = int(model.geom_bodyid[gi])
        if has_child[b]:                                       # structural link (calf/thigh/torso), not a contact
            continue
        cur = by_leaf.get(b)
        if cur is None or bottom[gi] < bottom[cur]:            # lowest colliding geom on this leaf body = its foot
            by_leaf[b] = gi
    feet = list(by_leaf.values())
    if not feet:                                              # degenerate (no leaf contacts): plain lowest-in-band
        lo = min(bottom.values())
        return [gi for gi in colliding if bottom[gi] < lo + band]
    lo = min(bottom[gi] for gi in feet)                       # drop a leaf that sits up high (head / raised arm)
    return [gi for gi in feet if bottom[gi] < lo + band]


def apply_sphere_feet(model, *, radius: float | None = None, band: float = 0.06, feet_only: bool = True) -> list:
    """In-place: convert the foot geoms to SPHERES (manifold-invariant contact) and, if ``feet_only``, make ONLY
    the foot spheres + the floor collide. Returns the foot geom indices. Applied on BOTH train + deploy compiles
    so the contact model matches (plan v2 T1.4). ``radius`` overrides the sphere radius (default: keep each foot
    geom's own radius = ``geom_size[0]``, which for a capsule/sphere is the cross-section radius)."""
    import mujoco
    import numpy as np
    feet = find_foot_geoms(model, band=band)
    feet_set = set(feet)
    for gi in feet:
        r = float(radius) if radius is not None else float(model.geom_size[gi][0]) or 0.02
        model.geom_type[gi] = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        model.geom_size[gi] = np.array([r, 0.0, 0.0], dtype=model.geom_size.dtype)
    if feet_only:
        for gi in range(model.ngeom):
            if int(model.geom_bodyid[gi]) == 0:               # floor / worldbody geoms: leave them colliding
                continue
            on = 1 if gi in feet_set else 0                   # feet collide; every other body geom does not
            model.geom_contype[gi] = on
            model.geom_conaffinity[gi] = on
    return feet
