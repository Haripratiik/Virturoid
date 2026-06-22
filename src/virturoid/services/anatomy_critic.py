"""Geometric design critic — the computable VERIFY half of the render->verify->repair loop.

The CAD research the user provided is explicit: don't judge generated bodies by "looks similar"; judge them by
executable validity + measurable proportion/intent checks, then repair. This module loads a compiled gene in
MuJoCo and returns structured, thresholded critiques (the kind of signal an iterative improvement loop or an
LLM-repair step consumes) — proportions, silhouette, limb separation, bilateral symmetry, part-size balance,
ground contact. It is morphology-aware (a humanoid is judged against human canon; a quadruped against a wide
stable stance). Pure measurement, no rendering; pair it with a render for the visual half.
"""

from __future__ import annotations


def critique_gene(gene, *, expected: str | None = None) -> dict:
    """Return ``{ok, score, height_m, issues:[{check, severity, detail, value}], measures:{...}}`` for a gene.

    ``expected`` overrides the body kind (humanoid/quadruped/legged/arm); otherwise inferred from robot_class.
    Best-effort: if MuJoCo can't load the body it returns a single ``compile`` issue (the worst failure)."""
    try:
        import numpy as np
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "score": 0.0, "issues": [{"check": "import", "severity": "fatal",
                "detail": f"deps unavailable: {exc}", "value": None}], "measures": {}}

    kind = (expected or getattr(gene, "robot_class", "") or "").lower()
    try:
        xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene, meshed=False))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    except Exception as exc:  # noqa: BLE001 - invalid geometry is the worst critique
        return {"ok": False, "score": 0.0, "issues": [{"check": "compile", "severity": "fatal",
                "detail": f"gene does not compile/stand: {exc}", "value": None}], "measures": {}}

    # world geom positions (skip the floor plane at idx 0 / any plane)
    import numpy as np
    keep = [g for g in range(m.ngeom) if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
    P = d.geom_xpos[keep]
    rb = m.geom_rbound[keep]
    lo, hi = (P - rb[:, None]).min(0), (P + rb[:, None]).max(0)
    ext = hi - lo
    height = float(ext[2])
    width = float(max(ext[0], ext[1]))
    issues: list[dict] = []

    def add(check, sev, detail, value=None):
        issues.append({"check": check, "severity": sev, "detail": detail, "value": value})

    measures = {"height_m": round(height, 3), "width_m": round(width, 3),
                "aspect_w_h": round(width / height, 3) if height > 1e-6 else 0.0,
                "n_geoms": len(keep)}

    # 1) part-size balance: no single part should dominate the body (the "can-tower" failure). Use
    # SHAPE-AWARE volume from geom_size, NOT the bounding-sphere rb**3 proxy — that artifact over-counted
    # flat slab torsos/chassis (a low wide box has a huge bounding sphere) and falsely flagged credible
    # bodies, which then exhausted the LLM verify->repair loop chasing a phantom defect.
    GT = mujoco.mjtGeom
    vols = []
    for gi in keep:
        t = int(m.geom_type[gi]); s = m.geom_size[gi]
        if t == int(GT.mjGEOM_BOX):
            v = 8.0 * float(s[0]) * float(s[1]) * float(s[2])
        elif t == int(GT.mjGEOM_SPHERE):
            v = (4.0 / 3.0) * np.pi * float(s[0]) ** 3
        elif t == int(GT.mjGEOM_CYLINDER):
            v = np.pi * float(s[0]) ** 2 * (2.0 * float(s[1]))
        elif t == int(GT.mjGEOM_CAPSULE):
            r0 = float(s[0]); v = np.pi * r0 ** 2 * (2.0 * float(s[1])) + (4.0 / 3.0) * np.pi * r0 ** 3
        elif t == int(GT.mjGEOM_ELLIPSOID):
            v = (4.0 / 3.0) * np.pi * float(s[0]) * float(s[1]) * float(s[2])
        else:                                            # mesh / other: bounding-sphere proxy
            v = (4.0 / 3.0) * np.pi * float(m.geom_rbound[gi]) ** 3
        vols.append(max(v, 1e-9))
    vols = np.array(vols)
    body_vol = float(vols.sum()) or 1.0
    biggest = float(vols.max()) / body_vol
    measures["biggest_part_frac"] = round(biggest, 3)
    if biggest > 0.55:
        add("part_balance", "high", f"one part is {biggest:.0%} of the body volume — it dominates the silhouette", biggest)

    # 2) bilateral symmetry in y (limbed bodies should be left/right symmetric)
    if kind in ("humanoid", "biped", "quadruped", "legged"):
        ys = P[:, 1]
        asym = abs(float(ys.mean()))
        measures["y_centroid_offset_m"] = round(asym, 3)
        if asym > 0.04:
            add("symmetry", "med", f"body is not left/right symmetric (y-centroid off by {asym:.3f} m)", asym)

    if kind in ("humanoid", "biped"):
        # 3) silhouette: a standing humanoid is tall but not a needle; W:H ~0.28-0.5
        ar = measures["aspect_w_h"]
        if ar < 0.22:
            add("silhouette", "high", f"too narrow/stick-like (W:H={ar:.2f}; want ~0.3-0.45)", ar)
        elif ar > 0.6:
            add("silhouette", "med", f"too wide for a humanoid (W:H={ar:.2f})", ar)
        # 4) lateral limb separation: arms+legs must spread in y, not stack on the centerline
        spread = float(np.percentile(np.abs(P[:, 1]), 90))
        measures["limb_y_spread_m"] = round(spread, 3)
        if spread < 0.09:
            add("limb_separation", "high", f"limbs hug the centerline (90th-pct |y|={spread:.3f} m) — reads as a column", spread)
        # 5) plausible human height
        if not (0.8 <= height <= 2.2):
            add("scale", "med", f"height {height:.2f} m is outside a human range (0.8-2.2)", height)
        # 6) head on top: highest geom should sit above the vertical mid-line by a clear margin
        if (float(P[:, 2].max()) - lo[2]) < 0.55 * height:
            add("head_top", "med", "no clear head/upper mass at the top third", None)

    if kind in ("quadruped", "legged"):
        ar = measures["aspect_w_h"]
        if ar < 0.8:
            add("stance", "high", f"too tall/narrow for a legged body (W:H={ar:.2f}; want wider-than-tall)", ar)
        # feet near the floor
        if lo[2] > 0.06:
            add("ground_contact", "med", f"lowest geom is {lo[2]:.3f} m off the floor — not standing on feet", lo[2])

    # score: 1.0 minus weighted penalties
    w = {"fatal": 1.0, "high": 0.25, "med": 0.12, "low": 0.05}
    score = max(0.0, 1.0 - sum(w.get(i["severity"], 0.1) for i in issues))
    return {"ok": not any(i["severity"] in ("fatal", "high") for i in issues),
            "score": round(score, 3), "height_m": round(height, 3),
            "issues": issues, "measures": measures}
