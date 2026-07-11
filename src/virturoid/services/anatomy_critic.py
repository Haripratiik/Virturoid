"""Geometric design critic — the computable VERIFY half of the render->verify->repair loop.

The CAD research the user provided is explicit: don't judge generated bodies by "looks similar"; judge them by
executable validity + measurable proportion/intent checks, then repair. This module loads a compiled gene in
MuJoCo and returns structured, thresholded critiques (the kind of signal an iterative improvement loop or an
LLM-repair step consumes) — proportions, silhouette, limb separation, bilateral symmetry, part-size balance,
ground contact. It is morphology-aware (a humanoid is judged against human canon; a quadruped against a wide
stable stance). Pure measurement, no rendering; pair it with a render for the visual half.
"""

from __future__ import annotations

_TIP_RATIO_MAX = 3.0        # a tip may be up to ~3x its limb's median girth (real feet/paws are 1-2x)
_TIP_RADIUS_FLOOR = 0.05    # tiny tips never flag (a 0.03 pad on a 0.01 leg is fine at any ratio)
_GROUND_BAND_FRAC = 0.35    # "ground extremity" = tip sits in the bottom 35% of the body height (not a head)


def _oversized_ground_extremities(gene, m, d, z_floor: float, height: float):
    """Yield (tip_segment, chain_median_radius, ratio) for every terminal limb-tip near the ground whose radius
    dwarfs its own limb chain. Structural (gene topology) + positional (compiled z) — role-name-free, so it
    generalizes to any morphology the LLM invents. Best-effort: a body-name lookup miss skips that tip."""
    try:
        children: dict[str, int] = {}
        for s in gene.segments:
            if s.parent:
                children[s.parent] = children.get(s.parent, 0) + 1
        for s in gene.segments:
            if children.get(s.name):                       # not a terminal segment
                continue
            # walk the parent chain collecting the actuated limb radii above this tip
            chain = []
            node = s
            while node is not None and node.parent is not None:
                node = next((p for p in gene.segments if p.name == node.parent), None)
                if node is None or node.parent is None:    # stop at the root (the torso is not limb girth)
                    break
                if node.joint_type in ("revolute", "prismatic"):
                    chain.append(float(node.radius_m))
            if len(chain) < 2:                             # not a real limb (a head on a short neck etc.)
                continue
            tip_r = float(s.radius_m)
            if tip_r < _TIP_RADIUS_FLOOR:
                continue
            chain_sorted = sorted(chain)
            med = chain_sorted[len(chain_sorted) // 2]
            if med <= 1e-6 or tip_r / med <= _TIP_RATIO_MAX:
                continue
            try:                                           # positional guard: only GROUND extremities (not heads)
                bid = m.body(s.name).id
                if float(d.xpos[bid][2]) > z_floor + _GROUND_BAND_FRAC * max(height, 1e-6):
                    continue
            except (KeyError, ValueError):
                continue
            yield s, med, tip_r / med
    except Exception:  # noqa: BLE001 - the critic is best-effort; a check must never crash the loop
        return


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
    # group volumes PER PART (body name), and judge the ROOT torso separately from appendages: measured on our
    # own verified bodies, a legitimate quadruped's torso is 44-61% of total volume (lynx 61%, elephant 56%) —
    # a 0.55 blanket cap rejected REAL torsos and exhausted the live LLM repair loop on a phantom defect. The
    # can-tower failure this check exists for is a root at ~75%+ with vestigial stubs, or any single APPENDAGE
    # (a drum foot, a giant head) above ~55%.
    per_body: dict[str, float] = {}
    for gi, v in zip(keep, vols):
        try:
            nm = m.body(int(m.geom_bodyid[gi])).name
        except Exception:  # noqa: BLE001
            nm = str(int(m.geom_bodyid[gi]))
        per_body[nm] = per_body.get(nm, 0.0) + float(v)
    root_seg = next((s for s in gene.segments if s.parent is None), None)
    root_name = root_seg.name if root_seg is not None else ""
    root_frac = per_body.get(root_name, 0.0) / body_vol
    nonroot = {k: v for k, v in per_body.items() if k != root_name}
    big_name, big_v = (max(nonroot.items(), key=lambda kv: kv[1]) if nonroot else ("", 0.0))
    big_frac = big_v / body_vol
    measures["biggest_part_frac"] = round(max(root_frac, big_frac), 3)
    measures["root_volume_frac"] = round(root_frac, 3)
    if root_frac > 0.75:
        add("part_balance", "high", f"the root body is {root_frac:.0%} of the total volume — the limbs are "
            f"vestigial stubs on a monolith; grow the appendages or shrink the torso", root_frac)
    if big_frac > 0.55:
        add("part_balance", "high", f"appendage '{big_name}' is {big_frac:.0%} of the body volume — one part "
            f"dominates the silhouette; shrink it relative to the body", big_frac)

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

    # 7) EXTREMITY PROPORTION (fidelity keystone, measured live): a limb's TIP part (foot/paw/hand) must be
    # proportionate to the limb that carries it. An LLM authored an otherwise-sensible dog whose feet were
    # 0.19 m-radius boxes on 0.02 m-radius legs — visually four drums that swallowed the legs, and the walk
    # CROUCHed (fitness -0.22) — and every prior gate (validity, hygiene, part-balance, stance) passed it.
    # The check is structural + positional: a TERMINAL segment at the BOTTOM of the body (a ground extremity,
    # not a head) whose radius is >3x the median radius of its own limb chain is flagged HIGH, with a teaching
    # detail the LLM repair loop consumes.
    for tip, chain_med, ratio in _oversized_ground_extremities(gene, m, d, lo[2], height):
        add("extremity_proportion", "high",
            f"'{tip.name}' (radius {tip.radius_m:.3f} m) is {ratio:.1f}x the girth of the limb carrying it "
            f"(~{chain_med:.3f} m) — a foot/hand must be comparable to its limb. CONCRETE FIX: author every "
            f"foot part with girth {max(0.015, chain_med):.3f}-{max(0.02, 2.0 * chain_med):.3f} m (about the "
            f"leg's own girth; a real robot foot pad is ~0.02-0.05 m), or OMIT foot parts entirely — a leg "
            f"with segments>=3 receives a proper welded foot pad automatically", round(ratio, 2))

    # score: 1.0 minus weighted penalties
    w = {"fatal": 1.0, "high": 0.25, "med": 0.12, "low": 0.05}
    score = max(0.0, 1.0 - sum(w.get(i["severity"], 0.1) for i in issues))
    return {"ok": not any(i["severity"] in ("fatal", "high") for i in issues),
            "score": round(score, 3), "height_m": round(height, 3),
            "issues": issues, "measures": measures}
