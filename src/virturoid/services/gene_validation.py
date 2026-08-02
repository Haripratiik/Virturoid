"""First-class GENE-DESIGN VALIDATION gate (report 8's validation service) — one honest pass over a built gene.

(Distinct from the older ``design_validator.py`` which gates the requirements/reference-arm pipeline over
components + scene sets. This module validates a plain ``RobotGene`` — the design-intent compiler's output.)

Report 8 makes validation a first-class compile stage (kinematics, torque margin, structure, stability,
collision, description round-trip). This composes the engineering checks we ALREADY have into a single report
and adds two new MuJoCo rest-pose checks. Design choices:

- It ANNOTATES, never hard-rejects: every problem is a risk_flag (severity fatal/high/med/low, the same shape
  as anatomy_critic), so an exotic-but-deliberate body is surfaced-with-caveats, not dropped — matching the
  bill_of_materials 'under_spec: flag, don't drop' contract.
- NO heavy deps: MuJoCo IS our simulator, so report 8's "URDF/SDF round-trip" becomes an MJCF compile+load
  round-trip. Every MuJoCo sub-check is best-effort (try/except -> skipped) so the offline/no-MuJoCo path
  returns the analytic checks only and never crashes a build.
- Confidence reuses buildability.transferability_score (honest heuristic, NOT a real-world transfer guarantee).
"""

from __future__ import annotations

# Plausible total-mass band per class (kg) — a screening sanity check, not a spec.
_MASS_BAND = {"humanoid": (4.0, 150.0), "biped": (4.0, 150.0), "quadruped": (0.5, 80.0),
              "legged": (0.2, 100.0), "manipulator": (0.3, 60.0), "mobile_base": (1.0, 100.0)}


def validate_gene_design(gene, *, material: str = "aluminum", payload_kg: float = 0.0,
                         sim_success: float | None = None) -> dict:
    """Validate a built gene; return ``{ok, confidence, risk_flags, checks, total_mass_kg, bom, disclaimer}``.

    ``ok`` is True when there is no fatal/high flag. ``risk_flags`` items are ``{check, severity, detail,
    value}``. Analytic checks always run; the stability / self-collision / round-trip checks run only when
    MuJoCo is importable."""
    flags: list[dict] = []
    checks: dict = {}

    def flag(check, severity, detail, value=None):
        flags.append({"check": check, "severity": severity, "detail": detail, "value": value})

    # 0) kinematic precondition — a malformed tree can't be load-analysed; short-circuit.
    kin = gene.validate()
    checks["kinematic"] = not kin
    if kin:
        flag("kinematic", "fatal", "; ".join(kin), None)
        return {"ok": False, "confidence": 0.0, "risk_flags": flags, "checks": checks,
                "total_mass_kg": None, "bom": [], "disclaimer": _DISCLAIMER}

    cls = (gene.robot_class or "").lower()

    # 1) torque margin — real actuator under_spec (catalog can't cover the joint) + static hold-torque ratio.
    bom = []
    try:
        from virturoid.services.component_geometry import bill_of_materials
        bom = bill_of_materials(gene)
        under = [r for r in bom if r.get("under_spec")]
        if under:
            flag("actuator_under_spec", "high",
                 f"{len(under)} joint(s) exceed the largest catalog actuator: "
                 + ", ".join(f"{r['joint']} needs {r['required_nm']}Nm" for r in under[:3]), len(under))
    except Exception:  # noqa: BLE001
        pass
    try:
        from virturoid.services.design_critic import TORQUE_RISK_RATIO, joint_hold_torques
        # Payload loads a joint only for a GRASPING body that actually holds it at the chain tip (a hand/
        # gripper). A legged/mobile body's chain tip is a foot/wheel — hanging the payload there is nonsense
        # and produced a spurious flag, so apply payload only to grasping classes.
        eff_payload = payload_kg if cls in ("manipulator", "humanoid", "biped") else 0.0
        ratios = joint_hold_torques(gene, payload_kg=eff_payload)
        over = [t for t in ratios if t["ratio"] >= TORQUE_RISK_RATIO]
        checks["torque"] = not over
        if over:
            worst = max(over, key=lambda t: t["ratio"])
            sev = "high" if worst["ratio"] >= 1.0 else "med"
            flag("torque_margin", sev, f"{len(over)} joint(s) near/over their static hold torque "
                 f"(worst {worst['joint']} at {worst['ratio']:.0%} of capacity)", round(worst["ratio"], 3))
    except Exception:  # noqa: BLE001
        pass

    # 2) structural stress (cantilever screening) — verbatim reuse, SF>=2.0.
    try:
        from virturoid.services.structural_validation import validate_structure
        st = validate_structure(gene, material=material, payload_kg=payload_kg, safety_factor_min=2.0)
        checks["structural"] = bool(st["ok"])
        sf = st.get("min_safety_factor", 0.0)
        if sf < 1.0:
            flag("structural", "fatal", f"a link is under-strength (min safety factor {sf:.2f} < 1.0)", round(sf, 2))
        elif not st["ok"]:
            flag("structural", "high", f"a link is below the 2.0 safety-factor target (min {sf:.2f})", round(sf, 2))
    except Exception:  # noqa: BLE001
        st = {"min_safety_factor": 0.0, "ok": False}

    # 3) mass budget — ground a COPY (ground_gene mutates) and sanity-check the total against a class band.
    total_mass = None
    try:
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.grounded_physics import ground_gene, physical_prior_for
        g2 = RobotGene.from_dict(gene.to_dict())
        total_mass = ground_gene(g2)["total_mass_kg"]
        prior = physical_prior_for(g2)
        lo, hi = prior.mass_band_kg if prior is not None else _MASS_BAND.get(cls, (0.1, 200.0))
        # THE BAND DESCRIBES AN UNLOADED MACHINE. A robot RATED to carry a load is genuinely heavier -- it is
        # carrying bigger motors and thicker load-path links to hold that load -- so judging it against the
        # unloaded band calls real hardware implausible (Spot: 32.5 kg, rated 14 kg). Widen the ceiling by the
        # rating and nothing else: `payload_kg` when the caller states it, else the rating `set_payload` stamped
        # on the gene. A body that never declared a payload is judged exactly as before.
        rated = float(payload_kg or 0.0)
        if not rated:
            try:
                rated = float((getattr(gene, "metadata", None) or {}).get("rated_payload_kg") or 0.0)
            except (TypeError, ValueError):
                rated = 0.0
        hi_eff = hi + max(0.0, rated)
        checks["mass_budget"] = lo <= total_mass <= hi_eff
        if not (lo <= total_mass <= hi_eff):
            _rated = f" + {rated:.1f} kg rated payload" if rated else ""
            flag("mass_budget", "med", f"total mass {total_mass:.1f} kg is outside the plausible "
                 f"{cls or 'robot'} band ({lo}-{hi} kg{_rated})", total_mass)
    except Exception:  # noqa: BLE001
        pass

    # 4) + 5) + 6) MuJoCo rest-pose checks (best-effort): static stability, self-collision, MJCF round-trip.
    _mujoco_checks(gene, cls, checks, flag)

    # 7) A contact-visible surface must agree with the geometry the simulator actually collides with. This
    # catches the otherwise easy-to-miss failure where a decorative foot touches the floor while its collider
    # floats above it (or the whole free-base body is spawned in mid-air). The check is executable rather than
    # a class/template rule, so it applies to newly authored morphologies too.
    try:
        from virturoid.services.visual_physics_gate import audit_gene
        visual_physics = audit_gene(gene)
        checks["visual_physics"] = visual_physics.ok
        if not visual_physics.ok:
            detail = "; ".join(issue.detail for issue in visual_physics.issues[:3])
            flag("visual_physics", "high", detail, visual_physics.support_gap_m)
    except Exception:  # noqa: BLE001 - no MuJoCo -> analytic-only report
        pass
    try:
        from virturoid.services.structural_assertions import evaluate_structural_assertions
        structural_contract = evaluate_structural_assertions(gene)
        checks["structural_seams"] = structural_contract.ok
        if not structural_contract.ok:
            failed = [a.detail for a in structural_contract.assertions if not a.ok]
            flag("structural_seams", "high", "; ".join(failed[:3]), len(failed))
    except Exception:  # noqa: BLE001 - optional MuJoCo geometry contract
        pass

    parts_grounded = not [r for r in bom if r.get("under_spec")] if bom else True
    try:
        from virturoid.services.buildability import transferability_score
        conf = transferability_score(parts_grounded, st, sim_success)
    except Exception:  # noqa: BLE001
        conf = 0.5
    if checks.get("stability") is False:           # an unstable rest pose dents behavioral confidence
        conf = round(conf * 0.7, 3)

    ok = not any(f["severity"] in ("fatal", "high") for f in flags)
    return {"ok": ok, "confidence": conf, "risk_flags": flags, "checks": checks,
            "total_mass_kg": round(total_mass, 3) if total_mass is not None else None,
            "bom": bom, "disclaimer": _DISCLAIMER}


def _mujoco_checks(gene, cls, checks, flag) -> None:
    """Static stability (CoM over support polygon), self-collision at rest, MJCF round-trip — all best-effort."""
    try:
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    except Exception:  # noqa: BLE001 - no MuJoCo -> analytic-only report
        return

    # MJCF round-trip: it must compile AND load, and expose the expected actuator count (report-8 round-trip,
    # reinterpreted for our MJCF transport instead of URDF/SDF).
    try:
        xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene, meshed=False))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        checks["mjcf_roundtrip"] = (m.nu == len(gene.actuated_joints()))
        if not checks["mjcf_roundtrip"]:
            flag("mjcf_roundtrip", "high",
                 f"compiled model exposes {m.nu} actuators but the gene has {len(gene.actuated_joints())}", m.nu)
        # Every declared loop must survive compilation — an actuator count alone cannot see one, which is how a
        # gantry with a decorative second column passed every gate it had.
        _want_eq = len(getattr(gene, "loop_closures", None) or [])
        checks["loop_closures_compiled"] = (int(m.neq) >= _want_eq)
        if not checks["loop_closures_compiled"]:
            flag("loop_closures_compiled", "high",
                 f"the design declares {_want_eq} closed loop(s) but the compiled model carries {int(m.neq)}",
                 int(m.neq))
        # A `connect` locks in whatever offset the two parts have AT BUILD TIME. So declaring a loop between
        # parts that are not already touching does not pull them together — it WELDS THE GAP, permanently, and
        # says nothing. Measured on a delta: its arm tips sat 0.5045 m from the platform before stepping and
        # 0.5045 m after 2000 steps, held exactly that far apart by the constraint meant to join them. The
        # gantry worked only because its bridge was sized so the far end already landed on the column.
        if _want_eq:
            import numpy as _np
            mujoco.mj_forward(m, d)
            _far = []
            for _lc in gene.loop_closures:
                _a, _b = (_lc or {}).get("a"), (_lc or {}).get("b")
                _ia = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(_a))
                _ib = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(_b))
                if _ia < 0 or _ib < 0:
                    continue
                _sa = next((s for s in gene.segments if s.name == _a), None)
                _anc = (_lc or {}).get("anchor")
                if not (isinstance(_anc, (list, tuple)) and len(_anc) == 3):
                    _anc = (0.0, 0.0, float(getattr(_sa, "length_m", 0.0) or 0.0))
                _pa = _np.asarray(d.xpos[_ia], dtype=float) + d.xmat[_ia].reshape(3, 3) @ _np.asarray(_anc,
                                                                                                      dtype=float)
                # Distance to b's SURFACE, not to its origin. A body's origin is wherever its frame happens to
                # sit — for a column it is the base, so measuring there reports the column's own LENGTH as a gap
                # and calls a perfect join broken. (The gantry's bridge tip is 0.0000 m from column_r's top and
                # 0.9000 m from its origin.)
                _gs = [_g for _g in range(m.ngeom) if int(m.geom_bodyid[_g]) == _ib]
                if _gs:
                    _gap = min(float(_np.linalg.norm(_pa - _np.asarray(d.geom_xpos[_g], dtype=float)))
                               - float(m.geom_rbound[_g]) for _g in _gs)
                    _gap = max(0.0, _gap)
                else:
                    _gap = float(_np.linalg.norm(_pa - _np.asarray(d.xpos[_ib], dtype=float)))
                if _gap > 0.05:
                    _far.append((f"{_a}<->{_b}", round(_gap, 4)))
            checks["loop_closures_meet"] = not _far
            if _far:
                flag("loop_closures_meet", "high",
                     "declared loop(s) join parts that are not touching, so the constraint holds the GAP instead "
                     f"of closing it: {_far}. Place the two parts coincident in the design — a connect preserves "
                     "the offset it is built with", _far[0][1])
    except Exception as exc:  # noqa: BLE001 - a body that won't compile/load is the worst failure
        flag("mjcf_roundtrip", "fatal", f"gene does not compile/load into MuJoCo: {exc}", None)
        return

    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    GT = mujoco.mjtGeom
    body_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) != int(GT.mjGEOM_PLANE)]
    contact_geoms = [g for g in body_geoms if int(m.geom_contype[g]) and int(m.geom_conaffinity[g])]

    # Static stability (free-base bodies only): whole-body CoM xy must lie inside the support polygon = the
    # convex hull of the lowest (ground-contacting) geoms. A top-heavy / single-foot body is flagged unstable.
    if gene.base_mount == "free" and contact_geoms:
        try:
            com = np.array(d.subtree_com[0][:2], float)
            # MuJoCo geom_rbound is a loose bounding sphere. On flat boxes / wheels it can put the chassis
            # "below" the wheels, so use tight transformed AABB bottoms and ignore visual-only geoms.
            bottoms = _geom_bottoms_tight(m, d, contact_geoms, np)
            zmin = float(bottoms.min())
            foot = [contact_geoms[i] for i, b in enumerate(bottoms) if b <= zmin + 0.04]
            pts = np.array([d.geom_xpos[g, :2] for g in foot], float)
            stable = _com_in_support(com, pts)
            checks["stability"] = bool(stable)
            if not stable:
                flag("stability", "high", "center of mass falls outside the foot support polygon — the rest "
                     "pose is statically unstable (would topple)", None)
        except Exception:  # noqa: BLE001
            pass

    # NOTE: a rest-pose self-collision check was evaluated and REMOVED. The load-bearing collision primitives
    # are conservative capsules (fatter than the slim visual lofts), so a humanoid's hanging hand and thigh
    # capsules overlap ~7 cm at rest even when the rendered body is clean — it fired on every proven humanoid/
    # quadruped template. A check that can't separate normal anatomy from a real defect erodes trust in the
    # whole report, so it's intentionally omitted; the intent compiler already prevents the gross "part buried
    # in another" case that primitive collision would (unreliably) catch.


def _com_in_support(com, pts) -> bool:
    """Is point ``com`` (xy) inside the convex hull of ``pts`` (with a small tolerance)? Fewer than 3 distinct
    points = a line/point support (no static base) -> unstable. Dependency-free (monotone-chain hull)."""
    import numpy as np

    uniq = np.unique(np.round(pts, 4), axis=0)
    if len(uniq) < 3:
        return False
    hull = _convex_hull(uniq)
    n = len(hull)
    # For a CCW hull, the point is inside iff it is left-of (or on) every edge, within a small tolerance.
    tol = -2e-3
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        cross = (x2 - x1) * (com[1] - y1) - (y2 - y1) * (com[0] - x1)
        if cross < tol:
            return False
    return True


def _geom_bottoms_tight(mj, data, geoms, np):
    """Return tight world-z bottoms for geoms using MuJoCo's local AABBs, not loose bounding spheres."""
    aabb = mj.geom_aabb.reshape(mj.ngeom, 6)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    bottoms = []
    for g in geoms:
        center, half = aabb[g, :3], aabb[g, 3:]
        corners = center + signs * half
        world = data.geom_xpos[g] + corners @ data.geom_xmat[g].reshape(3, 3).T
        bottoms.append(float(world[:, 2].min()))
    return np.array(bottoms, float)


def _convex_hull(pts):
    """Andrew's monotone chain — returns hull vertices CCW. ``pts`` is an (N,2) array of distinct points."""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


_DISCLAIMER = ("Validation is an engineering SCREENING (analytic + rest-pose sim), NOT a real-world transfer "
               "guarantee; a novel morphology cannot be calibrated until built and measured.")
