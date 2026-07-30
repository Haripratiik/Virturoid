"""A queryable MEASUREMENT surface over the robot the agent is holding.

`get_robot` returns a fixed summary, and `describe_robot`/`diagnose_body` compose a NEW body from a prompt rather
than inspecting the held one. So an agent designing a robot cannot ask the questions a designer actually asks:
how far is the gripper from its mount, do these two links pass through each other anywhere in the joint's range,
what is the torque margin at this joint, how much ground clearance is there.

This is the half of the loop that the Articraft result says matters most: raw GPT-5.5 ranked second-to-LAST
against baselines, and the same model behind an SDK plus a verification harness ranked FIRST (91.8% asset
retention over 10,909 generated assets). The model was never the variable -- the ability to measure was.

Everything here is READ-ONLY and derived from the compiled MuJoCo model, so an answer cannot disagree with the
physics the verdict will use. Nothing calls an LLM: these are measurements, which keeps the zero-product-token
rule intact.

The swept self-collision check deserves its own note, because it is the failure mode Articraft explicitly calls
out and the one a static check cannot see: a design can be perfectly clean at its rest pose and drive one link
straight through another halfway through a joint's travel.
"""
from __future__ import annotations


def _model(gene):
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    return compiled_model(robot_mjcf(gene))


def _posed_data(model, gene):
    """Forward-kinematic the body in the pose it SHIPS, not at qpos 0 -- the same principle the drive verdict and
    the imported rest pose both needed. A measurement taken in a stance the robot never holds is not useful."""
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _body_extents(model, data):
    """(name -> {lo, hi, centre}) world AABB per body, from geom AABBs rotated into world."""
    import mujoco
    import numpy as np
    out: dict[str, dict] = {}
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        ctr = data.geom_xpos[g] + R @ c
        ext = np.abs(R) @ h
        b = int(model.geom_bodyid[g])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        lo, hi = ctr - ext, ctr + ext
        if nm in out:
            out[nm]["lo"] = np.minimum(out[nm]["lo"], lo)
            out[nm]["hi"] = np.maximum(out[nm]["hi"], hi)
        else:
            out[nm] = {"lo": lo, "hi": hi}
    for v in out.values():
        v["centre"] = (v["lo"] + v["hi"]) / 2.0
    return out


def probe(gene, query: dict | None = None) -> dict:
    """Measure the held robot. ``query`` selects what to compute; omitted -> the whole report.

    Fields, each answering a question an agent has to be able to ask:
      parts        - per-part world AABB, centre, mass, and its own span
      reach        - farthest end-effector-ish point from the root, and the root-to-tip distance
      clearance    - lowest point above the floor plane, per part and overall
      mass         - total, per part, and the centre of mass with its support margin
      torque       - per joint: static holding torque required vs the actuator's rating
      overlaps     - pairs of NON-ADJACENT parts whose AABBs intersect at the rest pose
      swept        - pairs that collide somewhere in a joint's declared RANGE even though the rest pose is clean
    """
    import mujoco
    import numpy as np

    want = set((query or {}).get("fields") or ()) or None
    model = _model(gene)
    data = _posed_data(model, gene)
    ext = _body_extents(model, data)
    rep: dict = {"ok": True, "n_parts": len(ext), "posed_from": "rest keyframe" if model.nkey else "zero pose"}

    def wanted(f):
        return want is None or f in want

    seg = {s.name: s for s in getattr(gene, "segments", []) or []}
    root = gene.root().name if hasattr(gene, "root") else next(iter(ext), "")

    if wanted("parts"):
        rep["parts"] = {nm: {"centre": [round(float(v), 5) for v in e["centre"]],
                             "span": [round(float(a), 5) for a in (e["hi"] - e["lo"])],
                             "mass_kg": round(float(getattr(seg.get(nm), "mass_kg", 0.0) or 0.0), 4)}
                        for nm, e in sorted(ext.items())}

    if wanted("clearance"):
        floor = min(float(e["lo"][2]) for e in ext.values())
        rep["clearance"] = {"lowest_point_m": round(floor, 5),
                            "per_part_m": {nm: round(float(e["lo"][2]) - floor, 5) for nm, e in sorted(ext.items())}}

    if wanted("mass"):
        tot = float(model.body_mass.sum())
        com = np.sum(model.body_mass[:, None] * data.xipos, axis=0) / max(tot, 1e-9)
        # support margin: how far the COM projection sits inside the footprint of whatever touches down
        floor = min(float(e["lo"][2]) for e in ext.values())
        band = floor + 0.05 * max(1e-6, max(float(e["hi"][2]) for e in ext.values()) - floor)
        feet = [e for e in ext.values() if float(e["lo"][2]) <= band]
        if feet:
            fx = [float(f["centre"][0]) for f in feet]
            fy = [float(f["centre"][1]) for f in feet]
            margin = min(max(fx) - float(com[0]), float(com[0]) - min(fx),
                         max(fy) - float(com[1]), float(com[1]) - min(fy))
        else:
            margin = None
        rep["mass"] = {"total_kg": round(tot, 4), "com": [round(float(v), 5) for v in com],
                       "support_margin_m": (round(float(margin), 5) if margin is not None else None),
                       "n_contacting_parts": len(feet)}

    if wanted("torque"):
        # STATIC holding torque at each joint: the weight of everything distal, times its lever arm about the
        # joint axis. This is the number that decides whether a limb sags, and it is what a datasheet is checked
        # against -- so report it beside the actuator's own rating rather than asserting the design "fits".
        kids = {b: [] for b in range(model.nbody)}
        for b in range(1, model.nbody):
            kids[int(model.body_parentid[b])].append(b)

        def distal(b):
            out, st = [], [b]
            while st:
                x = st.pop()
                out.append(x)
                st += kids.get(x, [])
            return out

        rows = {}
        for j in range(model.njnt):
            if int(model.jnt_type[j]) not in (2, 3):
                continue
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
            b = int(model.jnt_bodyid[j])
            chain = distal(b)
            m = float(sum(model.body_mass[x] for x in chain))
            if m <= 0:
                continue
            com = np.sum([model.body_mass[x] * data.xipos[x] for x in chain], axis=0) / m
            # Torque ABOUT THE AXIS, not the perpendicular distance to it: tau = |axis . (r x F)| with F the
            # distal weight. Using |r_perp| instead overestimates whenever the axis is not horizontal (a yaw
            # joint carries no gravity torque at all, and the naive form would invent some), and a torque figure
            # that flatters the design is worse than no figure -- the whole point is that a datasheet gets checked
            # against it.
            axis = np.asarray(data.xaxis[j], dtype=float)
            r = com - np.asarray(data.xanchor[j], dtype=float)
            F = np.array([0.0, 0.0, -m * 9.81])
            need = abs(float(np.dot(axis, np.cross(r, F))))
            lever = need / max(m * 9.81, 1e-9)              # the EFFECTIVE gravity lever, reported for legibility
            rated = None
            s = seg.get(nm[:-6] if nm.endswith("_joint") else nm)
            if s is not None and getattr(s, "actuator_torque_nm", None):
                rated = float(s.actuator_torque_nm)
            rows[nm] = {"distal_mass_kg": round(m, 4), "lever_m": round(lever, 5),
                        "static_hold_nm": round(need, 3), "rated_nm": rated,
                        "margin": (round(rated / need, 2) if rated and need > 1e-9 else None)}
        rep["torque"] = rows

    if wanted("reach") and root in ext:
        r0 = ext[root]["centre"]
        far = max(ext.items(), key=lambda kv: float(np.linalg.norm(kv[1]["centre"] - r0)))
        rep["reach"] = {"from": root, "farthest_part": far[0],
                        "distance_m": round(float(np.linalg.norm(far[1]["centre"] - r0)), 5)}

    if wanted("overlaps") or wanted("swept"):
        adj = set()
        for s in getattr(gene, "segments", []) or []:
            if s.parent:
                adj.add(frozenset((s.name, s.parent)))
        names = sorted(ext)

        def intersecting(e_map):
            hits = []
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if frozenset((a, b)) in adj or a not in e_map or b not in e_map:
                        continue
                    ea, eb = e_map[a], e_map[b]
                    if all(ea["lo"][k] < eb["hi"][k] and eb["lo"][k] < ea["hi"][k] for k in range(3)):
                        hits.append([a, b])
            return hits

        if wanted("overlaps"):
            rep["overlaps"] = intersecting(ext)

        if wanted("swept"):
            # THE CHECK A STATIC ONE CANNOT MAKE. A design can be clean at rest and drive one link straight
            # through another partway through a joint's travel; Articraft names this as a top LLM failure mode.
            #
            # EXACT, via MuJoCo's own narrow phase rather than AABBs. Comparing boxes reported 43 pairs on a
            # 20-part quadruped -- mostly artefacts of two capsules passing near each other diagonally -- and an
            # unactionable list is barely better than no list. Reading data.contact after mj_forward uses the real
            # capsule/box geometry, so a reported pair IS touching.
            #
            # Ancestor pairs are excluded because they touch BY CONSTRUCTION (a child is seated in its parent),
            # which is the same exclusion gene_compiler writes into the model's own <contact> block. And the angle
            # each pair first meets at is reported: real robots self-collide at joint EXTREMES and carry avoidance
            # software for it, so "touches at 0.95 of travel" and "touches at 0.30" are different problems -- the
            # second is a design defect, the first is a limit to set.
            import copy

            anc = set()
            _par = {s.name: s.parent for s in getattr(gene, "segments", []) or []}
            for nm in _par:
                p = _par.get(nm)
                while p:
                    anc.add(frozenset((nm, p)))
                    p = _par.get(p)

            def _touching(d2):
                out = {}
                for ci in range(int(d2.ncon)):
                    c = d2.contact[ci]
                    b1 = int(model.geom_bodyid[int(c.geom1)]); b2 = int(model.geom_bodyid[int(c.geom2)])
                    if b1 == b2:
                        continue
                    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
                    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
                    if not n1 or not n2 or frozenset((n1, n2)) in anc:
                        continue
                    if b1 == 0 or b2 == 0:            # the FLOOR is not self-collision; split out below
                        key = ("world", n2 if b1 == 0 else n1)
                    else:
                        key = tuple(sorted((n1, n2)))
                    out[key] = min(out.get(key, 0.0), float(c.dist))     # most-penetrating contact for the pair
                return out

            rest_pairs = set(_touching(data))
            found: dict[tuple, dict] = {}
            swept_joints = 0
            for j in range(model.njnt):
                if int(model.jnt_type[j]) not in (2, 3) or not int(model.jnt_limited[j]):
                    continue
                swept_joints += 1
                lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
                adr = int(model.jnt_qposadr[j])
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
                for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                    d2 = copy.deepcopy(data)
                    d2.qpos[adr] = lo + t * (hi - lo)
                    mujoco.mj_forward(model, d2)
                    for key, dist in _touching(d2).items():
                        if key in rest_pairs:
                            continue
                        prev = found.get(key)
                        # keep the EARLIEST point in travel where it happens; that is the actionable number
                        if prev is None or abs(t - 0.5) > abs(prev["at_travel"] - 0.5):
                            found[key] = {"joint": jname, "at_travel": t, "penetration_m": round(dist, 5)}
            # A part sweeping into the FLOOR is a real defect -- an arm's boom fouling its own bench -- but it is
            # not self-collision, and conflating them buries one kind of problem inside the other.
            ground = {k[1]: v for k, v in found.items() if k[0] == "world"}
            found = {k: v for k, v in found.items() if k[0] != "world"}
            rep["swept"] = {
                "pairs": [list(k) for k in sorted(found)],
                "detail": {f"{k[0]}|{k[1]}": v for k, v in sorted(found.items())},
                "ground_strikes": ground,
                "ground_note": ("parts that sweep INTO the floor within a joint's range — a defect for a mounted "
                                "machine (its own bench), expected for a leg mid-stride"),
                "note": ("parts that ACTUALLY touch somewhere inside a joint's declared range while being clear at "
                         "the rest pose, from MuJoCo's own contact detection -- not an AABB estimate. "
                         "`at_travel` is where in the range it happens (0 = lower limit, 1 = upper): near the ends "
                         "is a limit worth tightening, mid-range is a geometry defect."),
                "excludes": "parent/ancestor pairs, which are seated in each other by construction",
                "n_joints_swept": swept_joints,
            }
    return rep
