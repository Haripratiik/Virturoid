"""AI-native STATEFUL tools (docs/ai_native_plan.md P1) — the tool surface for INCREMENTAL, session-held
work that both frontends (the MCP server + the in-app assistant) share. Unlike the stateless
``agent_tools`` (each composes a robot fresh from a prompt), these operate on a robot/scene HELD under an id
in ``session_state``, so an agent can create once and then EDIT in place ("make it taller"), simulate, verify,
and undo. Every tool returns the compact verdict contract ``{ok, ...compact fields..., artifacts?}`` — never a
raw sim/gene dump (SWE-agent ACI). Registered into ``agent_tools.TOOLS`` so the MCP server exposes them too.
"""
from __future__ import annotations

from pathlib import Path

_RENDER_DIR = Path("build/agent_renders")


# ------------------------------------------------------------------ helpers
def _summary(gene, robot_id: str | None = None) -> dict:
    """Compact, LLM-legible read of a robot: class, discovered appendages (GEN-1, structural), size, mass."""
    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.edit_operators import _dominant_material, _standing_height
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    from virturoid.services.task_matched_eval import robot_kind
    app = "?"
    try:
        am = build_appendage_map(compiled_model(robot_mjcf(gene)))
        app = ({"legs": am.n_legs, "wheels": am.n_wheels, "arms": am.n_arms, "spine": am.spine is not None})
    except Exception:  # noqa: BLE001
        pass
    out = {"robot_class": gene.robot_class, "kind": robot_kind(gene), "n_segments": len(gene.segments),
           "dof": len(gene.actuated_joints()), "appendages": app,
           "standing_height_m": _standing_height(gene),
           "total_mass_kg": round(sum(s.mass_kg for s in gene.segments), 3),
           "material": _dominant_material(gene), "end_effector": gene.end_effector_type,
           "design_source": getattr(gene, "design_source", "unknown"),
           "composition_notes": list(getattr(gene, "composition_notes", []) or [])}
    if robot_id:
        out["robot_id"] = robot_id
    return out


def _pose_manipulator_for_render(m, d, gene) -> None:
    """A static rest pose (qpos=0) renders an ARM as a straight vertical pole — which reads as broken. For a
    manipulator, bend its hinge joints into a natural 'ready' pose (an elbow bend + a slight reach) so the still
    shows a real articulated arm. Legged/mobile bodies look right standing, so they're left at rest."""
    import mujoco
    rc = (getattr(gene, "robot_class", "") or "").lower()
    sp = (getattr(gene, "species", "") or "").lower()
    if "manipulator" not in rc and "arm" not in rc and "arm" not in sp:
        return
    bends = [0.5, -1.1, 1.0, -0.6, 0.5]                         # shoulder out, elbow in, wrist — an L-shaped reach
    ai = 0
    for jn in range(m.njnt):
        if m.jnt_type[jn] != mujoco.mjtJoint.mjJNT_HINGE:       # only revolute arm joints (leave gripper slides)
            continue
        adr = int(m.jnt_qposadr[jn])
        a = bends[ai] if ai < len(bends) else 0.3
        if m.jnt_limited[jn]:                                   # clamp the bend into the joint's real range
            lo, hi = float(m.jnt_range[jn][0]), float(m.jnt_range[jn][1])
            a = min(max(a, lo + 0.05 * (hi - lo)), hi - 0.05 * (hi - lo))
        d.qpos[adr] = a
        ai += 1


def _render_gene(gene, tag: str, *, azimuth: float = 50.0, elevation: float = -16.0) -> str | None:
    import os
    os.environ.setdefault("MUJOCO_GL", "glfw")
    try:
        import mujoco
        import PIL.Image

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z
        _RENDER_DIR.mkdir(parents=True, exist_ok=True)
        spawn_z = standing_spawn_z(gene)
        # Render the MESHED model (the true geometry the app viewport shows), NOT the crude box collider — the
        # non-meshed render drew a chassis as its tiny bounding box, so a wheeled body read as a small box with
        # oversized disconnected wheels. Fall back to the primitive model only if meshing fails.
        try:
            xml = gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=spawn_z)
        except Exception:  # noqa: BLE001
            xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn_z)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
        _pose_manipulator_for_render(m, d, gene)               # bend an arm into a natural pose (no-op for others)
        mujoco.mj_forward(m, d)
        rr = mujoco.Renderer(m, height=420, width=560); cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.15]; cam.distance, cam.azimuth, cam.elevation = 1.9, float(azimuth), float(elevation)
        rr.update_scene(d, camera=cam); img = PIL.Image.fromarray(rr.render().copy()); rr.close()
        path = _RENDER_DIR / f"{tag}.png"; img.save(path)
        return str(path)
    except Exception:  # noqa: BLE001 - rendering is value-add; the edit/verdict still stands
        return None


# swim/fly INTENT words. AQUATIC bodies are simulated in water (T7, _honest_swim) and AERIAL bodies are flown as
# quadcopters (aerial.py, _honest_fly) — both real tiers now. These lists are the fall-through guard: if a
# swim/fly prompt somehow reaches a LAND verdict, _flag_physics_envelope flags it as a land proxy rather than
# letting it masquerade as the real capability.
# canonical aquatic vocabulary lives in aquatic.py — importing it (rather than keeping a second copy) is what
# stops the two lists drifting apart: they HAD drifted (this one knew octopus/squid, the routing one knew
# manta/salmon/koi), so an octopus prompt was judged by a land-walk verdict it could never pass.
from virturoid.services.aquatic import AQUATIC_ENV_WORDS as _AQUATIC_WORDS  # noqa: E402
_AERIAL_WORDS = ("fly", "flying", "aerial", "drone", "quadcopter", "helicopter", "hover", "aircraft", "winged")


def _env_words(prompt: str, words) -> bool:
    """WORD-BOUNDARY match (not substring) so 'wh-EEL-ed'/'sp-RAY' don't false-trigger the fluid/aerial tier."""
    import re
    p = (prompt or "").lower()
    return any(re.search(rf"\b{re.escape(w)}\b", p) for w in words)


def _flag_physics_envelope(res: dict, prompt: str, kind: str) -> None:
    """If the prompt implies a swim/fly envelope but the body fell through to a LAND verdict, annotate it
    honestly instead of letting a land-gait verdict masquerade as the real capability. AQUATIC bodies are
    simulated in water (T7) and AERIAL bodies are flown as quadcopters (aerial.py) — when either was actually
    simulated, ``res['kind']`` is already 'aquatic'/'aerial' and the verdict stands."""
    if res.get("kind") in ("aquatic", "aerial"):
        return                                                 # already simulated in its own medium — verdict stands
    env = "aquatic" if _env_words(prompt, _AQUATIC_WORDS) else ("aerial" if _env_words(prompt, _AERIAL_WORDS) else None)
    if env and kind in ("legged", "mobile"):
        res["physics_envelope"] = env
        res["credible_walk"] = False
        # Reaching here means the body was NOT simulated in its own medium (that case returned at the top), so
        # BOTH notes must say "land-based proxy". The aquatic note used to claim "it was simulated in water (see
        # swim_m)" — false on this branch, and newly reachable now that radial cephalopods are correctly
        # recognised as aquatic while keeping their (non-undulator) morphology.
        res["envelope_note"] = (
            "this prompt implies an AERIAL body; Virturoid flies quadcopters (build a drone and verify it), but "
            "this body was composed as a terrestrial one, so the verdict above is a LAND-BASED PROXY." if env == "aerial"
            else "this prompt implies an AQUATIC body, but it was composed as a non-undulator (e.g. a radial "
                 "cephalopod) and evaluated on land, so the verdict above is a LAND-BASED PROXY, not a swim "
                 "result. Virturoid swims serial-spine undulators (build a fish/eel and verify it).")


def _swim_model(gene):
    """T7: compile the gene into a WATER medium — a fluid <option> (density 1000, viscosity ~1e-3, gravity 0
    for neutral buoyancy) + per-geom ``fluidshape='ellipsoid'`` drag on the collision geoms. This is a REAL
    MuJoCo fluid model (verified: an idealised undulator swims ~0.85 m); density MUST be set at compile so the
    ellipsoid ``geom_fluid`` coefficients are populated."""
    import re
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    xml = compile_gene_to_mjcf(gene, include_floor=False)
    mm = re.search(r"<option\b([^>]*?)(/?)>", xml)
    if mm:
        attrs = re.sub(r'\s(density|viscosity|gravity)="[^"]*"', "", mm.group(1))
        xml = (xml[:mm.start()] + f'<option{attrs} density="1000" viscosity="0.0009" gravity="0 0 0"{mm.group(2)}>'
               + xml[mm.end():])
    else:
        xml = re.sub(r"(<mujoco\b[^>]*>)", r'\1<option density="1000" viscosity="0.0009" gravity="0 0 0"/>', xml, count=1)

    def _add(g):
        s = g.group(0)
        return s if ('type="plane"' in s or 'contype="0"' in s or "fluidshape=" in s) else s[:-2] + ' fluidshape="ellipsoid"/>'
    xml = re.sub(r"<geom\b[^>]*?/>", _add, xml)
    return mujoco.MjModel.from_xml_string(xml)


def _honest_swim(gene, *, steps: int = 2500) -> dict:
    """T7: the honest SWIM verdict — put the body in water (neutral buoyancy) and drive its joints with a
    PD-tracked travelling wave, measuring REAL net planar thrust. Like the gait/drive verdicts it never lies:
    a body whose geometry can't generate thrust honestly reads DOES NOT SWIM. The fluid physics is real; swim
    performance (like walk performance) is body/control-dependent (the general swim controller is a frontier)."""
    import numpy as np
    import mujoco
    try:
        m = _swim_model(gene)
    except Exception as exc:  # noqa: BLE001
        return {"kind": "aquatic", "verdict": f"could not build fluid model ({type(exc).__name__})", "swim_m": 0.0}
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    if m.nu == 0:
        return {"kind": "aquatic", "verdict": "NO ACTUATORS (cannot swim)", "survived": True, "swim_m": 0.0}
    bid, _free = _base_body_id(m)
    qadr = [m.jnt_qposadr[int(m.actuator_trnid[u, 0])] for u in range(m.nu)]
    vadr = [m.jnt_dofadr[int(m.actuator_trnid[u, 0])] for u in range(m.nu)]
    frc = m.actuator_forcerange[:, 1].copy(); frc[frc <= 0] = 3.0
    from virturoid.services.aquatic import SWIM_WAVE
    w = SWIM_WAVE
    p0 = np.array(d.xpos[bid]).copy()
    for t in range(steps):
        ph = 2 * np.pi * t * m.opt.timestep * w["freq"]
        for k in range(m.nu):                                   # PD-track the tuned travelling wave head->tail
            tgt = w["amp"] * np.sin(ph - k * w["wavenum"])
            d.ctrl[k] = float(np.clip(w["kp"] * (tgt - d.qpos[qadr[k]]) - w["kd"] * d.qvel[vadr[k]],
                                      -frc[k], frc[k]))
        mujoco.mj_step(m, d)
    swim = float(np.hypot(*(np.array(d.xpos[bid])[:2] - p0[:2])))
    verdict = (f"SWIMS ({swim:.2f} m undulatory thrust)" if swim > 0.15
               else f"DOES NOT SWIM ({swim:.2f} m — this body's geometry yields little thrust)")
    return {"kind": "aquatic", "verdict": verdict, "survived": True, "swim_m": round(swim, 3),
            "n_actuators": int(m.nu),
            "note": "REAL MuJoCo fluid sim (water + neutral buoyancy); thrust is body/geometry-dependent"}


def _honest_fly(gene, *, target=(1.5, 0.0, 1.2), steps: int = 2000) -> dict:
    """The honest FLY verdict — put the quadcopter in the air and drive it with the geometric flight controller
    (four rotor THRUST forces along body-up; desired-acceleration -> desired-attitude -> rotation-matrix attitude
    error -> body torque -> rotor allocation). REAL MuJoCo rigid-body dynamics + gravity; the ONLY added physics
    is the rotor thrust a real quadcopter's motors produce. Like the gait/swim verdicts it never lies: a body
    that cannot reach and hold its target honestly reads DOES NOT FLY. Attitude torque is inertia-normalized
    (``I_assembly @ alpha_des``), so the same gains fly any drone size.

    The DEFAULT target is a LATERAL waypoint (fly forward 1.5 m + up), so the verdict tests real flight — takeoff,
    BANK, translate, arrive, hold — not a straight-up hover a non-translating body could fake (un-gameable: a
    hover-only body reaches ~0 m of a lateral waypoint and honestly reads HOVERS/DOES NOT FLY)."""
    import mujoco
    import numpy as np

    from virturoid.services.aerial import FLY_GAINS
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    try:
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=0.18))
    except Exception as exc:  # noqa: BLE001
        return {"kind": "aerial", "verdict": f"could not build flight model ({type(exc).__name__})", "flew_m": 0.0}
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    bid, is_free = _base_body_id(m)
    if not is_free:
        return {"kind": "aerial", "verdict": "NOT A FLOATING BODY (cannot fly)", "survived": True, "flew_m": 0.0}
    md = getattr(gene, "metadata", None) or {}
    offs = md.get("rotor_offsets") or [[0.16, 0.16], [-0.16, 0.16], [-0.16, -0.16], [0.16, -0.16]]
    L = float(md.get("rotor_L") or max(abs(offs[0][0]), 1e-3))
    mass = float(sum(m.body_mass)); g = 9.81
    # assembly rotational inertia (body frame) = the angular block of the free joint's 6x6 mass matrix
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM)
    adr = int(m.jnt_dofadr[[j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]])
    I_ang = M[adr + 3:adr + 6, adr + 3:adr + 6].copy()
    kp = np.array(FLY_GAINS["kp"]); kd = np.array(FLY_GAINS["kd"]); KR = FLY_GAINS["KR"]; KW = FLY_GAINS["KW"]
    pref = np.array(target, float); tilts, zs, dists = [], [], []
    for t in range(steps):
        p = d.qpos[0:3].copy(); v = d.qvel[0:3].copy()
        R = d.xmat[bid].reshape(3, 3).copy(); omega = d.qvel[adr + 3:adr + 6].copy()
        a_des = kp * np.clip(pref - p, -FLY_GAINS["vcap"], FLY_GAINS["vcap"]) - kd * v + np.array([0, 0, g])
        a_des[2] = max(a_des[2], 2.0)                          # keep positive collective (never command < ~0.2 g)
        b3d = a_des / (np.linalg.norm(a_des) + 1e-9)           # desired body-up (thrust) direction
        b2d = np.cross(b3d, np.array([1.0, 0.0, 0.0])); b2d /= (np.linalg.norm(b2d) + 1e-9)
        Rd = np.column_stack([np.cross(b2d, b3d), b2d, b3d])   # desired attitude (yaw fixed to +x heading)
        eRm = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([eRm[2, 1], eRm[0, 2], eRm[1, 0]])       # vee(attitude error), body frame
        thrust = mass * float(a_des @ R[:, 2])                 # collective, projected onto the CURRENT body-up
        tau = I_ang @ (-KR * eR - KW * omega)                  # inertia-normalized attitude torque (body frame)
        T4 = thrust / 4.0; tr = tau[0] / (4 * L); tp = tau[1] / (4 * L)
        f = [T4 + tr - tp, T4 + tr + tp, T4 - tr + tp, T4 - tr - tp]   # X-config rotor allocation
        d.qfrc_applied[:] = 0
        up = R[:, 2]
        for i, (ox, oy) in enumerate(offs):
            pt = d.xpos[bid] + R @ np.array([ox, oy, 0.0])     # rotor world position
            mujoco.mj_applyFT(m, d, up * max(0.0, float(f[i])), np.zeros(3), pt, bid, d.qfrc_applied)
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return {"kind": "aerial", "verdict": f"UNSTABLE (diverged at {t * m.opt.timestep:.1f}s) — DOES NOT FLY",
                    "survived": False, "flew_m": 0.0}
        if t % 5 == 0:
            zs.append(float(p[2])); dists.append(float(np.linalg.norm(pref - p)))
            roll = np.arctan2(R[2, 1], R[2, 2]); pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
            tilts.append(float(max(abs(roll), abs(pitch))))
    fp = d.qpos[0:3]
    dist = float(np.linalg.norm(pref - fp))
    n = len(zs)
    airborne = n > 200 and min(zs[n // 4:]) > 0.5              # stayed off the ground for the last 3/4 of the run
    settled = float(np.mean(dists[-40:])) if len(dists) >= 40 else dist   # held near target at the end
    max_tilt_deg = float(np.degrees(max(tilts))) if tilts else 0.0
    reached = airborne and settled < 0.30
    flew = float(np.linalg.norm(fp[:2]))                       # horizontal translation from spawn (un-gameable-by-hover)
    verdict = (f"FLIES ({flew:.2f} m horizontal to waypoint, arrived {dist:.2f} m off at {max_tilt_deg:.0f} deg bank)"
               if reached
               else (f"HOVERS but does not reach target ({settled:.2f} m off)" if airborne
                     else f"DOES NOT FLY (never sustained altitude, {settled:.2f} m off target)"))
    return {"kind": "aerial", "verdict": verdict, "survived": bool(airborne),
            "flew_m": round(float(np.linalg.norm(fp[:2])), 3), "dist_to_target": round(dist, 3),
            "final_pos": [round(float(x), 3) for x in fp], "target": [round(float(x), 3) for x in pref],
            "max_tilt_deg": round(max_tilt_deg, 1), "reached_target": bool(reached),
            "note": "REAL MuJoCo rigid-body + gravity; four rotor thrust forces (a quadcopter's actual actuation), "
                    "geometric flight controller, inertia-normalized attitude"}


def _auto_bank_gait(gene, r, base_params) -> str | None:
    """FLYWHEEL SELF-UPDATE: bank the working gait a CREDIBLE walk just demonstrated, so the flywheel COMPOUNDS on
    ordinary build+verify — not only under explicit training or a manual GPU night run (the reason the bank sat
    empty: nothing on the ordinary path ever wrote to it). bank_gait is keep-best + keyed by morphology, so
    re-verifying the same body updates ONE skill (a stronger later gait replaces a weaker one), never floods."""
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT, _DeployResult, bank_gait
    from virturoid.services.gait_quality import classify
    from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    params = {**_DEFAULT_GAIT, **{k: float(v) for k, v in (base_params or {}).items()}}
    # Self-validate credibility from the rollout itself (defense in depth): bank_gait rejects a non-credible slide,
    # so the flywheel self-update never banks a body that merely SLID forward even if a caller forgot to gate on it.
    holder = _DeployResult(params, {"forward": float(r.get("forward", 0.0)),
                                    "height_ratio": float(r.get("height_ratio", 1.0)),
                                    "survived": bool(r.get("survived", False)),
                                    "credible": classify(r).startswith("CREDIBLE")})
    with MemoryDB(DEFAULT_DB_PATH) as db:
        # GROW THE TRANSFER LEDGER (P3) from real usage: when enabled, banking a credible walk also cross-evaluates
        # it on the K nearest banked bodies (2K bounded rollouts), so the physics-verified transfer corpus the gated
        # metric learns from accrues from ordinary builds. Off by default (keeps quick verify fast); a batch/night
        # run or deliberate build sets VIRTUROID_GROW_TRANSFER_LEDGER=1 so the moat compounds without slowing iteration.
        import os
        return bank_gait(db, gene, holder,
                         cross_eval=os.environ.get("VIRTUROID_GROW_TRANSFER_LEDGER", "0") == "1")


def _record_gait_lesson(gene, failure_code: str, operator: str, *, improvement: float,
                        root_cause: str | None = None) -> None:
    """LESSON WRITE on the common LOCOMOTION path (which the reasoned-redesign lesson writer skips, leaving the
    lessons store dead). A deploy-select disagreement IS a physics-verified failure->fix: the banked hint didn't
    transfer to THIS body, and ``operator`` is what fixed it. Keyed by class, idempotent keep-best, so it becomes
    grounding for the next similar body (surfaced by ``gait_hints.mine_gait_hints`` via ``lessons_for_class``)."""
    try:
        from virturoid.services.gait_flywheel import _class_of
        from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MemoryDB(DEFAULT_DB_PATH) as db:
            db.record_lesson(_class_of(gene), failure_code, operator, task_type="locomotion",
                             improvement=round(float(improvement), 4), root_cause=root_cause,
                             source_gene=getattr(gene, "id", None))
    except Exception:  # noqa: BLE001 - lessons are an accelerant; never break a verdict to write one
        pass


def _honest_serpentine(gene, *, steps: int = 2000, render: bool = False, tag: str = "serpentine") -> dict:
    """The honest LAND-SERPENTINE verdict for a LIMBLESS serial spine (a snake): a lateral travelling wave down
    the spine crawls it forward against ground friction (morph_policy.serpentine_rollout). A snake has no legs,
    so the leg crawl gait can't drive it (it falls); this drives the spine directly. Never lies — a spine whose
    undulation yields no net thrust reads DOES NOT CRAWL."""
    from virturoid.services.morph_policy import serpentine_rollout
    r = serpentine_rollout(gene, steps=steps, record_qpos=render)
    m = float(r.get("planar_m", 0.0))
    verdict = (f"CRAWLS (serpentine, {m:.2f} m lateral undulation)" if m > 0.15 and r.get("finite")
               else f"DOES NOT CRAWL ({m:.2f} m — this spine's undulation yields little ground thrust)")
    out = {"kind": "legged", "verdict": verdict, "survived": bool(r.get("survived")), "gait_source": "serpentine",
           "forward_m": round(m, 3), "credible_walk": False, "n_actuators": r.get("n_actuators"),
           "note": "limbless serial spine -> LAND serpentine (lateral travelling wave), not a leg gait"}
    if render and r.get("qpos_frames"):
        gif = _render_gait_gif(gene, r["qpos_frames"], tag)
        if gif:
            out["artifacts"] = [gif]
    return out


def _honest_biped(gene, *, steps: int = 1500) -> dict | None:
    """A BIPED (2-legged) body: honestly report whether it STANDS (static balance, PD-holding its stance) vs the
    multi-leg crawl gait that just FELLS it (that wave gait is for >=4 legs). Returns None if the body isn't a
    biped or can't even stand (let the fall verdict stand). A humanoid STANDS statically but DYNAMIC bipedal
    WALKING is a learned-control frontier (a scripted gait can't balance a walking biped) — say exactly that,
    rather than a flat 'FELL' that implies it can't balance at all."""
    import mujoco
    import numpy as np

    from virturoid.services.appendage_map import build_appendage_map
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    if build_appendage_map(m).n_legs != 2 or m.nu == 0:
        return None                                              # not a biped -> let the normal gait verdict stand
    # FIRST: the biped's BEST real controller — a banked LEARNED walk policy if one exists (the GPU-trained
    # humanoid), else the trot. If it moves the body FORWARD while upright, report THAT honest walk (the multi-leg
    # crawl wave gait that fells it is just the wrong controller for 2 legs).
    try:
        from virturoid.services.learn_locomotion import locomotion_episode
        lr = locomotion_episode(gene, horizon=steps)
        fwd = float(lr.get("forward_m", 0.0))
        if fwd > 0.3 and bool(lr.get("upright")):
            src = "learned policy" if lr.get("source") == "learned" else "trot gait"
            return {"kind": "legged", "verdict": f"WALKS FORWARD ({fwd:.2f} m, upright, {src})",
                    "survived": True, "gait_source": "biped_" + str(lr.get("source", "trot")),
                    "forward_m": round(fwd, 3), "credible_walk": False,
                    "note": "biped: forward locomotion via its best controller, not the multi-leg crawl wave gait"}
    except Exception:  # noqa: BLE001 - fall through to the static-balance check
        pass
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2]) or 1.0
    qref = d.qpos.copy()
    qadr = [m.jnt_qposadr[int(m.actuator_trnid[u, 0])] for u in range(m.nu)]
    vadr = [m.jnt_dofadr[int(m.actuator_trnid[u, 0])] for u in range(m.nu)]
    frc = m.actuator_forcerange[:, 1].copy(); frc[frc <= 0] = 8.0
    upr_frac = 0
    for t in range(steps):                                       # PD-hold the standing stance (static balance)
        for k in range(m.nu):
            d.ctrl[k] = float(np.clip(80.0 * (qref[qadr[k]] - d.qpos[qadr[k]]) - 4.0 * d.qvel[vadr[k]],
                                      -frc[k], frc[k]))
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            break
        q = d.qpos[3:7]; upr = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
        if float(d.qpos[2]) > 0.6 * z0 and upr > 0.7:
            upr_frac += 1
    stands = upr_frac > 0.8 * steps
    if not stands:
        return None                                              # can't even stand -> the honest FALL verdict stands
    return {"kind": "legged", "verdict": "STANDS (static balance); dynamic bipedal walking is a learned-control "
                                         "frontier (a scripted gait can't balance a walking biped — needs a "
                                         "learned policy / the GPU trainer)",
            "survived": True, "gait_source": "biped_stand", "forward_m": 0.0, "credible_walk": False,
            "upright_frac": round(upr_frac / max(1, steps), 3),
            "note": "biped: static balance holds; the multi-leg crawl wave gait is the wrong controller for 2 legs"}


def _learned_gait_attempt(gene) -> dict | None:
    """v7-F1 (master_plan_v7 §1): deploy the best banked LEARNED MorphPolicy for this body and judge it under the
    SAME un-gameable bar as the scripted gait — ``classify()`` including the ROLL/PITCH gate (the recall screen
    records qpos frames, v7-F2). This is the missing product half of the deploy gap: trained policies were banked
    by ``train_gene_on_gpu``/``transfer_train_morph`` but the verdict path only ever ran the scripted crawl, so
    training quality was invisible to the product. Returns ``{"out": verdict_dict, "rollout": r}`` when a banked
    policy is credible ON THIS BODY, else ``None`` (nothing banked / not credible here / no DB)."""
    from virturoid.services.gait_quality import classify, orientation_summary
    from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
    from virturoid.services.policy_flywheel import recall_morph_policy
    if not DEFAULT_DB_PATH.exists():
        return None
    with MemoryDB(DEFAULT_DB_PATH) as db:
        policy, r = recall_morph_policy(gene, db, with_rollout=True)   # screens credibility on THIS body
    if policy is None or not r:
        return None
    verdict = classify(r)
    if not verdict.startswith("CREDIBLE"):                    # the screen and the verdict share one bar; belt+braces
        return None
    o = orientation_summary(r.get("qpos_frames") or [])
    out = {"kind": "legged", "verdict": verdict, "survived": bool(r.get("survived")),
           "gait_source": "learned_policy",
           "forward_m": round(float(r.get("forward", 0)), 3),
           "speed_mps": round(float(r.get("speed", 0)), 3), "cadence": round(float(r.get("cadence", 0)), 1),
           "support_frac": round(float(r.get("support_frac", 0)), 2), "height_ratio": r.get("height_ratio"),
           "roll_max_deg": o.get("roll_max"), "pitch_max_deg": o.get("pitch_max"),
           "note": "walked by a banked LEARNED MorphPolicy (flywheel skill) — the scripted gait was not credible "
                   "on this body"}
    return {"out": out, "rollout": r}


def _honest_gait(gene, *, steps: int = 1200, render: bool = False, tag: str = "gait") -> dict:
    """Run the general scripted gait and return the ANTI-GOODHART verdict (survived+cadence+support+upright+
    forward, forward == actual displacement) — the honesty gate as a tool result, never a raw qpos dump."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.gait_quality import classify, orientation_summary
    # A LIMBLESS serial spine (a snake) has no legs for the crawl gait — drive it as a land SERPENTINE undulator
    # instead (else it just falls / scores 0). The land analogue of routing a fish to the swim tier.
    try:
        from virturoid.services.aquatic import _is_serial_spine
        if _is_serial_spine(gene):
            return _honest_serpentine(gene, steps=max(steps, 2000), render=render, tag=f"{tag}_serp")
    except Exception:  # noqa: BLE001 - serpentine routing is value-add; fall back to the leg gait on any error
        pass
    # FLYWHEEL: use the best banked LEARNED gait for this body's morphology if one exists (recalled by embedding),
    # so the product's legged robots walk with learned control that compounds over builds — else the shipped default.
    gait_params: dict = {}
    gait_source = "default_crawl"
    try:
        # FLYWHEEL = HINTS, NOT COPY-PASTE. Rather than deploy one banked body's exact params verbatim (a trap on a
        # slightly-different body), start from the mined HINT REGION — where credible walks CLUSTER across bodies,
        # auto-derived from data (gait_hints). A quick verify uses this data-driven prior; the ``adapt_gait`` tool
        # runs the full per-body fit from the same hints. The deploy-select below still guards it vs the default.
        from virturoid.services.gait_hints import mine_gait_hints
        from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
        if DEFAULT_DB_PATH.exists():
            with MemoryDB(DEFAULT_DB_PATH) as _db:
                _h = mine_gait_hints(_db, gene=gene)              # VECTOR-nearest robots seed the deploy hint
            if _h.get("n", 0) >= 2:                              # ≥2 banked walks -> a real mined region to hint from
                _p = _h["prior"]
                gait_params = {k: float(_p[k]) for k in ("freq", "hip_amp", "knee_amp", "duty", "kp", "kd")
                               if k in _p}
                gait_source = "flywheel_hint"                    # a data-driven hint region, not a copied policy
    except Exception:  # noqa: BLE001 - the flywheel is an accelerant; a miss just uses the default gait
        gait_params = {}
    r = crawl_gait_rollout(gene, steps=steps, record_qpos=True, **gait_params)
    # DEPLOY-SELECT safety net: a recalled gait must never make THIS body walk worse than the shipped default
    # (gene-construction paths differ, so a banked gait may not fit every body). When a gait was recalled, ALSO
    # run the default and keep whichever is CREDIBLE (tie-break: further) — so a mismatched banked SLIDE can never
    # beat a credible default. The flywheel only ever helps.
    if gait_params:
        r_def = crawl_gait_rollout(gene, steps=steps, record_qpos=True)
        cred_r = classify(r).startswith("CREDIBLE")
        cred_def = classify(r_def).startswith("CREDIBLE")
        better_def = (cred_def and not cred_r) or (
            cred_def == cred_r and abs(float(r_def.get("forward", 0))) > abs(float(r.get("forward", 0))))
        if better_def:
            # the banked hint underperformed the default ON THIS BODY -> a verified failure->fix lesson (the
            # locomotion path's only lesson source; feeds mine_gait_hints for the next similar body)
            _record_gait_lesson(gene, "banked_gait_underperformed", "deploy_default_crawl",
                                improvement=abs(float(r_def.get("forward", 0))) - abs(float(r.get("forward", 0))),
                                root_cause=f"recalled '{gait_source}' gait slid/underperformed the shipped default")
            r, gait_source = r_def, "default_crawl"
    o = orientation_summary(r.get("qpos_frames") or [])
    out = {"kind": "legged", "verdict": classify(r), "survived": bool(r.get("survived")),
           "gait_source": gait_source,
           "forward_m": round(float(r.get("forward", 0)), 3),
           "speed_mps": round(float(r.get("speed", 0)), 3), "cadence": round(float(r.get("cadence", 0)), 1),
           "support_frac": round(float(r.get("support_frac", 0)), 2), "height_ratio": r.get("height_ratio"),
           "roll_max_deg": o.get("roll_max"), "pitch_max_deg": o.get("pitch_max")}
    # v7-F1 LEARNED-CONTROL DEPLOY: when the scripted gait is NOT credible, this body may still walk with a banked
    # LEARNED policy — the product verdict must use the robot's BEST controller, not only the scripted prior.
    # Never-regress by construction: the learned rollout must itself be classify()-CREDIBLE (same bar, roll/pitch
    # included) or the scripted verdict stands; a credible scripted walk skips this entirely (cheap fast path).
    if not str(out["verdict"]).startswith("CREDIBLE"):
        try:
            learned = _learned_gait_attempt(gene)
            if learned is not None:
                out, r = learned["out"], learned["rollout"]   # adopt the learned walk (+ its frames for the render)
        except Exception:  # noqa: BLE001 - learned recall is an accelerant; the scripted verdict stands on any error
            pass
    # BIPED honesty: the multi-leg crawl wave gait FELLS a 2-legged body (wrong controller). If it's a biped that
    # STANDS statically, report that + flag dynamic walking as the learned-control frontier — not a flat 'FELL'
    # implying it can't balance at all. Only runs when the crawl was NOT a credible walk (walkers are unaffected).
    if not str(out["verdict"]).startswith("CREDIBLE"):
        try:
            biped = _honest_biped(gene)
            if biped is not None:
                return biped
        except Exception:  # noqa: BLE001 - the biped-stand check is value-add; keep the gait verdict on any error
            pass
    # NOTE on flywheel ADAPTATION at verify time: the `adapt_gait` tool warm-starts a short search from the mined
    # hints to FIT a gait to a body — but on a THIN corpus it can't reliably rescue a far-out-of-distribution body
    # (a "large quadruped" needs ~2.9 Hz; the hints, from a few near-1.5 Hz walks, seed too far away, so a bounded
    # search finds a slide, not a credible walk). Wiring it into every failed verify added ~15 s with no reliable
    # payoff on a cold corpus, so it stays an EXPLICIT tool (adapt_gait) the agent invokes; the quick verify keeps
    # its honest FELL. As the corpus banks credible walks for diverse bodies, the hints enrich and this earns its
    # place in the deploy path — the flywheel improving with usage, exactly as intended.
    # FLYWHEEL SELF-UPDATE: a CREDIBLE walk is a working controller -> bank it so future similar bodies recall it
    # (the compounding loop, now driven by ordinary verify, not just explicit training). Best-effort + keep-best.
    # A LEARNED-policy walk is excluded: it is ALREADY banked as a policy skill, and banking its rollout under
    # crawl-gait params would poison the gait corpus with params that never produced that rollout (v7-F1).
    if str(out["verdict"]).startswith("CREDIBLE") and out["survived"] and out.get("gait_source") != "learned_policy":
        try:
            _base = gait_params or (getattr(gene, "metadata", None) or {}).get("gait_params") or {}
            banked = _auto_bank_gait(gene, r, _base)
            if banked:
                out["flywheel_banked"] = banked
        except Exception:  # noqa: BLE001 - self-update is an accelerant; never let it break a verdict
            pass
    if render and r.get("qpos_frames"):
        gif = _render_gait_gif(gene, r["qpos_frames"], tag)
        if gif:
            out["artifacts"] = [gif]
    return out


def _base_body_id(model):
    """The free-base body (its 6-DOF free joint), or the first non-world body for a fixed base."""
    import mujoco
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j]), True
    return (1 if model.nbody > 1 else 0), False


def _compose_scene_xml(gene, sg) -> str | None:
    """B2: robot MJCF + the held scene's OBSTACLES spliced into the worldbody (the robot MJCF already brings a
    floor, so the scene's own floor is skipped). Returns None if the composed model won't compile — the caller
    then falls back to the bare-floor rollout with an honest note."""
    import mujoco
    from virturoid.services.morph_policy import robot_mjcf
    from virturoid.services.mujoco_exporter import _scene_objects_xml
    xml = robot_mjcf(gene)
    objs = [o for o in sg.objects if getattr(o, "object_type", "") not in ("floor",)]
    if not objs:
        return None
    geoms = "\n".join(_scene_objects_xml(objs))
    composed = xml.replace("</worldbody>", geoms + "\n</worldbody>", 1)
    try:
        mujoco.MjModel.from_xml_string(composed)               # validate it compiles before committing to it
        return composed
    except Exception:  # noqa: BLE001 - a scene material/name clash -> fall back to bare floor, honestly noted
        return None


def _honest_drive(gene, *, steps: int = 800, world_xml: str | None = None) -> dict:
    """B1: the DRIVE verdict for a wheeled/mobile body — the wheeled analogue of the gait verdict, VERIFIED so
    it can't be gamed: 'DRIVES' requires the wheels to be in GROUND CONTACT, actually SPINNING, and the base
    to stay upright at ~constant height — not a slide, a tip, or wheels spinning in the air. B2: ``world_xml``
    (robot composed into a scene) makes the drive contend with real obstacles."""
    import numpy as np
    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    model = compiled_model(world_xml or robot_mjcf(gene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    bid, _free = _base_body_id(model)
    if model.nu == 0:
        return {"kind": "mobile", "verdict": "NO ACTUATORS (cannot drive)", "survived": True, "forward_m": 0.0,
                "note": "task capability (navigation/goal-reach) via evaluate_held"}
    # identify wheel geoms (the cylinders), the floor plane, and each actuator's driven joint dof
    wheel_geoms = {g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER}
    floor_geoms = {g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
    wheel_r = float(np.mean([model.geom_size[g][0] for g in wheel_geoms])) if wheel_geoms else 0.05
    dofs = [int(model.jnt_dofadr[int(model.actuator_trnid[u, 0])]) for u in range(model.nu)]
    for _ in range(60):                                             # brief settle to steady wheel contact
        mujoco.mj_step(model, data)
    p0 = np.array(data.xpos[bid]); z0 = float(p0[2]); up_min = 1.0
    hi, lo = model.actuator_ctrlrange[:, 1], model.actuator_ctrlrange[:, 0]
    frc = model.actuator_forcerange[:, 1]
    drive = np.where(hi > lo, 0.85 * hi, np.where(frc > 0, 0.7 * frc, 2.0))
    contact_steps, spins = 0, []
    for _ in range(steps):
        data.ctrl[:] = drive
        mujoco.mj_step(model, data)
        up_min = min(up_min, float(data.xmat[bid].reshape(3, 3)[2, 2]))
        touching = any((c.geom1 in wheel_geoms and c.geom2 in floor_geoms) or
                       (c.geom2 in wheel_geoms and c.geom1 in floor_geoms) for c in data.contact[:data.ncon])
        contact_steps += int(touching)
        spins.append(float(np.mean(np.abs([data.qvel[j] for j in dofs]))))
    p1 = np.array(data.xpos[bid])
    fwd = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    dz = float(p1[2] - z0)
    spin = float(np.mean(spins))                                    # mean wheel angular speed (rad/s)
    contact_frac = contact_steps / max(1, steps)
    dt = float(model.opt.timestep)
    roll_ref = spin * steps * dt * wheel_r                          # distance if the spin were pure rolling
    slip = 1.0 - min(1.0, fwd / roll_ref) if roll_ref > 1e-6 else 1.0
    upright = up_min > 0.5 and dz > -0.15
    if not upright:
        verdict = "TIPPED (lost upright / fell while driving)"
    elif spin < 0.3:
        verdict = "STUCK (wheels do not turn)"
    elif contact_frac < 0.2:
        verdict = "WHEELS OFF GROUND (spinning in the air, no traction)"
    elif fwd < 0.12:
        verdict = "SPINS IN PLACE (wheels turn + touch, but no travel)"
    else:
        verdict = f"DRIVES ({fwd:.2f} m, wheel-slip {int(slip * 100)}%)"
    return {"kind": "mobile", "verdict": verdict, "survived": bool(upright and contact_frac > 0.2),
            "forward_m": round(fwd, 3), "upright_min": round(up_min, 2),
            "wheel_spin_radps": round(spin, 2), "wheel_ground_contact_frac": round(contact_frac, 2),
            "n_actuators": int(model.nu), "note": "task capability (navigation/goal-reach) via evaluate_held"}


def _honest_reach(gene, *, steps: int = 500, world_xml: str | None = None) -> dict:
    """B1: the REACH verdict for a manipulator — sweep the joints and measure the end-effector's workspace
    travel + base stability. An arm is NOT scored on locomotion; its capability verdict is evaluate_held."""
    import numpy as np
    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    model = compiled_model(world_xml or robot_mjcf(gene))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    if model.nu == 0:
        return {"kind": "manipulator", "verdict": "NO ACTUATORS (cannot articulate)", "survived": True, "reach_span_m": 0.0}
    ee = None                                                       # the end-effector segment's body
    for name in (s.name for s in gene.segments if getattr(s, "is_end_effector", False)):
        try:
            ee = model.body(name).id; break
        except (KeyError, ValueError):
            continue
    if ee is None:
        ee, _ = _base_body_id(model); ee = model.nbody - 1          # fall back to the last (distal) body
    lo = model.actuator_ctrlrange[:, 0].copy(); hi = model.actuator_ctrlrange[:, 1].copy()
    mid = np.where(hi > lo, 0.5 * (lo + hi), 0.0); amp = np.where(hi > lo, 0.5 * (hi - lo), 0.6)
    pts = []
    for t in range(steps):
        data.ctrl[:] = mid + amp * np.sin(2 * np.pi * (t / 180.0) + np.linspace(0, 3.0, model.nu))
        mujoco.mj_step(model, data)
        if t % 5 == 0:
            pts.append(np.array(data.xpos[ee]))
    pts = np.array(pts)
    span = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 0.0
    # TRUE kinematic reach = sum of the actuated link lengths (how far the arm extends), not the small sine-sweep
    # bbox diagonal (which read a misleading ~0.07 m). The sweep now only proves the joints MOVE (liveness).
    reach_m = round(sum(float(getattr(s, "length_m", 0.0) or 0.0) for s in gene.actuated_joints()), 3)
    moves = span > 0.02
    if not moves:
        verdict = "STUCK (end-effector barely moves)"
    else:
        verdict = f"ARTICULATES (reach {reach_m:.2f} m)"
    return {"kind": "manipulator", "verdict": verdict, "survived": True, "reach_m": reach_m,
            "reach_span_m": round(span, 3), "n_actuators": int(model.nu),
            "note": "reach = kinematic link length; grasp/pick-place capability via evaluate_held / run_task"}


def _honest_grasp(gene) -> dict | None:
    """Run the REAL friction grasp-and-lift an arm implies and return a verdict (PICKS UP / GRASP WEAK), or None if
    this body can't grasp (no gripper) — so verify_robot shows a manipulator's true pick-up capability, not just reach."""
    try:
        from virturoid.services.grasp_eval import evaluate_grasp_lift
        r = evaluate_grasp_lift(gene)
        sr = float(r.get("success_rate", 0.0))
    except Exception:  # noqa: BLE001 - not a grippered arm / grasp path unavailable -> reach verdict stands alone
        return None
    verdict = "PICKS UP (real friction grasp + lift)" if sr >= 0.5 else f"GRASP WEAK (success {sr:.0%})"
    return {"verdict": verdict, "success_rate": round(sr, 3), "detail": r}


def _render_gait_gif(gene, qpos_frames, tag: str) -> str | None:
    import os
    os.environ.setdefault("MUJOCO_GL", "glfw")
    try:
        import mujoco
        import PIL.Image

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        _RENDER_DIR.mkdir(parents=True, exist_ok=True)
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        d = mujoco.MjData(m); frames = []
        for qp in qpos_frames[::max(1, len(qpos_frames) // 40)]:
            d.qpos[:] = qp; mujoco.mj_forward(m, d)
            rr = mujoco.Renderer(m, height=360, width=480); cam = mujoco.MjvCamera()
            cam.lookat[:] = [float(qp[0]), float(qp[1]), 0.15]; cam.distance, cam.azimuth, cam.elevation = 1.9, 125, -12
            rr.update_scene(d, camera=cam); frames.append(PIL.Image.fromarray(rr.render().copy())); rr.close()
        path = _RENDER_DIR / f"{tag}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=70, loop=0)
        return str(path)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ robot tools
def create_robot(args: dict) -> dict:
    """Compose a robot from a prompt, ground it, and HOLD it under a new robot_id for editing. Returns the id +
    a compact summary + a render. This is the entry point for a conversational session."""
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.task_matched_eval import robot_kind
    prompt = args["prompt"]
    # Walkable-by-default: a legged body gets a wide walkable stance so it ACTUALLY walks (fanned quad walks
    # ~0.65 m with auto-tune vs ~0.09 m un-fanned). ensure_walkable_quad is a no-op for non-quad morphologies,
    # so manipulators/rovers/etc. are byte-identical.
    gene = compose_robot(prompt, ensure_walkable=bool(args.get("ensure_walkable", True)))
    # Walk-tune the gait per body: the crawl defaults are quad-tuned, so a hexapod/octopod only creeps (~0.2 m)
    # out of the box. tune_crawl_gait finds this body's credible op-point and caches it on the gene (the quad
    # short-circuits on its first, already-credible rollout, so it stays cheap + byte-identical).
    if robot_kind(gene) == "legged" and args.get("tune_gait", True):
        # NB (flywheel_breakthrough_plan §3.M / §5d): in-place stance_repair was TRIED here and REVERTED — measured
        # 0/5 product-path walk-rate lift (composer already fans offline; the dominant failure is fore-aft LURCH,
        # not lateral roll-over, which lateral splay cannot fix). stance_repair.py is kept for the factory
        # verify-build (default-gait gate), not the hot path.
        try:
            from virturoid.services.morph_policy import tune_crawl_gait
            tune_crawl_gait(gene)
        except Exception:  # noqa: BLE001 - a tune failure must never block the build; defaults still apply
            pass
    rid = S.put_robot(gene, prompt=prompt)
    out = {"ok": True, **_summary(gene, rid), "prompt": prompt}
    try:                                                       # WS-G: the robotics AI grounds a novel concept in
        from virturoid.services.agent_tools import safe_build_path   # the nearest VERIFIED concepts (advisory
        from virturoid.services.concept_grounding import ground_concept   # similarity, never a silent route)
        cg = ground_concept(safe_build_path(None, "memory"), prompt, query_gene=gene)
        if not cg.get("routed") and cg.get("grounding"):
            out["concept_grounding"] = cg
    except Exception:  # noqa: BLE001 - grounding is value-add; never blocks a build
        pass
    img = _render_gene(gene, rid)
    if img:
        out["artifacts"] = [img]
    return out


def get_robot(args: dict) -> dict:
    from virturoid.services import session_state as S
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'; call create_robot first"}
    return {"ok": True, **_summary(gene, args["robot_id"]), **S.robot_meta(args["robot_id"])}


def edit_robot(args: dict) -> dict:
    """Apply typed LOCALIZED edit ops to a held robot (e.g. taller = scale_group legs length 1.2). Lands as ONE
    undo step; returns the DIFF + new summary. ``ops`` is a list of ``{op, args}`` (discover them with the
    ``op:"list"`` request). Special ops (no other args needed): ``{op:"undo"}`` reverts the last edit,
    ``{op:"list"}`` returns the operator catalog."""
    from virturoid.services import edit_operators as EO
    from virturoid.services import session_state as S
    rid = args.get("robot_id")
    ops = args.get("ops")
    # single-verb form: edit_robot {robot_id, op:"undo"|"list"} — folds undo_robot + edit_ops into one tool (G-G)
    verb = args.get("op") or (ops[0].get("op") if ops and len(ops) == 1 and set(ops[0]) <= {"op"} else None)
    if verb == "list":
        return {"ok": True, "operators": EO.op_specs()}
    if verb == "undo":
        gene = S.undo_robot(rid)
        if gene is None:
            return {"ok": False, "error": "nothing to undo"}
        return {"ok": True, "diffs": [{"op": "undo"}], "summary": _summary(gene, rid),
                "structural": False, **S.robot_meta(rid)}
    gene = S.get_robot(rid)
    if gene is None:
        return {"ok": False, "error": f"no robot '{rid}'; call create_robot first"}
    if not ops:
        return {"ok": False, "error": "provide ops: [{op, args}]; discover them with {op:'list'}, undo with {op:'undo'}"}
    try:
        new_gene, diffs = EO.apply_ops(gene, ops)
    except EO.EditError as exc:
        return {"ok": False, "error": str(exc)}                # teaching error (how to fix), not a crash
    label = ",".join(d.get("op", "edit") for d in diffs)
    S.commit_robot(rid, new_gene, label=label)
    out = {"ok": True, "diffs": diffs, "summary": _summary(new_gene, rid),
           "structural": any(d.get("structural") for d in diffs), **S.robot_meta(rid)}
    img = _render_gene(new_gene, f"{rid}_{S.robot_meta(rid)['undo_depth']}")
    if img:
        out["artifacts"] = [img]
    return out


def undo_robot(args: dict) -> dict:
    from virturoid.services import session_state as S
    gene = S.undo_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": "nothing to undo"}
    return {"ok": True, "restored": _summary(gene, args["robot_id"]), **S.robot_meta(args["robot_id"])}


def edit_ops(_args: dict) -> dict:
    """Discover the typed edit operators + their args (so an agent can fill them)."""
    from virturoid.services.edit_operators import op_specs
    return {"ok": True, "operators": op_specs()}


def render_view(args: dict) -> dict:
    from virturoid.services import session_state as S
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'"}
    img = _render_gene(gene, f"{args['robot_id']}_view", azimuth=float(args.get("azimuth", 50.0)),
                       elevation=float(args.get("elevation", -16.0)))
    return {"ok": bool(img), "artifacts": [img] if img else [], "error": None if img else "render unavailable"}


def simulate_gait(args: dict) -> dict:
    """Run the general scripted gait on a held robot and return the honest verdict (+ optional GIF)."""
    from virturoid.services import session_state as S
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'"}
    res = _honest_gait(gene, steps=int(args.get("steps", 1200)), render=bool(args.get("render", False)),
                       tag=f"{args['robot_id']}_gait")
    return {"ok": True, **res}


def verify_robot(args: dict) -> dict:
    """The anti-hallucination gate as a tool: an honest motion verdict FOR THE BODY'S KIND (B1), so an agent
    never gets a bogus gait verdict on a rover or an arm. Dispatches on structural ``robot_kind``: legged ->
    gait (survived/cadence/forward), mobile -> DRIVE (real travel + upright), manipulator -> REACH (workspace
    span), spray/other -> honest structural read (no locomotion verdict). ``mode``: ``full`` (default; long +
    a GIF for legged) or ``quick`` (fast iterate check). Folds simulate_gait (G-G)."""
    from virturoid.services import session_state as S
    from virturoid.services.task_matched_eval import robot_kind
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'"}
    quick = str(args.get("mode", "full")).lower() == "quick"
    kind = robot_kind(gene)
    # B2: run the motion verdict IN a held scene (obstacles matter most for drive/reach). robot_and_scene share
    # one world; a scene the model can't compose falls back to bare floor with an honest note.
    world_xml, scene_note = None, None
    sid = args.get("scene_id")
    if sid:
        sc = S.get_scene(sid)
        if sc is None:
            scene_note = f"scene '{sid}' not found — ran on a bare floor"
        elif kind == "legged":
            scene_note = "legged gait verdict is obstacle-free; use run_task/evaluate_held for scene navigation"
        else:
            world_xml = _compose_scene_xml(gene, _scene_from_dict(sc))
            scene_note = (f"composed into scene '{sid}'" if world_xml
                          else f"scene '{sid}' had no obstacles or would not compose — bare floor")
    _prompt = (S.robot_meta(args["robot_id"]) or {}).get("prompt", "")
    _meta = getattr(gene, "metadata", None) or {}
    _aerial = bool(_meta.get("rotor_offsets")) or getattr(gene, "robot_class", "") == "aerial"
    # aquatic by STRUCTURE (kind) or by swim-intent prompt over a spine body (legged/mobile fallback)
    _aquatic = kind == "aquatic" or (_env_words(_prompt, _AQUATIC_WORDS) and kind in ("legged", "mobile"))
    try:
        if _aerial:
            # AERIAL tier: a quadcopter is FLOWN with real rotor thrust forces + a geometric flight controller
            # (aerial.py / _honest_fly) — real MuJoCo rigid-body dynamics + gravity, never a land proxy.
            res = _honest_fly(gene, steps=int(args.get("steps", 1600 if quick else 2200)))
            res["credible_walk"] = False
        elif _aquatic:
            # T7: a swim-intent body is SIMULATED IN WATER (real MuJoCo fluid), not given a land proxy.
            res = _honest_swim(gene, steps=int(args.get("steps", 1500 if quick else 2500)))
            res["credible_walk"] = False
        elif kind == "legged":
            # quick uses 800 steps (not 400) so cadence/support register — 400 read a credible walk as a SLIDE.
            steps = int(args.get("steps", 800 if quick else 1500))
            res = _honest_gait(gene, steps=steps, render=not quick, tag=f"{args['robot_id']}_verify")
            res["credible_walk"] = res["verdict"].startswith("CREDIBLE")
        elif kind == "mobile":
            res = _honest_drive(gene, steps=int(args.get("steps", 400 if quick else 800)), world_xml=world_xml)
            res["credible_walk"] = False                       # not a walker; "drives" is its success signal
        elif kind == "manipulator":
            res = _honest_reach(gene, steps=int(args.get("steps", 300 if quick else 500)), world_xml=world_xml)
            res["credible_walk"] = False
            if not quick:                                      # also run the REAL friction grasp (the capability
                _grasp = _honest_grasp(gene)                   # an arm actually implies) and fold it into the verdict
                if _grasp is not None:
                    res["grasp"] = _grasp
                    res["verdict"] = f"{res['verdict']}; {_grasp['verdict']}"
        else:                                                  # spray / unknown envelope: no locomotion verdict
            res = {"kind": kind, "verdict": f"{kind.upper()}: no locomotion verdict for this kind",
                   "credible_walk": False,
                   "note": "use evaluate_held for this body's task-matched capability score"}
        if scene_note:
            res["scene_id"] = sid
            res["scene_note"] = scene_note
        _flag_physics_envelope(res, (S.robot_meta(args["robot_id"]) or {}).get("prompt", ""), kind)
        # a camera-equipped robot's ONBOARD camera + CV is exercised here (not just the rangefinder): render its
        # own functional robot_cam (FOV + resolution from the real camera part) at a target and report what it SEES.
        if not quick:
            try:
                from virturoid.services.camera_perception import robot_sees_target
                cam = robot_sees_target(gene)
                if cam.get("has_camera"):
                    res["vision"] = {k: cam[k] for k in ("camera_part", "fovy_deg", "render_px", "sees",
                                                         "vision_trained", "perception") if k in cam}
            except Exception:  # noqa: BLE001 - vision is value-add; the motion verdict still stands
                pass
    except Exception as exc:  # noqa: BLE001 - an odd body must yield an honest error, not a crash
        res = {"kind": kind, "verdict": f"could not simulate ({type(exc).__name__})", "credible_walk": False,
               "error": str(exc)[:200]}
    res["mode"] = "quick" if quick else "full"
    return {"ok": True, **res}


# ------------------------------------------------------------------ scene tools
def create_scene(args: dict) -> dict:
    """Build a themed scene for a task and hold it under a scene_id. theme in scene_themes.THEMES."""
    from virturoid.services import scene_themes as T
    from virturoid.services import session_state as S
    theme = args.get("theme", "warehouse"); task = args.get("task", "navigation")
    sg = T.build_scene(task=task, theme=theme, seed=int(args.get("seed", 0)))
    issues = sg.validate()
    sid = S.put_scene(sg.to_dict(), task=task, theme=theme)
    return {"ok": True, "scene_id": sid, "theme": theme, "task": task, "n_objects": len(sg.objects),
            "valid": bool(issues.ok) if hasattr(issues, "ok") else True, "themes_available": T.theme_names()}


def edit_scene(args: dict) -> dict:
    """Re-theme a held scene ('house instead of warehouse') keeping the task/robot/layout. ops:
    [{op:'swap_theme', args:{theme:'house'}}]."""
    from virturoid.schemas.scenes import SceneGraph
    from virturoid.services import scene_themes as T
    from virturoid.services import session_state as S
    sid = args["scene_id"]
    d = S.get_scene(sid)
    if d is None:
        return {"ok": False, "error": f"no scene '{sid}'; call create_scene first"}
    ops = args.get("ops") or []
    sg = SceneGraph.from_dict(d) if hasattr(SceneGraph, "from_dict") else _scene_from_dict(d)
    theme = None
    for spec in ops:
        if spec.get("op") == "swap_theme":
            theme = (spec.get("args") or {}).get("theme")
            if theme not in T.THEMES:
                return {"ok": False, "error": f"unknown theme '{theme}'; known: {T.theme_names()}"}
            sg = T.apply_theme(sg, theme)
        else:
            return {"ok": False, "error": f"unknown scene op '{spec.get('op')}'; supported: swap_theme"}
    S.commit_scene(sid, sg.to_dict(), theme=theme)
    return {"ok": True, "scene_id": sid, "theme": theme, "n_objects": len(sg.objects), **S.scene_meta(sid)}


def _scene_from_dict(d: dict):
    """Reconstruct a SceneGraph from its dict (schema has no from_dict — build objects by hand)."""
    from virturoid.schemas.scenes import SceneGraph, SceneObject
    objs = [SceneObject(name=o["name"], object_type=o.get("object_type", "cube"), category=o.get("category"),
                        material=o.get("material"), size_xyz=tuple(o["size_xyz"]) if o.get("size_xyz") else None,
                        pose_xyz_rpy=tuple(o.get("pose_xyz_rpy") or (0, 0, 0, 0, 0, 0))) for o in d.get("objects", [])]
    return SceneGraph(id=d.get("id") or "scene", name=d.get("name", ""),
                      backend_targets=d.get("backend_targets", ["mujoco"]),
                      robot_spawn_xyz_rpy=tuple(d["robot_spawn_xyz_rpy"]) if d.get("robot_spawn_xyz_rpy") else None,
                      objects=objs, variation_parameters=d.get("variation_parameters", {}),
                      bounds=tuple(map(tuple, d["bounds"])) if d.get("bounds") else None)


# ------------------------------------------------------------------ job tools (long-running: handle + poll)
def start_training(args: dict) -> dict:
    """Kick off a REAL build+train job (minutes) and return a job_id immediately — poll with get_job. Uses the
    in-process job_registry (autonomous_build with train=True). The portable MCP long-job pattern."""
    from virturoid.services import job_registry as J
    prompt = args.get("prompt")
    if not prompt and args.get("robot_id"):
        from virturoid.services import session_state as S
        prompt = (S.robot_meta(args["robot_id"]) or {}).get("prompt")
    if not prompt:
        return {"ok": False, "error": "provide a prompt (or a robot_id created from one)"}
    job = J.create("autonomous_build", {"prompt": prompt, "train": bool(args.get("train", True)),
                                        "target": float(args.get("target", 0.8))}, Path(args.get("build_root") or "build/agent_builds"))
    return {"ok": True, "job_id": job.get("id"), "status": job.get("status"),
            "note": "poll get_job(job_id, since) for progress + result"}


def get_job(args: dict) -> dict:
    """Poll a job: status + new progress events since a cursor + result when finished."""
    from virturoid.services import job_registry as J
    jid = args["job_id"]
    pair = J.events_since(jid, int(args.get("since", 0)))
    if pair is None:
        return {"ok": False, "error": f"no job '{jid}'"}
    view, events = pair
    return {"ok": True, "status": view.get("status"), "events": [{"stage": e["stage"], "message": e["message"]}
            for e in events], "next_since": int(args.get("since", 0)) + len(events), "result": view.get("result")}


# name -> {description, parameters, handler, heavy} — merged into agent_tools.TOOLS
AI_NATIVE_TOOLS: dict[str, dict] = {
    "create_robot": {"description": "Compose a robot from a prompt and HOLD it under a robot_id for incremental "
                     "editing; returns id + summary + render. Start a design session here.", "heavy": True,
                     "handler": create_robot, "parameters": {"type": "object", "required": ["prompt"], "properties": {
                         "prompt": {"type": "string"}, "ensure_walkable": {"type": "boolean", "default": False}}}},
    "get_robot": {"description": "Compact summary of a held robot (class, discovered appendages, height, mass).",
                  "heavy": False, "handler": get_robot, "parameters": {"type": "object", "required": ["robot_id"],
                  "properties": {"robot_id": {"type": "string"}}}},
    "edit_ops": {"description": "Discover the typed LOCALIZED edit operators (scale_group/set_height/set_material/"
                 "set_leg_count) and their args.", "heavy": False, "handler": edit_ops,
                 "parameters": {"type": "object", "properties": {}}},
    "edit_robot": {"description": "Apply typed LOCALIZED edits to a held robot (e.g. taller = ops:[{op:'scale_group',"
                   "args:{group:'legs',dims:'length',factor:1.2}}]) — never regenerates. Lands as one undo step; "
                   "returns the diff. Also: op:'list' returns the operator catalog, op:'undo' reverts the last edit.",
                   "heavy": True, "handler": edit_robot, "parameters": {"type": "object", "required": ["robot_id"],
                   "properties": {"robot_id": {"type": "string"}, "ops": {"type": "array", "items": {"type": "object"},
                   "description": "list of {op, args} localized edits"},
                   "op": {"type": "string", "enum": ["undo", "list"], "description": "single-verb shortcut instead of ops"}}}},
    "undo_robot": {"description": "Undo the last edit on a held robot (one step).", "heavy": False,
                   "handler": undo_robot, "parameters": {"type": "object", "required": ["robot_id"],
                   "properties": {"robot_id": {"type": "string"}}}},
    "render_view": {"description": "Render a held robot to a PNG (an agent should SEE what it built). Returns a path.",
                    "heavy": True, "handler": render_view, "parameters": {"type": "object", "required": ["robot_id"],
                    "properties": {"robot_id": {"type": "string"}, "azimuth": {"type": "number"}, "elevation": {"type": "number"}}}},
    "simulate_gait": {"description": "Run the general scripted gait on a held robot; returns the HONEST verdict "
                      "(survived/cadence/support/upright/forward). Real MuJoCo.", "heavy": True, "handler": simulate_gait,
                      "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                          "robot_id": {"type": "string"}, "steps": {"type": "integer"}, "render": {"type": "boolean"}}}},
    "verify_robot": {"description": "The anti-hallucination gate: honest gait metrics + verdict + a GIF, so a walk "
                     "is never claimed without traces.", "heavy": True, "handler": verify_robot,
                     "parameters": {"type": "object", "required": ["robot_id"], "properties": {"robot_id": {"type": "string"}}}},
    "create_scene": {"description": "Build a THEMED scene (warehouse/house/kitchen/lab/yard) for a task and hold it.",
                     "heavy": False, "handler": create_scene, "parameters": {"type": "object", "properties": {
                         "task": {"type": "string", "default": "navigation"}, "theme": {"type": "string", "default": "warehouse"},
                         "seed": {"type": "integer", "default": 0}}}},
    "edit_scene": {"description": "Re-theme a held scene ('house instead of warehouse'): ops:[{op:'swap_theme',"
                   "args:{theme:'house'}}]. Keeps task/robot/layout.", "heavy": False, "handler": edit_scene,
                   "parameters": {"type": "object", "required": ["scene_id", "ops"], "properties": {
                       "scene_id": {"type": "string"}, "ops": {"type": "array", "items": {"type": "object"}}}}},
    "start_training": {"description": "Kick off a real build+train job (minutes); returns a job_id immediately. "
                       "Poll with get_job.", "heavy": False, "handler": start_training,
                       "parameters": {"type": "object", "properties": {"prompt": {"type": "string"},
                       "robot_id": {"type": "string"}, "train": {"type": "boolean", "default": True}}}},
    "get_job": {"description": "Poll a training/build job: status + new progress events + result.", "heavy": False,
                "handler": get_job, "parameters": {"type": "object", "required": ["job_id"], "properties": {
                    "job_id": {"type": "string"}, "since": {"type": "integer", "default": 0}}}},
}
