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
           "material": _dominant_material(gene), "end_effector": gene.end_effector_type}
    if robot_id:
        out["robot_id"] = robot_id
    return out


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
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
        rr = mujoco.Renderer(m, height=420, width=560); cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.15]; cam.distance, cam.azimuth, cam.elevation = 1.9, float(azimuth), float(elevation)
        rr.update_scene(d, camera=cam); img = PIL.Image.fromarray(rr.render().copy()); rr.close()
        path = _RENDER_DIR / f"{tag}.png"; img.save(path)
        return str(path)
    except Exception:  # noqa: BLE001 - rendering is value-add; the edit/verdict still stands
        return None


# T7-lite: swim/fly INTENT words. Virturoid has no fluid/aerial physics tier yet, so a body meant to swim or
# fly gets routed to a terrestrial body and a LAND verdict — honest to FLAG that the verdict is only a land
# proxy, never to silently present it as the real thing. The full tier (MuJoCo fluid + rotor/thruster
# actuators + swim-intent routing) is the larger T7.
_AQUATIC_WORDS = ("swim", "swimming", "aquatic", "underwater", "submarine", "eel", "fish", "shark", "whale",
                  "dolphin", "manatee", "narwhal", "seahorse", "jellyfish", "octopus", "squid", "stingray")
_AERIAL_WORDS = ("fly", "flying", "aerial", "drone", "quadcopter", "helicopter", "hover", "aircraft", "winged")


def _env_words(prompt: str, words) -> bool:
    """WORD-BOUNDARY match (not substring) so 'wh-EEL-ed'/'sp-RAY' don't false-trigger the fluid/aerial tier."""
    import re
    p = (prompt or "").lower()
    return any(re.search(rf"\b{re.escape(w)}\b", p) for w in words)


def _flag_physics_envelope(res: dict, prompt: str, kind: str) -> None:
    """If the prompt implies a swim/fly envelope we don't yet simulate, annotate the verdict honestly instead
    of letting a land-gait verdict masquerade as the real capability (T7-lite)."""
    if res.get("kind") == "aquatic":
        return                                                 # T7: already simulated in water — the swim verdict stands
    env = "aquatic" if _env_words(prompt, _AQUATIC_WORDS) else ("aerial" if _env_words(prompt, _AERIAL_WORDS) else None)
    if env and kind in ("legged", "mobile"):
        res["physics_envelope"] = env
        res["credible_walk"] = False
        res["envelope_note"] = (
            f"this prompt implies a {env.upper()} body; Virturoid simulates AQUATIC bodies in water (T7) but has "
            f"no AERIAL tier yet, so this {env} verdict is a LAND-BASED PROXY — treat it as unsupported-envelope." if env == "aerial"
            else f"this prompt implies an aquatic body; it was simulated in water (see swim_m).")


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
    p0 = np.array(d.xpos[bid]).copy()
    for t in range(steps):
        ph = 2 * np.pi * t * m.opt.timestep * 1.3
        for k in range(m.nu):                                   # PD-track a travelling wave head->tail
            tgt = 0.7 * np.sin(ph - k * 1.1)
            d.ctrl[k] = float(np.clip(8.0 * (tgt - d.qpos[qadr[k]]) - 0.3 * d.qvel[vadr[k]], -frc[k], frc[k]))
        mujoco.mj_step(m, d)
    swim = float(np.hypot(*(np.array(d.xpos[bid])[:2] - p0[:2])))
    verdict = (f"SWIMS ({swim:.2f} m undulatory thrust)" if swim > 0.15
               else f"DOES NOT SWIM ({swim:.2f} m — this body's geometry yields little thrust)")
    return {"kind": "aquatic", "verdict": verdict, "survived": True, "swim_m": round(swim, 3),
            "n_actuators": int(m.nu),
            "note": "REAL MuJoCo fluid sim (water + neutral buoyancy); thrust is body/geometry-dependent"}


def _honest_gait(gene, *, steps: int = 1200, render: bool = False, tag: str = "gait") -> dict:
    """Run the general scripted gait and return the ANTI-GOODHART verdict (survived+cadence+support+upright+
    forward, forward == actual displacement) — the honesty gate as a tool result, never a raw qpos dump."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    from scripts.verify_gait import classify, orientation_summary
    # FLYWHEEL: use the best banked LEARNED gait for this body's morphology if one exists (recalled by embedding),
    # so the product's legged robots walk with learned control that compounds over builds — else the shipped default.
    gait_params: dict = {}
    gait_source = "default_crawl"
    try:
        from virturoid.services.gait_flywheel import recall_gait
        from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
        if DEFAULT_DB_PATH.exists():
            with MemoryDB(DEFAULT_DB_PATH) as _db:
                _p = recall_gait(_db, gene)
            if _p:
                gait_params = {k: float(_p[k]) for k in ("freq", "hip_amp", "knee_amp", "duty", "kp", "kd")
                               if k in _p}
                gait_source = "learned_flywheel"
    except Exception:  # noqa: BLE001 - the flywheel is an accelerant; a miss just uses the default gait
        gait_params = {}
    r = crawl_gait_rollout(gene, steps=steps, record_qpos=True, **gait_params)
    # DEPLOY-SELECT safety net: a recalled gait must never make THIS body walk worse than the shipped default
    # (gene-construction paths differ slightly, so a banked gait may not fit every body). If it underperforms,
    # re-run the default on this exact gene and keep whichever actually walks further — the flywheel only ever helps.
    if gait_params and (not r.get("survived") or abs(float(r.get("forward", 0))) < 0.30):
        r_def = crawl_gait_rollout(gene, steps=steps, record_qpos=True)
        if abs(float(r_def.get("forward", 0))) >= abs(float(r.get("forward", 0))):
            r, gait_source = r_def, "default_crawl"
    o = orientation_summary(r.get("qpos_frames") or [])
    out = {"kind": "legged", "verdict": classify(r), "survived": bool(r.get("survived")),
           "gait_source": gait_source,
           "forward_m": round(float(r.get("forward", 0)), 3),
           "speed_mps": round(float(r.get("speed", 0)), 3), "cadence": round(float(r.get("cadence", 0)), 1),
           "support_frac": round(float(r.get("support_frac", 0)), 2), "height_ratio": r.get("height_ratio"),
           "roll_max_deg": o.get("roll_max"), "pitch_max_deg": o.get("pitch_max")}
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
    verdict = f"ARTICULATES (reach span {span:.2f} m)" if span > 0.05 else "STUCK (end-effector barely moves)"
    return {"kind": "manipulator", "verdict": verdict, "survived": True, "reach_span_m": round(span, 3),
            "n_actuators": int(model.nu), "note": "task capability (grasp / pick-place) via evaluate_held"}


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
    prompt = args["prompt"]
    gene = compose_robot(prompt, ensure_walkable=bool(args.get("ensure_walkable", False)))
    rid = S.put_robot(gene, prompt=prompt)
    out = {"ok": True, **_summary(gene, rid), "prompt": prompt}
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
    _aquatic = _env_words(_prompt, _AQUATIC_WORDS)
    try:
        if _aquatic and kind in ("legged", "mobile"):
            # T7: a swim-intent body is SIMULATED IN WATER (real MuJoCo fluid), not given a land proxy.
            res = _honest_swim(gene, steps=int(args.get("steps", 1500 if quick else 2500)))
            res["credible_walk"] = False
        elif kind == "legged":
            steps = int(args.get("steps", 400 if quick else 1500))
            res = _honest_gait(gene, steps=steps, render=not quick, tag=f"{args['robot_id']}_verify")
            res["credible_walk"] = res["verdict"].startswith("CREDIBLE")
        elif kind == "mobile":
            res = _honest_drive(gene, steps=int(args.get("steps", 400 if quick else 800)), world_xml=world_xml)
            res["credible_walk"] = False                       # not a walker; "drives" is its success signal
        elif kind == "manipulator":
            res = _honest_reach(gene, steps=int(args.get("steps", 300 if quick else 500)), world_xml=world_xml)
            res["credible_walk"] = False
        else:                                                  # spray / unknown envelope: no locomotion verdict
            res = {"kind": kind, "verdict": f"{kind.upper()}: no locomotion verdict for this kind",
                   "credible_walk": False,
                   "note": "use evaluate_held for this body's task-matched capability score"}
        if scene_note:
            res["scene_id"] = sid
            res["scene_note"] = scene_note
        _flag_physics_envelope(res, (S.robot_meta(args["robot_id"]) or {}).get("prompt", ""), kind)
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
