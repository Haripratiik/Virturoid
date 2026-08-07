"""AI-native STATEFUL tools (docs/ai_native_plan.md P1) — the tool surface for INCREMENTAL, session-held
work that both frontends (the MCP server + the in-app assistant) share. Unlike the stateless
``agent_tools`` (each composes a robot fresh from a prompt), these operate on a robot/scene HELD under an id
in ``session_state``, so an agent can create once and then EDIT in place ("make it taller"), simulate, verify,
and undo. Every tool returns the compact verdict contract ``{ok, ...compact fields..., artifacts?}`` — never a
raw sim/gene dump (SWE-agent ACI). Registered into ``agent_tools.TOOLS`` so the MCP server exposes them too.
"""
from __future__ import annotations

import re
from pathlib import Path

_RENDER_DIR = Path("build/agent_renders")


def _render_dir() -> Path:
    """The render directory as an ABSOLUTE path, resolved at CALL time.

    ``_RENDER_DIR`` is relative, so ``str(dir / name)`` handed back to a caller was a CWD-relative string like
    ``build\\agent_renders\\robot_x_view.png``. An MCP host launched from a different working directory than the
    one the server process happens to sit in cannot open that -- the picture existed on disk and the path to it
    did not resolve. Resolving here (rather than freezing an absolute constant at import) keeps the existing
    behaviour that a test or a run under a different CWD writes under THAT CWD's ``build/``; only the string we
    report changes, from relative to absolute. Matches what ``safe_build_path``/``export_held`` already return.
    """
    return (Path.cwd() / _RENDER_DIR).resolve()


# ------------------------------------------------------------------ helpers
def _summary(gene, robot_id: str | None = None, prompt: str = "") -> dict:
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
    out["composition_notes"] += _requested_vs_built_notes(prompt, out)
    if robot_id:
        out["robot_id"] = robot_id
    return out


# Words-per-unit multipliers the summary can compare against. Kept tiny on purpose: only EXPLICIT NUMBERS in
# the prompt are checked (leg counts, metres of height) — fuzzy adjectives are the composer's job, not this.
_REQ_LEGS = re.compile(r"\b(\d{1,4})\s*(?:-|\s)?legs?\b|\b(\d{1,4})\s*(?:-|\s)?legged\b")
_REQ_HEIGHT_M = re.compile(r"\b(\d+(?:\.\d+)?)[\s-]*(?:m|meter|metre)s?\b(?:\s*tall)?", re.IGNORECASE)


def _requested_vs_built_notes(prompt: str, summary: dict) -> list[str]:
    """SURFACE, never hide, a wildly-coerced explicit request (final-drive red-team finding 2026-07-22).

    Measured: "a 500-meter tall robot with 1000 legs" silently built a 0.16 m, 16-leg arachnid — the response
    carried a generic archetype note and NOTHING about either coercion, which is precisely the silent
    substitution the honesty architecture forbids. The build itself is allowed to clamp (a 500 m robot is not
    simulable); staying QUIET about it is not. Additive + best-effort: never raises, never blocks a build.
    """
    notes: list[str] = []
    try:
        p = (prompt or "").lower()
        m = _REQ_LEGS.search(p)
        if m:
            asked = int(next(g for g in m.groups() if g))
            built = (summary.get("appendages") or {}).get("legs") if isinstance(summary.get("appendages"), dict) else None
            if isinstance(built, int) and asked > 0 and built > 0 and not (0.5 <= built / asked <= 2.0):
                notes.append(f"Requested {asked} legs; built {built} (structural/simulation cap). "
                             "The count was coerced, not silently honored.")
        # Metres are only a HEIGHT claim when the prompt says so ("500-meter tall") — "0.9 m reach" or
        # "2 m long" are different dimensions with their own checks, and an arm's standing_height is a base-disc
        # cosmetic value that would false-positive against them (caught by the guard probe on first try).
        m = _REQ_HEIGHT_M.search(p) if re.search(r"\btall\b|\bheight\b|\bhigh\b", p) else None
        if m:
            asked_h = float(m.group(1))
            built_h = summary.get("standing_height_m")
            if isinstance(built_h, (int, float)) and asked_h > 0 and built_h and not (0.5 <= built_h / asked_h <= 2.0):
                notes.append(f"Requested ~{asked_h:g} m tall; built {built_h:.3g} m (dimension priors clamp "
                             "unsimulable scales). The size was coerced, not silently honored.")
    except Exception:  # noqa: BLE001 - the note is additive; a parse failure must not sink the build
        return notes
    return notes


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


def _select_mujoco_gl() -> str:
    """Choose a MuJoCo GL backend that can actually render HERE, and record the choice in MUJOCO_GL.

    This used to hardcode ``setdefault("glfw")``. glfw needs a display, so on a HEADLESS Linux box -- exactly
    where a robotics engineer evaluates this, and where the GPU trainer runs -- every render silently failed and
    the gallery/verdict images just went missing. An explicit MUJOCO_GL always wins (that contract is unchanged);
    otherwise prefer glfw when a display exists, else EGL (GPU, headless), else OSMesa (software).
    """
    import os
    import sys
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]                       # caller/CI knows best -- never override
    if sys.platform.startswith(("win", "darwin")) or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        backend = "glfw"
    else:
        backend = "egl" if os.path.exists("/dev/dri") or os.environ.get("NVIDIA_VISIBLE_DEVICES") else "osmesa"
    os.environ["MUJOCO_GL"] = backend
    return backend


def _render_gene(gene, tag: str, *, azimuth: float = 50.0, elevation: float = -16.0,
                 collision: bool = False) -> str | None:
    """The path to a rendered PNG, or ``None``. Kept for callers that treat a render as value-add; anything that
    must EXPLAIN a missing picture should call :func:`_render_gene_detail` and report the reason."""
    return _render_gene_detail(gene, tag, azimuth=azimuth, elevation=elevation, collision=collision)[0]


def _render_gene_detail(gene, tag: str, *, azimuth: float = 50.0, elevation: float = -16.0,
                        collision: bool = False) -> tuple[str | None, str | None]:
    """Render the gene and return ``(absolute_png_path, failure_reason)`` -- exactly one of the two is set.

    This used to be one function that returned ``str | None`` and swallowed every exception, so a failed render
    was indistinguishable from a render nobody asked for: the caller got ``None`` and had nothing to tell the
    engineer. The reason is now carried out, and the path is ABSOLUTE and verified to exist on disk before it is
    handed back -- a render tool must never report a path that does not open.
    """
    import os
    backend = _select_mujoco_gl()
    try:
        import mujoco
        import PIL.Image

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z
        out_dir = _render_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        # SHOW THE BODY THAT IS ACTUALLY SIMULATED when asked. ``collision=True`` builds the exact model the
        # gait/verdict path runs (compile_gene_to_mjcf, no visual mesh layer) at the exact spawn height it uses,
        # so "what you see" and "what gets verified" can be put side by side instead of taken on trust.
        spawn_z = standing_spawn_z(gene, meshed=not collision)
        # Render the MESHED model (the true geometry the app viewport shows), NOT the crude box collider — the
        # non-meshed render drew a chassis as its tiny bounding box, so a wheeled body read as a small box with
        # oversized disconnected wheels. Fall back to the primitive model only if meshing fails.
        if collision:
            xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn_z)
        else:
            try:
                xml = gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=spawn_z)
            except Exception:  # noqa: BLE001
                xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn_z)
        m = mujoco.MjModel.from_xml_string(xml)
        scene_option = mujoco.MjvOption()
        drawn = None
        if collision:
            # Hide every geom that does NOT collide (mass=0 contype=0 cosmetics: motor cans, collars, hubs,
            # fairings) so what is left on screen is exactly the set of bodies the verdict is computed from.
            drawn = []
            for _i in range(m.ngeom):
                if int(m.geom_bodyid[_i]) == 0:
                    continue                                    # keep the floor
                if int(m.geom_contype[_i]) == 0 and int(m.geom_conaffinity[_i]) == 0:
                    m.geom_rgba[_i] = (0.0, 0.0, 0.0, 0.0)
                else:
                    m.geom_rgba[_i] = (0.95, 0.45, 0.15, 0.95)   # the colliders, unmistakably
                    m.geom_matid[_i] = -1                        # a material's rgba would win over the colour above
                    drawn.append(_i)
            # ...and TURN THAT GROUP ON. MjvOption defaults to geomgroup [1,1,1,0,0,0], and an imported robot
            # parks its collision geoms in group 3 by the Menagerie convention (visual meshes in 0/2). So on
            # every imported model -- Go2, G1, Spot, ANYmal -- the collision view drew an empty floor: 13 of 13
            # colliders were in a group the renderer had switched off, and the 13 cosmetics that WERE in a
            # visible group had just been set to alpha 0 by the loop above. The tool that exists to show "the
            # exact bodies the verdict is computed on" showed nothing, and reported a path to the picture of it.
            scene_option.geomgroup[:] = 1
        d = mujoco.MjData(m)
        # SHOW THE POSE THE BODY SHIPS IN. mj_resetData goes to qpos0 — every joint at zero — so a design that
        # DECLARED where it rests was rendered in a stance it never holds. Measured: a SCARA and a rail rendered
        # byte-identically with and without their rest angles, and an excavator that stands 1.248 m tall in
        # simulation was drawn lying flat on the ground. The sim paths (standing_spawn_z, _honest_drive) already
        # reset to the keyframe for exactly this reason; the render was the one place left behind, which is the
        # one place a customer actually looks.
        _declared = bool((getattr(gene, "metadata", None) or {}).get("rest_pose"))
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        else:
            mujoco.mj_resetData(m, d)
        if not _declared:
            # The canned arm bend is a fallback for bodies that never said how they rest. A declared stance is
            # the designer's own answer and must not be overwritten by our guess at one.
            _pose_manipulator_for_render(m, d, gene)
        mujoco.mj_forward(m, d)
        rr = mujoco.Renderer(m, height=420, width=560); cam = mujoco.MjvCamera()
        # FRAME THE ACTUAL BODY. A fixed lookat z=0.15 / distance=1.9 is sized for a quadruped, so a tall body
        # (the humanoid) rendered with its HEAD CUT OFF in the gallery. Fit to the body's bounding box instead,
        # excluding world geoms (the floor plane is effectively infinite and would blow the box up). The 3.2x
        # factor is chosen so a typical quad still frames at ~1.9 -- existing renders are preserved, tall bodies
        # simply stop being cropped.
        # Frame what will actually be DRAWN: in the collision view the cosmetics are alpha-0, so including them
        # in the box zooms the camera out around bodies nobody can see.
        _bg = drawn if drawn else [i for i in range(m.ngeom) if int(m.geom_bodyid[i]) != 0]
        if _bg:
            import numpy as _np
            _p = _np.asarray(d.geom_xpos)[_bg]
            _lo, _hi = _p.min(axis=0), _p.max(axis=0)
            _c = (_lo + _hi) / 2.0
            _extent = float(_np.linalg.norm(_hi - _lo)) or 0.6
            cam.lookat[:] = [float(_c[0]), float(_c[1]), float(_c[2])]
            cam.distance = float(min(8.0, max(1.2, 2.4 * _extent)))   # fills the frame; verified uncropped on a humanoid
        else:
            cam.lookat[:] = [0.0, 0.0, 0.15]; cam.distance = 1.9
        cam.azimuth, cam.elevation = float(azimuth), float(elevation)
        rr.update_scene(d, camera=cam, scene_option=scene_option)
        frame = rr.render().copy(); img = PIL.Image.fromarray(frame); rr.close()
        # A UNIFORM frame is not a picture of a robot. On a headless box with the wrong GL backend MuJoCo hands
        # back an all-black buffer and every downstream step happily reports a path to it, so the one artefact a
        # customer actually looks at is a black rectangle with a green tick over it. Catch it here and say so;
        # the test is deliberately strict (every pixel identical) so a dark-but-real render is never rejected.
        if int(frame.max()) == int(frame.min()):
            return None, (f"the renderer produced a blank frame (every pixel identical, MUJOCO_GL={backend!r}) — "
                          f"the GL backend cannot draw here; set MUJOCO_GL=egl (GPU, headless) or osmesa "
                          f"(software) and retry")
        path = out_dir / f"{tag}.png"; img.save(path)
        if not path.is_file() or path.stat().st_size == 0:
            return None, f"the render was written to {path} but the file is missing or empty"
        return str(path), None
    except Exception as exc:  # noqa: BLE001 - rendering is value-add; the edit/verdict still stands
        return None, f"{type(exc).__name__}: {exc}"


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


def _fit_margin_for(gene, params: dict) -> dict | None:
    """The robustness margin ALREADY MEASURED for exactly ``params`` on this body, or ``None``.

    THE ONLY FREE MARGIN ON THE VERIFY PATH. ``gait_flywheel.fit_gait_for_body`` runs before verify on a body the
    shipped default cannot walk; when it adopts an operating point it measures that point's fragility margin and
    stashes it on ``metadata['gait_fit']``. Verify then deploys exactly those parameters, so the margin already in
    hand describes exactly the row about to be banked — no extra rollout buys it.

    IDENTITY IS EXACT, not approximate. A banked row inherits a margin only if every fitted parameter is bit-equal
    to the one being deployed: measured on the grounded authored cat, a 2.4e-5 relative change in ``freq`` flips
    +0.958 m CREDIBLE WALK to +0.500 m FELL, so "close enough" would attach an error bar to a different
    controller. A body that walks on the shipped default never searched and has no margin at all — that is the
    common case, and it banks explicitly ungated rather than borrowing one.
    """
    from virturoid.services.gait_flywheel import _FIT_PARAMS
    fit = (getattr(gene, "metadata", None) or {}).get("gait_fit") or {}
    fitted = fit.get("params") if isinstance(fit.get("params"), dict) else None
    if not (fit.get("adopted") and fitted and "robustness_rel" in fit):
        return None
    for k in _FIT_PARAMS:
        if float(fitted.get(k, float("nan"))) != float(params.get(k, float("nan"))):
            return None                                   # a different controller: its error bar is not this one's
    return {"robustness_rel": fit.get("robustness_rel"), "probes": fit.get("robustness_probes") or {},
            "steps": fit.get("horizon_steps"), "per_param_rel": fit.get("robustness_per_param")}


#: Why the ordinary build path banks without an error bar, recorded ON the row (``bank_gate_reason``) so a later
#: mining run can exclude it by fact rather than by inference. Measuring here is not a rounding error: the
#: fragility ladder is 4-12 rollouts at the settling horizon (~10-60 s), on EVERY verify_robot call.
_VERIFY_UNGATED_REASON = ("ordinary verify path: no fragility margin was measured for these parameters and none "
                          "was inherited from a per-body fit; measuring one here would put 4-12 settling-horizon "
                          "rollouts on every build")


def _auto_bank_gait(gene, r, base_params) -> str | None:
    """FLYWHEEL SELF-UPDATE: bank the working gait a CREDIBLE walk just demonstrated, so the flywheel COMPOUNDS on
    ordinary build+verify — not only under explicit training or a manual GPU night run (the reason the bank sat
    empty: nothing on the ordinary path ever wrote to it). bank_gait is keep-best + keyed by morphology, so
    re-verifying the same body updates ONE skill (a stronger later gait replaces a weaker one), never floods.

    THE PROVENANCE IS PART OF THE WRITE. This is the widest door into the bank and it cannot afford to measure a
    fragility margin (see ``_VERIFY_UNGATED_REASON``), so it passes the one the per-body fit already paid for when
    there is one and declares itself ungated when there is not. What it must never do is bank silently: a row with
    no stamp is indistinguishable from a row that predates the gate entirely.

    AND WHEN THAT INHERITED MARGIN SAYS FRAGILE, NOTHING IS BANKED. Measured 2026-08-07 on the composed
    ``a robot dog``: the fitter searched, adopted freq 2.6189 / kp 59.57, measured it 0/4 at every rung of the
    ladder, and ``learn_gait_flywheel`` correctly refused it — the DB held zero rows. Verify then deployed the
    same operating point, got CREDIBLE WALK at its 800-step horizon, and banked it. So the one op-point the
    product had explicitly measured as one lucky float was the one the flywheel would serve as a warm start, and
    ``fit_gait_for_body``'s own disclosure ("it is NOT banked for reuse", ``gait_flywheel.py:954``) and verify's
    own ``robustness_note`` ("the gait is not banked for reuse") were both false about it. Declining here costs
    zero rollouts, matches what every other gate-aware caller already does, and makes those two sentences true.
    """
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT, _DeployResult, bank_gait
    from virturoid.services.gait_quality import classify
    from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    params = {**_DEFAULT_GAIT, **{k: float(v) for k, v in (base_params or {}).items()}}
    rob = _fit_margin_for(gene, params)
    if rob is not None and rob.get("robustness_rel") is None:
        return None                       # measured, and it is one lucky float — the bank holds controllers
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
        return bank_gait(db, gene, holder, door="verify_robot",
                         robustness=rob, ungated_reason=(None if rob else _VERIFY_UNGATED_REASON),
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


def _record_gait_hint_outcome(gene, hint_rollout: dict, default_rollout: dict, *, selected_default: bool,
                              source: str) -> None:
    """Record every attempted recall deployment, including wins, losses and ties.

    Lessons remain useful diagnosis for losses; this provenance event is the unbiased denominator needed to tell
    whether reuse helps at all. The measured delta is like-for-like absolute forward travel at one horizon.

    ``delta`` IS THE COUNTERFACTUAL AND HAS ALWAYS BEEN ONE: it is (recalled - default), i.e. what WOULD have
    happened had the recalled gait been deployed blind. It is NOT what the robot shipped with, because the
    deploy-select above keeps the winner. Those two numbers have opposite signs on the live bank and the gap is
    not small -- the flywheel-hint arm counterfactuals to -0.1288 m and SHIPS +0.0949 m -- so a reader who takes
    ``delta`` for an outcome concludes the product makes robots worse when measurably it does not. That reading
    cost a day. ``shipped_delta`` is therefore recorded next to it and the counterfactual is kept, because the
    only reason we can see any of this is that someone recorded both arms.

    THE KIND NAMES THE MECHANISM. ``flywheel_hint`` (a mined cross-body region) and ``tuned_for_this_body``
    (this body's OWN fitted op-point) are different claims and were being pooled under ``gait_hint_deploy``,
    where 1598 rows of the latter diluted the former's mean by roughly 4x. They are banked apart now; both are
    still recorded, so no series is lost, only separated.
    """
    try:
        from virturoid.services.gait_flywheel import structural_gait_key
        from virturoid.services.gait_quality import classify
        from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        hint_forward = abs(float(hint_rollout.get("forward", 0.0)))
        default_forward = abs(float(default_rollout.get("forward", 0.0)))
        shipped_forward = default_forward if selected_default else hint_forward
        structure_key = structural_gait_key(gene)
        kind = "gait_hint_deploy" if source == "flywheel_hint" else "gait_own_point_deploy"
        with MemoryDB(DEFAULT_DB_PATH) as db:
            RoboticsVectorMemory(db).record_provenance(
                "gene", getattr(gene, "id", "") or structure_key,
                parent_type="gait_hint_region", parent_id=structure_key, kind=kind,
                delta=round(hint_forward - default_forward, 6),
                meta={"source": source, "selected": "default" if selected_default else "hint",
                      "hint_forward_m": hint_forward, "default_forward_m": default_forward,
                      "hint_credible": classify(hint_rollout).startswith("CREDIBLE"),
                      "default_credible": classify(default_rollout).startswith("CREDIBLE"),
                      # what the robot ACTUALLY walked away with, vs the same body's shipped default
                      "shipped_forward_m": shipped_forward,
                      "shipped_delta": round(shipped_forward - default_forward, 6),
                      "delta_is_counterfactual": True},
            )
    except Exception:  # noqa: BLE001 - measurement must never break a deploy verdict
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

    from virturoid.services.body_kind import measured_legs
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    # the leg count comes from body_kind (the one structural counter), not a private appendage-map call: a
    # Booster T1 measured ZERO legs here, so the biped block declined and a humanoid was judged by the
    # multi-leg crawl gait that fells it -- the exact verdict this function exists to prevent.
    if measured_legs(gene) != 2 or m.nu == 0:
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
    forward+RATE-HOLD, forward == actual displacement) — the honesty gate as a tool result, never a raw qpos dump.

    ``steps`` IS PART OF THE CLAIM. At or above ``gait_quality._SETTLE_MIN_STEPS`` the verdict also asserts the
    body was STILL WALKING at the end; below it the verdict is only about the horizon it saw, and ``settled``
    says so in the result. The full (non-quick) legged verify runs the settling horizon precisely because that
    is where the claim reaches a customer — measured, the grounded authored cat read CREDIBLE WALK at 1500 and
    fell at step 2014 (task #267).

    The recorded TRACE is subsampled with the horizon (``frame_every`` scales so ~300 frames are kept whatever
    the horizon), so a 4x longer physics run does NOT make the replay GIF 4x longer or 4x more expensive to
    render — the extra cost is simulation only.
    """
    from virturoid.services.morph_policy import crawl_gait_rollout
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    from virturoid.services.gait_quality import classify, orientation_summary, settling
    frame_every = max(5, int(steps) // 300)
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
    # THIS BODY'S OWN TUNED OP-POINT FIRST. tune_crawl_gait fits (freq, hip, knee, kp, kd) to this gene, confirms
    # it on a second longer rollout, and caches it on metadata['gait_params'] -- but verify never read it, so a
    # body we had just tuned was still judged under the shipped default. Measured: the horse tuned to a confirmed
    # CREDIBLE WALK at 1.26 m and then verified as "LURCHES (pitch 31 / roll 44)", because the verdict came from
    # a gait fitted to a different body. Params measured ON THIS BODY beat both the generic default and a mined
    # cross-body hint region, and the deploy-select comparison below still guards it against the default.
    try:
        _own = (getattr(gene, "metadata", None) or {}).get("gait_params") or {}
        _own = {k: float(_own[k]) for k in ("freq", "hip_amp", "knee_amp", "kp", "kd") if k in _own}
        if _own:
            gait_params, gait_source = _own, "tuned_for_this_body"
    except Exception:  # noqa: BLE001 - a malformed cache must never block the verdict
        gait_params = {}
    try:
        if gait_params:
            raise LookupError("this body has its own measured op-point; no cross-body hint needed")
        # FLYWHEEL = HINTS, NOT COPY-PASTE. Rather than deploy one banked body's exact params verbatim (a trap on a
        # slightly-different body), start from the mined HINT REGION — where credible walks CLUSTER across bodies,
        # auto-derived from data (gait_hints). A quick verify uses this data-driven prior; the ``adapt_gait`` tool
        # runs the full per-body fit from the same hints. The deploy-select below still guards it vs the default.
        #
        # VIRTUROID_DISABLE_GAIT_HINTS=1 skips the recall entirely. Design-Bench sets it so the REGRESSION GATE
        # is hermetic: measured 2026-07-22, verdict@1 read 0.50 on an empty DB and 0.55 on a banked one — the
        # gate number floated with whatever the session had banked, which made CI flicker at the floor. The
        # PRODUCT keeps hints on; the BENCH measures the composer+compiler alone, deterministically.
        import os as _os
        from virturoid.services.gait_hints import mine_gait_hints
        from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
        if _os.environ.get("VIRTUROID_DISABLE_GAIT_HINTS") == "1":
            raise LookupError("gait hints disabled for hermetic benchmarking")
        if DEFAULT_DB_PATH.exists():
            with MemoryDB(DEFAULT_DB_PATH) as _db:
                _h = mine_gait_hints(_db, gene=gene)              # VECTOR-nearest robots seed the deploy hint
            if _h.get("n", 0) >= 2:                              # ≥2 banked walks -> a real mined region to hint from
                _p = _h["prior"]
                gait_params = {k: float(_p[k]) for k in ("freq", "hip_amp", "knee_amp", "kp", "kd")
                               if k in _p}
                gait_source = "flywheel_hint"                    # a data-driven hint region, not a copied policy
    except Exception:  # noqa: BLE001 - the flywheel is an accelerant; a miss just uses the default gait
        if gait_source != "tuned_for_this_body":                 # never discard THIS body's own measured op-point
            gait_params = {}
    r = crawl_gait_rollout(gene, steps=steps, record_qpos=True, frame_every=frame_every, **gait_params)
    # EXACTLY THE KWARGS THAT PRODUCED ``r``, tracked separately from the SELECTION variable above because the
    # deploy-select below can swap the rollout without swapping the parameters. It is what the flywheel banks:
    # a row whose ``gait_params`` did not produce its own ``forward_m`` is evidence about nothing.
    deployed_params = dict(gait_params)
    # DEPLOY-SELECT safety net: a recalled gait must never make THIS body walk worse than the shipped default
    # (gene-construction paths differ, so a banked gait may not fit every body). When a gait was recalled, ALSO
    # run the default and keep whichever is CREDIBLE (tie-break: further) — so a mismatched banked SLIDE can never
    # beat a credible default. The flywheel only ever helps.
    if gait_params:
        # THE DEFAULT ARM HAS TO ACTUALLY BE THE DEFAULT. ``crawl_gait_rollout`` falls back to
        # ``gene.metadata['gait_params']`` for every gait kwarg it is not handed (morph_policy.py:1285-1299), so
        # calling it bare on a body that carries a tuned cache RE-RUNS THE SAME GAIT. The safety net then
        # compared the tuned op-point against itself. Measured 2026-08-07 against the live bank: all 1598
        # ``tuned_for_this_body`` provenance rows carry delta EXACTLY 0.000 — one wasted full-length rollout
        # each, and the guard this block advertises ("never worse than the shipped default") could not fire even
        # once on those 11 bodies. Naming the constants makes the baseline real at no extra rollout cost, and
        # turns 1598 vacuous zeros into a real measurement of what the per-body tune is worth.
        r_def = crawl_gait_rollout(gene, steps=steps, record_qpos=True, frame_every=frame_every,
                                   **_DEFAULT_GAIT)
        cred_r = classify(r).startswith("CREDIBLE")
        cred_def = classify(r_def).startswith("CREDIBLE")
        better_def = (cred_def and not cred_r) or (
            cred_def == cred_r and abs(float(r_def.get("forward", 0))) > abs(float(r.get("forward", 0))))
        _record_gait_hint_outcome(gene, r, r_def, selected_default=better_def, source=gait_source)
        if better_def:
            # the banked hint underperformed the default ON THIS BODY -> a verified failure->fix lesson (the
            # locomotion path's only lesson source; feeds mine_gait_hints for the next similar body)
            _record_gait_lesson(gene, "banked_gait_underperformed", "deploy_default_crawl",
                                improvement=abs(float(r_def.get("forward", 0))) - abs(float(r.get("forward", 0))),
                                root_cause=f"recalled '{gait_source}' gait slid/underperformed the shipped default")
            r, gait_source = r_def, "default_crawl"
            deployed_params = {}       # the SHIPPED DEFAULT produced this rollout — bank ITS params, not the loser's
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
            # ...with the parameters that produced THIS rollout. It used to be `gait_params or metadata`, which
            # after a deploy-select swap banked the LOSING recalled parameters under the winning DEFAULT run's
            # distance and credibility — a row whose controller never produced its own evidence.
            banked = _auto_bank_gait(gene, r, deployed_params)
            if banked:
                out["flywheel_banked"] = banked
        except Exception:  # noqa: BLE001 - self-update is an accelerant; never let it break a verdict
            pass
    # SAY WHAT WAS ACTUALLY MEASURED, on every branch (scripted / learned policy / biped stand). A walk verdict
    # is a claim about a horizon, and until task #267 the horizon was invisible: the grounded authored cat's
    # "CREDIBLE WALK" was measured at 1500 steps and the body fell at 2014. `horizon_steps` says how long anyone
    # looked; `settled` says whether that was long enough to check the body was STILL walking at the end;
    # `travel_rate_m_per_1000` shows the profile a net displacement hides; `robustness_rel` is the error bar —
    # how much relative perturbation this operating point survives — read from the fit that measured it (never
    # re-measured here, which would put N extra rollouts on every verify).
    _s = settling(r)
    out["horizon_steps"] = int(steps)
    out["settled"] = _s is not None
    if _s is not None:
        out["travel_rate_m_per_1000"] = {int(k): round(v, 3) for k, v in _s["rates"].items()}
        out["holds_rate"] = bool(_s["holds_rate"])
    _fit = (getattr(gene, "metadata", None) or {}).get("gait_fit") or {}
    if out.get("gait_source") == "tuned_for_this_body" and "robustness_rel" in _fit:
        out["robustness_rel"] = _fit["robustness_rel"]
        out["robustness_probes"] = _fit.get("robustness_probes")
        # PER PARAMETER, because that is the half a reader can act on. The joint scalar says "how far can
        # everything move at once" and averages the axes together; measured (task #267) the grounded authored
        # cat is below 1e-5 on FOUR of five while the canonical template is >=1e-1 on all five, and the
        # difference between "keep the step frequency within 10%" and "one part in 10^5 of step frequency rolls
        # it over" is the whole deployability question. Read from the fit that measured it; never re-measured
        # here, which would put N extra rollouts on every verify.
        out["robustness_per_param"] = _fit.get("robustness_per_param")
        if _fit.get("robustness_note"):
            out["robustness_note"] = _fit["robustness_note"]
        # A NULL MARGIN MUST NOT READ AS "not measured". It was measured and the answer was "none" — no perturbed
        # copy of this operating point walks, so the verdict above describes one lucky float rather than a
        # controller. Measured on the grounded authored cat, that is the difference between a walk that survives
        # to step 12000 and one that is on the floor by 8754 with the verdict taken at 6000 (task #267).
        if _fit.get("fragile"):
            out["robustness_note"] = ("FRAGILE operating point: no perturbed copy of it survives even a "
                                      "0.001 relative change, so this verdict is not reproducible on hardware "
                                      "and the gait is not banked for reuse")
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


def _ensure_scene_materials(xml: str) -> str:
    """M10 (2026-07-24 audit): the authored-scene geoms reference SHARED materials (mat_gray/red/blue/metal/
    shell) that a ROBOT mjcf's <asset> may not define -- measured: robot_mjcf defines mat_metal/mat_shell but
    NOT mat_gray/red/blue, so every sorting/pick-place scene failed to compile and fell back to a BARE FLOOR
    with no obstacles. Inject any missing scene material into the composed model's <asset> so the authored
    scene actually renders + simulates (rgba matches mujoco_exporter's canonical scene palette)."""
    defs = {
        "mat_gray":  '<material name="mat_gray" rgba="0.5 0.5 0.5 1"/>',
        "mat_red":   '<material name="mat_red" rgba="0.8 0.1 0.1 1"/>',
        "mat_blue":  '<material name="mat_blue" rgba="0.1 0.2 0.8 1"/>',
        "mat_metal": '<material name="mat_metal" rgba="0.47 0.49 0.53 1"/>',
        "mat_shell": '<material name="mat_shell" rgba="0.20 0.42 0.72 1"/>',
    }
    missing = [d for name, d in defs.items() if f'name="{name}"' not in xml]
    if not missing:
        return xml
    inject = "\n    " + "\n    ".join(missing)
    if "<asset>" in xml:
        return xml.replace("<asset>", "<asset>" + inject, 1)
    m = re.search(r"<mujoco\b[^>]*>", xml)                     # no <asset> -> create one right after <mujoco ...>
    if m:
        return xml[:m.end()] + "\n  <asset>" + inject + "\n  </asset>" + xml[m.end():]
    return xml


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
    composed = _ensure_scene_materials(composed)              # M10: define mat_gray/red/blue so it doesn't bare-floor
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
    mujoco.mj_resetData(model, data)
    # DRIVE THE BODY IN THE POSE IT SHIPS. Resetting to qpos 0 discards the model's own rest keyframe, and for a
    # mobile MANIPULATOR that pose is the whole difference between a robot and a wheelie: measured on the bench's
    # own frozen gene, the arm chains 0.952 m straight up from a 0.32 x 0.38 m wheelbase, putting the COM at
    # 0.164 m against a 0.160 m half-wheelbase -- so it tips at about 1 g, and full wheel traction IS about 1 g.
    # Same body, arm stowed: up_min -0.131 (flipped past horizontal) -> +0.988 (upright), and it drives FURTHER,
    # 0.833 -> 1.010 m. This is why the design-bench hybrid family read 1.0 -> 0.0 (#246): the verdict was
    # judging every mobile manipulator with its arm held vertically, which no real one drives with.
    # Same principle as the imported rest pose: evaluate the robot in the stance its design specifies.
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    bid, _free = _base_body_id(model)
    if model.nu == 0:
        return {"kind": "mobile", "verdict": "NO ACTUATORS (cannot drive)", "survived": True, "forward_m": 0.0,
                "note": "task capability (navigation/goal-reach) via evaluate_held"}
    # identify wheel geoms (the cylinders), the floor plane, and each actuator's driven joint dof
    # WHEELS ARE NAMED, NOT GUESSED FROM SHAPE. "Every cylinder is a wheel" is wrong on any wheeled robot that
    # also has cylindrical structure: measured on the bench's mobile manipulator, shoulder/j1/j2/wrist all count
    # as wheels because `_mech_beam` renders arm links as cylinders. That corrupts three things at once -- the
    # mean wheel RADIUS (arm links 0.03 m averaged in with real 0.07 m wheels), the wheel-CONTACT count (an arm
    # link brushing the floor reads as a wheel on the ground), and the SPIN check. Prefer the gene's own wheel
    # names, which the composer and the importer both set; fall back to shape only when nothing is named.
    _wheel_names = {s.name for s in getattr(gene, "segments", []) or []
                    if "wheel" in (s.name or "").lower() or "caster" in (s.name or "").lower()
                    or (getattr(s, "role", "") or "").lower() == "wheel"}
    wheel_geoms = {g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
                   in _wheel_names} if _wheel_names else set()
    if not wheel_geoms:                                             # unnamed (e.g. a raw import): shape is all we have
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
    # Drive each joint with a GRAVITY-COMPENSATED PD to a swept angle target — the way any real manipulator runs
    # (and the way physics_evaluator drives it), NOT a raw torque sweep. Without the qfrc_bias term a torque-
    # actuated arm just sags under gravity, and an AMENDED arm that embodied real motor mass reads a dishonest
    # STUCK (#220) — the arm is fine; the strawman open-loop controller simply couldn't hold it up. Gravity
    # compensation makes the sweep a test of the ARM's articulation, not of a controller that ignores dynamics.
    acts = []                                                       # (ctrl_slot, qpos_adr, dof_adr, mid, amp, fmax, phase)
    for i in range(model.nu):
        if int(model.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue                                                # only joint actuators sweep a joint angle
        jid = int(model.actuator_trnid[i, 0])
        qadr = int(model.jnt_qposadr[jid]); vadr = int(model.jnt_dofadr[jid])
        if model.jnt_limited[jid]:
            jlo, jhi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        else:
            jlo, jhi = -1.5, 1.5
        fr = model.actuator_forcerange[i]
        fmax = float(fr[1]) if bool(model.actuator_forcelimited[i]) and fr[1] > fr[0] else 500.0
        acts.append((i, qadr, vadr, 0.5 * (jlo + jhi), 0.45 * (jhi - jlo), fmax, 3.0 * i / max(1, model.nu)))
    kp, kd = 40.0, 4.0
    pts = []
    for t in range(steps):
        for slot, qadr, vadr, mid_a, amp_a, fmax, phase in acts:
            desired = mid_a + amp_a * np.sin(2 * np.pi * (t / 180.0) + phase)
            cmd = data.qfrc_bias[vadr] + kp * (desired - data.qpos[qadr]) - kd * data.qvel[vadr]
            data.ctrl[slot] = float(np.clip(cmd, -fmax, fmax))
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
    _select_mujoco_gl()                                      # headless-safe backend (see _select_mujoco_gl)
    try:
        import mujoco
        import PIL.Image

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        gif_dir = _render_dir()                                  # ABSOLUTE, for the same reason _render_gene is
        gif_dir.mkdir(parents=True, exist_ok=True)
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        d = mujoco.MjData(m); frames = []
        # FRAME THE ACTUAL BODY, as _render_gene already does. A fixed lookat z=0.15 / distance=1.9 is sized for a
        # quadruped: the humanoid's walk GIF -- the single most-looked-at artifact the gallery produces -- was
        # cropped to its shins, so the page asserted CREDIBLE WALK over a video of two disembodied legs. Measured
        # ONCE from the standing pose rather than per frame, so the camera does not breathe with the gait; only
        # the horizontal pan tracks the body. Excludes world geoms (the floor plane would blow the box up).
        _span, _dz = 0.6, 0.15
        try:
            import numpy as _np
            mujoco.mj_forward(m, d)
            _bg = [i for i in range(m.ngeom) if int(m.geom_bodyid[i]) != 0]
            if _bg:
                _p = _np.asarray(d.geom_xpos)[_bg]
                _lo, _hi = _p.min(axis=0), _p.max(axis=0)
                _span = float(_np.linalg.norm(_hi - _lo)) or 0.6
                _dz = float((_lo[2] + _hi[2]) / 2.0)          # look at the body's mid-height, not the floor
        except Exception:  # noqa: BLE001 - framing is cosmetic; a fixed camera beats no GIF
            pass
        _dist = float(min(8.0, max(1.2, 2.4 * _span)))        # same 2.4x factor _render_gene is tuned at
        for qp in qpos_frames[::max(1, len(qpos_frames) // 40)]:
            d.qpos[:] = qp; mujoco.mj_forward(m, d)
            rr = mujoco.Renderer(m, height=360, width=480); cam = mujoco.MjvCamera()
            cam.lookat[:] = [float(qp[0]), float(qp[1]), _dz]
            cam.distance, cam.azimuth, cam.elevation = _dist, 125, -12
            rr.update_scene(d, camera=cam); frames.append(PIL.Image.fromarray(rr.render().copy())); rr.close()
        path = gif_dir / f"{tag}.gif"
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
    #
    # ORDER IS THE WHOLE POINT (docs/breaking_the_cotuning_wall.md §1.2, and anatomy_compiler.py's own note at the
    # substitution site). ensure_walkable_quad DECIDES whether to throw the customer's design away and ship a
    # template instead, and it decides by measuring the body -- so it must run on the body we actually ship, with
    # a controller that fits it. Composing with ensure_walkable=True ran that decision HERE, on an UNGROUNDED,
    # UNTUNED body: a horse weighs 4.2 kg at this point and 21 kg once grounded, and every authored quadruped
    # measures 0.000 m at the shipped freq 1.5 / kp 32 op-point (which is one other robot's hand-tuned numbers).
    # So the substitution fired on a measurement artefact. Defer the decision: compose the AUTHORED body, ground
    # it, fit an op-point to it, and only then ask whether it can walk.
    want_walkable = bool(args.get("ensure_walkable", True))
    gene = compose_robot(prompt, ensure_walkable=False)
    # GROUND THE HELD BODY (Stage 2.1). The mass/torque grounding used to run only at export/build time, so the
    # robot the product VERIFIED and TRAINED was not the robot it exported and told the customer to build:
    # measured, a "robot dog" was held at 3.57 kg (-76% vs a Go2's 15 kg) while its own export shipped 13.50 kg
    # -- a 3.8x split-brain, with every CREDIBLE WALK verdict earned on the styrofoam twin. Grounding here makes
    # the held body THE buildable body, so verify/train/evaluate/render and the export all describe one robot.
    # Measured across archetypes: dog 3.57->13.50 kg (Go2 15), arm 2.96->19.00 (FR3 18.3), rover 4.20->110
    # (MiR250 94), hexapod 7.53->14.42 -- and no verdict regressed (the dog walks FURTHER grounded: 1.70 vs
    # 0.64 m). ground_and_repair is idempotent, so export_held's own grounding becomes a no-op for these bodies
    # and still covers imported/amended ones. Fail-open: a grounding error must never block a build.
    try:
        from virturoid.services.gene_build import ground_and_repair
        ground_and_repair(gene)
    except Exception:  # noqa: BLE001 - grounding is the fidelity layer, never a build blocker
        pass
    # Walk-tune the gait per body AFTER grounding, so the cached op-point is tuned for the REAL mass the body
    # ships with (tuning the styrofoam twin would hand the shipped robot a gait fitted to the wrong inertia).
    if robot_kind(gene) == "legged" and args.get("tune_gait", True):
        # NB (flywheel_breakthrough_plan §3.M / §5d): in-place stance_repair was TRIED here and REVERTED — measured
        # 0/5 product-path walk-rate lift (composer already fans offline; the dominant failure is fore-aft LURCH,
        # not lateral roll-over, which lateral splay cannot fix). stance_repair.py is kept for the factory
        # verify-build (default-gait gate), not the hot path.
        #
        # fit_gait_for_body REPLACES tune_crawl_gait here. Same job, but the 6-row grid tuner measured 2.1 s on the
        # grounded authored dog and returned "no robustly-credible open-loop crawl for this body (use learn_gait)"
        # -- i.e. it declined, and the body then went to the substitution gate carrying nothing. The flywheel path
        # recalls a structural prior, searches (bounded, credible-early-stop), re-checks at the DEPLOY horizon and
        # refuses anything that does not beat the shipped default. Bodies that already walk are left
        # byte-identical.
        #
        # THE COST, RE-MEASURED 2026-08-05 on this checkout, because the figures written here when the fit
        # landed ("~0.5 s for a body that already walks; the authored dog 39 evals / 11.4 s and a CREDIBLE
        # WALK") predate the 6000-step settling horizon and the 3 seed restarts that arrived in the same
        # commit, and are now wrong by an order of magnitude in both directions:
        #
        #     grounded authored hexapod    24.2 s     3 evals   adopts
        #     grounded authored cat        56.0 s    87 evals   adopts
        #     grounded authored dog       124.7 s   360 evals   ADOPTS NOTHING
        #     this whole function, dog    143.6 s              (compose 0.11 s, ground <0.01 s -- it is the fit)
        #
        # The dog no longer comes out a credible walk at all: at the horizon that can see the fall, the honest
        # answer is the expensive one. Nothing here should be trimmed to make that cheaper. What a TEST SUITE
        # does about it is VIRTUROID_GAIT_FIT_CACHE / VIRTUROID_SKIP_GAIT_FIT -- see gait_flywheel._FIT_CACHE.
        try:
            from virturoid.services.gait_flywheel import fit_gait_for_body
            # cache=True lets a STRUCTURALLY IDENTICAL body reuse an identical fit -- and it does nothing at all
            # unless VIRTUROID_GAIT_FIT_CACHE=1, which only a test suite sets. A product run has the flag unset
            # and searches every body every time, exactly as before.
            fit_gait_for_body(gene, cache=True)
        except Exception:  # noqa: BLE001 - a tune failure must never block the build; defaults still apply
            pass
    # NOW the walkability decision, on the grounded body with its own operating point. A body is only replaced if
    # it fails WITH A CONTROLLER OF ITS OWN, which is the only version of that question worth asking.
    if want_walkable and (gene.robot_class or "") not in ("aerial", "aquatic"):
        try:
            from virturoid.services.anatomy_compiler import ensure_walkable_quad
            swapped = ensure_walkable_quad(gene, prompt)
            if swapped is not gene:
                # A SUBSTITUTE comes out of the COMPOSER, i.e. ungrounded — it used to inherit this function's
                # grounding because the swap happened before it. Ground and fit it too, or a substituted robot
                # would ship as the styrofoam twin the grounding pass exists to prevent. A body returned with a
                # WIDENED STANCE is the same grounded body it went in as (only mount rotations moved) and already
                # carries an op-point fitted to that stance, so it is left exactly as measured.
                gene = swapped
                if not (getattr(gene, "metadata", None) or {}).get("grounding"):
                    try:
                        from virturoid.services.gene_build import ground_and_repair as _gar
                        _gar(gene)
                        if robot_kind(gene) == "legged" and args.get("tune_gait", True):
                            from virturoid.services.gait_flywheel import fit_gait_for_body
                            fit_gait_for_body(gene, cache=True)
                    except Exception:  # noqa: BLE001 - grounding is the fidelity layer, never a build blocker
                        pass
        except Exception:  # noqa: BLE001 - best-effort; never block a build on the walkability check
            pass
    rid = S.put_robot(gene, prompt=prompt)
    out = {"ok": True, **_summary(gene, rid, prompt=prompt), "prompt": prompt}
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


def probe_robot(args: dict) -> dict:
    """MEASURE the held robot: a read-only query surface, not a fixed summary.

    ``get_robot`` returns one canned summary and ``describe_robot``/``diagnose_body`` compose a NEW body from a
    prompt rather than inspecting the held one, so the questions a designer actually asks had no answer -- how far
    is the gripper from its mount, do these two links pass through each other anywhere in a joint's range, what is
    the torque margin here, how much ground clearance is there.

    ``fields`` selects what to compute (omit for everything):
    ``parts`` ``reach`` ``clearance`` ``mass`` ``torque`` ``overlaps`` ``swept``.

    ``swept`` is the one a static check cannot do: it samples each limited joint across its declared range and
    reports pairs that collide only AWAY from the rest pose -- a design can be perfectly clean as it stands and
    drive one link straight through another halfway through its travel.

    Everything is derived from the compiled MuJoCo model, so an answer cannot disagree with the physics the
    verdict will use, and no LLM is called -- these are measurements, so the zero-product-token rule holds."""
    from virturoid.services import session_state as S
    from virturoid.services.robot_probe import probe
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'; call create_robot first"}
    try:
        return {"ok": True, "robot_id": args["robot_id"], **probe(gene, args)}
    except Exception as exc:  # noqa: BLE001 - a measurement failing must not take the session down
        return {"ok": False, "error": f"could not measure '{args['robot_id']}' ({type(exc).__name__}: {exc})"}


def scope_amend(args: dict) -> dict:
    """What would this amend touch, and what stops being true once it lands — WITHOUT editing anything.

    An edit otherwise just happens: ask for a taller robot, get a new robot back, with no statement of what was
    meant to change, what was meant to stay, or which established facts the change invalidates. This returns
    ``{editable, preserved, invalidates}`` first, so the change can be shown or refused before it commits.

    ``editable`` includes everything BELOW a named part, because a descendant's placement is defined relative to
    its parent and moves whether or not it was named. ``preserved`` is a CLAIM, not a finding -- after the edit,
    ``edit_robot`` reports what actually moved, which is the check that turns the promise into a fact. An
    unclassified operator is assumed to touch everything: a missed recheck is a silent wrong answer, an extra one
    costs seconds of simulation."""
    from virturoid.services import session_state as S
    from virturoid.services.change_impact import scope
    gene = S.get_robot(args.get("robot_id"))
    if gene is None:
        return {"ok": False, "error": f"no robot '{args.get('robot_id')}'; call create_robot first"}
    ops = args.get("ops") or ([{"op": args["op"], "args": args.get("args") or {}}] if args.get("op") else [])
    if not ops:
        return {"ok": False, "error": "pass ops:[{op,args}] (or op/args) — the edit you are considering"}
    return {"ok": True, "robot_id": args["robot_id"], **scope(gene, ops)}


def assert_design(args: dict) -> dict:
    """State what the design MEANT, and be checked against it.

    ``probe_robot`` answers questions, but only the ones you think to ask. This is the other half: declare intent
    as claims the harness verifies -- "the foot should touch the ground", "the gripper stays within 40 cm of its
    mount", "these two are SUPPOSED to overlap, that is a joint housing not a defect".

    It targets one failure mode specifically: a design where every part passes its local checks and the assembly
    still means nothing. An assertion is the only artefact that records what the assembly was FOR, so it is the
    only thing that makes that kind of wrongness detectable.

    ``assertions`` is a list of ``{kind, a, b?, max_m?, min_m?, reason?}``; call with ``{"kind": "list"}`` for the
    vocabulary. Passing ``persist: true`` stores them on the robot so a later ``edit_robot`` re-checks the
    ORIGINAL intent -- an amend that quietly breaks what the design was for is caught by the design's own words.
    ``allow_*`` forms dismiss one false positive in writing, with the reason kept, instead of loosening a
    threshold for every design that follows."""
    from virturoid.services import session_state as S
    from virturoid.services.design_assertions import check, describe, validate
    if str(args.get("kind") or "") == "list" or args.get("list"):
        return {"ok": True, "assertions": describe()}
    gene = S.get_robot(args.get("robot_id"))
    if gene is None:
        return {"ok": False, "error": f"no robot '{args.get('robot_id')}'; call create_robot first"}
    spec = args.get("assertions")
    if spec is not None:
        bad = validate(spec)
        if bad:
            return {"ok": False, "error": "; ".join(bad), "assertions": describe()}
    try:
        out = check(gene, spec)
    except Exception as exc:  # noqa: BLE001 - a check failing must not take the session down
        return {"ok": False, "error": f"could not check '{args['robot_id']}' ({type(exc).__name__}: {exc})"}
    if args.get("persist") and spec:
        gene.metadata = {**(getattr(gene, "metadata", None) or {}), "assertions": list(spec)}
        S.put_robot(gene, robot_id=args["robot_id"], label="assertions")
    return {"ok": True, "robot_id": args["robot_id"], **out}


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
    if bool(args.get("gate_non_regression", True)):
        before = _design_non_regression_signature(gene)
        after = _design_non_regression_signature(new_gene)
        if after > before:
            return {
                "ok": False,
                "error": "edit auto-reverted because deterministic design findings regressed",
                "reverted": True,
                "before": {"high_or_fatal": before[0], "weighted_findings": before[1]},
                "after": {"high_or_fatal": after[0], "weighted_findings": after[1]},
                "proposed_diffs": diffs,
            }
    # AN EDIT THAT DISMEMBERS THE ROBOT IS NOT A SUCCESS. ``evaluate_structural_assertions`` measures the
    # visible gap across every parent->child seam and is already a HARD gate on ``submit_design``
    # (agent_design_tools), but the amend path never called it -- so a ``set_height`` that pulled a humanoid's
    # thighs off its pelvis and left its shins hanging in mid-air returned ``ok: true`` with a tidy diff and no
    # warning. It is deliberately a NO-NEW-DETACHMENT gate, not an absolute one: a lossy imported twin can
    # arrive with seams already open, and refusing to let the customer edit their own robot because of a defect
    # we introduced at ingest would be worse than the bug. Only seams this edit BROKE block the commit.
    seams = _newly_detached_seams(gene, new_gene)
    if seams and bool(args.get("gate_connectivity", True)):
        return {
            "ok": False,
            "error": (f"edit REJECTED: it detached {len(seams)} link(s) from the body — "
                      + "; ".join(f"{s['seam']} now separated by {s['gap_m']:.3f} m" for s in seams[:4])
                      + (f" (+{len(seams) - 4} more)" if len(seams) > 4 else "")
                      + ". The robot would render as floating pieces. Nothing was applied; try a smaller "
                        "change, or an op that rebuilds the topology (set_leg_count / add_limb)."),
            "detached_seams": seams[:12],
            "reverted": True,
            "proposed_diffs": diffs,
        }
    label = ",".join(d.get("op", "edit") for d in diffs)
    S.commit_robot(rid, new_gene, label=label)
    out = {"ok": True, "diffs": diffs, "summary": _summary(new_gene, rid),
           "structural": any(d.get("structural") for d in diffs), **S.robot_meta(rid)}
    img = _render_gene(new_gene, f"{rid}_{S.robot_meta(rid)['undo_depth']}")
    if img:
        out["artifacts"] = [img]
    return out


def _seam_gaps(gene) -> dict:
    """``{seam_name: gap_m}`` for every parent->child edge, from the shared structural-assertion measurement.

    Measured on the MESHED model — the body the customer actually looks at. The primitive measurement cannot
    see the failure this gate exists to catch: an imported robot is drawn from its own baked STLs, so an edit
    can lengthen every collider (which stays mated) while the drawn links come apart.
    """
    try:
        from virturoid.services.structural_assertions import evaluate_structural_assertions
        rep = evaluate_structural_assertions(gene, meshed=True)
    except Exception:  # noqa: BLE001 - unmeasurable is not the same as detached; the caller treats it as unknown
        return {}
    out = {}
    for a in rep.assertions:
        if a.name.startswith("seam:") and isinstance(a.value, (int, float)):
            out[a.name] = float(a.value)
    return out


def _newly_detached_seams(before, after, *, tol_m: float = 0.006) -> list[dict]:
    """Seams this edit BROKE: mated (or absent) before, detached after.

    Relative on purpose — an imported customer robot is a lossy re-derivation of their model and can arrive
    with seams already over budget. Blocking every edit on those would make their own robot uneditable, which
    is a worse failure than the one being fixed. A seam that was already open only counts if the edit made it
    materially worse (>1 mm), so an edit cannot hide behind a pre-existing gap either.
    """
    b, a = _seam_gaps(before), _seam_gaps(after)
    if not a:
        return []
    bad = []
    for name, gap in a.items():
        if gap <= tol_m:
            continue
        was = b.get(name)
        if was is not None and was > tol_m and gap <= was + 0.001:
            continue                                   # already detached before this edit, and not made worse
        bad.append({"seam": name[5:], "gap_m": round(gap, 4), "was_m": (round(was, 4) if was is not None else None)})
    return sorted(bad, key=lambda r: -r["gap_m"])


def _design_non_regression_signature(gene) -> tuple[int, int]:
    """Deterministic ordering used to accept/revert a critique-driven edit.

    High/fatal findings dominate, then a weighted total. If a verifier is unavailable, it contributes no
    finding rather than inventing evidence; the normal schema gate in ``apply_ops`` still applies.
    """
    severities: list[str] = []
    try:
        from virturoid.services.gene_validation import validate_gene_design
        severities += [f["severity"] for f in validate_gene_design(gene)["risk_flags"]]
    except Exception:  # noqa: BLE001
        pass
    try:
        from virturoid.services.anatomy_critic import critique_gene
        severities += [f["severity"] for f in critique_gene(gene)["issues"]]
    except Exception:  # noqa: BLE001
        pass
    try:
        from virturoid.services.visual_physics_gate import audit_gene
        severities += ["high" for _ in audit_gene(gene).issues]
    except Exception:  # noqa: BLE001
        pass
    weight = {"fatal": 8, "high": 5, "med": 2, "low": 1}
    hard = sum(severity in ("fatal", "high") for severity in severities)
    return hard, sum(weight.get(severity, 0) for severity in severities)


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
    """Render the held robot. ``view``: ``visual`` (default — the detailed surface) or ``collision`` (the exact
    bodies the physics verdict is computed on). Either way the payload DISCLOSES which model was drawn and
    proves the two share one set of colliders — see :func:`render_parity`."""
    from virturoid.services import session_state as S
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'"}
    view = str(args.get("view", "visual")).lower()
    if view not in ("visual", "collision"):
        return {"ok": False, "error": f"unknown view '{view}'; choose 'visual' or 'collision'"}
    img, why = _render_gene_detail(gene, f"{args['robot_id']}_{'collision' if view == 'collision' else 'view'}",
                                   azimuth=float(args.get("azimuth", 50.0)),
                                   elevation=float(args.get("elevation", -16.0)),
                                   collision=(view == "collision"))
    # THE PATH IS THE PRODUCT. This is the tool an engineer calls to SEE the robot, so it either hands back one
    # absolute path to a file that opens, or it says why not in words someone can act on. It used to report the
    # picture only under ``artifacts`` (so the obvious ``result['path']`` read as None on a render that had in
    # fact succeeded), as a CWD-relative string, and on failure as the bare phrase "render unavailable".
    if img is None:
        return {"ok": False, "path": None, "artifacts": [], "view": view,
                "error": f"could not render '{args['robot_id']}': {why}",
                "robot_id": args["robot_id"]}
    p = Path(img)
    if not p.is_file():                                        # never report a path that does not open
        return {"ok": False, "path": None, "artifacts": [], "view": view,
                "error": f"the renderer reported {p} but no file is there", "robot_id": args["robot_id"]}
    out = {"ok": True, "path": str(p), "artifacts": [str(p)], "bytes": p.stat().st_size, "view": view,
           "error": None, "robot_id": args["robot_id"]}
    # SAY WHICH ROBOT THIS PICTURE IS. The visual render adds detailed surface meshes that the gait sim does not
    # build, so "is the thing I'm looking at the thing you verified?" is a fair question with a measurable
    # answer. Attach it rather than leaving the customer to trust a comment in the source.
    out["parity"] = render_parity(gene)
    return out


_GEOM_TYPE_NAME = {0: "plane", 1: "heightfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
                   5: "cylinder", 6: "box", 7: "mesh"}


def _contact_disclosure(mv, band: float = 0.02) -> dict:
    """Name the surface the GROUND CONTACT is actually resolved on, per body that reaches the floor.

    ``render_parity`` proves the drawn body and the simulated body share one collider set. This answers the
    sharper question a picture invites, and the one place where a pretty render can quietly mislead: the foot
    you SEE is a generated visual mesh, while the contact the walk verdict is computed from is resolved on the
    link's collision PRIMITIVE. Those are different surfaces even when they are the same size — a mesh sole and
    a capsule are not the same contact patch — and the honest move is to say which one the solver used rather
    than let a boot-shaped render imply a contact shape nobody simulated.

    Reported per foot-like body: the collider's primitive type and size, and whether that body is DRAWN with a
    mesh. ``drawn_with_mesh`` true + a non-mesh ``collider_type`` is exactly the render/sim asymmetry to
    disclose. Measurement only — this function never changes a model.
    """
    import mujoco
    import numpy as np

    try:
        d = mujoco.MjData(mv)
        if mv.nkey:
            mujoco.mj_resetDataKeyframe(mv, d, 0)
        mujoco.mj_forward(mv, d)
        aabb = mv.geom_aabb.reshape(mv.ngeom, 6)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)

        def low(i):
            c, h = aabb[i, :3], aabb[i, 3:]
            w = d.geom_xpos[i] + (c + signs * h) @ d.geom_xmat[i].reshape(3, 3).T
            return float(w[:, 2].min())

        robot = [i for i in range(mv.ngeom) if int(mv.geom_bodyid[i]) != 0]
        colliders = [i for i in robot if int(mv.geom_contype[i]) or int(mv.geom_conaffinity[i])]
        if not colliders:
            return {"surfaces": [], "note": "this body has no collidable geometry"}
        floor = min(low(i) for i in colliders)
        mesh_bodies = {int(mv.geom_bodyid[i]) for i in robot if int(mv.geom_type[i]) == 7}
        out = []
        for i in sorted(colliders, key=low):
            if low(i) > floor + band:
                continue
            bid = int(mv.geom_bodyid[i])
            out.append({
                "body": mujoco.mj_id2name(mv, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}",
                "collider_type": _GEOM_TYPE_NAME.get(int(mv.geom_type[i]), str(int(mv.geom_type[i]))),
                "collider_size_m": [round(float(v), 5) for v in mv.geom_size[i] if float(v) > 0.0],
                "drawn_with_mesh": bid in mesh_bodies,
            })
        drawn = [s for s in out if s["drawn_with_mesh"] and s["collider_type"] != "mesh"]
        return {
            "surfaces": out,
            "render_differs_from_contact": bool(drawn),
            "note": ("the surfaces listed are the ones the solver resolves ground contact on. Where "
                     "drawn_with_mesh is true the render draws a detailed visual mesh over that link while "
                     "contact is computed on the collider primitive named here — the picture and the contact "
                     "patch are different surfaces, so judge footing from collider_type, not from the boot in "
                     "the image. Render them with render_view(view='collision') to see them directly."
                     if drawn else
                     "every ground-contact surface here is drawn as the same primitive the solver uses."),
        }
    except Exception as exc:  # noqa: BLE001 - a disclosure that cannot be measured says so
        return {"error": f"{type(exc).__name__}: {exc}",
                "note": "could not measure which surface carries ground contact"}


def render_parity(gene) -> dict:
    """Measured proof that the RENDERED body and the SIMULATED body are one robot.

    ``render_view`` builds ``gene_to_meshed_mjcf`` while the gait/verdict path builds ``compile_gene_to_mjcf``,
    and the two do look different in a geom census — on a held Menagerie Go2, 43 geoms with 13 meshes versus 47
    geoms with 21 capsules and 25 cylinders. That census is misleading: those extra cylinders are motor cans,
    collars and hubs emitted at ``mass=0 contype=0 conaffinity=0``, and the mesh visuals are emitted the same
    way. What actually decides a verdict is the COLLIDER set and the inertial properties, and those are built by
    the same function from the same gene. Measured identical on the Go2, G1, Spot, and on composed dogs and
    hexapods: 14/14 colliders matched by body, type, size and position, total mass to 1e-9, and the same spawn
    height. This function re-measures that on demand so the claim is checkable rather than asserted.
    """
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z

    def census(xml):
        m = mujoco.MjModel.from_xml_string(xml)
        colliders = sorted(
            (int(m.geom_bodyid[i]), int(m.geom_type[i]), tuple(np.round(m.geom_size[i], 6)),
             tuple(np.round(m.geom_pos[i], 6)))
            for i in range(m.ngeom)
            if int(m.geom_contype[i]) != 0 or int(m.geom_conaffinity[i]) != 0)
        return m, colliders

    try:
        z_v, z_s = standing_spawn_z(gene), standing_spawn_z(gene, meshed=False)
        mv, cv = census(gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=z_v))
        ms, cs = census(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=z_s))
        same = (cv == cs and abs(float(sum(mv.body_mass)) - float(sum(ms.body_mass))) < 1e-9
                and abs(z_v - z_s) < 1e-9)
        return {
            "colliders_identical": cv == cs, "n_colliders": [len(cv), len(cs)],
            "mass_identical": abs(float(sum(mv.body_mass)) - float(sum(ms.body_mass))) < 1e-9,
            "spawn_z_identical": abs(z_v - z_s) < 1e-9,
            "geoms_total": [int(mv.ngeom), int(ms.ngeom)], "visual_meshes": int(mv.nmesh),
            "same_physics": bool(same),
            "contact": _contact_disclosure(mv),
            "note": ("the rendered body and the simulated body share ONE set of colliders, masses and spawn "
                     "height; the extra geoms on each side are cosmetic (mass=0, contype=0) — motor cans and "
                     "collars on the sim model, surface meshes on the rendered one. "
                     "Call render_view with view='collision' to SEE the bodies the verdict is computed on."
                     if same else
                     "WARNING: the rendered body and the simulated body DIFFER in physics — the picture is not "
                     "the robot being verified."),
        }
    except Exception as exc:  # noqa: BLE001 - parity is a disclosure; an unmeasurable one says so
        return {"same_physics": None, "error": f"{type(exc).__name__}: {exc}",
                "note": "could not measure render/sim parity for this body"}


# ------------------------------------------------------- WHOSE CONTROLLER IS THIS VERDICT ABOUT? (imported bodies)
#
# A motion verdict is never a claim about a BODY. It is a claim about a BODY UNDER A CONTROLLER, and until this
# block existed we published the first sentence while only ever measuring the second.
#
# THE DEFECT, MEASURED 2026-08-04 on the real MuJoCo Menagerie Go2 driven through `agent_tools.call_tool`:
# ingest reads its mass as 15.206 kg (exactly the manufacturer's) and `verify_robot` then answers
# "FELL by YAW-DRIFT (roll 151 / pitch 86 / yaw 174 deg max), survived false". Every number in that sentence is
# true and the sentence is useless, because a Go2 walks perfectly well — with UNITREE's controller. We drove the
# customer's machine with OUR generic crawl gait (`morph_policy.crawl_gait_rollout` at its shipped freq 1.5 /
# kp 32, or a hint region mined from OTHER robots' banked walks), it fell, and we reported the fall in a
# grammatical form an engineer reads as a finding about their robot. It is a finding about our gait.
#
# THE RULE. `verify_robot` may state a motion verdict ABOUT THE CUSTOMER'S ROBOT only when the controller that
# produced it is one we produced FOR THAT BODY — an operating point fitted to this gene (`tuned_for_this_body`)
# or a policy trained on it (`learned_policy`). For an IMPORTED body under a generic prior, the honest output is
# a REFUSAL to rule on locomotion, the facts we can measure without a controller, the generic-gait rollout kept
# verbatim but explicitly labelled as ours, and the two things that would actually answer the question.
#
# WHAT THIS DOES NOT DO. It does not touch `gait_quality.classify`, which is the un-gameable verdict and this
# product's differentiator. The classifier was never wrong; it was answering a question the customer did not ask.
# Its string is carried through byte-identical under `under_our_generic_gait.verdict`. A body WE composed is
# untouched on every field — our generic gait IS the shipped controller for a body we designed, so the verdict
# there is exactly the claim we are entitled to make.
#
# WHAT A ROBOT PACKAGE ACTUALLY SHIPS (swept, all 63 MuJoCo Menagerie packages, 2026-08-04): 39/63 carry at least
# one keyframe (`home` x30, `stand`, `retract`, `pickup`, `hover`); 46/63 declare position servos with real
# gains, 11/63 pure torque motors. That is a POSE and a JOINT-LEVEL SERVO. **Not one of the 63 ships a
# locomotion controller** — no gait, no policy, no trajectory. The thing we would need in order to answer "does
# your robot walk" is precisely the thing that is never in the box, which is why the honest answer is a refusal
# plus a route, and not a better guess.
#: ``gait_source`` values that mean "a controller WE produced FOR THIS BODY". Everything else — ``default_crawl``
#: (the crawl gait's shipped constants) and ``flywheel_hint`` (a region mined from OTHER bodies' banked walks) —
#: is a generic prior fitted to no particular robot.
_FITTED_TO_THIS_BODY = ("tuned_for_this_body", "learned_policy", "biped_learned")


def _import_provenance(gene) -> dict | None:
    """``{...}`` when this body came in through the import path, else ``None`` (we composed it ourselves).

    Keyed on ``metadata['imported_from']``, which ``robot_import.import_robot`` sets once and which
    ``gene_build`` already trusts to decide whether the customer's geometry may be regenerated — so an imported
    body is recognised here by exactly the marker the rest of the system recognises it by, not a second guess.
    """
    meta = getattr(gene, "metadata", None) or {}
    if not str(meta.get("imported_from") or ""):
        return None
    return {"source": str(meta.get("imported_from")),
            "mass_from_source": str(meta.get("mass_source") or "") == "source_model",
            "torque_from_source": str(meta.get("torque_source") or "") == "source_model",
            "declared_rest_pose": meta.get("rest_pose_source")}


def _controller_free_facts(gene, *, hold_steps: int = 1200) -> dict:
    """What can be said about an imported machine WITHOUT having its controller. Four measurements, each a fact
    about the ROBOT rather than about a gait:

    * ``joint_limits_respected`` — is the pose the customer's own model declares inside every joint range it
      declares? Pure kinematics; no actuator is commanded.
    * ``self_collision_at_home`` — do any of the customer's own links interpenetrate at that pose? Read off
      MuJoCo's own contact list (exact geometry), not an AABB approximation.
    * ``static_holding_torque`` — the torque each joint must produce to hold the limb distal to it at that
      pose, versus the torque limit the customer's model declares for it. Zero rollouts, zero controller: it is
      arithmetic over the compiled model, and it is the number a datasheet is checked against.
    * ``stands_under_gravity`` — the one dynamic check, and the only one here that commands anything: hold every
      joint at the declared pose with a gravity-compensated position hold, torques CLAMPED to the customer's own
      declared limits, and watch the base for ``hold_steps``. A posture hold is not locomotion — it commands one
      constant setpoint and never sequences a step — so it cannot be mistaken for a gait, and the clamp is what
      keeps it from being vacuous: gravity compensation that exceeds the declared limit gets cut off and the
      machine sags, which is exactly the honest failure to report.

    Every one of these is measured on ``compile_gene_to_mjcf``'s model — the same model the verdict and the
    ``render_view(view='collision')`` picture come from — so nothing here can disagree with what is on screen.
    """
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z

    meta = getattr(gene, "metadata", None) or {}
    pose_src = meta.get("rest_pose_source")
    out: dict = {"declared_home_pose": (
        str(pose_src) if pose_src else
        "your model declares no rest keyframe — everything below was measured at its ZERO pose, which for a "
        "legged robot means straight legs and is not a stance any real machine holds")}

    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # --- kinematic: is the declared pose legal? -----------------------------------------------------------
    over = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (2, 3) or not bool(m.jnt_limited[j]):
            continue
        q = float(d.qpos[int(m.jnt_qposadr[j])])
        lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
        if q < lo - 1e-6 or q > hi + 1e-6:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            over.append({"joint": nm, "angle_rad": round(q, 4), "range_rad": [round(lo, 4), round(hi, 4)]})
    out["joint_limits_respected"] = {"ok": not over, "violations": over[:8],
                                     "n_limited_joints": int(sum(1 for j in range(m.njnt)
                                                                 if bool(m.jnt_limited[j])))}

    # --- kinematic: does the body pass through itself at that pose? ---------------------------------------
    pen = []
    for c in range(int(d.ncon)):
        con = d.contact[c]
        b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
        if b1 == 0 or b2 == 0 or float(con.dist) > -1e-4:       # floor contact / touching, not interpenetrating
            continue
        pen.append({"a": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body{b1}",
                    "b": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body{b2}",
                    "depth_mm": round(-float(con.dist) * 1000.0, 2)})
    out["self_collision_at_home"] = {"ok": not pen, "pairs": pen[:8]}

    # --- static: what torque does the declared pose demand of each joint? ---------------------------------
    # qfrc_bias at rest IS the generalized force gravity+bias demands of each DOF; forcerange is what the
    # customer's model says that joint can produce. No controller is involved in either number.
    #
    # SAY EXACTLY WHAT THIS IS. The body is spawned 2 mm clear of the floor, so this is each joint carrying the
    # limb DISTAL TO IT — the number a datasheet is checked against, and the same one ``probe_robot(fields=
    # ['torque'])`` reports. It is NOT the ground-reaction share a standing robot's legs carry; that load shows
    # up in ``stands_under_gravity``, where the hold runs clamped to these very limits and a joint that cannot
    # carry its share saturates. Naming this "holds its own weight" would overclaim a hanging-limb torque.
    worst, tight = None, []
    for u in range(m.nu):
        if int(m.actuator_trntype[u]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        j = int(m.actuator_trnid[u, 0])
        need = abs(float(d.qfrc_bias[int(m.jnt_dofadr[j])]))
        fr = m.actuator_forcerange[u]
        cap = float(fr[1]) if bool(m.actuator_forcelimited[u]) and fr[1] > fr[0] else None
        if cap is None or cap <= 0:
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        frac = need / cap
        if worst is None or frac > worst[1]:
            worst = (nm, frac, need, cap)
        if frac > 1.0:
            tight.append({"joint": nm, "needs_nm": round(need, 2), "declared_limit_nm": round(cap, 2)})
    out["static_holding_torque"] = {
        "ok": not tight, "over_limit": tight[:8],
        "worst_joint": (None if worst is None else
                        {"joint": worst[0], "uses_frac_of_declared_limit": round(worst[1], 3),
                         "needs_nm": round(worst[2], 2), "declared_limit_nm": round(worst[3], 2)}),
        "note": ("torque each joint must produce to hold the limb DISTAL TO IT at the declared pose, against "
                 "the torque limit YOUR model declares for that joint"
                 + (" (your numbers, read at import)" if str(meta.get("torque_source") or "") == "source_model"
                    else " (estimated by us — your model declared none)")
                 + ". Ground-reaction load is not in this number; see stands_under_gravity, where the hold runs "
                   "clamped to these same limits")}

    # --- dynamic: does it stay up? -------------------------------------------------------------------------
    # TILT IS MEASURED RELATIVE TO THE SPAWN ATTITUDE, never as an absolute Euler angle. An IMPORTED root carries
    # the rotation that aligns its reconstructed link frame (`gene_compiler._pose_keyframe`), so the Go2's base
    # quaternion reads roll 90.0 / pitch 76.5 deg AT STEP ZERO, before any physics runs — a body standing
    # perfectly square on four feet. An absolute gate would have failed every imported robot at t=0 and called it
    # a topple, which is the same class of false statement this whole block exists to remove.
    from virturoid.services.morph_policy import posture_hold_gains

    free_base = any(int(m.jnt_type[j]) == 0 for j in range(m.njnt))
    if not free_base:
        out["stands_under_gravity"] = {"applicable": False,
                                       "note": "this body is mounted, not free-standing — there is no stance to hold"}
        return out
    kp_v, kd_v, tau_v = posture_hold_gains(m, d)
    tgt = np.array(d.qpos, dtype=float)
    acts = [(u, int(m.jnt_qposadr[int(m.actuator_trnid[u, 0])]), int(m.jnt_dofadr[int(m.actuator_trnid[u, 0])]),
             float(tgt[int(m.jnt_qposadr[int(m.actuator_trnid[u, 0])])]))
            for u in range(m.nu) if int(m.actuator_trntype[u]) == int(mujoco.mjtTrn.mjTRN_JOINT)]
    base = next((int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == 0), 1)
    up0 = np.array(d.xmat[base]).reshape(3, 3)[:, 2].copy()
    z0 = float(d.qpos[2])
    sat, z_half = 0, z0
    for t in range(int(hold_steps)):
        for u, qadr, vadr, want in acts:
            cmd = d.qfrc_bias[vadr] + kp_v[u] * (want - d.qpos[qadr]) - kd_v[u] * d.qvel[vadr]
            if abs(cmd) > tau_v[u]:
                sat += 1
            d.ctrl[u] = float(np.clip(cmd, -tau_v[u], tau_v[u]))
        mujoco.mj_step(m, d)
        if t == int(hold_steps) // 2:
            z_half = float(d.qpos[2])
    up1 = np.array(d.xmat[base]).reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(float(np.dot(up0, up1)), -1.0, 1.0))))
    z1 = float(d.qpos[2])
    jerr = max((abs(float(d.qpos[q]) - w) for _u, q, _v, w in acts), default=0.0)
    sat_frac = sat / max(1, int(hold_steps) * max(1, len(acts)))
    # STANDING IS CONVERGENCE, NOT A SMALL NUMBER. Judging the raw settle against a fraction of spawn height
    # mislabels a small robot: an imported google_barkour_vb settles 20 mm on a 125 mm base — 16% — while sitting
    # dead level (0.0 deg tilt, 0.011 rad tracking error, ZERO torque saturation). That is a machine standing
    # still, and a percentage gate called it a collapse. What separates standing from collapsing is whether the
    # base STOPPED MOVING: over the second half of the hold, a stander has converged and a collapser has not.
    settling = abs(z1 - z_half)
    stands = (tilt < 15.0 and jerr < 0.15 and settling < 0.005 and z1 > 0.5 * max(z0, 1e-9))
    # STATIC STABILITY DECIDES WHETHER "did not stand" MEANS ANYTHING. A body with >=3 ground contacts can be
    # held up by posture alone; a biped cannot — balancing 2 legs needs an ACTIVE controller, which is precisely
    # the thing we said we do not have. Reporting "your humanoid did not stand" without that caveat would be a
    # brand-new false implication, so the finding is marked inconclusive instead of being stated. An UNKNOWN leg
    # count is inconclusive too, never presumed stable: an imported Agility Cassie measures 0 legs here, and
    # presuming stability would have published exactly that false statement about a biped.
    try:
        from virturoid.services.body_kind import measured_legs
        legs = measured_legs(gene)
    except Exception:  # noqa: BLE001
        legs = None
    statically_stable = legs is not None and int(legs) >= 3
    secs = round(int(hold_steps) * float(m.opt.timestep), 2)
    out["stands_under_gravity"] = {
        "applicable": True, "held_s": secs,
        "stands": True if stands else (False if statically_stable else None),
        "conclusive": bool(stands or statically_stable),
        "base_height_m": [round(z0, 4), round(z1, 4)], "base_sag_m": round(z0 - z1, 4),
        "still_settling_m": round(settling, 5), "tilt_from_spawn_deg": round(tilt, 1),
        "joint_tracking_err_rad": round(jerr, 4), "torque_saturated_frac": round(sat_frac, 3),
        "n_legs": legs,
        "how": ("your declared home pose held by a joint position hold, torques clamped to the limits your own "
                "model declares. A posture hold is not a gait: it commands one constant setpoint and never "
                "sequences a step"),
    }
    if not stands and not statically_stable:
        out["stands_under_gravity"]["why_inconclusive"] = (
            f"we measure {legs} ground-contact limbs on this body, so it is not statically stable — staying "
            f"upright needs an ACTIVE balance controller, which is exactly what we do not have for your robot. "
            f"That it went down under a PASSIVE posture hold says nothing about whether your machine balances.")
    return out


def _reframe_for_imported_body(res: dict, gene, kind: str) -> None:
    """Rewrite a verdict on an IMPORTED body so it cannot be read as a claim about the customer's machine.

    In place, and only when two things are true at once: the body came in through the import path, AND the
    controller that produced the verdict is a generic prior of ours rather than something fitted to this body.
    A body we composed keeps every field byte-for-byte; so does an imported body under a controller we trained
    or fitted FOR it, because there the verdict IS about a controller-body pair we are entitled to describe.
    """
    prov = _import_provenance(gene)
    if prov is None:
        return
    src = str(res.get("gait_source") or "")
    if src in _FITTED_TO_THIS_BODY:                              # our controller, fitted to THEIR body -> a fair claim
        res["controller_provenance"] = {
            "whose": "ours, fitted to this body",
            "what": src,
            "verdict_is_about": "your body under the controller we produced for it — not your own controller",
            "note": "if you have your own controller, import it (import_onnx_policy / sandbox_policy / "
                    "import_control_script) and re-verify: this number is our best, not your machine's ceiling"}
        return

    original = str(res.get("verdict") or "")
    if kind == "legged" and original:
        try:
            facts = _controller_free_facts(gene)
        except Exception as exc:  # noqa: BLE001 - a refusal must still be a refusal when the facts won't measure
            facts = {"error": f"{type(exc).__name__}: {exc}",
                     "note": "the controller-free measurements could not be taken on this body"}
        stand = facts.get("stands_under_gravity") or {}
        _lead = "NO LOCOMOTION VERDICT — we do not have your robot's controller"
        if stand.get("stands"):
            head = (f"{_lead}. What we CAN state: it STANDS under gravity at its own declared home pose "
                    f"({stand['held_s']} s, sag {stand['base_sag_m']:.3f} m, tilt "
                    f"{stand['tilt_from_spawn_deg']:.1f} deg from spawn)")
        elif stand.get("applicable") and not stand.get("conclusive"):
            head = (f"{_lead}, and with {stand.get('n_legs')} legs it cannot be balanced by a posture hold "
                    f"either — staying up needs the active balance controller we do not have. Structural facts "
                    f"we CAN state are below")
        elif stand.get("applicable"):
            head = (f"{_lead}. What we CAN state: it did NOT hold its own declared home pose under gravity "
                    f"(sag {stand.get('base_sag_m')} m, tilt {stand.get('tilt_from_spawn_deg')} deg from spawn) "
                    f"— read the stance/torque figures below before reading anything into the gait numbers")
        else:
            head = _lead
        res["locomotion_verdict"] = None
        res["verdict"] = head
        res["measured_without_your_controller"] = facts
        res["under_our_generic_gait"] = {
            "verdict": original,                                 # gait_quality.classify(), byte-identical
            "gait": f"our generic crawl gait ({src or 'default_crawl'})",
            "forward_m": res.get("forward_m"), "survived": res.get("survived"),
            "horizon_steps": res.get("horizon_steps"),
            "means": ("this is a measurement of OUR gait on YOUR body. Our crawl gait is a fixed open-loop "
                      "pattern fitted to no particular robot; it was never tuned for yours and is not what your "
                      "robot runs. Read it as evidence about our controller, NOT as a finding about your "
                      "machine — a robot that walks on your bench will still fall here."),
        }
    elif kind == "manipulator" and isinstance(res.get("grasp"), dict):
        g = res["grasp"]
        det = (g.get("detail") or {}) if isinstance(g.get("detail"), dict) else {}
        reasons = {str(a.get("reason") or "") for a in (det.get("attempts") or []) if isinstance(a, dict)}
        no_gripper = bool(reasons) and reasons <= {"no_grasp_site"}
        reach = original.split(";")[0].strip()                   # ARTICULATES/STUCK — kinematics, controller-free
        if no_gripper:
            res["verdict"] = f"{reach}; NO GRIPPER in your model — there is nothing here to grasp with"
            g["means"] = ("your model ships no gripper site, so no grasp was attempted. This is a structural "
                          "fact about the file you gave us, not a capability judgement.")
        else:
            res["verdict"] = f"{reach}; NO GRASP VERDICT — we do not have your gripper's controller"
            g["means"] = ("this is a measurement of OUR generic grasp-and-lift script on YOUR gripper. It was "
                          "never tuned for your hand and is not what your robot runs. Read it as evidence "
                          "about our script, NOT as a finding about your gripper.")
        g["under"] = "our generic grasp-and-lift script"
        res["grasp_verdict_scope"] = "our script on your gripper"
    elif kind == "manipulator":
        # QUICK mode on an arm: the only thing measured is REACH, and reach is kinematics — link lengths and
        # joint ranges, no controller anywhere in it. Attaching "under our controller, not yours" to a ruler
        # would be a caveat with nothing behind it, so the verdict stands and only the provenance block is added.
        pass
    elif original:
        # mobile / aerial / aquatic / anything else: those verdicts DO come out of a controller of ours (the
        # drive script, the flight controller), so the scope has to travel with the sentence.
        res["verdict"] = f"{original} — under OUR generic controller, not yours"

    # BE PRECISE ABOUT WHAT WE DID AND DID NOT GET. "We do not have your controller" is a vague complaint on its
    # own; paired with the list of what we DID take from the customer's file it becomes a checkable statement,
    # and it forestalls the reasonable objection "my model has a home keyframe, why didn't you use it" — we did.
    took = ["the joint ranges your model declares"]
    if prov["mass_from_source"]:
        took.insert(0, "your per-link masses, verbatim")
    if prov["torque_from_source"]:
        took.insert(0, "your declared actuator torque limits, verbatim")
    if prov["declared_rest_pose"]:
        took.append(f"your declared rest pose ({prov['declared_rest_pose']})")
    res["controller_provenance"] = {
        "whose": "ours, generic — fitted to no particular robot",
        "what": src or "generic scripted controller",
        "yours_was_not_supplied": True,
        "verdict_is_about": "our controller on your body",
        "what_we_took_from_your_model": took,
        "what_your_model_could_not_supply": (
            "a controller. A robot description ships geometry, inertia, limits and at most a rest POSE plus "
            "joint-level servo gains — never a gait, a policy or a trajectory. Swept across all 63 MuJoCo "
            "Menagerie packages: 39 carry a keyframe, 46 declare position servos, and ZERO ship a locomotion "
            "controller. The one thing needed to answer 'does this robot walk' is the one thing never in the box"),
        "why_this_matters": ("a motion verdict is a claim about a BODY UNDER A CONTROLLER. We have your body "
                             "and not your controller, so we decline to rule on what your robot can do."),
    }
    res["imported"] = prov
    res["to_get_a_real_verdict"] = [
        {"if": "you already have a controller for this robot",
         "do": "import_onnx_policy (a .onnx policy), sandbox_policy (inline Python), or import_control_script — "
               "then verify_robot again; the verdict then describes YOUR controller on YOUR body"},
        {"if": "you want US to produce one for this body",
         "do": "adapt_gait fits an operating point to this specific gene, and train_held learns a policy on it. "
               "Either makes the verdict a claim about a controller measured on YOUR robot"},
        {"if": "you only need the structural facts",
         "do": "probe_robot (mass / torque margins / clearance / self-collision through the joint ranges) and "
               "render_view(view='collision') — none of those need a controller at all"},
    ]


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
    a GIF for legged) or ``quick`` (fast iterate check). Folds simulate_gait (G-G).

    A verdict on an IMPORTED body is additionally scoped to the controller that produced it — see
    ``_reframe_for_imported_body``. We have the customer's robot and not the customer's controller, so on a
    generic prior this tool DECLINES to rule on locomotion and reports what it can measure without one."""
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
            # FULL uses the SETTLING horizon, because the full verify is where the walk claim reaches a customer
            # and 1500 was the horizon the claim was false at: the grounded authored cat read CREDIBLE WALK at
            # 1500 and fell at step 2014 — the credible walk WAS the fall, happening 514 steps after we stopped
            # looking (task #267). QUICK stays 800 and now says `settled: false`, so a screen cannot be mistaken
            # for the claim. The GIF is unaffected: _honest_gait subsamples the trace with the horizon.
            from virturoid.services.gait_flywheel import _SETTLE_STEPS
            steps = int(args.get("steps", 800 if quick else _SETTLE_STEPS))
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
        # WHOSE CONTROLLER WAS THAT? An imported robot driven by our generic prior gets its verdict reframed so
        # the sentence cannot be read as a claim about the customer's machine (see the block above _honest_gait's
        # callers). No-op for every body we composed, and for an imported body under a controller fitted to it.
        try:
            _reframe_for_imported_body(res, gene, kind)
        except Exception:  # noqa: BLE001 - the reframe is a disclosure; never let it swallow a measured verdict
            pass
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
    # M11 honesty: build_scene silently falls back to 'warehouse' for an unknown theme while the caller echoes
    # the requested one -- reject up front with the known list, the same way edit_scene/apply_theme already do,
    # so we never hand back a warehouse labelled 'marsbase'.
    if theme not in T.THEMES:
        return {"ok": False, "error": f"unknown theme '{theme}'; known: {T.theme_names()}"}
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
    "probe_robot": {"description": "MEASURE the held robot: a queryable read-only surface over the COMPILED "
                    "MuJoCo model, so an answer cannot disagree with the physics the verdict uses. `fields` "
                    "selects what to compute (omit for everything): parts/reach/clearance/mass/torque/overlaps/"
                    "swept. `swept` is the one a static check cannot make — it samples each limited joint across "
                    "its declared range and reports pairs that collide only AWAY from the rest pose.",
                    "heavy": True, "handler": probe_robot, "parameters": {"type": "object",
                    "required": ["robot_id"], "properties": {"robot_id": {"type": "string"},
                    "fields": {"type": "array", "description": "which measurements to compute; omit for the "
                               "whole report", "items": {"type": "string", "enum": [
                                   "parts", "reach", "clearance", "mass", "torque", "overlaps", "swept"]}}}}},
    "assert_design": {"description": "State what the design MEANT and be checked against it — the other half of "
                      "probe_robot, which only answers the questions you think to ask. `assertions` is a list of "
                      "{kind, a, b?, max_m?, min_m?, reason?}; kind:'list' returns the vocabulary. persist:true "
                      "stores them on the robot so a later edit_robot re-checks the ORIGINAL intent.",
                      "heavy": True, "handler": assert_design, "parameters": {"type": "object",
                      "required": ["robot_id"], "properties": {"robot_id": {"type": "string"},
                      "assertions": {"type": "array", "items": {"type": "object"},
                                     "description": "list of {kind, a, b?, max_m?, min_m?, reason?}"},
                      "persist": {"type": "boolean", "default": False},
                      "kind": {"type": "string", "enum": ["list"],
                               "description": "single-verb shortcut: 'list' returns the assertion vocabulary"}}}},
    "edit_ops": {"description": "Discover the typed LOCALIZED edit operators (scale_group/set_height/set_material/"
                 "set_leg_count) and their args.", "heavy": False, "handler": edit_ops,
                 "parameters": {"type": "object", "properties": {}}},
    "scope_amend": {"description": "DRY-RUN an amend: what it would touch (`editable`), what it claims to leave "
                    "alone (`preserved`), and which established facts it invalidates — WITHOUT editing anything, "
                    "so the change can be shown or refused before it commits. Takes the same ops as edit_robot.",
                    "heavy": False, "handler": scope_amend, "parameters": {"type": "object",
                    "required": ["robot_id"], "properties": {"robot_id": {"type": "string"},
                    "ops": {"type": "array", "items": {"type": "object"}, "description": "the edit you are "
                            "considering, as [{op, args}] — the same shape edit_robot takes"},
                    "op": {"type": "string", "description": "single-op shortcut instead of ops"},
                    "args": {"type": "object", "description": "args for the single-op shortcut"}}}},
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
    "render_view": {"description": "Render a held robot to a PNG (an agent should SEE what it built). "
                    "view='visual' (default) draws the detailed surface; view='collision' draws the EXACT bodies "
                    "the physics verdict is computed on. Returns `path` — an ABSOLUTE path to a PNG that exists on "
                    "disk — plus a measured render/sim parity read; on failure ok=false and `error` says why.",
                    "heavy": True, "handler": render_view, "parameters": {"type": "object", "required": ["robot_id"],
                    "properties": {"robot_id": {"type": "string"}, "azimuth": {"type": "number"},
                                   "elevation": {"type": "number"},
                                   "view": {"type": "string", "enum": ["visual", "collision"]}}}},
    "simulate_gait": {"description": "Run the general scripted gait on a held robot; returns the HONEST verdict "
                      "(survived/cadence/support/upright/forward). Real MuJoCo.", "heavy": True, "handler": simulate_gait,
                      "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                          "robot_id": {"type": "string"}, "steps": {"type": "integer"}, "render": {"type": "boolean"}}}},
    "verify_robot": {"description": "The anti-hallucination gate: honest gait metrics + verdict + a GIF, so a walk "
                     "is never claimed without traces. On an IMPORTED robot driven by our generic gait it returns "
                     "NO LOCOMOTION VERDICT — we have your body, not your controller — plus the facts measurable "
                     "without one (stands at your declared home pose, joint limits, self-collision, torque "
                     "margins) and the route to a real verdict.", "heavy": True, "handler": verify_robot,
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
