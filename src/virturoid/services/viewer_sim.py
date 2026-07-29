"""Produce 3D-viewer data: geometry metadata + per-frame world poses (Phase UI).

Runs one real pick-and-place episode while recording every geom's world pose per
frame, plus static geom metadata (type/size/color), so the browser can render and
play back the robot actually sorting blocks — faithful to the physics, with a
simple frontend (no in-browser kinematics).
"""

from __future__ import annotations

import json
from pathlib import Path

_GEOM_TYPE = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}


def _locomotion_replay_model(package_dir: Path) -> tuple[Path | None, dict]:
    """Return the best visual replay model and public mesh metadata for a package.

    New gene packages carry a detailed model plus portable STL URIs.  Older packages retain the primitive
    ``robot_only.xml`` fallback, so adding visual fidelity never invalidates an existing export.
    """
    visual = package_dir / "simulation" / "robot_visual.xml"
    index_path = package_dir / "simulation" / "viewer_mesh_index.json"
    if visual.exists() and index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("model_uri") == "simulation/robot_visual.xml":
                meshes = index.get("meshes")
                if isinstance(meshes, dict):
                    return visual, meshes
        except Exception:  # noqa: BLE001 - fall back to a valid primitive-only package
            pass
    robot_only = package_dir / "simulation" / "robot_only.xml"
    return (robot_only if robot_only.exists() else None), {}


def _package_relative_uri(package_dir: Path, path: Path) -> str:
    return path.relative_to(package_dir).as_posix()


def simulate_episode_for_viewer(package_dir: Path, scene_set_uri: str = "simulation/scene_set.json", scene_index: int = 0) -> dict:
    import mujoco

    package_dir = Path(package_dir)
    compiled = json.loads((package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8"))
    by_scene = {e["scene_id"]: e["mujoco_xml"] for e in compiled.get("scenes", [])}
    scene_set = json.loads((package_dir / scene_set_uri).read_text(encoding="utf-8"))
    scene = scene_set["scenes"][min(scene_index, len(scene_set["scenes"]) - 1)]
    scene_model_path = package_dir / by_scene[scene["id"]]
    model = mujoco.MjModel.from_xml_path(str(scene_model_path))
    replay_model_uri = _package_relative_uri(package_dir, scene_model_path)

    geoms = _geom_metadata(model)
    frames: list = []
    purpose = scene_set.get("purpose")
    is_locomotion = purpose == "locomotion"
    is_navigation = (not is_locomotion) and (purpose == "navigation"
                                             or any(o["name"] == "goal_zone" for o in scene["objects"]))

    if is_locomotion:
        # A walker replays its GAIT, not a pick/nav script. Replay on the robot's OWN model when the build saved
        # it (simulation/robot_only.xml): the compiled SCENE model reorders geoms/joints, which mismaps the
        # per-joint gait into a lurch, while the bare robot model matches what the gait was trained on -> it walks.
        replay_model, mesh_uris = _locomotion_replay_model(package_dir)
        if replay_model is not None:
            try:
                model = mujoco.MjModel.from_xml_path(str(replay_model))
                geoms = _geom_metadata(model, mesh_uris)
                replay_model_uri = _package_relative_uri(package_dir, replay_model)
            except Exception:  # noqa: BLE001 - an optional visual asset must not break a valid primitive replay
                robot_only = package_dir / "simulation" / "robot_only.xml"
                if replay_model != robot_only and robot_only.exists():
                    model = mujoco.MjModel.from_xml_path(str(robot_only))
                    geoms = _geom_metadata(model)
                    replay_model_uri = _package_relative_uri(package_dir, robot_only)
        # M1/#212: if the build saved the WINNING gait's qpos trace (the walkable crawl beat the trot), replay
        # THAT on this model so the viewport shows the same forward walk the headline scored -- not the trot's
        # drift. Guarded: only when the trace's nq matches THIS model (else the per-joint mapping would be wrong),
        # and it falls through to the live rollout on any mismatch/error.
        outcome = _replay_locomotion_qpos(model, package_dir, record_frames=frames)
        if outcome is None:
            outcome = _locomotion_episode(model, package_dir, record_frames=frames)
    elif is_navigation:
        from virturoid.services.navigation_controller import run_navigation_episode

        goal = next((o["pose_xyz_rpy"][:2] for o in scene["objects"] if o["name"] == "goal_zone"), [1.0, 0.0])
        obstacles = [o["pose_xyz_rpy"][:2] for o in scene["objects"] if o.get("object_type") == "obstacle"]
        try:
            nav = run_navigation_episode(model, goal, obstacles, record_frames=frames, frame_every=20)
            outcome = {"status": nav["status"], "failure_label": None if nav["status"] == "reached" else nav["status"],
                       "placed_count": 1 if nav["status"] == "reached" else 0, "block_count": 1}
        except Exception as exc:  # noqa: BLE001 - nav episode failed -> the settle fallback below keeps motion
            outcome = {"status": "render_fallback", "failure_label": str(exc)[:80],
                       "placed_count": 0, "block_count": 1}
    else:
        # Prefer a REAL contact-grasp sort replay the build recorded (gripper holds each block by CONTACT FRICTION,
        # NO _pin_block) -- so the demo shows actual physics, matching the "nothing is teleported" claim. Fall back
        # to the idealized-pin episode for older packages with no replay, or when a different scene is requested.
        rec = None
        contact_ep = package_dir / "simulation" / "contact_episode.json"
        if contact_ep.exists():
            try:
                cand = json.loads(contact_ep.read_text(encoding="utf-8"))
                if cand.get("scene_id") == scene["id"] and cand.get("frames"):
                    rec = cand
            except Exception:  # noqa: BLE001 - a malformed replay just falls back to the pin episode
                rec = None
        if rec is not None:
            geoms = rec.get("geoms", geoms)
            frames = rec["frames"]
            outcome = rec.get("outcome", {"status": "success", "failure_label": None,
                                          "placed_count": 0, "block_count": 0})
        else:
            from virturoid.services.pick_place_controller import run_pick_place_episode

            objects = {o["name"]: tuple(o["pose_xyz_rpy"][:3]) for o in scene["objects"]}
            # Replay the SAME task that was built/evaluated (object->target assignments), if present.
            ep_scene = {"objects": objects}
            if scene_set.get("assignments"):
                ep_scene["assignments"] = scene_set["assignments"]
            try:
                out = run_pick_place_episode(model, ep_scene, record_frames=frames, frame_every=10)
                outcome = {"status": out["status"], "failure_label": out["failure_label"],
                           "placed_count": out["placed_count"], "block_count": out["block_count"]}
            except Exception as exc:  # noqa: BLE001 - pick-place failed -> the settle fallback below keeps motion
                outcome = {"status": "render_fallback", "failure_label": str(exc)[:80],
                           "placed_count": 0, "block_count": 0}

    if not frames:  # GUARANTEE motion: the task episode produced no frames (failed/empty) -> settle under gravity
        _settle_episode(model, record_frames=frames)
        if isinstance(outcome, dict):
            outcome.setdefault("note", "settling fallback — the task episode produced no frames to replay")

    return {
        "scene_id": scene["id"],
        "purpose": scene_set.get("purpose"),
        "task": "locomotion" if is_locomotion else (
            "navigation" if is_navigation else (scene_set.get("task_type") or "pick_place_sort")),
        "geoms": geoms,
        "frames": frames,
        "frame_count": len(frames),
        "outcome": outcome,
        # Transparency: exactly what the software produced + which scene this episode is.
        "robot": _robot_summary(package_dir),
        "scene": _scene_summary(scene),
        "scene_index": min(scene_index, len(scene_set["scenes"]) - 1),
        "replay_model_uri": replay_model_uri,
        "scenes": [{"index": i, "id": s["id"], "name": s.get("name", s["id"])}
                   for i, s in enumerate(scene_set["scenes"])],
    }


def _load_locomotion_policy(package_dir: Path, feature_dim: int, models_dir: str = "models"):
    """Find a banked MorphPolicy for this package's species whose encoding matches (so the replay shows the
    LEARNED gait). Recipe policies carry an obs normalizer (obs_mean) the caller uses to pick recipe control."""
    from virturoid.services.morph_policy import MorphPolicy

    species = None
    try:
        genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
        species = genome.get("species")
    except Exception:  # noqa: BLE001
        species = None
    for cand in ([Path(models_dir) / ("learned_" + str(species).replace("/", "_") + ".npz")] if species else []):
        if cand.exists():
            try:
                pol = MorphPolicy.from_npz(str(cand))
                if pol.accepts_feature_dim(feature_dim):
                    return pol
            except Exception:  # noqa: BLE001
                pass
    return None


def _replay_locomotion_qpos(model, package_dir: Path, *, record_frames: list):
    """M1/#212: replay the build's saved WINNING-gait qpos trace (simulation/locomotion_qpos.json) on ``model``,
    recording geom frames, so the viewport shows the exact forward walk the build headline scored. Returns an
    outcome dict, or None to signal the caller to fall back to the live rollout -- when no trace exists, its nq
    doesn't match THIS model (the per-joint mapping would be wrong), or anything goes sideways."""
    import mujoco

    from virturoid.services.pick_place_controller import _capture_geom_frame

    trace_path = package_dir / "simulation" / "locomotion_qpos.json"
    if not trace_path.exists():
        return None
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        qpos_frames = trace.get("qpos_frames") or []
        if not qpos_frames or int(trace.get("nq", -1)) != int(model.nq):
            return None                                       # shape mismatch -> live rollout keeps the mapping right
        data = mujoco.MjData(model)
        for qp in qpos_frames:
            data.qpos[:] = qp
            mujoco.mj_forward(model, data)
            record_frames.append(_capture_geom_frame(data, model))
        if not record_frames:
            return None
        return {"status": "walked", "failure_label": None, "placed_count": 0, "block_count": 0,
                "forward_m": float(trace.get("forward_m", 0.0)), "note": "replay of the scored gait"}
    except Exception:  # noqa: BLE001 - a malformed/absent trace just falls back to the live rollout
        return None


def _locomotion_episode(model, package_dir: Path, *, record_frames: list, steps: int = 900,
                        frame_every: int = 6, models_dir: str = "models") -> dict:
    """Replay a walker's forward gait, recording geom poses per frame. Uses the banked policy for this species
    (recipe control — PD-to-default + obs-norm — when it carries a normalizer; otherwise a bounded residual on
    gravity comp). Returns the same outcome shape the other tasks use, plus forward travel + uprightness."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import MorphPolicy
    from virturoid.services.pick_place_controller import _capture_geom_frame

    gr = encode_robot(model)
    data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    bq = gr.base_qadr
    z0 = float(data.qpos[bq + 2]) if bq >= 0 else 1.0
    x0 = float(data.qpos[bq]) if bq >= 0 else 0.0
    banked = _load_locomotion_policy(package_dir, gr.feature_dim, models_dir)
    pol = banked if banked is not None else MorphPolicy(gr.feature_dim, seed=0)
    has_policy = banked is not None                  # a real banked policy -> apply its LEARNED residual (propulsion)
    normalizer = getattr(pol, "obs_mean", None) is not None  # a recipe policy MAY also carry an obs normalizer
    qadr = np.asarray(gr.qadr, int); vadr = np.asarray(gr.vadr, int); act_u = np.asarray(gr.act_u, int)
    clamps = np.asarray(gr.clamps, float); q_def = np.array([float(data.qpos[a]) for a in qadr])
    # Drive the replay with the SAME per-joint gains the policy was trained under (adaptive -> inertia-scaled),
    # else the scalar recipe — matches recipe_rollout_morph / rollout_view so the build replay == the trained gait.
    if getattr(pol, "adaptive_gains", False):
        from virturoid.services.morph_policy import recipe_gains
        kp_v, kd_v, as_v = recipe_gains(model, gr)
    else:                                            # scalar recipe — share the source-of-truth constants with training
        from virturoid.services.morph_policy import _ASCALE, _KD_REF, _KP_REF
        kp_v = np.full(gr.n_tokens, _KP_REF); kd_v = np.full(gr.n_tokens, _KD_REF); as_v = np.full(gr.n_tokens, _ASCALE)
    # TROT-CPG prior: the policy's own when CPG-trained, else the default. This feed-forward stepping rhythm is what
    # makes the body WALK in the replay, so a quad with no exactly-matching banked policy still shows a real upright
    # gait (bare CPG walks) instead of a random-policy collapse. Same control law as recipe_rollout_morph.
    from virturoid.services.morph_policy import CPG_DEFAULT, _trot_cpg_tokens
    cpg = getattr(pol, "cpg", None) or CPG_DEFAULT
    cpg_amp, cpg_phase, cpg_gate = _trot_cpg_tokens(model, gr, cpg)
    cpg_freq = float(cpg["freq"]); res_scale = float(cpg.get("residual_scale", 0.3))
    two_pi = 2.0 * np.pi; dt = float(model.opt.timestep)
    for t in range(steps):
        obs = pol.adapt_observation(gr.observe(model, data))
        a = (pol.act((obs - pol.obs_mean) / pol.obs_std) if normalizer else pol.act(obs)) if has_policy else None
        cphase = two_pi * cpg_freq * t * dt if cpg_gate else 0.0
        for k in range(gr.n_tokens):
            cpg_off = float(cpg_amp[k] * np.sin(cphase + cpg_phase[k])) if cpg_gate else 0.0
            resid = res_scale * as_v[k] * float(np.tanh(a[k])) if has_policy else 0.0
            tgt = q_def[k] + cpg_off + resid
            tau = kp_v[k] * (tgt - float(data.qpos[qadr[k]])) - kd_v[k] * float(data.qvel[vadr[k]])
            data.ctrl[act_u[k]] = float(np.clip(tau, -clamps[k], clamps[k]))
        mujoco.mj_step(model, data)
        if t % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))
        if not np.all(np.isfinite(data.qpos)):
            break
    forward = float(data.qpos[bq] - x0) if bq >= 0 else 0.0
    upright = (float(data.qpos[bq + 2]) > 0.5 * z0) if bq >= 0 else True
    status = "walked" if (forward > 0.15 and upright) else ("upright" if upright else "fell")
    return {"status": status, "failure_label": None if status == "walked" else status,
            "placed_count": 1 if status == "walked" else 0, "block_count": 1,
            "forward_m": round(forward, 3), "upright": bool(upright), "learned": has_policy}


def _settle_episode(model, *, record_frames: list, steps: int = 180, frame_every: int = 6) -> None:
    """Fallback replay: let the robot settle under gravity (no control), recording geom poses per frame. This
    GUARANTEES the viewer always has SOME motion to play even when the task-specific episode failed or produced
    no frames -- so a build is never a frozen, motionless scene in the viewport."""
    import mujoco
    import numpy as np

    from virturoid.services.pick_place_controller import _capture_geom_frame

    data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    for t in range(steps):
        mujoco.mj_step(model, data)
        if t % frame_every == 0:
            record_frames.append(_capture_geom_frame(data, model))
        if not np.all(np.isfinite(data.qpos)):
            break


def _camera_for_episode_frame(model, frame, *, padding: float = 2.2) -> tuple[list[float], float]:
    """Fit an orbit camera to the recorded non-plane geometry in an episode frame.

    Replay assets span a tabletop arm, a floor-scale rover, and a standing walker.  A fixed arm camera makes the
    latter appear as a tiny off-screen dot; bounds from the actual recorded poses keep the render readable without
    inventing a separate camera rule for every robot class.
    """
    import mujoco
    import numpy as np

    lows, highs = [], []
    for geom_id, pose in enumerate(frame):
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        radius = max(0.01, float(model.geom_rbound[geom_id]))
        pos = np.asarray(pose[:3], dtype=float)
        lows.append(pos - radius)
        highs.append(pos + radius)
    if not lows:
        return [0.4, 0.0, 0.15], 1.5
    lo, hi = np.min(np.asarray(lows), axis=0), np.max(np.asarray(highs), axis=0)
    center = (lo + hi) / 2.0
    extent = float(np.max(hi - lo))
    return [float(v) for v in center], max(0.45, float(padding) * extent)


def render_episode_frames(package_dir: Path, scene_index: int = 0, width: int = 720, height: int = 540,
                          azimuth: float = 135.0, elevation: float = -20.0, distance: float | None = None):
    """Render the recorded episode to a list of RGB numpy frames using MuJoCo's renderer.

    Reuses ``simulate_episode_for_viewer`` (which records every geom's world pose per frame) and
    replays it straight into ``mjv_updateScene`` by setting ``data.geom_xpos``/``geom_xmat`` — so
    the native desktop viewport shows the *actual* MuJoCo render of the same episode the web viewer
    plays, with no re-simulation and no qpos needed. Returns ``(view_dict, [HxWx3 uint8, ...])``.
    Needs MuJoCo with a working offscreen GL context.
    """
    import mujoco
    import numpy as np

    package_dir = Path(package_dir)
    view = simulate_episode_for_viewer(package_dir, scene_index=scene_index)
    # Re-open the exact model that simulated the recorded frames.  This matters for detailed visual meshes:
    # their extra geoms change the frame array length even though they do not affect collision physics.
    model_uri = view.get("replay_model_uri")
    if not model_uri:
        compiled = json.loads((package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8"))
        by_scene = {e["scene_id"]: e["mujoco_xml"] for e in compiled.get("scenes", [])}
        model_uri = by_scene[view["scene_id"]]
    model = mujoco.MjModel.from_xml_path(str(package_dir / str(model_uri)))
    renderer = mujoco.Renderer(model, height, width)
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    lookat, auto_distance = _camera_for_episode_frame(model, view.get("frames", [[]])[0])
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = (auto_distance if distance is None else distance), azimuth, elevation
    mat = np.zeros(9)
    images = []
    try:
        for frame in view["frames"]:
            for g, pose in enumerate(frame):
                data.geom_xpos[g] = pose[:3]
                mujoco.mju_quat2Mat(mat, np.asarray(pose[3:7], dtype=float))
                data.geom_xmat[g] = mat
            renderer.update_scene(data, cam)
            images.append(renderer.render().copy())
    finally:
        renderer.close()
    return view, images


def _robot_summary(package_dir: Path) -> dict:
    """A human-readable spec of the robot the software actually built (from the genome)."""
    def _load(rel: str) -> dict:
        p = package_dir / rel
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:  # noqa: BLE001
            return {}

    genome = _load("robot/robot_genome.json")
    autonomy = _load("reports/autonomy_report.json")
    contract = _load("reports/robot_package_contract.json")

    joints = [
        {"name": j.get("name"), "type": j.get("joint_type"),
         "parent": j.get("parent_link"), "child": j.get("child_link"),
         "actuator": j.get("actuator_component_id")}
        for j in genome.get("joints", [])
    ]
    species = autonomy.get("species") or contract.get("species") or genome.get("species")
    robot_class = autonomy.get("robot_class") or contract.get("robot_class") or _class_from_species(species)
    return {
        "name": genome.get("name") or species or "robot",
        "robot_class": robot_class,
        "species": species,
        "requested_species": autonomy.get("requested_species") or "",
        "requested_robot_class": autonomy.get("requested_robot_class") or "",
        "links": genome.get("links", []),
        "joints": joints,
        "end_effectors": genome.get("end_effectors", []),
        "sensors": [s.get("sensor_component_id") for s in genome.get("sensors", [])],
        "converged_design": autonomy.get("converged_design", {}),
        "task_type": autonomy.get("task_type") or contract.get("task_type"),
    }


def _class_from_species(species: str | None) -> str | None:
    """Infer the robot class from a species pattern when no report/contract is present."""
    if not species:
        return None
    root = species.split(".", 1)[0]
    return {
        "fixed_arm": "manipulator", "dual_arm": "manipulator",
        "mobile_base": "mobile_base", "mobile_robot": "mobile_base", "mobile_manipulator": "mobile_base",
        "humanoid": "humanoid", "legged_robot": "quadruped", "quadruped": "quadruped",
    }.get(root, root)


def _scene_summary(scene: dict) -> dict:
    """A human-readable spec of the scene the software generated for this episode."""
    objects = [
        {"name": o.get("name"), "type": o.get("object_type"),
         "pos": [round(float(v), 3) for v in o.get("pose_xyz_rpy", [0, 0, 0])[:3]]}
        for o in scene.get("objects", [])
    ]
    return {
        "id": scene.get("id"),
        "name": scene.get("name", scene.get("id")),
        "object_count": len(objects),
        "objects": objects,
    }


def _geom_metadata(model, mesh_uris: dict | None = None) -> list:
    import mujoco
    import numpy as np

    out = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        rgba = [round(float(v), 3) for v in model.geom_rgba[gid]]
        # Resolve material color when the geom uses a material instead of rgba.
        matid = int(model.geom_matid[gid])
        if matid >= 0:
            rgba = [round(float(v), 3) for v in model.mat_rgba[matid]]
        item = {
            "name": name,
            "type": _GEOM_TYPE.get(int(model.geom_type[gid]), "box"),
            "size": [round(float(v), 4) for v in model.geom_size[gid]],
            "rgba": rgba,
            "metalness": 0.08,
            "roughness": 0.68,
        }
        if matid >= 0:
            # MuJoCo 3 exposes native PBR values.  Older/generated models generally use
            # specular + shininess, so translate those conservatively instead of making
            # every part look like the same grey plastic in Studio.
            if hasattr(model, "mat_metallic"):
                item["metalness"] = round(float(model.mat_metallic[matid]), 3)
            else:
                item["metalness"] = round(0.55 * float(model.mat_specular[matid]), 3)
            if hasattr(model, "mat_roughness"):
                item["roughness"] = round(float(model.mat_roughness[matid]), 3)
            else:
                item["roughness"] = round(max(0.12, 1.0 - float(model.mat_shininess[matid])), 3)
        mesh = (mesh_uris or {}).get(name)
        if item["type"] == "mesh" and isinstance(mesh, dict) and isinstance(mesh.get("uri"), str):
            item["mesh_uri"] = mesh["uri"]
            item["mesh_scale"] = float(mesh.get("scale", 0.001))
        out.append(item)
    return out
