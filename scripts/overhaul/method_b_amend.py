"""METHOD B prototype -- REAL-MODEL AMENDMENT.

Real Menagerie models are fixed designs; customers want VARIANTS while keeping production-grade fidelity
(real meshes, real inertias, real actuators). This self-contained experiment customizes a real model at
the MJCF / MjSpec level and renders before/after, then VERIFIES each result compiles + stands in MuJoCo.

Three amendment kinds are prototyped:
  (1) SCALE      -- "a larger quadruped": uniformly scale Unitree Go1 (meshes, body offsets, geom sizes,
                    inertials, spawn height). Fidelity preserved because every mesh keeps its detail; we
                    only change the global metric size.
  (2) SWAP EE    -- "a Panda with a Robotiq gripper": detach Panda's built-in 2-finger hand, attach the
                    Robotiq 2F-85 (a *separate* Menagerie model) at the arm tip via mujoco.MjSpec.attach,
                    which auto-namespaces assets/defaults/actuators so two real models compose cleanly.
  (3) RECOLOR    -- "an orange Go1": rewrite material rgba only. Geometry/physics untouched -> always safe.

Nothing in shared src/ is edited. This is an experiment to compare against the other overhaul methods.

    PYTHONPATH=src python scripts/overhaul/method_b_amend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "build" / "overhaul" / "amend"
OUT.mkdir(parents=True, exist_ok=True)

from virturoid.services.real_model_library import load_real_model  # noqa: E402


# --------------------------------------------------------------------------------------------------
# rendering -- 4-view contact sheet, brightened headlight (Menagerie materials render dark by default)
# --------------------------------------------------------------------------------------------------
def render(model: mujoco.MjModel, out: Path, size: int = 460, pose_home: bool = True) -> dict:
    """Render a compiled model from 4 angles into one PNG. Returns standing/compile diagnostics."""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), size)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), size)
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]
    data = mujoco.MjData(model)
    if pose_home and model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    pts = data.geom_xpos[: model.ngeom]
    center = pts.mean(axis=0)
    extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
    renderer = mujoco.Renderer(model, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = max(0.6, extent * 2.3)
    shots = []
    for az, el in [(90, -12), (45, -20), (140, -22), (0, -85)]:
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(data, cam)
        shots.append(renderer.render())
    top = np.concatenate(shots[:2], axis=1)
    bot = np.concatenate(shots[2:], axis=1)
    Image.fromarray(np.concatenate([top, bot], axis=0)).save(out)
    renderer.close()
    return {"png": str(out), "extent_m": round(extent, 3)}


def _with_floor(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Add a ground plane if the model has none (raw Menagerie models ship floorless) so the standing
    check is meaningful -- without a floor a free-base locomotor just free-falls forever."""
    spec = spec.copy()
    has_plane = any(int(g.type) == int(mujoco.mjtGeom.mjGEOM_PLANE) for g in spec.geoms)
    if not has_plane:
        spec.worldbody.add_geom(name="_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[8, 8, 0.1], pos=[0, 0, 0])
    return spec


def stands(spec_or_model, settle_s: float = 1.5) -> dict:
    """Drop the model at its home pose, hold ctrl at the keyframe, and check it doesn't collapse/explode.
    Accepts an MjSpec (preferred -- a floor is injected) or a compiled MjModel (used as-is). Returns
    base-height drift + whether qpos stayed finite -- a cheap 'does this real-physics body stand'."""
    if isinstance(spec_or_model, mujoco.MjSpec):
        model = _with_floor(spec_or_model).compile()
    else:
        model = spec_or_model
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        if model.nu and model.key_ctrl.shape[1] >= model.nu:
            data.ctrl[:] = model.key_ctrl[0][: model.nu]
    else:
        mujoco.mj_resetData(model, data)
    # free-base z (locomotors) -- first free joint's qpos[2]
    z_adr = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            z_adr = int(model.jnt_qposadr[j]) + 2
            break
    z0 = float(data.qpos[z_adr]) if z_adr is not None else None
    steps = int(settle_s / model.opt.timestep)
    exploded = False
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            exploded = True
            break
    z1 = float(data.qpos[z_adr]) if z_adr is not None else None
    drift = round(abs(z1 - z0), 4) if z0 is not None else None
    upright = (drift is not None and drift < 0.15) if z0 is not None else True
    return {"exploded": exploded, "z0": z0, "z1": z1, "drift_m": drift,
            "upright_or_fixed_base": (not exploded) and upright}


# --------------------------------------------------------------------------------------------------
# (1) UNIFORM SCALE -- scale every spatial quantity of a real model by `factor`.
# --------------------------------------------------------------------------------------------------
def scale_model(info: dict, factor: float) -> mujoco.MjSpec:
    """Uniformly scale a real model via MjSpec. Scaling rules (length L, so volume ~L^3):
        body.pos, geom.pos/size/fromto, site.pos/size, mesh.scale  -> * factor
        inertial mass -> * factor^3 ; diaginertia/fullinertia -> * factor^5
        free-joint spawn height in every keyframe qpos[2]          -> * factor
    Meshes keep all their geometric detail, so fidelity is preserved -- only the metric size changes."""
    spec = mujoco.MjSpec.from_file(info["path"])
    f = float(factor)

    for g in spec.geoms:
        g.pos = np.asarray(g.pos) * f
        g.size = np.asarray(g.size) * f
        ft = np.asarray(g.fromto)
        if np.any(ft):                       # capsule/cylinder defined by endpoints
            g.fromto = ft * f
    for b in spec.bodies:
        b.pos = np.asarray(b.pos) * f
        b.mass = float(b.mass) * f**3
        b.inertia = np.asarray(b.inertia) * f**5
        b.ipos = np.asarray(b.ipos) * f
    for s in spec.sites:
        s.pos = np.asarray(s.pos) * f
        s.size = np.asarray(s.size) * f
    for m in spec.meshes:
        m.scale = np.asarray(m.scale) * f
    # scale the free-base spawn height in each keyframe so the larger body still spawns standing
    for k in spec.keys:
        q = np.asarray(k.qpos, dtype=float)
        if q.size >= 3:
            q[2] *= f                        # qpos[2] is the free-joint base height for these locomotors
            k.qpos = q
    return spec


# --------------------------------------------------------------------------------------------------
# (2) END-EFFECTOR SWAP -- detach Panda's hand, attach a Robotiq 2F-85 (separate real model) at the tip.
# --------------------------------------------------------------------------------------------------
def swap_gripper(arm_info: dict, gripper_info: dict) -> mujoco.MjSpec:
    """Compose two REAL Menagerie models: remove the Panda's integrated 2-finger hand and bolt a Robotiq
    2F-85 onto link7. Uses MjSpec.attach, which copies the gripper's meshes/materials/defaults/actuators
    with a 'rq_' prefix so nothing collides with the arm. Result is two production models, one robot."""
    arm = mujoco.MjSpec.from_file(arm_info["path"])

    # 1) delete the Panda's built-in hand subtree + the finger tendon/equality/actuator that drive it.
    hand = arm.body("hand")
    tip = hand.parent                         # link7 -- where the original hand mounted (pos 0 0 0.107)
    hand_offset = np.asarray(hand.pos).copy() # keep the same mount point/orientation as the OEM hand
    hand_quat = np.asarray(hand.quat).copy()
    arm.delete(hand)                          # recursively removes left/right finger bodies too
    for getter in ("tendon", "equality", "actuator"):
        for name in ("split", "actuator8"):
            try:
                arm.delete(getattr(arm, getter)(name))
            except Exception:                 # noqa: BLE001 - element may not exist under this name
                pass

    # 2) add a mounting frame on link7 at the OEM hand's pose, then attach the gripper spec there.
    gripper = mujoco.MjSpec.from_file(gripper_info["path"])
    frame = tip.add_frame()
    frame.pos = hand_offset
    frame.quat = hand_quat
    arm.attach(gripper, prefix="rq_", frame=frame)
    return arm


# --------------------------------------------------------------------------------------------------
# (3) RECOLOR -- rewrite a named material's rgba; geometry + physics untouched.
# --------------------------------------------------------------------------------------------------
def recolor(info: dict, material_rgba: dict[str, tuple]) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(info["path"])
    by_name = {m.name: m for m in spec.materials}
    for name, rgba in material_rgba.items():
        if name in by_name:
            by_name[name].rgba = np.asarray(rgba, dtype=float)
    return spec


def compile_report(spec: mujoco.MjSpec, label: str) -> tuple[mujoco.MjModel | None, dict]:
    try:
        model = spec.compile()
    except Exception as exc:  # noqa: BLE001
        return None, {"label": label, "compiled": False, "error": str(exc)[:300]}
    rep = {"label": label, "compiled": True, "bodies": int(model.nbody) - 1,
           "actuators": int(model.nu), "meshes": int(model.nmesh), "geoms": int(model.ngeom)}
    return model, rep


def main() -> int:
    results: list[dict] = []

    # ---- BEFORE baselines -----------------------------------------------------------------------
    go1 = load_real_model("a quadruped")
    panda = load_real_model("a panda arm")
    robotiq = load_real_model("a robotiq gripper")
    assert go1["ok"] and panda["ok"] and robotiq["ok"], (go1, panda, robotiq)

    go1_spec = mujoco.MjSpec.from_file(go1["path"])
    m = go1_spec.compile()
    r = render(m, OUT / "go1_before.png"); st = stands(go1_spec)
    print("BEFORE go1:", r, st)
    results.append({"amend": "scale", "stage": "before", **r, **st})

    panda_spec = mujoco.MjSpec.from_file(panda["path"])
    m = panda_spec.compile()
    r = render(m, OUT / "panda_before.png"); st = stands(panda_spec)
    print("BEFORE panda:", r, st)
    results.append({"amend": "swap", "stage": "before", **r, **st})

    # ---- (1) SCALE: 1.6x larger Go1 -------------------------------------------------------------
    spec = scale_model(go1, 1.6)
    model, rep = compile_report(spec, "go1_scaled_1.6x")
    print("SCALE compile:", rep)
    if model is not None:
        st = stands(spec)
        r = render(model, OUT / "go1_scaled_1.6x.png")
        print("SCALE render/stand:", r, st)
        results.append({"amend": "scale", "stage": "after", "factor": 1.6, **rep, **r, **st})

    # ---- (2) SWAP: Panda + Robotiq 2F-85 --------------------------------------------------------
    spec = swap_gripper(panda, robotiq)
    model, rep = compile_report(spec, "panda_robotiq2f85")
    print("SWAP compile:", rep)
    if model is not None:
        st = stands(spec)
        r = render(model, OUT / "panda_robotiq2f85.png")
        print("SWAP render/stand:", r, st)
        results.append({"amend": "swap", "stage": "after", **rep, **r, **st})

    # ---- (3) RECOLOR: orange Go1 ----------------------------------------------------------------
    spec = recolor(go1, {"dark": (0.85, 0.35, 0.05, 1.0)})
    model, rep = compile_report(spec, "go1_orange")
    print("RECOLOR compile:", rep)
    if model is not None:
        st = stands(spec)
        r = render(model, OUT / "go1_orange.png")
        print("RECOLOR render/stand:", r, st)
        results.append({"amend": "recolor", "stage": "after", **rep, **r, **st})

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
