"""Morphology-agnostic, TRANSFERABLE contact grasp — POC-A (the manipulation-moat frontier).

The GPU residual grasp (scripts/mjx_residual_grasp.py) is a flat MLP keyed to ONE arm's (obs_dim, nu) with a
hardcoded 2-finger contact reward — it cannot transfer across manipulators. This learns a per-token
``MorphPolicy`` RESIDUAL on the proven scripted top-down grasp base (``grasp_eval``): the policy sees the
per-joint morph-graph tokens + the object pose (relative to the TCP, via the perception channel) and emits a
per-joint arm-target residual; fingers stay on the scripted open→close schedule (POC-A). Because the policy is
per-token (one attention net for any joint count) and the contact reward is GRAPH-DERIVED (fingers = slide
joints, count distinct finger bodies touching the object — works for 2/3/N fingers, no hardcoded geom ids),
the SAME residual transfers across DIFFERENT arms/grippers. Diagnostic motivation: scripted grasp is 1.00 on
the tabletop arm but only 0.25 on an 8-DOF arm — that gap is the learnable, transferable target.

POC-A holds the control law at the known-good recipe and changes only representation (flat MLP → graph
attention) + reward (2-finger → N-finger). POC-B (learned finger control) builds on a proven POC-A.
See [[task-effectiveness-loop]], [[morph-policy-theme1]] (the locomotion analog), grasp_eval.py (the base).
"""

from __future__ import annotations

_GOAL_SLOTS = 3   # object-offset packed into the policy's perception ring


def _pack(offset):
    import numpy as np
    r = np.zeros(8, dtype=float)
    r[:_GOAL_SLOTS] = np.asarray(offset, dtype=float).ravel()[:_GOAL_SLOTS]
    return r


def encode_grasp(mj, gene):
    """Graph-derived grasp handles for ANY manipulator: arm vs finger actuator slots (revolute vs slide),
    finger bodies+geoms (for N-finger contact), the object geom/qpos, the TCP site, and the alignment from
    morph-graph TOKEN order -> arm actuator slots (so a per-token residual maps onto the scripted arm targets)."""
    import mujoco

    from virturoid.services.morph_graph import encode_robot
    nu = int(mj.nu)
    j_of = [int(mj.actuator_trnid[u, 0]) for u in range(nu)]
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    arm_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) != slide]
    fin_u = [u for u in range(nu) if int(mj.jnt_type[j_of[u]]) == slide]
    finger_bodies = {int(mj.jnt_bodyid[j_of[u]]) for u in fin_u}
    finger_geoms = {g for g in range(mj.ngeom) if int(mj.geom_bodyid[g]) in finger_bodies}
    box_geom = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    box_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box"))
    bq = int(mj.jnt_qposadr[box_jid]); box_body = int(mj.jnt_bodyid[box_jid])
    tcp = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    graph = encode_robot(mj)
    # token k drives actuator graph.act_u[k]; map each scripted arm slot -> its token (residual alignment)
    tok_of_act = {int(a): k for k, a in enumerate(graph.act_u)}
    arm_tok = [tok_of_act.get(u, -1) for u in arm_u]          # token index per arm actuator slot (-1 if none)
    fin_tok = [tok_of_act.get(u, -1) for u in fin_u]          # token index per finger actuator slot (POC-B)
    return {"nu": nu, "arm_u": arm_u, "fin_u": fin_u, "finger_bodies": finger_bodies,
            "finger_geoms": finger_geoms, "box_geom": box_geom, "bq": bq, "box_body": box_body,
            "tcp": tcp, "graph": graph, "arm_tok": arm_tok, "fin_tok": fin_tok}


_SETUP_CACHE: dict = {}


def _grasp_setup(gene, gx, gy, hover_z, lift_z):
    """Compile the arm+cube and solve the top-down IK poses ONCE per (body, object position) — cached, since
    they don't depend on the residual policy. Makes ES (thousands of rollouts) tractable: each rollout is then
    just a fresh MjData + the stepping loop. Returns everything the episode loop needs."""
    import mujoco
    import numpy as np

    from virturoid.schemas.scenes import SceneObject
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.grasp_eval import _set_grasp_contacts
    from virturoid.services.grasp_pose import arm_revolute_joints, best_top_down_pose, solve_top_down_grasp_pose

    cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=1.0, scale=1.0)
    xml = compile_gene_with_scene(gene, [cube])
    key = (xml, round(gx, 4), round(gy, 4), hover_z, lift_z)
    if key in _SETUP_CACHE:
        return _SETUP_CACHE[key]
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.opt.iterations = 30
    mj.opt.ls_iterations = 12
    h = encode_grasp(mj, gene)
    if h["tcp"] < 0 or h["box_geom"] < 0 or not h["arm_u"]:
        _SETUP_CACHE[key] = None
        return None
    nu, arm_u = h["nu"], h["arm_u"]
    j_of = [int(mj.actuator_trnid[u, 0]) for u in range(nu)]
    _set_grasp_contacts(mj, h["finger_bodies"], h["box_body"])
    arm_qadr = [int(mj.jnt_qposadr[j_of[u]]) for u in arm_u]
    arm_joints = arm_revolute_joints(mj)
    bq, tcp = h["bq"], h["tcp"]
    work = mujoco.MjData(mj); mujoco.mj_resetData(mj, work)
    palm = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, gene.end_effector().name))
    d0 = mujoco.MjData(mj); mujoco.mj_resetData(mj, d0); mujoco.mj_forward(mj, d0)
    box_z = float(d0.qpos[bq + 2])
    above_map, _tilt, _ = best_top_down_pose(mj, work, site_id=tcp, palm_body_id=palm,
                                             target=(gx, gy, hover_z), arm_joints=arm_joints, iters=250)

    def ik(z, seed_map):
        return solve_top_down_grasp_pose(mj, work, site_id=tcp, palm_body_id=palm, target=(gx, gy, z),
                                         arm_joints=arm_joints, seed=[seed_map[qa] for qa, _ in arm_joints], iters=250)

    at_map = ik(box_z, above_map); lift_map = ik(lift_z, at_map)
    setup = {
        "mj": mj, "h": h, "arm_joints": arm_joints, "arm_qadr": arm_qadr,
        "act_dof": np.array([int(mj.jnt_dofadr[j_of[u]]) for u in range(nu)]),
        "fr_lim": mj.actuator_forcerange[:, 1].copy(), "geom_body": {g: int(mj.geom_bodyid[g]) for g in h["finger_geoms"]},
        "q_above": np.array([above_map[qa] for qa, _ in arm_joints]),
        "q_at": np.array([at_map[qa] for qa, _ in arm_joints]),
        "q_lift": np.array([lift_map[qa] for qa, _ in arm_joints]),
    }
    _SETUP_CACHE[key] = setup
    return setup


def morph_grasp_rollout(gene, policy, gx, gy, *, residual_scale: float = 0.12, fin_residual_scale: float = 0.04,
                        ep_len: int = 260, kpj: float = 150.0, kdj: float = 18.0, kpf: float = 300.0,
                        kdf: float = 10.0, dq: float = 0.06, phases=(0.18, 0.5, 0.62), fclose: float = 0.05,
                        hover_z: float = 0.16, lift_z: float = 0.22, success_lift: float = 0.05,
                        record_frames=None, frame_every: int = 8) -> dict:
    """One grasp-lift with a per-token MorphPolicy RESIDUAL on the scripted arm targets. Mirrors
    grasp_eval.grasp_lift_attempt (same contact model + IK + phase-PD) so ``policy=None`` reproduces the
    scripted base; a policy adds ``residual_scale * tanh(act)`` (rad) to each arm joint target each step, fed
    the object offset (box - TCP) via the perception channel. Returns success/lift/contact + a dense reward."""
    import mujoco
    import numpy as np

    s = _grasp_setup(gene, gx, gy, hover_z, lift_z)
    if s is None:
        return {"success": False, "lifted": 0.0, "contacts": 0, "dense": -1.0, "reason": "no_grasp_handles"}
    mj, h = s["mj"], s["h"]
    nu, arm_u, fin_u = h["nu"], h["arm_u"], h["fin_u"]
    graph, arm_tok, fin_tok = h["graph"], h["arm_tok"], h["fin_tok"]
    box_geom, finger_geoms = h["box_geom"], h["finger_geoms"]
    geom_body = s["geom_body"]
    act_dof, fr_lim = s["act_dof"], s["fr_lim"]
    arm_qadr, arm_joints = s["arm_qadr"], s["arm_joints"]
    q_above, q_at, q_lift = s["q_above"], s["q_at"], s["q_lift"]
    KP = np.array([kpj if u in arm_u else kpf for u in range(nu)])
    KD = np.array([kdj if u in arm_u else kdf for u in range(nu)])
    bq, tcp = h["bq"], h["tcp"]

    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)
    for k in range(len(arm_joints)):
        d.qpos[arm_qadr[k]] = q_above[k]
    mujoco.mj_forward(mj, d)
    box_z0 = float(d.qpos[bq + 2]); q_ref = q_above.copy()
    p_settle, p_grip, p_lift = phases

    def contacts():
        bodies = set()
        for i in range(int(d.ncon)):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if box_geom in (g1, g2):
                for g in (g1, g2):
                    if g in finger_geoms:
                        bodies.add(geom_body[g])
        return len(bodies)

    max_both, grip_contact_steps = 0, 0
    for t in range(ep_len):
        frac = t / ep_len
        q_tgt = q_lift if frac >= p_lift else (q_above if frac < p_settle else q_at)
        q_ref = q_ref + np.clip(q_tgt - q_ref, -dq, dq)
        # --- per-token MorphPolicy residual (the LEARNED, transferable part): arm-pose refinement (POC-A) +
        #     finger close-amount control (POC-B, fixes a slipping grip). Fed the object offset (box - TCP). ---
        q_eff = q_ref.copy()
        f_close = fclose
        if policy is not None:
            box_pos = np.array(d.qpos[bq:bq + 3], dtype=float)
            act = policy.act(graph.observe(mj, d), ranges=_pack(box_pos - np.array(d.site_xpos[tcp], dtype=float)))
            for k in range(len(arm_joints)):
                tk = arm_tok[k] if k < len(arm_tok) else -1
                if 0 <= tk < len(act):
                    q_eff[k] = q_ref[k] + residual_scale * float(np.tanh(act[tk]))
            # fingers: close at least to fclose, up to fclose + fin_residual_scale (grip TIGHTER) — never looser
            ft = fin_tok[0] if fin_tok and 0 <= fin_tok[0] < len(act) else -1
            if ft >= 0:
                f_close = fclose + fin_residual_scale * 0.5 * (float(np.tanh(act[ft])) + 1.0)
        f_tgt = 0.0 if frac < p_grip else f_close
        q_star = np.zeros(nu)
        for k, u in enumerate(arm_u):
            q_star[u] = q_eff[k]
        for u in fin_u:
            q_star[u] = f_tgt
        q = d.qpos[act_dof]; qd = d.qvel[act_dof]
        d.ctrl[:] = np.clip(d.qfrc_bias[act_dof] + KP * (q_star - q) - KD * qd, -fr_lim, fr_lim)
        mujoco.mj_step(mj, d)
        if not np.all(np.isfinite(d.qpos)):
            return {"success": False, "lifted": 0.0, "contacts": 0, "dense": -1.0, "reason": "blew_up"}
        if record_frames is not None and t % frame_every == 0:
            from virturoid.services.pick_place_controller import _capture_geom_frame
            record_frames.append(_capture_geom_frame(d, mj))
        nc = contacts()
        if frac >= p_grip:
            max_both = max(max_both, nc)
            if nc >= 2:
                grip_contact_steps += 1
    lifted = float(d.qpos[bq + 2]) - box_z0
    d_tcp = float(np.linalg.norm(d.site_xpos[tcp] - d.qpos[bq:bq + 3]))
    success = bool(max_both >= 2 and lifted > success_lift and d_tcp < 0.08)
    # NON-GAMEABLE reward dominated by the REAL objective (FINAL sustained lift) + holding + success; the box
    # slipping away is penalized. (The earlier accumulated proximity+contact reward was reward-hacked: high
    # proxy, zero real lift.) grip_contact_steps rewards SUSTAINED (not transient) contact through the lift.
    reward = (8.0 * float(np.clip(lifted, 0.0, 0.12))
              + 0.3 * (grip_contact_steps / max(1, int(ep_len * (1 - phases[1]))))
              + (3.0 if success else 0.0)
              - 1.5 * max(0.0, d_tcp - 0.08))
    return {"success": success, "lifted": round(lifted, 4), "contacts": int(max_both),
            "dense": round(float(reward), 4), "d_tcp": round(d_tcp, 4),
            "reason": None if success else ("no_contact" if max_both < 2 else "no_lift" if lifted <= success_lift else "dropped")}


def _positions(seed: int = 0):
    return [(0.40, 0.0), (0.44, 0.06), (0.44, -0.06), (0.48, 0.0)]


def grasp_score(gene, policy, *, positions=None, residual_scale: float = 0.12, fin_residual_scale: float = 0.04) -> float:
    """ES fitness: mean lift-dominant grasp reward over object positions (blowups score -1). Graph-derived, so
    the SAME fitness works on any manipulator."""
    import numpy as np
    positions = positions or _positions()
    vals = [morph_grasp_rollout(gene, policy, gx, gy, residual_scale=residual_scale,
                                fin_residual_scale=fin_residual_scale)["dense"] for gx, gy in positions]
    return float(np.mean(vals)) if vals else -1.0


def evaluate_morph_grasp(gene, policy, *, positions=None, residual_scale: float = 0.12, fin_residual_scale: float = 0.04) -> dict:
    """Real friction-grasp success (grasp_eval's metric: >=2 finger contacts & lifted & TCP near box) with
    the learned residual applied — the transfer metric."""
    positions = positions or _positions()
    rs = [morph_grasp_rollout(gene, policy, gx, gy, residual_scale=residual_scale,
                              fin_residual_scale=fin_residual_scale) for gx, gy in positions]
    n = max(1, len(rs))
    return {"success_rate": round(sum(r["success"] for r in rs) / n, 3),
            "mean_lift": round(sum(r["lifted"] for r in rs) / n, 4),
            "mean_contacts": round(sum(r["contacts"] for r in rs) / n, 2)}


def grasp_feature_dim(gene) -> int:
    """The MorphPolicy feature_dim for ``gene`` in a grasp scene (a cube) — constant across manipulators, so a
    banked grasp residual recalls onto any arm whose grasp-scene encoding matches."""
    import mujoco

    from virturoid.schemas.scenes import SceneObject
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.morph_graph import encode_robot
    cube = SceneObject("box", "cube", (0.44, 0.0, 0.05, 0, 0, 0), mass_kg=0.03)
    return int(encode_robot(mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))).feature_dim)


def bank_grasp_residual(policy, gene, db, *, models_dir: str = "models", residual_scale: float = 0.12,
                        fin_residual_scale: float = 0.04, success_rate: float = 0.0) -> str:
    """Bank a transferable grasp residual into the SAME skill bank the locomotion flywheel uses, keyed by
    structural kind + 'grasp'. The residual SCALES (applied at rollout time, not in the weights) ride in
    ``base_config`` so recall reproduces the exact behavior. Returns the npz path. Idempotent keep-best."""
    from pathlib import Path

    from virturoid.services.task_matched_eval import robot_kind
    kind = robot_kind(gene)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    npz = str(Path(models_dir) / f"grasp_{kind}__{getattr(gene, 'id', 'imported')}.npz")
    policy.to_npz(npz, success_rate)
    db.record_skill(f"morph_{kind}_grasp", kind, "grasp", success_rate=float(success_rate), params_path=npz,
                    species=getattr(gene, "species", None), gene_id=getattr(gene, "id", None),
                    obs_dim=int(policy.feature_dim), act_dim=int(policy.hidden),
                    base_config={"residual_scale": float(residual_scale), "fin_residual_scale": float(fin_residual_scale)},
                    notes=f"transferable per-token grasp residual; success={success_rate:.3f}")
    # Per-species file bridge so the production grasp skill (_run_grasp_lift via models_dir) picks it up with no
    # db handle — mirrors locomotion's learned_<species>.npz. Scales ride in a sidecar json.
    import json
    sp = getattr(gene, "species", None) or getattr(gene, "robot_class", None) or "imported"
    base = Path(models_dir) / ("learned_" + str(sp).replace("/", "_") + "_grasp")
    policy.to_npz(str(base.with_suffix(".npz")), success_rate)
    base.with_suffix(".json").write_text(
        json.dumps({"residual_scale": float(residual_scale), "fin_residual_scale": float(fin_residual_scale)}),
        encoding="utf-8")
    return npz


def banked_grasp_policy(gene, *, models_dir: str = "models"):
    """File-keyed recall of THIS body's banked grasp residual (learned_<species>_grasp.npz + sidecar scales) —
    the execution-side reader for _run_grasp_lift, mirroring learn_locomotion.banked_policy_for. Returns
    ``(policy, {residual_scale, fin_residual_scale})`` or ``(None, None)``."""
    import json
    from pathlib import Path

    from virturoid.services.morph_policy import MorphPolicy
    sp = getattr(gene, "species", None) or getattr(gene, "robot_class", None) or "imported"
    base = Path(models_dir) / ("learned_" + str(sp).replace("/", "_") + "_grasp")
    if not base.with_suffix(".npz").exists():
        return None, None
    try:
        pol = MorphPolicy.from_npz(str(base.with_suffix(".npz")))
    except Exception:  # noqa: BLE001
        return None, None
    if pol.feature_dim != grasp_feature_dim(gene):
        return None, None
    sc = {"residual_scale": 0.12, "fin_residual_scale": 0.04}
    if base.with_suffix(".json").exists():
        try:
            sc = {**sc, **json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))}
        except Exception:  # noqa: BLE001
            pass
    return pol, sc


def recall_grasp_residual(gene, db):
    """Recall the best banked grasp residual for ``gene``'s structural kind (cross-manipulator). Returns
    ``(policy, {residual_scale, fin_residual_scale, success_rate})`` or ``(None, None)`` if none/incompatible."""
    from virturoid.services.morph_policy import MorphPolicy
    from virturoid.services.task_matched_eval import robot_kind
    meta = db.recall_skill(robot_kind(gene), "grasp")
    if not meta or not meta.get("params_path"):
        return None, None
    try:
        pol = MorphPolicy.from_npz(meta["params_path"])
    except Exception:  # noqa: BLE001
        return None, None
    if pol.feature_dim != grasp_feature_dim(gene):           # incompatible encoding -> cold start
        return None, None
    bc = meta.get("base_config") or {}
    return pol, {"residual_scale": float(bc.get("residual_scale", 0.12)),
                 "fin_residual_scale": float(bc.get("fin_residual_scale", 0.04)),
                 "success_rate": meta.get("success_rate")}


def train_grasp_residual(genes, *, generations: int = 24, pop: int = 24, seed: int = 0, warm_start=None,
                         residual_scale: float = 0.12, fin_residual_scale: float = 0.04, workers: int | None = None):
    """ES-train a per-token MorphPolicy grasp residual across ``genes`` (one body or a distribution). Returns
    ``(policy, history)``. One residual drives any manipulator (per-token attention), so it transfers."""
    import mujoco
    import numpy as np

    from virturoid.schemas.scenes import SceneObject
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_trainer import train_morph_es
    cube = SceneObject("box", "cube", (0.44, 0.0, 0.05, 0, 0, 0), mass_kg=0.03)
    fd = encode_robot(mujoco.MjModel.from_xml_string(compile_gene_with_scene(genes[0], [cube]))).feature_dim

    def fit(policy, gs):
        return float(np.mean([grasp_score(g, policy, residual_scale=residual_scale,
                                          fin_residual_scale=fin_residual_scale) for g in gs]))

    return train_morph_es(genes, feature_dim=fd, generations=generations, pop=pop, steps=2, seed=seed,
                          init_policy=warm_start, fitness_fn=fit, workers=workers or 1)
