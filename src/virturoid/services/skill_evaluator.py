"""Evaluate a banked learned SKILL the same way every policy is evaluated (startup plan §32.5).

A learned skill (e.g. the M3 residual grasp) is trained on the MJX/GPU backend and banked by the skill
flywheel. But a *training* curve is not an evaluation — §32.5 requires every policy be run through one
protocol: fixed baseline scenes, randomized scenes, known-failure (here: out-of-region) scenes, and a
reproducibility check, producing comparable EpisodeRecords + a Policy Card. This service does exactly
that on the CPU MuJoCo backend (no GPU needed): it reconstructs the trainer's control law from the
skill's ``base_config`` (so it is faithful) plus the learned residual from the saved params, runs the
policy, and reports honest held-out success — the antidote to "we trained something and hope it works".

It also doubles as a **cross-backend fidelity check**: if the CPU success matches the MJX training
success, the skill transfers across backends (the plan's backend-agnostic principle, §3.2).
"""

from __future__ import annotations

import math
from pathlib import Path

from virturoid.schemas.policy_card import CategoryResult, PolicyCard
from virturoid.schemas.runs import EpisodeRecord
from virturoid.services.mujoco_runner import mujoco_available


def _load_policy(params_path: str):
    """Load the policy (pi) MLP from the trainer's npz as a list of (W, b) numpy layers."""
    import numpy as np

    data = np.load(params_path)
    layers, i = [], 0
    while f"pi_w{i}" in data:
        layers.append((np.asarray(data[f"pi_w{i}"]), np.asarray(data[f"pi_b{i}"])))
        i += 1
    if not layers:
        raise ValueError(f"no policy (pi_*) params in {params_path}")
    return layers


def _mlp(layers, x):
    """Deterministic MLP forward (tanh hidden, linear output) — the policy MEAN action."""
    import numpy as np

    for w, b in layers[:-1]:
        x = np.tanh(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def _grasp_rollout(gene, cfg: dict, layers, gx: float, gy: float, record: bool = False, friction: float = 1.0):
    """One CPU-MuJoCo grasp episode with the learned policy; returns the outcome dict (and, if
    ``record``, an ``outcome["trajectory"]`` of per-step obs/action/reward/robot+object state for the
    demonstration-dataset system §41).

    Mirrors ``scripts/mjx_residual_grasp.py``: pre-fold to q_above, phase base (settle→descend→grip→lift)
    via gravity-comp joint-PD toward a ramped reference, plus the learned residual on top, with the same
    contact bitmasks (box↔table/finger, finger↔box, arm disabled). Deterministic for a given (gx, gy).
    """
    import numpy as np
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    kpj, kdj = float(cfg.get("kpj", 120.0)), float(cfg.get("kdj", 12.0))
    kpf, kdf = float(cfg.get("kpf", 300.0)), float(cfg.get("kdf", 10.0))
    alpha, dq = float(cfg.get("alpha", 0.3)), float(cfg.get("dq", 0.10))
    p_settle, p_grip, p_lift = cfg.get("phases", [0.15, 0.50, 0.65])
    L, fclose = int(cfg.get("ep_len", 200)), float(cfg.get("fclose", 0.045))

    cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=float(friction), scale=1.0)   # WS9: parameterized friction (hard grasp scene)
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))
    mj.opt.iterations = 20; mj.opt.ls_iterations = 10
    tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
    bq = int(mj.jnt_qposadr[box_jid]); bv = int(mj.jnt_dofadr[box_jid])
    BOX_G = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box")
    FL = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_geom")
    FR = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_geom")
    box_body = int(mj.jnt_bodyid[box_jid]); fl_b = int(mj.geom_bodyid[FL]); fr_b = int(mj.geom_bodyid[FR])
    for g in range(mj.ngeom):                                   # contact bitmasks (match the trainer)
        b = int(mj.geom_bodyid[g])
        mj.geom_contype[g], mj.geom_conaffinity[g] = (
            (0b11, 0b11) if b == box_body else (0b10, 0b10) if b in (fl_b, fr_b)
            else (0b01, 0b01) if b == 0 else (0, 0))

    nu = int(mj.nu)
    act_dof = [int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(nu)]
    arm = [u for u in range(nu) if int(mj.jnt_type[int(mj.actuator_trnid[u, 0])]) != 2]
    fin = [u for u in range(nu) if int(mj.jnt_type[int(mj.actuator_trnid[u, 0])]) == 2]
    arm_qadr = [int(mj.jnt_qposadr[int(mj.actuator_trnid[u, 0])]) for u in arm]
    fr_lim = mj.actuator_forcerange[:, 1]
    KP = np.array([kpj if u in arm else kpf for u in range(nu)])
    KD = np.array([kdj if u in arm else kdf for u in range(nu)])

    def ik(z):
        q = plan_joint_targets(mj, (gx, gy, z), iterations=6, candidates=40, site_name="grasp_site")[0]
        return np.array([q[u] for u in arm])

    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)
    box_z = float(d.qpos[bq + 2])
    q_above, q_at, q_lift = ik(0.15), ik(box_z), ik(0.20)
    for k, u in enumerate(arm):
        d.qpos[arm_qadr[k]] = q_above[k]                        # pre-fold to the hover pose
    mujoco.mj_forward(mj, d)
    box_z0 = float(d.qpos[bq + 2]); q_arm_ref = q_above.copy()

    def finger_box_contacts():
        fl = fr = 0
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if BOX_G in (g1, g2):
                if FL in (g1, g2): fl += 1
                if FR in (g1, g2): fr += 1
        return fl, fr

    max_both, lifted_max = 0, -1.0
    traj = {"obs": [], "actions": [], "rewards": [], "robot_qpos": [], "robot_qvel": [],
            "object_pose": []} if record else None
    prev_lift = 0.0
    for t in range(L):
        frac = t / L
        q_arm_tgt = q_lift if frac >= p_lift else (q_above if frac < p_settle else q_at)
        q_arm_ref = q_arm_ref + np.clip(q_arm_tgt - q_arm_ref, -dq, dq)
        f_tgt = 0.0 if frac < p_grip else fclose
        q_star = np.zeros(nu)
        for k, u in enumerate(arm):
            q_star[u] = q_arm_ref[k]
        for u in fin:
            q_star[u] = f_tgt
        # observation (must match the trainer's obs_of): qpos, qvel, tcp, box, tcp-box, lifted, phase
        boxp = d.qpos[bq:bq + 3]; tcpp = d.site_xpos[tcp]
        obs = np.concatenate([d.qpos, d.qvel, tcpp, boxp, tcpp - boxp,
                              [boxp[2] - box_z], [frac]])
        obs = np.nan_to_num(obs)
        mean = _mlp(layers, obs)
        q = d.qpos[np.array(act_dof)]; qd = d.qvel[np.array(act_dof)]
        base = d.qfrc_bias[np.array(act_dof)] + KP * (q_star - q) - KD * qd   # gravity-comp joint-PD
        ctrl = np.clip(base + np.clip(mean, -1, 1) * alpha * fr_lim, -fr_lim, fr_lim)  # + bounded residual
        if record:
            traj["obs"].append(obs.copy()); traj["actions"].append(ctrl.copy())
            traj["robot_qpos"].append(d.qpos[:nu].copy()); traj["robot_qvel"].append(d.qvel[:nu].copy())
            traj["object_pose"].append(d.qpos[bq:bq + 7].copy())
            lift_now = float(d.qpos[bq + 2]) - box_z0
            traj["rewards"].append(lift_now - prev_lift); prev_lift = lift_now  # per-step lift increment
        d.ctrl[:] = ctrl
        mujoco.mj_step(mj, d)
        if frac >= p_grip:
            fl, fr_c = finger_box_contacts()
            max_both = max(max_both, min(fl, fr_c))             # both fingers in contact
        lifted_max = max(lifted_max, float(d.qpos[bq + 2]) - box_z0)

    boxp = d.qpos[bq:bq + 3]
    d_tcp = float(np.linalg.norm(d.site_xpos[tcp] - boxp))
    lifted = float(d.qpos[bq + 2]) - box_z0
    box_speed = float(np.linalg.norm(d.qvel[bv:bv + 3]))
    success = bool(max_both >= 1 and lifted > 0.05 and d_tcp < 0.06)
    # Specific, actionable failure reason (§26 "failures are specific and useful"), not a bare label.
    if success:
        reason = None
    elif max_both < 1:
        reason = "no_grasp_contact"      # jaws never closed on the box (reach/approach problem)
    elif lifted <= 0.05:
        reason = "gripped_no_lift"        # both fingers contacted but the box didn't come off the table
    elif d_tcp >= 0.06:
        reason = "lifted_then_dropped"    # raised the box then lost it (slip / unstable grasp)
    else:
        reason = "low_lift"               # marginal lift just under the success threshold
    out = {"success": success, "lifted": round(lifted, 4), "lifted_max": round(lifted_max, 4),
           "both_contact": int(max_both >= 1), "d_tcp": round(d_tcp, 4),
           "box_speed": round(box_speed, 4), "failure_reason": reason}
    if record:
        out["trajectory"] = {k: np.asarray(v) for k, v in traj.items()}
    return out


def _grid(x_lo, x_hi, y_lo, y_hi, n):
    xs = [x_lo + (x_hi - x_lo) * (i + 0.5) / n for i in range(n)]
    ys = [y_lo + (y_hi - y_lo) * (j + 0.5) / n for j in range(n)]
    return [(x, y) for x in xs for y in ys]


def evaluate_skill(skill: dict, *, params_path: str | None = None, baseline_n: int = 3,
                   randomized_n: int = 8, seed: int = 0, task_graph_id: str = "grasp_task",
                   run_id: str = "skill_eval") -> tuple[PolicyCard, list[EpisodeRecord], dict]:
    """Run the §32.5 evaluation suite on a banked learned skill (CPU MuJoCo). Returns (card, episodes, summary).

    ``skill`` is a banked skill dict (from ``MemoryDB.recall_skill`` / a SKILL_CARD): it must carry
    ``region``, ``base_config``, ``params_path`` (overridable), ``gene_id``, ``species``, ``obs_dim``,
    ``act_dim``. The gripper gene is reconstructed from the seed arm (the skill is the grasp species).
    """
    if not mujoco_available():
        raise RuntimeError("MuJoCo is required to evaluate a skill.")
    import numpy as np

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper

    params_path = params_path or skill.get("params_path")
    if not params_path or not Path(params_path).exists():
        raise FileNotFoundError(f"policy params not found: {params_path}")
    layers = _load_policy(params_path)
    gene = add_parallel_gripper(tabletop_arm_gene())
    cfg = skill.get("base_config") or {}
    region = skill.get("region") or {"x": [0.43, 0.49], "y": [-0.08, 0.08]}
    xr, yr = region["x"], region["y"]

    episodes: list[EpisodeRecord] = []
    cats: list[CategoryResult] = []

    def run_scenes(category, points):
        succ, metric = 0, 0.0
        for i, (gx, gy) in enumerate(points):
            r = _grasp_rollout(gene, cfg, layers, float(gx), float(gy))
            succ += int(r["success"]); metric += r["lifted_max"]
            episodes.append(EpisodeRecord(
                id=f"{run_id}_{category}_{i:03d}",
                run_id=run_id, robot_genome_id=skill.get("gene_id", gene.id), task_graph_id=task_graph_id,
                scene_graph_id=f"{category}_{i:03d}_x{gx:.3f}_y{gy:.3f}", policy_id=skill.get("skill_id", "skill"),
                backend="mujoco", seed=seed, result="success" if r["success"] else "failure",
                metrics={**r, "box_x": round(float(gx), 4), "box_y": round(float(gy), 4)},
                failure_labels=[] if r["success"] else [r.get("failure_reason") or "failure"]))
        n = max(1, len(points))
        cr = CategoryResult(category=category, scene_count=len(points),
                            success_rate=round(succ / n, 4), mean_metric=round(metric / n, 4))
        cats.append(cr)
        return cr

    # §32.5: fixed baseline scenes (deterministic grid)
    run_scenes("baseline", _grid(xr[0], xr[1], yr[0], yr[1], baseline_n))
    # §32.5: randomized scenes (seeded, reproducible)
    rng = np.random.default_rng(seed)
    rand_pts = [(rng.uniform(xr[0], xr[1]), rng.uniform(yr[0], yr[1])) for _ in range(randomized_n)]
    run_scenes("randomized", rand_pts)
    # §32.5: known-failure scenes — genuinely outside the arm's reachable workspace (too close to the
    # base, too far, or too lateral). These SHOULD mostly fail; the card reports the measured boundary
    # honestly rather than asserting it.
    edge = [(0.22, 0.0), (0.62, 0.0), (0.46, 0.30), (0.30, 0.25)]
    run_scenes("known_failure", edge)

    # §32.5/§21.4: reproducibility — re-run the baseline; deterministic CPU MuJoCo must match exactly
    repro = [_grasp_rollout(gene, cfg, layers, gx, gy)["success"]
             for gx, gy in _grid(xr[0], xr[1], yr[0], yr[1], baseline_n)]
    base_cat = next(c for c in cats if c.category == "baseline")
    base_success = [e.result == "success" for e in episodes if e.scene_graph_id.startswith("baseline")]
    reproducible = bool(repro == base_success)

    in_dist = [c for c in cats if c.category in ("baseline", "randomized")]
    headline = round(sum(c.success_rate * c.scene_count for c in in_dist) /
                     max(1, sum(c.scene_count for c in in_dist)), 4)
    kf = next(c for c in cats if c.category == "known_failure")

    # Cluster the failures by SPECIFIC reason (§8.5/§26) with example scenes — actionable, not a bar label.
    import collections
    by_reason: dict[str, list[str]] = collections.defaultdict(list)
    for e in episodes:
        if e.result != "success":
            by_reason[(e.failure_labels or ["failure"])[0]].append(e.scene_graph_id)
    failure_clusters = [{"reason": reason, "count": len(scenes), "example_scenes": scenes[:3]}
                        for reason, scenes in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))]

    card = PolicyCard(
        id=skill.get("skill_id", "skill_card"),
        name=f"{skill.get('task_type','grasp')}_{skill.get('species','manipulator')}",
        policy_type="residual_rl_grasp", task_type=skill.get("task_type", "grasp"),
        compatible_robots=[skill.get("gene_id", gene.id)],
        compatible_tasks=[skill.get("task_type", "grasp")],
        observation_space=["qpos", "qvel", "grasp_site_xyz", "box_xyz", "tcp_minus_box", "lift_height", "phase"],
        action_space=f"residual_joint_torque[{skill.get('act_dim', gene_nu(gene))}]",
        valid_region=region, training_scenes=f"MJX residual RL, region x{xr} y{yr}",
        success_rate=headline, category_results=cats, reproducible=reproducible,
        failure_clusters=failure_clusters,
        known_failures=[f"beyond the reachable workspace (near-base/far/extreme-lateral): "
                        f"{kf.success_rate:.0%} success over {kf.scene_count} probe scenes "
                        "— the gripper TCP cannot reach down onto the box there"],
        safety_constraints=[f"joint torque clamped to forcerange; residual bounded to alpha={cfg.get('alpha',0.3)}"],
        required_sensors=["proprioception (qpos/qvel)", "object pose (grasp_site relative)"],
        runtime_requirements={"backend": "mujoco|mjx", "control_steps": cfg.get("ep_len", 200),
                              "base": "phase FK-IK joint-PD + bounded residual MLP"},
        limitations=["single tabletop cube ~0.05 m; fixed friction/mass (no domain randomization yet)"],
        params_uri=params_path, backend="mujoco")
    summary = {"headline_success": headline, "reproducible": reproducible,
               "categories": {c.category: {"n": c.scene_count, "success": c.success_rate,
                                           "mean_lift": c.mean_metric} for c in cats},
               "failure_clusters": failure_clusters, "episodes": len(episodes)}
    return card, episodes, summary


def gene_nu(gene) -> int:
    return len(gene.actuated_joints())


def evaluate_and_record(skill: dict, out_dir, *, memory=None, params_path: str | None = None,
                        **kw) -> tuple[PolicyCard, dict]:
    """Run the §32.5 evaluation and write the §12.3 outputs (policy card + evaluation result + episodes).

    Writes ``policy_card.json``, ``evaluation_episodes.jsonl`` and ``summary.json`` to ``out_dir``; if a
    ``MemoryDB`` is given, records the run and stamps the banked skill with its honest held-out success
    (``eval_success``) so the flywheel offers calibrated skills, not training-curve numbers.
    """
    import json

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    card, episodes, summary = evaluate_skill(skill, params_path=params_path, **kw)
    (out / "policy_card.json").write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    with (out / "evaluation_episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict(), sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if memory is not None:
        memory.record_run(
            prompt=f"held-out §32.5 evaluation of {card.name}", robot_class="manipulator",
            task_type=skill.get("task_type", "grasp"), converged_design=None,
            success_rate=summary["headline_success"], species=skill.get("species"),
            succeeded=summary["headline_success"] >= 0.8, backend="mujoco", design_source="skill_eval")
        # stamp the banked skill with the honest held-out number + reproducibility
        memory.record_skill(
            skill["skill_id"], "manipulator", skill.get("task_type", "grasp"),
            success_rate=skill.get("success_rate", 0.0), params_path=skill.get("params_path"),
            species=skill.get("species"), gene_id=skill.get("gene_id"),
            reward_spec=skill.get("reward_spec"), region=skill.get("region"),
            base_config=skill.get("base_config"), obs_dim=skill.get("obs_dim"),
            act_dim=skill.get("act_dim"),
            notes=f"held-out eval: {summary['headline_success']:.0%} (reproducible={summary['reproducible']})")
    return card, summary
