"""MJX PPO for the ATTENTION MorphPolicy — Theme 1's payoff: one policy that transfers across morphologies.

Trains the same attention-over-tokens architecture as the CPU ``morph_policy.MorphPolicy`` (per-token
embed → self-attention → per-token action), in MJX, on a composed body. The policy's parameter layout
MATCHES the NumPy MorphPolicy exactly (We, be, Wq, Wk, Wv, Wo, Wh, bh), so the trained ``--save`` npz
loads straight into the CPU MorphPolicy and can be applied ZERO-SHOT to a DIFFERENT body (any token
count) — the morphology-generalization claim, verified CPU-side after training.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_morph_attention.py --robot quadruped --iters 200 \
        --envs 1024 --save runs/morph_quad.npz
    PYTHONPATH=src ~/rl/bin/python scripts/mjx_morph_attention.py --smoke
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="quadruped")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--ep-len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.6)
    # --- anti-collapse recipe (the CPU-validated fix; see memory locomotion-collapse-fix) -----------------
    # Default ON: torque-residual-on-grav-comp control + no-termination reward is the COLLAPSE bug (the body
    # learns a fall-forward flop, not a gait). The recipe replaces it with position-PD-to-DEFAULT-pose control
    # (self-righting prior) + terminate/alive-mask (a fallen env earns 0 thereafter, killing floor-rolling) +
    # a clipped-non-negative velocity-TRACKING reward (kills the suicide-lunge). --legacy-control restores the
    # old (collapse-prone) torque-residual path for A/B comparison.
    ap.add_argument("--legacy-control", action="store_true", help="OLD torque-residual-on-grav-comp (collapse-prone)")
    ap.add_argument("--kp", type=float, default=32.0, help="PD position gain (recipe control)")
    ap.add_argument("--kd", type=float, default=1.5, help="PD velocity gain (recipe control)")
    ap.add_argument("--ascale", type=float, default=0.4, help="action offset scale around the default pose")
    ap.add_argument("--vtgt", type=float, default=0.3, help="target forward speed (m/s) for velocity tracking")
    ap.add_argument("--clear-w", type=float, default=2.5, help="foot-CLEARANCE reward weight (anti-shuffle: pay the "
                    "policy to LIFT feet off the floor — without it the recipe holds a stiff stand-and-shuffle)")
    # GAIT-QUALITY reward knobs the AI gait-critic (services/gait_critic.py) tunes to turn a slide into a WALK.
    ap.add_argument("--swing-w", type=float, default=0.0, help="HIGH-swing reward: pay for lifting one foot HIGH "
                    "(a real step), not just the mean clearance a shuffle satisfies")
    ap.add_argument("--slip-w", type=float, default=0.0, help="anti-SLIP/DRAG penalty: penalize moving forward "
                    "with BOTH feet planted (the slide hack that games fwd_vel without stepping)")
    ap.add_argument("--alt-w", type=float, default=0.5, help="SWING-PHASE reward: pay for having at least one foot "
                    "OFF the ground (stepping, any morphology) instead of standing on all feet. ON by default so "
                    "air-time can't be farmed by a synchronized bound (both feet leave together) — a real walk "
                    "needs single-support phases")
    ap.add_argument("--smooth-w", type=float, default=0.3, help="control-smoothness penalty weight")
    ap.add_argument("--prog-w", type=float, default=4.0, help="forward-PROGRESS reward (linear, 0 at standstill, "
                    "capped at vtgt) — the 'actually go forward' incentive that a Gaussian speed-tracker lacked "
                    "(a small vtgt makes standing score ~82 pct of a Gaussian, so it converges to standing)")
    ap.add_argument("--back-w", type=float, default=10.0, help="BACKWARD-base-x penalty weight (G4 parity fix): "
                    "the gait-quality rewards (air-time/alternation/clearance) pay for STEPPING regardless of "
                    "direction, so a body earns them while REVERSING a forward CPG prior — the MJX-vs-CPU backward "
                    "convergence. This penalty is applied OUTSIDE the max(0) gate, so a backward gait is driven "
                    "toward 0 reward (never a basin) while a forward gait is untouched.")
    # CANONICAL legged_gym (Rudin et al.) reward terms — the published dictionary that converts an upright slide
    # into a deliberate WALK (see docs/virturoid_research_dossier.md). feet_air_time is the single most important
    # anti-slide / anti-stand-still term: it pays the policy for the TIME each foot spends in the air per step
    # (rewarded on touchdown, target ~0.5 s), so a held stance or a foot-dragging slide earns nothing while real
    # swinging steps earn the reward. ON by default — it is what the height-only clearance/swing terms were missing.
    ap.add_argument("--air-w", type=float, default=1.0, help="feet_air_time reward weight (legged_gym): pay per "
                    "foot for swing TIME on touchdown (target 0.5 s) — the canonical anti-slide/anti-shuffle term. "
                    "LOWERED from 2.0: at 2.0 it dwarfed the upright incentive, so PPO farmed air-time by lunging "
                    "right up to the fall cliff (the iter-35 collapse: fwd_vel climbs while the body topples)")
    ap.add_argument("--torque-w", type=float, default=0.02, help="actuator-EFFORT penalty (legged_gym torque term): "
                    "penalize normalized PD torque (ctrl/clamp)^2 so the policy can't crank destabilizing targets "
                    "for free — gives the reward something shaping the APPROACH to a step, not just the fall cliff")
    ap.add_argument("--air-target", type=float, default=0.5, help="feet_air_time target swing duration (s)")
    ap.add_argument("--vz-w", type=float, default=0.4, help="lin_vel_z penalty (legged_gym): penalize vertical "
                    "base bouncing so the gait is forward, not a pogo")
    ap.add_argument("--wxy-w", type=float, default=0.04, help="ang_vel_xy penalty (legged_gym): penalize base "
                    "roll/pitch rate so the trunk stays level while stepping")
    ap.add_argument("--cpg", action="store_true", help="TROT-CPG PRIOR (residual RL): inject a feed-forward diagonal-"
                    "trot oscillation into the PD target (mirrors the CPU recipe's CPG_DEFAULT) and let PPO learn only "
                    "the propulsion/balance RESIDUAL. Empirically necessary on the quad: pure PPO from a standstill "
                    "gets STUCK in a lunge-and-fall local optimum (alive ~251/1000, flat across 105 iters) — it cannot "
                    "discover a gait from rest. Quad-only (gated by leg naming); a non-quad falls back to scalar recipe.")
    # SIM2REAL domain randomization (opt-in --dr): per-EPISODE per-env randomized dynamics so the policy is robust to
    # the reality gap. Default off -> the rollout is byte-identical to a non-DR run (every DR op is guarded by DR_ON).
    ap.add_argument("--dr", action="store_true", help="enable per-episode domain randomization (actuator gain, PD "
                    "stiffness, observation noise, random pushes) for sim2real robustness")
    ap.add_argument("--dr-gain", type=float, default=0.15, help="actuator-gain randomization: per-env torque scale "
                    "sampled uniformly in [1-x, 1+x] (motor strength / gear-efficiency variation)")
    ap.add_argument("--dr-pd", type=float, default=0.15, help="PD-stiffness randomization: per-env Kp/Kd scale in "
                    "[1-x, 1+x] (proxies joint friction/damping + controller-gain mismatch)")
    ap.add_argument("--dr-obs", type=float, default=0.02, help="observation noise std (Gaussian, per step) added to "
                    "the policy's normalized obs — sensor noise / state-estimation error")
    ap.add_argument("--dr-push", type=float, default=0.02, help="per-step probability of a random horizontal velocity "
                    "PUSH on the base (external disturbance robustness); 0 disables pushes")
    ap.add_argument("--dr-push-mag", type=float, default=0.5, help="push magnitude (m/s) of the random base kick")
    ap.add_argument("--adaptive", action="store_true", help="ADAPTIVE per-joint PD gains from each DOF's effective "
                    "inertia (mj_fullM diag), so a heavy biped/humanoid joint gets the stiffness flat kp=32 can't "
                    "provide (it folds otherwise). Mirrors CPU morph_policy.recipe_gains; anchored to 32/1.5 on the "
                    "reference quad. The 'any robot' leg of the recipe — needed to train a humanoid.")
    # EVAL mode (no training): load a saved policy, run ONE deterministic single-env episode ON MJX (the physics it
    # trained on — so survival here separates 'not converged' from the MJX->CPU sim gap) and dump the qpos trajectory
    # so it can be rendered locally WITHOUT a sim gap (replay the exact MJX motion).
    ap.add_argument("--eval-npz", default=None, help="evaluate a saved policy (skips training)")
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--dump-traj", default=None, help="save the eval qpos trajectory to this .npy for local render")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--init-npz", default=None, help="WARM-START the policy weights from a saved npz (chain short "
                    "bursts under the TDR limit, building on prior training — the AI-assisted gait loop)")
    ap.add_argument("--gene-json", default=None, help="train a RobotGene serialized to JSON (overrides --robot)")
    ap.add_argument("--mjcf-file", default=None, help="train an IMPORTED MJCF model directly (overrides --robot/--gene-json)")
    ap.add_argument("--film", action="store_true", help="Phase-5 tokenizer upgrade: FiLM joint-attribute "
                    "conditioning (adds Wfilm; the CPU MorphPolicy mirrors it)")
    ap.add_argument("--topo-bias", action="store_true", help="Phase-5 tokenizer upgrade: topology-aware "
                    "attention bias on kinematic hop distance (adds Wtopo)")
    ap.add_argument("--calf-phase", type=float, default=None, help="override the trot-CPG calf phase (rad); "
                    "per-body gait direction (quad walks fwd at 1.5708, a bilateral hexapod at 0.0)")
    ap.add_argument("--cpg-freq", type=float, default=None, help="override the trot-CPG frequency (Hz)")
    args = ap.parse_args(argv)
    if args.smoke:
        args.envs, args.iters, args.ep_len, args.minibatches = 2, 2, 20, 1

    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    import optax
    from mujoco import mjx

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements

    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    spawn_qpos = None
    if args.mjcf_file:                                    # an IMPORTED real robot (e.g. MuJoCo Menagerie Go1)
        mj = mujoco.MjModel.from_xml_path(args.mjcf_file)  # from_xml_path so mesh assets resolve
        args.robot = "imported"
        for g in range(mj.ngeom):                         # MJX-safe: cylinder COLLIDERS -> capsules
            if int(mj.geom_type[g]) == 5:
                mj.geom_type[g] = 3
        for u in range(mj.nu):                            # position SERVOS -> motor (torque) so PD-torque applies
            if int(mj.actuator_biastype[u]) != 0:
                mj.actuator_gaintype[u] = 0; mj.actuator_biastype[u] = 0
                mj.actuator_gainprm[u, :] = 0.0; mj.actuator_gainprm[u, 0] = 1.0
                mj.actuator_biasprm[u, :] = 0.0
                mj.actuator_ctrlrange[u] = mj.actuator_forcerange[u]   # ctrl is now FORCE; allow full torque range
        mj.opt.cone = 0                                   # PYRAMIDAL friction cone (MJX can't do elliptic+condim=1)
        for g in range(mj.ngeom):                         # ensure colliding geoms have friction (condim 3) for feet
            if int(mj.geom_contype[g]) and int(mj.geom_condim[g]) < 3:
                mj.geom_condim[g] = 3
        if mj.nkey > 0:
            spawn_qpos = np.array(mj.key_qpos[0], dtype=float)   # the robot's HOME/standing pose (the PD attractor)
        elif int(mj.jnt_type[0]) == 0:                           # floating base, NO keyframe (e.g. H1): STAND it —
            _d0 = mujoco.MjData(mj); mujoco.mj_forward(mj, _d0)  # joints at default, base lifted so feet clear floor
            _zmin = float(np.min(_d0.geom_xpos[:, 2])) if mj.ngeom else 0.0
            spawn_qpos = np.array(_d0.qpos, dtype=float); spawn_qpos[2] += (0.02 - _zmin)
        print(f"imported: nu={mj.nu} bodies={mj.nbody - 1} keyframe={'yes' if mj.nkey > 0 else 'no(stood it)'}",
              flush=True)
    else:
        if args.gene_json:                                # train an arbitrary body shipped from the app
            import json
            from virturoid.schemas.gene import RobotGene
            gene = RobotGene.from_dict(json.loads(Path(args.gene_json).read_text(encoding="utf-8")))
            print(f"gene-json: {gene.robot_class} ({len(gene.actuated_joints())} DOF)", flush=True)
        else:
            gene = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=args.robot, robot_class=args.robot))
        from virturoid.services.gene_compiler import standing_spawn_z
        # physics_only=True -> an MJX-SAFE model: no visual cylinder cans/collars/housings (MJX can't compile
        # cylinder collisions, and the contype=1 sweep below would force those cosmetic cylinders to collide and
        # crash mjx.put_model). Cylinder COLLIDERS become capsules. Colliders/joints/actuators are unchanged.
        mj = mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene), physics_only=True))
    if not args.mjcf_file:                                # generated bodies: force the physics-only geoms to collide
        for g in range(mj.ngeom):                         # (an imported robot keeps its native collision setup so
            mj.geom_contype[g] = 1; mj.geom_conaffinity[g] = 1   # visual meshes stay non-colliding -> MJX compiles)
    NACON, NJMAX = 64, 256
    mx = mjx.put_model(mj)

    graph = encode_robot(mj)
    NT = graph.n_tokens
    qadr = jp.asarray(graph.qadr); vadr = jp.asarray(graph.vadr)
    static = jp.asarray(graph.static)
    clamps = jp.asarray(graph.clamps)
    # Phase-5 tokenizer upgrades (opt-in, mirror the CPU MorphPolicy so trained weights transfer back).
    FILM, TOPO_BIAS, TOPO_BUCKETS = bool(args.film), bool(args.topo_bias), 8
    if TOPO_BIAS:                                          # static per-morphology hop-distance matrix (NT, NT)
        from virturoid.services.topo_pe import hop_distance_matrix
        HOP = jp.asarray(hop_distance_matrix(graph.parent), dtype=jp.int32)
    else:
        HOP = None
    base_qadr = graph.base_qadr
    base_vadr = int(mj.jnt_dofadr[graph.base_jid]) if graph.base_jid >= 0 else -1
    # STANDING height (the body now spawns on its feet) — used as a CONTINUOUS height reward so the policy
    # is pushed to walk TALL instead of finding a low crouch-scoot that a mere threshold bonus tolerated.
    _d0 = mujoco.MjData(mj)
    if spawn_qpos is not None:                            # an imported robot stands at its HOME keyframe, not qpos=0
        _d0.qpos[:] = spawn_qpos
    mujoco.mj_forward(mj, _d0)
    z_stand = float(_d0.qpos[base_qadr + 2]) if base_qadr >= 0 else 0.24
    nu = int(mj.nu)
    act_dof = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(nu)])  # legacy grav-comp
    # RECIPE control uses the PER-TOKEN graph arrays (qadr/vadr/clamps loaded above — one entry per ACTUATED joint,
    # in token order) EXACTLY like the CPU recipe_rollout_morph, so token k drives the right joint even when an
    # actuator is skipped (NT != nu). q_default is the default STANDING pose (the PD attractor — even a random
    # policy then hovers near standing); act_u scatters the per-token control back to actuator-indexed data.ctrl.
    act_u = jp.asarray(graph.act_u)
    q_default = jp.asarray([float(_d0.qpos[int(a)]) for a in graph.qadr])
    KP, KD, ASCALE, VTGT, CLEAR_W = args.kp, args.kd, args.ascale, args.vtgt, args.clear_w
    SWING_W, SLIP_W, ALT_W, SMOOTH_W, PROG_W = args.swing_w, args.slip_w, args.alt_w, args.smooth_w, args.prog_w
    BACK_W = args.back_w                                       # G4: explicit backward-base-x penalty
    AIR_W, AIR_TGT, VZ_W, WXY_W = args.air_w, args.air_target, args.vz_w, args.wxy_w   # legged_gym reward terms
    TORQUE_W = args.torque_w                              # actuator-effort penalty (anti-crank stabilizer)
    # TROT-CPG PRIOR (opt-in --cpg): per-token feed-forward amplitude/phase from the SAME logic as the CPU recipe
    # (morph_policy._trot_cpg_tokens + CPG_DEFAULT), so a GPU-trained policy transfers to / replays on the CPU path.
    # The phase is driven by the live sim clock data.time (no carry surgery; continuous across chunks, resets per ep).
    CPG_ON = False; CPG_FREQ = 0.0; RES_SCALE = 1.0; TWO_PI = 2.0 * float(np.pi)
    cpg_amp_j = jp.zeros(graph.n_tokens); cpg_phase_j = jp.zeros(graph.n_tokens)
    CPG_PARAMS = None
    if args.cpg:
        from virturoid.services.morph_policy import CPG_DEFAULT, _trot_cpg_tokens
        cpgd = dict(CPG_DEFAULT)                              # per-body gait overrides (direction/rhythm)
        if args.calf_phase is not None:
            cpgd["calf_phase"] = float(args.calf_phase)
        if args.cpg_freq is not None:
            cpgd["freq"] = float(args.cpg_freq)
        _amp, _phase, _gate = _trot_cpg_tokens(mj, graph, cpgd)
        if _gate:
            CPG_ON = True; CPG_FREQ = float(cpgd["freq"]); RES_SCALE = float(cpgd["residual_scale"])
            cpg_amp_j = jp.asarray(_amp); cpg_phase_j = jp.asarray(_phase); CPG_PARAMS = cpgd
            print(f"trot-CPG prior ON: freq={CPG_FREQ} calf_phase={cpgd['calf_phase']} res_scale={RES_SCALE} "
                  f"active={int((_amp != 0).sum())} tokens", flush=True)
        else:
            print("--cpg requested but body is not a recognized legged body; using scalar recipe", flush=True)
    # SIM2REAL DR knobs (static -> every DR op below is guarded by DR_ON, so --dr off is byte-identical).
    DR_ON = bool(args.dr); DR_GAIN = float(args.dr_gain); DR_PD = float(args.dr_pd)
    DR_OBS = float(args.dr_obs); DR_PUSH = float(args.dr_push); DR_PUSH_MAG = float(args.dr_push_mag)
    if DR_ON:
        print(f"domain randomization ON: gain+-{DR_GAIN} pd+-{DR_PD} obs_noise={DR_OBS} push_p={DR_PUSH}@{DR_PUSH_MAG}m/s", flush=True)
    if args.adaptive:
        # ADAPTIVE per-joint gains — inline mirror of CPU morph_policy.recipe_gains, computed from THIS box's own
        # model + graph so the token order is self-consistent (no cross-machine sync). Holds closed-loop wn/zeta
        # constant across morphologies: kp = clip(min(I_eff*wn^2, cap), 2, 192), cap floored at _KP_REF so the
        # reference quad stays EXACT; kd = clip(2*zeta*sqrt(kp*I_eff), 0.1, 24). KP/KD become (NT,) arrays that
        # broadcast over the (N, NT) PD math below unchanged.
        _M = np.zeros((mj.nv, mj.nv)); mujoco.mj_fullM(mj, _M, _d0.qM)
        _Ieff = np.maximum(1e-4, np.diag(_M)[np.asarray(graph.vadr, int)])
        _tau = np.asarray(graph.clamps, float)
        _I_REF, _KPR, _KDR = 0.04359, 32.0, 1.5
        _wn2 = _KPR / _I_REF; _zeta = _KDR / (2.0 * np.sqrt(_KPR * _I_REF))
        _cap = np.maximum(_KPR, 0.9 * _tau / max(1e-3, ASCALE))
        _kp = np.clip(np.minimum(_Ieff * _wn2, _cap), 2.0, 192.0)
        _kd = np.clip(2.0 * _zeta * np.sqrt(_kp * _Ieff), 0.1, 24.0)
        KP, KD = jp.asarray(_kp), jp.asarray(_kd)
        print(f"adaptive per-joint gains: kp[{_kp.min():.0f}..{_kp.max():.0f}] kd[{_kd.min():.2f}..{_kd.max():.2f}] "
              f"(median kp {np.median(_kp):.0f})", flush=True)
    # FEET (anti-shuffle): the lowest BODY geoms at the standing pose (exclude the worldbody/floor plane). The
    # reward pays the policy to lift these above their standing height, so it STEPS instead of dragging its feet.
    _gz0 = _d0.geom_xpos[:, 2]
    _bgeoms = [gi for gi in range(mj.ngeom) if int(mj.geom_bodyid[gi]) != 0]
    _zmin = min(float(_gz0[gi]) for gi in _bgeoms) if _bgeoms else 0.0
    _feet = [gi for gi in _bgeoms if float(_gz0[gi]) < _zmin + 0.05] or _bgeoms
    foot_idx = jp.asarray(_feet)
    foot_z0 = jp.asarray([float(_gz0[gi]) for gi in _feet])
    n_feet = len(_feet)
    DT = float(mj.opt.timestep)                          # per-step dt for feet_air_time accumulation
    F = static.shape[1] + 4 + 9
    H = args.hidden
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.99, 0.95, 0.2, 5e-3, 0.5, 3e-4
    N, L = args.envs, args.ep_len

    key = jax.random.PRNGKey(0)
    def W(k, shape, s=0.3):
        return jax.random.normal(k, shape) * s
    ks = jax.random.split(key, 9)
    att = {"We": W(ks[0], (F, H)), "be": jp.zeros(H),
           "Wq": W(ks[1], (H, H)), "Wk": W(ks[2], (H, H)), "Wv": W(ks[3], (H, H)), "Wo": W(ks[4], (H, H)),
           "Wh": W(ks[5], (H, 1)), "bh": jp.zeros(1)}
    if FILM:
        att["Wfilm"] = jp.zeros((F, 2 * H))               # -> (gamma, beta); zero-init = identity at start
    if TOPO_BIAS:
        att["Wtopo"] = jp.zeros(TOPO_BUCKETS + 1)         # learned bias per hop distance; zero-init = no bias
    vf = [(W(ks[6], (H, 64)) * (1 / H) ** 0.5, jp.zeros(64)), (W(ks[7], (64, 1)) * (1 / 64) ** 0.5, jp.zeros(1))]
    params = {"att": att, "vf": vf, "logstd": jp.zeros(1) - 0.5}
    if args.init_npz:                                    # WARM-START: chain short bursts (build on prior training)
        _d = np.load(args.init_npz)
        for _k in list(att.keys()):                       # includes Wfilm/Wtopo when those upgrades are on
            if _k in _d.files and att[_k].shape == _d[_k].shape:
                att[_k] = jp.asarray(_d[_k])
        params = {"att": att, "vf": vf, "logstd": jp.zeros(1) - 0.5}
        print(f"warm-started policy weights from {args.init_npz}", flush=True)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = opt.init(params)

    def obs_tokens(d):
        """(NT, F) per env — same feature layout as morph_graph.observe (so params transfer to CPU)."""
        qpos = d.qpos[qadr]; qvel = d.qvel[vadr]
        dyn = jp.stack([qpos, qvel, jp.sin(qpos), jp.cos(qpos)], axis=1)
        q = d.qpos[base_qadr:base_qadr + 7]; qd = d.qvel[base_vadr:base_vadr + 6]
        upright = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
        g = jp.array([1.0, q[2], upright, qd[0], qd[1], qd[2], qd[3], qd[4], qd[5]])
        return jp.nan_to_num(jp.concatenate([static, dyn, jp.tile(g, (NT, 1))], axis=1))

    from virturoid.services.morph_forward import attention_forward

    def policy_mean_pool(P, obs):
        """obs (NT, F) -> (action_means (NT,), pooled_embed (H,)). Uses the SHARED forward (morph_forward)
        so the GPU trainer is byte-identical to the CPU MorphPolicy — trained weights transfer to any body."""
        return attention_forward(P["att"], obs, H, xp=jp, film=FILM, hop=HOP, topo_bias=TOPO_BIAS)

    def vmlp(ps, x):
        x = jp.tanh(x @ ps[0][0] + ps[0][1])
        return x @ ps[1][0] + ps[1][1]

    def act_and_value(P, obs_b):                          # obs_b (N, NT, F)
        means, pools = jax.vmap(lambda o: policy_mean_pool(P, o))(obs_b)   # (N,NT),(N,H)
        vals = vmlp(P["vf"], pools)[:, 0]                                  # (N,)
        return means, vals

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    _spawn = jp.asarray(spawn_qpos) if spawn_qpos is not None else None  # imported robot's home pose

    def reset(key):
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        if _spawn is not None:                            # spawn every env at the robot's standing home pose
            data = data.replace(qpos=jp.broadcast_to(_spawn, (N, _spawn.shape[0])))
        return jax.vmap(lambda d: mjx.forward(mx, d))(data)

    def legacy_reward_of(d):
        qd = jp.nan_to_num(d.qvel[:, base_vadr:base_vadr + 3], posinf=0.0, neginf=0.0)
        fwd = jp.clip(qd[:, 0], -3.0, 3.0)
        quat = d.qpos[:, base_qadr + 3:base_qadr + 7]
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        z = jp.nan_to_num(d.qpos[:, base_qadr + 2], posinf=0.0, neginf=0.0)
        height = jp.clip(z / z_stand, 0.0, 1.0)
        # LEGACY (collapse-prone): forward GATED by height**2 + upright + stay-tall. Kept for --legacy-control A/B.
        return fwd * height * height + 0.3 * upright + 0.1 * height, fwd

    CHUNK = 5
    LEGACY = bool(args.legacy_control)

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, dr, carry, keys):
        std = jp.exp(params["logstd"])[0]
        gain_s, kp_s, kd_s = dr                          # per-env sim2real scales (all ones when --dr off)

        def body(carry, key):
            # alive (N,): 1.0 until this env first falls; air_time/last_contact (N, n_feet): feet_air_time state
            data, alive, a_prev, air_time, last_contact = carry
            obs = jax.vmap(obs_tokens)(data)              # (N, NT, F)
            pkey = key
            if DR_ON:                                     # sim2real: noisy obs + (below) randomized dynamics + pushes
                key, okey, pkey = jax.random.split(key, 3)
                obs = obs + DR_OBS * jax.random.normal(okey, obs.shape)
            means, val = act_and_value(params, obs)       # (N,NT),(N,)
            act = means + std * jax.random.normal(key, means.shape)
            logp = (-0.5 * (((act - means) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            a_clip = jp.clip(act, -1, 1)
            off = jp.tanh(act)                            # the bounded control OFFSET actually applied (recipe)
            if LEGACY:
                grav = data.qfrc_bias[:, act_dof]
                ctrl = jp.clip(grav + a_clip * args.alpha * clamps, -clamps, clamps)
            else:
                # RECIPE: position-PD to the DEFAULT standing pose + a small learned offset (self-righting prior),
                # per ACTUATED-JOINT token (qadr/vadr), then scattered to actuator-indexed ctrl via act_u.
                # CPG prior (if --cpg): feed-forward trot offset on the PD target; policy learns the RESIDUAL.
                cpg_off = (cpg_amp_j[None, :] * jp.sin(TWO_PI * CPG_FREQ * jp.reshape(data.time, (-1, 1)) + cpg_phase_j[None, :])
                           ) if CPG_ON else 0.0
                tgt = q_default + cpg_off + ASCALE * (RES_SCALE if CPG_ON else 1.0) * off   # (N, NT) per-token target
                KPe = KP * kp_s[:, None] if DR_ON else KP   # sim2real: per-env PD stiffness (proxy joint friction/damping)
                KDe = KD * kd_s[:, None] if DR_ON else KD
                tau = KPe * (tgt - data.qpos[:, qadr]) - KDe * data.qvel[:, vadr]
                ctrl_tok = jp.clip(tau, -clamps, clamps)                # (N, NT)
                if DR_ON:
                    ctrl_tok = jp.clip(ctrl_tok * gain_s[:, None], -clamps, clamps)   # per-env actuator-gain scale
                ctrl = jp.zeros((N, nu)).at[:, act_u].set(ctrl_tok)     # scatter token -> actuator index
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            if DR_ON and DR_PUSH > 0.0:                    # sim2real: random horizontal velocity PUSH on the base
                do_push = (jax.random.uniform(pkey, (N,)) < DR_PUSH).astype(jp.float32)
                kick = do_push[:, None] * DR_PUSH_MAG * jax.random.normal(jax.random.fold_in(pkey, 1), (N, 2))
                data2 = data2.replace(qvel=data2.qvel.at[:, base_vadr:base_vadr + 2].add(kick))
            qd = jp.nan_to_num(data2.qvel[:, base_vadr:base_vadr + 3], posinf=0.0, neginf=0.0)
            fwd = jp.clip(qd[:, 0], -3.0, 3.0)
            z = jp.nan_to_num(data2.qpos[:, base_qadr + 2], posinf=0.0, neginf=0.0)
            if LEGACY:
                rew_raw, _f = legacy_reward_of(data2)
                rew = rew_raw - 0.01 * jp.mean(a_clip ** 2, axis=-1)
                alive2 = alive                                          # no termination in legacy mode
                air_time2, last_contact2 = air_time, last_contact       # feet_air_time unused in legacy A/B path
            else:
                quat = data2.qpos[:, base_qadr + 3:base_qadr + 7]
                upr = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)   # world-up . body-up
                up = jp.clip((z - 0.5 * z_stand) / (0.2 * z_stand), 0.0, 1.0)   # CONTINUOUS upright RAMP (0 at the
                #   fall line z=0.5*z_stand, 1 at standing z>=0.7*z_stand): a smooth gradient OUT of a crouch instead
                #   of a binary cliff that paid exactly 0 the instant z dipped below 0.7 (so a half-collapsed body
                #   got no signal to climb back up). This is the core of the iter-35 collapse fix.
                fell = (z < 0.5 * z_stand).astype(jp.float32)           # FALLEN -> terminate
                alive2 = alive * (1.0 - fell)                           # 0 once fallen, stays 0 thereafter
                progress = jp.clip(fwd, -VTGT, VTGT)                  # forward speed up to vtgt; BACKWARD now PENALIZED
                #   (was clip(fwd, 0, vtgt) -> backward earned 0, not negative, so a policy could collect the
                #   gait-quality rewards while REVERSING a forward CPG prior; this is the humanoid backward-walk bug)
                sm = jp.mean((off - a_prev) ** 2, axis=-1)              # smoothness of the APPLIED control offset
                fz = data2.geom_xpos[:, foot_idx, 2]                    # (N, n_feet) foot heights
                clearance = jp.mean(jp.clip(fz - foot_z0, 0.0, 0.08), axis=-1)   # MEAN foot lift (anti-shuffle)
                # GAIT-QUALITY terms the AI gait-critic tunes to turn a SLIDE into a WALK: a HIGH single-foot lift
                # = a real step; an anti-SLIP penalty for moving forward with BOTH feet planted (the fwd_vel drag
                # hack — exactly the iter-30 failure: duty=1.0, clearance=0); an ALTERNATION reward for
                # single-support (exactly one foot planted) = stepping L/R, not dragging both.
                swing = jp.max(jp.clip(fz - foot_z0, 0.0, 0.12), axis=-1)        # (N,) highest foot lift = a step
                stance = (fz < foot_z0 + 0.02).astype(jp.float32)               # (N, n_feet) per-foot in-stance
                n_stance = jp.sum(stance, axis=-1)                              # (N,) feet on the ground
                both_planted = (n_stance >= fz.shape[-1]).astype(jp.float32)    # all feet down -> drag candidate
                # PARTIAL support = SOME feet down AND some up: a real stepping stance. Rewards neither standing
                # (all feet down) NOR bounding/leaping (all feet airborne) — the round-3 failure was a bound+dive.
                swing_phase = ((n_stance > 0.5) & (n_stance < fz.shape[-1] - 0.5)).astype(jp.float32)
                slip = both_planted * jp.abs(fwd)                              # forward motion while fully planted
                # feet_air_time (legged_gym, the canonical anti-slide term): reward the swing TIME of each foot,
                # paid only on TOUCHDOWN (first contact after being airborne), targeting AIR_TGT seconds. A held
                # stance accumulates no air time; a foot-dragging slide never lifts a foot, so neither earns it —
                # only a real swing-and-plant step does. `contact_filt` is the standard one-step bounce filter.
                contact = stance                                               # (N, n_feet) 1.0 if foot grounded
                contact_filt = jp.maximum(contact, last_contact)
                first_contact = (air_time > 0.0).astype(jp.float32) * contact_filt
                air_acc = air_time + DT
                air_reward = jp.sum((air_acc - AIR_TGT) * first_contact, axis=-1)   # (N,) summed over feet landing now
                air_time2 = air_acc * (1.0 - contact_filt)                     # reset feet now in contact
                last_contact2 = contact
                # lin_vel_z / ang_vel_xy stabilizers (legged_gym): keep the gait forward + the trunk level
                vz = jp.nan_to_num(data2.qvel[:, base_vadr + 2])
                wxy = jp.nan_to_num(data2.qvel[:, base_vadr + 3:base_vadr + 5])
                stab = VZ_W * vz ** 2 + WXY_W * jp.sum(wxy ** 2, axis=-1)
                # actuator-EFFORT penalty (legged_gym torque term): normalized PD torque (ctrl/clamp)^2 averaged over
                # tokens, so the policy pays for cranking destabilizing targets — the reward now shapes the APPROACH
                # to each step, not just the fall cliff. Always on (ungated) so it discourages flailing while crouched.
                effort = jp.mean((ctrl_tok / (clamps + 1e-6)) ** 2, axis=-1)
                step_r = jp.maximum(0.0, PROG_W * progress * up + 0.2 * up + 0.15 * jp.maximum(0.0, upr)
                                    + CLEAR_W * clearance * up + SWING_W * swing * up + ALT_W * swing_phase * up
                                    + AIR_W * air_reward * up
                                    - SLIP_W * slip - SMOOTH_W * sm - stab - TORQUE_W * effort)
                # G4 PARITY FIX: an explicit cost for BACKWARD base-x, applied OUTSIDE the max(0) so the stepping
                # rewards can't fund a reversed gait. Scales with how backward it goes -> backward is driven to 0
                # reward (never a basin) while a forward gait pays nothing. Kills the MJX backward-convergence.
                back_pen = BACK_W * jp.maximum(0.0, -fwd) * up
                rew = alive2 * jp.maximum(0.0, step_r - back_pen)       # non-negative; zero after a fall or if reversing
            # carry the action basis used for smoothness next step (applied offset for recipe; clipped for legacy)
            a_next = a_clip if LEGACY else off
            return (data2, alive2, a_next, air_time2, last_contact2), (obs, act, logp, rew, val, fwd, alive2)

        return jax.lax.scan(body, carry, keys)

    @jax.jit
    def finish(params, data_last, traj):
        obs, act, logp, rew, val, fwd, alive = traj
        _, last_val = act_and_value(params, jax.vmap(obs_tokens)(data_last))

        def gae(carry, x):
            nextval, adv = carry
            rew_t, val_t, alive_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = (delta + GAMMA * LAM * adv) * alive_t       # fallen steps contribute no gradient / no bootstrap
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val, alive), reverse=True)
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), fwd.mean(0).mean()

    def rollout(params, key):
        rkey, key = jax.random.split(key)
        data = reset(rkey)
        # SIM2REAL: sample per-env dynamics scales ONCE per episode (held across chunks); all ones when --dr off
        # (guarded so the non-DR key stream + rollout are byte-identical to before).
        if DR_ON:
            dk = jax.random.split(rkey, 3)
            gain_s = 1.0 + DR_GAIN * (2.0 * jax.random.uniform(dk[0], (N,)) - 1.0)
            kp_s = 1.0 + DR_PD * (2.0 * jax.random.uniform(dk[1], (N,)) - 1.0)
            kd_s = 1.0 + DR_PD * (2.0 * jax.random.uniform(dk[2], (N,)) - 1.0)
        else:
            gain_s = kp_s = kd_s = jp.ones(N)
        dr = (gain_s, kp_s, kd_s)
        # (data, alive-mask, prev-action, feet_air_time, last-contact) threaded across chunks; feet start grounded
        carry = (data, jp.ones(N), jp.zeros((N, nu)), jp.zeros((N, n_feet)), jp.ones((N, n_feet)))
        chunks = []
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            carry, out = roll_chunk(params, dr, carry, jax.random.split(ck, CHUNK))
            jax.block_until_ready(carry[0].qpos)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(7)]
        return finish(params, carry[0], traj)

    def loss_fn(params, obs, act, old_logp, adv, ret):
        std = jp.exp(params["logstd"])[0]
        means, val = act_and_value(params, obs)
        logp = (-0.5 * (((act - means) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
        ratio = jp.exp(logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio, -adv * jp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        ent = (params["logstd"] + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
        return pg + VF * ((val - ret) ** 2).mean() - ENT * ent

    @jax.jit
    def update(params, opt_state, obs, act, logp, adv, ret, key):
        B = obs.shape[0]; mb = B // args.minibatches

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, B)

            def mbstep(carry, i):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                g = jax.grad(loss_fn)(params, obs[idx], act[idx], logp[idx], adv[idx], ret[idx])
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), None
            (params, opt_state), _ = jax.lax.scan(mbstep, (params, opt_state), jp.arange(args.minibatches))
            return (params, opt_state, key), None
        (params, opt_state, _), _ = jax.lax.scan(epoch, (params, opt_state, key), jp.arange(args.epochs))
        return params, opt_state

    def _save(p, fwd):
        a = p["att"]
        # save in the MorphPolicy._ORDER so the CPU policy loads straight in (transfer to any body)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        # Phase-5 opt-in weights ride alongside (detected by key presence in MorphPolicy.from_npz).
        extra = {}
        if FILM:
            extra["Wfilm"] = np.asarray(a["Wfilm"])
        if TOPO_BIAS:
            extra["Wtopo"] = np.asarray(a["Wtopo"])
        if CPG_PARAMS is not None:                            # bank the EXACT CPG so CPU deploy matches training
            extra["cpg_arr"] = np.asarray([CPG_PARAMS["freq"], CPG_PARAMS["thigh_amp"], CPG_PARAMS["calf_amp"],
                                           CPG_PARAMS["calf_phase"], CPG_PARAMS["residual_scale"],
                                           1.0 if CPG_PARAMS.get("leg_flip") else 0.0])
        np.savez(args.save, We=np.asarray(a["We"]), be=np.asarray(a["be"]), Wq=np.asarray(a["Wq"]),
                 Wk=np.asarray(a["Wk"]), Wv=np.asarray(a["Wv"]), Wo=np.asarray(a["Wo"]),
                 Wh=np.asarray(a["Wh"]), bh=np.asarray(a["bh"]), **extra,
                 # meta layout MUST match MorphPolicy.from_npz: [F,H,NT,fwd, adaptive@4, cpg@5] so CPU replay reads
                 # the right flags (adaptive gains + the trot-CPG prior) and reproduces the GPU gait.
                 meta=np.asarray([F, H, NT, float(fwd),
                                  1.0 if args.adaptive else 0.0, 1.0 if CPG_ON else 0.0]))

    if args.eval_npz:                                   # EVAL: deterministic single-env MJX rollout (recipe control)
        dd = np.load(args.eval_npz)
        for kk in list(params["att"].keys()):             # includes Wfilm/Wtopo when the upgrades are on
            if kk in dd.files:
                params["att"][kk] = jp.asarray(dd[kk])

        @jax.jit
        def estep(d):
            obs = obs_tokens(d)
            means, _ = policy_mean_pool(params, obs)     # deterministic mean action (no exploration noise)
            cpg_off = (cpg_amp_j * jp.sin(TWO_PI * CPG_FREQ * d.time + cpg_phase_j)) if CPG_ON else 0.0
            tgt = q_default + cpg_off + ASCALE * (RES_SCALE if CPG_ON else 1.0) * jp.tanh(means)
            tau = KP * (tgt - d.qpos[qadr]) - KD * d.qvel[vadr]
            ctrl = jp.zeros(nu).at[act_u].set(jp.clip(tau, -clamps, clamps))
            return mjx.step(mx, d.replace(ctrl=ctrl))
        _d = mjx.make_data(mx, naconmax=NACON, njmax=NJMAX)
        if _spawn is not None:
            _d = _d.replace(qpos=_spawn)
        d = mjx.forward(mx, _d)
        z0 = float(d.qpos[base_qadr + 2]); x0 = float(d.qpos[base_qadr]); alive = args.eval_steps
        traj = []; clears = []
        for t in range(args.eval_steps):
            d = estep(d); jax.block_until_ready(d.qpos)
            traj.append(np.asarray(d.qpos)); z = float(d.qpos[base_qadr + 2])
            clears.append(float(np.mean(np.clip(np.asarray(d.geom_xpos[foot_idx, 2]) - np.asarray(foot_z0), 0, 0.08))))
            if z < 0.5 * z0:
                alive = t; break
        fwd = float(d.qpos[base_qadr]) - x0
        print(f"EVAL(MJX 1-env): forward={fwd:.3f} alive={alive}/{args.eval_steps} z0={z0:.3f} "
              f"final_z={float(d.qpos[base_qadr + 2]):.3f} clear_max={max(clears):.3f}", flush=True)
        if args.dump_traj:
            np.save(args.dump_traj, np.array(traj))
            print(f"dumped traj -> {args.dump_traj} shape {np.array(traj).shape}", flush=True)
        return 0

    print(f"ATTENTION robot={args.robot} tokens={NT} F={F} H={H} nu={nu} envs={N}", flush=True)
    t0 = time.time()
    ep_fwd = 0.0
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        obs, act, logp, adv, ret, ep_rew, ep_fwd = rollout(params, rk)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  fwd_vel={float(ep_fwd):+.3f}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)
        # CHECKPOINT often — the WSL2 GPU watchdog (TDR) kills long kernels mid-run (observed: a hang ~iter 20),
        # and every-30 lost everything before the first checkpoint. Every 10 keeps a fetchable policy (CPG-primed,
        # so it still walks) even when the box is killed early -- the difference between a usable run and a None.
        if args.save and it > 0 and it % 10 == 0:
            _save(params, ep_fwd); print(f"  [checkpoint @ iter {it}]", flush=True)
    print(f"done: attention {args.robot}, fwd_vel {float(ep_fwd):+.3f}, {time.time()-t0:.0f}s", flush=True)
    if args.save:
        _save(params, ep_fwd)
        print(f"saved attention policy -> {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
