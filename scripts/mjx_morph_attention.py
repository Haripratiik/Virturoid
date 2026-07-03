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
    ap.add_argument("--decimation", type=int, default=1, help="CONTROL DECIMATION (plan v2 T1.1, the confirmed "
                    "deploy-gap root-cause fix): hold each policy action for D physics substeps (PD + CPG clock "
                    "run every substep), so the policy decides at sim_hz/D instead of every step. 1 = act every "
                    "step (byte-identical to pre-decimation). At timestep 0.002, D=10 -> 50 Hz control (the "
                    "legged_gym/Playground canon). The control horizon is ep_len//D so total physics time is "
                    "unchanged. Reward is computed once per CONTROL step with dt scaled to D*sim_dt (legged_gym "
                    "computes reward at the control rate). Deploy with recipe_rollout_morph(decimation=D) to match.")
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
    ap.add_argument("--fwd-gate-w", type=float, default=0.0, help="G4 ROOT-CAUSE fix (the parity probe FALSIFIED "
                    "the MJX-vs-CPU sim-flip theory: pure-CPG CPU==MJX==~0). The true bug is REWARD DILUTION: the "
                    "gait-quality bonuses are direction-agnostic, so an IN-PLACE/BACKWARD march maxes them while the "
                    "lone forward term is swamped. In [0,1], the fraction by which those bonuses are GATED on forward "
                    "speed (fwd/vtgt): 0.0 = old ungated reward (byte-identical, suite/other recipes unchanged), "
                    "1.0 = quality pays only while moving forward. Use ~0.85 to fix the reversed-gait convergence.")
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
    ap.add_argument("--periodic-w", type=float, default=0.0,
                    help="PERIODIC gait-contact reward (plan v4 P1, Siekmann-style duty): penalize each foot for "
                         "staying in STANCE longer than half a gait cycle -- forces every foot to LIFT on the CPG "
                         "rhythm, so the learned residual can't suppress the CPG foot-lift into a forward SLIDE "
                         "(the measured hexapod failure). 0 = off (byte-identical).")
    ap.add_argument("--upright-hi", type=float, default=0.7, help="upright-ramp SATURATION height (fraction of "
                    "standing z where the continuous upright reward hits 1.0). Default 0.7 (byte-identical; the "
                    "quad walks). RAISE toward ~0.9 for bodies that FARM the reward by CROUCHING: at 0.7 a "
                    "0.67-height slide already earns ~85% upright, so PPO never stands tall enough to swing its "
                    "legs (the hexapod crouch-slide, cadence 0). Higher saturation -> a crouch earns far less -> "
                    "the policy stands up and STEPS. Fall line stays at 0.5*z_stand.")
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
    ap.add_argument("--contact-dr", action="store_true", help="plan v2 T1.5: per-env MODEL-param domain "
                    "randomization (canonical MuJoCo-Playground pattern: batch geom_friction/body_mass/dof_"
                    "armature/dof_frictionloss as vmapped mjx.Model leaves), targeting the MJX<->CPU CONTACT gap "
                    "for vigorous learned gaits. Off (default) -> in_axes=(None,0) = byte-identical single model.")
    ap.add_argument("--dr-scale", type=float, default=1.0, help="P3 (plan v4): scale the contact-DR range WIDTH "
                    "(deviation from 1.0). 1.0=canonical Go1/Barkour ranges (default, byte-identical); <1 = milder "
                    "(e.g. 0.4) so a warm-started nominal policy adds robustness WITHOUT toppling under too-wide DR.")
    ap.add_argument("--sphere-feet", action="store_true", help="plan v2 T1.4: convert the foot geoms to SPHERES "
                    "with FEET-ONLY collision. A sphere-plane contact is the IDENTICAL 1-point manifold in MJX and "
                    "C MuJoCo, so this removes the deploy-gap AT ITS SOURCE (capsule/box feet give a 2-point MJX "
                    "manifold that MPR resolves differently on CPU). Applied to the SAME model MJX trains on and "
                    "the CPU verifier deploys (banked in meta[8] so verify auto-matches). Off -> untouched.")
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
    ap.add_argument("--phase-obs", action="store_true", help="P1 (plan v4): feed the CPG master-phase gait "
                    "clock [sin,cos] as a percept token (Wrange) so the policy TIMES its steps (Siekmann); the "
                    "CPU MorphPolicy mirrors it via meta[9] + recipe_rollout_morph. Use with --cpg.")
    ap.add_argument("--calf-phase", type=float, default=None, help="override the trot-CPG calf phase (rad); "
                    "per-body gait direction (quad walks fwd at 1.5708, a bilateral hexapod at 0.0)")
    ap.add_argument("--cpg-freq", type=float, default=None, help="override the trot-CPG frequency (Hz)")
    ap.add_argument("--action-lpf", type=float, default=0.0, help="plan v2 T1.2: EMA low-pass on the action offset "
                    "at the control rate (Playground uses ~5Hz between policy and PD); fraction of the previous "
                    "offset retained, in [0,1). 0 = off (byte-identical). ~0.2-0.4 strips high-freq content the "
                    "policy would use to chase contact micro-dynamics. Deploy with recipe_rollout_morph(action_lpf=).")
    ap.add_argument("--keep-checkpoints", action="store_true", help="plan v2 T0.1: also save NUMBERED checkpoints "
                    "(runs/<name>_it{N}.npz every 10 iters) so DEPLOY-SIM checkpoint selection can pick best-by-CPU "
                    "(MJX reward can rise while CPU deploy flips sign -- never select on train-sim reward).")
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
    if args.sphere_feet:                                  # plan v2 T1.4: manifold-invariant sphere feet + feet-only
        from virturoid.services.sphere_feet import apply_sphere_feet   #   collision, applied to the model MJX trains
        _sf = apply_sphere_feet(mj)                       #   on so train==deploy (verify reads meta[8] + mirrors this).
        print(f"sphere-feet ON: {len(_sf)} foot geoms -> spheres + feet-only collision", flush=True)
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
    FGATE = args.fwd_gate_w                                    # G4 root-cause fix: gate gait-quality on forward speed
    ACTION_LPF = float(args.action_lpf)                       # T1.2: EMA low-pass on the action offset (0 = off)
    AIR_W, AIR_TGT, VZ_W, WXY_W = args.air_w, args.air_target, args.vz_w, args.wxy_w   # legged_gym reward terms
    TORQUE_W = args.torque_w                              # actuator-effort penalty (anti-crank stabilizer)
    PERIODIC_W = float(args.periodic_w)                   # P1 (plan v4): periodic gait-contact reward weight (0 = off)
    UPRIGHT_HI = max(0.55, float(args.upright_hi))        # upright-ramp saturation height (anti-crouch; default 0.7)
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
    # P1 periodic gait-contact reward: the expected MAX stance duration = half a gait cycle (Siekmann duty 0.5).
    # A foot grounded longer than this is "stuck" (a slide); the reward penalizes the excess, forcing every foot
    # to LIFT on the CPG rhythm so the learned residual can't cancel the CPG foot-lift into a forward slide.
    MAX_STANCE = (0.5 / CPG_FREQ) if (CPG_ON and CPG_FREQ > 0.0) else 0.35
    # SIM2REAL DR knobs (static -> every DR op below is guarded by DR_ON, so --dr off is byte-identical).
    DR_ON = bool(args.dr); DR_GAIN = float(args.dr_gain); DR_PD = float(args.dr_pd)
    CONTACT_DR = bool(args.contact_dr)                        # T1.5: per-env MODEL-param DR (vmapped mjx.Model)
    PHASE_OBS = bool(args.phase_obs)                          # P1: phase-clock obs token (Wrange percept); use with --cpg
    PERCEPT_DIM = 10                                          # == MorphPolicy.PERCEPT_DIM (8 range slots + 2 cmd/clock)
    DR_SCALE = float(args.dr_scale)                           # P3: contact-DR range-width scale (1.0 = canonical)
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
    DECIM = max(1, int(args.decimation))                 # control decimation (T1.1): physics substeps per action
    DT = float(mj.opt.timestep) * DECIM                  # CONTROL dt for feet_air_time/reward (legged_gym rate)
    if DECIM > 1:
        print(f"control decimation: D={DECIM} -> {1.0 / DT:.0f} Hz control over {1.0 / mj.opt.timestep:.0f} Hz "
              f"physics; control horizon {args.ep_len // DECIM} steps", flush=True)
    F = static.shape[1] + 4 + 9
    H = args.hidden
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.99, 0.95, 0.2, 5e-3, 0.5, 3e-4
    N, L = args.envs, max(5, args.ep_len // DECIM)       # L = CONTROL steps (physics time = L*DECIM ~ ep_len)

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
    if PHASE_OBS:                                         # P1: perception encoder for the [sin,cos] gait-clock token
        att["Wrange"] = W(ks[8], (PERCEPT_DIM, H)); att["brange"] = jp.zeros(H)   # ks[8] was previously unused
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

    def _phase_percept_1(tm):
        """scalar sim time -> (PERCEPT_DIM,) percept: the global CPG master-phase gait clock [sin,cos] in the last
        two slots (8,9), zeros elsewhere — EXACTLY what recipe_rollout_morph feeds on CPU (deploy==train)."""
        mp = TWO_PI * CPG_FREQ * tm
        return jp.concatenate([jp.zeros(PERCEPT_DIM - 2), jp.stack([jp.sin(mp), jp.cos(mp)])])

    def policy_mean_pool(P, obs, percept=None):
        """obs (NT, F) -> (action_means (NT,), pooled_embed (H,)). Uses the SHARED forward (morph_forward)
        so the GPU trainer is byte-identical to the CPU MorphPolicy — trained weights transfer to any body.
        ``percept`` (PERCEPT_DIM,) adds the P1 gait-clock token via Wrange (None -> proprioception-only)."""
        return attention_forward(P["att"], obs, H, xp=jp, film=FILM, hop=HOP, topo_bias=TOPO_BIAS, percept=percept)

    def vmlp(ps, x):
        x = jp.tanh(x @ ps[0][0] + ps[0][1])
        return x @ ps[1][0] + ps[1][1]

    def act_and_value(P, obs_b, percept_b=None):         # obs_b (N, NT, F); percept_b (N, PERCEPT_DIM) or None
        if percept_b is None:
            means, pools = jax.vmap(lambda o: policy_mean_pool(P, o))(obs_b)           # (N,NT),(N,H)
        else:                                            # P1: per-env gait clock vmapped alongside obs
            means, pools = jax.vmap(lambda o, pc: policy_mean_pool(P, o, pc))(obs_b, percept_b)
        vals = vmlp(P["vf"], pools)[:, 0]                                              # (N,)
        return means, vals

    # CONTACT-PARAM DR (T1.5): batch per-env mjx.Model leaves (friction/mass/armature/frictionloss) + a matching
    # in_axes Model; vmap step/forward over (model, data). The canonical MuJoCo-Playground pattern, targeting the
    # MJX<->CPU CONTACT gap for vigorous learned gaits. OFF -> mx_v=mx, mx_axes=None -> in_axes=(None,0) = the
    # original single-model path (byte-identical). make_data keeps the UNBATCHED mx (shapes don't depend on leaves).
    if CONTACT_DR:
        def _rand_model(rng):
            k = jax.random.split(rng, 4)
            def _rng(lo, hi):                                 # P3: scale the range's deviation from 1.0 by DR_SCALE
                return 1.0 - (1.0 - lo) * DR_SCALE, 1.0 + (hi - 1.0) * DR_SCALE   # DR_SCALE=1.0 -> (lo, hi) unchanged
            _fl, _fh = _rng(0.7, 1.4); _ml, _mh = _rng(0.9, 1.1)
            _al, _ah = _rng(1.0, 1.05); _ll, _lh = _rng(0.5, 2.0)
            fr = mx.geom_friction.at[:, 0].set(
                mx.geom_friction[:, 0] * jax.random.uniform(k[0], (mx.ngeom,), minval=_fl, maxval=_fh))
            ma = mx.body_mass * jax.random.uniform(k[1], (mx.nbody,), minval=_ml, maxval=_mh)
            ar = mx.dof_armature.at[6:].set(
                mx.dof_armature[6:] * jax.random.uniform(k[2], (mx.nv - 6,), minval=_al, maxval=_ah))
            fl = mx.dof_frictionloss.at[6:].set(
                mx.dof_frictionloss[6:] * jax.random.uniform(k[3], (mx.nv - 6,), minval=_ll, maxval=_lh))
            return fr, ma, ar, fl
        _fr, _ma, _ar, _fl = jax.vmap(_rand_model)(jax.random.split(jax.random.PRNGKey(7), N))
        mx_v = mx.replace(geom_friction=_fr, body_mass=_ma, dof_armature=_ar, dof_frictionloss=_fl)
        mx_axes = jax.tree_util.tree_map(lambda _x: None, mx).replace(
            geom_friction=0, body_mass=0, dof_armature=0, dof_frictionloss=0)
        print(f"contact-DR ON: per-env friction/mass/armature/frictionloss over {N} envs", flush=True)
    else:
        mx_v, mx_axes = mx, None
    step_v = jax.vmap(mjx.step, in_axes=(mx_axes, 0))

    _spawn = jp.asarray(spawn_qpos) if spawn_qpos is not None else None  # imported robot's home pose

    def reset(key):
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        if _spawn is not None:                            # spawn every env at the robot's standing home pose
            data = data.replace(qpos=jp.broadcast_to(_spawn, (N, _spawn.shape[0])))
        return jax.vmap(mjx.forward, in_axes=(mx_axes, 0))(mx_v, data)

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
            # alive (N,): 1.0 until this env first falls; air_time/last_contact (N, n_feet): feet_air_time state;
            # stance_time (N, n_feet): per-foot GROUNDED duration (P1 periodic reward; symmetric to air_time).
            data, alive, a_prev, air_time, last_contact, stance_time = carry
            obs = jax.vmap(obs_tokens)(data)              # (N, NT, F)
            pkey = key
            if DR_ON:                                     # sim2real: noisy obs + (below) randomized dynamics + pushes
                key, okey, pkey = jax.random.split(key, 3)
                obs = obs + DR_OBS * jax.random.normal(okey, obs.shape)
            pcpt = jax.vmap(_phase_percept_1)(data.time) if PHASE_OBS else None   # P1 gait clock (N, PERCEPT_DIM)
            means, val = act_and_value(params, obs, pcpt)   # (N,NT),(N,)
            act = means + std * jax.random.normal(key, means.shape)
            logp = (-0.5 * (((act - means) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            a_clip = jp.clip(act, -1, 1)
            off = jp.tanh(act)                            # the bounded control OFFSET actually applied (recipe)
            if ACTION_LPF > 0.0:                          # ACTION LPF (T1.2): EMA-smooth the applied offset at the
                off = ACTION_LPF * a_prev + (1.0 - ACTION_LPF) * off   # control rate (train==deploy); 0 -> unchanged.
            # CONTROL DECIMATION (T1.1): `_ctrl_of(d)` computes ctrl from the CURRENT substep state (CPG clock
            # advances via d.time, PD tracks the held target as the body moves) with the learned action `off`/
            # `a_clip` HELD. Step it DECIM times. DECIM=1 -> one step from the INPUT state = byte-identical.
            def _ctrl_of(d):
                if LEGACY:
                    grav = d.qfrc_bias[:, act_dof]
                    return jp.clip(grav + a_clip * args.alpha * clamps, -clamps, clamps), None
                # RECIPE: position-PD to the DEFAULT standing pose + a small learned offset (self-righting prior),
                # per ACTUATED-JOINT token (qadr/vadr), then scattered to actuator-indexed ctrl via act_u.
                # CPG prior (if --cpg): feed-forward trot offset on the PD target; policy learns the RESIDUAL.
                cpg_off = (cpg_amp_j[None, :] * jp.sin(TWO_PI * CPG_FREQ * jp.reshape(d.time, (-1, 1)) + cpg_phase_j[None, :])
                           ) if CPG_ON else 0.0
                tgt = q_default + cpg_off + ASCALE * (RES_SCALE if CPG_ON else 1.0) * off   # (N, NT) per-token target
                KPe = KP * kp_s[:, None] if DR_ON else KP   # sim2real: per-env PD stiffness (proxy joint friction/damping)
                KDe = KD * kd_s[:, None] if DR_ON else KD
                tau = KPe * (tgt - d.qpos[:, qadr]) - KDe * d.qvel[:, vadr]
                ct = jp.clip(tau, -clamps, clamps)                     # (N, NT)
                if DR_ON:
                    ct = jp.clip(ct * gain_s[:, None], -clamps, clamps)   # per-env actuator-gain scale
                return jp.zeros((N, nu)).at[:, act_u].set(ct), ct         # scatter token -> actuator index
            ctrl, ctrl_tok = _ctrl_of(data)                              # first substep from the INPUT state
            data2 = step_v(mx_v, data.replace(ctrl=ctrl))
            if DECIM > 1:                                                # (DECIM-1) more substeps HOLDING the action
                def _substep(d, _):
                    c, _ct = _ctrl_of(d)
                    return step_v(mx_v, d.replace(ctrl=c)), None
                data2, _ = jax.lax.scan(_substep, data2, None, length=DECIM - 1)
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
                stance_time2 = stance_time                              # P1 stance-time unused in legacy path
            else:
                quat = data2.qpos[:, base_qadr + 3:base_qadr + 7]
                upr = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)   # world-up . body-up
                up = jp.clip((z - 0.5 * z_stand) / ((UPRIGHT_HI - 0.5) * z_stand), 0.0, 1.0)   # CONTINUOUS upright
                #   RAMP: 0 at the fall line z=0.5*z_stand, 1 at standing z>=UPRIGHT_HI*z_stand (default 0.7): a smooth
                #   gradient OUT of a crouch instead of a binary cliff that paid 0 the instant z dipped (so a
                #   half-collapsed body got no signal to climb back up). --upright-hi RAISES the saturation so a
                #   crouch-slide can no longer farm ~full upright reward -> the body must stand tall to STEP.
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
                # P1 PERIODIC gait-contact reward (plan v4): accumulate per-foot GROUNDED time (reset the step a
                # foot lifts), and penalize the EXCESS over half a gait cycle -- so every foot MUST lift on the CPG
                # rhythm and the learned residual can't cancel the CPG foot-lift into a forward slide (the measured
                # hexapod failure). PERIODIC_W=0 -> periodic_pen=0 -> training byte-identical.
                stance_time2 = (stance_time + DT) * contact_filt               # grounded duration; 0 on liftoff
                periodic_pen = PERIODIC_W * jp.sum(jp.maximum(stance_time2 - MAX_STANCE, 0.0), axis=-1)
                # lin_vel_z / ang_vel_xy stabilizers (legged_gym): keep the gait forward + the trunk level
                vz = jp.nan_to_num(data2.qvel[:, base_vadr + 2])
                wxy = jp.nan_to_num(data2.qvel[:, base_vadr + 3:base_vadr + 5])
                stab = VZ_W * vz ** 2 + WXY_W * jp.sum(wxy ** 2, axis=-1)
                # actuator-EFFORT penalty (legged_gym torque term): normalized PD torque (ctrl/clamp)^2 averaged over
                # tokens, so the policy pays for cranking destabilizing targets — the reward now shapes the APPROACH
                # to each step, not just the fall cliff. Always on (ungated) so it discourages flailing while crouched.
                effort = jp.mean((ctrl_tok / (clamps + 1e-6)) ** 2, axis=-1)
                # G4 ROOT-CAUSE FIX (the parity probe FALSIFIED the sim-flip hypothesis: pure-CPG CPU==MJX==~0, no
                # flip). The real bug is REWARD DILUTION -- the gait-QUALITY bonuses (clearance/swing/alternation/
                # air-time) are direction-agnostic, so a tidy IN-PLACE or BACKWARD march maxes them while the lone
                # forward term is swamped -> the policy banks a clean reversed gait (cadence 13.3, upright .92, fwd
                # -0.29). GATE those bonuses on FORWARD speed so "nice stepping" only pays while going forward.
                # FGATE in [0,1] interpolates: 0 = old ungated reward (byte-identical -> suite/other recipes unchanged),
                # 1 = fully gated. alive(0.2*up)+upright(0.15) stay ungated (don't just fall); progress+back_pen stay directional.
                fgate = jp.clip(fwd / (VTGT + 1e-6), 0.0, 1.0)         # 0 standing/backward -> 1 at target forward speed
                gate_mult = (1.0 - FGATE) + FGATE * fgate              # FGATE=0 -> 1.0 (unchanged); FGATE=1 -> fgate
                step_r = jp.maximum(0.0, PROG_W * progress * up + 0.2 * up + 0.15 * jp.maximum(0.0, upr)
                                    + (CLEAR_W * clearance + SWING_W * swing + ALT_W * swing_phase
                                       + AIR_W * air_reward) * up * gate_mult
                                    - SLIP_W * slip - SMOOTH_W * sm - stab - TORQUE_W * effort - periodic_pen)
                # G4 PARITY FIX: an explicit cost for BACKWARD base-x, applied OUTSIDE the max(0) so the stepping
                # rewards can't fund a reversed gait. Scales with how backward it goes -> backward is driven to 0
                # reward (never a basin) while a forward gait pays nothing. Kills the MJX backward-convergence.
                back_pen = BACK_W * jp.maximum(0.0, -fwd) * up
                rew = alive2 * jp.maximum(0.0, step_r - back_pen)       # non-negative; zero after a fall or if reversing
            # carry the action basis used for smoothness next step (applied offset for recipe; clipped for legacy)
            a_next = a_clip if LEGACY else off
            outs = (obs, act, logp, rew, val, fwd, alive2)
            if PHASE_OBS:                                   # P1: carry the gait clock so loss_fn recomputes logp WITH it
                outs = outs + (pcpt,)
            return (data2, alive2, a_next, air_time2, last_contact2, stance_time2), outs

        return jax.lax.scan(body, carry, keys)

    @jax.jit
    def finish(params, data_last, traj):
        if PHASE_OBS:                                         # P1: the 8th buffer array is the per-step gait clock
            obs, act, logp, rew, val, fwd, alive, pcpt = traj
            last_pcpt = jax.vmap(_phase_percept_1)(data_last.time)   # bootstrap value uses the clock at data_last too
        else:
            obs, act, logp, rew, val, fwd, alive = traj; pcpt = last_pcpt = None
        _, last_val = act_and_value(params, jax.vmap(obs_tokens)(data_last), last_pcpt)

        def gae(carry, x):
            nextval, adv = carry
            rew_t, val_t, alive_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = (delta + GAMMA * LAM * adv) * alive_t       # fallen steps contribute no gradient / no bootstrap
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val, alive), reverse=True)
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), fwd.mean(0).mean(), pcpt

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
        carry = (data, jp.ones(N), jp.zeros((N, nu)), jp.zeros((N, n_feet)), jp.ones((N, n_feet)),
                 jp.zeros((N, n_feet)))                          # + stance_time (P1 periodic reward; feet start grounded)
        chunks = []
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            carry, out = roll_chunk(params, dr, carry, jax.random.split(ck, CHUNK))
            jax.block_until_ready(carry[0].qpos)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(len(chunks[0]))]  # 7, or 8 with phase-obs
        return finish(params, carry[0], traj)

    def loss_fn(params, obs, act, old_logp, adv, ret, percept=None):
        std = jp.exp(params["logstd"])[0]
        means, val = act_and_value(params, obs, percept)
        logp = (-0.5 * (((act - means) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
        ratio = jp.exp(logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio, -adv * jp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        ent = (params["logstd"] + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
        return pg + VF * ((val - ret) ** 2).mean() - ENT * ent

    @jax.jit
    def update(params, opt_state, obs, act, logp, adv, ret, key, percept=None):
        B = obs.shape[0]; mb = B // args.minibatches

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, B)

            def mbstep(carry, i):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                pc = percept[idx] if percept is not None else None       # P1: the gait clock for this minibatch
                g = jax.grad(loss_fn)(params, obs[idx], act[idx], logp[idx], adv[idx], ret[idx], pc)
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), None
            (params, opt_state), _ = jax.lax.scan(mbstep, (params, opt_state), jp.arange(args.minibatches))
            return (params, opt_state, key), None
        (params, opt_state, _), _ = jax.lax.scan(epoch, (params, opt_state, key), jp.arange(args.epochs))
        return params, opt_state

    def _save(p, fwd, path=None):
        dst = path or args.save
        a = p["att"]
        # save in the MorphPolicy._ORDER so the CPU policy loads straight in (transfer to any body)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        # Phase-5 opt-in weights ride alongside (detected by key presence in MorphPolicy.from_npz).
        extra = {}
        if FILM:
            extra["Wfilm"] = np.asarray(a["Wfilm"])
        if TOPO_BIAS:
            extra["Wtopo"] = np.asarray(a["Wtopo"])
        if PHASE_OBS:                                         # P1: bank the perception encoder so CPU deploy matches
            extra["Wrange"] = np.asarray(a["Wrange"]); extra["brange"] = np.asarray(a["brange"])
        if CPG_PARAMS is not None:                            # bank the EXACT CPG so CPU deploy matches training
            extra["cpg_arr"] = np.asarray([CPG_PARAMS["freq"], CPG_PARAMS["thigh_amp"], CPG_PARAMS["calf_amp"],
                                           CPG_PARAMS["calf_phase"], CPG_PARAMS["residual_scale"],
                                           1.0 if CPG_PARAMS.get("leg_flip") else 0.0])
        np.savez(dst, We=np.asarray(a["We"]), be=np.asarray(a["be"]), Wq=np.asarray(a["Wq"]),
                 Wk=np.asarray(a["Wk"]), Wv=np.asarray(a["Wv"]), Wo=np.asarray(a["Wo"]),
                 Wh=np.asarray(a["Wh"]), bh=np.asarray(a["bh"]), **extra,
                 # meta layout MUST match MorphPolicy.from_npz: [F,H,NT,fwd, adaptive@4, cpg@5, decimation@6,
                 # action_lpf@7, sphere_feet@8] so CPU replay reads the right flags (adaptive gains + trot-CPG prior +
                 # control rate + action filter + foot contact model) and reproduces the gait it was trained with.
                 meta=np.asarray([F, H, NT, float(fwd),
                                  1.0 if args.adaptive else 0.0, 1.0 if CPG_ON else 0.0, float(DECIM),
                                  float(ACTION_LPF),           # meta[7]: action LPF (T1.2) so deploy matches train
                                  1.0 if args.sphere_feet else 0.0,      # meta[8]: sphere feet (T1.4); deploy==train
                                  1.0 if PHASE_OBS else 0.0]))           # meta[9]: P1 phase-clock obs; deploy==train

    if args.eval_npz:                                   # EVAL: deterministic single-env MJX rollout (recipe control)
        dd = np.load(args.eval_npz)
        for kk in list(params["att"].keys()):             # includes Wfilm/Wtopo when the upgrades are on
            if kk in dd.files:
                params["att"][kk] = jp.asarray(dd[kk])

        @jax.jit
        def estep(d):
            obs = obs_tokens(d)
            _pc = _phase_percept_1(d.time) if PHASE_OBS else None   # P1: single-env gait clock (deploy==train)
            means, _ = policy_mean_pool(params, obs, _pc)   # deterministic mean action (no exploration noise)
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
        obs, act, logp, adv, ret, ep_rew, ep_fwd, pcpt = rollout(params, rk)   # pcpt = gait clock (None w/o phase-obs)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        if PHASE_OBS:                                        # P1: flatten + pass the clock so PPO minibatches carry it
            params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret),
                                       uk, flat(pcpt))
        else:
            params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  fwd_vel={float(ep_fwd):+.3f}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)
        # CHECKPOINT often — the WSL2 GPU watchdog (TDR) kills long kernels mid-run (observed: a hang ~iter 20),
        # and every-30 lost everything before the first checkpoint. Every 10 keeps a fetchable policy (CPG-primed,
        # so it still walks) even when the box is killed early -- the difference between a usable run and a None.
        if args.save and it > 0 and it % 10 == 0:
            _save(params, ep_fwd); print(f"  [checkpoint @ iter {it}]", flush=True)
            if args.keep_checkpoints:                        # plan v2 T0.1: keep NUMBERED checkpoints so deploy-sim
                _save(params, ep_fwd, path=args.save.replace(".npz", f"_it{it}.npz"))   # selection can pick best-by-CPU
                #   (the divergence curve: MJX reward rises while CPU-deploy can flip sign -> never trust the final)
    print(f"done: attention {args.robot}, fwd_vel {float(ep_fwd):+.3f}, {time.time()-t0:.0f}s", flush=True)
    if args.save:
        _save(params, ep_fwd)
        print(f"saved attention policy -> {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
