#!/usr/bin/env python3
"""G4 CPU<->MJX parity probe (docs/breakthrough_research_plan.md G4).

Replays the SAME pure-CPG open-loop gait (zero residual -> isolates the SIM from any learned policy) and reports
the NET base-x travel under four configurations, to localize the direction flip that makes a CPU-forward hexapod
train BACKWARD in MJX:

  A  CPU, robot_mjcf model,      iters=20   -- the KNOWN-forward config (what the CPU search selected)
  B  CPU, physics_only model,    iters=20   -- same solver as A, the MJX model  -> isolates the MODEL build
  C  MJX, physics_only model,    default    -- the ACTUAL training sim
  D  MJX, physics_only model,    iters=20   -- tests the Part-A fix (pin solver iterations to match CPU)

Read the signs:
  A>0, B<0  -> the physics_only/forced-collide MODEL flips direction (fix = model build, not solver)
  A>0, B>0, C<0 -> same model+ctrl, only the sim differs -> MJX solver (iters/fp32/manifold) is the culprit
  C<0, D>0  -> pinning iterations to 20 fixes it  (Part A works; land it in the trainer)
  C<0, D<0  -> iterations is NOT enough -> contact manifold / fp32 -> escalate to elliptic-cone / condim study

Sign disagreement is the hard signal; 30-50% magnitude drift between matched-sign configs is expected and fine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CPG = {"freq": 1.5, "thigh_amp": 0.6, "calf_amp": 0.8, "calf_phase": 0.0,   # 0.0 = the CPU-forward hexapod config
       "residual_scale": 0.3, "leg_flip": True}


def _load_gene(gene_json: str | None):
    if gene_json:
        from virturoid.schemas.gene import RobotGene
        return RobotGene.from_dict(json.loads(Path(gene_json).read_text(encoding="utf-8")))
    # fallback: a bilateral hexapod (the G4 body) straight from the fixture builder
    from virturoid.services.steerable_body import steerable_quadruped
    return steerable_quadruped(n_legs=6, bilateral=True)


def _cpu_model(gene):
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    return compiled_model(robot_mjcf(gene))


def _mjx_model(gene):
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    mj = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene), physics_only=True))
    for g in range(mj.ngeom):                       # trainer forces generated geoms to collide
        mj.geom_contype[g] = 1
        mj.geom_conaffinity[g] = 1
    return mj


def _cpg_arrays(model, cpg):
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import _trot_cpg_tokens
    graph = encode_robot(model)
    amp, phase, gate = _trot_cpg_tokens(model, graph, cpg)
    return graph, np.asarray(amp, float), np.asarray(phase, float), bool(gate)


def cpu_rollout(model, cpg, steps, kp=32.0, kd=1.5):
    import mujoco
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    graph, amp, phase, gate = _cpg_arrays(model, cpg)
    qadr = np.asarray(graph.qadr, int); vadr = np.asarray(graph.vadr, int)
    act_u = np.asarray(graph.act_u, int); clamps = np.asarray(graph.clamps, float)
    q_def = np.array([float(data.qpos[a]) for a in qadr])
    bq = graph.base_qadr; dt = float(model.opt.timestep); freq = float(cpg["freq"])
    x0 = float(data.qpos[bq]); traj = []
    for t in range(steps):
        cphase = 2.0 * np.pi * freq * t * dt
        for k in range(graph.n_tokens):
            off = float(amp[k] * np.sin(cphase + phase[k]))
            tgt = q_def[k] + off
            tau = kp * (tgt - float(data.qpos[qadr[k]])) - kd * float(data.qvel[vadr[k]])
            data.ctrl[act_u[k]] = float(np.clip(tau, -clamps[k], clamps[k]))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            break
        if t % 50 == 0:
            traj.append(round(float(data.qpos[bq]) - x0, 3))
    return float(data.qpos[bq]) - x0, gate, traj


def mjx_rollout(mj, cpg, steps, iters=None, kp=32.0, kd=1.5):
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    if iters is not None:
        mj.opt.iterations = int(iters)
    mx = mjx.put_model(mj)
    graph, amp, phase, gate = _cpg_arrays(mj, cpg)
    qadr = jp.asarray(np.asarray(graph.qadr, int)); vadr = jp.asarray(np.asarray(graph.vadr, int))
    act_u = jp.asarray(np.asarray(graph.act_u, int)); clamps = jp.asarray(np.asarray(graph.clamps, float))
    amp = jp.asarray(amp); phase = jp.asarray(phase)
    bq = graph.base_qadr; dt = float(mj.opt.timestep); freq = float(cpg["freq"])
    NACON, NJMAX = 64, 256
    d0 = mjx.forward(mx, mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))
    q_def = d0.qpos[qadr]
    x0 = float(d0.qpos[bq])

    def step_fn(d, t):
        cphase = 2.0 * jp.pi * freq * t * dt
        off = amp * jp.sin(cphase + phase)
        tgt = q_def + off
        tau = kp * (tgt - d.qpos[qadr]) - kd * d.qvel[vadr]
        tau = jp.clip(tau, -clamps, clamps)
        ctrl = jp.zeros(mx.nu).at[act_u].set(tau)
        d = mjx.step(mx, d.replace(ctrl=ctrl))
        return d, d.qpos[bq]

    ts = jp.arange(steps, dtype=jp.float32)
    d_final, xtraj = jax.lax.scan(jax.jit(step_fn), d0, ts)
    xtraj = np.asarray(xtraj) - x0
    return float(np.asarray(d_final.qpos[bq]) - x0), bool(gate), [round(float(xtraj[i]), 3)
                                                                   for i in range(0, len(xtraj), 50)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-json", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--calf-phase", type=float, default=0.0)
    ap.add_argument("--freq", type=float, default=1.5)
    ap.add_argument("--skip-mjx", action="store_true", help="CPU-only (for local dev without jax/mjx)")
    args = ap.parse_args()
    cpg = dict(CPG); cpg["calf_phase"] = args.calf_phase; cpg["freq"] = args.freq
    gene = _load_gene(args.gene_json)
    cls = getattr(gene, "robot_class", "?")
    print(f"body={cls} calf_phase={cpg['calf_phase']} freq={cpg['freq']} steps={args.steps}", flush=True)

    xa, ga, ta = cpu_rollout(_cpu_model(gene), cpg, args.steps)
    print(f"A  CPU robot_mjcf   iters=20  net_x={xa:+.3f}  gate={ga}  traj={ta}", flush=True)
    xb, gb, tb = cpu_rollout(_mjx_model(gene), cpg, args.steps)
    print(f"B  CPU physics_only iters=20  net_x={xb:+.3f}  gate={gb}  traj={tb}", flush=True)

    if not args.skip_mjx:
        xc, gc, tc = mjx_rollout(_mjx_model(gene), cpg, args.steps, iters=None)
        print(f"C  MJX physics_only default   net_x={xc:+.3f}  gate={gc}  traj={tc}", flush=True)
        xd, gd, td = mjx_rollout(_mjx_model(gene), cpg, args.steps, iters=20)
        print(f"D  MJX physics_only iters=20  net_x={xd:+.3f}  gate={gd}  traj={td}", flush=True)

        def sgn(x):
            return "+" if x > 0.02 else ("-" if x < -0.02 else "0")
        print(f"\nSIGNS  A={sgn(xa)} B={sgn(xb)} C={sgn(xc)} D={sgn(xd)}", flush=True)
        if sgn(xa) == "+" and sgn(xb) == "-":
            print("VERDICT: MODEL build flips direction (physics_only/forced-collide) -> fix the compile, not solver")
        elif sgn(xb) == "+" and sgn(xc) == "-" and sgn(xd) == "+":
            print("VERDICT: MJX solver iterations is the culprit -> Part A (pin iters=20 in trainer) FIXES it")
        elif sgn(xc) == "-" and sgn(xd) == "-":
            print("VERDICT: iterations NOT enough -> escalate to contact-manifold / elliptic-cone / fp study")
        else:
            print("VERDICT: inconclusive / no clean flip -> inspect trajectories")


if __name__ == "__main__":
    main()
