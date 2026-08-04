"""The bench rig: the ONE physical setup every Stage-1 measurement is taken in, so the excitation the engineer
runs, the replay we compare against it, and the sim2sim gate are all the same experiment.

Three decisions are baked in here, and each is a claim about the real world:

**The robot is on a stand, not on the floor.** Every free base is welded and the floor is removed. That is the
setup a bench sysid run actually uses -- legs hanging, nothing to fall off -- and it is also the setup that
ISOLATES what we are measuring. Hwangbo et al. attribute the dominant sim-to-real residual to actuator dynamics
and control-signal delay; contact is a different (and much harder) identification problem. A gap number measured
with feet on the ground would confound the two and could not name which one moved. So: no contact, by design,
and the report says the contact gap is NOT measured rather than pretending it was folded in.

**The controller is ours, and it is declared.** The joints are driven by a PD law whose gains are derived from
each DOF's own joint-space inertia so the closed loop has the same natural frequency on every morphology -- the
same principle as ``morph_policy.recipe_gains`` (which cannot be reused directly: it is keyed to morph-graph
TOKENS, and the bench model has no free base, so the token/DOF indices do not line up). The gains ship inside the
excitation plan. That matters twice over: an engineer cannot reproduce our number without them, and the delay
search in ``gap_report`` re-simulates the closed loop, which is only valid because we know what loop they ran.

**Torque is what the actuator applied, not what the controller asked for.** ``pd_replay`` records the commanded
torque AFTER the ``forcerange`` clamp and AFTER the delay buffer, because that is what a motor driver reports and
what an engineer logs off a CAN bus. Logging the pre-clamp command would make a saturating joint look like a
modelling error.
"""

from __future__ import annotations

# Closed-loop natural frequency of the bench PD, rad/s. 12 rad/s ~ 1.9 Hz: comfortably above the excitation band
# so the controller tracks it, comfortably below the joint's structural modes so the rig is not exciting the body.
BENCH_WN_RADPS = 12.0
BENCH_ZETA = 1.0                    # critically damped -- no overshoot into a joint limit
_DEFAULT_JOINT_LIMIT_RAD = 3.1416   # matches control_script_compiler._derive's fallback


def bench_model(gene):
    """Compile ``gene`` into the bench model: base WELDED, floor REMOVED.

    Returns ``(model, meta)``. ``meta['fixed_base']`` records whether we had to weld a free base, because a
    number measured on a welded quadruped is not a number measured on a walking one and the report must say so.
    """
    import dataclasses

    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    was_free = getattr(gene, "base_mount", "") == "free"
    bench_gene = dataclasses.replace(gene, base_mount="floor") if was_free else gene
    xml = compile_gene_to_mjcf(bench_gene, include_floor=False)
    # NOT morph_policy.compiled_model: that cache hands back a SHARED MjModel, and the sensitivity columns below
    # perturb dof_damping/frictionloss/armature in place. One mutated cache entry would silently change the
    # dynamics of every other rollout in the process.
    model = mujoco.MjModel.from_xml_string(xml)
    return model, {"fixed_base": True, "welded_free_base": was_free,
                   "floor": False, "contact_gap_measured": False}


def joint_dof_map(model, gene) -> dict:
    """``{segment_name: dof_index}`` for every actuated joint, resolved by NAME rather than by position.

    Order-independence is not pedantry here: the actuator list, the gene's segment list and the model's DOF list
    are built by three different passes, and a silent off-by-one would attribute one joint's gap to another --
    the single most damaging way this report could lie.
    """
    import mujoco

    out: dict = {}
    for seg in gene.actuated_joints():
        jid = -1
        for cand in (f"{seg.name}_joint", seg.name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cand)
            if jid >= 0:
                break
        if jid >= 0:
            out[seg.name] = int(model.jnt_dofadr[jid])
    return out


def bench_gains(model):
    """Per-DOF ``(kp, kd)`` from the joint-space inertia at the start pose, so the closed-loop natural frequency
    is the same on a 0.5 kg wrist and a 40 kg hip. Same principle as ``morph_policy.recipe_gains``; computed on
    model DOFs because the bench model has no free base and therefore no morph-graph token alignment."""
    import mujoco
    import numpy as np

    nv = model.nv
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    full = np.zeros((nv, nv))
    mujoco.mj_fullM(model, full, data.qM)
    inertia = np.maximum(np.diag(full).copy(), 1e-9)
    kp = inertia * (BENCH_WN_RADPS ** 2)
    kd = 2.0 * BENCH_ZETA * np.sqrt(np.maximum(kp * inertia, 1e-12))
    return kp, kd, inertia


def torque_ceiling(model):
    """Per-DOF |torque| ceiling the compiled model will actually enforce (``forcerange``), in N.m."""
    import numpy as np

    nv = model.nv
    ceil = np.full(nv, np.inf)
    for a in range(model.nu):
        trn = int(model.actuator_trnid[a, 0])
        adr = int(model.jnt_dofadr[trn])
        if 0 <= adr < nv:
            ceil[adr] = abs(float(model.actuator_forcerange[a, 1]))
    ceil[~np.isfinite(ceil)] = 1e6
    return ceil


#: A joint's start pose is pulled at least this fraction of the safe band's half-width in from either end, so
#: there is always room to swing symmetrically about it.
MIN_HEADROOM_FRAC = 0.25


def rest_pose(model):
    """The robot's own rest stance from the model's first keyframe, or zeros when it carries none."""
    import numpy as np

    nv = model.nv
    if model.nkey and model.key_qpos.shape[1] == model.nq and model.nq == nv:
        return np.array(model.key_qpos[0][:nv], dtype=float)
    return np.zeros(nv)


def start_pose(model, gene, backoff: float = 0.15, min_headroom_frac: float = MIN_HEADROOM_FRAC):
    """The pose the bench experiment starts and returns to: the robot's rest stance, moved as little as
    necessary to leave room to swing.

    The subtlety that bit: clipping the rest pose to the EDGE of the safe band leaves zero headroom, and a joint
    with zero headroom gets a zero amplitude and is dropped from the experiment. On the composed quadruped that
    silently removed all four knees -- the joints with the widest travel on the machine (2.6 rad) and the best
    excitation candidates -- because their rest angle of 0.0 sits just outside a 15% back-off of [-0.2, 2.4].
    The report said "not excitable: amplitude", which reads like a fact about the robot and was a fact about
    this function. So the clip is into an INNER band that guarantees headroom, and ``build_excitation`` tells
    the engineer which joints it had to move off the rest stance to position.
    """
    import numpy as np

    lo, hi = safe_band(model, gene, backoff=backoff)
    inset = float(min_headroom_frac) * 0.5 * (hi - lo)
    return np.clip(rest_pose(model), lo + inset, hi - inset)


def safe_band(model, gene, backoff: float = 0.15):
    """``(lo, hi)`` per DOF: the joint's declared range shrunk by ``backoff`` of its width at each end.

    The range comes from the GENE (what the customer declared) intersected with the compiled model's own
    ``jnt_range``, so a limit the compiler tightened cannot be exceeded by an excitation authored off the gene.
    """
    import numpy as np

    nv = model.nv
    lo = np.full(nv, -_DEFAULT_JOINT_LIMIT_RAD)
    hi = np.full(nv, _DEFAULT_JOINT_LIMIT_RAD)
    for j in range(model.njnt):
        adr = int(model.jnt_dofadr[j])
        if adr >= nv:
            continue
        if bool(model.jnt_limited[j]):
            lo[adr], hi[adr] = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
    dofs = joint_dof_map(model, gene)
    for seg in gene.actuated_joints():
        adr = dofs.get(seg.name)
        if adr is None:
            continue
        if seg.joint_lower is not None:
            lo[adr] = max(lo[adr], float(seg.joint_lower))
        if seg.joint_upper is not None:
            hi[adr] = min(hi[adr], float(seg.joint_upper))
    mid = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo) * (1.0 - 2.0 * float(backoff)), 0.0)
    return mid - half, mid + half


def pd_replay(model, q_cmd_log, *, kp, kd, q_start, ctrl_every: int, delay_ticks: int = 0):
    """Drive ``model`` with the bench PD following ``q_cmd_log`` (one row per LOG sample) and return the log an
    engineer would record: ``(t, q, qd, tau)``, all at the log rate.

    ``ctrl_every`` log rows elapse between control updates -- the controller runs slower than the physics, as it
    does on hardware. ``delay_ticks`` buffers whole CONTROL ticks between computing a torque and applying it:
    that is Hwangbo's control-signal delay, and it is applied to the torque rather than to the setpoint because a
    delayed setpoint is indistinguishable from a clock offset and would not be an actuator property at all.
    """
    import mujoco
    import numpy as np

    nv = model.nv
    dt = float(model.opt.timestep)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:nv] = q_start
    mujoco.mj_forward(model, data)

    ceil = torque_ceiling(model)
    dof2act = {}
    for a in range(model.nu):
        dof2act[int(model.jnt_dofadr[int(model.actuator_trnid[a, 0])])] = a

    n = int(np.asarray(q_cmd_log).shape[0])
    q = np.zeros((n, nv)); qd = np.zeros((n, nv)); tau = np.zeros((n, nv))
    # ``delay_ticks`` zero-torque ticks are pre-loaded, so delay_ticks=0 applies the freshly computed torque
    # immediately. An off-by-one here cancels between our own log and our own replay and would go unnoticed --
    # until it met a real log, where it would add a phantom control tick of latency to every joint.
    buf = [np.zeros(nv) for _ in range(int(delay_ticks))]
    applied = np.zeros(nv)
    for k in range(n):
        if k % int(ctrl_every) == 0:
            want = np.clip(kp * (q_cmd_log[k] - data.qpos[:nv]) - kd * data.qvel[:nv], -ceil, ceil)
            buf.append(want)
            applied = buf.pop(0)
        for adr, a in dof2act.items():
            data.ctrl[a] = applied[adr]
        q[k] = data.qpos[:nv]; qd[k] = data.qvel[:nv]; tau[k] = applied
        mujoco.mj_step(model, data)
    return np.arange(n) * dt, q, qd, tau


def inverse_torque(model, q, qd, qacc):
    """``qfrc_inverse``: the generalized force OUR model says is needed to produce the logged motion.

    This is the load-bearing call. Its self-consistency is exact -- replaying a rollout's own (q, qd, qacc)
    reproduces the applied torque to ~2e-15 N.m (``tests/test_sysid_stage1.py``) -- which is what lets the
    residual ``tau_measured - inverse_torque`` be read as "the physics our model is missing" rather than as
    numerical slop.
    """
    import mujoco
    import numpy as np

    nv = model.nv
    data = mujoco.MjData(model)
    out = np.zeros((int(np.asarray(q).shape[0]), nv))
    for k in range(out.shape[0]):
        data.qpos[:nv] = q[k]; data.qvel[:nv] = qd[k]; data.qacc[:nv] = qacc[k]
        mujoco.mj_inverse(model, data)
        out[k] = data.qfrc_inverse[:nv]
    return out


def central_derivative(series, dt):
    """Central-difference time derivative of a logged signal (velocity -> acceleration, position -> velocity).

    Deliberately the plainest estimator available, because it is the one an engineer's log will support -- real
    logs carry position and velocity, not acceleration. It is also, MEASURED, the single largest error source in
    the whole Stage-1 estimate: feeding the simulator's own ``qacc`` recovers an injected perturbation EXACTLY
    (0.00000 on all three parameters), while this central difference leaves a residual bias of ~0.017 N.m on
    frictionloss. That bias is not hidden -- ``gap_report`` measures it per joint as the noise floor and refuses
    to report any estimate underneath it. A 9-point Savitzky-Golay derivative was tried and was WORSE (0.030),
    because it smooths across the stick-slip transitions that carry the friction signal.
    """
    import numpy as np

    series = np.asarray(series, dtype=float)
    out = np.zeros_like(series)
    if series.shape[0] < 3:
        return out
    out[1:-1] = (series[2:] - series[:-2]) / (2.0 * float(dt))
    out[0], out[-1] = out[1], out[-2]
    return out
