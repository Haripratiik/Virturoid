"""CPU black-box trainer (OpenAI-ES) for the morphology-conditioned policy — the CPU-first proof that
ONE MorphPolicy is *learnable* and improves locomotion across a SET of composed bodies, before the same
architecture is scaled with PPO on MJX/GPU. Gradient-free (ES) so it needs no autodiff/torch on CPU.

Objective = forward (+x) base travel + an upright bonus, meaned over the training bodies — so a single
parameter set is pushed to move several morphologies (the MetaMorph "train on a distribution" idea).
"""

from __future__ import annotations


def forward_score(gene, policy, *, steps: int = 220, alpha: float = 0.5) -> float:
    """Locomotion fitness for one body: the DENSE per-step gait reward accumulated by ``rollout_morph`` —
    per step, forward velocity GATED by height_ratio^2 (strongly punishes the low crouch-scoot local optimum)
    plus a stay-tall bonus, meaned over the horizon. Research (the locomotion-quality sweep + the deep-research
    reward template: tracking + upright + alive − energy − smoothness) showed this dense, crouch-punishing,
    SUSTAINED-motion reward yields an upright walk where the old TERMINAL ``forward*height`` settled into a fast
    low crouch. Heavy penalty for blowups."""
    from virturoid.services.morph_policy import rollout_morph
    r = rollout_morph(gene, policy, steps=steps, alpha=alpha)
    if not r["finite"]:
        return -1.0
    return float(r.get("gait", 0.0))


def recipe_forward_score(gene, policy, *, steps: int = 900, normalizer=None, kp: float = 32.0,
                         kd: float = 1.5, ascale: float = 0.4, vtgt: float = 0.3, adaptive: bool = False,
                         cpg: dict | None = None) -> float:
    """Locomotion fitness under the ANTI-COLLAPSE RECIPE (PD-to-default control + obs-norm + terminate/alive +
    velocity-tracking) — the rollout that actually WALKS, as opposed to the torque-residual ``forward_score``
    that collapses to a fall-forward flop. Same signature shape as ``forward_score`` so the ES loop is shared.
    ``adaptive=True`` trains under inertia-scaled per-joint gains (recipe_gains) so the fitness the ES optimizes
    matches the gains deploy will use."""
    from virturoid.services.morph_policy import recipe_rollout_morph
    r = recipe_rollout_morph(gene, policy, steps=steps, normalizer=normalizer, kp=kp, kd=kd,
                             ascale=ascale, vtgt=vtgt, adaptive=adaptive, cpg=cpg)
    if not r["finite"]:
        return -1.0
    return float(r.get("gait", 0.0))


# --- BATCHED in-process population rollout ("the same sim, many times") --------------------------------
# Profiling the ES loop (see the batched-rollout study) found physics is only ~9% of a step — the cost is
# the XML recompile (~25%, now cached) and the PER-CANDIDATE Python policy forward. So the efficient CPU
# primitive isn't `mujoco.rollout` (threaded physics) — it's running the SAME body's sim for the WHOLE
# population in ONE process: compile once, step `B` MjData in lockstep, and collapse the B candidate policy
# forwards into ONE vectorized batched matmul (B different weight sets via batched einsum). This is the CPU
# analog of MJX's vmap-over-policies, and it composes with multi-core (each worker batches its chunk) and
# the GPU/MJX path (the big jobs). It's numerically equivalent to the serial forward_score (same physics,
# same math, rounded the same) — not bit-identical over long horizons (the batched matmul reorders floats,
# which chaos can amplify), so the bit-identical determinism guarantee stays with the serial/parallel path.
def _stack_params(param_batch, feature_dim: int, hidden: int = 32):
    """Stack B flat MorphPolicy param vectors into batched weight tensors (leading batch dim)."""
    import numpy as np

    from virturoid.services.morph_policy import MorphPolicy
    P = np.asarray(param_batch, dtype=float)
    B = P.shape[0]
    tmpl = MorphPolicy(feature_dim, hidden=hidden)
    W, i = {}, 0
    for k in MorphPolicy._ORDER:
        shp = tmpl._arrs[k].shape
        nsz = int(np.prod(shp)) if shp else 1
        W[k] = P[:, i:i + nsz].reshape((B,) + shp)
        i += nsz
    return W


def _act_batch(W, obs_batch):
    """Batched proprioceptive forward: B policies (own weights) each on its own (N, feature_dim) obs ->
    (B, N) actions. Mirrors MorphPolicy.act exactly (perception token off, as forward_score never senses)."""
    import numpy as np

    e = np.tanh(np.einsum("bnf,bfh->bnh", obs_batch, W["We"]) + W["be"][:, None, :])
    q = np.einsum("bnh,bhk->bnk", e, W["Wq"])
    k = np.einsum("bnh,bhk->bnk", e, W["Wk"])
    v = np.einsum("bnh,bhk->bnk", e, W["Wv"])
    h = e.shape[-1]
    sc = np.einsum("bnk,bmk->bnm", q, k) / np.sqrt(h)
    sc -= sc.max(axis=-1, keepdims=True)
    att = np.exp(sc)
    att /= att.sum(axis=-1, keepdims=True)
    z = np.einsum("bnh,bhk->bnk", np.einsum("bnm,bmh->bnh", att, v), W["Wo"])
    u = np.tanh(e + z)
    out = np.tanh(np.einsum("bnh,bhk->bnk", u, W["Wh"]) + W["bh"][:, None, :])
    return out[..., 0]


def batched_forward_scores(gene, param_batch, *, steps: int = 220, alpha: float = 0.5):
    """Evaluate a BATCH of policy param vectors on one body by running the SAME sim many times in ONE
    process: model compiled once, B envs stepped in lockstep, the population's policy forward done as a
    SINGLE vectorized op. Returns the per-row forward_score (numerically equal to the serial path)."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import compiled_model, robot_mjcf

    model = compiled_model(robot_mjcf(gene), solver_iterations=20)
    graph = encode_robot(model)
    B = len(param_batch)
    if graph.n_tokens == 0:
        return np.full(B, 0.05)                              # no actuated joints -> standing-tall bonus only
    W = _stack_params(param_batch, graph.feature_dim, hidden=32)   # MorphPolicy default hidden width
    act_u = np.asarray(graph.act_u, dtype=int)
    vadr = np.asarray(graph.vadr, dtype=int)
    clamps = np.asarray(graph.clamps, dtype=float)
    bq, bjid = graph.base_qadr, graph.base_jid

    datas = [mujoco.MjData(model) for _ in range(B)]
    for d in datas:
        mujoco.mj_resetData(model, d)
        mujoco.mj_forward(model, d)
    p0 = np.array([[d.qpos[bq], d.qpos[bq + 1]] for d in datas]) if bjid >= 0 else np.zeros((B, 2))
    z0 = np.array([float(d.qpos[bq + 2]) for d in datas]) if bjid >= 0 else np.zeros(B)
    alive = np.ones(B, dtype=bool)
    finite = np.ones(B, dtype=bool)
    # DENSE gait reward per env, matching rollout_morph's track_smooth template: per step, velocity-tracking
    # (exp(-(vx-v_target)^2/0.25)) gated by upright + a stay-tall bonus - an action-SMOOTHNESS penalty.
    dt = float(model.opt.timestep)
    z0s = np.where(z0 > 1e-6, z0, 1.0)
    prev_x = p0[:, 0].astype(float).copy()
    gait = np.zeros(B)
    prev_act = None
    v_target = 0.5

    for _ in range(steps):
        obs = np.zeros((B, graph.n_tokens, graph.feature_dim))
        for i, d in enumerate(datas):
            if alive[i]:
                obs[i] = graph.observe(model, d)
        acts = _act_batch(W, obs)                            # (B, N) — the whole population in one op
        for i, d in enumerate(datas):
            if not alive[i]:
                continue
            res = np.clip(acts[i], -1.0, 1.0) * alpha * clamps
            d.ctrl[act_u] = np.clip(d.qfrc_bias[vadr] + res, -clamps, clamps)
            mujoco.mj_step(model, d)
            if not np.all(np.isfinite(d.qpos)):
                alive[i] = False
                finite[i] = False
                continue
            if bjid >= 0:
                x = float(d.qpos[bq]); z = float(d.qpos[bq + 2])
                hr_t = min(1.0, max(0.0, z / z0s[i]))
                vx = (x - prev_x[i]) / dt
                up = 1.0 if z > 0.7 * z0s[i] else 0.0
                smooth = 0.0 if prev_act is None else float(np.mean((acts[i] - prev_act[i]) ** 2))
                gait[i] += float(np.exp(-((vx - v_target) ** 2) / 0.25)) * up + 0.3 * hr_t - 0.5 * smooth
                prev_x[i] = x
        prev_act = acts

    scores = np.empty(B)
    for i in range(B):
        scores[i] = -1.0 if not finite[i] else round(float(gait[i] / max(1, steps)), 4)   # matches forward_score
    return scores


# --- PARALLEL ES population evaluation ------------------------------------------------------------------
# The ES population is embarrassingly parallel: each candidate is an independent rollout. We fan the
# `pop` candidate evaluations out across CPU cores (each candidate's forward_score over the training
# bodies runs in its own process), so the SAME budget trains ~Ncores faster — or, in the same wall-clock,
# a much bigger pop/generation count trains a stronger policy. The eps are still drawn in the PARENT, so
# the result is bit-for-bit the same as the serial path (just computed in parallel). Worker state (the
# training bodies) is set ONCE per process via the Pool initializer, so only the small param vectors are
# shipped per task. Custom fitness_fn (the perception avoid task) stays serial — its closure isn't picklable.
_W: dict = {}


def _init_es_worker(gene_dicts, feature_dim, steps, alpha, recipe=False, norm=None, adaptive=False, cpg=None):
    import warnings
    warnings.filterwarnings("ignore")
    from virturoid.schemas.gene import RobotGene
    _W["genes"] = [RobotGene.from_dict(d) for d in gene_dicts]
    _W["fd"] = int(feature_dim)
    _W["steps"] = int(steps)
    _W["alpha"] = float(alpha)
    _W["recipe"] = bool(recipe)            # anti-collapse RECIPE fitness (PD/obs-norm/terminate/vel-track)
    _W["norm"] = norm                       # (mean, std) obs normalizer, banked with the policy
    _W["adaptive"] = bool(adaptive)         # inertia-scaled per-joint gains (recipe_gains) — the "any robot" path
    _W["cpg"] = cpg                          # trot-CPG prior params (the stepping rhythm ES refines a residual on)


def _es_eval_params(flat_params):
    import numpy as np

    from virturoid.services.morph_policy import MorphPolicy
    pol = MorphPolicy(_W["fd"])
    pol.set_params(np.asarray(flat_params, dtype=float))
    if _W.get("recipe"):
        return float(np.mean([recipe_forward_score(g, pol, steps=_W["steps"], normalizer=_W.get("norm"),
                                                   adaptive=_W.get("adaptive", False), cpg=_W.get("cpg"))
                              for g in _W["genes"]]))
    return float(np.mean([forward_score(g, pol, steps=_W["steps"], alpha=_W["alpha"]) for g in _W["genes"]]))


def _make_es_pool(genes, feature_dim, steps, alpha, workers, recipe=False, norm=None, adaptive=False, cpg=None):
    """A spawn Pool of ``workers`` processes, each pre-loaded with the training bodies. None on any failure
    (caller falls back to serial) so parallel training never blocks a build."""
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        return ctx.Pool(processes=int(workers), initializer=_init_es_worker,
                        initargs=([g.to_dict() for g in genes], feature_dim, steps, alpha, recipe, norm, adaptive, cpg))
    except Exception:  # noqa: BLE001
        return None


def train_morph_es(genes, *, feature_dim: int, generations: int = 12, pop: int = 16, sigma: float = 0.15,
                   lr: float = 0.06, steps: int = 220, alpha: float = 0.5, seed: int = 0, init_policy=None,
                   fitness_fn=None, workers: int = 1, eval_backend: str = "auto", recipe: bool = False,
                   normalizer=None, adaptive: bool = False, cpg: dict | None = None):
    """OpenAI-ES on the flat MorphPolicy params. Returns ``(best_policy, history)`` where history is the
    mean fitness per generation. ``genes`` is the body distribution to train one policy across.
    ``init_policy`` warm-starts from a banked policy (the flywheel) so a SIMILAR species trains faster.
    ``fitness_fn(policy, genes)`` overrides the default forward-locomotion objective (used by
    ``train_morph_avoid`` for the perception sense-and-avoid task) — the ES loop + elitism are shared.

    ``eval_backend`` selects how the population is scored each generation:
      * ``"batched"`` — run the SAME sim for the whole population in ONE process (compile once, B envs in
        lockstep, the policy forward as a single vectorized op); the efficient single-core primitive.
      * ``"parallel"`` — fan the population across ``workers`` CPU processes (bit-identical to serial).
      * ``"serial"`` — one candidate at a time.
      * ``"auto"`` (default) — ``parallel`` when ``workers>1``, else ``serial``; this keeps the default path
        BIT-IDENTICAL serial-vs-parallel. ``batched`` is opt-in (numerically equal, not bit-identical)."""
    import numpy as np

    from virturoid.services.morph_policy import MorphPolicy

    rng = np.random.default_rng(seed)
    policy = MorphPolicy(feature_dim, seed=seed)
    if init_policy is not None:
        prior_dim = getattr(init_policy, "feature_dim", None)
        if prior_dim == feature_dim:
            policy.set_params(init_policy.get_params())      # warm-start from prior learning
        else:
            import warnings
            warnings.warn(
                f"warm-start rejected: prior feature_dim={prior_dim!r}, requested feature_dim={feature_dim}; "
                "training starts cold instead of silently pretending the prior was reused",
                RuntimeWarning,
                stacklevel=2,
            )
    theta = policy.get_params()
    n = theta.size

    if eval_backend == "auto":
        eval_backend = "parallel" if (workers and workers > 1) else "serial"
    use_parallel = eval_backend == "parallel" and workers and workers > 1 and fitness_fn is None
    use_batched = eval_backend == "batched" and fitness_fn is None and not recipe   # batched is residual-only
    pool = (_make_es_pool(genes, feature_dim, steps, alpha, workers, recipe, normalizer, adaptive, cpg)
            if use_parallel else None)

    def fitness(vec) -> float:
        policy.set_params(vec)
        if fitness_fn is not None:
            return float(fitness_fn(policy, genes))
        if recipe:                                           # anti-collapse RECIPE locomotion fitness
            return float(np.mean([recipe_forward_score(g, policy, steps=steps, normalizer=normalizer,
                                                       adaptive=adaptive, cpg=cpg) for g in genes]))
        return float(np.mean([forward_score(g, policy, steps=steps, alpha=alpha) for g in genes]))

    def eval_pop(cands) -> "np.ndarray":
        if pool is not None:
            try:
                return np.array(pool.map(_es_eval_params, [c for c in cands]))
            except Exception:  # noqa: BLE001 - a worker died -> finish this run serially
                pass
        if use_batched:
            try:
                return np.mean([batched_forward_scores(g, cands, steps=steps, alpha=alpha) for g in genes], axis=0)
            except Exception:  # noqa: BLE001 - any batched hiccup -> serial
                pass
        return np.array([fitness(c) for c in cands])

    # Elitism: ES is noisy and can drift DOWN over a generation (especially warm-started, where theta
    # already starts near a good point). We keep the BEST theta seen and return that — so training can
    # never hand back a policy worse than where it began. This makes the flywheel monotonic: warm-starting
    # from a banked policy yields a result at least as good as the seed, never a regression (capstone bug:
    # a hexapod warm-started at 0.174 and a non-elitist ES returned a degraded 0.158).
    f0 = fitness(theta)
    history = [f0]
    best_theta, best_fit = theta.copy(), f0
    try:
        for _ in range(generations):
            eps = rng.normal(0, 1, (pop, n))
            fits = eval_pop(theta + sigma * eps)                  # PARALLEL over the population (or serial)
            adv = (fits - fits.mean()) / (fits.std() + 1e-8)     # fitness-shaped advantage
            theta = theta + (lr / (pop * sigma)) * (eps.T @ adv)
            f = fitness(theta)
            history.append(f)
            if f > best_fit:
                best_theta, best_fit = theta.copy(), f
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    policy.set_params(best_theta)
    return policy, history


def avoid_score(gene, policy, *, walls=None, goal_xy=(2.0, 0.0), n_rays: int = 8, max_range: float = 3.0,
                steps: int = 300, alpha: float = 0.5) -> float:
    """Sense-and-avoid fitness for ONE legged body in an UNKNOWN environment (perception Rung A).

    The body is driven by ``policy.act(obs, ranges, cmd)`` where ``ranges`` is the live rangefinder ring and
    ``cmd`` is a goal-biased velocity hint derived from the SENSED ranges (reactive_command). Reward = progress
    toward the goal (only while UPRIGHT — a fallen body stops earning, so it can't ragdoll into the goal) +
    a small stay-in-the-open bonus. Avoidance emerges from progress: walking into a wall blocks progress while
    skirting it in open space keeps earning, so the policy is pushed to USE the sensed ranges. No absolute
    range penalty — that would trip on the rangefinders sensing the body's OWN legs/the ground when it tips.
    Needs MuJoCo. Reuses exteroception (no new sim)."""
    import math

    import mujoco
    import numpy as np

    from virturoid.services.exteroception import read_ranges, with_rangefinders
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.reactive_nav import reactive_command

    model = mujoco.MjModel.from_xml_string(with_rangefinders(gene, walls=walls, n_rays=n_rays))
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    graph = encode_robot(model)
    if graph.base_jid < 0 or graph.n_tokens == 0:
        return -1.0
    bq = graph.base_qadr
    ray_ang = np.array([2 * math.pi * i / n_rays for i in range(n_rays)])
    goal = np.asarray(goal_xy, float)

    def _yaw():
        q = data.qpos[bq + 3:bq + 7]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    z0 = float(data.qpos[bq + 2])                                    # standing base height (uprightness gate)
    prev_gd = float(np.linalg.norm(goal - np.asarray(data.qpos[bq:bq + 2], float)))
    reward = 0.0
    for _ in range(steps):
        pos = np.asarray(data.qpos[bq:bq + 2], float)
        to = goal - pos
        goal_bearing = math.atan2(math.sin(math.atan2(to[1], to[0]) - _yaw()),
                                  math.cos(math.atan2(to[1], to[0]) - _yaw()))
        ranges = read_ranges(model, data, n_rays=n_rays, max_range=max_range)
        vx_cmd, wz_cmd = reactive_command(ranges, ray_ang, goal_bearing, max_range=max_range)
        obs = graph.observe(model, data)
        graph.apply(model, data, policy.act(obs, ranges=ranges, cmd=[vx_cmd, wz_cmd]), alpha=alpha)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return reward - 5.0                                      # blew up: heavy penalty
        if float(data.qpos[bq + 2]) < 0.4 * z0:                     # fell over -> stops earning (no ragdoll)
            break
        gd = float(np.linalg.norm(goal - np.asarray(data.qpos[bq:bq + 2], float)))
        reward += (prev_gd - gd)                                     # progress toward the goal (while upright)
        reward += 0.02 * float(np.mean(np.clip(ranges / max_range, 0.0, 1.0)))   # mild prefer-open bonus
        prev_gd = gd
    return float(reward)


def train_morph_avoid(genes, *, walls=None, goal_xy=(2.0, 0.0), feature_dim: int, n_rays: int = 8,
                      generations: int = 6, pop: int = 10, sigma: float = 0.15, lr: float = 0.06,
                      steps: int = 250, alpha: float = 0.5, seed: int = 0, init_policy=None):
    """ES-train a PERCEPTIVE MorphPolicy on the unknown-environment sense-and-avoid task — reusing the same
    (elitist) ES loop as locomotion via ``fitness_fn``. ``n_rays`` is standardized (8) so the perception
    weights transfer across morphologies just like the proprioceptive ones."""
    def fit(policy, gs):
        import numpy as np
        return float(np.mean([avoid_score(g, policy, walls=walls, goal_xy=goal_xy, n_rays=n_rays,
                                          steps=steps, alpha=alpha) for g in gs]))

    return train_morph_es(genes, feature_dim=feature_dim, generations=generations, pop=pop, sigma=sigma,
                          lr=lr, steps=steps, alpha=alpha, seed=seed, init_policy=init_policy, fitness_fn=fit)
