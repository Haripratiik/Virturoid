"""MorphPolicy: a permutation-equivariant, morphology-agnostic policy over the kinematic graph.

A SHARED MLP applied to every joint token from ``morph_graph`` — so a single parameter set drives a
3-DOF arm, an 8-DOF quadruped, or a 24-DOF spider (the token count varies; the per-token function does
not). This is the MetaMorph idea in its simplest permutation-equivariant form (Theme 1). NumPy forward
for CPU-first validation + black-box (CMA/ES) tuning; the learnable JAX/Flax transformer + PPO replace
the forward on the GPU box behind the SAME ``observe``/``apply`` interface.
"""

from __future__ import annotations


class MorphPolicy:
    """Per-token embed → ONE self-attention block (tokens coordinate) → per-token action head. Attention
    is permutation-equivariant and handles any token count, so one parameter set drives any morphology;
    the attention + positional feature let symmetric limbs break symmetry (phase a gait). Same forward
    on CPU (NumPy, here) and GPU (Flax, in ``scripts/mjx_morph_ppo``)."""

    # Exteroception: a fixed-width perception vector (a ring of n_rays=8 range sensors + a 2-slot
    # velocity command) fed through ONE extra global token. Fixed across morphologies so the perception
    # weights transfer like the rest of the policy (see [[perception-next-keystone]]).
    PERCEPT_DIM = 10                                          # 8 rangefinder distances + (vx_cmd, wz_cmd)

    def __init__(self, feature_dim: int, hidden: int = 32, seed: int = 0, *,
                 film: bool = False, topo_bias: bool = False, topo_buckets: int = 8):
        import numpy as np

        self.feature_dim = feature_dim
        self.hidden = hidden
        self.percept_dim = self.PERCEPT_DIM
        # Phase-5 tokenizer upgrades, OPT-IN so default policies are byte-identical (and n_params unchanged):
        #   film  -> FiLM joint-attribute conditioning (2603.00182): (1+gamma)*e + beta from the obs datasheet
        #   topo_bias -> topology-aware attention bias theta_{d(i,j)} on kinematic hop distance
        # Both weights are ZERO-init, so even when ON-but-untrained the forward is identical to OFF; GPU/ES
        # training moves them. Extra weights are appended LAST + tracked in an instance order, so a default
        # policy's parameter vector + n_params are exactly as before (banked policies load unchanged).
        self.film = bool(film)
        self.topo_bias = bool(topo_bias)
        self.topo_buckets = int(topo_buckets)
        # Anti-collapse RECIPE deploy metadata, banked WITH the policy (see recipe_rollout_morph): a non-None
        # obs_mean marks this as a recipe (PD-to-default + obs-norm) policy, so the deploy/replay paths drive it
        # with the SAME control + normalization it was trained under (a torque-residual rollout would NOT walk it).
        self.obs_mean = None
        self.obs_std = None
        # ADAPTIVE GAINS flag, banked WITH the policy (see recipe_gains): when True, every deploy/replay path
        # derives per-joint Kp/Kd from the body's effective inertia instead of the scalar 32/1.5 — so the recipe
        # that was tuned on the reference quadruped also walks a humanoid/heavy/spider body (the "any robot" leg).
        self.adaptive_gains = False
        self.cpg = None                                       # trot-CPG prior params (set when CPG-trained; see CPG_DEFAULT)
        self.decimation = 1                                   # control decimation (plan v2 T1.1): deploy hold-D-steps
        self.action_lpf = 0.0                                 # action EMA low-pass (plan v2 T1.2); 0 = off
        self.sphere_feet = False                              # sphere feet + feet-only collision (plan v2 T1.4); off = untouched
        rng = np.random.default_rng(seed)
        s = 0.3
        # NOTE: Wrange/brange are appended LAST so the rng draw order for We/Wq/Wk/Wv/Wo/Wh is unchanged
        # — a fresh policy's proprioceptive weights are bit-identical to before perception existed.
        self._arrs = {
            "We": rng.normal(0, s, (feature_dim, hidden)), "be": np.zeros(hidden),
            "Wq": rng.normal(0, s, (hidden, hidden)), "Wk": rng.normal(0, s, (hidden, hidden)),
            "Wv": rng.normal(0, s, (hidden, hidden)), "Wo": rng.normal(0, s, (hidden, hidden)),
            "Wh": rng.normal(0, s, (hidden, 1)), "bh": np.zeros(1),
            # Perception encoder: maps the percept vector straight into HIDDEN space (not feature_dim),
            # so the global perception token enters the attention block WITHOUT widening per-token obs.
            "Wrange": rng.normal(0, s, (self.PERCEPT_DIM, hidden)), "brange": np.zeros(hidden),
        }
        # Instance parameter order = base order (+ opt-in weights, appended last, zero-init = identity when off).
        self._order = list(self._ORDER)
        if self.film:
            self._arrs["Wfilm"] = np.zeros((feature_dim, 2 * hidden))   # -> (gamma, beta) per joint token
            self._order.append("Wfilm")
        if self.topo_bias:
            self._arrs["Wtopo"] = np.zeros(self.topo_buckets + 1)       # learned bias per hop distance 0..buckets
            self._order.append("Wtopo")

    def act(self, obs, ranges=None, cmd=None, hop=None):
        """(n_tokens, feature_dim) -> (n_tokens,) actions in [-1, 1].

        With ``ranges`` (a length-``n_rays`` distance vector) and optional ``cmd`` ([vx, wz] goal-velocity
        bias), a single GLOBAL perception token is appended and mixed through the SAME attention block, so
        the body can act on what it SENSES. ``ranges=None`` is byte-identical to the proprioception-only
        forward (perception-blind), so locomotion and every banked policy behave exactly as before.

        ``hop`` (an ``n_tokens x n_tokens`` kinematic hop-distance matrix, from ``topo_pe.hop_distance_matrix``)
        adds the topology-aware attention bias when this policy was built with ``topo_bias`` — else ignored.
        Both Phase-5 upgrades (FiLM, topo-bias) are identity when their weights are zero, so an off / untrained
        policy is byte-identical to before."""
        import numpy as np

        if obs.shape[0] == 0:
            return np.zeros(0)
        a = self._arrs
        e = np.tanh(obs @ a["We"] + a["be"])                 # (N, H) per-token embedding
        if self.film and "Wfilm" in a:                       # FiLM: modulate each token by its own datasheet
            gb = obs @ a["Wfilm"]                            # (N, 2H) -> (gamma, beta); zero-init -> identity
            e = (1.0 + gb[:, :self.hidden]) * e + gb[:, self.hidden:]
        if ranges is not None:
            r = np.asarray(ranges, dtype=float).ravel()
            c = np.asarray(cmd if cmd is not None else [0.0, 0.0], dtype=float).ravel()
            percept = np.concatenate([r, c])
            percept = (percept[: self.percept_dim] if percept.size >= self.percept_dim
                       else np.concatenate([percept, np.zeros(self.percept_dim - percept.size)]))
            pt = np.tanh(percept @ a["Wrange"] + a["brange"])  # (H,) global perception token in hidden space
            e = np.vstack([e, pt[None, :]])                    # (N+1, H): joints + one perception token
        q, k, v = e @ a["Wq"], e @ a["Wk"], e @ a["Wv"]      # (N(+1), H)
        scores = (q @ k.T) / np.sqrt(self.hidden)            # (N(+1), N(+1))
        if self.topo_bias and hop is not None and "Wtopo" in a:   # structure-aware attention bias theta_{d(i,j)}
            nj = obs.shape[0]
            hop = np.asarray(hop)
            if hop.shape == (nj, nj):
                tb = a["Wtopo"]
                scores[:nj, :nj] += tb[np.clip(hop.astype(int), 0, len(tb) - 1)]   # zero-init -> no change
        scores -= scores.max(axis=1, keepdims=True)
        att = np.exp(scores); att /= att.sum(axis=1, keepdims=True)
        z = (att @ v) @ a["Wo"]                              # cross-token mixing (joints attend to perception)
        u = np.tanh(e + z)                                   # residual
        out = np.tanh(u @ a["Wh"] + a["bh"])                 # (N(+1), 1)
        return out[: obs.shape[0], 0]                        # drop the perception token's action

    # flat parameter vector — for black-box optimization (ES) on CPU before MJX/PPO
    _ORDER = ("We", "be", "Wq", "Wk", "Wv", "Wo", "Wh", "bh", "Wrange", "brange")

    def get_params(self):
        import numpy as np
        return np.concatenate([self._arrs[k].ravel() for k in self._order])

    def set_params(self, flat):
        import numpy as np
        flat = np.asarray(flat, dtype=float)
        i = 0
        for kk in self._order:
            arr = self._arrs[kk]; n = arr.size
            arr[...] = flat[i:i + n].reshape(arr.shape); i += n

    @property
    def n_params(self) -> int:
        return sum(a.size for a in self._arrs.values())

    @classmethod
    def from_npz(cls, path):
        """Load a policy trained on the GPU (``mjx_morph_attention.py --save``) into a CPU MorphPolicy.
        The attention arch matches exactly, so the SAME weights apply to ANY body (cross-morphology
        transfer) — load once, evaluate on a quadruped, a hexapod, a spider, etc."""
        import numpy as np

        d = np.load(path)
        feature_dim, hidden = int(d["meta"][0]), int(d["meta"][1])
        # Phase-5 opt-in weights are detected by key presence (same philosophy as Wrange): a policy trained
        # with FiLM / topo-bias carries "Wfilm" / "Wtopo"; older npz have neither -> the plain policy.
        buckets = int(len(d["Wtopo"]) - 1) if "Wtopo" in d.files else 8
        p = cls(feature_dim, hidden=hidden, film=("Wfilm" in d.files),
                topo_bias=("Wtopo" in d.files), topo_buckets=buckets)
        # Load only keys present in the file: pre-perception npz (8 keys, no Wrange/brange) load fine and
        # run perception-blind (the seeded Wrange/brange are never read unless act() is given ranges).
        for k in p._order:
            if k in d.files:
                arr = np.asarray(d[k], dtype=float)
                if p._arrs[k].shape == arr.shape:
                    p._arrs[k][...] = arr
                else:
                    p._arrs[k] = arr                          # adopt the saved shape (e.g. a different topo_buckets)
        # RECIPE policies bank their obs normalizer alongside the weights (older npz files have neither key ->
        # obs_mean stays None -> they deploy on the legacy residual path exactly as before).
        p.obs_mean = np.asarray(d["obs_mean"], dtype=float) if "obs_mean" in d.files else None
        p.obs_std = np.asarray(d["obs_std"], dtype=float) if "obs_std" in d.files else None
        # ADAPTIVE-GAINS flag rides in meta[4] (older 4-element meta -> stays False -> scalar 32/1.5 path).
        meta = d["meta"]
        p.adaptive_gains = bool(float(meta[4]) > 0.5) if len(meta) > 4 else False
        # TROT-CPG flag rides in meta[5]: a CPG-trained residual MUST deploy under the same CPG prior (else the
        # learned residual drives a body with no stepping rhythm). Older 5-element meta -> None -> scalar recipe.
        p.cpg = CPG_DEFAULT if (len(meta) > 5 and float(meta[5]) > 0.5) else None
        if "cpg_arr" in d.files:                             # EXACT banked CPG (per-body calf_phase/freq) -> deploy==train
            c = np.asarray(d["cpg_arr"], dtype=float)
            p.cpg = {"freq": float(c[0]), "thigh_amp": float(c[1]), "calf_amp": float(c[2]),
                     "calf_phase": float(c[3]), "residual_scale": float(c[4]), "leg_flip": bool(c[5] > 0.5)}
        # CONTROL DECIMATION rides in meta[6] (plan v2 T1.1): a policy trained at 50 Hz (D=10) MUST deploy at the
        # same rate or the gait mismatches. Older <=6-element meta -> 1 (every-step, unchanged). recipe_rollout_morph
        # and verify_submission read policy.decimation so deploy==train automatically.
        p.decimation = int(float(meta[6])) if len(meta) > 6 else 1
        p.action_lpf = float(meta[7]) if len(meta) > 7 else 0.0   # meta[7]: action EMA low-pass (T1.2); deploy==train
        p.sphere_feet = bool(float(meta[8]) > 0.5) if len(meta) > 8 else False  # meta[8]: sphere feet (T1.4); deploy==train
        return p

    def to_npz(self, path, score: float = 0.0, *, normalizer=None):
        """Save this policy so it can be banked in the flywheel and reloaded (same format as from_npz). Pass
        ``normalizer=(mean, std)`` for a RECIPE policy so its obs normalizer is banked WITH the weights (the
        deploy/replay path then drives it under the same PD-to-default control + normalization it learned)."""
        import numpy as np
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        norm = normalizer
        if norm is None and getattr(self, "obs_mean", None) is not None:
            norm = (self.obs_mean, self.obs_std)              # fall back to the policy's own banked normalizer
        extra, recipe_flag = {}, 0.0
        if norm is not None:
            extra = {"obs_mean": np.asarray(norm[0], dtype=float), "obs_std": np.asarray(norm[1], dtype=float)}
            recipe_flag = 1.0
        np.savez(path, **{k: self._arrs[k] for k in self._order}, **extra,
                 meta=np.asarray([float(self.feature_dim), float(self.hidden), recipe_flag, float(score),
                                  1.0 if getattr(self, "adaptive_gains", False) else 0.0,
                                  1.0 if getattr(self, "cpg", None) else 0.0,
                                  float(getattr(self, "decimation", 1) or 1),      # meta[6]: control decimation
                                  float(getattr(self, "action_lpf", 0.0) or 0.0),     # meta[7]: action LPF
                                  1.0 if getattr(self, "sphere_feet", False) else 0.0]))  # meta[8]: sphere feet (T1.4)
        return str(path)


_STANDING_SPAWN_CACHE: dict = {}


def _standing_spawn_for(gene):
    """Free-base bodies must spawn STANDING ON THEIR FEET, not at the legacy fixed 0.1 m — that height puts a
    quadruped's feet ~0.15 m UNDER the floor, so the contact solver violently EJECTS it (base launches to ~7x
    standing height) and it can never find a gait. ``learn_locomotion`` already trains on a standing-spawn body,
    but every gene-based EVAL/skill path (``rollout_morph``, ``locomotion_episode``, batched scoring, the Gym
    env) compiled at 0.1 — so the policy was TRAINED on a standing body and DEPLOYED on a penetrating one (a
    +2.4 m swing in 2400-step travel, the difference between walking forward and bouncing backward). Centralise
    the standing height here so every path uses the SAME body as training. Cached per body (the settling probe
    is otherwise re-run on every rollout); fixed-base bodies (arms) are unaffected (spawn_z stays None)."""
    if getattr(gene, "base_mount", None) != "free":
        return None
    try:
        import json
        key = hash(json.dumps(gene.to_dict(), sort_keys=True, default=str))
    except Exception:  # noqa: BLE001 - fall back to identity if the gene isn't serialisable
        key = id(gene)
    z = _STANDING_SPAWN_CACHE.get(key)
    if z is None:
        from virturoid.services.gene_compiler import standing_spawn_z
        z = standing_spawn_z(gene, meshed=False)
        _STANDING_SPAWN_CACHE[key] = z
    return z


def robot_mjcf(robot) -> str:
    """Resolve a 'robot' to an MJCF string — either a composed RobotGene or a raw MJCF/URDF-derived XML
    string (an IMPORTED model). Lets the whole learn/evaluate pipeline work on bodies we DIDN'T compose.
    Free-base genes spawn STANDING (``standing_spawn_z``) so eval/deploy matches how they were trained."""
    if isinstance(robot, str):
        return robot
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    return compile_gene_to_mjcf(robot, spawn_z=_standing_spawn_for(robot))


# --- compiled-model cache -------------------------------------------------------------------------------
# Profiling the ES loop showed XML->MjModel compile (~13 ms) was ~25% of every rollout's wall-clock, and
# the SAME body is recompiled on every candidate evaluation (pop x generations times) — pure waste. The XML
# string fully determines MjModel, MjModel is READ-ONLY during stepping (all per-rollout state lives in
# MjData), so one compiled model is safely reused across every rollout of that body in this process. This
# is process-local (each multiprocessing worker keeps its own), so it composes with the parallel path; the
# result is bit-identical (deterministic compile), it's just ~25% less work. See the batched-rollout study.
from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=32)
def _compiled_model_cached(xml: str):
    import mujoco
    return mujoco.MjModel.from_xml_string(xml)


def compiled_model(xml: str):
    """Compile (or reuse) an MjModel for ``xml``. Reused across rollouts of the same body — MjModel is
    read-only during stepping, so sharing it is safe and bit-identical, just far cheaper than recompiling."""
    return _compiled_model_cached(xml)


def rollout_morph(gene, policy: MorphPolicy | None = None, *, steps: int = 600, alpha: float = 0.6,
                  seed: int = 0, with_ranges: bool = False, n_rays: int = 8, max_range: float = 3.0) -> dict:
    """Drive ``gene`` (a composed RobotGene OR a raw MJCF string for an imported model) for ``steps`` with
    a MorphPolicy via the morph-graph interface, as a bounded residual on gravity comp. Returns stability +
    locomotion stats. CPU MuJoCo. The SAME policy object drives any morphology (obs/action dims are fixed).

    ``with_ranges`` compiles a rangefinder ring on the body and feeds sensed distances to the policy's
    perception token each step (exteroception); the default (False) is the proprioception-only rollout,
    byte-identical to before."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot

    if with_ranges and not isinstance(gene, str):
        from virturoid.services.exteroception import read_ranges, with_rangefinders
        model = compiled_model(with_rangefinders(gene, walls=None, n_rays=n_rays))
    else:
        with_ranges = False                                  # raw-MJCF imports have no rangefinder ring
        model = compiled_model(robot_mjcf(gene))
    model.opt.iterations = 20
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    graph = encode_robot(model)
    if policy is None:
        policy = MorphPolicy(graph.feature_dim, seed=seed)
    if policy.feature_dim != graph.feature_dim:
        raise ValueError(f"policy feature_dim {policy.feature_dim} != graph {graph.feature_dim}")

    p0 = (np.array(data.qpos[graph.base_qadr:graph.base_qadr + 2], dtype=float)
          if graph.base_jid >= 0 else np.zeros(2))
    z0 = float(data.qpos[graph.base_qadr + 2]) if graph.base_jid >= 0 else 0.0
    # DENSE gait reward (accumulated per step) — the deep-research reward template that won the locomotion sweep:
    #   per step:  exp(-(vx - V_TARGET)^2 / 0.25) * upright_gate   # track a target forward SPEED (not max) while UP
    #            + 0.3 * height_ratio                              # stay-tall (alive) bonus
    #            - 0.5 * mean((a_t - a_{t-1})^2)                   # ACTION SMOOTHNESS (the key term)
    # The smoothness penalty turns a slow low crouch-scoot into a rhythmic UPRIGHT walk (+0.45m crouch -> +1.17m
    # upright in the A/B sweep); velocity-TRACKING (vs raw-max) gives a controlled gait, not a lunge; the upright
    # gate (z>0.7*z0) zeroes forward credit while crouched. Returned as ``gait``; other fields are unchanged.
    gait = 0.0
    dt = float(model.opt.timestep)
    z0s = z0 if z0 > 1e-6 else 1.0
    prev_x = float(p0[0])
    prev_act = None
    v_target = 0.5
    for _ in range(steps):
        obs = graph.observe(model, data)
        ranges = read_ranges(model, data, n_rays=n_rays, max_range=max_range) if with_ranges else None
        act = policy.act(obs, ranges=ranges)
        graph.apply(model, data, act, alpha=alpha)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return {"finite": False, "n_tokens": graph.n_tokens, "feature_dim": graph.feature_dim,
                    "base_displacement": 0.0, "upright": False, "gait": -1.0}
        if graph.base_jid >= 0:
            x = float(data.qpos[graph.base_qadr]); z = float(data.qpos[graph.base_qadr + 2])
            hr_t = min(1.0, max(0.0, z / z0s))
            vx = (x - prev_x) / dt
            up = 1.0 if z > 0.7 * z0s else 0.0
            smooth = 0.0 if prev_act is None else float(np.mean((act - prev_act) ** 2))
            gait += float(np.exp(-((vx - v_target) ** 2) / 0.25)) * up + 0.3 * hr_t - 0.5 * smooth
            prev_x = x
        prev_act = act
    if graph.base_jid >= 0:
        p1 = np.array(data.qpos[graph.base_qadr:graph.base_qadr + 2], dtype=float)
        disp = float(np.linalg.norm(p1 - p0))
        forward = float(p1[0] - p0[0])
        zf = float(data.qpos[graph.base_qadr + 2])
        upright = bool(zf > 0.5 * z0)
        height_ratio = float(min(1.0, max(0.0, zf / z0))) if z0 > 0 else 1.0   # continuous: stayed tall?
    else:
        disp, forward, upright, height_ratio = 0.0, 0.0, True, 1.0
    return {"finite": True, "n_tokens": graph.n_tokens, "feature_dim": graph.feature_dim,
            "base_displacement": round(disp, 3), "forward": round(forward, 3), "upright": upright,
            "height_ratio": round(height_ratio, 3), "gait": round(gait / max(1, steps), 4)}


# --- ANTI-COLLAPSE RECIPE rollout (the fix for "the learned gait doesn't walk") -------------------------
# rollout_morph above drives a torque RESIDUAL on gravity-comp: gravity-comp only holds the CURRENT (possibly
# falling) pose, so ES finds a fall-forward FLOP, not a gait (z 0.305 -> 0.08; see memory locomotion-collapse-fix).
# Three independent investigations converged on the fix, validated on CPU (a quad that WALKS upright 2000+/2400
# steps vs the old collapse): position-PD to the DEFAULT STANDING pose + a small learned offset (a self-righting
# prior — even a random policy hovers near standing), observation NORMALIZATION (the #1 ES-locomotion prerequisite),
# terminate-on-fall + a clipped-NON-NEGATIVE velocity-TRACKING reward (a fallen body earns 0 thereafter, killing
# floor-rolling; tracking a target SPEED instead of maximizing it kills the suicide-lunge). Kept as NEW functions
# so the default rollout_morph stays byte-identical (test safety); the locomotion skill path opts into these.
def recipe_obs_normalizer(gene, *, n_pol: int = 4, steps: int = 250, seed0: int = 100):
    """Running mean/std of observations under random policies (ARS V2's observation normalization). Returned as
    ``(mean, std)`` and BANKED with a recipe policy so deployment applies the SAME normalization training used."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot

    model = compiled_model(robot_mjcf(gene)); model.opt.iterations = 20
    graph = encode_robot(model)
    obs = []
    for s in range(n_pol):
        data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
        pol = MorphPolicy(graph.feature_dim, seed=seed0 + s)
        for _ in range(steps):
            o = graph.observe(model, data); obs.append(o.copy())
            graph.apply(model, data, pol.act(o), alpha=0.6); mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                break
    A = np.concatenate(obs, axis=0)
    return A.mean(0), A.std(0) + 1e-4


# --- Adaptive per-joint PD gains: the "any robot" generalization of the locomotion recipe -----------------
# The scalar recipe (kp=32, kd=1.5) is tuned for the reference quadruped, whose actuated DOFs carry a joint-
# space (effective) inertia around _I_REF. A humanoid hip carries ~30x that inertia, so a flat kp=32 gives it a
# sluggish closed-loop natural frequency wn=sqrt(kp/I) and it folds under gravity. Holding the closed-loop wn
# and damping ratio zeta CONSTANT across bodies makes kp scale with each joint's effective inertia (kp =
# I_eff*wn^2), so the SAME transient response transfers to ANY morphology — the recipe walks a quadruped, a
# humanoid, or a heavy body without hand-retuning. Gains are a deterministic function of the body, so we
# RECOMPUTE them wherever the recipe runs (cheap: one mj_fullM) rather than banking stale vectors that would be
# wrong on a different body; the policy carries only a boolean `adaptive_gains` flag telling deploy/replay to
# take this path. See memory [[recipe-gain-generalization]].
_KP_REF, _KD_REF, _ASCALE = 32.0, 1.5, 0.4
_I_REF = 0.04359          # reference quadruped's MEDIAN actuated-DOF joint-space inertia (mj_fullM diag); FROZEN
_KP_MIN, _KP_MAX, _KD_MIN, _KD_MAX = 2.0, 192.0, 0.1, 24.0


def _effective_inertia(model, graph):
    """Per-actuated-DOF effective (joint-space) inertia: the diagonal of the dense mass matrix at the default
    pose. ``mj_fullM`` densifies the sparse ``data.qM``; ``graph.vadr`` indexes the actuated DOFs. CPU MuJoCo."""
    import mujoco
    import numpy as np
    data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv), dtype=float); mujoco.mj_fullM(model, M, data.qM)
    vadr = np.asarray(graph.vadr, dtype=int)
    return np.maximum(1e-4, np.diag(M)[vadr].astype(float))


def adaptive_recommended(model, graph=None) -> bool:
    """True when the body's joint inertia departs materially from the reference quadruped the scalar recipe was
    tuned on — a humanoid/heavy/very-asymmetric body where a flat kp=32 gives the wrong closed-loop response and
    folds. Near-reference bodies (a standard quadruped/hexapod) return False, so they KEEP the tuned scalar gains
    untouched (the anchor guarantees no regression). This is the auto-gate the product uses to turn adaptive gains
    on only where they help. CPU MuJoCo."""
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    if graph is None:
        graph = encode_robot(model)
    if graph.n_tokens == 0:
        return False
    I = _effective_inertia(model, graph)
    ratio = float(np.median(I)) / _I_REF                        # how much heavier than the reference quad
    spread = float(np.max(I) / max(1e-6, np.min(I)))            # inertia heterogeneity across joints
    # ONE-SIDED: scalar kp=32 only FAILS in the too-soft direction (a HEAVY joint folds under gravity). A joint
    # LIGHTER than the reference is merely over-stiffened by kp=32, which is safe (more stiffness != collapse, and
    # the torque clip catches saturation) — so a light-jointed dog/spider keeps the validated scalar gait. Adaptive
    # turns on only for heavier-than-reference bodies (humanoid) or very heterogeneous ones (mixed heavy+light DOFs).
    return (ratio > 2.2) or (spread > 6.0)


def recipe_gains(model, graph=None, *, ascale: float = _ASCALE):
    """Per-joint ``(kp_vec, kd_vec, ascale_vec)`` for the locomotion recipe, derived from each actuated DOF's
    effective (joint-space) inertia so the closed-loop response (natural frequency, damping ratio) is CONSTANT
    across morphologies. At the reference inertia it reproduces the tuned scalar (~32/1.5/0.4), so enabling it
    never moves the anchor body; a humanoid/heavy body gets proportionally stiffer joints and stays upright.
    Inertia is read at the DEFAULT standing pose (the PD attractor). CPU MuJoCo; deterministic in the body."""
    import numpy as np

    from virturoid.services.morph_graph import encode_robot
    if graph is None:
        graph = encode_robot(model)
    n = graph.n_tokens
    if n == 0:
        z = np.zeros(0)
        return z, z, z
    I_eff = _effective_inertia(model, graph)                    # generalized inertia per actuated DOF
    tau_max = np.asarray(graph.clamps, dtype=float)              # per-token actuator torque limit
    wn2 = _KP_REF / _I_REF
    zeta = _KD_REF / (2.0 * np.sqrt(_KP_REF * _I_REF))
    # Cap kp so a FULL-scale offset command (|tanh|=1 -> ascale rad of error) demands at most ~0.9*tau_max — the
    # stiffness never routinely saturates the actuator (the dimensionally-correct reading of the "0.9*tau_max"
    # ceiling; a raw `min(kp, 0.9*tau_max)` would instead clamp the quad's own joints to ~12 and regress it).
    # FLOOR the cap at _KP_REF so it can NEVER clip a joint below the tuned scalar — this makes the anchor EXACT
    # (kp=32.0/kd=1.5 at the reference inertia) while still bounding the much-stiffer heavy-body joints.
    kp_cap = np.maximum(_KP_REF, 0.9 * tau_max / max(1e-3, float(ascale)))
    kp = np.clip(np.minimum(I_eff * wn2, kp_cap), _KP_MIN, _KP_MAX)
    kd = np.clip(2.0 * zeta * np.sqrt(kp * I_eff), _KD_MIN, _KD_MAX)
    asc = np.full(n, float(ascale), dtype=float)
    return kp, kd, asc


# Default trot-CPG params (validated by build/_cpg_sweep.py: bare CPG walks the composed quad ~0.5 m upright,
# cadence ~6/s — the prior that ES then refines, per docs/virturoid_research_dossier.md P1). A feed-forward
# oscillation on each leg's thigh+calf joints, diagonal limbs anti-phase, gives the body a STEPPING rhythm so
# ES learns only a propulsion/stabilization RESIDUAL instead of discovering a gait from a stand-still (which
# CPU-ES cannot do — it slides or collapses). residual_scale < 1 keeps the policy from cancelling the CPG.
CPG_DEFAULT = {"freq": 1.5, "thigh_amp": 0.6, "calf_amp": 0.8, "calf_phase": 1.5708,
               "residual_scale": 0.3, "leg_flip": True}


def _trot_cpg_tokens(model, graph, params: dict):
    """Per-token (amp, phase) arrays for a CPG gait prior, plus a gate. Two morphology paths:
      - QUADRUPED (procedural ``leg{i}_{j}`` naming, role 1 = thigh/hip-pitch, 2 = calf/knee): DIAGONAL trot.
      - BIPED / HUMANOID (anatomical ``left_*``/``right_*`` hip-PITCH + knee naming): left/right ANTI-PHASE
        alternating gait (only the sagittal leg joints oscillate; hip yaw/roll, ankles, arms, torso stay at the
        PD default + learned residual).
    Anything else gets amp 0 (-> scalar recipe). Token k's joint is found by name via its qpos address."""
    import re

    import mujoco
    import numpy as np
    n = graph.n_tokens
    amp = np.zeros(n); phase = np.zeros(n)
    qadr2name = {int(model.jnt_qposadr[j]): (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
                 for j in range(model.njnt)}
    th_a, ca_a, ca_ph = params["thigh_amp"], params["calf_amp"], params["calf_phase"]

    # PATH 1 — procedural QUADRUPED: 'leg{i}_{j}', diagonal trot (UNCHANGED; the banked quad walk depends on it).
    leg_re = re.compile(r"leg(\d+)_(\d+)")
    tok_leg = [None] * n; tok_role = [None] * n; legs = set()
    for k in range(n):
        m = leg_re.search(qadr2name.get(int(graph.qadr[k]), ""))
        if m:
            tok_leg[k] = int(m.group(1)); tok_role[k] = int(m.group(2)); legs.add(int(m.group(1)))
    legs = sorted(legs)
    if legs:
        if len(legs) == 4:
            base = [np.pi, 0.0, 0.0, np.pi] if params.get("leg_flip") else [0.0, np.pi, np.pi, 0.0]
            leg_phase = {legs[i]: base[i] for i in range(4)}  # diagonal pairs anti-phase (proven quad trot)
        else:
            # GENERAL N-leg body (hexapod/octopod/…): ALTERNATING TRIPOD (plan gap-closure N15). The bodies we
            # build are BILATERAL — steerable_quadruped mounts legs as (L,R) pairs front→back, so the leg index
            # enumerates FL,FR,ML,MR,HL,HR,... The old `π·(i%2)` grouped {0,2,4}=all-LEFT vs {1,3,5}=all-RIGHT →
            # a lateral PACE (both sides rock together) with no alternating support triangle → ≈0 net forward
            # thrust (the measured "hexapod won't walk"). The canonical hexapod tripod alternates
            # {FL,MR,HL} vs {FR,ML,HR}: within each L/R pair the two legs are anti-phase AND consecutive stations
            # flip, which is exactly `phase = π·((i//2 + i%2) % 2)` for the (L,R)-pairs enumeration (→ tripod A =
            # {0,3,4}, tripod B = {1,2,5}; generalizes to an alternating tetrapod for octopods). Quad path untouched.
            leg_phase = {lg: (np.pi * (((i // 2) + (i % 2)) % 2)) for i, lg in enumerate(legs)}
        for k in range(n):
            if tok_leg[k] is None:
                continue
            if tok_role[k] == 1:
                amp[k] = th_a; phase[k] = leg_phase[tok_leg[k]]
            elif tok_role[k] == 2:
                amp[k] = ca_a; phase[k] = leg_phase[tok_leg[k]] + ca_ph
        return amp, phase, True

    # PATH 2 — BIPED / HUMANOID by anatomical naming: left/right hip-PITCH (thigh) + knee/shin (calf), L/R
    # anti-phase. Side is matched by full word ("left"/"right") OR by the ``l_``/``r_`` prefix our generated
    # humanoid uses (``l_thigh_joint`` / ``r_shin_joint``), and the calf by knee/calf/shin — so OUR generated
    # humanoid gets the stepping prior too (without it, CPU-ES can only stand-and-drift, never step).
    side_phase = {"left": 0.0, "right": np.pi}                # alternating-leg gait
    found = {"left": False, "right": False}
    for k in range(n):
        nm = qadr2name.get(int(graph.qadr[k]), "").lower()
        if "left" in nm or nm.startswith("l_"):
            side = "left"
        elif "right" in nm or nm.startswith("r_"):
            side = "right"
        else:
            continue
        if ("hip" in nm and "pitch" in nm) or "thigh" in nm:  # thigh-equivalent (sagittal hip swing)
            amp[k] = th_a; phase[k] = side_phase[side]; found[side] = True
        elif "knee" in nm or "calf" in nm or "shin" in nm:    # calf-equivalent (knee / shin)
            amp[k] = ca_a; phase[k] = side_phase[side] + ca_ph; found[side] = True
    if found["left"] and found["right"]:
        return amp, phase, True
    return amp, phase, False                                  # not a recognizable legged body -> scalar recipe


def extract_gait_params(gene) -> dict | None:
    """Extract the deterministic trot-CPG gait as exportable joint-target params, or None if ``gene`` isn't a
    recognized legged body (manipulator / unknown morphology). This is the SAME feed-forward gait
    ``recipe_rollout_morph`` drives -- ``target[j] = default_pose[j] + amplitude[j]*sin(2*pi*freq*t + phase[j])`` --
    MINUS the learned residual, so a deployed PD / ros2_control loop tracking these joint-position targets
    reproduces the exported walk. Pure data (no policy weights), so it round-trips to a standalone controller."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot

    model = compiled_model(robot_mjcf(gene)); model.opt.iterations = 20
    data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    graph = encode_robot(model)
    if graph.base_jid < 0 or graph.n_tokens == 0:               # fixed-base / no actuators: nothing to walk
        return None
    amp, phase, gate = _trot_cpg_tokens(model, graph, CPG_DEFAULT)
    if not gate:                                                # not a recognizable legged body -> no gait program
        return None
    qadr = np.asarray(graph.qadr, dtype=int)
    act_u = np.asarray(graph.act_u, dtype=int)
    default_pose = [float(data.qpos[a]) for a in qadr]          # the DEFAULT standing pose (the gait baseline)
    joint_names: list[str] = []
    limits: list[list[float]] = []
    for u in act_u:                                             # one actuated joint per token, token order preserved
        jid = int(model.actuator_trnid[int(u), 0])
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        joint_names.append(nm or f"joint_{jid}")
        if bool(model.jnt_limited[jid]):
            limits.append([float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])])
        else:
            limits.append([-3.14159265, 3.14159265])
    return {
        "policy_type": "trot_cpg_gait",
        "joint_names": joint_names,
        "default_pose": default_pose,
        "amplitude": [float(a) for a in np.asarray(amp).ravel()],
        "phase_offset": [float(p) for p in np.asarray(phase).ravel()],
        "frequency_hz": float(CPG_DEFAULT["freq"]),
        "position_limits": limits,
    }


def upright_height_ratio(n_feet: int) -> float:
    """Per-body "upright" stance-height threshold tau: how high (as a fraction of the standing z0) the trunk must
    ride to count as *genuinely upright for this morphology* (WS3). A hexapod/octopod is statically stable in a
    LOW tripod crouch and every hexapod-RL paper walks at ~0.15 m with NO upright reward at all
    (arXiv:2511.03167) -- so gating a low, properly-stepping tripod as "not upright" (the old hardcoded z>0.7*z0)
    was a quad-tuned cliff, not a morphology-correct bar. Quad/biped keep 0.70 (L1 baseline byte-identical);
    more legs -> a lower natural stance. Paired with the ``support_frac`` companion so the bar moves SIDEWAYS to
    morphology-correct (a low tripod must still show real stepping support), never DOWN to "easy"."""
    if n_feet >= 7:
        return 0.50                                             # octopod / decapod: broad, low, statically stable
    if n_feet >= 5:
        return 0.55                                             # hexapod: alternating-tripod low stance
    return 0.70                                                 # quad / biped / tripod: unchanged (L1 preserved)


def recipe_rollout_morph(gene, policy: MorphPolicy | None = None, *, steps: int = 900, kp: float = 32.0,
                         kd: float = 1.5, ascale: float = 0.4, vtgt: float = 0.3, normalizer=None,
                         adaptive: bool = False, cpg: dict | None = None, model_perturb=None,
                         seed: int = 0, record_frames: bool = False, frame_every: int = 5,
                         decimation: int = 1, action_lpf: float = 0.0, sphere_feet: bool = False,
                         command_schedule: list | None = None) -> dict:
    """Drive ``gene`` with the anti-collapse RECIPE: position-PD to the default standing pose + learned offset,
    obs normalization (``normalizer=(mean,std)``), terminate-on-fall, clipped-non-negative velocity-tracking
    reward. Returns ``gait`` (recipe reward meaned over the horizon — the training fitness), ``forward`` travel,
    ``height_ratio``, ``alive`` step count, ``survived``, and ``frames`` (for rendering). CPU MuJoCo; the SAME
    policy object drives any morphology (this is what actually produces an UPRIGHT WALK)."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot

    # deploy==train: a banked policy carries its own decimation/LPF/sphere-feet config (meta[6..8]); adopt it when
    # the caller didn't override, so ANY rollout of a banked policy reproduces its training contact model + rate.
    if policy is not None:
        sphere_feet = sphere_feet or bool(getattr(policy, "sphere_feet", False))
        if decimation == 1:
            decimation = int(getattr(policy, "decimation", 1) or 1)
        if action_lpf == 0.0:
            action_lpf = float(getattr(policy, "action_lpf", 0.0) or 0.0)
    model = compiled_model(robot_mjcf(gene))
    if sphere_feet or model_perturb is not None:             # BOTH transforms mutate the model in place, but
        import copy                                          # compiled_model returns a SHARED lru_cached MjModel —
        model = copy.deepcopy(model)                         # mutating it would corrupt the cache (and DR would
    model.opt.iterations = 20                                # COMPOUND across rollouts). Copy first when we will mutate.
    if sphere_feet:                                          # T1.4: manifold-invariant sphere feet + feet-only collision
        from virturoid.services.sphere_feet import apply_sphere_feet   # (default off -> untouched -> byte-identical)
        apply_sphere_feet(model)                             # applied on train + deploy compile so contact model matches
    if model_perturb is not None:                            # sim2real: randomize dynamics in-place (gain/mass/damping)
        model_perturb(model)                                 # default None -> untouched -> byte-identical rollout
    data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    graph = encode_robot(model)
    if policy is None:
        policy = MorphPolicy(graph.feature_dim, seed=seed)
    if policy.feature_dim != graph.feature_dim:
        raise ValueError(f"policy feature_dim {policy.feature_dim} != graph {graph.feature_dim}")
    # Phase-5 topo-bias: a policy trained WITH the topology attention bias must DEPLOY with it too, or the
    # learned control mismatches. Compute the body's hop-distance matrix once (None for non-topo policies ->
    # act() ignores it -> byte-identical). Morphology-agnostic: recomputed per body, so it transfers.
    hop = None
    if getattr(policy, "topo_bias", False):
        from virturoid.services.topo_pe import hop_distance_matrix
        hop = np.asarray(hop_distance_matrix(graph.parent), dtype=int)
    if graph.base_jid < 0 or graph.n_tokens == 0:            # fixed-base / no actuators: nothing to walk
        return {"finite": True, "gait": 0.0, "forward": 0.0, "height_ratio": 1.0, "alive": steps,
                "survived": True, "speed": 0.0, "frames": [], "n_tokens": graph.n_tokens,
                "feature_dim": graph.feature_dim}
    if normalizer is None and getattr(policy, "obs_mean", None) is not None:
        normalizer = (policy.obs_mean, policy.obs_std)       # a banked recipe policy carries its own normalizer
    mean = std = None
    if normalizer is not None:
        mean = np.asarray(normalizer[0], dtype=float); std = np.asarray(normalizer[1], dtype=float)
    bq = graph.base_qadr; z0 = float(data.qpos[bq + 2]) or 1.0
    qadr = np.asarray(graph.qadr, dtype=int); vadr = np.asarray(graph.vadr, dtype=int)
    act_u = np.asarray(graph.act_u, dtype=int); clamps = np.asarray(graph.clamps, dtype=float)
    q_def = np.array([float(data.qpos[a]) for a in qadr])    # the DEFAULT standing pose (the PD attractor)
    # PER-JOINT GAINS: adaptive (inertia-scaled, the "any robot" path) when requested OR when the banked policy
    # was trained adaptive; otherwise broadcast the scalar recipe gains. Scalars stay byte-identical (np.full of a
    # constant), so default callers + test_recipe_forward_score_matches_rollout are unchanged.
    if adaptive or bool(getattr(policy, "adaptive_gains", False)):
        # always derive adaptive gains at the canonical _ASCALE so train==score==replay==build-replay exactly
        # (the replay/build sites call recipe_gains with the default too — never forward a per-rollout ascale here).
        kp_v, kd_v, as_v = recipe_gains(model, graph)
    else:
        kp_v = np.broadcast_to(np.asarray(kp, dtype=float), (graph.n_tokens,))
        kd_v = np.broadcast_to(np.asarray(kd, dtype=float), (graph.n_tokens,))
        as_v = np.broadcast_to(np.asarray(ascale, dtype=float), (graph.n_tokens,))
    # TROT-CPG PRIOR (gated): feed-forward leg oscillation gives the body a stepping rhythm so the policy learns
    # only a propulsion/stabilization RESIDUAL. Inactive (cpg_on False / res_scale 1.0) -> identical scalar recipe.
    cpg_on = False; res_scale = 1.0; cpg_amp = cpg_phase = None; cpg_freq = 0.0; _two_pi = 2.0 * np.pi
    if cpg is None:
        cpg = getattr(policy, "cpg", None)                   # a banked CPG-trained policy carries its own CPG prior
    if cpg:
        cpg_amp, cpg_phase, _gate = _trot_cpg_tokens(model, graph, cpg)
        if _gate:
            cpg_on = True; cpg_freq = float(cpg["freq"]); res_scale = float(cpg.get("residual_scale", 0.3))
    x0 = float(data.qpos[bq]); px = x0; R = 0.0; a_prev = None; alive = steps; hr = 1.0
    dt = float(model.opt.timestep)
    frames = [] if record_frames else None
    capture = None
    if record_frames:
        from virturoid.services.pick_place_controller import _capture_geom_frame
        capture = _capture_geom_frame
    # foot-contact CADENCE + sustained-upright tracking, so the locomotion gate can tell a real WALK (feet lift and
    # alternate, trunk held high) from an upright SLIDE (feet never leave the floor -> 0 cadence) — the anti-Goodhart
    # signal the gameable raw-displacement KPI lacked.
    _gz0 = np.asarray(data.geom_xpos[:, 2], dtype=float)
    _body_g = [gi for gi in range(model.ngeom) if int(model.geom_bodyid[gi]) != 0]
    if _body_g:
        _zmin = min(float(_gz0[gi]) for gi in _body_g)
        feet = [gi for gi in _body_g if float(_gz0[gi]) < _zmin + 0.05] or _body_g
    else:
        feet = []
    feet_idx = np.asarray(feet, dtype=int)
    fz0 = _gz0[feet_idx] if len(feet_idx) else np.zeros(0)
    c_prev = (np.asarray(data.geom_xpos[feet_idx, 2]) < fz0 + 0.02) if len(feet_idx) else np.zeros(0, bool)
    lifts = 0; up_steps = 0; support_steps = 0
    tau_up = upright_height_ratio(len(feet_idx))            # per-body upright stance-height threshold (WS3)
    # COMMAND-TRACKING (WS8): with a command_schedule = [(vx_cmd, n_steps), ...] measure how well the body's
    # forward SPEED tracks a time-varying commanded speed. A constant "always walk forward" gait can't slow/stop/
    # reverse -> high error -> low track_score. L6_command_track is scored on tracking, so a constant gait FAILS.
    cmd_seq = None; terr = 0.0; tsteps = 0
    cmd_conditioned = bool(getattr(policy, "command_conditioned", False)) if policy is not None else False
    _zero_rng = None
    if command_schedule:
        cmd_seq = []
        for vx_c, n in command_schedule:
            cmd_seq.extend([float(vx_c)] * int(n))
        if cmd_conditioned:                                 # a conditioned policy READS the command (perception token)
            _zero_rng = np.zeros(max(0, int(getattr(policy, "percept_dim", 8)) - 2))
    dec = max(1, int(decimation))                            # CONTROL DECIMATION (plan v2 T1.1): recompute the
    lpf = float(action_lpf)                                  #   learned action only every `dec` physics steps and
    a = a_filt = None                                        #   HOLD it between (the CPG clock + PD loop still run
    for t in range(steps):                                   #   every step). dec=1 + action_lpf=0 -> byte-identical.
        if t % dec == 0:
            obs = graph.observe(model, data)
            if mean is not None:
                obs = (obs - mean) / std
            if cmd_conditioned and cmd_seq is not None:      # feed the current command to a conditioned policy (WS8)
                _vc = cmd_seq[min(t, len(cmd_seq) - 1)]
                a_raw = policy.act(obs, ranges=_zero_rng, cmd=[_vc, 0.0], hop=hop)
            else:
                a_raw = policy.act(obs, hop=hop)             # ACTION LPF (plan v2 T1.2): EMA-smooth the offset at the
            a_filt = a_raw if a_filt is None else lpf * a_filt + (1.0 - lpf) * a_raw   # control rate (Playground ~5Hz
            a = a_filt                                       #   filter); lpf=0 -> a=a_raw -> unchanged.
        cphase = _two_pi * cpg_freq * t * dt if cpg_on else 0.0
        for k in range(graph.n_tokens):
            cpg_off = float(cpg_amp[k] * np.sin(cphase + cpg_phase[k])) if cpg_on else 0.0
            tgt = q_def[k] + cpg_off + res_scale * as_v[k] * float(np.tanh(a[k]))
            tau = kp_v[k] * (tgt - float(data.qpos[qadr[k]])) - kd_v[k] * float(data.qvel[vadr[k]])
            data.ctrl[act_u[k]] = float(np.clip(tau, -clamps[k], clamps[k]))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            alive = t; break
        if record_frames and t % frame_every == 0:
            frames.append(capture(data, model))
        q = data.qpos[bq + 3:bq + 7]
        upr = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)   # world-up . body-up (orientation)
        z = float(data.qpos[bq + 2]); hr = min(1.0, max(0.0, z / z0)); x = float(data.qpos[bq])
        vx = (x - px) / dt; px = x
        if cmd_seq is not None:                                   # accumulate command-tracking error (WS8)
            _vc = cmd_seq[min(t, len(cmd_seq) - 1)]
            terr += (vx - _vc) ** 2; tsteps += 1
        if z < 0.5 * z0:                                          # FALLEN -> terminate (stop earning)
            alive = t; break
        up = min(1.0, max(0.0, (z - 0.5 * z0) / (0.2 * z0)))      # CONTINUOUS upright ramp (mirror of the GPU
        #   trainer fix): a smooth gradient out of a crouch instead of a binary z>0.7 cliff that paid exactly 0
        #   the instant z dipped, leaving a half-collapsed body no signal to climb back up.
        sm = 0.0 if a_prev is None else float(np.mean((a - a_prev) ** 2))
        R += max(0.0, float(np.exp(-((vx - vtgt) ** 2) / 0.25)) * up + 0.2 * up + 0.15 * max(0.0, upr) - 0.3 * sm)
        a_prev = a
        if z > tau_up * z0 and upr > 0.6:                         # sustained-upright fraction at the body's MORPHOLOGY
            up_steps += 1                                        #   stance height (tau_up: quad/biped 0.7, hex 0.55,
        #   octopod+ 0.5 -- WS3 per-body upright; a low tripod is upright FOR A HEXAPOD, cite arXiv:2511.03167)
        if len(feet_idx):                                         # foot LIFTOFF count -> cadence (a slide never lifts)
            c_now = np.asarray(data.geom_xpos[feet_idx, 2]) < fz0 + 0.02
            lifts += int(np.sum(c_prev & ~c_now))
            ng = int(np.sum(c_now))                              # TRIPOD-SUPPORT companion (WS3): a genuine stepping
            support_steps += int(0 < ng < len(feet_idx))        #   step has SOME feet down (support) AND some up
            c_prev = c_now                                       #   (swing) -- a flat slide plants ALL feet every step
    forward = float(data.qpos[bq] - x0)
    _T = max(1, alive) * dt
    cadence = round(lifts / _T, 2)                                # foot liftoffs per second (0 for a stiff slide)
    upright_frac = round(up_steps / max(1, alive), 3)            # fraction of alive steps upright at stance height
    support_frac = round(support_steps / max(1, alive), 3)      # fraction of alive steps in genuine stepping support
    out = {"finite": True, "gait": round(R / max(1, steps), 4), "forward": round(forward, 3),
           "height_ratio": round(hr, 3), "alive": alive, "survived": bool(alive >= steps and hr > 0.6),
           "speed": round(forward / max(1, alive) / dt, 3), "frames": frames or [],
           "cadence": cadence, "upright_frac": upright_frac, "support_frac": support_frac,
           "upright_tau": round(tau_up, 3), "n_feet": int(len(feet_idx)),
           "n_tokens": graph.n_tokens, "feature_dim": graph.feature_dim}
    if cmd_seq is not None:                                       # WS8: RMS speed-tracking error -> a legged_gym
        track_err = (terr / max(1, tsteps)) ** 0.5              #   exp(-err^2/sigma^2) score (higher = better; a
        out["track_err"] = round(track_err, 4)                 #   constant gait mistracks a varied command -> low)
        out["track_score"] = round(float(np.exp(-(track_err ** 2) / 0.05)), 4)
    return out


def recipe_robustness(gene, policy: MorphPolicy | None = None, *, n: int = 8, gain: float = 0.15,
                      mass: float = 0.1, damping: float = 0.2, friction: float = 0.3, seed: int = 0,
                      steps: int = 900, **kw) -> dict:
    """SIM2REAL robustness eval: run ``n`` rollouts under RANDOMIZED dynamics — per-trial actuator-gain, link
    mass+inertia, joint damping, and ground/foot friction perturbations applied to the compiled model — and
    report how often the policy still WALKS (survives + steps + moves forward). A policy that overfits one set
    of dynamics degrades here; a domain-randomized / robust one holds up. Uses recipe_rollout_morph's
    model_perturb hook, so the rollout itself is unchanged (this is the CPU mirror of the GPU --dr eval)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n):
        g = float(1.0 + gain * (2 * rng.random() - 1))
        m = float(1.0 + mass * (2 * rng.random() - 1))
        d = float(1.0 + damping * rng.random())             # damping only ADDS (friction rarely negative)
        f = float(1.0 + friction * (2 * rng.random() - 1))

        def _pert(model, _g=g, _m=m, _d=d, _f=f):
            model.actuator_gainprm[:, 0] *= _g              # motor gain (actuator strength)
            model.body_mass[1:] *= _m; model.body_inertia[1:] *= _m   # link mass + inertia (skip world body 0)
            model.dof_damping[:] *= _d                      # joint damping / friction
            model.geom_friction[:, 0] *= _f                 # tangential ground/foot friction
        res.append(recipe_rollout_morph(gene, policy, steps=steps, model_perturb=_pert, **kw))
    walked = [r for r in res if r.get("survived") and r.get("cadence", 0.0) >= 1.0 and r.get("forward", 0.0) > 0.15]
    fwd = [float(r.get("forward", 0.0)) for r in res]
    return {"n": n, "survival_rate": round(len(walked) / max(1, n), 3),
            "mean_forward": round(float(np.mean(fwd)), 3), "min_forward": round(float(min(fwd)), 3),
            "mean_cadence": round(float(np.mean([float(r.get("cadence", 0.0)) for r in res])), 2)}
