"""Gymnasium-compatible environments (plan §39) — a STANDARD RL interface over Virturoid bodies, so any
off-the-shelf trainer (Stable-Baselines3, CleanRL, RLlib) can learn a controller, not just our in-house ES.

We expose LOCOMOTION here because it is fully torque-controlled end to end (``morph_graph``: observe -> act ->
apply -> step), so a Gym env is a faithful, learnable wrapper. Pick-place is intentionally NOT exposed yet: its
scripted controller uses an idealized PIN grasp (``pick_place_controller._pin_block``) — a real friction grasp
is the documented open problem, so an RL env over it would be redundant (reach is already solved) or blocked on
that grasp. When the learned grasp lands, a VirturoidPickPlaceEnv slots in behind the same interface.

gymnasium is OPTIONAL: if installed, ``VirturoidLocomotionEnv`` subclasses ``gymnasium.Env`` and registers
``Virturoid-Locomotion-v0``; if not, the SAME class works as a duck-typed env (reset/step/terminated/truncated)
so the interface is usable and testable now, and upgrades to a real gym.Env the moment the package appears.
"""

from __future__ import annotations

try:  # gymnasium is optional — degrade to a duck-typed base when absent.
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
    _Base = _gym.Env
    _HAS_GYM = True
except Exception:  # noqa: BLE001
    _gym = None
    _spaces = None
    _Base = object
    _HAS_GYM = False


class _Box:
    """Minimal Box stand-in (low/high/shape/dtype + sample) for when gymnasium is absent — same attributes
    the standard space exposes, so downstream code reads shape/low/high identically."""

    def __init__(self, low, high, shape, dtype="float32"):
        import numpy as np
        self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self):
        import numpy as np
        return np.random.uniform(self.low, self.high).astype(self.dtype)


def _box(low, high, shape):
    return _spaces.Box(low=low, high=high, shape=shape, dtype="float32") if _HAS_GYM else _Box(low, high, shape)


class VirturoidLocomotionEnv(_Base):
    """Drive ANY Virturoid body (a composed ``RobotGene`` or a raw MJCF string) for forward locomotion under a
    standard Gym contract.

      * observation: the flattened ``morph_graph`` per-token features ``(n_tokens * feature_dim,)`` — the SAME
        obs a ``MorphPolicy`` consumes (reshape to ``(n_tokens, feature_dim)`` to use a banked policy).
      * action: per-actuated-joint command in ``[-1, 1]`` ``(n_tokens,)`` — applied as a bounded residual on
        gravity compensation (``alpha`` scales it), exactly as ``rollout_morph`` does.
      * reward: per-step forward (+x) base progress gated by uprightness (a fallen body stops earning) — the
        dense analog of ``forward_score``.
      * terminated: the body fell (base below ``fall_frac`` of its standing height) or the sim blew up.
      * truncated: the step budget ``horizon`` elapsed.

    The model is compiled ONCE via the shared cache (``morph_policy.compiled_model``)."""

    metadata = {"render_modes": []}

    def __init__(self, gene, *, horizon: int = 1000, alpha: float = 0.6, fall_frac: float = 0.5):
        import numpy as np

        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import compiled_model, robot_mjcf

        self._np = np
        self._model = compiled_model(robot_mjcf(gene), solver_iterations=20)
        self._graph = encode_robot(self._model)
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.fall_frac = float(fall_frac)
        self.n_tokens = self._graph.n_tokens
        self.feature_dim = self._graph.feature_dim

        obs_dim = self.n_tokens * self.feature_dim
        self.observation_space = _box(-np.inf, np.inf, (obs_dim,))
        self.action_space = _box(-1.0, 1.0, (max(1, self.n_tokens),))

        self._data = None
        self._t = 0
        self._x0 = 0.0
        self._z0 = 1.0

    # -- gym API -----------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        import mujoco
        np = self._np
        if seed is not None and _HAS_GYM:
            super().reset(seed=seed)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        g = self._graph
        if g.base_jid >= 0:
            self._x0 = float(self._data.qpos[g.base_qadr])
            self._z0 = float(self._data.qpos[g.base_qadr + 2]) or 1.0
        else:
            self._x0, self._z0 = 0.0, 1.0
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        import mujoco
        np = self._np
        g = self._graph
        a = np.asarray(action, dtype=float).ravel()[: self.n_tokens]
        prev_x = float(self._data.qpos[g.base_qadr]) if g.base_jid >= 0 else 0.0

        g.apply(self._model, self._data, a, alpha=self.alpha)
        mujoco.mj_step(self._model, self._data)
        self._t += 1

        terminated = False
        if not np.all(np.isfinite(self._data.qpos)):
            return self._obs(), -1.0, True, False, {"blew_up": True}

        if g.base_jid >= 0:
            x = float(self._data.qpos[g.base_qadr])
            z = float(self._data.qpos[g.base_qadr + 2])
            upright = z > self.fall_frac * self._z0
            reward = (x - prev_x) * (1.0 if upright else 0.0)   # forward credit only while upright
            if not upright:
                terminated = True
        else:
            reward = 0.0
        truncated = self._t >= self.horizon
        return self._obs(), float(reward), bool(terminated), bool(truncated), {"t": self._t}

    def _obs(self):
        np = self._np
        if self.n_tokens == 0:
            return np.zeros(0, dtype="float32")
        return self._graph.observe(self._model, self._data).astype("float32").ravel()

    # -- convenience for our own MorphPolicy (reshape the flat obs back to tokens) -----------------------
    def act_with_morph_policy(self, policy, obs):
        return policy.act(self._np.asarray(obs, dtype=float).reshape(self.n_tokens, self.feature_dim))


def register_envs() -> bool:
    """Register Virturoid envs with gymnasium (idempotent). Returns True iff gymnasium is installed."""
    if not _HAS_GYM:
        return False
    try:
        _gym.register(id="Virturoid-Locomotion-v0",
                      entry_point="virturoid.services.gym_env:VirturoidLocomotionEnv")
    except Exception:  # noqa: BLE001 - already registered
        pass
    return True
