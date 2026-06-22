"""Morphology graph: a body-agnostic observation/action interface so ONE policy drives ANY composed robot.

Theme 1 of the comprehensive roadmap (MetaMorph / morphology-conditioned control). A single policy can
only generalize across the bodies we compose if it sees each robot as a GRAPH of like-typed tokens — one
per actuated joint — with a FIXED-width feature vector per token (local proprioception + the joint's
structural role in the kinematic tree + shared global base state). Then a per-token (permutation-
equivariant) policy applies unchanged to a 3-DOF arm, an 8-DOF quadruped, or a 24-DOF spider.

This module is the keystone representation — pure NumPy + MuJoCo introspection, validated CPU-first
before any MJX/GPU training (the project's CPU→single-env-MJX→GPU discipline). The learnable transformer
+ PPO live on top of this interface (``morph_policy`` + ``scripts/mjx_morph_ppo``), not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MorphGraph:
    """Encoded robot: per-actuated-joint tokens + tree structure + the runtime obs/action interface."""
    act_u: list = field(default_factory=list)        # actuator index per token
    qadr: list = field(default_factory=list)         # qpos address per token
    vadr: list = field(default_factory=list)         # dof/qvel address per token
    parent: list = field(default_factory=list)       # parent token index (-1 = attaches to the base)
    static: object = None                            # (N, S) static structural features (np.ndarray)
    clamps: object = None                            # (N,) actuator force clamp
    base_jid: int = -1                               # freejoint id (-1 if fixed base)
    base_qadr: int = -1

    @property
    def n_tokens(self) -> int:
        return len(self.act_u)

    # per-token feature layout: [static S] + [qpos, qvel, sin, cos] + [base: z, upright, linvel3, angvel3]
    GLOBAL_DIM = 4 + 9                                # dynamic-per-token (4) + base-global broadcast (9)

    @property
    def feature_dim(self) -> int:
        return (self.static.shape[1] if self.static is not None else 0) + self.GLOBAL_DIM

    def observe(self, model, data):
        """Build the (n_tokens, feature_dim) observation. Same feature_dim for EVERY morphology."""
        import numpy as np

        n = self.n_tokens
        qpos = np.array([data.qpos[a] for a in self.qadr]) if n else np.zeros(0)
        qvel = np.array([data.qvel[v] for v in self.vadr]) if n else np.zeros(0)
        dyn = np.stack([qpos, qvel, np.sin(qpos), np.cos(qpos)], axis=1) if n else np.zeros((0, 4))
        # shared global base state (broadcast to every token) — free base only, else zeros
        if self.base_jid >= 0:
            q = data.qpos[self.base_qadr:self.base_qadr + 7]
            qd = data.qvel[int(model.jnt_dofadr[self.base_jid]):int(model.jnt_dofadr[self.base_jid]) + 6]
            quat = q[3:7]
            upright = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)   # world-z . body-z
            g = np.array([q[2], upright, qd[0], qd[1], qd[2], qd[3], qd[4], qd[5]], dtype=float)
        else:
            g = np.zeros(8)
        glob = np.concatenate([[1.0 if self.base_jid >= 0 else 0.0], g])   # (9,)
        glob = np.tile(glob, (n, 1)) if n else np.zeros((0, 9))
        return np.nan_to_num(np.concatenate([self.static, dyn, glob], axis=1)) if n else np.zeros((0, self.feature_dim))

    def apply(self, model, data, actions, *, alpha: float = 1.0, gravity_comp: bool = True):
        """Write per-token actions to ctrl as a BOUNDED residual on gravity compensation — so an
        untrained policy roughly holds pose (stable) and learns the residual (our proven recipe)."""
        import numpy as np

        for k in range(self.n_tokens):
            base = float(data.qfrc_bias[self.vadr[k]]) if gravity_comp else 0.0
            res = float(np.clip(actions[k], -1.0, 1.0)) * alpha * self.clamps[k]
            data.ctrl[self.act_u[k]] = float(np.clip(base + res, -self.clamps[k], self.clamps[k]))


def encode_robot(model) -> MorphGraph:
    """Encode a compiled MuJoCo model into a body-agnostic ``MorphGraph`` (one token per actuated joint)."""
    import mujoco
    import numpy as np

    g = MorphGraph()
    # base = freejoint, if any
    g.base_jid = next((j for j in range(model.njnt) if int(model.jnt_type[j]) == 0), -1)
    g.base_qadr = int(model.jnt_qposadr[g.base_jid]) if g.base_jid >= 0 else -1

    joint_of_token = {}        # joint id -> token index (for parent lookup)
    body_token = {}            # body id of the joint -> token index
    static_rows = []
    for u in range(int(model.nu)):
        j = int(model.actuator_trnid[u, 0])
        jt = int(model.jnt_type[j])
        if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue                                  # skip free/ball (the base) — tokens are actuated DOFs
        tok = len(g.act_u)
        g.act_u.append(u)
        g.qadr.append(int(model.jnt_qposadr[j]))
        g.vadr.append(int(model.jnt_dofadr[j]))
        joint_of_token[j] = tok
        body = int(model.jnt_bodyid[j])
        body_token[body] = tok
        axis = [float(v) for v in model.jnt_axis[j]]
        link_off = float(np.linalg.norm(model.body_pos[body]))
        mass = float(model.body_mass[body])
        # geom radius proxy for this body's first geom
        radius = 0.0
        for gi in range(int(model.ngeom)):
            if int(model.geom_bodyid[gi]) == body:
                radius = float(model.geom_size[gi][0]); break
        is_hinge = 1.0 if jt == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.0
        static_rows.append([is_hinge, 1.0 - is_hinge, axis[0], axis[1], axis[2],
                            link_off, radius, mass])    # parent/depth filled below

    # parent token + depth: walk up the body tree to the nearest ancestor body that owns a token
    parents, depths = [], []
    for tok, u in enumerate(g.act_u):
        body = int(model.jnt_bodyid[int(model.actuator_trnid[u, 0])])
        p, depth, b = -1, 0, int(model.body_parentid[body])
        while b > 0:
            if b in body_token:
                p = body_token[b]; break
            b = int(model.body_parentid[b])
        # depth = number of token-ancestors
        b2, d = int(model.body_parentid[body]), 0
        while b2 > 0:
            if b2 in body_token:
                d += 1
            b2 = int(model.body_parentid[b2])
        parents.append(p); depths.append(d)
    g.parent = parents

    n_children = [0] * len(g.act_u)
    for p in parents:
        if p >= 0:
            n_children[p] += 1
    n = len(static_rows)
    rows = []
    for k, srow in enumerate(static_rows):
        # positional id (token order ~ tree traversal) breaks symmetry between otherwise-identical limbs
        # so an attention policy can phase them differently (MetaMorph's morphology positional embedding).
        pos = k / max(1, n - 1)
        rows.append(srow + [depths[k] / 10.0, n_children[k] / 4.0, pos])
    g.static = np.array(rows, dtype=float) if rows else np.zeros((0, 11))
    g.clamps = np.array([float(model.actuator_forcerange[u, 1]) or 10.0 for u in g.act_u], dtype=float)
    return g
