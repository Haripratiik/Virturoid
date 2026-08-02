"""GEN-1 (docs/generality_plan.md) — STRUCTURAL appendage discovery. Classify a compiled body's kinematic
chains by FUNCTION (leg / wheel / arm / spine / other) from CONTACT + KINEMATICS, never from link names or a
``robot_class`` label. This is the keystone the general gait engine (GEN-2), stance auditor (GEN-3), class
dispatch (GEN-4) and eval (GEN-6) all consume, so ONE mechanism sees any prompted body — a spider's 8 legs, a
hexapod's 6, a snake's spine, a rover's wheels — where the old name-regex CPG saw only ``leg{n}_{l|r}`` quads.

Measured rules (probed on the live build path, docs/generality_plan.md §1):
  * a token = one actuated joint; chains = maximal root->leaf paths in the token tree (``MorphGraph.parent``).
  * GROUNDED = the chain's lowest geom (incl. welded foot bodies) sits within ``ground_tol`` of the body's
    lowest geom overall. Legs/wheels/spine are grounded; arms + neck/tail are not (they ride high, z~0.44).
  * WHEEL = grounded chain with a CONTINUOUS (unlimited) hinge (a free-spinning driven wheel).
  * SPINE = the body IS one long grounded serial chain (len >= ``spine_min``) with no other grounded chains
    (a snake/eel) — undulates, doesn't step.
  * LEG = a grounded, limited-joint chain when there are >=2 grounded chains (a stepping limb).
  * ARM = a non-grounded chain that reaches up into free space (a manipulator).
  * OTHER = a short non-grounded appendage (neck/tail/head) — held at the PD default, never driven by a gait.
Per-token joint role from the WORLD-frame axis at rest: abduction (world fore-aft x), stride/lift (world
lateral y, shallow=stride hip, deep=lift knee), yaw/undulate (world vertical z). Axis logic matches the
existing ``_trot_cpg_tokens`` PATH 1b so a banked gait's tokens line up.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Appendage:
    kind: str                       # "leg" | "wheel" | "arm" | "spine" | "other"
    tokens: list                    # token indices along the main path, SHALLOW -> DEEP
    roles: list                     # per-token role, aligned with tokens: abduction|stride|lift|yaw|spin|other
    side: float = 0.0               # +1 left (world +y), -1 right, 0 center — from the root joint anchor
    tip_xy: tuple = (0.0, 0.0)      # world (x, y) of the deepest (foot/tip) body — for the support polygon
    tip_body: int = 0               # body id of the foot/tip — for measuring per-joint foot effect (radial legs)
    grounded: bool = False

    @property
    def stride_tokens(self) -> list:
        """The fore-aft SWING joints (world-y) — what a gait oscillates to step; abduction (x) is held."""
        return [t for t, r in zip(self.tokens, self.roles) if r in ("stride", "lift")]


@dataclass
class AppendageMap:
    legs: list = field(default_factory=list)
    wheels: list = field(default_factory=list)
    arms: list = field(default_factory=list)
    spine: object = None            # an Appendage or None
    others: list = field(default_factory=list)
    n_tokens: int = 0

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def n_wheels(self) -> int:
        return len(self.wheels)

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def kind(self) -> str:
        """Structure-derived locomotion class (GEN-4) — replaces the ``robot_class`` label for dispatch."""
        if self.n_legs >= 1:
            return "legged"
        if self.spine is not None:
            return "limbless"
        if self.n_wheels >= 2:
            return "wheeled"
        if self.n_arms >= 1:
            return "manipulator"
        return "unknown"

    def foot_xy(self) -> list:
        """World (x, y) of every leg tip — the support-polygon vertices (GEN-3 stance auditor)."""
        return [lg.tip_xy for lg in self.legs]


def _world_axis_role(wa) -> str:
    """Classify a joint's WORLD-frame axis into a locomotion role. x=fore-aft (abduction/roll),
    y=lateral (fore-aft stride swing about it), z=vertical (yaw/undulation)."""
    ax = int(max(range(3), key=lambda i: abs(wa[i])))
    return {0: "abduction", 1: "stride", 2: "yaw"}[ax]


def build_appendage_map(model, *, ground_tol: float = 0.10, spine_min: int = 6) -> AppendageMap:
    """Discover + classify the body's appendages from the compiled model. Pure geometry/contact + kinematics —
    no names, no ``robot_class``. ``ground_tol`` = how close (m) a chain's lowest geom must ride to the body's
    overall lowest to count as ground-contacting; ``spine_min`` = min chain length to read as a spine."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot

    graph = encode_robot(model)
    n = graph.n_tokens
    amap = AppendageMap(n_tokens=n)
    if n == 0:
        return amap
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # per-token joint id / body id / world axis / limited flag / side (anchor world-y sign)
    jid = [int(model.actuator_trnid[graph.act_u[t], 0]) for t in range(n)]
    bod = [int(model.jnt_bodyid[j]) for j in jid]
    limited = [bool(model.jnt_limited[j]) for j in jid]
    waxis, side = [], []
    for t in range(n):
        b = bod[t]
        wa = data.xmat[b].reshape(3, 3) @ np.asarray(model.jnt_axis[jid[t]])
        waxis.append(wa)
        side.append(1.0 if float(data.xanchor[jid[t]][1]) >= 0 else -1.0)

    # lowest geom-z per BODY (incl. bodies that carry a welded foot with no token)
    gz = np.asarray(data.geom_xpos[:, 2])
    body_geoms: dict = {}
    for gi in range(int(model.ngeom)):
        body_geoms.setdefault(int(model.geom_bodyid[gi]), []).append(gi)
    def body_low(b):
        gs = body_geoms.get(b)
        return min(float(gz[gi]) for gi in gs) if gs else 9.9
    zmin = min((body_low(b) for b in body_geoms if b != 0), default=0.0)

    # token tree -> children; roots have parent -1 (attach straight to the free base)
    kids: dict = {}
    for t, p in enumerate(graph.parent):
        kids.setdefault(p, []).append(t)
    roots = [t for t in range(n) if graph.parent[t] == -1]

    # TIP = the KINEMATICALLY deepest body under a chain's first body (incl. branches + welded feet/grippers).
    # A leg's tip is the foot (rides low, near the floor); an arm's tip is the gripper (rides high, in free
    # space) — so grounded-ness must test the TIP's height, NOT the chain's lowest point (an arm mounts its
    # BASE low, which fooled a min-z test into calling it grounded). This is the leg/arm discriminator.
    def chain_tip(main_tokens):
        root_b = bod[main_tokens[0]]
        best_b, best_depth = bod[main_tokens[-1]], -1
        for bb in range(int(model.nbody)):
            pb, depth, found = bb, 0, False
            while pb > 0:
                if pb == root_b:
                    found = True; break
                pb = int(model.body_parentid[pb]); depth += 1
            if found and depth > best_depth:
                best_depth, best_b = depth, bb
        return body_low(best_b), (float(data.xpos[best_b][0]), float(data.xpos[best_b][1])), best_b

    # build each root's main path (root -> deepest leaf via single-child descent; branch = end-effector/gripper)
    def main_paths(r, depth=0):
        """The chains under token ``r``. Normally one: descend while there is a single child and stop at a
        branch (a gripper's fingers are not part of the arm).

        A ONE-TOKEN stub that branches is not an appendage at all, though — it is a TRUNK (a waist/pelvis yaw
        joint sitting between the free base and the hips), and stopping there collapsed everything below it
        into a single stub. Measured on Booster T1: both legs hang off one ``Waist`` token, so the body came
        out with ZERO legs and one 1-token "other", which then disabled the biped honesty verdict and left a
        humanoid to be judged by the multi-leg crawl gait. Every imported robot with an actuated waist above
        the hips lost its legs the same way. So descend PAST a bare trunk and let each branch be its own
        chain. A trunk >= 2 tokens long is a real appendage (an arm ending in fingers) and is left alone."""
        path = [r]; cur = r
        while cur in kids:
            ch = kids[cur]
            if len(ch) != 1:                            # branch point (e.g. gripper fingers) -> stop the main path
                break
            cur = ch[0]; path.append(cur)
        if len(path) == 1 and len(kids.get(path[0], ())) >= 2 and depth < 8:
            return [p for c in kids[path[0]] for p in main_paths(c, depth + 1)]
        return [path]

    chains = []
    for r in roots:
        for path in main_paths(r):
            tip_z, tip_xy, tip_b = chain_tip(path)
            grounded = (tip_z - zmin) < ground_tol
            chains.append({"root": path[0], "path": path, "grounded": grounded, "tip_xy": tip_xy,
                           "tip_body": tip_b, "unlimited": any(not limited[t] for t in path)})

    grounded_chains = [c for c in chains if c["grounded"]]

    # A BOLTED-DOWN BODY DOES NOT STAND ON WHAT IT IS BOLTED TO. Grounded-ness is measured against the body's
    # own lowest geom, which on a bench-mounted assembly is the bench mount itself — so every chain that hangs
    # anywhere near it reads as ground-contacting. Measured: a Robotiq 2F-85's two fingers came out as two LEGS
    # (n_legs=2 -> family "humanoid"), a UR5e's single 6-joint chain came out as a SNAKE'S SPINE, and a LEAP
    # hand's four fingers came out as a QUADRUPED — so an imported dexterous hand was verified with a walk
    # rubric and offered a quadruped crawl gait.
    #
    # A fixed base is only promoted on evidence of a real STANCE (body_kind.MIN_STANCE_CHAINS /
    # STANCE_LEVEL_FRAC): enough chains reaching down TOGETHER, to a common level. That is what still keeps
    # #214's quadruped legged — MuJoCo fused its static torso into the world, and its four level limbs qualify.
    # FLOATING bodies are untouched: they really do stand on whatever reaches the floor.
    floating = any(int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(int(model.njnt)))
    if not floating:
        from virturoid.services.body_kind import MIN_STANCE_CHAINS, STANCE_LEVEL_FRAC
        zmax = max((max(float(gz[gi]) for gi in gs) for b, gs in body_geoms.items() if b != 0), default=zmin)
        tips = [body_low(c["tip_body"]) for c in grounded_chains]
        level = bool(tips) and (max(tips) - min(tips)) <= STANCE_LEVEL_FRAC * max(zmax - zmin, 1e-6)
        if len(grounded_chains) < MIN_STANCE_CHAINS or not level:
            for c in chains:
                c["grounded"] = False
            grounded_chains = []

    # SPINE: a single grounded serial chain that IS the body (long, no sibling grounded chains)
    if len(grounded_chains) == 1 and len(grounded_chains[0]["path"]) >= spine_min:
        c = grounded_chains[0]
        amap.spine = Appendage(kind="spine", tokens=c["path"],
                               roles=[_world_axis_role(waxis[t]) for t in c["path"]],
                               side=0.0, tip_xy=c["tip_xy"], tip_body=c["tip_body"], grounded=True)
        return amap

    for c in chains:
        toks = c["path"]
        roles = [_world_axis_role(waxis[t]) for t in toks]
        # a leg's DEEPEST world-y (stride) joint is the knee/lift; shallower world-y ones are the hip stride
        yidx = [i for i, r in enumerate(roles) if r == "stride"]
        if len(yidx) >= 2:
            roles[yidx[-1]] = "lift"
        app = Appendage(kind="", tokens=toks, roles=roles, side=side[c["root"]], tip_xy=c["tip_xy"],
                        tip_body=c["tip_body"], grounded=c["grounded"])
        if c["grounded"] and c["unlimited"] and len(toks) <= 2:
            app.kind = "wheel"; app.roles = ["spin"] * len(toks); amap.wheels.append(app)
        elif c["grounded"] and len(grounded_chains) >= 2:
            app.kind = "leg"; amap.legs.append(app)
        elif not c["grounded"] and len(toks) >= 2:
            app.kind = "arm"; amap.arms.append(app)
        else:
            app.kind = "other"; amap.others.append(app)   # neck / tail / head / lone stub — held at PD default
    return amap
