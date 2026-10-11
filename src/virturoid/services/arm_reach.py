"""How far an arm REACHES, measured by forward kinematics, never by adding up link lengths.

``verify_robot`` used to print ``sum(length_m)`` over every actuated segment as the arm's reach. That sum
counts things a reach does not contain: the two finger joints of a gripper, every offset link that folds back
on itself, and the full length of every link whether or not the joint ranges ever let the chain straighten.
Measured on the Menagerie arms against their datasheets, that rule read a Franka Panda (855 mm) at 1.708 m,
a UR5e (850 mm) at 1.274 m and a Kinova Gen3 (902 mm) at 1.045 m: the Panda at twice its real reach, in the
headline verdict, in the spec sheet and on the verification certificate.

What is measured instead is the question a reach answers: how far from its own first joint's AXIS can the end
effector get? Over 6000 joint configurations sampled inside the limits the model declares, the largest
perpendicular distance from the first joint's axis line on the end effector's chain to the end-effector body's
origin. For an arm standing on a table with a vertical first joint that is its horizontal reach, the radius of
the work envelope datasheets quote; for a wall- or ceiling-mounted arm it is the same radius about its own first
axis, so the number does not depend on where the arm is bolted down. The end-effector ORIGIN is where the arm's
own flange or palm starts, so a tool or finger length is not in the number. A first joint that SLIDES has no axis
to be distant from, and its reach is measured from that joint's anchor point instead.

WHY THE AXIS AND NOT THE FIRST JOINT'S ANCHOR POINT, which the first version measured from. Over the 14 Menagerie
arms with a published reach (2026-10-11, same samples): from the anchor, median error 6% but worst +44% and a +7%
bias, because a source that anchors a vertical first joint well below the shoulder adds that offset whenever the
arm points straight up (a KUKA iiwa 14 read 1.102 m for its 0.820 m, a Flexiv Rizon 4 1.126 m for its 0.780 m,
and a composed arm with a base yaw under a column 1.14 m for about 0.85 m of horizontal reach); from the axis,
median 9%, worst 16% and no bias (Panda 0.856 for 0.855, UR5e 0.925 for 0.850, Gen3 0.841 for 0.902, KUKA 0.900,
Flexiv 0.903). A headline number that is sometimes 44% too long is the wrong trade for a slightly better median.

It is kinematics only: no controller, no dynamics, no gravity. Three limits are worth knowing when reading it:

* it is a sampled maximum, so it can only come in UNDER the true maximum, by the amount the 6000 samples miss
  the most extended pose. Measured over seeds 0-4 it moves by 0.001 m on the Panda, the UR5e and the Gen3 and by
  0.006 m on a composed 6-DOF arm (``tests/test_arm_reach.py`` holds it under 2%);
* every joint is sampled INDEPENDENTLY inside its own range. A linkage the source closes with an equality
  constraint (a four-bar, a coupled finger, a parallelogram elbow) is sampled as if it were open, so a
  configuration the constraint forbids can be counted and such an arm can read a little LONG;
* datasheets do not agree on where reach is measured to (wrist centre, flange, tool point), so the residual
  error is partly convention: across the 14 arms it runs from -14% (Sawyer) to +16% (Flexiv).
"""
from __future__ import annotations

#: How many joint configurations a reach estimate samples. 6000 is the count every number in the module
#: docstring was measured at; it costs about 0.04 s of forward kinematics on a 7-DOF arm once the model is
#: compiled (0.08-0.13 s on the first call, compile included).
REACH_SAMPLES = 6000

#: The seed of the sample. Fixed, so the same body reads the same reach in every call and every process.
REACH_SEED = 0


def _chain_joints(model, body: int) -> list[int]:
    """Hinge and slide joints on the path from ``body`` up to the world, distal first."""
    import mujoco
    movable = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    out: list[int] = []
    b = int(body)
    while b > 0:
        start, n = int(model.body_jntadr[b]), int(model.body_jntnum[b])
        out.extend(j for j in range(start, start + n) if int(model.jnt_type[j]) in movable)
        b = int(model.body_parentid[b])
    return out


def end_effector_body(model, gene) -> int | None:
    """The compiled body of the segment ``gene`` flags as its end effector, or ``None`` when none compiles."""
    for seg in getattr(gene, "segments", None) or []:
        if not getattr(seg, "is_end_effector", False):
            continue
        try:
            return int(model.body(seg.name).id)
        except (KeyError, ValueError):
            continue
    return None


def arm_reach(gene, *, model=None, samples: int = REACH_SAMPLES, seed: int = REACH_SEED) -> dict | None:
    """The arm's reach by forward kinematics, or ``None`` when the body has no articulated end-effector chain.

    ``{"reach_m", "from_joint", "measured_from", "to_body", "samples", "seed", "method"}``: ``reach_m`` is the
    largest perpendicular distance from ``from_joint``'s axis line (the first hinge on the end effector's chain;
    ``measured_from`` "axis") to ``to_body``'s origin over ``samples`` configurations drawn uniformly inside each
    chain joint's declared range, or the distance from that joint's anchor when it slides (``measured_from``
    "anchor"). An unlimited hinge is drawn over a full turn; an unlimited slide is held at its zero, since its
    travel is not in the model. ``model`` may be passed when the caller already compiled the gene's own MJCF.
    """
    import mujoco
    import numpy as np

    if model is None:
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        model = compiled_model(robot_mjcf(gene))
    ee = end_effector_body(model, gene)
    if ee is None:
        return None
    chain = _chain_joints(model, ee)
    if not chain:
        return None
    first = chain[-1]                                           # the most proximal joint on the chain
    lo = np.zeros(len(chain))
    hi = np.zeros(len(chain))
    for k, j in enumerate(chain):
        if bool(model.jnt_limited[j]):
            lo[k], hi[k] = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        elif int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
            lo[k], hi[k] = -np.pi, np.pi
    qadr = np.array([int(model.jnt_qposadr[j]) for j in chain])
    draws = np.random.default_rng(int(seed)).uniform(lo, hi, size=(int(samples), len(chain)))
    data = mujoco.MjData(model)
    about_axis = int(model.jnt_type[first]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    best = 0.0
    for row in draws:
        data.qpos[qadr] = row
        mujoco.mj_kinematics(model, data)
        v = data.xpos[ee] - data.xanchor[first]
        if about_axis:                                          # drop the component along the first joint's axis
            a = data.xaxis[first] / (float(np.linalg.norm(data.xaxis[first])) or 1.0)
            v = v - a * float(v @ a)
        best = max(best, float(np.linalg.norm(v)))
    return {"reach_m": round(best, 3),
            "from_joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, first) or f"joint {first}",
            "measured_from": "axis" if about_axis else "anchor",
            "to_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, ee) or f"body {ee}",
            "samples": int(samples), "seed": int(seed),
            "method": ("forward kinematics: the farthest the end-effector origin gets from the first joint's "
                       + ("axis" if about_axis else "anchor") + " on its chain, over joint configurations sampled "
                       "inside the declared limits")}
