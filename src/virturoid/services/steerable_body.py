"""Steerable legged bodies — legs with a hip-YAW (abduction) DOF so the robot can actually TURN.

A composed quadruped whose leg joints are all sagittal (fore-aft swing + knee) can walk forward but
physically cannot yaw — like a real quadruped with no hip abduction. Learned steerable navigation (the
frog-maze) therefore needs a body that *can* turn: each leg gets a hip-yaw joint (vertical axis) ahead of
the fore-aft propulsion joints. This is the platform's morphology-gate principle in action — the body
must support the task. Shared by the trainer (``scripts/mjx_morph_vel``) and the follower
(``services/nav_learned``) so both build the identical body. Relates to [[morph-policy-theme1]].
"""

from __future__ import annotations

import math


def _leg_links(dofs_per_leg: int):
    """Canonical per-DOF leg chain. 2: [hip-pitch, knee] (sagittal, can't yaw); 3: + a leading hip-YAW
    (abduction → turning); 4: + an ankle pitch. A topology search picks among these."""
    yaw = {"length": 0.06, "axis": (0, 0, 1), "torque": 12.0}      # hip yaw (abduction) -> turning
    hip = {"length": 0.12, "axis": (0, 1, 0), "torque": 16.0}      # hip pitch (fore-aft propulsion)
    knee = {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}
    ankle = {"length": 0.08, "axis": (0, 1, 0), "torque": 8.0}
    return {2: [hip, knee], 3: [yaw, hip, knee], 4: [yaw, hip, knee, ankle]}[dofs_per_leg]


def steerable_quadruped(species: str = "frog", *, n_legs: int = 4, dofs_per_leg: int = 3):
    """A frog-like legged body. ``dofs_per_leg`` selects the leg chain (3 = hip-yaw+hip-pitch+knee, the
    hip-yaw being what lets a learned policy turn). ``n_legs`` places legs at corners (4) or radially."""
    from virturoid.services.morphology_builder import MorphologyBuilder

    b = MorphologyBuilder("legged", base_mount="free", species=species)
    b.base("torso", shape="box", length=0.10, radius=0.16, mass=2.5)
    if n_legs == 4:
        mounts = [(0.14, 0.12), (0.14, -0.12), (-0.14, 0.12), (-0.14, -0.12)]
    else:  # radial arrangement for other leg counts
        mounts = [(0.15 * math.cos(2 * math.pi * i / n_legs), 0.15 * math.sin(2 * math.pi * i / n_legs))
                  for i in range(n_legs)]
    links = _leg_links(dofs_per_leg)
    for i, (dx, dy) in enumerate(mounts):
        b.limb(f"leg{i}", [dict(s) for s in links], parent="torso",
               mount_offset=(dx, dy, -0.03), mount_euler=(math.pi, 0, 0))
    return b.build()
