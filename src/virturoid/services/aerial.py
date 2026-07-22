"""Make aerial robots actually FLY — a quadcopter body + a geometric flight controller that closes the aerial
frontier on CPU. This mirrors the swim close (see :mod:`aquatic`): the physics was assumed intractable, then an
experiment cracked it.

A quadcopter needs NO aerodynamics model — only four rotor THRUST forces applied along the body-up axis
(``mujoco.mj_applyFT`` at four rotor points). The controller is the standard **geometric / acceleration-based**
one, which a naive nested-PID controller cannot match once the drone has to translate (it flips):

  desired acceleration  ->  desired body-up (thrust) direction  ->  rotation-matrix attitude error
  ->  body torque  ->  allocate to four rotor forces.

MEASURED (scratchpad experiment, then this module): flies to arbitrary targets — hover (0,0,1.2), far
(2.5,-1.5,1.8), backward (-1.2,0,0.9), high (0.3,0.3,2.5) — all with ~0.00 m final error and sane 3-13 deg
banking. The attitude gains are INERTIA-NORMALIZED (they command a desired angular *acceleration*, multiplied by
the assembly inertia at run time), so the SAME gains fly a drone of any mass/size.

``build_quadcopter`` composes an ORIGINAL drone (a central hub + four rotors + a downward camera pod), and
``ai_native_tools._honest_fly`` drives it. Like the swim/gait verdicts, the flight verdict never lies: a body
that cannot reach its target honestly reads DOES NOT FLY.
"""
from __future__ import annotations

import re

from virturoid.schemas.gene import GeneSegment, RobotGene

_AERIAL_WORDS = ("drone", "quadcopter", "quadrotor", "quadrocopter", "multirotor", "multicopter", "hexacopter",
                 "octocopter", "quadcopter", "aerial", "flying", "fly", "flies", "hover", "hovering",
                 "helicopter", "aircraft", "uav", "copter", "airborne")
_AERIAL_RE = re.compile(r"\b(" + "|".join(sorted(set(_AERIAL_WORDS))) + r")\b", re.I)

# Geometric flight controller gains, INERTIA-NORMALIZED so one set flies any drone:
#   * kp/kd  — outer position loop -> desired world acceleration (per-axis; z stiffer to hold altitude).
#   * KR/KW  — inner attitude loop as a desired ANGULAR ACCELERATION (rad/s^2); the run-time torque is
#              ``I_assembly @ alpha_des`` so the closed-loop attitude bandwidth is mass/inertia-invariant.
#   * vcap   — desired-velocity cap (m/s of position-error feedforward) so big steps don't demand a huge tilt.
# (KR=533, KW=65 reproduce the measured hand-tuned drone: torque = I*(-KR*eR - KW*w) == -9*eR - 1.1*w at I=0.0169.)
FLY_GAINS = {"kp": (3.0, 3.0, 8.0), "kd": (2.6, 2.6, 5.0), "KR": 533.0, "KW": 65.0, "vcap": 1.2}


def is_aerial_prompt(prompt: str) -> bool:
    """True if the prompt asks for a flying/rotorcraft body (word-boundary, so 'butter-FLY' won't false-fire is
    handled by the caller's context; 'fly/flies/flying/drone/quadcopter/...' are the real triggers)."""
    return bool(_AERIAL_RE.search(prompt or ""))


def build_quadcopter(prompt: str = "a quadcopter drone", *, span_m: float = 0.25,
                     mass_kg: float = 0.9) -> RobotGene:
    """Compose an ORIGINAL quadcopter gene: a flat central hub, four rotor disks at the corners (welded), and a
    downward camera/sensor pod (the end effector). Floating base (freejoint via ``base_mount='free'``). The four
    rotor thrust points live in ``metadata['rotor_offsets']`` (± ``span_m`` in x and y) so the flight controller
    applies thrust where the rotors are — the same geometry that renders in the viewport."""
    import math
    L = round(float(span_m), 4)
    hub_len = 0.045                                            # thin flat hub (extends along local +z)
    rotor_z = -hub_len / 2.0                                   # bring rotors to the hub mid-plane (child z = hub_len + mo_z)
    hub = GeneSegment(name="hub", shape="box", length_m=hub_len, radius_m=max(0.06, 0.42 * L),
                      mass_kg=round(0.45 * mass_kg, 4), joint_type=None, material="carbon_fiber")
    segs = [hub]
    corners = [(L, L), (-L, L), (-L, -L), (L, -L)]
    for i, (x, y) in enumerate(corners):
        theta = math.atan2(y, x)                               # azimuth of this arm/rotor from the hub center
        # ARM BOOM (welded, visual): a thin beam from the hub center out to the rotor. euler (pi/2, theta+pi/2, 0)
        # maps the segment's local +z (its length axis) to the horizontal direction (cos t, sin t, 0), so the beam
        # lies flat and points at the rotor; length = hypot(x,y) reaches the corner exactly.
        segs.append(GeneSegment(
            name=f"arm{i}", parent="hub", shape="box", length_m=round(math.hypot(x, y), 4), radius_m=0.012,
            mass_kg=round(0.03 * mass_kg, 4), joint_type="fixed", mount_offset=(0.0, 0.0, rotor_z),
            mount_euler=(math.pi / 2, theta + math.pi / 2, 0.0), material="carbon_fiber"))
        segs.append(GeneSegment(
            name=f"rotor{i}", parent="hub", shape="cylinder", length_m=0.016, radius_m=round(0.34 * L, 4),
            mass_kg=round(0.06 * mass_kg, 4), joint_type="fixed", mount_offset=(x, y, rotor_z),
            material="carbon_fiber"))
    # camera/sensor pod hangs just under the hub center (a real drone's gimbal) and is the single end effector
    segs.append(GeneSegment(name="camera_pod", parent="hub", shape="box", length_m=0.03, radius_m=0.022,
                            mass_kg=round(0.09 * mass_kg, 4), joint_type="fixed",
                            mount_offset=(0.0, 0.0, -(hub_len + 0.02)), is_end_effector=True, material="aluminum"))
    gene = RobotGene(
        id="aerial_quadcopter", species="aerial.quadcopter", robot_class="aerial", segments=segs,
        base_mount="free", end_effector_type="none",
        metadata={"aerial": True, "rotor_offsets": [[x, y] for (x, y) in corners], "rotor_L": L,
                  "prompt": prompt})
    gene.design_source = "aerial_platform"
    # HONESTY (final-drive finding 2026-07-22): "a bird robot that flies" returned this quadcopter with EMPTY
    # composition_notes and design_source "unknown" — the render has zero bird traits, so the substitution was
    # invisible at the tool surface. The flight verdict was already honest; the provenance now is too.
    p = (prompt or "").lower()
    if any(w in p for w in ("bird", "wing", "flap", "eagle", "hawk", "butterfly", "dragonfly", "bat", "insect")):
        gene.composition_notes = [
            "A rotor-based quadcopter platform was selected to realize flight; fixed-wing and flapping-wing "
            "body plans are not yet supported, so the requested animal form was NOT preserved.",
        ]
    else:
        gene.composition_notes = ["Deterministic quadcopter platform selected for the aerial request."]
    return gene
