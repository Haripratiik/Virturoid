"""Compositional morphology builder: assemble ANY robot from a few parametric building blocks.

This is the LEGO substrate the agents drive: instead of hand-coding each species, an agent composes a
robot from primitives — a welded base, revolute/prismatic/fixed links chained or branched, symmetric
limbs, and end-effectors — and gets back a validated ``RobotGene`` that compiles to MuJoCo and
auto-places into the species tree (``species_discovery``). A handful of operations cover serial arms,
parallel-jaw grippers, quadrupeds, and multi-arm torsos; the same kit extends to anything expressible
as a kinematic tree. "Builder composes blocks, schema/physics disposes" — ``build()`` validates.

    arm = (MorphologyBuilder("manipulator", base_mount="table")
           .base("base").link("shoulder", 0.25, axis=(0,0,1)).link("upper", 0.25, axis=(0,1,0))
           .link("fore", 0.2, axis=(0,1,0)).gripper().build())
"""

from __future__ import annotations

import math

from virturoid.schemas.gene import GeneSegment, RobotGene

_EE_FOR = {"gripper": "gripper", "suction": "suction", "spray": "spray_nozzle", "hook": "hook"}


class MorphologyBuilder:
    def __init__(self, robot_class: str = "manipulator", *, base_mount: str = "table",
                 species: str | None = None, gene_id: str | None = None):
        self.robot_class = robot_class
        self.base_mount = base_mount
        self.species = species
        self.gene_id = gene_id
        self._segments: list[GeneSegment] = []
        self._cursor: str | None = None       # links attach here unless an explicit parent is given
        self._ee_type = "none"

    # ---- primitives ---------------------------------------------------------
    def base(self, name: str = "base", *, shape: str = "box", length: float = 0.1,
             radius: float = 0.04, mass: float = 1.0, geometry: dict | None = None) -> "MorphologyBuilder":
        """The welded root block (mounted to the world). ``geometry`` (optional) is an arbitrary CAD shape
        spec (cad_geometry.realize_shape) so the base can be any shape, not just a primitive."""
        self._segments.append(GeneSegment(name=name, parent=None, shape=shape, length_m=length,
                                          radius_m=radius, mass_kg=mass, joint_type=None, geometry=geometry))
        self._cursor = name
        return self

    def link(self, name: str, length: float, *, radius: float = 0.03, mass: float = 0.3,
             joint: str = "revolute", axis=(0.0, 0.0, 1.0), lower: float | None = -3.14,
             upper: float | None = 3.14, torque: float = 10.0, shape: str = "capsule",
             parent: str | None = None, mount_offset=(0.0, 0.0, 0.0),
             mount_euler=(0.0, 0.0, 0.0), geometry: dict | None = None) -> "MorphologyBuilder":
        """Add a jointed link as a child of the current tip (or an explicit ``parent``), and move the
        cursor onto it. ``joint`` is revolute | prismatic | fixed. ``geometry`` (optional) is an arbitrary
        CAD shape spec so this block can be ANY shape (extrude/revolve/tapered/…), not just a primitive."""
        if not self._segments:
            raise ValueError("call base() before adding links")
        p = parent or self._cursor
        self._segments.append(GeneSegment(
            name=name, parent=p, shape=shape, length_m=length, radius_m=radius, mass_kg=mass,
            joint_type=(None if joint == "fixed" else joint), joint_axis=tuple(axis),
            joint_lower=lower, joint_upper=upper, mount_offset=tuple(mount_offset),
            mount_euler=tuple(mount_euler), geometry=geometry,
            actuator_torque_nm=(None if joint == "fixed" else torque)))
        self._cursor = name
        return self

    def wheel(self, name: str, *, radius: float = 0.06, thickness: float = 0.04, parent: str | None = None,
              mount_offset=(0.0, 0.0, 0.0), torque: float = 6.0, mass: float = 0.3) -> "MorphologyBuilder":
        """Add a driven WHEEL: a cylinder disc whose spin axis is horizontal (rolls forward). Modeled as
        a continuous revolute joint about the link's local +z, with ``mount_euler`` laying that axis flat
        — so a composed mobile base has real rolling wheels, not vertical capsules."""
        if not self._segments:
            raise ValueError("call base() before adding wheels")
        self._segments.append(GeneSegment(
            name=name, parent=parent or self._cursor, shape="cylinder", length_m=thickness,
            radius_m=radius, mass_kg=mass, joint_type="revolute", joint_axis=(0.0, 0.0, 1.0),
            joint_lower=None, joint_upper=None, mount_offset=tuple(mount_offset),
            mount_euler=(math.pi / 2, 0.0, 0.0), actuator_torque_nm=torque))
        return self   # cursor stays on the chassis so multiple wheels attach to it

    def at(self, name: str) -> "MorphologyBuilder":
        """Move the insertion cursor to an existing segment (to start a branch)."""
        if not any(s.name == name for s in self._segments):
            raise ValueError(f"no segment named {name!r} to branch from")
        self._cursor = name
        return self

    def limb(self, prefix: str, links: list[dict], *, parent: str, mount_offset=(0.0, 0.0, 0.0),
             mount_euler=(0.0, 0.0, 0.0)) -> "MorphologyBuilder":
        """Attach a chain of links (a limb) at ``mount_offset`` from ``parent`` — call repeatedly with
        different offsets to build symmetric robots (quadruped legs, multi-arm torsos). ``mount_euler``
        orients the limb's first link: links extend along +z, so a quadruped LEG needs ``(pi,0,0)`` to
        point DOWN (else the legs stick up and the body can't stand)."""
        self.at(parent)
        # Only these kwargs are valid for link(); an LLM-authored spec may also carry "name"/"length"
        # (handled here) or extra descriptive keys — filter so any well-formed recipe composes.
        allowed = {"radius", "mass", "joint", "axis", "lower", "upper", "torque", "shape", "geometry"}
        for i, raw in enumerate(links):
            spec = dict(raw)
            length = spec.pop("length", 0.2)
            kwargs = {k: v for k, v in spec.items() if k in allowed}
            mo = mount_offset if i == 0 else (0.0, 0.0, 0.0)
            me = mount_euler if i == 0 else (0.0, 0.0, 0.0)
            self.link(f"{prefix}_{i}", length, parent=(parent if i == 0 else None),
                      mount_offset=mo, mount_euler=me, **kwargs)
        return self

    def gripper(self, *, span: float = 0.10, finger_len: float = 0.06, finger_radius: float = 0.012,
                close: float = 0.045, torque: float = 8.0, prefix: str = "") -> "MorphologyBuilder":
        """Add a parallel-jaw gripper (two prismatic fingers) on the current tip; sets the ee type.

        ``prefix`` namespaces the finger segments (e.g. ``"l_"``/``"r_"``) so a BIMANUAL robot can carry
        two grippers without colliding on the default ``finger_l``/``finger_r`` names."""
        palm = self._cursor
        if palm is None:
            raise ValueError("call base()/link() before gripper()")
        for name, y, ax in ((f"{prefix}finger_l", span / 2, (0.0, -1.0, 0.0)),
                            (f"{prefix}finger_r", -span / 2, (0.0, 1.0, 0.0))):
            self._segments.append(GeneSegment(
                name=name, parent=palm, shape="box", length_m=finger_len, radius_m=finger_radius,
                mass_kg=0.03, joint_type="prismatic", joint_axis=ax, joint_lower=0.0,
                joint_upper=close, mount_offset=(0.0, y, 0.0), actuator_torque_nm=torque))
        self._mark_palm_ee(palm)               # palm is the ee -> compiler emits the grasp_site TCP
        self._ee_type = "gripper"
        return self

    def hand(self, *, n_fingers: int = 3, span: float = 0.10, finger_len: float = 0.05,
             finger_radius: float = 0.01, close: float = 0.04, torque: float = 6.0) -> "MorphologyBuilder":
        """Add a multi-finger hand: ``n_fingers`` prismatic fingers spaced on a ring around the tip, each
        closing radially toward the center — a dexterous end-effector for varied objects. n=2 is the
        parallel gripper; n≥3 gives an enveloping grasp."""
        if n_fingers < 2:
            raise ValueError("a hand needs >= 2 fingers")
        if n_fingers == 2:
            return self.gripper(span=span, finger_len=finger_len, finger_radius=finger_radius,
                                close=close, torque=torque)
        palm = self._cursor
        if palm is None:
            raise ValueError("call base()/link() before hand()")
        r = span / 2
        for i in range(n_fingers):
            th = 2 * math.pi * i / n_fingers
            cx, cy = math.cos(th), math.sin(th)
            self._segments.append(GeneSegment(
                name=f"finger_{i}", parent=palm, shape="box", length_m=finger_len, radius_m=finger_radius,
                mass_kg=0.025, joint_type="prismatic", joint_axis=(-cx, -cy, 0.0),  # close toward center
                joint_lower=0.0, joint_upper=close, mount_offset=(round(r * cx, 5), round(r * cy, 5), 0.0),
                actuator_torque_nm=torque))
        self._mark_palm_ee(palm)               # palm is the ee -> compiler emits the grasp_site TCP
        self._ee_type = "gripper"
        return self

    def _mark_palm_ee(self, palm: str) -> None:
        """Mark the palm (not a finger) as the end-effector so the compiler emits the grasp_site TCP at
        the finger mid-plane — else build() defaults the ee to a finger leaf and no grasp TCP is created."""
        for s in self._segments:
            s.is_end_effector = (s.name == palm)

    def end_effector(self, kind: str = "gripper", name: str | None = None) -> "MorphologyBuilder":
        """Mark a segment as the end-effector and set the ee type (gripper/suction/spray/hook)."""
        name = name or self._cursor
        self._ee_type = _EE_FOR.get(kind, "none")
        for s in self._segments:
            s.is_end_effector = (s.name == name)
        return self

    # ---- finalize -----------------------------------------------------------
    def build(self) -> RobotGene:
        """Validate and return the composed gene (raises if it is not a buildable kinematic tree)."""
        if not any(s.is_end_effector for s in self._segments):
            # default the end-effector to the deepest leaf so the gene validates
            leaves = [s for s in self._segments if not any(c.parent == s.name for c in self._segments)]
            if leaves:
                leaves[-1].is_end_effector = True
        gid = self.gene_id or f"built_{self.robot_class}_{len(self._segments)}seg"
        gene = RobotGene(
            id=gid, species=self.species or f"{self.robot_class}.built",
            robot_class=self.robot_class, segments=list(self._segments),
            base_mount=self.base_mount, end_effector_type=self._ee_type)
        issues = gene.validate()
        if issues:
            raise ValueError(f"composed morphology is invalid: {'; '.join(issues)}")
        return gene
