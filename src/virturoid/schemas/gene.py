"""Robot gene: a parametric, amendable encoding of a robot body.

The "gene" is the generative encoding the AI layer proposes and the compiler turns into
a real robot (URDF/MJCF). Per the architecture plan it uses a **parent-template +
parameter-diff** scheme so a new species is cheaply *amended* from an ancestor rather than
built from scratch — the core flywheel move (customer 2's spray-paint humanoid branches
from customer 1's box-lifter; same body, different end effector/task).

A gene is a kinematic tree of ``GeneSegment`` nodes (links) joined by typed joints, plus a
base mount and an end-effector kind. Morphology is intentionally separate from the task:
the same humanoid gene can lift boxes or spray paint by swapping the end effector + task,
not the body.

Validation enforces it is a real kinematic tree (single root, no cycles, known parents,
positive dimensions, one end effector) — "LLM proposes, schema/physics disposes".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Joint type -> MJCF joint kind (compiler maps these; "fixed" emits no joint = rigid weld).
JOINT_TYPES = {"revolute", "prismatic", "fixed"}
SHAPES = {"capsule", "box", "cylinder", "sphere"}
BASE_MOUNTS = {"floor", "table", "torso", "free"}  # "free" = a floating base (mobile robots that drive)
END_EFFECTORS = {"gripper", "suction", "spray_nozzle", "hook", "none"}


@dataclass
class GeneSegment:
    """One rigid link in the body, with the joint that attaches it to its parent."""
    name: str
    parent: str | None = None          # parent segment name; None = root (mounted to world)
    shape: str = "capsule"             # capsule | box | cylinder
    length_m: float = 0.1              # extent along the segment's local +z
    radius_m: float = 0.03             # capsule/cylinder radius, or box half-width in x/y
    mass_kg: float = 0.2
    joint_type: str | None = None      # revolute | prismatic | fixed | None(root: welded base)
    joint_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    joint_lower: float | None = None   # rad (revolute) or m (prismatic)
    joint_upper: float | None = None
    mount_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # fixed rotation of this segment vs its parent's tip
    mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # translational offset from parent tip (e.g. side-by-side gripper fingers)
    actuator_torque_nm: float | None = None  # drives the compiler's actuator gain (= the SELECTED motor's peak once grounded)
    torque_req_nm: float | None = None        # the joint's fixed torque REQUIREMENT; keeps grounding idempotent
    #                                           (re-grounding sizes from this, not from the last motor's peak)
    is_end_effector: bool = False
    cross_section: tuple[float, float] | None = None  # box (local x,y) half-extents; overrides the square radius_m.
    #                                    Lets a segment be LATERALLY COMPRESSED (a fish/eel body) for real
    #                                    undulatory swim thrust (drag anisotropy). None = square from radius_m.
    geometry: dict | None = None       # optional ARBITRARY CAD shape spec (cad_geometry.realize_shape): the
    #                                    blueprint can make this block any shape (extrude/revolve/tapered/…),
    #                                    not just the primitive shape/length/radius above. None = use primitive.
    material: str | None = None        # render-only material key (steel|aluminum|carbon_fiber|shell|metal|
    #                                    rubber) -> viewport colour/finish; physics ignores it. None = default.


@dataclass
class RobotGene:
    """A complete, amendable robot body encoding."""
    id: str
    species: str                       # taxonomy pattern, e.g. humanoid.upper_body.box_lift
    robot_class: str                   # manipulator | humanoid | quadruped | mobile_base
    segments: list[GeneSegment] = field(default_factory=list)
    parent_species: str | None = None  # the tree node this gene was amended from
    base_mount: str = "table"          # floor | table | torso
    end_effector_type: str = "gripper"
    # How the body was obtained is product evidence, not an ephemeral Python
    # attribute.  A fallback must survive serialization and be visible in every
    # export/UI consumer; otherwise an authored design and a generic substitute
    # are indistinguishable after the build ends.
    design_source: str = "unknown"
    composition_notes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ---- tree helpers -------------------------------------------------------
    def segment(self, name: str) -> GeneSegment | None:
        return next((s for s in self.segments if s.name == name), None)

    def root(self) -> GeneSegment | None:
        return next((s for s in self.segments if s.parent is None), None)

    def end_effector(self) -> GeneSegment | None:
        return next((s for s in self.segments if s.is_end_effector), None)

    def children_of(self, name: str) -> list[GeneSegment]:
        return [s for s in self.segments if s.parent == name]

    def actuated_joints(self) -> list[GeneSegment]:
        return [s for s in self.segments if s.joint_type in ("revolute", "prismatic")]

    # ---- validation ---------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of issues; empty means the gene is a valid, buildable kinematic tree."""
        issues: list[str] = []
        if not self.segments:
            return ["gene has no segments"]
        names = [s.name for s in self.segments]
        if len(set(names)) != len(names):
            issues.append("segment names must be unique")
        name_set = set(names)

        roots = [s for s in self.segments if s.parent is None]
        if len(roots) != 1:
            issues.append(f"expected exactly one root segment, found {len(roots)}")

        for s in self.segments:
            if s.parent is not None and s.parent not in name_set:
                issues.append(f"segment {s.name!r} has unknown parent {s.parent!r}")
            if s.shape not in SHAPES:
                issues.append(f"segment {s.name!r} shape {s.shape!r} not in {sorted(SHAPES)}")
            if s.length_m <= 0 or s.radius_m <= 0 or s.mass_kg <= 0:
                issues.append(f"segment {s.name!r} must have positive length/radius/mass")
            if s.joint_type is not None and s.joint_type not in JOINT_TYPES:
                issues.append(f"segment {s.name!r} joint_type {s.joint_type!r} not in {sorted(JOINT_TYPES)}")
            if s.parent is None and s.joint_type not in (None, "fixed"):
                issues.append(f"root segment {s.name!r} must be a welded/fixed base (joint_type None or 'fixed')")

        # Acyclicity: every parent chain must terminate at the root.
        if not issues:
            for s in self.segments:
                seen, node = set(), s
                while node is not None and node.parent is not None:
                    if node.name in seen:
                        issues.append("kinematic tree has a cycle")
                        break
                    seen.add(node.name)
                    node = self.segment(node.parent)

        ees = [s for s in self.segments if s.is_end_effector]
        if len(ees) != 1:
            issues.append(f"exactly one segment must be the end effector, found {len(ees)}")
        if self.base_mount not in BASE_MOUNTS:
            issues.append(f"base_mount {self.base_mount!r} not in {sorted(BASE_MOUNTS)}")
        if self.end_effector_type not in END_EFFECTORS:
            issues.append(f"end_effector_type {self.end_effector_type!r} not in {sorted(END_EFFECTORS)}")
        return issues


    # ---- serialization (so the species tree can store + reload buildable genes) --------
    def to_dict(self) -> dict:
        return {
            "id": self.id, "species": self.species, "robot_class": self.robot_class,
            "parent_species": self.parent_species, "base_mount": self.base_mount,
            "end_effector_type": self.end_effector_type,
            "design_source": self.design_source,
            "composition_notes": list(self.composition_notes),
            "metadata": dict(self.metadata),
            "segments": [
                {"name": s.name, "parent": s.parent, "shape": s.shape, "length_m": s.length_m,
                 "radius_m": s.radius_m, "mass_kg": s.mass_kg, "joint_type": s.joint_type,
                 "joint_axis": list(s.joint_axis), "joint_lower": s.joint_lower, "joint_upper": s.joint_upper,
                 "mount_euler": list(s.mount_euler), "mount_offset": list(s.mount_offset),
                 "actuator_torque_nm": s.actuator_torque_nm, "torque_req_nm": s.torque_req_nm,
                 "is_end_effector": s.is_end_effector,
                 "cross_section": list(s.cross_section) if s.cross_section else None,
                 "geometry": s.geometry, "material": s.material}
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobotGene":
        return cls(
            id=d["id"], species=d["species"], robot_class=d["robot_class"],
            parent_species=d.get("parent_species"), base_mount=d.get("base_mount", "table"),
            end_effector_type=d.get("end_effector_type", "gripper"),
            design_source=str(d.get("design_source", "unknown")),
            composition_notes=list(d.get("composition_notes", [])),
            metadata=dict(d.get("metadata", {})),
            segments=[
                GeneSegment(
                    name=s["name"], parent=s.get("parent"), shape=s.get("shape", "capsule"),
                    length_m=s.get("length_m", 0.1), radius_m=s.get("radius_m", 0.03), mass_kg=s.get("mass_kg", 0.2),
                    joint_type=s.get("joint_type"), joint_axis=tuple(s.get("joint_axis", (0, 0, 1))),
                    joint_lower=s.get("joint_lower"), joint_upper=s.get("joint_upper"),
                    mount_euler=tuple(s.get("mount_euler", (0, 0, 0))),
                    mount_offset=tuple(s.get("mount_offset", (0, 0, 0))),
                    actuator_torque_nm=s.get("actuator_torque_nm"), torque_req_nm=s.get("torque_req_nm"),
                    is_end_effector=s.get("is_end_effector", False),
                    cross_section=(tuple(s["cross_section"]) if s.get("cross_section") else None),
                    geometry=s.get("geometry"), material=s.get("material"),
                )
                for s in d["segments"]
            ],
        )


def amend_gene(
    parent: RobotGene,
    *,
    new_id: str,
    species: str,
    segment_overrides: dict[str, dict] | None = None,
    add_segments: list[GeneSegment] | None = None,
    remove_segments: list[str] | None = None,
    end_effector_type: str | None = None,
    base_mount: str | None = None,
    metadata: dict | None = None,
) -> RobotGene:
    """Derive a child gene from a parent by a parameter-diff (the flywheel amendment op).

    This is how a new customer's robot reuses an existing species: keep the parent body,
    override only what differs (scale a link, swap the end effector, add/remove a segment).
    ``segment_overrides`` maps segment name -> field overrides (e.g. {"length_m": 0.4}).
    """
    removed = set(remove_segments or [])
    segments: list[GeneSegment] = []
    for s in parent.segments:
        if s.name in removed:
            continue
        ov = (segment_overrides or {}).get(s.name)
        segments.append(replace(s, **ov) if ov else replace(s))
    segments.extend(add_segments or [])
    return RobotGene(
        id=new_id,
        species=species,
        robot_class=parent.robot_class,
        segments=segments,
        parent_species=parent.species,
        base_mount=base_mount or parent.base_mount,
        end_effector_type=end_effector_type or parent.end_effector_type,
        metadata={**parent.metadata, **(metadata or {}), "amended_from": parent.id},
    )
