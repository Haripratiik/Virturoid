"""Canonical kinematic layout shared by the URDF and MJCF exporters.

The Robot Genome stores links, joints, limits, and actuator bindings, but not
concrete link geometry. This module derives one deterministic layout (link
lengths, masses, geom sizes, joint origins) from the genome plus the generated
CAD models, so the URDF, the MJCF, and the CAD bill of materials all describe
the same physical arm. Both exporters consume this layout instead of computing
their own offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.cad import CadModel
from virturoid.schemas.robot import RobotGenome, RobotJoint

# World geometry shared with the scene exporter.
TABLE_TOP_Z = 0.025

# Link name -> CAD model id, so geometry/mass come from the generated CAD.
_LINK_CAD_ID = {
    "base_link": "cad_arm_base",
    "upper_link": "cad_upper_link",
    "forearm_link": "cad_forearm_link",
    "wrist_link": "cad_wrist_camera_mount",
    "gripper_link": "cad_gripper_mount",
}

# Fallback link geometry (meters / kg) when a CAD model is not supplied.
# Layout is a floor-reaching arm: short base pedestal + short upper (shoulder)
# bracket, then a long forearm and an extended wrist that swing down to the table.
_LINK_DEFAULTS = {
    "base_link": {"length": 0.07, "half_xy": 0.06, "mass": 0.45},
    "upper_link": {"length": 0.07, "half_xy": 0.03, "mass": 0.22},
    "forearm_link": {"length": 0.33, "half_xy": 0.022, "mass": 0.18},
    "wrist_link": {"length": 0.16, "half_xy": 0.02, "mass": 0.035},
    "gripper_link": {"length": 0.07, "half_xy": 0.03, "mass": 0.16},
}


@dataclass
class LinkLayout:
    name: str
    parent: str | None
    length_m: float
    half_xy_m: float
    mass_kg: float
    origin_xyz: tuple[float, float, float]
    joint: RobotJoint | None = None
    is_end_effector: bool = False
    children: list["LinkLayout"] = field(default_factory=list)


def compute_arm_layout(robot: RobotGenome, cad_models: list[CadModel] | None = None) -> LinkLayout:
    """Return the root LinkLayout (with nested children) for an arm genome."""
    cad_by_id = {model.id: model for model in (cad_models or [])}
    joint_by_child = {joint.child_link: joint for joint in robot.joints}
    child_links = set(joint_by_child)
    parent_links = {joint.parent_link for joint in robot.joints}

    # Root: a link that parents a joint but is never a child (the mounted base).
    roots = [link for link in robot.links if link in parent_links and link not in child_links]
    root_name = roots[0] if roots else (robot.links[0] if robot.links else "base_link")

    # The last actuated link carries the gripper; floating links attach to it.
    ee_link = robot.joints[-1].child_link if robot.joints else root_name
    # The true end-effector tip is a floating link (the gripper) if present,
    # otherwise the last actuated link. Exactly one link holds the EE site.
    floating = [link for link in robot.links if link != root_name and link not in child_links]
    ee_tip = floating[-1] if floating else ee_link

    layouts: dict[str, LinkLayout] = {}
    for link in robot.links:
        geom = _link_geometry(link, cad_by_id)
        layouts[link] = LinkLayout(
            name=link,
            parent=None,
            length_m=geom["length"],
            half_xy_m=geom["half_xy"],
            mass_kg=geom["mass"],
            origin_xyz=(0.0, 0.0, 0.0),
            joint=joint_by_child.get(link),
            is_end_effector=(link == ee_tip),
        )

    # Wire parents and joint origins (each body sits at its parent's distal tip).
    for link, layout in layouts.items():
        joint = joint_by_child.get(link)
        if joint is not None:
            layout.parent = joint.parent_link
            parent = layouts.get(joint.parent_link)
            layout.origin_xyz = (0.0, 0.0, parent.length_m if parent else 0.0)
        elif link != root_name:
            # Floating link (e.g. gripper): fix it to the end-effector tip.
            layout.parent = ee_link
            ee = layouts.get(ee_link)
            layout.origin_xyz = (0.0, 0.0, ee.length_m if ee else 0.0)

    for link, layout in layouts.items():
        if layout.parent and layout.parent in layouts:
            layouts[layout.parent].children.append(layout)

    # Mount the base on the table surface so the arm sits on the bench.
    layouts[root_name].origin_xyz = (0.0, 0.0, TABLE_TOP_Z)
    return layouts[root_name]


def iter_links(root: LinkLayout):
    yield root
    for child in root.children:
        yield from iter_links(child)


def _link_geometry(link: str, cad_by_id: dict[str, CadModel]) -> dict:
    defaults = _LINK_DEFAULTS.get(link, {"length": 0.1, "half_xy": 0.025, "mass": 0.1})
    cad_id = _LINK_CAD_ID.get(link)
    cad = cad_by_id.get(cad_id) if cad_id else None
    if cad is None:
        return dict(defaults)
    length = defaults["length"]
    half_xy = defaults["half_xy"]
    # An explicit link_length_mm (set by the co-design optimizer) overrides the
    # default/bbox-derived length for any link, so the body geometry is tunable.
    explicit_length = (cad.editable_parameters or {}).get("link_length_mm")
    if explicit_length:
        return {"length": round(float(explicit_length) / 1000.0, 4), "half_xy": half_xy, "mass": cad.mass_kg or defaults["mass"]}
    if cad.bounding_box_mm:
        bx, by, bz = cad.bounding_box_mm
        if link in {"upper_link", "forearm_link"}:
            # Link length is the long CAD axis; cross-section from the other two.
            length = round(bx / 1000.0, 4)
            half_xy = round(max(by, bz) / 2000.0, 4)
        # The base keeps its pedestal default height (the CAD plate is the
        # footprint, not the mount height); only its mass comes from CAD.
    mass = cad.mass_kg if cad.mass_kg else defaults["mass"]
    return {"length": length, "half_xy": half_xy, "mass": mass}
