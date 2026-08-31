"""Executable alignment gate between contact-visible geometry and physical collision geometry."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class VisualPhysicsIssue:
    code: str
    geom: str
    detail: str


@dataclass
class VisualPhysicsReport:
    ok: bool
    support_gap_m: float | None
    checked_contact_visuals: int
    issues: list[VisualPhysicsIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "support_gap_m": self.support_gap_m,
                "checked_contact_visuals": self.checked_contact_visuals,
                "issues": [asdict(issue) for issue in self.issues]}


def audit_mjcf(xml: str, *, max_air_gap_m: float = 0.01, penetration_tol_m: float = 0.005,
               containment_tol_m: float = 0.001, contact_band_m: float = 0.03,
               enforce_floor_contact: bool = True, check_contact_visuals: bool | None = None) -> VisualPhysicsReport:
    """Check spawn contact and containment for cosmetic geometry close enough to visually imply contact."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)

    def bounds(geom_id: int):
        center, half = model.geom_aabb[geom_id, :3], model.geom_aabb[geom_id, 3:]
        local = center + signs * half
        world = data.geom_xpos[geom_id] + local @ data.geom_xmat[geom_id].reshape(3, 3).T
        return world.min(axis=0), world.max(axis=0)

    robot_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) != 0]
    physical = [i for i in robot_geoms if int(model.geom_contype[i]) or int(model.geom_conaffinity[i])]
    cosmetic = [i for i in robot_geoms if not int(model.geom_contype[i]) and not int(model.geom_conaffinity[i])]
    issues: list[VisualPhysicsIssue] = []
    if not physical:
        issues.append(VisualPhysicsIssue("no_physical_geometry", "", "robot has no collidable geometry"))
        return VisualPhysicsReport(False, None, 0, issues)
    physical_bounds = {i: bounds(i) for i in physical}
    support_z = min(lo[2] for lo, _ in physical_bounds.values())
    if enforce_floor_contact:
        if support_z > max_air_gap_m:
            issues.append(VisualPhysicsIssue("airborne_spawn", "", f"lowest collider is {support_z:.4f} m above floor"))
        if support_z < -penetration_tol_m:
            issues.append(VisualPhysicsIssue("spawn_penetration", "", f"lowest collider is {support_z:.4f} m below floor"))
    checked = 0
    if check_contact_visuals is None:
        check_contact_visuals = enforce_floor_contact
    for geom_id in cosmetic:
        if not check_contact_visuals:
            break
        lo, hi = bounds(geom_id)
        if lo[2] > support_z + contact_band_m:
            continue
        checked += 1
        same_body = [physical_bounds[i] for i in physical if model.geom_bodyid[i] == model.geom_bodyid[geom_id]]
        contained = any(np.all(lo >= plo - containment_tol_m) and np.all(hi <= phi + containment_tol_m)
                        for plo, phi in same_body)
        if not contained:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            issues.append(VisualPhysicsIssue(
                "contact_visual_outside_collider", name,
                "a visual-only surface near the floor extends beyond every collider on its body"))
    return VisualPhysicsReport(not issues, round(float(support_z), 6), checked, issues)


def audit_gene(gene) -> VisualPhysicsReport:
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    spawn = standing_spawn_z(gene, clearance=0.002, meshed=False)
    # Infer support from authored structure rather than robot_class. A free drone is
    # intentionally airborne; a novel walker with foot/leg/wheel semantics is not.
    support_words = {"foot", "feet", "paw", "hoof", "wheel", "leg", "tread", "track"}

    def support_tokens(segment) -> set[str]:
        role = str((getattr(segment, "geometry", None) or {}).get("semantic_role", ""))
        text = f"{getattr(segment, 'name', '')}_{role}".lower().replace("-", "_")
        return {token for token in text.split("_") if token}

    expects_ground_support = getattr(gene, "base_mount", None) == "free" and any(
        support_tokens(segment) & support_words for segment in getattr(gene, "segments", [])
    )
    return audit_mjcf(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn),
        enforce_floor_contact=expects_ground_support,
        check_contact_visuals=expects_ground_support,
    )


def require_visual_physics_alignment(gene) -> VisualPhysicsReport:
    report = audit_gene(gene)
    if not report.ok:
        details = "; ".join(f"{issue.code}:{issue.geom}" for issue in report.issues)
        raise ValueError(f"visual/physics alignment gate failed: {details}")
    return report
