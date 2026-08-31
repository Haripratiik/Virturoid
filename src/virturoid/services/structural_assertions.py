"""Executable, per-design topology and seam assertions for generated robot bodies."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructuralAssertion:
    name: str
    ok: bool
    detail: str
    value: float | int | str | None = None


@dataclass(frozen=True)
class StructuralAssertionReport:
    ok: bool
    topology_edges: tuple[tuple[str, str], ...]
    assertions: tuple[StructuralAssertion, ...]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "topology_edges": [list(edge) for edge in self.topology_edges],
            "assertions": [asdict(assertion) for assertion in self.assertions],
        }


def evaluate_structural_assertions(gene, *, max_visible_seam_gap_m: float = 0.006,
                                   meshed: bool = False) -> StructuralAssertionReport:
    """Author and execute gap/topology contracts from this design's graph and rendered geometry.

    Bounds include decorative brackets and hubs, so a mechanical mate may bridge deliberately separated
    colliders without contaminating simulation contact geometry.

    ``meshed`` measures the model the CUSTOMER SEES (``gene_to_meshed_mjcf``) instead of the primitive one. It
    matters for imported robots: their links are drawn from the customer's own baked STL, which no amend can
    resize, so an edit could lengthen the colliders (staying mated, and therefore invisible to the primitive
    measurement) while the drawn body came apart into floating chunks. Off by default so the existing
    ``submit_design`` gate keeps measuring exactly what it always measured.
    """
    issues = gene.validate()
    edges = tuple((segment.parent, segment.name) for segment in gene.segments if segment.parent is not None)
    assertions: list[StructuralAssertion] = [
        StructuralAssertion("kinematic_tree", not issues,
                            "single rooted acyclic tree" if not issues else "; ".join(issues), len(edges))
    ]
    if issues:
        return StructuralAssertionReport(False, edges, tuple(assertions))

    try:
        import mujoco
        import numpy as np
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z

        _build = gene_to_meshed_mjcf if meshed else compile_gene_to_mjcf
        model = mujoco.MjModel.from_xml_string(
            _build(gene, include_floor=True, spawn_z=standing_spawn_z(gene, meshed=meshed))
        )
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)

        def geom_bounds(geom_id: int):
            center, half = model.geom_aabb[geom_id, :3], model.geom_aabb[geom_id, 3:]
            world = data.geom_xpos[geom_id] + (center + signs * half) @ data.geom_xmat[geom_id].reshape(3, 3).T
            return world.min(axis=0), world.max(axis=0)

        by_body: dict[str, list[tuple]] = {}
        for body_id in range(1, model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                by_body[body_name] = [geom_bounds(i) for i in range(model.ngeom)
                                      if int(model.geom_bodyid[i]) == body_id]

        def union(bounds):
            return np.min([lo for lo, _ in bounds], axis=0), np.max([hi for _, hi in bounds], axis=0)

        for parent, child in edges:
            if not by_body.get(parent) or not by_body.get(child):
                assertions.append(StructuralAssertion(
                    f"seam:{parent}->{child}", False, "one side of the seam has no rendered geometry"))
                continue
            plo, phi = union(by_body[parent]); clo, chi = union(by_body[child])
            separation = np.maximum(0.0, np.maximum(plo - chi, clo - phi))
            gap = float(np.linalg.norm(separation))
            assertions.append(StructuralAssertion(
                f"seam:{parent}->{child}", gap <= max_visible_seam_gap_m,
                ("visible bracket/hub/link geometry mates across this edge" if gap <= max_visible_seam_gap_m
                 else f"visible parent-child seam is detached by {gap:.4f} m"),
                round(gap, 6),
            ))
    except Exception as exc:  # noqa: BLE001 - an unexecutable contract is itself a failed assertion
        assertions.append(StructuralAssertion("mjcf_geometry_contract", False,
                                              f"could not execute structural geometry assertions: {exc}"))
    return StructuralAssertionReport(all(assertion.ok for assertion in assertions), edges, tuple(assertions))


def require_structural_assertions(gene) -> StructuralAssertionReport:
    report = evaluate_structural_assertions(gene)
    if not report.ok:
        failed = [assertion.name for assertion in report.assertions if not assertion.ok]
        raise ValueError("structural assertion gate failed: " + ", ".join(failed[:6]))
    return report
