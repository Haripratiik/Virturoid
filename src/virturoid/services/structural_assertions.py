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
    measured_on: str = "drawn"     # "drawn" (the meshed model) or "colliders" (the primitive model)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "topology_edges": [list(edge) for edge in self.topology_edges],
            "assertions": [asdict(assertion) for assertion in self.assertions],
            "measured_on": self.measured_on,
        }


def evaluate_structural_assertions(gene, *, max_visible_seam_gap_m: float = 0.006,
                                   meshed: bool = True) -> StructuralAssertionReport:
    """Author and execute gap/topology contracts from this design's graph and rendered geometry.

    Bounds include decorative brackets and hubs, so a mechanical mate may bridge deliberately separated
    colliders without contaminating simulation contact geometry.

    MEASURED ON THE BODY THE CUSTOMER SEES (``gene_to_meshed_mjcf``) by default, because what this reports is a
    VISIBLE seam. The drawn model keeps every collider where it was and adds the drawn shells, so it can close a
    gap the bare colliders show but cannot open one. The default used to be the colliders alone, and that made
    the critique wrong about the product's own default body: measured 2026-10-11, every composed walker
    (quadruped, tripod, hexapod, octopod) carried a HIGH "visible parent-child seam is detached by 0.037-0.094 m"
    on its head, because its trunk collider is deliberately shorter than the trunk it draws and the head sits at
    the drawn nose. In the drawn model every one of them mates. ``meshed=False`` still measures the colliders
    alone; if the drawn model cannot be built, the colliders are measured instead and ``measured_on`` says so.

    The drawn model matters most for imported robots: their links are drawn from the customer's own baked STL,
    which no amend can resize, so an edit could lengthen the colliders (staying mated, and therefore invisible
    to the collider measurement) while the drawn body came apart into floating chunks.

    A PART WELDED TO ITS PARENT'S RIGID PIECE IS ATTACHED THROUGH IT. A quadcopter's rotor is a fixed child of
    the hub that sits on the tip of a fixed arm, which is also a child of the hub: the arm carries the rotor and
    the rotor never touches the hub, so the bare parent-to-child test called all four rotors detached (0.085 m).
    The parent side of a seam is the parent plus every body welded to it, up or down, outside the child's own
    subtree; the child side is the child plus the bodies welded below it. A jointed sibling never bridges: it
    moves, and the part it touched would float as soon as it did.
    """
    issues = gene.validate()
    edges = tuple((segment.parent, segment.name) for segment in gene.segments if segment.parent is not None)
    assertions: list[StructuralAssertion] = [
        StructuralAssertion("kinematic_tree", not issues,
                            "single rooted acyclic tree" if not issues else "; ".join(issues), len(edges))
    ]
    if issues:
        return StructuralAssertionReport(False, edges, tuple(assertions))

    measured_on = "drawn" if meshed else "colliders"
    try:
        import mujoco
        import numpy as np
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z

        spawn_z = standing_spawn_z(gene)
        xml = None
        if meshed:
            try:
                xml = gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=spawn_z)
            except Exception:  # noqa: BLE001 - the colliders alone answer conservatively (they only add gaps)
                measured_on = "colliders"
        if xml is None:
            xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn_z)
        model = mujoco.MjModel.from_xml_string(xml)
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

        boxes = {name: union(bounds) for name, bounds in by_body.items() if bounds}
        welded = {s.name for s in gene.segments if s.parent is not None and s.joint_type in (None, "fixed")}
        parent_of = {child: parent for parent, child in edges}
        children: dict[str, list[str]] = {}
        for parent, child in edges:
            children.setdefault(parent, []).append(child)

        def subtree(name: str) -> set[str]:
            out, stack = set(), [name]
            while stack:
                node = stack.pop()
                if node not in out:
                    out.add(node)
                    stack.extend(children.get(node, ()))
            return out

        def rigid_piece(name: str, exclude: set[str], *, upward: bool) -> set[str]:
            out, stack = set(), [name]
            while stack:
                node = stack.pop()
                if node in out or node in exclude:
                    continue
                out.add(node)
                stack.extend(c for c in children.get(node, ()) if c in welded)
                if upward and node in welded and parent_of.get(node) is not None:
                    stack.append(parent_of[node])
            return out

        def box_gap(a: str, b: str) -> float:
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            return float(np.linalg.norm(np.maximum(0.0, np.maximum(alo - bhi, blo - ahi))))

        for parent, child in edges:
            if not by_body.get(parent) or not by_body.get(child):
                assertions.append(StructuralAssertion(
                    f"seam:{parent}->{child}", False, "one side of the seam has no rendered geometry"))
                continue
            gap, via = box_gap(parent, child), parent
            if gap > max_visible_seam_gap_m:
                child_side = rigid_piece(child, set(), upward=False) & set(boxes)
                parent_side = rigid_piece(parent, subtree(child), upward=True) & set(boxes)
                for a in parent_side:
                    for b in child_side:
                        g = box_gap(a, b)
                        if g < gap:
                            gap, via = g, a
            mated = gap <= max_visible_seam_gap_m
            if not mated:
                detail = f"visible parent-child seam is detached by {gap:.4f} m"
            elif via != parent:
                detail = f"visible geometry mates across this edge through {via!r}, welded to {parent!r}"
            else:
                detail = "visible bracket/hub/link geometry mates across this edge"
            assertions.append(StructuralAssertion(f"seam:{parent}->{child}", mated, detail, round(gap, 6)))
    except Exception as exc:  # noqa: BLE001 - an unexecutable contract is itself a failed assertion
        assertions.append(StructuralAssertion("mjcf_geometry_contract", False,
                                              f"could not execute structural geometry assertions: {exc}"))
    return StructuralAssertionReport(all(assertion.ok for assertion in assertions), edges, tuple(assertions),
                                     measured_on)


def require_structural_assertions(gene) -> StructuralAssertionReport:
    report = evaluate_structural_assertions(gene)
    if not report.ok:
        failed = [assertion.name for assertion in report.assertions if not assertion.ok]
        raise ValueError("structural assertion gate failed: " + ", ".join(failed[:6]))
    return report
