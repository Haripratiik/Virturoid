"""MAP-Elites design archive (Pillar 1 Curator / POET): illuminate a grid of DIVERSE trainable bodies.

Instead of collapsing the design search onto a single best body, keep the best (body + its best-response
controller + honest score) per morphology NICHE. This is the Curator role from the agentic loop
(docs/usp_and_two_pillar_plan.md Pillar 1): it covers the design space and guards against
diversity-collapse — the failure where a shared representation/controller homogenizes every design,
which BOTH RoboCat and the morphological-pretraining paper flag as the key risk for a self-improving
loop. Each filled cell is also a warm-start seed for its niche (retrieve the same-niche elite and amend).

The niche descriptor is a coarse, interpretable morphology signature (DOF x limb-count x size-class);
insertion is per-cell elitism (a candidate only displaces its cell's incumbent if it scores higher). The
archive's QD-score (sum of elite scores) and coverage (filled cells) are the Curator's health metrics —
rising QD-score with maintained coverage is illumination; rising score with collapsing coverage is the
warning sign. Pure-Python and dependency-free; JSON-persistable.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.gene import RobotGene


def descriptor(gene: RobotGene) -> tuple[int, int, int]:
    """A coarse morphology niche key: (DOF band, limb count, size class).

    - DOF band groups actuated-joint counts (0-2, 3-5, ... ) so a 3-DOF arm and a 12-DOF quad differ.
    - Limb count = number of leaf chains (legs/arms/fingers), the gross body plan.
    - Size class bins total link length (tabletop / small / medium / large).
    Deterministic and cheap — the same body always lands in the same cell.
    """
    from virturoid.services.morphology_embedding import _depth_and_branching

    dof = len(gene.actuated_joints())
    _depth, _br, _bp, n_leaves, _reach, _children = _depth_and_branching(gene)
    total_length = sum(s.length_m for s in gene.segments)
    d_dof = min(dof, 18) // 3
    d_limbs = min(int(n_leaves), 8)
    d_size = 0 if total_length < 0.5 else 1 if total_length < 1.0 else 2 if total_length < 2.0 else 3
    return (d_dof, d_limbs, d_size)


class MapElitesArchive:
    """A grid of morphology niches, each holding the best-scoring (body, controller) found for it."""

    def __init__(self, descriptor_fn=descriptor) -> None:
        self.cells: dict[tuple, dict] = {}
        self._descriptor_fn = descriptor_fn

    def insert(self, gene: RobotGene, score: float, *, controller: dict | None = None,
               meta: dict | None = None) -> str:
        """Try to place ``gene`` (score = its honest best-response fitness) in its niche. Returns
        ``'added'`` (new cell), ``'improved'`` (beat the cell's incumbent), or ``'rejected'`` (worse)."""
        key = self._descriptor_fn(gene)
        cur = self.cells.get(key)
        if cur is not None and score <= cur["score"]:
            return "rejected"
        action = "improved" if cur is not None else "added"
        self.cells[key] = {
            "descriptor": list(key), "score": round(float(score), 4),
            "gene": gene.to_dict() if hasattr(gene, "to_dict") else gene,
            "controller": controller, "meta": meta or {},
        }
        return action

    def nearest_elite(self, gene: RobotGene) -> dict | None:
        """The elite occupying ``gene``'s niche (a same-body-plan warm-start seed), or ``None``."""
        return self.cells.get(self._descriptor_fn(gene))

    def elites(self) -> list[dict]:
        return list(self.cells.values())

    def coverage(self) -> int:
        """Number of filled niches (illumination breadth)."""
        return len(self.cells)

    def qd_score(self) -> float:
        """Quality-Diversity score: sum of the elite scores across all filled niches (the Curator's
        headline metric — rises as the archive finds better bodies in MORE niches)."""
        return round(sum(c["score"] for c in self.cells.values()), 4)

    def best(self) -> dict | None:
        return max(self.cells.values(), key=lambda c: c["score"]) if self.cells else None

    def summary(self) -> dict:
        b = self.best()
        return {"coverage": self.coverage(), "qd_score": self.qd_score(),
                "best_score": (b["score"] if b else None),
                "best_descriptor": (b["descriptor"] if b else None)}

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # keys are tuples -> serialize as a list of cells (each already carries its descriptor)
        p.write_text(json.dumps({"cells": list(self.cells.values())}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, descriptor_fn=descriptor) -> "MapElitesArchive":
        arc = cls(descriptor_fn=descriptor_fn)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for cell in data.get("cells", []):
            arc.cells[tuple(cell["descriptor"])] = cell
        return arc
