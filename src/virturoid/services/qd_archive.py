"""Quality-Diversity archive (breakthrough plan WS2/N2 + the metrics dashboard) — a MAP-Elites grid that the
night-shift banks VERIFIED candidates into, keeping the highest-fitness ELITE per behavior-descriptor cell.

Why QD and not a flat top-K: an autonomous explorer that only keeps the single best design collapses to one
family and stops discovering. A MAP-Elites archive keeps the best design *per behavior niche* (e.g. per
(n_legs, forward-speed) cell), so diversity is preserved and the frontier keeps expanding -- the substrate for
the compounding knowledge stock. The plan's dashboard reads three numbers off it: ANNECS-V (count of novel
cells ever filled -- must grow ~linearly or the descriptor space is saturated), QD-score (sum of elite
fitnesses -- total captured value), and coverage (filled / total cells). Pure data structure, no deps -- the
night-shift and the flywheel write to it; tests drive it with synthetic descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Elite:
    item: object                     # the banked artifact (gene/policy handle/record) — opaque to the archive
    descriptor: tuple                # the raw behavior descriptor
    fitness: float
    cell: tuple                      # the discretized cell index


@dataclass
class QDArchive:
    """MAP-Elites archive over ``dims`` = list of ``(name, lo, hi)``; each descriptor dim is binned into ``bins``
    buckets. ``add`` keeps the elite (max fitness) per cell and reports whether the cell was newly filled."""
    dims: list
    bins: int = 8
    cells: dict = field(default_factory=dict)          # cell -> Elite
    _novel_ever: int = 0                               # cells filled at least once (monotone; ANNECS-V)

    def _cell(self, descriptor) -> tuple:
        if len(descriptor) != len(self.dims):
            raise ValueError(f"descriptor has {len(descriptor)} dims, archive expects {len(self.dims)}")
        idx = []
        for v, (_name, lo, hi) in zip(descriptor, self.dims):
            if hi <= lo:
                idx.append(0)
                continue
            frac = (float(v) - lo) / (hi - lo)
            b = int(frac * self.bins)
            idx.append(min(self.bins - 1, max(0, b)))   # clamp out-of-range descriptors into the edge cells
        return tuple(idx)

    def add(self, item, descriptor, fitness: float) -> dict:
        """Insert a candidate. Keeps it only if its cell is empty or it beats the cell's current elite. Returns
        ``{added, novel_cell, replaced, cell}`` so the caller (night-shift) can log a bank / novel-cell event."""
        cell = self._cell(descriptor)
        cur = self.cells.get(cell)
        novel = cur is None
        if cur is not None and fitness <= cur.fitness:
            return {"added": False, "novel_cell": False, "replaced": False, "cell": cell}
        self.cells[cell] = Elite(item=item, descriptor=tuple(descriptor), fitness=float(fitness), cell=cell)
        if novel:
            self._novel_ever += 1
        return {"added": True, "novel_cell": novel, "replaced": not novel, "cell": cell}

    def elites(self) -> list:
        return list(self.cells.values())

    def best(self) -> Elite | None:
        return max(self.cells.values(), key=lambda e: e.fitness) if self.cells else None

    def coverage(self) -> float:
        """Filled cells / total cells (0..1)."""
        total = self.bins ** len(self.dims)
        return len(self.cells) / total if total else 0.0

    def qd_score(self) -> float:
        """Sum of elite fitnesses -- total captured value across niches."""
        return float(sum(e.fitness for e in self.cells.values()))

    def novel_cells_filled(self) -> int:
        """ANNECS-V proxy: number of DISTINCT cells ever filled (monotone non-decreasing across a run)."""
        return self._novel_ever

    def snapshot(self) -> dict:
        """A compact dashboard reading (for the night-shift journal / UI)."""
        return {"filled": len(self.cells), "coverage": round(self.coverage(), 4),
                "qd_score": round(self.qd_score(), 4), "annecs_v": self.novel_cells_filled(),
                "best_fitness": round(self.best().fitness, 4) if self.cells else None}
