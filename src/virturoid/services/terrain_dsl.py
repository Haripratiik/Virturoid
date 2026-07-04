"""Procedural terrain DSL (breakthrough v5, B2). A small, composable library of ATOMIC terrain tiles — flat,
slope, stairs, waves, rough/stones, friction patches — each realized as a numpy heightfield (+ per-tile friction)
that compiles to a MuJoCo ``<hfield>`` the MJX trainer can drop under any robot. Terrain is simultaneously the
best-documented locomotion fix (a robot that only ever trains on a plane learns a plane-specific gait) and the
substrate the Curriculum Author (B3) manipulates.

Design constraints:
- **MJX-compatible**: one hfield asset, one static floor geom. Capsule/sphere feet (already the trainer default)
  roll over hfield facets cleanly; box feet catch. Heights live in [0, 1] and scale by the geom's z-size.
- **Deterministic**: every generator takes a ``seed`` and a ``difficulty`` in [0, 1]; same inputs -> same field.
  This is what makes the Rudin per-env curriculum reproducible and pre-registrable on VIRT-Bench.
- **Difficulty is monotone**: difficulty 0 is (near-)flat for every tile type, so a curriculum can always fall
  back to something learnable; higher difficulty raises amplitude/step-height/slope/roughness.
- **Pure numpy**: no MuJoCo import needed to BUILD a field (import only to emit XML), so it is trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TILE_TYPES = ("flat", "slope", "stairs", "waves", "rough", "stones", "friction")


@dataclass
class Terrain:
    """A realized terrain patch: an (nrow, ncol) heightfield in [0, 1], the metric size it spans, the physical
    z-scale that maps 1.0 -> ``height_m`` metres, and a per-cell friction multiplier field (1.0 == nominal)."""
    heights: np.ndarray                 # (nrow, ncol) float in [0, 1]
    size_m: float                       # side length of the square patch (metres)
    height_m: float                     # physical elevation of height==1.0 (metres)
    friction: np.ndarray                # (nrow, ncol) tangential-friction multiplier
    tile: str = "flat"
    difficulty: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def resolution(self) -> int:
        return self.heights.shape[0]


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, n)
    return np.meshgrid(u, u)


def make_tile(tile: str, *, difficulty: float = 0.3, seed: int = 0, n: int = 64, size_m: float = 8.0,
              height_m: float = 0.25) -> Terrain:
    """Build one atomic terrain tile. ``difficulty`` in [0, 1] monotonically scales the hard parameter of each
    tile (slope angle, step height, wave amplitude, roughness, stone density, friction spread)."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    xx, yy = _grid(n)
    h = np.zeros((n, n), dtype=float)
    fric = np.ones((n, n), dtype=float)
    meta: dict = {}

    if tile == "flat":
        pass
    elif tile == "slope":
        grade = d * 0.6                                  # up to ~31 deg over the patch
        h = yy * grade
        meta["grade"] = grade
    elif tile == "stairs":
        n_steps = int(2 + round(d * 8))                  # 2..10 steps across the patch
        step_h = d * 0.9                                 # normalized step rise (scaled by height_m at emit time)
        h = np.floor(yy * n_steps) / max(1, n_steps) * step_h
        meta["n_steps"] = n_steps; meta["step_h_m"] = step_h * height_m
    elif tile == "waves":
        k = 1.5 + d * 4.5                                # spatial frequency
        amp = d
        h = 0.5 * amp * (1.0 + np.sin(2 * np.pi * k * xx) * np.cos(2 * np.pi * k * yy))
        meta["wavelength_m"] = size_m / max(1e-6, k)
    elif tile == "rough":
        amp = d
        base = rng.standard_normal((n, n))
        # smooth the white noise a little so it is walkable roughness, not spikes
        for _ in range(2):
            base = 0.25 * (np.roll(base, 1, 0) + np.roll(base, -1, 0) + np.roll(base, 1, 1) + np.roll(base, -1, 1))
        base -= base.min()
        h = amp * base / max(1e-6, base.max())
        meta["rms"] = float(np.std(h))
    elif tile == "stones":
        n_stones = int(d * 60)                            # discrete raised stepping-stones
        rad = max(1, int(n * (0.10 - 0.05 * d)))          # smaller stones at higher difficulty
        for _ in range(n_stones):
            cy, cx = rng.integers(0, n, size=2)
            yl, yh = max(0, cy - rad), min(n, cy + rad); xl, xh = max(0, cx - rad), min(n, cx + rad)
            h[yl:yh, xl:xh] = np.maximum(h[yl:yh, xl:xh], 0.4 + 0.6 * d * rng.random())
        meta["n_stones"] = n_stones
    elif tile == "friction":
        # a flat tile with slippery PATCHES (the classic ice-patch DR): friction dips in random blobs
        n_patch = int(1 + d * 6)
        lo = 1.0 - 0.8 * d                                # slipperiest patch friction multiplier
        for _ in range(n_patch):
            cy, cx = rng.random(2)
            r = 0.08 + 0.12 * rng.random()
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
            fric[m] = lo
        meta["min_friction_mult"] = lo
    else:
        raise ValueError(f"unknown terrain tile {tile!r} (known: {TILE_TYPES})")

    h = np.clip(h, 0.0, 1.0)
    return Terrain(heights=h, size_m=float(size_m), height_m=float(height_m), friction=fric,
                   tile=tile, difficulty=d, meta=meta)


def compose_row(tiles: list[Terrain], *, gap: int = 0) -> Terrain:
    """Concatenate tiles left-to-right into one wide patch (a Rudin-style terrain 'course'). All tiles must share
    resolution + height scale; friction fields concatenate alongside."""
    if not tiles:
        raise ValueError("compose_row needs at least one tile")
    n = tiles[0].resolution
    assert all(t.resolution == n for t in tiles), "all tiles must share resolution"
    hs = np.concatenate([t.heights for t in tiles], axis=1)
    fr = np.concatenate([t.friction for t in tiles], axis=1)
    total_w = sum(t.size_m for t in tiles)
    return Terrain(heights=hs, size_m=total_w, height_m=tiles[0].height_m, friction=fr,
                   tile="row", difficulty=max(t.difficulty for t in tiles),
                   meta={"tiles": [t.tile for t in tiles]})


def hfield_asset_xml(t: Terrain, *, name: str = "terrain") -> str:
    """The ``<hfield>`` asset element (goes inside ``<asset>``). size = (radius_x, radius_y, z_top, z_base). The
    ELEVATION DATA is not inlined (large fields belong in ``mjModel.hfield_data``); call :func:`hfield_data`
    after compile and assign it. nrow/ncol define the grid the data must match."""
    nrow, ncol = t.heights.shape
    rx = t.size_m / 2.0
    ry = (t.size_m * ncol / max(1, nrow)) / 2.0 if ncol != nrow else rx
    return (f'<hfield name="{name}" nrow="{nrow}" ncol="{ncol}" '
            f'size="{rx:.4f} {ry:.4f} {max(1e-3, t.height_m):.4f} 0.1"/>')


def hfield_geom_xml(t: Terrain, *, name: str = "terrain", pos=(0.0, 0.0, 0.0), friction: float = 1.0) -> str:
    """The static floor ``<geom>`` that renders the hfield. ``friction`` is the nominal tangential friction; the
    per-cell :attr:`Terrain.friction` field (slippery patches) is applied via per-contact overrides if the caller
    supports them, else it is metadata."""
    return (f'<geom name="{name}_floor" type="hfield" hfield="{name}" '
            f'pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" material="mat_ground" '
            f'friction="{friction:.3f} 0.01 0.001" contype="1" conaffinity="1"/>')


def hfield_data(t: Terrain) -> np.ndarray:
    """Flat, row-major elevation array in [0, 1] to assign to ``mjModel.hfield_data`` after compiling a model that
    declares this terrain's :func:`hfield_asset_xml`. MuJoCo scales it by the asset's z_top (== ``height_m``)."""
    return t.heights.flatten().astype(np.float32)


@dataclass
class RudinCurriculum:
    """Rudin et al. (2021) game-inspired terrain curriculum: N parallel envs each hold a difficulty LEVEL on a
    ladder. An env that SUCCEEDS (travelled far enough / stayed up) is promoted a level; one that FAILS is
    demoted. Difficulty rises only as fast as the fleet can handle, and a solved fleet's mean level is a single,
    honest 'how hard can it walk' scalar. Pure state (numpy) so the MJX trainer just reads ``levels`` per env."""
    n_envs: int
    n_levels: int = 10
    promote_frac: float = 0.6          # travelled >= promote_frac * patch length -> harder
    demote_frac: float = 0.2           # travelled <  demote_frac  * patch length -> easier
    levels: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.levels is None:
            # start everyone low; a spread-out start is fine too but low-start is the safe default
            self.levels = np.zeros(self.n_envs, dtype=np.int32)

    def difficulty(self) -> np.ndarray:
        """Per-env difficulty in [0, 1] from the integer level ladder."""
        return self.levels.astype(float) / max(1, self.n_levels - 1)

    def update(self, distance_frac: np.ndarray, alive: np.ndarray) -> dict:
        """Promote/demote each env from its episode result. ``distance_frac`` = travelled / patch-length,
        ``alive`` = survived (didn't fall). Returns a summary for logging."""
        distance_frac = np.asarray(distance_frac, dtype=float)
        alive = np.asarray(alive, dtype=bool)
        promote = alive & (distance_frac >= self.promote_frac)
        demote = (~alive) | (distance_frac < self.demote_frac)
        self.levels = np.clip(self.levels + promote.astype(np.int32) - demote.astype(np.int32),
                              0, self.n_levels - 1)
        return {"mean_level": float(self.levels.mean()), "max_level": int(self.levels.max()),
                "promoted": int(promote.sum()), "demoted": int(demote.sum()),
                "mean_difficulty": float(self.difficulty().mean())}
