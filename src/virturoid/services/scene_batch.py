"""Scenes into the vectorized training loop (scene-gen plan S5, CPU enabler). The measured gap: the MJX trainer
never sees a generated scene — it trains thousands of envs on ONE bare model, so multi-scene diversity can't
reach the policy. MJX vmaps over DATA, not over model STRUCTURE, so the training-compatible mechanism is: compile
ONE model with K object SLOTS, and have each env's reset write the object poses/sizes of a DIFFERENT scene drawn
from the TRAIN pool. Absent objects are parked out of view + made non-colliding. This gives genuine per-env scene
variation (the anti-overfitting signal) inside a single MJX model.

This module owns the CPU half: it PACKS a SceneFamily's train pool into fixed-slot per-env arrays (poses, sizes,
presence) that the MJX reset consumes, assigns a scene to every env, and reports coverage. The held-out pool is
NEVER packed into training (the honest split). The MJX reset-randomization wiring that reads these arrays is the
GPU-validation surface. Pure numpy + deterministic -> unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# object types that are per-env variable (the manipulable/steerable content); walls/floor are structural and stay
# in the compiled model, not packed per-env.
_PACKABLE = {"cube", "box", "obstacle", "zone", "container"}


@dataclass
class SceneBatch:
    """Fixed-slot packing of a scene pool for MJX per-env reset. K = max packable objects across the pool. Arrays
    are (n_scenes, K, ...) so the trainer indexes ``[env_scene_id[e]]`` to get env e's scene."""
    k_slots: int
    scene_ids: list[str]
    poses: np.ndarray          # (n_scenes, K, 7) xyz + quat (wxyz); parked objects sit far below the floor
    sizes: np.ndarray          # (n_scenes, K, 3) full extents (m)
    presence: np.ndarray       # (n_scenes, K) 1.0 if the slot is a real object in that scene, else 0.0
    categories: list[list[str]]

    @property
    def n_scenes(self) -> int:
        return len(self.scene_ids)


_PARK = (0.0, 0.0, -5.0)       # where absent slots go: below the floor, contacts disabled by the reset via presence


def pack_scenes(scenes, *, k_slots: int | None = None) -> SceneBatch:
    """Pack a list of SceneGraphs into fixed K-slot arrays. K defaults to the max packable-object count in the
    pool. Objects beyond a scene's count are parked (presence 0). Deterministic (object order preserved)."""
    per_scene = []
    for s in scenes:
        objs = [o for o in s.objects if o.object_type in _PACKABLE]
        per_scene.append(objs)
    K = k_slots or max((len(o) for o in per_scene), default=1)
    n = len(scenes)
    poses = np.zeros((n, K, 7), dtype=float); poses[:, :, 3] = 1.0        # identity quat
    poses[:, :, :3] = _PARK
    sizes = np.full((n, K, 3), 0.01, dtype=float)
    presence = np.zeros((n, K), dtype=float)
    cats: list[list[str]] = []
    for i, objs in enumerate(per_scene):
        row_cats = []
        for j, o in enumerate(objs[:K]):
            x, y, z, *_ = o.pose_xyz_rpy
            poses[i, j, :3] = (x, y, z)
            sizes[i, j] = o.size_xyz if o.size_xyz is not None else (0.05, 0.05, 0.05)
            presence[i, j] = 1.0
            row_cats.append(o.category or o.object_type)
        row_cats += [""] * (K - len(row_cats))
        cats.append(row_cats)
    return SceneBatch(k_slots=K, scene_ids=[s.id for s in scenes], poses=poses, sizes=sizes,
                      presence=presence, categories=cats)


def assign_scenes_to_envs(n_envs: int, n_scenes: int, *, seed: int = 0) -> np.ndarray:
    """Assign a training scene to each of ``n_envs`` MJX envs — round-robin then shuffled so every scene gets
    near-equal env coverage (balanced multi-scene training) and the assignment is deterministic per seed."""
    base = np.arange(n_envs) % max(1, n_scenes)
    return np.random.default_rng(seed).permutation(base)


def build_training_batch(family, n_envs: int, *, seed: int = 0) -> dict:
    """The S5 entry point: pack the family's TRAIN pool (held-out excluded) and assign a scene to every env.
    Returns everything the MJX reset needs: the SceneBatch arrays + per-env scene ids + a coverage report. The
    held-out pool is deliberately absent — training must never touch it."""
    batch = pack_scenes(family.train)
    env_scene = assign_scenes_to_envs(n_envs, batch.n_scenes, seed=seed)
    counts = np.bincount(env_scene, minlength=batch.n_scenes)
    return {"batch": batch, "env_scene_id": env_scene,
            "coverage": {"n_scenes": batch.n_scenes, "k_slots": batch.k_slots,
                         "min_envs_per_scene": int(counts.min()), "max_envs_per_scene": int(counts.max())},
            "held_out_ids": [s.id for s in family.held_out]}
