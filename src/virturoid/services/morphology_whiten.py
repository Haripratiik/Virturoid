"""Whiten the morphology feature vector so cosine distance actually discriminates.

The raw feature vector lives in the positive orthant (every feature >= 0), which squeezes cosine into a narrow
high band — measured, dog/hexapod/octopus all sit at cosine 0.91-1.0. Standardizing each feature (subtract the
corpus mean, divide by the corpus std) MEAN-CENTERS the space, so cosine becomes a Pearson correlation across
features and differences that were swamped (a 14-DoF dog vs a 42-DoF centipede) open up. This also RETIRES the
hand-tuned ``_SCALES`` magic numbers: the scale of every feature is now fit from data variance.

Design:
  * ``fit_whitener(genes)`` — per-feature mean/std over a representative corpus (deterministic).
  * ``apply_whiten(vec, w)`` — (v - mean) / std, with std clamped to >=eps so a CONSTANT feature maps to 0 and
    the DIMENSION IS PRESERVED (no silent truncation downstream; a dead feature simply contributes nothing).
  * persisted JSON at ``build/models/body_whitener.json``; absent -> callers use the raw rich vector (still an
    improvement over the 29-D one). Pure-python, no numpy required.
"""
from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.gene import RobotGene
from virturoid.services.install_paths import anchored
from virturoid.services.morphology_embedding import RICH_FEATURE_NAMES, embed_gene_rich

#: ANCHORED (see ``services.install_paths``): ``load_whitener`` returns None when absent and ``apply_whiten``
#: then returns the RAW vector, so a CWD-relative default silently un-whitened the embedding index -- the
#: exact mass-tyranny pathology Wave 1 exists to remove -- with nothing logged and no error raised.
DEFAULT_WHITENER_PATH = anchored("build/models/body_whitener.json")
_EPS = 1e-6
_cache: dict | None = None
_cache_mtime: float | None = None


def fit_whitener(genes: list[RobotGene], *, embed_fn=embed_gene_rich, clamp_frac: float = 0.25) -> dict:
    """Fit per-feature mean/std over the corpus embeddings, plus a std FLOOR = ``clamp_frac`` * max(std). The floor
    is robust scaling: it caps how much a LOW-variance feature can be amplified, so a feature that is rare in the
    fit corpus (e.g. a gripper's fingers when most bodies are legged) can't dominate cosine by being divided by a
    tiny std. Without it, whitening trades the mass-tyranny pathology for a rare-feature-tyranny one. Deterministic;
    needs >=2 genes."""
    vecs = [embed_fn(g) for g in genes]
    if len(vecs) < 2:
        raise ValueError("need >=2 genes to fit a whitener")
    dim = len(vecs[0])
    mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    std = [(sum((v[i] - mean[i]) ** 2 for v in vecs) / len(vecs)) ** 0.5 for i in range(dim)]
    floor = clamp_frac * (max(std) or 1.0)
    return {"names": list(RICH_FEATURE_NAMES), "mean": mean, "std": std, "floor": floor, "n_fit": len(vecs)}


def apply_whiten(vec: list[float], w: dict | None) -> list[float]:
    """Standardize ``vec`` by the fitted stats with a std floor (robust scaling). Dim-preserving: a constant
    feature -> ~0 (its value equals the mean); a rare feature is scaled, not exploded."""
    if not w:
        return list(vec)
    mean, std = w["mean"], w["std"]
    floor = max(w.get("floor", 0.0), _EPS)
    m = min(len(vec), len(mean))
    return [(vec[i] - mean[i]) / max(std[i], floor) for i in range(m)]


def save_whitener(w: dict, path: Path | None = None) -> Path:
    path = path or DEFAULT_WHITENER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(w))
    return path


def load_whitener(path: Path | None = None) -> dict | None:
    """Load the persisted whitener, cached and invalidated on file mtime. None if absent."""
    global _cache, _cache_mtime
    path = path or DEFAULT_WHITENER_PATH
    if not path.exists():
        return None
    mt = path.stat().st_mtime
    if _cache is not None and _cache_mtime == mt:
        return _cache
    _cache = json.loads(path.read_text())
    _cache_mtime = mt
    return _cache


def default_representative_prompts() -> list[str]:
    """A deterministic spread of bodies to fit the whitener on — spans the archetypes the index must separate."""
    return [
        "a small quadruped robot dog", "a large heavy quadruped robot", "a medium quadruped robot",
        "a six-legged hexapod robot", "an eight-legged spider robot", "a many-legged centipede robot",
        "an octopus robot with eight arms", "a two-legged humanoid robot", "a two-legged raptor robot",
        "a six-legged crab robot", "a snake robot that slithers", "an eel robot that swims",
        "a six-axis robot arm on a table", "a small tabletop gripper arm", "a four-wheeled rover robot",
        "a wheeled delivery robot", "a bird drone that flies", "a monopod hopping robot",
        "a tripod three-legged robot", "a horse quadruped robot", "a gecko climbing robot",
        "a humanoid with two arms", "a heavy industrial arm", "a turtle robot with a shell",
    ]


def fit_default_whitener(prompts: list[str] | None = None, *, save: bool = True) -> dict:
    """Compose the representative bodies and fit + persist the default whitener. CPU, no MuJoCo (compose only)."""
    from virturoid.services.morphology_composer import compose_robot
    prompts = prompts or default_representative_prompts()
    genes = []
    for p in prompts:
        try:
            genes.append(compose_robot(p))
        except Exception:  # noqa: BLE001 - skip a prompt that fails to compose; fit on the rest
            continue
    w = fit_whitener(genes)
    if save:
        save_whitener(w)
    return w
