"""Manufacture the PHYSICS-VERIFIED transfer matrix — the ground truth the robotics embedding is graded against.

The moat is "distance in the body-vector predicts control transfer". To *measure* that (never assert it) we need
labelled transfer outcomes: does body A's working gait actually produce a CREDIBLE walk when replayed on body B?
We already own the machinery — ``search_gait`` finds each body its own deployable gait, ``evaluate_gait`` replays
any gait on any body with an un-gameable credible-walk verdict (signed forward + upright + cadence). So for a fixed
zoo of bodies we compute, once and cache:

  * each body's OWN reference gait (``search_gait``) + whether it's a credible source at all,
  * ``transfer[i][j]`` = body i's gait rolled out on body j → credible & survived (+ the forward_m for a graded label).

This cache is FIXED ground truth (physics); ``embedding_eval`` then scores any embedding against it in milliseconds
(recomputing only the cheap vectors), so every embedding change is proven by a measured before/after. Deterministic
(seed 0 throughout); CPU MuJoCo. Run once via ``build_transfer_corpus``; consumers read ``load_corpus``.
"""
from __future__ import annotations

import json
from pathlib import Path

from virturoid.services.install_paths import anchored

# a deliberately diverse zoo: a strong LEGGED core (where gait-hint retrieval actually happens and today's embedding
# is blind, cosine 0.91-1.0) + serial/wheeled/arm outliers so the eval also catches "homeless body" retrieval.
DEFAULT_ZOO: list[tuple[str, str]] = [
    # a broad QUADRUPED core (reliable gait sources -> a dense, discriminative transfer matrix where borrowing
    # actually happens) + many-leg / biped / serial / wheeled / arm bodies as targets and outliers. Graduated
    # sizes/proportions give the transfer matrix FINER-than-class structure for a learned metric to exploit.
    ("tquad",  "a tiny quadruped robot"),
    ("sdog",   "a small quadruped robot dog"),
    ("mquad",  "a medium quadruped robot"),
    ("lquad",  "a large heavy quadruped robot"),
    ("hquad",  "a huge heavy quadruped robot"),
    ("horse",  "a tall horse quadruped robot with long legs"),
    ("gecko",  "a small wide gecko lizard robot with four splayed legs"),
    ("cat",    "an agile cat quadruped robot"),
    ("turtle", "a squat turtle robot with four short legs"),
    ("bull",   "a stocky bull quadruped robot"),
    ("hexa",   "a six-legged hexapod robot"),
    ("whexa",  "a wide six-legged hexapod robot"),
    ("ant",    "a six-legged ant robot"),
    ("crab",   "a six-legged crab robot"),
    ("spider", "an eight-legged spider robot"),
    ("wspider","a wide eight-legged spider robot"),
    ("octo",   "an octopus robot with eight arms"),
    ("cent",   "a many-legged centipede robot"),
    ("mille",  "a many-legged millipede robot"),
    ("biped",  "a two-legged humanoid robot"),
    ("raptor", "a two-legged raptor dinosaur robot"),
    ("snake",  "a snake robot that slithers"),
    ("eel",    "an eel robot that swims"),
    ("arm",    "a six-axis robot arm on a table"),
    ("grip",   "a small tabletop gripper arm"),
    ("rover",  "a four-wheeled rover robot"),
]

#: ANCHORED (see ``services.install_paths``) -- a measured-transfer corpus this install BUILT, not something
#: that should appear or vanish with the shell's pwd. ``load_corpus`` treats absent as empty.
DEFAULT_CACHE = anchored("build/data/transfer_corpus.json")


def _compose(prompt: str):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt)


def build_transfer_corpus(zoo: list[tuple[str, str]] | None = None, *, out_path: Path | None = None,
                          generations: int = 4, pop: int = 10, steps: int = 500, cred_forward: float = 0.25,
                          progress=None) -> dict:
    """Compose the zoo, find each body its own gait, and cross-evaluate every gait on every body → the transfer
    matrix. Caches to ``out_path``. ``transfer[i][j]`` is 1 iff body i's gait walks CREDIBLY (survived + upright +
    forward>=``cred_forward``) on body j. Slow (one-time); deterministic."""
    from virturoid.services.gait_search import evaluate_gait, search_gait
    zoo = zoo or DEFAULT_ZOO
    out_path = out_path or DEFAULT_CACHE

    def log(m):
        if progress:
            progress(m)

    bodies = []
    for bid, prompt in zoo:
        g = _compose(prompt)
        res = search_gait(g, generations=generations, pop=pop, steps=steps, seed=0)
        self_cred = bool(res.best_survived and res.best_credible and res.best_forward >= cred_forward)
        bodies.append({
            "id": bid, "prompt": prompt, "robot_class": getattr(g, "robot_class", None),
            "species": getattr(g, "species", None), "gene": g.to_dict(), "ref_gait": dict(res.best_params),
            "self_credible": self_cred, "self_forward": round(float(res.best_forward), 4),
        })
        log(f"[gait] {bid:7s} own gait forward={res.best_forward:+.3f} credible={self_cred}")

    n = len(bodies)
    transfer = [[0] * n for _ in range(n)]
    forward = [[0.0] * n for _ in range(n)]
    genes = [_reconstruct(b["gene"]) for b in bodies]
    for i, src in enumerate(bodies):
        if not src["self_credible"]:
            continue                                             # a body with no credible gait is not a source
        gait_i = src["ref_gait"]
        for j in range(n):
            r = evaluate_gait(genes[j], gait_i, steps=steps)
            cred = bool(r.get("survived") and r.get("credible") and float(r.get("forward", 0)) >= cred_forward)
            transfer[i][j] = 1 if cred else 0
            forward[i][j] = round(float(r.get("forward", 0)), 4)
        row = "".join(str(x) for x in transfer[i])
        log(f"[xfer] {src['id']:7s} -> {row}  ({sum(transfer[i])}/{n} bodies can borrow it)")

    corpus = {"bodies": bodies, "transfer": transfer, "forward": forward,
              "params": {"generations": generations, "pop": pop, "steps": steps, "cred_forward": cred_forward}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=1))
    log(f"SAVED {out_path}  ({n} bodies, {sum(sum(r) for r in transfer)} credible transfers)")
    return corpus


def _reconstruct(d: dict):
    from virturoid.schemas.gene import RobotGene
    return RobotGene.from_dict(d)


def load_corpus(path: Path | None = None) -> dict | None:
    path = path or DEFAULT_CACHE
    if not path.exists():
        return None
    corpus = json.loads(path.read_text())
    for b in corpus["bodies"]:
        b["_gene"] = _reconstruct(b["gene"])
    return corpus
