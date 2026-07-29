"""Blinded, preregistered pairwise benchmark for robot-render believability.

The harness never judges its own images. It freezes prompts, views, pair order, endpoint and exclusions before
votes are collected, emits a public A/B ballot without the side key, and scores candidate preference with a
Wilson confidence interval. That turns "looks less crude" into a repeatable product metric.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROBOTS = {
    "dog": "a robot dog",
    "horse": "a robot horse",
    "hexapod": "a six-legged hexapod robot",
    "humanoid": "a humanoid robot",
    "arm": "a 6-axis robot arm with a gripper",
    "rover": "a four-wheeled warehouse rover",
}
VIEWS = {"front": 0, "front34": 45, "rear34": 135}
PROTOCOL_VERSION = "believability-pairwise-v1"


@dataclass(frozen=True)
class Pair:
    pair_id: str
    robot: str
    prompt: str
    view: str
    azimuth_deg: int
    image_a: str
    image_b: str
    candidate_side: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_protocol(baseline_dir: str | Path, candidate_dir: str | Path, out_dir: str | Path, *,
                     seed: int = 20260728) -> dict:
    """Freeze the 6x3 comparison and write separate public ballot/private key JSON artifacts."""
    baseline_dir, candidate_dir, out_dir = Path(baseline_dir), Path(candidate_dir), Path(out_dir)
    missing = [str(root / f"{robot}_{view}.png") for root in (baseline_dir, candidate_dir)
               for robot in ROBOTS for view in VIEWS if not (root / f"{robot}_{view}.png").is_file()]
    if missing:
        raise FileNotFoundError("missing fixed benchmark renders: " + ", ".join(missing))
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for robot, prompt in ROBOTS.items():
        for view, azimuth in VIEWS.items():
            baseline = (baseline_dir / f"{robot}_{view}.png").resolve()
            candidate = (candidate_dir / f"{robot}_{view}.png").resolve()
            candidate_side = "A" if rng.getrandbits(1) == 0 else "B"
            image_a, image_b = ((candidate, baseline) if candidate_side == "A" else (baseline, candidate))
            pairs.append(Pair(f"{robot}:{view}", robot, prompt, view, azimuth,
                              str(image_a), str(image_b), candidate_side))
    fingerprint = hashlib.sha256(json.dumps(
        [{"pair_id": p.pair_id, "a": _sha256(Path(p.image_a)), "b": _sha256(Path(p.image_b)),
          "candidate_side": p.candidate_side} for p in pairs], sort_keys=True).encode()).hexdigest()[:16]
    preregistration = {
        "protocol": PROTOCOL_VERSION, "protocol_id": fingerprint, "seed": seed,
        "primary_endpoint": "candidate wins / decisive A-or-B votes",
        "target_preference": 0.70, "confidence_interval": "95% Wilson score interval",
        "tie_policy": "reported separately and excluded from the binomial preference endpoint",
        "exclusions": ["choice is not A, B, or tie", "pair_id is not in the frozen ballot"],
        "instructions": "Choose the robot that looks more like believable, buildable robotic hardware. "
                        "Judge mechanical plausibility, proportion, articulation legibility and prompt fidelity.",
        "robots": ROBOTS, "views": VIEWS, "n_pairs": len(pairs),
    }
    ballot = {**preregistration, "pairs": [
        {k: v for k, v in asdict(pair).items() if k != "candidate_side"} for pair in pairs]}
    key = {"protocol": PROTOCOL_VERSION, "protocol_id": fingerprint,
           "candidate_side_by_pair": {pair.pair_id: pair.candidate_side for pair in pairs}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preregistration.json").write_text(json.dumps(preregistration, indent=2), encoding="utf-8")
    (out_dir / "ballot.json").write_text(json.dumps(ballot, indent=2), encoding="utf-8")
    (out_dir / "private_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return {"protocol_id": fingerprint, "n_pairs": len(pairs),
            "preregistration": str((out_dir / "preregistration.json").resolve()),
            "ballot": str((out_dir / "ballot.json").resolve()),
            "private_key": str((out_dir / "private_key.json").resolve())}


def _wilson(wins: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    p = wins / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def score_votes(key: dict | str | Path, votes: list[dict] | str | Path) -> dict:
    """Score blinded votes against the separately-held side key."""
    if isinstance(key, (str, Path)):
        key = json.loads(Path(key).read_text(encoding="utf-8"))
    if isinstance(votes, (str, Path)):
        votes = json.loads(Path(votes).read_text(encoding="utf-8"))
    if key.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("unsupported or missing protocol version")
    side_by_pair = key.get("candidate_side_by_pair") or {}
    candidate_wins = baseline_wins = ties = invalid = 0
    by_pair: dict[str, dict] = {}
    for vote in votes:
        pair_id = str(vote.get("pair_id", ""))
        choice = str(vote.get("choice", "")).strip().upper()
        if pair_id not in side_by_pair or choice not in ("A", "B", "TIE"):
            invalid += 1
            continue
        row = by_pair.setdefault(pair_id, {"candidate_wins": 0, "baseline_wins": 0, "ties": 0})
        if choice == "TIE":
            ties += 1
            row["ties"] += 1
        elif choice == side_by_pair[pair_id]:
            candidate_wins += 1
            row["candidate_wins"] += 1
        else:
            baseline_wins += 1
            row["baseline_wins"] += 1
    decisive = candidate_wins + baseline_wins
    lo, hi = _wilson(candidate_wins, decisive)
    preference = candidate_wins / decisive if decisive else None
    return {"protocol_id": key.get("protocol_id"), "valid_votes": decisive + ties, "invalid_votes": invalid,
            "decisive_votes": decisive, "candidate_wins": candidate_wins, "baseline_wins": baseline_wins,
            "ties": ties, "candidate_preference": round(preference, 4) if preference is not None else None,
            "ci95": [round(lo, 4), round(hi, 4)],
            "passes_70_percent_target": bool(preference is not None and preference >= 0.70),
            "by_pair": by_pair}
