"""Evidence-bundle retrieval + universal transfer screening (dossier R4 + R3).

R4 — turn skill recall into AUDIT-READY evidence: :func:`retrieve_evidence` reads banked skills from the
``MemoryDB`` and returns ranked :class:`TrainingEvidenceBundle`s carrying held-out success, valid region,
failure clusters, reward provenance, and IO compatibility — so the training compiler ranks and rejects
candidates on measured evidence, not nearest-neighbour vibes.

R3 — generalize transfer screening beyond locomotion: :func:`screen_transfer` takes candidates + an injected
cheap zero-shot ``screen_fn`` (grasp/push/reach/locomotion all plug in), ranks by measured transfer, and
REJECTS negative transfer before any expensive training. Deterministic, backend-free (the screen_fn owns physics).
"""

from __future__ import annotations

import json
from typing import Callable

from virturoid.schemas.training_evidence import TrainingEvidenceBundle


def _loads(value) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}


def _failure_clusters(conn, robot_class: str, task_type: str) -> list[str]:
    try:
        rows = conn.execute(
            "SELECT label, SUM(count) AS n FROM failures WHERE robot_class=? AND task_type=? "
            "GROUP BY label ORDER BY n DESC", (robot_class, task_type)).fetchall()
        return [r["label"] for r in rows]
    except Exception:  # noqa: BLE001 - failures table optional / schema drift
        return []


def _bundle_from_row(row, *, obs_dim, act_dim, conn, match_kind: str) -> TrainingEvidenceBundle:
    io_compatible = (obs_dim is not None and act_dim is not None
                     and row["obs_dim"] == obs_dim and row["act_dim"] == act_dim)
    heldout = float(row["success_rate"]) if row["success_rate"] is not None else None
    # evidence score: held-out success is the spine; IO-compatible and exact matches rank higher, cross-class lower.
    kind_weight = {"exact_io": 1.25, "class_task": 1.0, "same_class": 0.8, "cross_class": 0.6}[match_kind]
    score = round((heldout or 0.0) * kind_weight * (1.15 if io_compatible else 1.0), 4)
    return TrainingEvidenceBundle(
        id=f"evidence_{row['skill_id']}",
        candidate_skill_id=row["skill_id"],
        robot_class=row["robot_class"],
        task_type=row["task_type"],
        heldout_success=heldout,
        valid_region=_loads(row["region"]),
        failure_clusters=_failure_clusters(conn, row["robot_class"], row["task_type"]),
        reward_spec_ref=_loads(row["reward_spec"]) or None,
        io_compatible=bool(io_compatible),
        params_path=row["params_path"],
        match_kind=match_kind,
        provenance_edge={"species": row["species"], "gene_id": row["gene_id"]},
        evidence_score=score,
    )


def retrieve_evidence(db, robot_class: str, task_type: str, *, species: str | None = None,
                      obs_dim: int | None = None, act_dim: int | None = None,
                      k: int = 5, cross_class: bool = True) -> list[TrainingEvidenceBundle]:
    """Return ranked :class:`TrainingEvidenceBundle`s for a (class, task), best evidence first.

    Widens from exact class+task to same-class-any-task to (optionally) cross-class, tagging each match kind, so a
    caller sees not just "the nearest skill" but WHY it might transfer and what failures/region it carries.
    """
    conn = db.conn
    seen: set[str] = set()
    bundles: list[TrainingEvidenceBundle] = []

    def _collect(rows, match_kind):
        for row in rows:
            if row["skill_id"] in seen:
                continue
            seen.add(row["skill_id"])
            bundles.append(_bundle_from_row(row, obs_dim=obs_dim, act_dim=act_dim, conn=conn,
                                            match_kind=match_kind))

    exact = conn.execute(
        "SELECT * FROM skills WHERE robot_class=? AND task_type=? ORDER BY success_rate DESC",
        (robot_class, task_type)).fetchall()
    # tag IO-exact matches specially so directly-reusable policies rank at the top.
    for row in exact:
        if row["skill_id"] in seen:
            continue
        seen.add(row["skill_id"])
        kind = ("exact_io" if (obs_dim is not None and row["obs_dim"] == obs_dim
                               and act_dim is not None and row["act_dim"] == act_dim) else "class_task")
        bundles.append(_bundle_from_row(row, obs_dim=obs_dim, act_dim=act_dim, conn=conn, match_kind=kind))

    _collect(conn.execute("SELECT * FROM skills WHERE robot_class=? AND task_type!=? ORDER BY success_rate DESC",
                          (robot_class, task_type)).fetchall(), "same_class")
    if cross_class:
        _collect(conn.execute("SELECT * FROM skills WHERE robot_class!=? AND task_type=? ORDER BY success_rate DESC",
                              (robot_class, task_type)).fetchall(), "cross_class")

    bundles.sort(key=lambda b: b.evidence_score, reverse=True)
    return bundles[:k]


def screen_transfer(candidates: list, screen_fn: Callable[[object], float], *,
                    reject_below: float = 0.0, labeler: Callable[[object], str] | None = None) -> dict:
    """Universal zero-shot transfer screening (generalizes locomotion transfer_seed to any task).

    Runs the cheap ``screen_fn`` on each candidate (a zero-shot rollout score: forward metres, grasp success,
    push distance, ...), ranks descending, and REJECTS candidates whose measured transfer is ``<= reject_below``
    (negative transfer) BEFORE any expensive training. The screen_fn owns the physics; this is task-agnostic.
    Returns ``{ranked, survivors, rejected, best}``.
    """
    label = labeler or (lambda c: getattr(c, "id", None) or str(c))
    scored = []
    for cand in candidates:
        try:
            score = float(screen_fn(cand))
        except Exception as exc:  # noqa: BLE001 - a candidate that errors is rejected, not fatal
            scored.append({"candidate": label(cand), "score": None, "error": str(exc), "accepted": False})
            continue
        scored.append({"candidate": label(cand), "score": round(score, 5),
                       "accepted": score > reject_below})
    scored.sort(key=lambda s: (s["score"] is not None, s["score"] if s["score"] is not None else -1e9),
                reverse=True)
    survivors = [s for s in scored if s["accepted"]]
    rejected = [s for s in scored if not s["accepted"]]
    return {
        "ranked": scored,
        "survivors": survivors,
        "rejected": rejected,
        "best": survivors[0] if survivors else None,
        "negative_transfer_rejected": len(rejected),
    }
