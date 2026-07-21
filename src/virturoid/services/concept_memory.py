"""Open-world concept memory for robot requests.

``robot_class`` in older Virturoid services is an execution family (arm, legged
body, mobile base, ...), not a closed catalogue of what people may ask for.  This
module preserves a user's arbitrary concept separately, records it before a build,
and promotes it only after the existing physics-backed package path produces
evidence.  The lifecycle is deliberately small and auditable:

    candidate -> evaluated -> verified

Candidate concepts never change routing.  Only verified concepts can provide a
recalled execution route for a later underspecified request.  That prevents both
silent class collapse and unsafe self-training from a single failed rollout.
"""

from __future__ import annotations

from pathlib import Path


def observe_request(memory_dir: Path, concept: str, prompt: str, *,
                    aliases: list[str] | None = None) -> dict | None:
    """Persist a new or repeated concept and its model-proposed exact aliases."""
    if not (concept or "").strip():
        return None
    from virturoid.services.memory_db import MemoryDB

    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        return db.observe_concept(concept, prompt, aliases=aliases)


def recall_verified_route(memory_dir: Path, concept: str, *,
                          aliases: list[str] | None = None) -> dict | None:
    """Return a verified route for this label or a model-proposed exact alias.

    The LLM supplies semantic equivalences.  Storage only checks their normalized
    text exactly; it never does fuzzy matching or invents an alias, so a new term
    cannot silently inherit an unrelated execution route.
    """
    if not (concept or "").strip():
        return None
    from virturoid.services.memory_db import MemoryDB

    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        record = db.concept_for_alias(concept)
        if record is None:
            for alias in aliases or []:
                record = db.concept_for_alias(alias)
                if record is not None:
                    break
    if record and record.get("state") == "verified" and record.get("execution_family"):
        return record
    return None


def promote_after_evaluation(memory_dir: Path, concept: str, *, execution_family: str,
                             task_type: str, species_pattern: str | None,
                             success_rate: float, target_success_rate: float,
                             aliases: list[str] | None = None) -> dict | None:
    """Record one package/evaluation outcome and promote only on target attainment."""
    if not (concept or "").strip():
        return None
    from virturoid.services.memory_db import MemoryDB

    with MemoryDB(Path(memory_dir) / "virturoid_memory.db") as db:
        return db.promote_concept(
            concept,
            execution_family=execution_family,
            task_type=task_type,
            species_pattern=species_pattern,
            success_rate=success_rate,
            target_success_rate=target_success_rate,
            aliases=aliases,
        )
