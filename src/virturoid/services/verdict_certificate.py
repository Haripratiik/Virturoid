"""Verdict certificate — the moat's honesty, travelling WITH the export.

Isaac/MuJoCo hand you a physics engine; they don't certify that THIS design's control was verified (and not
gamed) before you build it. We do. Every export can carry a machine-readable certificate: the un-gameable
physics verdict (signed forward + upright + cadence; the SAME rollout that deploys — deploy==measure), the
honest checks, and the flywheel provenance (which vector-nearest prior robot seeded it). The positioning writes
itself — the robot "arrives in Isaac already verified" — and it makes the verification gate visible at the exact
moment a user touches NVIDIA's stack. Pure formatting over a verdict the caller already measured (so it is
testable without a rollout); the flywheel provenance is best-effort.
"""
from __future__ import annotations

_CHECK_KEYS = ("survived", "forward_m", "speed_mps", "cadence", "support_frac", "height_ratio",
               "roll_max_deg", "pitch_max_deg", "distance_m", "reach_m", "planar_m")


def _flywheel_provenance(gene, memory_dir: str) -> dict | None:
    """Best-effort: the vector-nearest banked body this design could have borrowed control from (similarity +
    class) — the retrieval provenance that makes the flywheel visible on the artifact. None if nothing banked."""
    try:
        from pathlib import Path

        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        db_path = Path(memory_dir) / "virturoid_memory.db"
        if not db_path.exists():
            return None
        with MemoryDB(db_path) as db:
            vm = RoboticsVectorMemory(db)
            if vm.count("body") == 0:
                vm.index_species_bodies()
            hits = vm.nearest_bodies(gene, k=1, min_sim=0.0)
            if not hits:
                return None
            h = hits[0]
            return {"nearest_prior_body": h.get("obj_id"), "similarity": round(float(h.get("similarity", 0)), 3),
                    "meta": {k: (h.get("meta") or {}).get(k) for k in ("robot_class", "species") if (h.get("meta") or {}).get(k)}}
    except Exception:  # noqa: BLE001 - provenance is a nice-to-have on the certificate, never a blocker
        return None


def build_certificate(gene, verdict: dict, *, task: str = "", robot_id: str | None = None,
                      memory_dir: str = "build/memory") -> dict:
    """Format an already-measured physics ``verdict`` (the output of ``verify_robot``/``_honest_walk``) into a
    signed verification certificate. Pure + deterministic given the verdict."""
    from virturoid.services.task_matched_eval import robot_kind
    v = verdict or {}
    verdict_str = str(v.get("verdict", "unverified"))
    credible = bool(verdict_str.upper().startswith(("CREDIBLE", "WALKS", "DRIVES", "REACHES", "FLIES", "SWIMS", "CRAWLS"))
                    or v.get("credible") or v.get("credible_walk"))
    checks = {k: v[k] for k in _CHECK_KEYS if k in v and v[k] is not None}
    return {
        "artifact": "virturoid_verification_certificate",
        "version": 1,
        "robot_id": robot_id,
        "species": getattr(gene, "species", None),
        "robot_class": getattr(gene, "robot_class", None),
        "kind": robot_kind(gene),
        "task": task,
        "verdict": verdict_str,
        "credible": credible,
        "gait_source": v.get("gait_source"),
        "checks": checks,
        "verified_with": ("MuJoCo physics; un-gameable verdict (signed forward + upright + cadence). The verdict "
                          "is signed by the SAME rollout that deploys (deploy==measure) — a slide/lurch cannot "
                          "pass as a walk."),
        "flywheel_provenance": _flywheel_provenance(gene, memory_dir),
        "disclaimer": ("Physics-verified in simulation, not on hardware. Sim-to-real bring-up (actuator/friction "
                       "identification, safety) is the integrator's responsibility; this certifies the design was "
                       "verified in-sim before export, not that it is deployed."),
    }
