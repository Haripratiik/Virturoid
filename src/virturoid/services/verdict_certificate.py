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
                      memory_dir: str = "build/memory", body_parity: dict | None = None) -> dict:
    """Format an already-measured physics ``verdict`` (the output of ``verify_robot``/``_honest_walk``) into a
    signed verification certificate. Pure + deterministic given the verdict.

    ``body_parity`` is a :func:`grounded_physics.fingerprint_delta` between the body that was MEASURED and the
    body the package SHIPS. The ``deploy==measure`` line used to be printed unconditionally, and it was simply
    false whenever the two differed -- measured, an exported Go2 certificate quoted forward_m -0.188 / cadence
    4.4 / support 0.62 from a fresh 800-step rollout while the customer's own ``verify_robot`` had reported
    -0.692 / 3.3 / 0.74, on a body 8 kg lighter. Pass the comparison and the claim tells the truth; omit it
    (the in-session case, where the verdict and the gene are the same object) and it stands as before.

    ``deploy_is_measure`` is TRI-STATE. True = measured, on this body. False = measured, on a different body.
    **None = never measured** — the case a two-valued flag had no way to express and therefore got wrong: a
    verdict that reports ``could not simulate`` carries no checks, and both the flag and ``body_parity.same``
    used to read True on it. ``rollout_ran`` says the same thing without needing the tri-state read."""
    from virturoid.services.task_matched_eval import robot_kind
    v = verdict or {}
    verdict_str = str(v.get("verdict", "unverified"))
    credible = bool(verdict_str.upper().startswith(("CREDIBLE", "WALKS", "DRIVES", "REACHES", "FLIES", "SWIMS", "CRAWLS"))
                    or v.get("credible") or v.get("credible_walk"))
    checks = {k: v[k] for k in _CHECK_KEYS if k in v and v[k] is not None}
    # DID A ROLLOUT ACTUALLY HAPPEN? ``deploy_is_measure`` and ``body_parity.same`` are both claims about a
    # MEASUREMENT -- "the numbers above came from the body this package ships". Measured on Menagerie's flybody,
    # whose twin does not compile: ``verify_robot`` returned ``verdict: "could not simulate (ValueError)"`` with
    # no checks at all, and this function printed ``deploy_is_measure: true``, ``body_parity.same: true`` and
    # the sentence "the verdict is signed by the SAME rollout that deploys" -- three assertions about a rollout
    # that never ran. An empty ``checks`` is the direct evidence: there are no numbers for the claim to be about.
    # The answer is tri-state, because "no measurement" is not "the body changed" and must not read as either.
    measured = bool(checks)
    _same_body = body_parity is None or bool(body_parity.get("same"))
    _verified_with = ("MuJoCo physics; un-gameable verdict (signed forward + upright + cadence). The verdict "
                      "is signed by the SAME rollout that deploys (deploy==measure) — a slide/lurch cannot "
                      "pass as a walk.")
    if not measured:
        _why_unmeasured = (f"the verdict is {verdict_str!r}"
                           + (f" ({str(v.get('error'))[:160]})" if v.get("error") else "")
                           + " and carries no measured checks")
        # NOTE the phrasing: the literal token "deploy==measure" is the CLAIM, and callers (including this
        # repo's own tests) check for it by substring. It must not appear even inside a sentence denying it.
        _verified_with = (
            f"NOTHING WAS MEASURED. No rollout produced a number on this body: {_why_unmeasured}. This "
            "certificate therefore makes no claim that the deployed body is the measured one — there is no "
            "measured one. It is not evidence that the robot works, does not work, or was run at all.")
        parity_out = {"same": None, "reason": _why_unmeasured + ", so there is no measured body for the "
                                              "shipped body to be the same as",
                      "held_vs_shipped": body_parity}
    elif not _same_body:
        _verified_with = (
            "MuJoCo physics; un-gameable verdict (signed forward + upright + cadence) — a slide/lurch cannot "
            "pass as a walk. DEPLOY != MEASURE: the numbers above were measured on a body that is NOT the one "
            f"in this package ({body_parity.get('n_links_changed')} link(s) differ; total mass "
            f"{body_parity.get('total_mass_kg')} kg, delta {body_parity.get('delta_mass_kg')} kg). Re-verify "
            "the shipped body before relying on these numbers.")
        parity_out = body_parity
    else:
        parity_out = body_parity
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
        "verified_with": _verified_with,
        # tri-state: True = measured on this body; False = measured on a different body; None = never measured.
        "deploy_is_measure": (_same_body if measured else None),
        "rollout_ran": measured,
        "body_parity": parity_out,
        "flywheel_provenance": _flywheel_provenance(gene, memory_dir),
        "disclaimer": ("Physics-verified in simulation, not on hardware. Sim-to-real bring-up (actuator/friction "
                       "identification, safety) is the integrator's responsibility; this certifies the design was "
                       "verified in-sim before export, not that it is deployed."),
    }
