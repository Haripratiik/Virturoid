"""Verdict certificate — the moat's honesty, travelling WITH the export.

Isaac/MuJoCo hand you a physics engine; they don't certify that THIS design's control was verified (and not
gamed) before you build it. We do. Every export can carry a machine-readable certificate: the un-gameable
physics verdict (signed forward + upright + cadence), the honest checks, and the flywheel provenance (which
vector-nearest prior robot seeded it). The positioning writes itself — the robot "arrives in Isaac already
verified" — and it makes the verification gate visible at the exact moment a user touches NVIDIA's stack. Pure
formatting over a verdict the caller already measured (so it is testable without a rollout); the flywheel
provenance is best-effort.

**deploy==measure is EARNED, not asserted.** It is a claim about a ROLLOUT, and a rollout is a body AND a
controller. Both halves are checked here — ``body_parity`` (the measured body vs the shipped one) and
``controller_parity`` (the control law the verdict was measured under vs the one
``software/control_program.json`` deploys) — and the headline sentence is only printed when both hold. Measured
on a real Menagerie Unitree Go2 before the second half existed: a certificate signing a ``default_crawl``
rollout at +0.119 m, in a package whose shipped controller was a ``trot_cpg_gait`` measured separately, both
correct, printing "the verdict is signed by the SAME rollout that deploys" over the disagreement.
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


#: ``gait_source`` prefixes that name a CRAWL-WAVE rollout. ``verify_robot`` reports the door a controller came
#: through, not the control law it implements, so the mapping from source to law lives here — it is what lets the
#: certificate say whether the rollout it signs and the program the package deploys are the same kind of machine.
_CRAWL_SOURCES = ("default_crawl", "tuned_for_this_body", "flywheel_hint")


def _control_law_of(gait_source: str) -> str | None:
    """The CONTROL LAW behind a ``gait_source`` label, or None when it names no scripted law."""
    s = str(gait_source or "")
    if s.startswith(_CRAWL_SOURCES):
        return "crawl_wave_gait"
    if s == "learned_policy":
        return "learned_policy"
    if s == "serpentine":
        return "serpentine"
    if s.startswith("biped_"):
        return "biped_stand"
    return None


#: the crawl operating point, as the shipped program spells it vs as the deploy path spells it
_OP_KEYS = (("frequency_hz", "freq"), ("hip_amp", "hip_amp"), ("knee_amp", "knee_amp"), ("kp", "kp"), ("kd", "kd"))


def _measured_operating_point(gene, gait_source: str) -> dict | None:
    """The crawl parameters the SIGNED rollout actually ran, recovered from the same places ``_honest_gait`` reads
    them: this body's own tuned cache for a ``tuned_for_this_body*`` verdict, the shipped default for
    ``default_crawl``. ``None`` when the source names no recoverable point (a mined ``flywheel_hint``, a learned
    policy, a non-crawl law) — and None must read as UNKNOWN, never as agreement."""
    s = str(gait_source or "")
    try:
        if s.startswith("tuned_for_this_body"):
            own = (getattr(gene, "metadata", None) or {}).get("gait_params") or {}
            got = {k: float(own[k]) for _, k in _OP_KEYS if k in own}
            return got or None
        if s == "default_crawl":
            from virturoid.services.gait_flywheel import _DEFAULT_GAIT
            return {k: float(_DEFAULT_GAIT[k]) for _, k in _OP_KEYS if k in _DEFAULT_GAIT} or None
    except Exception:  # noqa: BLE001 - an unrecoverable op-point is UNKNOWN, which is the safe answer
        return None
    return None


def _controller_parity(gene, verdict: dict, shipped: dict | None) -> dict:
    """Did the rollout this certificate signs use the controller the package DEPLOYS?

    ``body_parity`` answers the same question about the BODY and has since an exported Go2 certificate was caught
    quoting a rollout from a body 8 kg lighter. The controller half went unasked, and on a real Menagerie Go2 the
    answer was no: the certificate signed a ``default_crawl`` rollout (+0.119 m, "NO LOCOMOTION VERDICT") while
    ``software/control_program.json`` shipped a ``trot_cpg_gait`` (a different control law, a different PD
    attractor, a different rate) — and the certificate printed "the verdict is signed by the SAME rollout that
    deploys (deploy==measure)" over the top of it. Two correct measurements of two different machines.

    TWO THINGS ARE CHECKED, because matching the control law is not enough. ``extract_crawl_gait_params`` runs its
    OWN ``tune_crawl_gait`` search at export time, independent of the operating point the deploy path reads off
    ``metadata['gait_params']`` — measured on a generated quadruped, the held body deploys freq 2.056 / hip 0.746
    / knee 0.770 while the package ships freq 2.55 / hip 0.9 / knee 1.0. Same law, different machine. So
    ``control_law_match`` and ``operating_point_match`` are reported separately and ``same`` is their conjunction.

    Tri-state throughout: True = the signed rollout ran the shipped controller; False = it demonstrably ran a
    different one; **None = we cannot tell** (no controller stamp, or an operating point that is not recoverable
    — e.g. a mined ``flywheel_hint``).
    """
    measured_src = str((verdict or {}).get("gait_source") or "")
    measured_law = _control_law_of(measured_src)
    measured_op = _measured_operating_point(gene, measured_src)
    if not shipped:
        return {"same": None, "control_law_match": None, "operating_point_match": None,
                "measured_controller": {"gait_source": measured_src or None, "control_law": measured_law,
                                        "operating_point": measured_op},
                "shipped_controller": None,
                "reason": ("this package records no deployable control program for this body, so there is nothing "
                           "for the signed rollout to be compared against. The verdict above describes the "
                           "controller named in measured_controller and nothing else.")}
    shipped_law = str(shipped.get("policy_type") or "") or None
    law_match = bool(measured_law and shipped_law and measured_law == shipped_law)
    shipped_op = {k: shipped[p] for p, k in _OP_KEYS if shipped.get(p) is not None} or None
    op_match: bool | None
    if not law_match:
        op_match = None                      # a different law makes the parameter comparison meaningless
    elif measured_op is None or shipped_op is None:
        op_match = None
    else:
        keys = set(measured_op) & set(shipped_op)
        op_match = bool(keys) and all(
            abs(float(measured_op[k]) - float(shipped_op[k])) <= 1e-6 + 1e-3 * abs(float(shipped_op[k]))
            for k in keys)
    same = True if (law_match and op_match is True) else (None if (law_match and op_match is None) else False)
    out = {
        "same": same,
        "control_law_match": law_match,
        "operating_point_match": op_match,
        "measured_controller": {"gait_source": measured_src or None, "control_law": measured_law,
                                "operating_point": measured_op,
                                "forward_m": (verdict or {}).get("forward_m")},
        "shipped_controller": {"control_law": shipped_law,
                               "entrypoint": shipped.get("entrypoint"),
                               "parameters_file": shipped.get("parameters_file"),
                               "program_fingerprint": shipped.get("program_fingerprint"),
                               "control_frequency_hz": shipped.get("control_frequency_hz"),
                               "pd_gains": shipped.get("pd_gains"),
                               "operating_point": shipped_op,
                               "verified_walk": shipped.get("verified_walk"),
                               "forward_m": shipped.get("sim_forward_m"),
                               "verdict": shipped.get("sim_verdict"),
                               "measured_by": shipped.get("measured_by")},
        "cross_check": ("compare program_fingerprint with the same field in software/control_program.json: equal "
                        "means these two files describe the same controller"),
    }
    _both = (f"the signed rollout travelled {(verdict or {}).get('forward_m')} m and the deployed program "
             f"travelled {shipped.get('sim_forward_m')} m ({shipped.get('sim_verdict')})")
    if same is True:
        out["reason"] = (f"the signed rollout and the deployed program run the same {shipped_law!r} control law at "
                         f"the same operating point. They remain two SEPARATE rollouts at their own budgets and "
                         f"rates, so their distances need not be equal — {_both}; each is labelled with the run "
                         "that produced it.")
    elif same is None:
        out["reason"] = (f"both run the {shipped_law!r} control law, but the operating point the verdict was "
                         f"measured at cannot be recovered from gait_source {measured_src!r}, so this certificate "
                         f"does NOT claim they are the same controller. {_both}.")
    elif not law_match:
        out["reason"] = (
            f"the verdict above was measured under {measured_law or measured_src or 'an unnamed controller'} and "
            f"this package deploys {shipped_law!r} ({shipped.get('parameters_file')}). They are different "
            f"controllers, so the distances differ by construction: {_both}. Neither number is wrong; they are "
            "about different machines.")
    else:
        out["reason"] = (
            f"same {shipped_law!r} control law, DIFFERENT OPERATING POINT: the verdict was measured at "
            f"{measured_op} and the package deploys {shipped_op} (the export runs its own gait search, "
            f"independent of the point the deploy path reads). {_both}.")
    return out


def _deploy_is_measure(measured: bool, same_body: bool, same_controller: bool | None) -> bool | None:
    """The conjunction of the two halves, as a tri-state. ``False`` beats ``None`` (a known mismatch is a finding,
    not an unknown); ``None`` beats ``True`` (an unchecked half cannot be asserted)."""
    if not measured:
        return None
    if not same_body or same_controller is False:
        return False
    return True if same_controller is True else None


def build_certificate(gene, verdict: dict, *, task: str = "", robot_id: str | None = None,
                      memory_dir: str = "build/memory", body_parity: dict | None = None,
                      shipped_controller: dict | None = None) -> dict:
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
    used to read True on it. ``rollout_ran`` says the same thing without needing the tri-state read.

    ``shipped_controller`` closes the OTHER half of the same claim, and it is the half that was still false after
    ``body_parity`` landed. "The verdict is signed by the SAME rollout that deploys" is a statement about a
    ROLLOUT, and a rollout is a body AND a controller; only the body was ever checked. Measured on a real
    Menagerie Unitree Go2 with the current code: the certificate signed a ``default_crawl`` rollout at +0.119 m
    ("NO LOCOMOTION VERDICT") on a body that matched exactly, and the same package's
    ``software/control_program.json`` shipped a ``trot_cpg_gait`` — a different control law, from a different PD
    attractor, at a different rate — while the certificate printed ``deploy_is_measure: true`` and the
    deploy==measure sentence over the top. The claim now requires both halves; if the shipped controller is
    unknown or different, the sentence is replaced rather than qualified, because a reader quoting one line out
    of this file must not be able to quote a false one. Pass the stamp
    ``gene_build.exported_controller_stamp(gene)`` (the export writer records it after it has MEASURED the file it
    writes); omit it and ``controller_parity.same`` reads None — unknown, never assumed true.
    """
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
    ctl_parity = _controller_parity(gene, v, shipped_controller)
    _same_ctl = ctl_parity.get("same")                 # True / False / None (unknown — no controller stamp)
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
        # THE BODY MATCHES. Does the CONTROLLER? The deploy==measure sentence claims a rollout, not a mesh.
        _preamble = ("MuJoCo physics; un-gameable verdict (signed forward + upright + cadence) — a slide/lurch "
                     "cannot pass as a walk, and it was measured on the body this package ships. ")
        if _same_ctl is False:
            _verified_with = (_preamble + "THE CONTROLLER IS NOT THE ONE THIS PACKAGE DEPLOYS: "
                              + ctl_parity["reason"]
                              + " See controller_parity for both runs; do not read either distance as the other "
                                "controller's.")
        elif ctl_parity["shipped_controller"] is None:
            _law = ctl_parity["measured_controller"]["control_law"] or "an unnamed controller"
            _verified_with = (_preamble + "NO DEPLOYABLE CONTROL PROGRAM IS RECORDED for this body, so there is "
                              "no deployed controller for the signed rollout to be the same as. This certificate "
                              f"claims only what it measured: {_law} (gait_source {v.get('gait_source')!r}). See "
                              "controller_parity.")
        elif _same_ctl is None:
            _verified_with = (_preamble + "WHETHER IT IS THE DEPLOYED CONTROLLER CANNOT BE ESTABLISHED: "
                              + ctl_parity["reason"] + " See controller_parity.")
        else:
            _verified_with = (
                "MuJoCo physics; un-gameable verdict (signed forward + upright + cadence) — a slide/lurch cannot "
                "pass as a walk. deploy==measure holds on BOTH halves: the same body AND the same controller "
                f"({ctl_parity['shipped_controller']['control_law']} at the same operating point) this package "
                "deploys. The two runs are still separate rollouts at their own budgets; controller_parity "
                "carries each with its own distance.")
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
        # THE WHOLE CLAIM, both halves, still tri-state. True = a rollout ran, on the shipped body, under the
        # shipped control law. False = a rollout ran and one of those two is wrong. None = nothing was measured,
        # OR the body matches but WHICH CONTROLLER DEPLOYS IS NOT RECORDED — because "we did not check" is not
        # "it matches", and reading True off an unchecked half is exactly how this field came to be wrong.
        "deploy_is_measure": _deploy_is_measure(measured, _same_body, _same_ctl),
        "deploy_is_measure_parts": {"same_body": (_same_body if measured else None),
                                    "same_controller": (_same_ctl if measured else None)},
        "rollout_ran": measured,
        "body_parity": parity_out,
        "controller_parity": ctl_parity,
        "flywheel_provenance": _flywheel_provenance(gene, memory_dir),
        "disclaimer": ("Physics-verified in simulation, not on hardware. Sim-to-real bring-up (actuator/friction "
                       "identification, safety) is the integrator's responsibility; this certifies the design was "
                       "verified in-sim before export, not that it is deployed."),
    }
