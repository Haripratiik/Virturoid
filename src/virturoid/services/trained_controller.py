"""WHERE A TRAINED CONTROLLER LANDS — the write side every training door was missing.

Until 2026-08-08 the training tools were a read-only loop. ``reward_loop._train_reward``,
``gait_hints._adapt_gait`` and ``input_training_tools._learn_gait`` all did ``session_state.get_robot(rid)``,
searched a controller for that body, returned the numbers — and NEVER called ``put_robot``/``commit_robot``.
Every ``put_robot`` caller in the repo was a design/ingest/edit path. So the held gene was byte-identical before
and after training, and ``verify_robot`` afterwards re-measured the UNTRAINED controller and said so in its own
provenance field. An engineer who trained a real Menagerie Go2 twice, by two different tools, read
``gait_source: flywheel_hint`` both times — the product telling the truth about a result it had discarded.

**THE CONTRACT, chosen deliberately.** Training APPLIES to the held robot by default, and returns the artifact
either way. The alternative — return an artifact the engineer explicitly applies — is defensible and is what
shipped, but what shipped had no apply verb at all, which is how "returns an artifact" decayed into "discards
the result". So:

  * ``apply="auto"`` (the default) commits the trained controller **when the training run's own un-gameable
    verdict says it is a credible walk**, and refuses otherwise WITH the verdict as the reason. Training that
    did not produce a walk does not get to overwrite a controller that might.
  * ``apply="never"`` is the artifact-only contract: nothing is mutated, and ``params`` comes back so
    ``apply_gait`` can land it later. That is the dry run.
  * ``apply="always"`` lands it regardless — for the engineer who wants to look at a non-credible operating
    point in the viewer. It is never silent: the report says the verdict it overrode.

**REVERSIBLE, because it is an edit.** The commit goes through ``session_state.commit_robot``, which pushes the
previous gene onto the same undo stack ``edit_robot {ops:[{op:'undo'}]}`` pops. Training is now an edit to the
robot, with an edit's guarantees, rather than a side channel.

**WHY ``metadata['gait_params']``.** That is the slot ``ai_native_tools._honest_gait`` reads FIRST when it picks
a controller to deploy for the verdict — ahead of the mined flywheel hint and ahead of the shipped default. So
landing here is not a bookkeeping entry: it is what the next ``verify_robot`` actually runs. ``gait_provenance``
rides alongside it, and verify reports it, so ``gait_source`` names the door that produced the controller
(``tuned_for_this_body::train_reward``) instead of leaving "fitted to this body" to stand for four different
origins.

**WHAT THE SAFETY NET DOES AND DOES NOT COVER** — this paragraph used to claim "landing a trained gait can
never make a robot verify WORSE than it did before", and that is FALSE, measured 2026-08-08 through
``call_tool`` on an authored horse:

    verify (before)             3.305 m   tuned_for_this_body            CREDIBLE WALK
    train_held apply="auto"     solved, beats_default (0.868 vs 0.471 at the search's 600-step deploy horizon)
    verify (after)              2.107 m   tuned_for_this_body::train_held  FELL by YAW-DRIFT
    edit_robot op:"undo"        3.305 m   tuned_for_this_body            CREDIBLE WALK

Verify's deploy-select re-runs the SHIPPED DEFAULT alongside the landed gait and keeps whichever is credible —
so a landing can never lose to the default. It does not re-run whatever the body was carrying BEFORE, and every
door's own gate is judged at a search-length horizon (600 steps) while the verdict is judged at the settling
horizon (6000): a point can genuinely beat the default over 600 steps and fall at 2000. So a robot that already
carried a fitted controller CAN come out behind, and the two protections that make that recoverable are the
ones this module provides — the report names the ``previous_params`` it replaced, and ``undo`` restores them.
"""
from __future__ import annotations

import math
import time

#: The five scalars the CPG controller reads. Same tuple as ``gait_search.PARAM_NAMES``; a controller is all
#: five or it is not a controller (a partial dict silently inherits the rest from whatever the body carried).
GAIT_KEYS: tuple[str, ...] = ("freq", "hip_amp", "knee_amp", "kp", "kd")

#: The apply contract, as an agent-facing vocabulary. See the module docstring for what each one promises.
APPLY_MODES: tuple[str, ...] = ("auto", "always", "never")

_UNDO_HINT = "edit_robot {robot_id, ops:[{op:'undo'}]} restores the controller this replaced"


def normalize_gait_params(params) -> tuple[dict, str]:
    """``(clean_params, "")`` or ``({}, why_not)``.

    Rejects rather than repairs. Every value must be finite and inside the interval ``gait_search`` samples that
    parameter from — searched results are ``_clip``-ed into that box by construction (``gait_search.py:96``), so
    a value outside it did not come from a search and is not an operating point any search could reproduce.
    """
    if not isinstance(params, dict):
        return {}, (f"gait params must be an object carrying {', '.join(GAIT_KEYS)}; "
                    f"got {type(params).__name__}")
    missing = [k for k in GAIT_KEYS if k not in params]
    if missing:
        return {}, (f"missing gait parameter(s) {', '.join(missing)} — a controller is all five of "
                    f"{', '.join(GAIT_KEYS)}; a partial dict silently inherits the rest from the old gait")
    try:
        from virturoid.services.gait_search import _HI, _LO
    except Exception:  # noqa: BLE001 - no search module -> range cannot be checked; take the values as given
        _LO, _HI = {}, {}
    out: dict = {}
    for k in GAIT_KEYS:
        try:
            v = float(params[k])
        except (TypeError, ValueError):
            return {}, f"gait parameter {k}={params[k]!r} is not a number"
        if not math.isfinite(v):
            return {}, f"gait parameter {k} is not finite ({v!r})"
        if k in _LO and not (_LO[k] <= v <= _HI[k]):
            return {}, (f"gait parameter {k}={v:g} is outside the searchable range "
                        f"[{_LO[k]:g}, {_HI[k]:g}] — no gait search can reach it, so it is not an "
                        f"operating point this platform can reproduce or bank")
        out[k] = v
    return out, ""


def held_gait(robot_id: str) -> dict:
    """The controller the held robot currently deploys: ``{params, provenance, source}``. Empty params mean it
    deploys whatever verify picks for it (a mined flywheel hint, or the shipped default)."""
    from virturoid.services import session_state as S
    gene = S.get_robot(str(robot_id or ""))
    if gene is None:
        return {"params": {}, "provenance": None, "source": None}
    md = getattr(gene, "metadata", None) or {}
    own = md.get("gait_params") or {}
    clean = {k: float(own[k]) for k in GAIT_KEYS if isinstance(own.get(k), (int, float))}
    return {"params": clean, "provenance": md.get("gait_provenance"),
            "source": ("tuned_for_this_body" if clean else None)}


def apply_trained_gait(robot_id: str, params, *, door: str, apply: str = "auto", credible: bool | None = None,
                       verdict: str = "", evidence: dict | None = None) -> dict:
    """Land a trained controller on the held robot. Never raises; returns what happened and why.

    ``door`` names the tool that produced it (``train_reward`` / ``learn_gait`` / ``adapt_gait`` /
    ``apply_gait``) and becomes the suffix on the ``gait_source`` the next verdict reports. ``credible`` is the
    training run's OWN un-gameable verdict — the gate ``apply="auto"`` consults; ``None`` means the door could
    not judge it, which is treated as not-credible (a door that cannot say whether it produced a walk does not
    get to overwrite a controller silently).
    """
    from virturoid.services import session_state as S
    mode = str(apply or "auto").lower()
    rid = str(robot_id or "")
    base = {"applied": False, "robot_id": rid, "door": str(door), "apply_mode": mode,
            "params": None, "previous_params": None, "undo": None}
    if mode not in APPLY_MODES:
        return {**base, "reason": f"apply must be one of {', '.join(APPLY_MODES)} (got {mode!r})",
                "error": f"apply must be one of {', '.join(APPLY_MODES)} (got {mode!r})"}
    clean, why = normalize_gait_params(params)
    if why:
        return {**base, "reason": f"nothing to apply: {why}"}
    base["params"] = clean
    prev = held_gait(rid)
    base["previous_params"] = prev["params"] or None
    if mode == "never":
        return {**base, "reason": "apply='never' — the trained controller is returned as an artifact and the "
                                  "held robot is untouched; land it later with apply_gait {robot_id, params}",
                "apply_with": {"tool": "apply_gait", "args": {"robot_id": rid, "params": clean}}}
    if mode == "auto" and not bool(credible):
        return {**base, "reason": (f"apply='auto' and this run did not produce a credible walk"
                                   f"{f' ({verdict})' if verdict else ''}, so the robot keeps the controller it "
                                   f"had. Re-run with more budget, or pass apply='always' to land it anyway and "
                                   f"look at it — the report will say it overrode a non-credible verdict."),
                "apply_with": {"tool": "apply_gait", "args": {"robot_id": rid, "params": clean}}}
    gene = S.get_robot(rid)
    if gene is None:
        return {**base, "reason": f"no held robot '{rid}' to apply the trained controller to"}
    try:
        from virturoid.schemas.gene import RobotGene
        fresh = RobotGene.from_dict(gene.to_dict())              # a COPY: the undo entry must not alias the new gene
        md = dict(getattr(fresh, "metadata", None) or {})
        md["gait_params"] = dict(clean)
        md["gait_provenance"] = {"door": str(door), "applied_at": round(time.time(), 3),
                                 "credible": bool(credible), "verdict": str(verdict or ""),
                                 **{k: v for k, v in (evidence or {}).items() if v is not None}}
        fresh.metadata = md
        if not S.commit_robot(rid, fresh, label=f"trained::{door}"):
            return {**base, "reason": f"no held robot '{rid}' to apply the trained controller to"}
    except Exception as exc:  # noqa: BLE001 - a landing failure must be reported, never swallowed into "applied"
        return {**base, "reason": f"could not commit the trained controller: {type(exc).__name__}: {exc}",
                "error": f"{type(exc).__name__}: {exc}"}
    src = f"tuned_for_this_body::{door}"
    out = {**base, "applied": True, "undo": _UNDO_HINT,
            "gait_source_after": src,
            "reason": (f"committed to the held robot as its deployed controller; verify_robot now runs THESE "
                       f"parameters and reports gait_source '{src}'")}
    if base["previous_params"]:
        # IT REPLACED SOMETHING THAT WAS ALSO FITTED TO THIS BODY. Verify's deploy-select guards the landing
        # against the shipped DEFAULT, not against the controller the body was already carrying, and the gate
        # that admitted this one was judged over a search-length horizon rather than the settling one — so this
        # particular replacement is the case where a landing can leave the robot verifying worse. Measured on an
        # authored horse: 3.305 m CREDIBLE -> 2.107 m FELL, and undo put it back. Saying so is the difference
        # between an engineer who re-verifies and one who does not.
        out["replaced"] = (f"this robot already carried a controller fitted to it "
                           f"({base['previous_params']}); re-verify, and undo if the new one is worse — "
                           f"verify's deploy-select only guards the landing against the SHIPPED DEFAULT")
    if mode == "always" and credible is False:
        # ``credible is False`` and ``credible is None`` are different facts: one run measured a verdict and it
        # was not a walk, the other never took one (``apply_gait`` runs no physics). Only the first is an
        # override of something, and saying "we overrode a non-credible verdict" about a measurement nobody made
        # is the same class of overclaim as the verdict this whole change exists to fix.
        out["override"] = (f"apply='always' landed an operating point whose own verdict was "
                           f"{verdict or 'not credible'} — verify's deploy-select will still prefer the shipped "
                           f"default if this one walks less well, and will then report 'default_crawl'")
    return out
