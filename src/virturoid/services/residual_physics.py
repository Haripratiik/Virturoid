"""Residual / learned-physics interface (plan Phase F): scoped, honest, deferred.

The architecture research was blunt: a PINN is the WRONG tool for rigid articulated robots —
MuJoCo already solves those exactly, and PINNs are weakest at contact (our core regime). So
this module does NOT add a PINN to the rigid-body loop. It defines the *interface* where a
learned-physics correction legitimately belongs, scoped to the cases the research endorsed:

- **Sim-to-real residual** — a small learned model that corrects MuJoCo's prediction toward
  measured reality (Learned Residual Physics). Default is identity (no correction): with no
  real-world data, MuJoCo is the best estimate, and identity is the honest no-op.
- **Soft / continuum / fluid domains** — where there IS a governing PDE and no cheap exact
  solver, a PINN or (preferably) a neural operator (DeepONet/FNO) implements this interface.
  Those require the new physics domain + training data and are explicitly deferred — the
  placeholders raise rather than pretend.

Rigid-body callers use ``IdentityResidual`` and keep MuJoCo as ground truth.
"""

from __future__ import annotations

from typing import Protocol


class ResidualPhysicsModel(Protocol):
    """Corrects a simulator's predicted next-state toward reality. ``domain`` tags applicability."""
    domain: str

    def correct(self, state, sim_next):
        """Return a corrected next-state given the current state and the simulator's prediction."""


class IdentityResidual:
    """No correction — MuJoCo's prediction is used as-is. The honest default for rigid bodies
    with no real-world residual data (a learned residual is only meaningful once we have
    measured sim-to-real error to train on)."""

    domain = "rigid_body"

    def correct(self, state, sim_next):
        return sim_next


class _DeferredModel:
    """Base for not-yet-implemented learned-physics models: callable interface, honest error."""

    domain = "deferred"
    _why = "not implemented"

    def correct(self, state, sim_next):  # noqa: D401
        raise NotImplementedError(self._why)


class PINNResidual(_DeferredModel):
    """A PINN residual — ONLY for soft/continuum/deformable domains with a governing PDE.

    Deferred: requires the soft-body physics domain + PDE residual + training data. Do not use
    for rigid-body dynamics (use MuJoCo + IdentityResidual)."""

    domain = "soft_body"
    _why = ("PINNResidual is deferred to soft/continuum/deformable domains with a governing PDE "
            "(see docs/ai_architecture_plan.md Phase F). For rigid robots, MuJoCo is the exact "
            "ground truth — use IdentityResidual.")


class NeuralOperatorResidual(_DeferredModel):
    """A neural operator (DeepONet/FNO) — preferred over a PINN for cross-geometry fields
    (fluid/aero/thermal). Deferred until that domain + data exist."""

    domain = "field"
    _why = ("NeuralOperatorResidual is deferred to fluid/aero/thermal field domains "
            "(see docs/ai_architecture_plan.md Phase F).")


def get_residual_model(domain: str = "rigid_body") -> ResidualPhysicsModel:
    """Route to the right residual model for a physics domain.

    rigid_body -> IdentityResidual (MuJoCo is exact). soft/continuum -> PINN (deferred).
    fluid/aero/thermal -> neural operator (deferred). Keeps the rigid loop PINN-free.
    """
    d = (domain or "rigid_body").lower()
    if d in ("rigid_body", "rigid", "articulated"):
        return IdentityResidual()
    if d in ("soft_body", "soft", "continuum", "deformable", "cable"):
        return PINNResidual()
    if d in ("fluid", "aero", "thermal", "field"):
        return NeuralOperatorResidual()
    return IdentityResidual()
