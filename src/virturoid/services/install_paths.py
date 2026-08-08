"""Where the installation's own directories live — the ONE anchoring rule.

WHY THIS MODULE EXISTS. Three separate incidents in one week, all the same shape: a path that reads
like a constant and is actually a function of the process's working directory.

  1. ``memory_db.DEFAULT_DB_PATH = Path("build")/"memory"/...`` — the test suite grew the developer's
     real gait bank 97 -> 101 rows, four of them classed ``totally_made_up_xyz``.       (fixed f77d092)
  2. ``agent_tools.safe_build_path(None, "memory")`` -> ``<cwd>/build/memory`` — a SECOND default rule
     that never read the redirect, so 231 ``runs`` rows leaked AFTER the first fix was declared
     verified.                                                                          (fixed f77d092)
  3. ``ui_server._resolve_build_root`` kept a bare relative ``build/ui_verify`` — the same install
     showed different robots from different directories.                                (fixed 248afca)

Each was found by accident, after shipping. The common cause is not carelessness at any one site: it
is that "where does the bank live" had no owner, so every caller re-derived it, and re-derivations
drift. This module is that owner for the ANCHOR half of the question.

THE RULE, in one line: a destination that belongs to the INSTALL is anchored to the checkout; a
destination that belongs to the CALLER's invocation stays relative to the caller.

  * Anchored (use ``anchored()``): the gait bank, the session store, the policy bank, the fitted
    whitener/metric, the demo set. These are properties of "which Virturoid am I running", so
    ``cd ..`` must not change them. Several are TRACKED IN GIT, which settles it — a checkout has
    them and a bare wheel does not, so "no checkout" and "no artifact" are the same fact. Anchored
    or absent; never "wherever I happen to be standing".
  * Legitimately relative: a URI *inside* a built robot package (``reports/spec_sheet.json``,
    ``simulation/scene_set.json``) — those are joined to a package root by their caller and are
    portable by design. And an ``argparse`` default the user sees and can override on the command
    line. Neither is this bug; see ``tests/test_no_cwd_relative_destinations.py``.

HOW TO ANCHOR. Prefer an explicit env override, falling back to ``anchored(...)``::

    def _dir() -> Path:
        return Path(os.environ.get("VIRTUROID_SESSIONS_DIR") or anchored("build/sessions"))

The env override is what lets a test session, a corpus-factory night and the product each point
somewhere different in one process; the anchored default is what stops ``cd`` from doing it silently.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not decide *policy* — whether a given write is
allowed, redirected or refused. That lives at the writer's own chokepoint (``MemoryDB.__init__`` for
the bank, and it should stay there). This module only answers "anchored to what". Keep it
dependency-free so anything, including the schemas layer, can import it.
"""
from __future__ import annotations

import os
from pathlib import Path

#: ``.../src/virturoid/services/install_paths.py`` -> parents[3] is the repo root of a source checkout.
_HERE = Path(__file__).resolve()


def checkout_root() -> Path | None:
    """The source checkout this module was imported from, or ``None`` when it is an installed wheel.

    Detected by structure (``pyproject.toml`` + ``src/virturoid/``) rather than by a fixed number of
    ``parents``, so a re-layout does not silently start returning site-packages.
    """
    for parent in _HERE.parents:                 # .../services -> .../virturoid -> .../src -> <checkout>
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "virturoid").is_dir():
            return parent
    return None


def anchored(relative: str | os.PathLike[str]) -> Path:
    """``<checkout>/<relative>`` in a source checkout; the unchanged relative path otherwise.

    The wheel case genuinely has nothing to anchor to, and inventing a location (a temp dir, the user's
    home) is worse than the honest fallback: it would put artifacts somewhere nobody was told about. So
    a wheel keeps today's behaviour, and every caller that has a shipped artifact to find should treat
    "no checkout" as "no artifact" and say so, the way ``ui_server._resolve_build_root`` already does.

    An absolute ``relative`` is returned untouched — callers pass user-supplied paths through here.
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    root = checkout_root()
    return (root / p) if root else p


def policy_bank_dir(models_dir: str | os.PathLike[str] = "models") -> Path:
    """Resolve a learned-policy bank directory: absolute stays put, relative anchors to the install.

    The bank's conventional name is the bare relative ``"models"``, and it is not only a default -- three call
    sites pass the literal EXPLICITLY (``gene_build``'s headline scoring, ``desktop``'s "is a policy banked?"
    check, ``capability_registry``'s ``params.get("models_dir", "models")``), which is why this anchors at the
    point of RESOLUTION rather than at the argument defaults. Fixing only the defaults is what turned incident 1
    into incident 2: a second rule that still computed the old answer.

    Two measured facts make this a genuine defect rather than tidiness. ``models/learned_quadruped.composed.npz``
    and ``models/morph_frog_vel.npz`` are TRACKED IN GIT -- shipped artifacts, so like the ``build/ui_verify``
    demo set they are anchored or absent, never "wherever I am". And ``banked_policy_for`` returns None on a
    miss, which every caller reads as "no learned policy yet": from one directory up, the same install silently
    deploys the scripted fallback instead of the banked walker, and reports no error. ``tests/test_nav_learned.py``
    already anchors this exact directory with ``Path(__file__).parent.parent / "models"`` -- the test knew, and
    production did not.

    Writes are safe to anchor here in a way the memory bank's were NOT: banking is KEEP-BEST
    (``learn_locomotion`` refuses to clobber a stronger banked policy for the species), so a stray run can only
    improve the bank, never pollute it with a row that must then be measured against forever.

    ``VIRTUROID_MODELS_DIR`` overrides, mirroring ``VIRTUROID_MEMORY_DIR``/``VIRTUROID_SESSIONS_DIR``.
    """
    p = Path(models_dir)
    if p.is_absolute():
        return p
    override = os.environ.get("VIRTUROID_MODELS_DIR")
    if override and str(p) == "models":          # only the CONVENTIONAL name is redirected; a caller that asked
        return Path(override)                    # for some other relative dir gets that dir, anchored
    return anchored(p)
