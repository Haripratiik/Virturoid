"""A LONG CALL THAT PRINTS NOTHING IS INDISTINGUISHABLE FROM A HANG.

MEASURED 2026-08-10 across 20 real ``create_robot`` calls through ``agent_tools.call_tool``: 0.5 s to 634 s.
The 634 s call produced no output at all until it returned, and three bodies spent a full 360-gait search and
adopted nothing. A trial user cannot tell that apart from a wedged process, and the same defect was already
fixed once today in ``scripts/run_mvp_demo.py`` (9m39s of silence -> a 15-second heartbeat). This module is that
fix moved into the LIBRARY, so it reaches every caller of the tool rather than one script.

TWO OUTPUTS, and they are deliberately different in kind:

  * a **console heartbeat** on stderr, for a human watching a terminal -- stage in, stage out, and a line every
    ``_HEARTBEAT_S`` in between so the clock visibly moves;
  * a **machine-readable stage table** returned in the tool result, for the agent that called the tool -- which
    never sees stderr and today gets one opaque ``took_s``.

STDERR, NEVER STDOUT, and this is load-bearing rather than stylistic: ``virturoid.mcp_server`` speaks JSON-RPC
over stdout, so a single progress line printed there corrupts the protocol frame and takes the session down. The
writer is also wrapped in a bare ``except`` -- a progress line must never be able to fail a build, and pytest's
capture replaces ``sys.stderr`` with an object that is closed at teardown, which a daemon heartbeat thread can
outlive by a few milliseconds.

DEFAULT ON FOR A PRODUCT RUN, OFF UNDER PYTEST. The suite calls ``create_robot`` about fifty times; a heartbeat
per call is noise in a log nobody reads, so ``pytest`` in ``sys.modules`` turns printing off. ``VIRTUROID_PROGRESS``
overrides in both directions (``1`` forces it on, ``0`` forces it off) so the behaviour itself stays testable.
The stage TABLE is always recorded either way -- it costs a ``time.monotonic()`` per stage and it is the half a
caller can act on.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import sys
import threading
import time

#: Seconds between "still running" lines. Same cadence ``run_mvp_demo`` settled on: short enough that a reader
#: never waits long enough to doubt, long enough that a 10-minute stage prints 40 lines rather than 600.
_HEARTBEAT_S = 15.0

_CURRENT: contextvars.ContextVar = contextvars.ContextVar("virturoid_build_progress", default=None)


def printing_enabled() -> bool:
    """Should the console heartbeat actually print? See the module docstring for why pytest is special."""
    flag = os.environ.get("VIRTUROID_PROGRESS")
    if flag is not None:
        return flag.strip() == "1"
    return "pytest" not in sys.modules


def _ascii(s: str) -> str:
    """Windows consoles default to cp1252 and cannot encode an em-dash. Sanitise at the print site: a progress
    line that raises ``UnicodeEncodeError`` inside a daemon thread is worse than no progress line."""
    return (str(s).replace("—", "-").replace("–", "-").replace("’", "'")
            .encode("ascii", "replace").decode("ascii"))


class BuildProgress:
    """Records stage timings for the result, and optionally narrates them to stderr.

    Not thread-safe by design of use: stages are entered and left on one call's thread. The heartbeat thread only
    READS ``_t0``/``_label`` and only writes to stderr, so it cannot perturb a verdict.
    """

    def __init__(self, label: str, *, printing: bool | None = None, budget_s: float | None = None,
                 fit_gait: bool = True, evals_budget: int | None = None) -> None:
        self.label = label
        # BUILD-WIDE POLICY, not just narration. ``create_robot`` is not the only thing that fits a gait during
        # a build -- ``anatomy_compiler.ensure_walkable_quad`` calls the fitter twice more, on bodies
        # ``create_robot`` never sees. MEASURED: ``tune_gait: false`` therefore did NOT turn the fit off; the
        # walkability gate ran a full one anyway and overwrote the "skipped" disclosure with a real result. A
        # flag on the build reaches every fitter inside it without threading a parameter through code that has
        # no business knowing about the tool's arguments.
        self.fit_gait = bool(fit_gait)
        self.printing = printing_enabled() if printing is None else bool(printing)
        self.stages: list[dict] = []
        self.notes: list[str] = []
        self._t_start = time.monotonic()
        # THE BUILD'S SEARCH CLOCK: ONE budget, shared by every search inside the build, STARTED BY THE FIRST
        # SEARCH THAT NEEDS IT rather than at construction. Both halves are measured corrections.
        #
        # ONE clock, because ``create_robot`` runs the gait fit up to three times (the authored body, the
        # walkability gate's candidate, then a substituted body) — three independent 180 s ceilings is a
        # 9-minute call that honoured every one of them.
        #
        # STARTED LATE, because starting it here made a slow COMPOSER cancel the search. MEASURED: a biped whose
        # composer took 253.65 s reached its fit with 0.0 s left and ran ZERO of 360 evaluations — the customer
        # waited 259 s and got a body with no fitted controller and no measurement of one, which is the outcome
        # the fit exists to prevent. Nothing can interrupt the composer, so charging the search for it only
        # moves the loss onto the one stage that was doing useful work.
        self.budget_s = None if (budget_s is None or float(budget_s) <= 0) else float(budget_s)
        self.deadline: float | None = None
        # THE SECOND BUDGET, AND IT EXISTS BECAUSE THE FIRST ONE IS NOT REPRODUCIBLE. A wall clock buys a bound
        # on the WAIT, but what it stops is not a property of the robot: MEASURED 2026-08-12, two runs of the
        # same "an eight-legged spider robot" build on the same box, both stopped by the same 180 s ceiling,
        # stopped at 144 of 360 gaits and at 70 of 360. That is why ``tests/conftest.py`` has to pin the clock
        # OFF and why every budget test asserts on a disclosure rather than a duration. An EVALUATION budget bounds
        # the same search in a unit the physics owns rather than the hardware, so "this build was budgeted" is a
        # fact a test can pin and a customer can reproduce. Same ONE-LEDGER rule as the clock: shared by every
        # fit in the build, because three fits each honouring "spend up to N evaluations" spend 3N.
        self.evals_budget = None if (evals_budget is None or int(evals_budget) <= 0) else int(evals_budget)
        self.evals_spent = 0

    def evals_remaining(self) -> int | None:
        """Evaluations this build may still spend on gait search, or ``None`` when unbounded (the default)."""
        if self.evals_budget is None:
            return None
        return max(0, self.evals_budget - self.evals_spent)

    def spend_evals(self, n: int) -> None:
        """Charge ``n`` gait evaluations to the build's shared ledger. A no-op on an unbounded build, but the
        running total is still kept so a caller can report what a build actually cost."""
        self.evals_spent += max(0, int(n))

    def claim_deadline(self) -> float | None:
        """The shared search deadline, STARTING IT on first call. ``None`` when this build is unbounded.

        Idempotent: the second and third fits of a build get the remainder of the first one's budget, not a
        fresh one. Call it at the moment real search work is about to begin — a fit that returns early (the
        body's default already walks, the controller does not apply, fitting is off) must not start the clock
        for the fits that follow it.
        """
        if self.budget_s is not None and self.deadline is None:
            self.deadline = time.monotonic() + self.budget_s
        return self.deadline

    # ---------------------------------------------------------------- output
    def _write(self, line: str) -> None:
        if not self.printing:
            return
        try:
            sys.stderr.write(_ascii(line) + "\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 - a progress line may never fail a build (closed stream at teardown)
            pass

    def _clock(self) -> str:
        s = int(time.monotonic() - self._t_start)
        return f"[{s // 60:d}:{s % 60:02d}]"

    def say(self, msg: str) -> None:
        """A one-off line that is not a stage -- e.g. what a fit MAY cost on this body, said BEFORE it runs.

        Recorded on ``notes`` as well as printed, because the agent-facing caller cannot read stderr and the
        up-front cost statement is exactly the thing it needs in order to decide whether to wait.
        """
        self.notes.append(str(msg))
        self._write(f"{self._clock()} {self.label}: {msg}")

    # ---------------------------------------------------------------- stages
    @contextlib.contextmanager
    def stage(self, name: str, note: str = ""):
        """Time one stage, heartbeat while it runs, and append ``{stage, seconds}`` to the table.

        Re-entering the same ``name`` ACCUMULATES into one row rather than appending a second: ``create_robot``
        grounds and fits twice when the walkability gate substitutes a body, and two rows called "gait_fit"
        would read as two different stages to anyone summing the table.
        """
        t0 = time.monotonic()
        self._write(f"{self._clock()} {self.label}/{name} ...{(' ' + note) if note else ''}")
        stop = threading.Event()

        def _beat() -> None:
            while not stop.wait(_HEARTBEAT_S):
                self._write(f"{self._clock()}   ... still running: {self.label}/{name} "
                            f"({int(time.monotonic() - t0)}s)")

        thread = threading.Thread(target=_beat, daemon=True) if self.printing else None
        if thread is not None:
            thread.start()
        failed = False
        try:
            yield self
        except BaseException:
            failed = True
            raise
        finally:
            stop.set()
            secs = round(time.monotonic() - t0, 2)
            for row in self.stages:
                if row["stage"] == name:
                    row["seconds"] = round(row["seconds"] + secs, 2)
                    break
            else:
                self.stages.append({"stage": name, "seconds": secs})
            self._write(f"{self._clock()} {'x' if failed else 'done'} {self.label}/{name} in {secs}s")

    def table(self) -> list[dict]:
        """The stage breakdown, slowest first -- the shape a reader asks 'where did the time go?' in."""
        return sorted((dict(r) for r in self.stages), key=lambda r: -r["seconds"])

    def elapsed_s(self) -> float:
        return round(time.monotonic() - self._t_start, 2)


@contextlib.contextmanager
def build_progress(label: str, *, printing: bool | None = None, budget_s: float | None = None,
                   fit_gait: bool = True, evals_budget: int | None = None):
    """Install a ``BuildProgress`` for the duration of a call, so nested code can reach it without plumbing.

    A CONTEXTVAR RATHER THAN AN ARGUMENT, because the stages that matter are three and four frames down
    (``gait_flywheel.fit_gait_for_body`` inside ``ai_native_tools.create_robot``) and threading a reporter
    through every signature between them would be a much larger and much more breakable change than the problem
    warrants. Nesting is honoured: an inner ``build_progress`` restores the outer one on exit, and ``current()``
    returns ``None`` outside any of them, which is what makes every emit site a safe no-op by default.
    """
    rep = BuildProgress(label, printing=printing, budget_s=budget_s, fit_gait=fit_gait,
                        evals_budget=evals_budget)
    token = _CURRENT.set(rep)
    try:
        yield rep
    finally:
        _CURRENT.reset(token)


def current() -> BuildProgress | None:
    """The active reporter, or ``None``. Every caller must tolerate ``None`` -- most code runs outside a build."""
    return _CURRENT.get()


@contextlib.contextmanager
def stage(name: str, note: str = ""):
    """``with stage("gait_fit"):`` from anywhere -- a no-op when no build is in progress."""
    rep = current()
    if rep is None:
        yield None
        return
    with rep.stage(name, note) as r:
        yield r


def say(msg: str) -> None:
    """Emit a one-off progress note if a build is in progress; otherwise do nothing."""
    rep = current()
    if rep is not None:
        rep.say(msg)
