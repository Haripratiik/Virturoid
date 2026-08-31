#!/usr/bin/env python
"""The tests CI enforces on every change, each one tied to a regression this repo actually shipped.

WHY THIS FILE EXISTS RATHER THAN A LIST OF PATHS IN THE WORKFLOW YAML
--------------------------------------------------------------------
CI here has failed the same way twice, and both failures were invisible from the outside:

  * 2026-07-21 -- the two PR jobs named SIX test files out of ~1590 tests. Every regression outside those six
    was invisible for ten days; the first full local run since 07-11 surfaced two real breaks.
  * The full suite is ~68 minutes. A gate nobody can wait for is a gate nobody runs, so a fast tier is not
    optional -- but a fast tier that quietly shrinks is exactly the 2026-07-21 failure again.

So the enforced set lives in code, next to the failure each entry exists to catch, and the runner AUDITS its
own coverage afterwards:

  1. every listed file must still exist -- a rename empties a YAML path list silently, and loudly here;
  2. every listed file must contribute at least one EXECUTED test -- not collected, not skipped, executed;
  3. the tier prints how much of the suite it does NOT cover, every run, in its own output.

Rule 2 is the load-bearing one and it is aimed at a specific lie this repo has told before: 78 tests green on a
subset while the full suite held 12 reds. The corpus tier below runs against the real MuJoCo Menagerie, and
those tests `pytest.skip` when the corpus is absent -- so without rule 2 a runner with no corpus reports a
green tier having verified nothing at all.

WHAT THIS TIER CANNOT DO
------------------------
It is a REGRESSION RATCHET, not a substitute for the suite. It cannot catch a break in code no listed test
touches, it runs no GPU and no live LLM, and it will not notice a NEW failure mode until someone writes a test
for it and adds it here. The nightly full-suite job remains the authority. `python scripts/ci_gate.py list`
prints that sentence together with the uncovered-file count, so it is on the console of every run rather than
in a document nobody opens.

USAGE
    python scripts/ci_gate.py list                 # the table, plus what the tier misses
    python scripts/ci_gate.py run --tier fast      # PR gate: no external corpus needed
    python scripts/ci_gate.py run --tier corpus    # needs the MuJoCo Menagerie checkout
    python scripts/ci_gate.py run --tier all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


@dataclass(frozen=True)
class Gate:
    """One enforced test file and the shipped defect that justifies it.

    ``failure`` is not documentation for its own sake. A gate whose ``failure`` cannot name a real incident is
    noise, and deleting it is a legitimate change; this field is what makes that judgement possible later.
    """
    path: str
    failure: str
    corpus: bool = False          # needs the real MuJoCo Menagerie checkout to execute
    tags: tuple[str, ...] = field(default_factory=tuple)


#: The PR gate. Nothing here needs a network or an external corpus, so it runs identically on a laptop and a
#: cold GitHub runner.
FAST: tuple[Gate, ...] = (
    Gate("tests/test_suite_does_not_write_the_real_bank.py",
         "2026-08-07: the ordinary verify path banked into the developer's own memory. One session took the "
         "live bank 97 -> 101 locomotion rows, four of them carrying a fixture's body class. The bank is the "
         "substrate the evidence gates measure against, so the suite was editing its own evidence."),
    Gate("tests/test_bank_gate_doors.py",
         "2026-08-06 (#276): three of the four doors that write a banked row passed NO fragility margin, and a "
         "row with no margin read as 'fine'. Rows must record which door banked them and what was measured."),
    Gate("tests/test_bank_gate_provenance.py",
         "2026-08-07 (#276, fifth door): a legitimate re-bank REPLACED base_config and erased the provenance "
         "stamp, laundering a suite-authored row into 'unattributed'. Observed live, minutes after stamping."),
    Gate("tests/test_bank_provenance.py",
         "The audit pass that decides suite/real/unattributed for every banked row. If it mis-stamps, every "
         "later claim about how much of the moat is real is wrong in the same direction."),
    Gate("tests/test_tool_registration.py",
         "2026-08-05 (#269): tools the product advertised did not exist. An agent-facing surface that lists a "
         "capability it cannot perform is the most expensive kind of dishonesty we ship."),
    Gate("tests/test_learned_deploy_wiring.py",
         "2026-08-02 (#260): a build reported a RANDOM policy's numbers as its headline result, because the "
         "controller was inferred from artifact contents instead of being recorded."),
    Gate("tests/test_import_verify_honesty.py",
         "2026-07-24 (#234/B2): ingest silently swapped the customer's robot for a template and reported "
         "success on the substitute. The swap must be disclosed and undoable."),
    Gate("tests/test_export_consistency.py",
         "2026-08-03 (#264): the exported package was not the body that had been verified. What ships and what "
         "was measured must be the same robot."),
    Gate("tests/test_export_gate.py",
         "The gene-path export gate: an unverified package must not be exportable as if it were verified."),
    Gate("tests/test_export_gait_honesty.py",
         "An exported controller must not claim a gait the body was never measured under."),
    Gate("tests/test_ros2_export.py",
         "2026-08-06 (#272): exporters silently dropped physics the source model carried. Found by re-reading "
         "the emitted files rather than trusting the writer."),
    Gate("tests/test_isaac_export.py",
         "Same defect class as ROS2, second backend: USD must carry what the twin actually has."),
    Gate("tests/test_verdict_certificate.py",
         "2026-08-06 (#275): a twin that diverged to NaN still produced a verdict certificate. MuJoCo's "
         "mj_checkAcc calls mj_resetData, so a finiteness check written the obvious way passes vacuously."),
    Gate("tests/test_certificate_v2.py",
         "The NASA-STD-7009 certificate schema: a certificate that omits its own limits is worse than none."),
    Gate("tests/test_status_honesty.py",
         "2026-08-03 (#263): the status bar said 'unverified' while Verify said 'EXPORT BLOCKED' and the "
         "header said 'valid' -- three surfaces, three verdicts, one robot."),
    Gate("tests/test_ui_honesty.py",
         "Same incident: a green chip appeared on a build that produced nothing."),
    Gate("tests/test_honesty_scorecard.py",
         "The scorecard is the single place a claim is compared to its evidence; if it drifts, every other "
         "surface inherits the drift."),
    Gate("tests/test_design_delta_honest.py",
         "2026-08-04 (#252): an amend compiled with the right topology and did not READ as the requested "
         "change. 'The graph changed' is not evidence the robot did."),
    Gate("tests/test_design_bench.py",
         "2026-08-05 (#246): the hybrid family regressed 1.0 -> 0.0 and NOTHING noticed, because the bench "
         "reported an aggregate. Per-family verdicts are the gate."),
    Gate("tests/test_moat_panel.py",
         "The Memory tab's read model. It renders a currently-unflattering bank (1 of 101 rows gated, dominant "
         "recall kind negative); a surface that can only render wins would be worse than no surface."),
    Gate("tests/test_ci_gate.py",
         "2026-07-21: the PR jobs gated on six test files out of ~1590 and nobody noticed for ten days. This "
         "gate guards its own definition -- a renamed enforced file fails the suite instead of quietly "
         "emptying the list, and the audit's two refusals are tested rather than assumed."),
)

#: Needs a real MuJoCo Menagerie checkout. Every one of these SKIPS without it -- which is why the runner
#: refuses to call the tier green when they skip, instead of reporting the skips as passes.
CORPUS: tuple[Gate, ...] = (
    Gate("tests/test_multiroot_twin_is_simulable.py", corpus=True,
         failure="2026-08-06 (#271): 4 of 63 Menagerie packages declare multiple roots and ALL FOUR produced a "
                 "broken twin that reported valid=True. Every downstream number is computed by STEPPING that twin."),
    Gate("tests/test_imported_verdict_honesty.py", corpus=True,
         failure="2026-07-25 (#218): ingested legged and arm bodies were judged by the wrong rubric, producing "
                 "a dishonest 'TIPPED while driving' for a robot that does not drive."),
    Gate("tests/test_exported_couplings.py", corpus=True,
         failure="2026-08-06 (#272): URDF <mimic> was emitted for 0 of 24 coupled models. The customer's "
                 "kinematic constraints were being dropped on the way out."),
    Gate("tests/test_same_robot_throughout.py", corpus=True,
         failure="2026-08-03 (#264): render, verification and export disagreed about which robot they were "
                 "describing."),
    Gate("tests/test_mesh_carrythrough.py", corpus=True,
         failure="2026-07-29 (#215): ingest rebuilt a generic proxy while the customer's own meshes sat unused."),
)

ALL: tuple[Gate, ...] = FAST + CORPUS

#: Executed-test floors. They exist to catch a tier that silently shrinks WITHOUT any file disappearing -- a
#: parametrised sweep losing its parameters, a class-level skip, a fixture that stops yielding. Measured
#: 2026-08-07 on this tree: fast executed 176 in 2m43s, corpus executed 97 in 7m57s. The floors sit ~6% under
#: those so a single legitimate deletion does not flake the gate. Raise them when a tier legitimately grows;
#: never lower one to make a run pass -- lowering it is how the gate stops meaning anything.
FLOORS = {"fast": 165, "corpus": 90, "all": 255}

#: A skip whose reason matches any of these means the ENVIRONMENT could not run the test, not that the test
#: chose to opt out. In a tier that claims to cover those tests, that is a FAILURE, not a pass.
#:
#: The corpus tests do not all phrase it the same way -- measured 2026-08-07 against a runner with the cache
#: hidden, the 83 skips split across "is not in the local Menagerie cache", "is not cached locally
#: (robot_descriptions fetches on demand)" and "model X not present at <path>". Matching only the first
#: caught 43 of them; the floor check caught the rest, but a gate should not depend on its backstop.
_ENV_SKIP_MARKERS = ("menagerie", "not cached", "not present at", "not installed", "needs mujoco",
                     "no module named", "requires mujoco")


def tier(name: str) -> tuple[Gate, ...]:
    return {"fast": FAST, "corpus": CORPUS, "all": ALL}[name]


def missing(gates: tuple[Gate, ...]) -> list[str]:
    return [g.path for g in gates if not (REPO_ROOT / g.path).exists()]


def uncovered_files() -> list[str]:
    """Test files in tests/ that NO tier enforces. Printed on every run; this is the honest denominator."""
    gated = {g.path for g in ALL}
    return sorted(
        f"tests/{p.name}" for p in TESTS_DIR.glob("test_*.py") if f"tests/{p.name}" not in gated
    )


def print_table(name: str) -> None:
    gates = tier(name)
    print(f"CI gate tier '{name}': {len(gates)} enforced test files\n")
    for g in gates:
        mark = "MISSING " if not (REPO_ROOT / g.path).exists() else ""
        print(f"  {mark}{g.path}{'  [needs Menagerie]' if g.corpus else ''}")
        print(f"      {g.failure}")
    uncov = uncovered_files()
    total = len(uncov) + len(ALL)
    print(f"\n  LIMITS OF THIS GATE -- read them before trusting a green run:")
    print(f"    * it enforces {len(ALL)} of {total} test files; {len(uncov)} are NOT covered by any tier.")
    print(f"    * it runs no GPU, no live LLM and no multi-hour corpus job.")
    print(f"    * it is a ratchet on defects we have ALREADY shipped. A new failure mode is invisible to it")
    print(f"      until someone writes a test and adds it to this file.")
    print(f"    * the nightly full suite (~68 min) remains the authority. Green here is not green overall.")


def _audit(junit: Path, gates: tuple[Gate, ...], floor: int) -> list[str]:
    """Read the junit report and return the reasons this tier must not be called green. Empty == clean."""
    problems: list[str] = []
    try:
        root = ET.parse(junit).getroot()
    except (OSError, ET.ParseError) as exc:
        return [f"could not read the junit report at {junit}: {exc}"]

    executed: dict[str, int] = {g.path: 0 for g in gates}
    # pytest reports a case as file="tests/test_x.py" OR classname="tests.test_x.SomeTests" depending on
    # version and test style, so match on the module STEM found in either -- comparing whole strings silently
    # matched nothing and reported every gate empty (observed while building this).
    by_stem = {Path(g.path).stem: g.path for g in gates}
    env_skipped: list[str] = []
    total_executed = 0
    for case in root.iter("testcase"):
        raw = (case.get("file") or "") + "." + (case.get("classname") or "")
        tokens = set(raw.replace("\\", ".").replace("/", ".").split("."))
        path = next((by_stem[t] for t in tokens if t in by_stem), None)
        skip = case.find("skipped")
        if skip is not None:
            reason = f"{skip.get('message', '')} {skip.text or ''}".lower()
            if any(m in reason for m in _ENV_SKIP_MARKERS):
                env_skipped.append(f"{case.get('classname')}::{case.get('name')} -- {skip.get('message', '')}")
            continue
        total_executed += 1
        if path:
            executed[path] += 1

    for path, n in executed.items():
        if n == 0:
            problems.append(f"{path} contributed ZERO executed tests -- the gate is present but empty")
    if env_skipped:
        problems.append(
            f"{len(env_skipped)} test(s) were skipped because the ENVIRONMENT could not run them, in a tier "
            f"that claims to cover them. This is the 'green on a subset' failure. First: {env_skipped[0]}")
    if total_executed < floor:
        problems.append(f"only {total_executed} tests executed; this tier's recorded floor is {floor}. "
                        f"Something stopped being collected.")
    return problems


def run(name: str, extra: list[str]) -> int:
    gates = tier(name)
    gone = missing(gates)
    if gone:
        print("REFUSING TO RUN. Enforced test files are missing (renamed or deleted):", file=sys.stderr)
        for p in gone:
            print(f"  {p}", file=sys.stderr)
        print("Fix the path in scripts/ci_gate.py, or delete the gate and say in the commit why the defect it "
              "guarded can no longer happen.", file=sys.stderr)
        return 2

    print_table(name)
    junit = REPO_ROOT / f"ci-gate-{name}.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", f"--junitxml={junit}",
           *[g.path for g in gates], *extra]
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    env = dict(os.environ)
    # The suite's own conftest redirects the memory bank, but a hermetic run is also the documented way to keep
    # a CI machine from reading a developer .env that is not there.
    env.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
    rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode

    problems = _audit(junit, gates, FLOORS.get(name, 1))
    if problems:
        print("\nGATE AUDIT FAILED -- this run must not be reported as green:", file=sys.stderr)
        for p in problems:
            print(f"  * {p}", file=sys.stderr)
        return max(rc, 3)
    if rc == 0:
        print(f"\nGate '{name}' passed, and its coverage audit passed. "
              f"Remember what it does not cover (printed above).")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lst = sub.add_parser("list", help="print the enforced set and this gate's limits")
    lst.add_argument("--tier", default="all", choices=("fast", "corpus", "all"))
    rn = sub.add_parser("run", help="run a tier and audit its own coverage")
    rn.add_argument("--tier", default="fast", choices=("fast", "corpus", "all"))
    rn.add_argument("pytest_args", nargs="*", help="extra arguments passed straight to pytest")
    args = ap.parse_args()
    if args.cmd == "list":
        print_table(args.tier)
        return 0
    return run(args.tier, list(args.pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
