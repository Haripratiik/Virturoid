"""Status honesty: the product's whole pitch is "honest by construction", so the UI must never
contradict itself about (a) whether a job produced what was asked for, or (b) what may be claimed
about a robot package.

Two contradictions this suite pins down:

1. **"SUCCEEDED" over "success 0%" / "No robot package was generated".** ``job_registry`` stamped
   ``status="succeeded"`` on any worker that returned without raising -- including the honest
   clarify/refusal path, which deliberately builds nothing. An honest refusal is a GOOD outcome and
   must stay distinct from an ERROR, so the fix is a third terminal status (``no_output``), never
   ``failed``.

2. **Four labels for one robot.** ``valid`` (contract.ok), ``unverified`` (footer), ``EXPORT
   BLOCKED`` (Verify tab) and ``Buildable=False`` (console) were each derived independently, so a
   green VALID chip could sit on the same screen as EXPORT BLOCKED. There is now ONE derivation --
   ``services.package_status.package_status`` -- served on ``/api/packages`` and rendered by every
   surface, with distinct facts carrying distinct labels instead of a generic "valid".
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.schemas.autonomy import AutonomyDecision, AutonomyReport  # noqa: E402
from virturoid.services import job_registry  # noqa: E402
from virturoid.ui_server import create_server  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"succeeded", "failed", "cancelled", "no_output"}


def package_status(package_dir: Path) -> dict:
    """Imported lazily so the job-level contradictions above still run (and fail) independently."""
    from virturoid.services.package_status import package_status as impl

    return impl(package_dir)


def _wait_for_job(job_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = job_registry.get(job_id)
        if job and job["status"] in TERMINAL:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


def _clarify_report(prompt: str) -> AutonomyReport:
    """The real shape autonomous_build returns for an impossible/ambiguous prompt (it stops BEFORE
    generating an unrequested body and writes no robot)."""
    return AutonomyReport(
        id="autonomy_clarify_test",
        prompt=prompt,
        feasible=False,
        succeeded=False,
        decisions=[AutonomyDecision(
            iteration=0, stage="clarify_intent",
            action="stopped before generating an unrequested fallback body",
            detail="no recognizable robot body plan or task", success_before=0.0, success_after=0.0)],
        notes=["No robot package was generated. No recognizable robot body plan or task was found."],
    )


def _built_report(prompt: str) -> AutonomyReport:
    return AutonomyReport(
        id="autonomy_built_test",
        prompt=prompt,
        robot_class="quadruped",
        species="quadruped_generic",
        task_type="locomotion",
        feasible=True,
        succeeded=False,          # a REAL robot that walks badly is still a real robot
        final_success_rate=0.3,
        decisions=[AutonomyDecision(
            iteration=0, stage="build_and_validate", action="built the gene",
            detail="segments=9", success_before=0.0, success_after=0.3)],
    )


def _install_stub_build(report_for, *, write_package: bool):
    """Patch autonomous_build with a stub that mimics the real side effects (report on disk, and a
    robot package only when the build actually produced one). Returns a restore callable."""
    import virturoid.services.autonomous_build as ab

    original = ab.autonomous_build

    def stub(prompt, output_dir, *, target_success_rate=0.8, train=False, memory_dir=None, progress=None, **kw):
        out = Path(output_dir)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "autonomy_report.json").write_text("{}", encoding="utf-8")
        if progress is not None:
            progress({"stage": "start", "message": f"Designing a robot for: {prompt}"})
        if write_package:
            (out / "robot").mkdir(parents=True, exist_ok=True)
            (out / "robot" / "robot.urdf").write_text("<robot name='r'/>", encoding="utf-8")
            if progress is not None:
                progress({"stage": "build", "message": "Built and validated a quadruped."})
        elif progress is not None:
            progress({"stage": "needs_clarification", "message": "Need a clearer body plan or task."})
        return report_for(prompt)

    ab.autonomous_build = stub
    return lambda: setattr(ab, "autonomous_build", original)


def _write_package(root: Path, name: str, *, contract_ok: bool | None, safe_to_export: bool | None,
                   buildable: bool | None = None) -> Path:
    """A minimal package that /api/packages will list (it has a URDF), with the honesty reports the
    four status surfaces read."""
    pkg = root / name
    (pkg / "robot").mkdir(parents=True, exist_ok=True)
    (pkg / "robot" / "robot.urdf").write_text("<robot name='r'/>", encoding="utf-8")
    (pkg / "reports").mkdir(parents=True, exist_ok=True)
    if contract_ok is not None:
        (pkg / "reports" / "robot_package_contract.json").write_text(
            json.dumps({"ok": contract_ok, "robot_class": "quadruped", "species": "dog"}), encoding="utf-8")
    if safe_to_export is not None:
        (pkg / "reports" / "product_readiness_ledger.json").write_text(
            json.dumps({
                "safe_to_export": safe_to_export,
                "highest_attained": "physics_evaluated" if safe_to_export else "real_cad_exported",
                "required": ["schema_valid", "physics_evaluated"],
                "issues": [] if safe_to_export else ["required stage 'physics_evaluated' is 'not_run' (not real)"],
                "stages": [
                    {"stage": "schema_valid", "status": "attained"},
                    {"stage": "physics_evaluated", "status": "attained" if safe_to_export else "not_run"},
                ],
            }), encoding="utf-8")
    if buildable is not None:
        (pkg / "reports" / "buildability_report.json").write_text(
            json.dumps({"buildable": buildable, "confidence": 0.4,
                        "issues": [] if buildable else ["joint 'hip' needs 90.0 Nm but the strongest actuator is 52.0 Nm"]}),
            encoding="utf-8")
    return pkg


# --------------------------------------------------------------------------------------------
# 1. Job-level chip
# --------------------------------------------------------------------------------------------

class JobOutcomeHonestyTests(unittest.TestCase):
    def test_build_that_generated_no_package_is_not_succeeded(self):
        """The reported bug: a green SUCCEEDED chip directly above 'success 0%' and 'No robot
        package was generated'."""
        restore = _install_stub_build(_clarify_report, write_package=False)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = job_registry.create("autonomous_build", {"prompt": "build a robot that flies to Mars"},
                                          Path(tmp))
                done = _wait_for_job(job["id"])
        finally:
            restore()
        self.assertNotEqual("succeeded", done["status"],
                            "a build that produced no robot package must never report success")
        self.assertEqual("no_output", done["status"])
        self.assertFalse(done["result"]["package_written"])
        self.assertIn("No robot package was generated", done["result"]["blocked_reason"])

    def test_honest_refusal_is_not_reported_as_an_error(self):
        """The product's honest-failure handling is GOOD -- a refusal must not be relabelled a
        failure/error (no error string, no error event, and the reason is preserved)."""
        restore = _install_stub_build(_clarify_report, write_package=False)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = job_registry.create("autonomous_build", {"prompt": "build a robot that flies to Mars"},
                                          Path(tmp))
                done = _wait_for_job(job["id"])
                _, events = job_registry.events_since(job["id"], 0)
        finally:
            restore()
        self.assertNotEqual("failed", done["status"])
        self.assertIsNone(done["error"])
        self.assertNotIn("error", [e["stage"] for e in events])
        self.assertTrue(done["result"]["requires_clarification"])

    def test_build_that_produced_a_robot_still_succeeds(self):
        """Guard against over-correction: a real package with weak task success is still a
        successful BUILD -- the success% pill tells the performance story."""
        restore = _install_stub_build(_built_report, write_package=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                job = job_registry.create("autonomous_build", {"prompt": "build a quadruped that walks"},
                                          Path(tmp))
                done = _wait_for_job(job["id"])
        finally:
            restore()
        self.assertEqual("succeeded", done["status"])
        self.assertTrue(done["result"]["package_written"])
        self.assertIsNone(done["result"]["blocked_reason"])
        self.assertEqual(0.3, done["result"]["final_success_rate"])

    def test_tool_jobs_keep_their_statuses(self):
        job = job_registry.create("tool", {"tool": "list_tools", "args": {}}, Path("build"))
        self.assertEqual("succeeded", _wait_for_job(job["id"])["status"])
        bad = job_registry.create("tool", {"tool": "no_such_tool", "args": {}}, Path("build"))
        self.assertEqual("failed", _wait_for_job(bad["id"])["status"])


# --------------------------------------------------------------------------------------------
# 2. Per-robot status
# --------------------------------------------------------------------------------------------

class PackageStatusHonestyTests(unittest.TestCase):
    def test_export_blocked_package_is_never_labelled_valid(self):
        """The reported bug: header chip VALID (green) beside Verify's EXPORT BLOCKED."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _write_package(Path(tmp), "dog", contract_ok=True, safe_to_export=False)
            st = package_status(pkg)
        self.assertEqual("EXPORT BLOCKED", st["label"])
        self.assertEqual("bad", st["kind"])
        self.assertNotIn("valid", st["label"].lower())
        # the contract fact is still reported -- under its own name, not as a generic verdict
        self.assertTrue(st["contract_ok"])
        self.assertFalse(st["safe_to_export"])
        self.assertIn("physics_evaluated", st["detail"])

    def test_export_ready_package_matches_the_verify_tab_wording(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _write_package(Path(tmp), "arm", contract_ok=True, safe_to_export=True)
            st = package_status(pkg)
        self.assertEqual("EXPORT-READY", st["label"])
        self.assertEqual("ok", st["kind"])

    def test_missing_ledger_is_unverified_not_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _write_package(Path(tmp), "old", contract_ok=True, safe_to_export=None)
            st = package_status(pkg)
        self.assertEqual("UNVERIFIED", st["label"])
        self.assertEqual("muted", st["kind"])
        self.assertIsNone(st["safe_to_export"])

    def test_broken_contract_is_package_incomplete_and_outranks_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _write_package(Path(tmp), "broken", contract_ok=False, safe_to_export=True)
            st = package_status(pkg)
        self.assertEqual("PACKAGE INCOMPLETE", st["label"])
        self.assertEqual("bad", st["kind"])

    def test_buildability_is_a_separate_named_fact_not_a_generic_verdict(self):
        """`[buildability] Buildable=False` in the console must be reported under its own label --
        and must never sit under a green claim with nothing said about it."""
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _write_package(Path(tmp), "ungrounded", contract_ok=True, safe_to_export=True, buildable=False)
            st = package_status(pkg)
        self.assertFalse(st["buildable"])
        self.assertTrue(any("actuator" in n for n in st["notes"]), st["notes"])
        with tempfile.TemporaryDirectory() as tmp:
            pkg = _write_package(Path(tmp), "grounded", contract_ok=True, safe_to_export=True, buildable=True)
            self.assertTrue(package_status(pkg)["buildable"])

    def test_packages_api_serves_the_one_status(self):
        """Every surface renders the SAME object: it has to be on the wire."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_package(root, "blocked_dog", contract_ok=True, safe_to_export=False)
            _write_package(root, "ready_arm", contract_ok=True, safe_to_export=True)
            server = create_server("127.0.0.1", 0, root)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                payload = json.loads(urlopen(f"http://{host}:{port}/api/packages", timeout=10).read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
        by_id = {p["id"]: p for p in payload["packages"]}
        self.assertEqual("EXPORT BLOCKED", by_id["blocked_dog"]["status"]["label"])
        self.assertEqual("bad", by_id["blocked_dog"]["status"]["kind"])
        self.assertEqual("EXPORT-READY", by_id["ready_arm"]["status"]["label"])
        self.assertEqual("ok", by_id["ready_arm"]["status"]["kind"])


# --------------------------------------------------------------------------------------------
# 3. The rendering surfaces (no JS test runner in this repo -- guard the binding in source)
# --------------------------------------------------------------------------------------------

class StatusRenderingSurfaceTests(unittest.TestCase):
    SURFACES = (
        "frontend/src/app/TitleBar.tsx",            # header chip
        "frontend/src/panels/inspector/Inspector.tsx",
        "frontend/src/panels/library/LibraryWorkspace.tsx",
    )

    def _src(self, rel: str) -> str:
        return (REPO_ROOT / rel).read_text(encoding="utf-8")

    def _assert_contains(self, rel: str, needle: str) -> None:
        self.assertTrue(needle in self._src(rel), f"{rel} must reference {needle!r}")

    def _assert_absent(self, rel: str, needle: str) -> None:
        self.assertFalse(needle in self._src(rel), f"{rel} must no longer contain {needle!r}")

    def test_surfaces_render_the_shared_status_and_not_a_generic_valid_pill(self):
        for rel in self.SURFACES:
            with self.subTest(surface=rel):
                self._assert_contains(rel, "status")
                self._assert_absent(rel, 'label="valid"')
                self._assert_absent(rel, '? "valid"')

    def test_status_bar_and_verify_agree_on_wording(self):
        # the footer must render the shared label, not its own private verified/unverified verdict
        self._assert_absent("frontend/src/app/StatusBar.tsx", '"unverified"')
        self._assert_contains("frontend/src/app/StatusBar.tsx", "gate.label")
        self._assert_contains("frontend/src/panels/verify/VerifyWorkspace.tsx", "robotStatus")

    def test_job_chip_has_one_label_derivation(self):
        for needle in ("jobStatusLabel", "jobStatusKind", "no_output"):
            self._assert_contains("frontend/src/state/jobs.ts", needle)
        for rel in ("frontend/src/panels/agent/RunCard.tsx", "frontend/src/panels/dock/Dock.tsx"):
            with self.subTest(surface=rel):
                self._assert_contains(rel, "jobStatusLabel")
                self._assert_contains(rel, "jobStatusKind")
        # the poller must treat the new terminal status as terminal, or a refused build spins forever
        self._assert_contains("frontend/src/api/jobs.ts", "no_output")


if __name__ == "__main__":
    unittest.main()
