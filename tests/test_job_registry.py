import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from virturoid.services import job_registry
from virturoid.ui_server import FRONTEND_DIST, create_server


def _wait_for_job(job_id: str, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = job_registry.get(job_id)
        if job and job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s}s")


class JobRegistryTests(unittest.TestCase):
    def test_light_tool_job_completes_with_events(self):
        job = job_registry.create("tool", {"tool": "list_tools", "args": {}}, Path("build"))
        # create() returns immediately; a light tool may already have finished.
        self.assertIn(job["status"], {"queued", "running", "succeeded"})

        finished = _wait_for_job(job["id"])
        self.assertEqual("succeeded", finished["status"])
        self.assertTrue(finished["result"]["ok"])

        found = job_registry.events_since(job["id"], 0)
        self.assertIsNotNone(found)
        view, events = found
        self.assertEqual(view["id"], job["id"])
        self.assertGreaterEqual(len(events), 2)  # start + done
        self.assertEqual(0, events[0]["seq"])

        # ?since pagination returns only the tail
        _, tail = job_registry.events_since(job["id"], len(events))
        self.assertEqual([], tail)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            job_registry.create("not_a_kind", {}, Path("build"))

    def test_failed_tool_job_reports_error(self):
        job = job_registry.create("tool", {"tool": "no_such_tool", "args": {}}, Path("build"))
        finished = _wait_for_job(job["id"])
        self.assertEqual("failed", finished["status"])
        self.assertIn("no_such_tool", finished["error"])


class StudioAndJobsHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.server = create_server("127.0.0.1", 0, Path(self.tmpdir.name))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmpdir.cleanup()

    def test_studio_route_serves_spa_or_build_hint(self):
        if (FRONTEND_DIST / "index.html").exists():
            html = urlopen(f"{self.base}/studio", timeout=5).read().decode("utf-8")
            self.assertIn("Virturoid Studio", html)
            # SPA fallback: unknown non-asset path also serves the shell
            fallback = urlopen(f"{self.base}/studio/some/deep/route", timeout=5).read().decode("utf-8")
            self.assertIn("Virturoid Studio", fallback)
        else:
            from urllib.error import HTTPError

            with self.assertRaises(HTTPError) as ctx:
                urlopen(f"{self.base}/studio", timeout=5)
            self.assertEqual(404, ctx.exception.code)

    def test_legacy_shell_unchanged_at_root(self):
        html = urlopen(f"{self.base}/", timeout=5).read().decode("utf-8")
        self.assertIn("Virturoid Local Build Workbench", html)

    def test_jobs_http_roundtrip(self):
        body = json.dumps({"kind": "tool", "args": {"tool": "list_tools", "args": {}}}).encode("utf-8")
        req = Request(f"{self.base}/api/jobs", data=body, headers={"Content-Type": "application/json"})
        created = json.loads(urlopen(req, timeout=5).read().decode("utf-8"))
        job_id = created["job"]["id"]
        self.assertTrue(job_id)

        _wait_for_job(job_id)
        payload = json.loads(urlopen(f"{self.base}/api/jobs/{job_id}?since=0", timeout=5).read().decode("utf-8"))
        self.assertEqual("succeeded", payload["job"]["status"])
        self.assertGreaterEqual(len(payload["events"]), 2)

        listing = json.loads(urlopen(f"{self.base}/api/jobs", timeout=5).read().decode("utf-8"))
        self.assertTrue(any(j["id"] == job_id for j in listing["jobs"]))


if __name__ == "__main__":
    unittest.main()
