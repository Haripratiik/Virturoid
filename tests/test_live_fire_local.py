"""Plan v3 M2/WS7a — LOCAL live-fire wiring, proven hermetically with ZERO external spend.

Stands up an in-process OpenAI-compatible ``/v1/chat/completions`` stub (the same protocol Ollama/vLLM speak),
points ``VIRTUROID_LLM_BACKEND=local`` at it, and drives the REAL harness through it end to end:
  * make_routed_llm("proposer") -> LocalLLM -> propose_designs returns a real edit over the HTTP path;
  * generate_detector -> the fail-CLOSED calibration gate trusts a detector that separates pass/fail fixtures.
No network, no API key, no real model -- this is the "the live-fire path is wired" proof the paid runbook then
repeats against a served model on the GPU box (scripts/serve_local_llm.md)."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer


class _OpenAIStub(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat endpoint: inspects the requested schema (embedded in the system prompt by
    both OpenAILLM and LocalLLM) and returns a schema-shaped JSON, wrapped as a chat completion."""

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}")
        system = body["messages"][0]["content"]
        if '"proposals"' in system:
            content = json.dumps({"proposals": [
                {"edit_kind": "cpg", "params": {"calf_phase": 0.2}, "rationale": "small phase to face forward"}]})
        elif "detect(ep)" in system or '"code"' in system:
            content = json.dumps({"code": "def detect(ep):\n    s = float(ep.get('success_rate', 0.0))\n"
                                          "    return {'ok': s >= 0.5, 'score': s}",
                                  "rationale": "threshold the success_rate metric"})
        elif '"viable"' in system:
            content = json.dumps({"viable": True, "reason": "distinct, addresses the failure"})
        else:
            content = json.dumps({"edit_kind": "cpg", "params": {"calf_phase": 0.0}})
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class LocalLiveFireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _OpenAIStub)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def setUp(self):
        import os
        self._saved = {k: os.environ.get(k) for k in
                       ("VIRTUROID_LLM_BACKEND", "VIRTUROID_LOCAL_LLM_URL", "VIRTUROID_LOCAL_LLM_MODEL",
                        "VIRTUROID_NO_LOCAL_ENV")}
        os.environ["VIRTUROID_NO_LOCAL_ENV"] = "1"             # hermetic: ignore any project .env
        os.environ["VIRTUROID_LLM_BACKEND"] = "local"
        os.environ["VIRTUROID_LOCAL_LLM_URL"] = f"http://127.0.0.1:{self.port}/v1"
        os.environ["VIRTUROID_LOCAL_LLM_MODEL"] = "stub-model"
        # get_llm caches the loaded-.env flag; force a re-read is unnecessary (env vars win over the file).

    def tearDown(self):
        import os
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_local_backend_resolves_and_drives_the_proposer(self):
        from virturoid.services.llm_client import make_routed_llm
        from virturoid.services.search_operators import propose_designs
        llm = make_routed_llm("proposer")
        self.assertIsNotNone(llm)                              # backend=local resolved a real client
        art = {"summary_text": "VERDICT: FAIL — walks_backward", "next_actions": ["flip direction"]}
        out = propose_designs("make it walk forward", art, [], llm)   # REAL operator over the HTTP path
        self.assertTrue(out and out[0]["edit_kind"] == "cpg")
        self.assertAlmostEqual(out[0]["params"]["calf_phase"], 0.2)

    def test_local_detector_codegen_through_failclosed_calibration(self):
        from virturoid.services.code_sandbox import generate_detector
        from virturoid.services.llm_client import make_routed_llm
        llm = make_routed_llm("codegen")
        res = generate_detector("grasp a cube",
                                pass_fixtures=[{"success_rate": 0.9}, {"success_rate": 0.7}],
                                fail_fixtures=[{"success_rate": 0.1}, {"success_rate": 0.2}], llm=llm)
        self.assertTrue(res["trusted"])                        # ran in the sandbox + separated the fixtures
        self.assertIn("detect", res["code"])
        self.assertTrue(res["calibration"]["pass_ok"] and res["calibration"]["fail_ok"])

    def test_role_aware_local_routing(self):
        # M2a: breadth roles get the FAST model, deep roles the STRONG model, when the overrides are set.
        import os
        from virturoid.services.llm_client import get_llm
        os.environ["VIRTUROID_LOCAL_LLM_FAST_MODEL"] = "fast-7b"
        os.environ["VIRTUROID_LOCAL_LLM_BUILD_MODEL"] = "strong-70b"
        try:
            self.assertEqual(get_llm("proposer").model, "fast-7b")      # breadth -> cheap
            self.assertEqual(get_llm("diagnostician").model, "strong-70b")  # depth -> strong
            self.assertEqual(get_llm("scene").model, "fast-7b")
        finally:
            os.environ.pop("VIRTUROID_LOCAL_LLM_FAST_MODEL", None)
            os.environ.pop("VIRTUROID_LOCAL_LLM_BUILD_MODEL", None)


if __name__ == "__main__":
    unittest.main()
