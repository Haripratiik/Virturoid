import threading
import tempfile
import unittest
from urllib.request import urlopen
from pathlib import Path

from virturoid.ui_server import build_ui_html, create_server, run_build_from_payload


class UiServerTests(unittest.TestCase):
    def test_ui_shell_contains_build_form_and_api_hook(self):
        html = build_ui_html()

        self.assertIn("Virturoid Local Build Workbench", html)
        self.assertIn('id="buildForm"', html)
        self.assertIn("/api/build", html)
        self.assertIn("Train controller when supported", html)

    def test_api_helper_builds_mobile_package_under_build_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = {
                "prompt": "Build a mobile robot base that can drive and navigate indoors to deliver parts.",
                "output_name": "mobile_ui_test",
                "train": False,
            }

            response = run_build_from_payload(payload, Path(tmpdir))
            output_dir = Path(response["result"]["output_dir"])

            self.assertEqual("mobile_base", response["result"]["selected_robot_class"])
            self.assertTrue(response["result"]["package_valid"])
            self.assertEqual("/package/mobile_ui_test/reports/workbench.html", response["workbench_url"])
            self.assertTrue((output_dir / "reports" / "workbench.html").exists())
            self.assertTrue((output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").exists())

    def test_http_server_serves_ui_shell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            server = create_server("127.0.0.1", 0, Path(tmpdir))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                html = urlopen(f"http://{host}:{port}/", timeout=5).read().decode("utf-8")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertIn("Virturoid Local Build Workbench", html)
        self.assertIn("/api/build", html)


if __name__ == "__main__":
    unittest.main()
