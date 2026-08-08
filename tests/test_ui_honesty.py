"""The package API surfaces the build's honest signals (§4.1/§4.8A/§4.8E) so the Build Console can show
requested-vs-achieved + the BOM<->sim fidelity gap, not just the spec sheet."""

import json
import tempfile
import unittest
from pathlib import Path

from virturoid.ui_server import package_honesty_summary


class UIHonestyTests(unittest.TestCase):
    def test_reads_fidelity_and_compliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp)
            (child / "reports").mkdir()
            (child / "reports" / "bom_sim_fidelity.json").write_text(
                json.dumps({"mass_fidelity_ratio": 2.26, "flags": ["optimistic"]}), encoding="utf-8")
            (child / "reports" / "spec_compliance.json").write_text(
                json.dumps({"all_honored": True, "constraints": [{"constraint": "height_m"}]}), encoding="utf-8")
            h = package_honesty_summary(child)
            self.assertEqual(h["mass_fidelity_ratio"], 2.26)
            self.assertEqual(h["fidelity_flags"], 1)
            self.assertTrue(h["spec_all_honored"])
            self.assertEqual(h["spec_constraints"], 1)

    def test_none_when_no_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(package_honesty_summary(Path(tmp)))

    def test_resolve_build_root_keeps_a_root_that_has_packages(self):
        from virturoid.ui_server import _resolve_build_root
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "myroot"
            (root / "pkg1" / "robot").mkdir(parents=True)              # a built package -> no demo fallback
            (root / "pkg1" / "robot" / "robot.urdf").write_text("<robot name='x'/>", encoding="utf-8")
            self.assertEqual(_resolve_build_root(root), root)

    def test_a_root_the_library_would_list_as_empty_is_treated_as_empty(self):
        """The resolver and the list must share one predicate: a half-written package (a bare ``robot/`` with
        no URDF and no compiled scenes) used to satisfy the resolver, suppress the demo fallback, and then
        list zero robots -- an empty library served from a root just declared non-empty."""
        import virturoid.ui_server as ui
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "myroot"
            (root / "half_written" / "robot").mkdir(parents=True)      # no robot.urdf, no scenes
            original = ui.checkout_root
            ui.checkout_root = lambda: None
            try:
                self.assertFalse(ui._has_packages(root))
            finally:
                ui.checkout_root = original


class BuildRootIsAnchoredToTheCheckoutTests(unittest.TestCase):
    """The Robot Library must not depend on the shell's pwd.

    Measured before this was anchored: launching `python scripts/run_ui.py --ui studio --web` from one
    directory ABOVE the checkout resolved the build root to `<that dir>/build/ui_workbench`, found no demo set
    (`build/ui_verify` was CWD-relative too) and served an EMPTY library from a clone that has four tracked demo
    packages in it. Same command, same clone, different pwd, different product."""

    def test_demo_set_is_found_in_the_checkout_not_the_cwd(self):
        import virturoid.ui_server as ui
        with tempfile.TemporaryDirectory() as tmp:
            fake_checkout = Path(tmp) / "checkout"
            demo_pkg = fake_checkout / "build" / "ui_verify" / "demo_pkg"
            (demo_pkg / "robot").mkdir(parents=True)
            (demo_pkg / "robot" / "robot.urdf").write_text("<robot name='x'/>", encoding="utf-8")
            empty_root = Path(tmp) / "workbench"                       # the user's own root: nothing built yet
            empty_root.mkdir()
            original = ui.checkout_root
            ui.checkout_root = lambda: fake_checkout                   # noqa: E731 - narrow, restored below
            try:
                self.assertEqual(ui._resolve_build_root(empty_root),
                                 fake_checkout / "build" / "ui_verify")
            finally:
                ui.checkout_root = original

    def test_no_packages_anywhere_returns_the_requested_root(self):
        import virturoid.ui_server as ui
        with tempfile.TemporaryDirectory() as tmp:
            empty_root = Path(tmp) / "workbench"
            empty_root.mkdir()
            original = ui.checkout_root
            ui.checkout_root = lambda: None                            # noqa: E731 - installed-wheel case
            try:
                # No demo set to fall back to; it must not silently retarget somewhere else.
                self.assertEqual(ui._resolve_build_root(empty_root), empty_root)
            finally:
                ui.checkout_root = original


class EmptyLibraryExplainsItselfTests(unittest.TestCase):
    """An empty library is a state, not a bug -- and the API has to say which, or a first-run reader cannot
    tell 'nothing built yet' from 'pointed at the wrong folder'."""

    def _packages(self, root: Path) -> dict:
        import json as _json
        import urllib.request
        from threading import Thread
        from virturoid.ui_server import create_server
        server = create_server("127.0.0.1", 0, root)
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/packages") as r:
                return _json.loads(r.read().decode("utf-8"))
        finally:
            server.shutdown()

    def test_empty_payload_names_the_directory_and_the_way_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self._packages(Path(tmp))
            self.assertEqual(data["packages"], [])
            empty = data["empty"]
            self.assertEqual(Path(empty["scanned"]), Path(tmp).resolve())
            self.assertTrue(empty["next_steps"])
            self.assertTrue(any("run_mvp_demo" in s for s in empty["next_steps"]))

    def test_populated_payload_has_no_empty_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pkg1" / "robot").mkdir(parents=True)
            (Path(tmp) / "pkg1" / "robot" / "robot.urdf").write_text("<robot name='x'/>", encoding="utf-8")
            data = self._packages(Path(tmp))
            self.assertEqual([p["id"] for p in data["packages"]], ["pkg1"])
            self.assertNotIn("empty", data)


class StudioOwnsTheRootWhenAskedForTests(unittest.TestCase):
    """`--ui studio` and then the LEGACY console at `/` is how a first-run evaluator concludes the app is
    broken while Studio runs one path segment away."""

    def _get(self, root: Path, path: str, ui: str):
        import urllib.error
        import urllib.request
        from threading import Thread
        from virturoid.ui_server import create_server

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        server = create_server("127.0.0.1", 0, root, ui=ui)
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}{path}"
            try:
                with urllib.request.build_opener(_NoRedirect).open(url) as r:
                    return r.status, r.headers.get("Location"), r.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.headers.get("Location"), b""
        finally:
            server.shutdown()

    def test_root_redirects_to_studio_when_studio_was_requested(self):
        from virturoid.ui_server import FRONTEND_DIST
        if not (FRONTEND_DIST / "index.html").exists():
            self.skipTest("Studio bundle not built in this checkout")
        with tempfile.TemporaryDirectory() as tmp:
            status, location, _ = self._get(Path(tmp), "/", ui="studio")
            self.assertEqual(status, 302)
            self.assertEqual(location, "/studio/")

    def test_legacy_console_keeps_an_address_of_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, _, body = self._get(Path(tmp), "/legacy", ui="studio")
            self.assertEqual(status, 200)
            self.assertIn(b"Virturoid", body)

    def test_default_root_is_unchanged_for_the_legacy_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, _, body = self._get(Path(tmp), "/", ui="legacy")
            self.assertEqual(status, 200)
            self.assertIn(b"Virturoid", body)


class CommittedStudioBundleIsSelfConsistentTests(unittest.TestCase):
    """The clone must contain the assets its own index.html asks for.

    ``frontend/dist/`` is in .gitignore and its files are force-added, so `npm run build` emits NEW
    content-hashed filenames that git does not pick up on its own. Rebuild, commit, and a fresh clone gets an
    index.html referencing `assets/index-<newhash>.js` that is not in the repository -- Studio loads as a blank
    page, on the exact path a first-run evaluator takes. This caught itself during the first-run pass: the
    rebuild landed and the new bundle was untracked until it was force-added."""

    def test_every_asset_index_html_references_is_tracked(self):
        import re
        import subprocess
        from virturoid.ui_server import FRONTEND_DIST, checkout_root
        root = checkout_root()
        if root is None or not (FRONTEND_DIST / "index.html").exists():
            self.skipTest("not a source checkout with a built bundle")
        try:
            tracked = subprocess.run(["git", "ls-files", "frontend/dist"], cwd=root, capture_output=True,
                                     text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git unavailable")
        if tracked.returncode != 0 or not tracked.stdout.strip():
            self.skipTest("frontend/dist is not tracked in this checkout")
        have = {line.strip() for line in tracked.stdout.splitlines() if line.strip()}
        html = (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")
        refs = sorted(set(re.findall(r'(?:src|href)="/studio/(assets/[A-Za-z0-9._-]+)"', html)))
        self.assertTrue(refs, "index.html references no /studio/assets/* files - parser out of date?")
        missing = [r for r in refs if f"frontend/dist/{r}" not in have]
        self.assertEqual(missing, [], f"index.html references untracked assets {missing}; "
                                      f"run `git add -f frontend/dist` after rebuilding Studio")


if __name__ == "__main__":
    unittest.main()
