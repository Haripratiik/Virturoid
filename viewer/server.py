"""Local HTTP server for the Virturoid 3D viewer.

Serves the viewer UI and read-only access to generated packages under build/.
Does not import or modify virturoid package code.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

VIEWER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_ROOT.parent
DEFAULT_BUILD_ROOT = REPO_ROOT / "build"
URDF_REL = Path("robot") / "robot.urdf"
SCENE_INDEX_REL = Path("simulation") / "mujoco" / "compiled_scene_index.json"


def find_packages(build_root: Path) -> list[dict]:
    build_root = build_root.resolve()
    if not build_root.exists():
        return []

    packages: list[dict] = []
    seen: set[str] = set()
    for urdf_path in sorted(build_root.rglob("robot.urdf")):
        if urdf_path.parent.name != "robot":
            continue
        package_dir = urdf_path.parent.parent.resolve()
        try:
            rel = package_dir.relative_to(build_root).as_posix()
        except ValueError:
            continue
        if rel in seen:
            continue
        seen.add(rel)

        scenes: list[dict] = []
        index_path = package_dir / SCENE_INDEX_REL
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                index = {}
            for item in index.get("scenes", []):
                xml_uri = str(item.get("mujoco_xml", "")).replace("\\", "/")
                if not xml_uri:
                    continue
                scenes.append(
                    {
                        "scene_id": item.get("scene_id", ""),
                        "purpose": item.get("purpose", ""),
                        "mujoco_xml": xml_uri,
                        "object_count": item.get("object_count", 0),
                    }
                )

        mvp_scene = "simulation/mujoco/mvp_scene.xml"
        mobile_scene = package_dir / "simulation" / "mujoco" / "mobile_base_scene.xml"
        if mobile_scene.exists():
            mvp_scene = "simulation/mujoco/mobile_base_scene.xml"

        packages.append(
            {
                "id": rel,
                "name": package_dir.name,
                "urdf": "robot/robot.urdf",
                "mvp_scene": mvp_scene,
                "scene_count": len(scenes),
                "scenes": scenes,
            }
        )

    packages.sort(key=lambda item: item["id"].lower())
    return packages


def _content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if path.suffix == ".urdf":
        return "application/xml; charset=utf-8"
    if path.suffix == ".xml":
        return "application/xml; charset=utf-8"
    if path.suffix in {".js", ".mjs"}:
        return "text/javascript; charset=utf-8"
    return "application/octet-stream"


def _safe_relative_path(raw: str) -> Path | None:
    cleaned = unquote(raw).replace("\\", "/").lstrip("/")
    if not cleaned or ".." in cleaned.split("/"):
        return None
    return Path(cleaned)


class ViewerHandler(BaseHTTPRequestHandler):
    build_root: Path = DEFAULT_BUILD_ROOT
    viewer_root: Path = VIEWER_ROOT

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/":
            return self._send_file(self.viewer_root / "index.html")

        if route == "/api/packages":
            payload = {"build_root": str(self.build_root.resolve()), "packages": find_packages(self.build_root)}
            return self._send_json(payload)

        if route.startswith("/viewer/"):
            rel = _safe_relative_path(route.removeprefix("/viewer/"))
            if rel is None:
                return self._send_error(HTTPStatus.BAD_REQUEST, "Invalid viewer path.")
            target = (self.viewer_root / rel).resolve()
            if self.viewer_root.resolve() not in [target, *target.parents]:
                return self._send_error(HTTPStatus.FORBIDDEN, "Outside viewer root.")
            if not target.exists() or target.is_dir():
                return self._send_error(HTTPStatus.NOT_FOUND, "Viewer asset not found.")
            return self._send_file(target)

        if route.startswith("/package/"):
            rel = _safe_relative_path(route.removeprefix("/package/"))
            if rel is None:
                return self._send_error(HTTPStatus.BAD_REQUEST, "Invalid package path.")
            build_root = self.build_root.resolve()
            target = (build_root / rel).resolve()
            if build_root not in [target, *target.parents]:
                return self._send_error(HTTPStatus.FORBIDDEN, "Outside build root.")
            if not target.exists() or target.is_dir():
                return self._send_error(HTTPStatus.NOT_FOUND, "Package file not found.")
            return self._send_file(target)

        return self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        self._send_bytes(data, _content_type(path))

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload, indent=2).encode("utf-8"), "application/json; charset=utf-8", status)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        message = format % args
        if re.search(r"GET /(viewer/|api/|package/)", message):
            return
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Virturoid 3D URDF / scene viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--build-root", default=str(DEFAULT_BUILD_ROOT))
    args = parser.parse_args()

    build_root = Path(args.build_root).resolve()
    build_root.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.daemon_threads = True
    ViewerHandler.build_root = build_root
    ViewerHandler.viewer_root = VIEWER_ROOT

    url = f"http://{args.host}:{args.port}"
    print(f"Virturoid 3D viewer running at {url}")
    print(f"Serving packages from: {build_root}")
    print("Open the URL in your browser, pick a build package, and switch between Robot (URDF) and Scene (MuJoCo).")
    server.serve_forever()


if __name__ == "__main__":
    main()
