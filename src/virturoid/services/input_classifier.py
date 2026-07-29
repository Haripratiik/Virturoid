"""Whole-project artifact classifier + Project Graph — Phase 2 of the Input Ingestion Research Plan.

The enterprise promise is "drop in what you have; Virturoid tells you what it found". This service scans a
folder or a zip, classifies every file into the plan's categories (robot models, meshes, CAD, ROS, controllers/
policies, logs/datasets, docs, BOM, config), checksums each one, and produces a Project Graph summary: recognized
vs unrecognized files, category counts, the first runnable simulation target, and blocking issues. It reuses the
Phase-0 :class:`InputBundle` / :class:`InputArtifact` records and depends only on the standard library
(AGENTS.md: deterministic, backend-free, reproducible via checksums).
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile

from virturoid.schemas.input_bundle import InputArtifact, InputBundle, ParseStatus

# extension -> (artifact_type, category). Category is the coarse Project-Graph bucket.
_EXT_MAP: dict[str, tuple[str, str]] = {
    ".urdf": ("robot_model", "model"), ".xacro": ("robot_model", "model"), ".mjcf": ("robot_model", "model"),
    ".sdf": ("world_or_model", "model"), ".usd": ("robot_model", "model"), ".usda": ("robot_model", "model"),
    ".usdc": ("robot_model", "model"), ".usdz": ("robot_model", "model"), ".xml": ("xml_model", "model"),
    ".stl": ("mesh", "mesh"), ".obj": ("mesh", "mesh"), ".dae": ("mesh", "mesh"), ".glb": ("mesh", "mesh"),
    ".fbx": ("mesh", "mesh"), ".ply": ("mesh", "mesh"),
    ".step": ("cad", "cad"), ".stp": ("cad", "cad"), ".iges": ("cad", "cad"), ".igs": ("cad", "cad"),
    ".sldprt": ("cad", "cad"), ".sldasm": ("cad", "cad"), ".f3d": ("cad", "cad"),
    ".onnx": ("policy", "policy"), ".pt": ("policy", "policy"), ".pth": ("policy", "policy"),
    ".jit": ("policy", "policy"), ".npz": ("policy", "policy"), ".ckpt": ("policy", "policy"),
    ".safetensors": ("policy", "policy"),
    ".mcap": ("log", "log"), ".bag": ("log", "log"), ".db3": ("log", "log"),
    ".hdf5": ("dataset", "dataset"), ".h5": ("dataset", "dataset"), ".parquet": ("dataset", "dataset"),
    ".mp4": ("video", "log"), ".mkv": ("video", "log"), ".avi": ("video", "log"),
    ".md": ("doc", "doc"), ".pdf": ("doc", "doc"), ".txt": ("doc", "doc"), ".rst": ("doc", "doc"),
    ".csv": ("bom", "bom"), ".xlsx": ("bom", "bom"), ".xls": ("bom", "bom"),
    ".yaml": ("config", "config"), ".yml": ("config", "config"), ".json": ("config", "config"),
    ".toml": ("config", "config"),
    ".py": ("script", "controller"), ".launch": ("ros_launch", "ros"),
}

IGNORE_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "__MACOSX",
})
IGNORE_FILES = frozenset({".DS_Store", "Thumbs.db"})


def classify_name(name: str) -> tuple[str, str, bool]:
    """Classify a file *by name* -> (artifact_type, category, recognized). Filename cues beat extension."""
    base = os.path.basename(name)
    lowered = base.lower()
    ext = os.path.splitext(lowered)[1]

    # Filename-specific ROS/BOM cues that override the raw extension.
    if lowered == "package.xml":
        return "ros_package", "ros", True
    if lowered.endswith(".launch.py") or lowered.endswith(".launch.xml"):
        return "ros_launch", "ros", True
    if "ros2_control" in lowered and ext in {".xacro", ".urdf", ".xml", ".yaml", ".yml"}:
        return "ros_control", "ros", True
    if re.search(r"\bbom\b|bill_of_materials|parts_list", lowered) and ext in {".csv", ".xlsx", ".json", ".yaml", ".yml"}:
        return "bom", "bom", True
    if "controller" in lowered and ext == ".yaml":
        return "controller_config", "ros", True

    if ext in _EXT_MAP:
        artifact_type, category = _EXT_MAP[ext]
        return artifact_type, category, True
    return "unknown", "unknown", False


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_") or "project"


_XML_ROOTS = (("<mujoco", "robot_model"), ("<robot", "robot_model"), ("<sdf", "world_or_model"))


def sniff_xml_model(path: str) -> str | None:
    """Return the artifact type for a ``.xml`` whose ROOT ELEMENT says it is a robot description, else None.

    ``.xml`` is genuinely ambiguous (configs, launch files, ROS manifests), so the extension map parks it at the
    non-committal ``xml_model`` -- but ``project_graph_summary`` only accepts ``robot_model``/``world_or_model``
    as a sim target, so a real MJCF was scanned, recognized, and then silently excluded. That mattered because
    MuJoCo models are conventionally shipped as ``.xml``, not ``.mjcf`` -- the whole MuJoCo Menagerie is. Measured
    on Menagerie's unitree_go2 folder: robot_models=[], "No robot description found", lane_used=none, and
    ingest_project then synthesized a DEFAULT body and reported "Ready to edit/verify" -- a 67 kg humanoid
    standing in for a 15 kg quadruped, with no warning that the customer's own robot had never been read.

    Deciding by CONTENT keeps package.xml/launch files where they belong while letting real models through.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8", errors="ignore").lower()
    except OSError:
        return None
    for token, artifact_type in _XML_ROOTS:
        if token in head:
            return artifact_type
    return None


def _make_artifact(index: int, rel_path: str, size: int | None, checksum: str | None,
                   full_path: str | None = None) -> InputArtifact:
    artifact_type, category, recognized = classify_name(rel_path)
    if artifact_type == "xml_model" and full_path:
        sniffed = sniff_xml_model(full_path)
        if sniffed:
            artifact_type, category, recognized = sniffed, "model", True
    return InputArtifact(
        id=f"art_{index:04d}",
        artifact_type=artifact_type,
        media_type=category,
        checksum=checksum,
        size_bytes=size,
        parser="extension_classifier",
        parse_status=ParseStatus.OK if recognized else ParseStatus.UNRECOGNIZED,
        extracted_refs=[rel_path.replace(os.sep, "/")],
    )


def scan_folder(root: str, *, bundle_id: str | None = None, max_files: int = 20000) -> InputBundle:
    """Walk a project folder and return an :class:`InputBundle` of classified, checksummed artifacts.

    Skips VCS/build/junk directories and dotfiles; unrecognized files are kept with an ``UNRECOGNIZED`` status
    so they surface in the dashboard rather than vanishing.
    """
    root = os.path.abspath(root)
    artifacts: list[InputArtifact] = []
    index = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            if filename in IGNORE_FILES or filename.startswith("."):
                continue
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root)
            try:
                size: int | None = os.path.getsize(full)
            except OSError:
                size = None
            artifacts.append(_make_artifact(index, rel, size, _sha256_file(full), full_path=full))
            index += 1
            if index >= max_files:
                return InputBundle(id=bundle_id or f"bundle_{_slug(os.path.basename(root))}",
                                   source_path=root, root_folder=root, artifacts=artifacts)
    return InputBundle(id=bundle_id or f"bundle_{_slug(os.path.basename(root))}",
                       source_path=root, root_folder=root, artifacts=artifacts)


def scan_zip(zip_path: str, *, bundle_id: str | None = None, max_files: int = 20000) -> InputBundle:
    """Classify the entries of a zip *without extracting* it (checksums computed from entry bytes)."""
    artifacts: list[InputArtifact] = []
    index = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            name = info.filename
            if info.is_dir():
                continue
            base = os.path.basename(name)
            if base in IGNORE_FILES or base.startswith(".") or any(
                part in IGNORE_DIRS for part in name.split("/")
            ):
                continue
            try:
                data = archive.read(info)
                checksum: str | None = hashlib.sha256(data).hexdigest()
                size: int | None = len(data)
            except (OSError, KeyError, zipfile.BadZipFile):
                checksum, size = None, info.file_size
            artifacts.append(_make_artifact(index, name, size, checksum))
            index += 1
            if index >= max_files:
                break
    return InputBundle(id=bundle_id or f"bundle_{_slug(os.path.basename(zip_path))}",
                       source_path=os.path.abspath(zip_path), root_folder=None, artifacts=artifacts)


def project_graph_summary(bundle: InputBundle) -> dict:
    """The migration dashboard: what was recognized, by category, the first runnable sim target, and blockers."""
    by_category: dict[str, int] = {}
    recognized = 0
    unrecognized = 0
    robot_models: list[str] = []
    for artifact in bundle.artifacts:
        category = artifact.media_type or "unknown"
        by_category[category] = by_category.get(category, 0) + 1
        if artifact.parse_status == ParseStatus.UNRECOGNIZED:
            unrecognized += 1
        else:
            recognized += 1
        if artifact.artifact_type in {"robot_model", "world_or_model"} \
                and artifact.parse_status != ParseStatus.UNRECOGNIZED:
            robot_models.append(artifact.extracted_refs[0] if artifact.extracted_refs else artifact.id)

    blockers: list[str] = []
    if not robot_models:
        blockers.append(
            "No robot description (URDF/MJCF/SDF/USD) found — cannot build a simulation-ready twin.")

    return {
        "bundle_id": bundle.id,
        "source_path": bundle.source_path,
        "total_files": len(bundle.artifacts),
        "recognized": recognized,
        "unrecognized": unrecognized,
        "by_category": by_category,
        "robot_models": robot_models,
        "first_runnable_sim_target": robot_models[0] if robot_models else None,
        "has_ros": by_category.get("ros", 0) > 0,
        "has_policies": by_category.get("policy", 0) > 0,
        "has_logs": by_category.get("log", 0) + by_category.get("dataset", 0) > 0,
        "has_cad": by_category.get("cad", 0) > 0,
        "has_bom": by_category.get("bom", 0) > 0,
        "blockers": blockers,
    }
