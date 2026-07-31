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


# ---------------------------------------------------------------------------------------------------------
# WHICH FILE IS *THE ROBOT*?  (the model picker)
#
# ``first_runnable_sim_target`` used to be ``robot_models[0]`` -- directory-walk order, i.e. effectively
# alphabetical. Measured over 20 MuJoCo Menagerie packages through the real ingest path, that got 11 wrong:
#     franka_emika_panda    -> hand.xml       (the bare GRIPPER: 3 bodies / 2 DOF, vs panda.xml's 12 / 8)
#     hello_robot_stretch_3 -> scene.xml      (the robot PLUS a 1.2 m, 46.7 kg table welded on as robot links)
#     boston_dynamics_spot  -> scene.xml      (a floor-only wrapper -> import fails -> Spot silently replaced)
#     shadow_hand           -> keyframes.xml  (an include fragment -> import fails -> replaced by a generic arm)
#     ... plus ur5e, skydio_x2, pal_talos, booster_t1, stanford_tidybot, trossen_vx300s, wonik_allegro
#
# So we SCORE the candidates on evidence instead. The weights below are deliberately ordered:
#
#   1. STRUCTURE first, because it is the only signal that is never ambiguous. An MJCF whose root is <mujoco>
#      but which declares no <body> is an INCLUDE FRAGMENT -- keyframes/assets/defaults/actuator blocks meant to
#      be pulled into someone else's worldbody. ``robot_import.import_robot`` already REFUSES exactly this file
#      (see its "AN INCLUDE FRAGMENT IS NOT A MODEL" guard), so a picker that hands one over is guaranteeing a
#      failed import. We drop them from the candidate set entirely, which also removes floor-only scene.xml.
#      For a file that DOES declare bodies, ``<include>`` gives a direction: a scene <include>s the robot and
#      then adds props, so the INCLUDEE outranks the INCLUDER.
#
#   2. NAME second, because the packaging convention ("spot.xml lives in boston_dynamics_spot/") is what
#      actually identifies WHICH robot a multi-model folder is about. It is what separates spot.xml from
#      spot_arm.xml and panda.xml from panda_nohand.xml -- two pairs where both files are real, importable
#      robots and no amount of compiling tells you which one the customer means.
#
#   3. COMPILED COUNTS last, and CAPPED. A real MuJoCo compile is the highest-confidence evidence that a file
#      IS a robot at all (2 DOF vs 8 DOF; 0 movable joints vs 32), and that is how we use it: as a gate plus a
#      small tie-break. It is a BAD signal for *which* robot, and using it raw would re-introduce the bug from
#      the other side -- stretch_3/scene.xml has MORE bodies than stretch.xml (it owns the table), and
#      spot_arm.xml has MORE actuators than spot.xml. Hence min(actuators, 24) * 0.25, which can never overturn
#      a name or structure verdict, only order candidates the earlier evidence left tied.
#
# Compiling is opt-in (``compile_probe``): this module is stdlib-only and ``inspect_project_bundle`` advertises
# itself as metadata-only, so the probe is injected by ``ingest_project``, which already compiles models anyway.
# It is also only run on candidates within ``PROBE_MARGIN`` of the leader, so the common single-model folder
# costs zero compiles.
# ---------------------------------------------------------------------------------------------------------
_INCLUDE_RE = re.compile(r"""<include\b[^>]*\bfile\s*=\s*["']([^"']+)["']""")
_BODY_RE = re.compile(r"<body\b")
# FIRST element wins, not "does <mujoco> appear anywhere". A real URDF may legally carry a <mujoco><compiler/>
# EXTENSION block -- model_import._inject_mujoco_compiler exists precisely to add one and to "respect a
# customer-provided <mujoco> block" -- and a URDF declares <link>, never <body>. Matching on the substring
# would therefore read the customer's own URDF as a bodyless MJCF fragment and discard it.
_ROOT_ELEM_RE = re.compile(r"<(mujoco|robot|sdf)\b", re.I)
_PLANE_RE = re.compile(r"""type\s*=\s*["']plane["']""")
_MODEL_READ_CAP = 4_000_000          # a model file bigger than this is meshes-as-text; head is enough to judge
PROBE_MARGIN = 25.0                  # only compile candidates this close to the leader (0 compiles when clear)
PROBE_TOP_K = 4
# Directories that hold a robot's SUPPORT files. A top-level robot description is never inside one of these,
# and per-part MJCF modules often are (flexiv_rizon4 ships assets/link0/link0.xml ... assets/link6/link6.xml,
# each a valid <mujoco> with bodies). Deliberately excludes urdf/ , xacro/ and description/ — the standard ROS
# layout DOES put the real robot in urdf/.
_SUPPORT_DIRS = frozenset({"assets", "asset", "meshes", "mesh", "visual", "collision", "textures", "materials"})


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _name_tokens(value: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", value.lower()) if t}


def _package_name_score(stem: str, package: str) -> tuple[float, str]:
    """How strongly does a file's stem say "I am the robot this package is named after"?"""
    s, p = _norm_name(stem), _norm_name(package)
    if not s or not p:
        return 0.0, ""
    if s == p:
        return 50.0, f"stem matches the package name {package!r} exactly"
    if len(s) >= 2 and s in p:
        return 45.0, f"stem {stem!r} is the robot named by the package {package!r}"
    if len(p) >= 2 and p in s:
        return 30.0, f"stem {stem!r} extends the package name {package!r}"
    if _name_tokens(stem) & _name_tokens(package):
        return 18.0, f"stem shares a word with the package {package!r}"
    return 0.0, ""


def _convention_score(stem: str) -> tuple[float, list[str]]:
    """Filename conventions every robot-description corpus follows (Menagerie, ROS packages, Isaac assets)."""
    low, toks = stem.lower(), _name_tokens(stem)
    score, reasons = 0.0, []
    if low.startswith("scene") or "scene" in toks:
        score -= 55.0
        reasons.append("named scene* — a world, not the robot")
    if low.startswith("keyframe") or any(t.startswith("keyframe") for t in toks):
        score -= 55.0
        reasons.append("named keyframe* — recorded poses, not the robot")
    if toks & {"test", "tests", "example", "examples", "demo", "tutorial", "sample"}:
        score -= 35.0
        reasons.append("named as a test/example/demo asset")
    if low.startswith("mjx") or "mjx" in toks:
        score -= 14.0
        reasons.append("an mjx (GPU-backend) variant of the same robot")
    if "right" in toks and "left" not in toks:
        # Pure tie-break for mirrored pairs (shadow_hand, wonik_allegro ship left_hand.xml + right_hand.xml and
        # nothing distinguishes them). Right-handed is the convention upstream `robot_descriptions` picks for
        # both; the point is only that the answer is deterministic and is still THE CUSTOMER'S robot.
        score += 1.0
    return score, reasons


def _read_model_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read(_MODEL_READ_CAP)
    except OSError:
        return None


def _model_facts(text: str | None) -> dict:
    """Structural facts a model file states about itself (no XML parser: real-world files are often malformed)."""
    if text is None:
        return {"readable": False, "mjcf": False, "fragment": False, "includes": [], "bodies": 0, "plane": False}
    root = _ROOT_ELEM_RE.search(text)
    mjcf = bool(root) and root.group(1).lower() == "mujoco"
    bodies = len(_BODY_RE.findall(text))
    return {"readable": True, "mjcf": mjcf, "bodies": bodies,
            # same rule robot_import.import_robot refuses on: a <mujoco> root that declares no <body> only
            # contributes keyframes/assets/defaults/actuators to a parent that <include>s it.
            "fragment": mjcf and bodies == 0,
            "includes": _INCLUDE_RE.findall(text), "plane": bool(_PLANE_RE.search(text))}


def _resolve_include(including_ref: str, target: str) -> str:
    """MuJoCo resolves <include file=...> relative to the INCLUDING file's directory."""
    base = os.path.dirname(including_ref.replace(os.sep, "/"))
    joined = f"{base}/{target}" if base else target
    return os.path.normpath(joined).replace(os.sep, "/")


def rank_model_candidates(refs, *, root_folder: str | None = None, compile_probe=None) -> list[dict]:
    """Rank candidate robot-description files best-first, with the evidence for each ranking attached.

    ``refs`` are bundle-relative paths. ``root_folder`` (folder scans only; a zip bundle has none) unlocks the
    structural evidence. ``compile_probe(abs_path) -> {"ok", "bodies", "actuators"} | None`` unlocks the
    compiled evidence and is only called for candidates still in contention.
    """
    refs = [r.replace(os.sep, "/") for r in refs]
    if not refs:
        return []

    facts: dict[str, dict] = {}
    for ref in refs:
        text = _read_model_text(os.path.join(root_folder, ref.replace("/", os.sep))) if root_folder else None
        facts[ref] = _model_facts(text)

    # An include fragment is not a model: it is guaranteed to fail the import it would be handed to. Dropping it
    # (rather than ranking it last) means a folder holding ONLY fragments honestly reports "no robot description
    # found" instead of nominating a file we know cannot load.
    candidates = [r for r in refs if not facts[r]["fragment"]]
    dropped = [r for r in refs if facts[r]["fragment"]]
    if len(candidates) <= 1:
        return [{"ref": r, "score": 0.0, "reasons": ["only importable candidate" if dropped else "only candidate"],
                 "actuators": None, "bodies": facts[r]["bodies"] or None} for r in candidates]
    known = set(candidates)

    # <include> gives DIRECTION, and the direction depends on which way the two files sit in the tree:
    #   * a SCENE includes a sibling ("scene.xml" -> "spot.xml") and then bolts a floor and a table on top, so
    #     the includee is the robot and the includer is the wrapper;
    #   * an ASSEMBLY ROOT includes files BELOW it ("flexiv_rizon4.xml" -> "assets/link0/link0.xml"), so the
    #     includer is the robot and the includees are its own parts.
    # Measured: without this split the picker chose flexiv_rizon4/assets/link0/link0.xml — one link of the arm.
    depth = {r: r.count("/") for r in candidates}
    wraps_of: dict[str, list[str]] = {r: [] for r in candidates}      # includes a sibling/shallower model
    parts_of: dict[str, list[str]] = {r: [] for r in candidates}      # includes files deeper in the tree
    wrapped_by: dict[str, list[str]] = {r: [] for r in candidates}
    part_of: dict[str, list[str]] = {r: [] for r in candidates}
    for ref in candidates:
        for target in facts[ref]["includes"]:
            resolved = _resolve_include(ref, target)
            if resolved not in known or resolved == ref:
                continue
            if depth[resolved] > depth[ref]:
                parts_of[ref].append(resolved)
                part_of[resolved].append(ref)
            else:
                wraps_of[ref].append(resolved)
                wrapped_by[resolved].append(ref)

    root_pkg = os.path.basename(os.path.normpath(root_folder)) if root_folder else ""
    scored: list[dict] = []
    for ref in candidates:
        f = facts[ref]
        stem = os.path.splitext(os.path.basename(ref))[0]
        parts = ref.split("/")[:-1]
        support = [p for p in parts if p.lower() in _SUPPORT_DIRS]
        score, reasons = 0.0, []

        # NAME. The immediate parent directory names the package ("spot.xml" in "boston_dynamics_spot/"), except
        # under a support dir, where it names an asset group ("link0/") and would falsely score a perfect match.
        packages = [root_pkg] if support else [p for p in (parts[-1] if parts else "", root_pkg) if p]
        best_name = max((_package_name_score(stem, p) for p in packages), default=(0.0, ""))
        if best_name[0]:
            score += best_name[0]
            reasons.append(best_name[1])
        conv, conv_reasons = _convention_score(stem)
        score += conv
        reasons.extend(conv_reasons)

        if support:
            score -= 25.0
            reasons.append(f"lives under {support[0]}/ — a support asset, not the top-level robot")
        if depth[ref]:
            score -= min(depth[ref] * 6.0, 24.0)                      # a robot's own file sits above its parts

        if wraps_of[ref]:
            score -= 40.0
            reasons.append(f"<include>s the sibling model {wraps_of[ref][0]} — a wrapper, prefer what it includes")
            if f["bodies"] or f["plane"]:
                score -= 20.0
                reasons.append("and adds its own floor/props on top of the included robot")
        if wrapped_by[ref]:
            score += 20.0
            reasons.append(f"is <include>d by {wrapped_by[ref][0]} — it is the robot that scene wraps")
        if parts_of[ref]:
            score += 10.0
            reasons.append(f"assembles {len(parts_of[ref])} part file(s) below it — the assembly root")
        if part_of[ref]:
            score -= 30.0
            reasons.append(f"is a part <include>d by {part_of[ref][0]} — a component, not the whole robot")

        scored.append({"ref": ref, "score": score, "reasons": reasons,
                       "actuators": None, "bodies": f["bodies"] or None})

    scored.sort(key=lambda c: (-c["score"], c["ref"]))

    # COMPILED EVIDENCE, only where it can still change the answer.
    if compile_probe is not None and root_folder and scored:
        lead = scored[0]["score"]
        contenders = [c for c in scored if lead - c["score"] <= PROBE_MARGIN][:PROBE_TOP_K]
        if len(contenders) > 1:
            for cand in contenders:
                probe = None
                try:
                    probe = compile_probe(os.path.join(root_folder, cand["ref"].replace("/", os.sep)))
                except Exception:  # noqa: BLE001 - a probe failure must never sink the ranking
                    probe = None
                if not probe:
                    continue
                bodies, actuators = int(probe.get("bodies") or 0), int(probe.get("actuators") or 0)
                cand["bodies"], cand["actuators"] = bodies, actuators
                if not probe.get("ok", True):
                    cand["score"] -= 35.0
                    cand["reasons"].append("does not compile in MuJoCo — not a runnable sim target")
                    continue
                if bodies <= 1:
                    cand["score"] -= 30.0
                    cand["reasons"].append(f"compiles to {bodies} body — a prop, not a robot")
                elif actuators <= 0:
                    cand["score"] -= 20.0
                    cand["reasons"].append("compiles with no movable joints — nothing to drive")
                else:
                    cand["score"] += 25.0 + min(actuators, 24) * 0.25
                    cand["reasons"].append(f"real compile: {bodies} bodies, {actuators} actuators")
            # explicit tie-break the plan asks for: equal evidence -> the one with more actuators wins
            scored.sort(key=lambda c: (-c["score"], -(c["actuators"] or 0), -(c["bodies"] or 0), c["ref"]))
    return scored


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


def project_graph_summary(bundle: InputBundle, *, compile_probe=None) -> dict:
    """The migration dashboard: what was recognized, by category, the first runnable sim target, and blockers.

    ``compile_probe`` (optional) lets a caller that already has MuJoCo loaded contribute compiled body/actuator
    counts to the model pick; see ``rank_model_candidates``. Without it the pick is static (stdlib only), which
    is what ``inspect_project_bundle`` advertises.
    """
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

    # PICK BY EVIDENCE, NOT ALPHABET (see rank_model_candidates). robot_models is returned best-first so the
    # list itself is the ranking, and model_ranking carries the reasons so the choice is auditable.
    ranking = rank_model_candidates(robot_models, root_folder=bundle.root_folder, compile_probe=compile_probe)
    robot_models = [c["ref"] for c in ranking]

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
        "model_ranking": [{"ref": c["ref"], "score": round(float(c["score"]), 2), "reasons": c["reasons"],
                           "bodies": c.get("bodies"), "actuators": c.get("actuators")} for c in ranking[:6]],
        "first_runnable_sim_target": robot_models[0] if robot_models else None,
        "has_ros": by_category.get("ros", 0) > 0,
        "has_policies": by_category.get("policy", 0) > 0,
        "has_logs": by_category.get("log", 0) + by_category.get("dataset", 0) > 0,
        "has_cad": by_category.get("cad", 0) > 0,
        "has_bom": by_category.get("bom", 0) > 0,
        "blockers": blockers,
    }
