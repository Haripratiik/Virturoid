"""Inbound dataset/log importer — Input Ingestion plan, Phase 5.

Recovers episode counts, rates, modalities, and candidate observation/action specs from an imported
demonstration/log dataset so prior data can seed training (LeRobot lesson: datasets are first-class input).
Grounded where the standard library / available deps allow, honest where they do not:

  * virturoid_npz   — the repo's own demonstration dataset (index.json + npz): REAL episode/obs/action shapes.
  * lerobot         — meta/info.json (fps, features, totals) + optional parquet column schema (pyarrow): REAL.
  * robomimic_hdf5  — real shapes if h5py is present; else format-detected with a "install h5py" note.
  * mcap/rosbag/db3 — magic-byte format detection + a note that full topic extraction needs the mcap/rosbags lib.

Deterministic, local-only (no network), standard-library core.
"""

from __future__ import annotations

import json
import os

from virturoid.schemas.dataset_import import DatasetImportSpec

# Leading magic bytes -> format id.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89HDF\r\n\x1a\n", "robomimic_hdf5"),
    (b"\x89MCAP", "mcap"),
    (b"#ROSBAG", "rosbag"),
    (b"SQLite format 3\x00", "ros2_db3"),
    (b"PAR1", "lerobot"),
)


def sniff_format(path: str) -> str:
    """Identify a dataset by extension, then by leading magic bytes (extension wins when unambiguous)."""
    lowered = path.lower()
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "meta", "info.json")):
            return "lerobot"
        if os.path.exists(os.path.join(path, "index.json")):
            return "virturoid_npz"
        return "unknown"
    ext = os.path.splitext(lowered)[1]
    by_ext = {".hdf5": "robomimic_hdf5", ".h5": "robomimic_hdf5", ".mcap": "mcap",
              ".bag": "rosbag", ".db3": "ros2_db3", ".parquet": "lerobot", ".json": None}
    if ext in by_ext and by_ext[ext]:
        return by_ext[ext]
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
        for magic, fmt in _MAGIC:
            if head.startswith(magic):
                return fmt
    except OSError:
        pass
    return "unknown"


def _import_virturoid_npz(path: str, spec_id: str) -> DatasetImportSpec:
    import numpy as np

    index_path = path if path.endswith("index.json") else os.path.join(path, "index.json")
    root = os.path.dirname(index_path)
    index = json.loads(open(index_path, encoding="utf-8").read())
    total = sum(int(ep.get("length", 0)) for ep in index)
    obs_dim = index[0].get("obs_dim") if index else None
    act_dim = index[0].get("action_dim") if index else None
    obs_keys: list[str] = []
    if index:
        try:
            arr = np.load(os.path.join(root, index[0]["uri"]))
            obs_keys = [k for k in arr.files if k not in ("dones",)]
            if "actions" in arr.files:
                act_dim = int(arr["actions"].shape[1])
        except Exception:  # noqa: BLE001 - the index still gives counts
            pass
    return DatasetImportSpec(
        id=spec_id, source=path, format="virturoid_npz", episodes=len(index), total_frames=total,
        modalities=["proprioception", "action"], candidate_observation_keys=obs_keys or ["obs"],
        candidate_action_dim=int(act_dim) if act_dim else None,
        warnings=[] if index else ["empty dataset index"])


def _import_lerobot(path: str, spec_id: str) -> DatasetImportSpec:
    warnings: list[str] = []
    info_path = (os.path.join(path, "meta", "info.json") if os.path.isdir(path)
                 else path if path.endswith(".json") else None)
    episodes = frames = 0
    fps = None
    modalities: list[str] = []
    obs_keys: list[str] = []
    act_dim = None
    if info_path and os.path.exists(info_path):
        info = json.loads(open(info_path, encoding="utf-8").read())
        fps = info.get("fps")
        episodes = int(info.get("total_episodes", 0) or 0)
        frames = int(info.get("total_frames", 0) or 0)
        features = info.get("features", {}) or {}
        for name, feat in features.items():
            if name.startswith("observation"):
                obs_keys.append(name)
                modalities.append(name.split(".")[-1])
            if name == "action" and isinstance(feat, dict):
                shape = feat.get("shape") or []
                act_dim = int(shape[0]) if shape else None
    elif path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            names = list(schema.names)
            obs_keys = [n for n in names if n.startswith("observation")]
            modalities = [n.split(".")[-1] for n in obs_keys]
            frames = pq.read_metadata(path).num_rows
            episodes = 1
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"parquet read failed: {exc}")
    else:
        warnings.append("no meta/info.json found for the LeRobot dataset.")
    return DatasetImportSpec(
        id=spec_id, source=path, format="lerobot", episodes=episodes, total_frames=frames,
        rate_hz=float(fps) if fps else None, modalities=sorted(set(modalities)),
        candidate_observation_keys=obs_keys, candidate_action_dim=act_dim, warnings=warnings)


def _import_robomimic_hdf5(path: str, spec_id: str) -> DatasetImportSpec:
    try:
        import h5py
    except ImportError:
        return DatasetImportSpec(
            id=spec_id, source=path, format="robomimic_hdf5",
            warnings=["h5py not installed; install it to extract episode/obs/action shapes."])
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            return DatasetImportSpec(id=spec_id, source=path, format="robomimic_hdf5",
                                     warnings=["no /data group (not a robomimic dataset?)."])
        demos = list(data.keys())
        first = data[demos[0]] if demos else None
        act_dim = None
        obs_keys: list[str] = []
        frames = 0
        for name in demos:
            g = data[name]
            if "actions" in g:
                frames += int(g["actions"].shape[0])
        if first is not None:
            if "actions" in first:
                act_dim = int(first["actions"].shape[1]) if first["actions"].ndim > 1 else 1
            if "obs" in first and hasattr(first["obs"], "keys"):
                obs_keys = list(first["obs"].keys())
            elif "obs" in first:
                obs_keys = ["obs"]
    return DatasetImportSpec(
        id=spec_id, source=path, format="robomimic_hdf5", episodes=len(demos), total_frames=frames,
        modalities=["proprioception"], candidate_observation_keys=obs_keys, candidate_action_dim=act_dim)


def _import_log(path: str, fmt: str, spec_id: str) -> DatasetImportSpec:
    """MCAP / ROS bag / ROS2 db3: detect the format; full topic/rate extraction needs the mcap/rosbags lib."""
    note = {
        "mcap": "MCAP log detected; install the 'mcap' reader to extract topics/rates/schemas.",
        "rosbag": "ROS 1 bag detected; install 'rosbags' to extract topics/rates/TF.",
        "ros2_db3": "ROS 2 db3 (sqlite) bag detected; install 'rosbags' to extract topics/rates/TF.",
    }.get(fmt, "log detected.")
    return DatasetImportSpec(id=spec_id, source=path, format=fmt, warnings=[note])


def import_dataset(path: str) -> DatasetImportSpec:
    """Import a demonstration/log dataset (file or directory) into a :class:`DatasetImportSpec`."""
    spec_id = "ds_" + os.path.basename(path.rstrip("/\\")).lower().replace(".", "_")
    if not os.path.exists(path):
        return DatasetImportSpec(id=spec_id, source=path, warnings=[f"path not found: {path}"])
    fmt = sniff_format(path)
    if fmt == "virturoid_npz":
        return _import_virturoid_npz(path, spec_id)
    if fmt == "lerobot":
        return _import_lerobot(path, spec_id)
    if fmt == "robomimic_hdf5":
        return _import_robomimic_hdf5(path, spec_id)
    if fmt in ("mcap", "rosbag", "ros2_db3"):
        return _import_log(path, fmt, spec_id)
    return DatasetImportSpec(id=spec_id, source=path, format="unknown",
                             warnings=["could not identify the dataset format (extension + magic bytes)."])
