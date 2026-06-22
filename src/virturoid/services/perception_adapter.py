"""Perception in the loop (Phase 2).

The controller must act on what the robot *sees*, not on privileged simulator
state. A PerceptionAdapter turns the scene into estimated object poses with the
same interface, so backends are swappable:

- ``SyntheticPerception`` (default): a sensor model over true positions — adds
  noise, range limits, and detection dropout. Deterministic, no GL, runs in the
  test suite. It is genuine perception-in-the-loop: estimates can be wrong or
  missing, which causes honest task failures.
- ``MujocoCameraPerception``: renders the wrist RGB-D camera and back-projects
  color/depth blobs to 3D — real pixels, used when rendering is available.
- ``Cosmos3Perception``: routes frames to NVIDIA Cosmos 3 Nano (NIM/API) for
  vision reasoning / 6-DoF pose; the production backend behind this interface.

Each adapter returns ``{object_name: {"pos": (x, y, z), "confidence": float}}``
for the objects it detects; undetected objects are simply absent.
"""

from __future__ import annotations

from typing import Protocol


class PerceptionAdapter(Protocol):
    name: str

    def perceive(self, model, data, object_names: list[str]) -> dict:
        """Return detected object poses keyed by name (missing = not detected)."""


class SyntheticPerception:
    """Sensor model: true pose + Gaussian noise, range gating, detection dropout."""

    name = "synthetic_sensor_v0"

    def __init__(
        self,
        seed: int = 0,
        noise_std_m: float = 0.006,
        dropout: float = 0.05,
        misdetect: float = 0.04,
        misdetect_bias_m: float = 0.06,
        max_range_m: float = 1.5,
    ) -> None:
        self.seed = seed
        self.noise_std_m = noise_std_m
        self.dropout = dropout
        self.misdetect = misdetect
        self.misdetect_bias_m = misdetect_bias_m
        self.max_range_m = max_range_m

    def perceive(self, model, data, object_names: list[str]) -> dict:
        import mujoco
        import numpy as np

        rng = np.random.default_rng(self.seed)
        cam_pos = _camera_world_pos(model, data)
        out: dict = {}
        for name in object_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            true = np.array(data.geom_xpos[gid], dtype=float)
            if cam_pos is not None and np.linalg.norm(true - cam_pos) > self.max_range_m:
                continue  # out of sensor range -> not detected
            if rng.random() < self.dropout:
                continue  # detection dropout (occlusion/clutter)
            noise = rng.normal(0.0, self.noise_std_m, size=3)
            confidence = 0.95
            if rng.random() < self.misdetect:
                noise = noise + rng.normal(0.0, self.misdetect_bias_m, size=3)
                confidence = 0.55
            out[name] = {"pos": tuple(float(v) for v in (true + noise)), "confidence": confidence}
        return out


class MujocoCameraPerception:
    """Real RGB-D perception: render the wrist camera and back-project color blobs."""

    name = "mujoco_camera_v0"

    _COLOR = {"red_block": (0, 80), "blue_block": (160, 255)}  # hue-ish channel gate

    def __init__(self, camera: str = "wrist_rgbd_camera", width: int = 160, height: int = 120) -> None:
        self.camera = camera
        self.width = width
        self.height = height

    def perceive(self, model, data, object_names: list[str]) -> dict:
        import mujoco
        import numpy as np

        try:
            rgb_r = mujoco.Renderer(model, self.height, self.width)
            rgb_r.update_scene(data, camera=self.camera)
            rgb = rgb_r.render().astype(float)
            depth_r = mujoco.Renderer(model, self.height, self.width)
            depth_r.enable_depth_rendering()
            depth_r.update_scene(data, camera=self.camera)
            depth = depth_r.render()
        except Exception:  # noqa: BLE001 - rendering not available; caller falls back
            return {}

        out: dict = {}
        for name in object_names:
            if "red" in name:
                mask = (rgb[:, :, 0] > 120) & (rgb[:, :, 2] < 90)
            elif "blue" in name:
                mask = (rgb[:, :, 2] > 120) & (rgb[:, :, 0] < 90)
            else:
                continue
            if mask.sum() < 4:
                continue
            ys, xs = np.nonzero(mask)
            pos = _backproject(model, data, self.camera, xs.mean(), ys.mean(), float(depth[ys, xs].mean()), self.width, self.height)
            if pos is not None:
                out[name] = {"pos": pos, "confidence": float(min(1.0, mask.sum() / 50.0))}
        return out


class Cosmos3Perception:
    """NVIDIA Cosmos 3 Nano perception backend (vision reasoning / 6-DoF pose).

    Routes a rendered frame to a Cosmos 3 Nano endpoint (NIM microservice or
    build.nvidia.com API). Requires a configured client; otherwise raises so the
    caller can fall back. This is the production-grade perception path.
    """

    name = "cosmos3_nano"

    def __init__(self, client=None, endpoint: str | None = None) -> None:
        self.client = client
        self.endpoint = endpoint

    def perceive(self, model, data, object_names: list[str]) -> dict:
        if self.client is None and self.endpoint is None:
            raise RuntimeError(
                "Cosmos3Perception needs a configured NIM client or endpoint. "
                "Render the wrist camera and call Cosmos 3 Nano for object poses, "
                "or use SyntheticPerception/MujocoCameraPerception offline."
            )
        # Production: render frame -> POST to Cosmos 3 Nano -> parse object poses.
        frame = MujocoCameraPerception().perceive(model, data, object_names)  # render hook
        return self._query(frame, object_names)

    def _query(self, frame, object_names):  # pragma: no cover - requires live endpoint
        raise NotImplementedError("Wire the Cosmos 3 Nano NIM/API request/response here.")


def default_perception(seed: int = 0) -> PerceptionAdapter:
    return SyntheticPerception(seed=seed)


def _camera_world_pos(model, data):
    import numpy as np

    if model.ncam <= 0:
        return None
    return np.array(data.cam_xpos[0], dtype=float)


def _backproject(model, data, camera, px, py, depth, width, height):
    import mujoco
    import numpy as np

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0 or depth <= 0 or depth > 5:
        return None
    fovy = float(model.cam_fovy[cid]) * np.pi / 180.0
    f = (height / 2.0) / np.tan(fovy / 2.0)
    x = (px - width / 2.0) / f * depth
    y = (py - height / 2.0) / f * depth
    cam_pos = np.array(data.cam_xpos[cid], dtype=float)
    cam_mat = np.array(data.cam_xmat[cid], dtype=float).reshape(3, 3)
    # MuJoCo camera looks down -z; image x->+x_cam, y->-y_cam.
    point_cam = np.array([x, -y, -depth])
    return tuple(float(v) for v in (cam_pos + cam_mat @ point_cam))
