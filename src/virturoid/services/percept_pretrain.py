"""Percept-prediction pretrain (breakthrough v5, B4 Phase B): ground the tiny vision encoder BEFORE any RL, by
teaching it to predict the PRIVILEGED rangefinder signal from the camera image. This is the teacher-student
setup's first half — the rangefinder ring (``exteroception``) is the geometry-truth teacher, the camera is the
student input, and the pretrained readout is a warm-start for the MJWarp DAgger student (Phase C, GPU).

Why it matters: a randomly-initialized encoder feeds RL noise; an encoder whose features already linearly predict
'how far is the obstacle in each direction' feeds RL a geometrically-meaningful representation, which is the
documented reason vision RL is otherwise slow and sample-hungry. We measure exactly that with a held-out R^2:
does the image carry the rangefinder geometry?

CPU + numpy: the readout is a closed-form ridge regression over the (frozen) encoder features — deterministic and
testable. Fine-tuning the conv weights and the DAgger loop are the jax/MJWarp extensions; this module owns the
DATA pipeline, the OBJECTIVE, and the METRIC so those extensions have a validated target to hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from virturoid.services.vision_encoder import TinyVisionEncoder, to_encoder_input


def raw_features(images: np.ndarray) -> np.ndarray:
    """Flattened greyscale pixels of the (already 16x16x3) images — a linearly-decodable representation. Used to
    validate the pretrain OBJECTIVE/METRIC independently of any encoder, and as the ``feature_fn`` a caller can
    pass when the frozen encoder is not yet trained."""
    imgs = np.asarray(images, dtype=np.float64)
    grey = imgs.mean(axis=-1).reshape(imgs.shape[0], -1)      # (N, 256)
    return grey


@dataclass
class PerceptPredictor:
    """Image -> rangefinder-length vector via a chosen feature representation + a closed-form ridge readout.
    ``feature_fn`` defaults to the (frozen) TinyVisionEncoder; pass :func:`raw_features` to isolate the objective
    from the encoder, or a jax-trained encoder's forward once Phase-B conv fine-tuning lands. ``fit`` solves the
    readout in closed form (deterministic), ``score`` reports held-out R^2 — 'is the geometry decodable from this
    representation'. A frozen RANDOM encoder scores ~0 here, which is the documented motivation for training it."""
    n_ranges: int
    encoder: TinyVisionEncoder = None
    feature_fn: object = None
    W: np.ndarray = None               # (feat+1, n_ranges) readout incl. bias
    ridge: float = 1e-2

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = TinyVisionEncoder(feat=48, seed=0)
        if self.feature_fn is None:
            self.feature_fn = lambda ims: np.stack(
                [np.asarray(self.encoder.encode(to_encoder_input(im))) for im in ims], axis=0)

    def _features(self, images: np.ndarray) -> np.ndarray:
        """Represent a batch of HxWx3 images as (N, d) features via ``feature_fn``, with a bias column appended."""
        feats = np.asarray(self.feature_fn(images), dtype=np.float64)
        return np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)

    def fit(self, images: np.ndarray, ranges: np.ndarray) -> "PerceptPredictor":
        """Closed-form ridge readout from encoder features to the rangefinder targets. ``images`` (N,H,W,3),
        ``ranges`` (N, n_ranges) in [0, 1] (normalized distances). W = (FᵀF + λI)⁻¹ FᵀY."""
        F = self._features(np.asarray(images, dtype=np.float32))
        Y = np.asarray(ranges, dtype=np.float64)
        A = F.T @ F + self.ridge * np.eye(F.shape[1])
        self.W = np.linalg.solve(A, F.T @ Y)
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        assert self.W is not None, "call fit() first"
        return self._features(np.asarray(images, dtype=np.float32)) @ self.W

    def score(self, images: np.ndarray, ranges: np.ndarray) -> dict:
        """Held-out fit quality: R^2 (1 - SSE/SST; 0 == no better than predicting the mean) + RMSE, over all range
        beams pooled. R^2 > 0 means the image genuinely carries the rangefinder geometry."""
        Y = np.asarray(ranges, dtype=np.float64)
        P = self.predict(images)
        sse = float(np.sum((P - Y) ** 2))
        sst = float(np.sum((Y - Y.mean(axis=0)) ** 2)) + 1e-12
        return {"r2": 1.0 - sse / sst, "rmse": float(np.sqrt(np.mean((P - Y) ** 2)))}


def pretrain(images: np.ndarray, ranges: np.ndarray, *, holdout: float = 0.25, seed: int = 0,
             ridge: float = 1e-2, feature_fn=None) -> dict:
    """Full Phase-B pretrain on a collected (images, ranges) dataset: split train/held-out, fit the readout,
    report held-out R^2 (the warm-start quality). ``feature_fn`` chooses the representation (default frozen
    encoder; pass :func:`raw_features` to validate the objective). Returns ``{"predictor","train","heldout","n"}``."""
    images = np.asarray(images, dtype=np.float32); ranges = np.asarray(ranges, dtype=np.float64)
    n = len(images)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_hold = max(1, int(n * holdout))
    te, tr = idx[:n_hold], idx[n_hold:]
    pp = PerceptPredictor(n_ranges=ranges.shape[1], ridge=ridge, feature_fn=feature_fn).fit(images[tr], ranges[tr])
    return {"predictor": pp, "train": pp.score(images[tr], ranges[tr]),
            "heldout": pp.score(images[te], ranges[te]), "n": n}


def collect_percept_dataset(model, data, poses, *, camera: str = "robot_cam", n_beams: int = 8,
                            max_range: float = 4.0, px: int = 128):
    """Render (image, rangefinder) pairs across ``poses`` for the Phase-B dataset. Requires a MuJoCo model with a
    ``camera`` and a rangefinder ring (``exteroception.inject_rangefinders``); GL-dependent, so callers gate on
    availability. Returns (images (N,16,16,3), ranges (N, n_beams) normalized to [0,1]). Kept thin: the offline
    re-render at scale is the jax/MJWarp job; this validates the pipeline shape on CPU."""
    import mujoco

    from virturoid.services.vision_encoder import render_camera
    from virturoid.services.exteroception import read_ranges

    images, ranges = [], []
    for qpos in poses:
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        img = render_camera(model, data, camera=camera, px=px)
        images.append(to_encoder_input(img))
        r = np.asarray(read_ranges(model, data), dtype=float)[:n_beams]
        ranges.append(np.clip(r / max_range, 0.0, 1.0))
    return np.asarray(images, dtype=np.float32), np.asarray(ranges, dtype=np.float64)
