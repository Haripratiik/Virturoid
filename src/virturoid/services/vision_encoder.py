"""Tiny vision encoder + camera pipeline -- the 'small CV model' for visual perception.

Deliberately TINY, following the Squint recipe (Fast Visual RL for Sim-to-Real, 2026): a 2-conv CNN over a
16x16 RGB image. Small is the whole point -- a tiny encoder lets the GPU render + encode thousands of parallel
envs, so we can train far more (a ResNet/ViT/VLM backbone would cut env count 10-100x). NumPy forward here for
CPU-first validation; the SAME tiny shape ports straight to a Flax/torch encoder for GPU vision RL with
Madrona-MJX batch rendering, and deploys to the robot's Jetson Orin (already in the BOM). NVIDIA's heavy
foundation/VLM models are the DEPLOY target, not the training encoder. See [[perception-next-keystone]].
"""
from __future__ import annotations

CAM_RENDER_PX = 128   # render high-ish, then area-downsample (anti-alias) to ENCODER_PX -- the Squint trick
ENCODER_PX = 16       # the tiny encoder input resolution


def render_camera(model, data, camera="robot_cam", *, px: int = CAM_RENDER_PX):
    """Render ``camera``'s RGB view as a (px, px, 3) uint8 image. Requires a GL context (works on this box)."""
    import mujoco

    renderer = mujoco.Renderer(model, height=px, width=px)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    renderer.close()
    return img


def to_encoder_input(img, *, px: int = ENCODER_PX):
    """Area-downsample an (H, W, 3) uint8 image to (px, px, 3) float in [0, 1] -- the encoder's tiny input."""
    import numpy as np

    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    fh, fw = h // px, w // px
    img = img[: fh * px, : fw * px].reshape(px, fh, px, fw, 3).mean(axis=(1, 3))   # block-average pooling
    return img / 255.0


def _conv2d_s2_relu(x, weight, bias):
    """3x3 stride-2 'valid' convolution + ReLU. x (Cin, H, W) -> (Cout, (H-3)//2+1, (W-3)//2+1). NumPy; tiny."""
    import numpy as np

    cin, h, w = x.shape
    cout = weight.shape[0]
    ho, wo = (h - 3) // 2 + 1, (w - 3) // 2 + 1
    out = np.empty((cout, ho, wo), dtype=np.float32)
    for i in range(ho):
        for j in range(wo):
            patch = x[:, i * 2:i * 2 + 3, j * 2:j * 2 + 3]
            out[:, i, j] = np.tensordot(weight, patch, axes=([1, 2, 3], [0, 1, 2])) + bias
    return np.maximum(out, 0.0)


class TinyVisionEncoder:
    """2-conv CNN over a 16x16 RGB image -> a feature vector (the Squint-size visual encoder). Shared between
    actor and critic in RL. Pure-NumPy forward (CPU validation); the weight shapes mirror a Flax/torch CNN so a
    GPU-trained encoder loads straight in."""

    def __init__(self, *, c1: int = 16, c2: int = 32, feat: int = 64, seed: int = 0):
        import numpy as np

        rng = np.random.default_rng(seed)
        self.c1, self.c2, self.feat = c1, c2, feat
        self.W1 = rng.normal(0, 0.1, (c1, 3, 3, 3)).astype(np.float32); self.b1 = np.zeros(c1, np.float32)
        self.W2 = rng.normal(0, 0.1, (c2, c1, 3, 3)).astype(np.float32); self.b2 = np.zeros(c2, np.float32)
        # 16 ->(s2)-> 7 ->(s2)-> 3, so c2 x 3 x 3 features into the projection
        self.Wf = (rng.normal(0, 0.1, (c2 * 3 * 3, feat)) * 0.1).astype(np.float32); self.bf = np.zeros(feat, np.float32)

    def encode(self, enc_input):
        """(16, 16, 3) float image in [0, 1] -> (feat,) feature vector in [-1, 1]."""
        import numpy as np

        x = np.asarray(enc_input, dtype=np.float32).transpose(2, 0, 1)   # (3, 16, 16)
        x = _conv2d_s2_relu(x, self.W1, self.b1)                          # (c1, 7, 7)
        x = _conv2d_s2_relu(x, self.W2, self.b2)                          # (c2, 3, 3)
        return np.tanh(x.reshape(-1) @ self.Wf + self.bf)                 # (feat,)

    @property
    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.Wf.size + self.bf.size)
