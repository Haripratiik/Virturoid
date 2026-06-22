"""Behavior cloning: distill the FK-IK planner into a fast learned policy (Phase 3).

The scripted controller plans each waypoint with an expensive cross-entropy
forward-kinematics IK search. Here we treat that planner as an *expert* and learn
a feedforward policy that maps a target point (x, y, z) directly to joint targets,
trained by supervised regression on expert demonstrations. The result is a real
learned control policy that needs no per-step search at inference — the first
rung of the learned-policy ladder (BC -> RL/diffusion/VLA behind the same hook).
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.services.mujoco_runner import mujoco_available


class BCPolicy:
    """Learned target policy: (x, y, z) -> joint position targets via polynomial regression."""

    def __init__(self, weights, joint_names, ranges):
        self.weights = weights  # (n_joints, n_features)
        self.joint_names = joint_names
        self.ranges = ranges

    @staticmethod
    def features(x: float, y: float, z: float):
        return [x, y, z, x * x, y * y, z * z, x * y, x * z, y * z, 1.0]

    def predict(self, point) -> list:
        import numpy as np

        f = np.array(self.features(point[0], point[1], point[2]))
        raw = self.weights @ f
        return [float(min(max(raw[i], self.ranges[i][0], -3.14), max(self.ranges[i][1], -3.13))) for i in range(len(raw))]

    @classmethod
    def from_dict(cls, d):
        import numpy as np

        return cls(np.array(d["weights"]), d["joint_names"], [tuple(r) for r in d["ranges"]])


def train_bc_policy(package_dir: Path, samples: int = 240, seed: int = 11) -> dict:
    """Collect expert (FK-IK) demonstrations across the workspace and fit a BC policy."""
    if not mujoco_available():
        raise RuntimeError("MuJoCo is not installed; run `pip install mujoco` to train a BC policy.")

    import mujoco
    import numpy as np

    from virturoid.services.pick_place_controller import plan_joint_targets

    package_dir = Path(package_dir)
    genome = json.loads((package_dir / "robot" / "robot_genome.json").read_text(encoding="utf-8"))
    compiled = json.loads((package_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8"))
    xml_uri = compiled["scenes"][0]["mujoco_xml"]
    model = mujoco.MjModel.from_xml_path(str(package_dir / xml_uri))

    joint_names = [j["name"] for j in genome["joints"]]
    ranges = []
    for j in genome["joints"]:
        limit = j.get("limit") or {}
        ranges.append((limit.get("lower", -3.14) or -3.14, limit.get("upper", 3.14) or 3.14))

    rng = np.random.default_rng(seed)
    feats, targets = [], []
    # Cover the whole task workspace the controller visits: block grasp annulus
    # (r ~0.30-0.42) AND the bin region (r ~0.58), across grasp/place/hover heights.
    for _ in range(samples):
        r = rng.uniform(0.28, 0.60)
        theta = rng.uniform(-0.7, 0.7)
        x, y = r * np.cos(theta), r * np.sin(theta)
        z = rng.choice([0.06, 0.13, 0.20, 0.25])  # grasp, place, hover heights
        expert, dist = plan_joint_targets(model, (x, y, z), iterations=8, candidates=60, seed=int(rng.integers(0, 9999)))
        if dist < 0.12:  # keep only good demonstrations
            feats.append(BCPolicy.features(x, y, z))
            targets.append(expert)

    F = np.array(feats)
    T = np.array(targets)
    # Closed-form least squares per joint: W = (FᵀF)⁻¹ FᵀT.
    weights, *_ = np.linalg.lstsq(F, T, rcond=None)
    weights = weights.T  # (n_joints, n_features)
    residual = float(np.mean(np.abs(F @ weights.T - T)))

    policy = BCPolicy(weights, joint_names, ranges)
    artifact = {
        "id": f"bc_policy_{genome['id']}",
        "policy_type": "behavior_cloning_polynomial",
        "expert": "fk_ik_cross_entropy_planner",
        "joint_names": joint_names,
        "ranges": [list(r) for r in ranges],
        "weights": weights.tolist(),
        "feature_spec": ["x", "y", "z", "x2", "y2", "z2", "xy", "xz", "yz", "bias"],
        "demonstrations": len(feats),
        "train_joint_residual_rad": round(residual, 4),
        "notes": [
            "Learned from expert FK-IK demonstrations (behavior cloning).",
            "Maps a target point directly to joint targets with no inference-time search.",
        ],
    }
    out = package_dir / "software" / "controller" / "bc_policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # Policy card (plan §12.4).
    card = {
        "policy_id": artifact["id"],
        "type": artifact["policy_type"],
        "trained_on": f"{len(feats)} expert demonstrations",
        "compatible_robot": genome["id"],
        "inputs": "target point (x, y, z) in base frame",
        "outputs": f"{len(joint_names)} joint position targets",
        "train_joint_residual_rad": artifact["train_joint_residual_rad"],
        "limitations": "Distills the reach planner; grasp/transport logic still scripted.",
    }
    (package_dir / "software" / "controller" / "bc_policy_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    return {"policy": policy, "artifact": artifact, "card": card, "demonstrations": len(feats), "residual_rad": artifact["train_joint_residual_rad"]}
