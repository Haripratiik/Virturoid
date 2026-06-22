"""Demonstration & dataset system (startup plan §41 / §32.3 / §12.1).

Turns a competent controller's rollouts into a learning-ready expert-demonstration dataset, so the
plan's first learning type — imitation learning (§12.1) — and offline RL have data to train on. It
implements the §41.3 pipeline: collect episodes -> keep successful expert trajectories -> validate
obs/action shapes -> write a portable per-episode dataset + a §41.3 dataset card. When ``h5py`` is
present it ALSO writes a robomimic-shaped HDF5 (``data/demo_{i}/{obs,actions,rewards,dones}``, §41.4)
for direct use with existing offline-IL pipelines; without it the portable NPZ form carries the same
content (and converts trivially later). The first source is the LEARNED grasp skill (the flywheel
closing on itself: a learned policy generates the demonstrations that train the next policy).
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.demonstration import DatasetCard, DemonstrationEpisode


def record_grasp_demonstrations(skill: dict, *, params_path: str | None = None, n: int = 12,
                                seed: int = 0, only_success: bool = True) -> list[dict]:
    """Roll out the banked grasp skill over randomized in-region scenes and capture expert trajectories.

    Returns a list of episode dicts with arrays (obs, actions, rewards, robot_qpos/qvel, object_pose)
    + metadata. By default keeps only SUCCESSFUL episodes (expert demonstrations for IL).
    """
    import numpy as np

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper
    from virturoid.services.mujoco_runner import mujoco_available
    from virturoid.services.skill_evaluator import _grasp_rollout, _load_policy

    if not mujoco_available():
        raise RuntimeError("MuJoCo is required to record demonstrations.")
    params_path = params_path or skill.get("params_path")
    if not params_path or not Path(params_path).exists():
        raise FileNotFoundError(f"policy params not found: {params_path}")
    layers = _load_policy(params_path)
    gene = add_parallel_gripper(tabletop_arm_gene())
    cfg = skill.get("base_config") or {}
    region = skill.get("region") or {"x": [0.43, 0.49], "y": [-0.08, 0.08]}
    xr, yr = region["x"], region["y"]
    rng = np.random.default_rng(seed)

    episodes = []
    for i in range(n):
        gx, gy = float(rng.uniform(*xr)), float(rng.uniform(*yr))
        r = _grasp_rollout(gene, cfg, layers, gx, gy, record=True)
        if only_success and not r["success"]:
            continue
        tr = r["trajectory"]
        episodes.append({
            "episode_id": f"grasp_demo_{i:03d}", "success": r["success"],
            "scene_params": {"box_x": round(gx, 4), "box_y": round(gy, 4)},
            "task_language": "grasp the cube and lift it off the table",
            "robot_genome_id": skill.get("gene_id", gene.id), "source": "simulation",
            "return": float(np.sum(tr["rewards"])), **tr,
        })
    return episodes


def write_demonstration_dataset(episodes: list[dict], out_dir, *, robot: str, task: str,
                                obs_modalities: list[str], action_space: str,
                                domain_randomization: dict | None = None,
                                training_permission: str = "full_opt_in") -> DatasetCard:
    """Write the §41.3 dataset: per-episode NPZ + index.json + dataset_card.json (+ robomimic HDF5 if
    h5py). Validates obs/action shape consistency across episodes (§41.3 step 6). Returns the card."""
    import numpy as np

    out = Path(out_dir); (out / "episodes").mkdir(parents=True, exist_ok=True)
    if not episodes:
        raise ValueError("no demonstrations to write (no successful episodes recorded?)")

    obs_dim = int(episodes[0]["obs"].shape[1]); act_dim = int(episodes[0]["actions"].shape[1])
    index, total_T, n_success = [], 0, 0
    for ep in episodes:
        obs, act = ep["obs"], ep["actions"]
        if obs.shape[1] != obs_dim or act.shape[1] != act_dim:               # §41.3 step 6: shape validation
            raise ValueError(f"episode {ep['episode_id']} shape mismatch: obs{obs.shape} act{act.shape}")
        T = int(obs.shape[0]); total_T += T; n_success += int(ep["success"])
        uri = f"episodes/{ep['episode_id']}.npz"
        dones = np.zeros(T, dtype=np.int8); dones[-1] = 1
        np.savez(out / uri, obs=obs, actions=act, rewards=ep["rewards"], dones=dones,
                 robot_qpos=ep["robot_qpos"], robot_qvel=ep["robot_qvel"], object_pose=ep["object_pose"])
        meta = DemonstrationEpisode(
            episode_id=ep["episode_id"], length=T, success=bool(ep["success"]), obs_dim=obs_dim,
            action_dim=act_dim, task_language=ep["task_language"], scene_params=ep["scene_params"],
            robot_genome_id=ep["robot_genome_id"], source=ep["source"], uri=uri,
            return_=round(float(ep["return"]), 4))
        index.append(meta.__dict__)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    robomimic = _maybe_write_robomimic_hdf5(out, episodes)
    formats = ["virturoid_npz"] + (["robomimic_hdf5"] if robomimic else [])
    card = DatasetCard(
        id=f"demos_{task}_{robot}", robot=robot, task=task, sources=["simulation"],
        episodes=len(episodes), success_rate=round(n_success / max(1, len(episodes)), 4),
        observation_modalities=obs_modalities, action_space=action_space,
        obs_dim=obs_dim, action_dim=act_dim, total_transitions=total_T,
        known_biases=["expert is the residual-RL grasp policy (not human); in-region scenes only"],
        domain_randomization=domain_randomization or {"box_xy": "uniform over the valid region"},
        training_permission=training_permission, formats=formats, robomimic_compatible=robomimic)
    (out / "dataset_card.json").write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    return card


def load_demonstration_dataset(dataset_dir):
    """Load a written dataset back into stacked (obs, actions, rewards, dones, episode_lengths).

    The consumer-side seam: imitation-learning / offline-RL trainers read demonstrations through this,
    so the dataset is a real input to learning, not an isolated artifact.
    """
    import numpy as np

    d = Path(dataset_dir)
    index = json.loads((d / "index.json").read_text(encoding="utf-8"))
    obs, act, rew, dones, lengths = [], [], [], [], []
    for ep in index:
        arr = np.load(d / ep["uri"])
        obs.append(arr["obs"]); act.append(arr["actions"]); rew.append(arr["rewards"])
        dones.append(arr["dones"]); lengths.append(int(ep["length"]))
    return {"obs": np.concatenate(obs), "actions": np.concatenate(act),
            "rewards": np.concatenate(rew), "dones": np.concatenate(dones),
            "episode_lengths": lengths}


def fit_bc_baseline(dataset_dir, *, val_frac: float = 0.2, ridge: float = 1.0, seed: int = 0) -> dict:
    """Baseline imitation learning (§12.1): ridge regression obs→action over the demonstrations, with a
    held-out split. Proves the dataset is *learnable* (a real IL input) and reports held-out fit. This
    is a sanity/consumability baseline, not a deployable controller — torque-space BC needs a richer
    model; the point is the demonstrations → policy path works end to end.
    """
    import numpy as np

    data = load_demonstration_dataset(dataset_dir)
    X, Y = data["obs"], data["actions"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X)); X, Y = X[perm], Y[perm]
    n_val = max(1, int(len(X) * val_frac)); Xtr, Ytr, Xval, Yval = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]
    # closed-form ridge with a bias feature
    Phi = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ Ytr)
    Pval = np.hstack([Xval, np.ones((len(Xval), 1))])
    pred = Pval @ W
    mse = float(np.mean((pred - Yval) ** 2))
    var = float(np.mean((Yval - Yval.mean(axis=0)) ** 2)) or 1.0
    return {"transitions": int(len(X)), "val_transitions": int(len(Xval)),
            "val_mse": round(mse, 5), "val_r2": round(1.0 - mse / var, 4),
            "obs_dim": int(X.shape[1]), "action_dim": int(Y.shape[1])}


def _maybe_write_robomimic_hdf5(out: Path, episodes: list[dict]) -> bool:
    """Write a robomimic-shaped HDF5 if h5py is available (§41.4). Returns whether it was written."""
    try:
        import h5py
    except ImportError:
        return False
    import numpy as np

    with h5py.File(out / "demonstrations.hdf5", "w") as f:
        data = f.create_group("data")
        total = 0
        for i, ep in enumerate(episodes):
            g = data.create_group(f"demo_{i}")
            T = int(ep["obs"].shape[0]); total += T
            g.create_dataset("obs", data=np.asarray(ep["obs"], dtype=np.float32))
            g.create_dataset("actions", data=np.asarray(ep["actions"], dtype=np.float32))
            g.create_dataset("rewards", data=np.asarray(ep["rewards"], dtype=np.float32))
            dones = np.zeros(T, dtype=np.int64); dones[-1] = 1
            g.create_dataset("dones", data=dones)
            g.attrs["num_samples"] = T
        data.attrs["total"] = total
    return True
