"""§4.3 — the MimicGen-style demo factory: 1 scripted skill -> many VERIFIED demonstrations, all in sim.

The data bottleneck is the whole field's escape problem (robot action data can't be scraped). Sim removes the
two hardest costs of generating it — free resets and a programmatic success_fn — so we can REPLAY one working
scripted grasp under randomized conditions and REJECTION-SAMPLE the successes into a demonstration dataset. That
dataset is the data engine that would seed an imitation / VLA head (the answer to "where's your foundation-model
data?"). This module is the engine; the consumer (Diffusion Policy / ACT / a VLA) trains on its output.

The augmentation factor (successful demos per scripted skill) and the yield (success rate under randomization)
are the honest KPIs — both reported, neither inflated (every demo passed the real grasp success_fn).
"""

from __future__ import annotations


def generate_grasp_demos(gene, *, n: int = 48, seed: int = 0, reach=(0.34, 0.46), lateral: float = 0.12,
                         record_frames: bool = False, **grasp_kw) -> dict:
    """Replay the scripted friction grasp under randomized object pose + grasp-parameter jitter, keeping only the
    successes. Returns ``{n_attempts, n_success, yield, augmentation_x, demos}`` — ``demos`` are the verified
    demonstrations (scene + the action params that worked + the measured outcome; optionally the motion frames).
    Needs MuJoCo (the grasp eval). The arm should be a grasp-capable gene (e.g. compose_robot('grasp and lift a
    box on a table', llm=None))."""
    import numpy as np

    from virturoid.services.grasp_eval import grasp_lift_attempt
    rng = np.random.default_rng(seed)
    demos: list[dict] = []
    attempts = 0
    for _ in range(int(n)):
        gx = float(rng.uniform(reach[0], reach[1]))            # randomized object pose in the reachable zone
        gy = float(rng.uniform(-lateral, lateral))
        fclose = float(rng.uniform(0.03, 0.07))               # grasp-parameter augmentation (jaw closing depth)
        res = grasp_lift_attempt(gene, gx, gy, fclose=fclose, record_frames=record_frames, **grasp_kw)
        attempts += 1
        if res.get("success"):
            demo = {"scene": {"object_xy": [round(gx, 4), round(gy, 4)]},
                    "action_params": {"fclose": round(fclose, 4)},
                    "outcome": {"lifted_m": res.get("lifted"), "d_tcp": res.get("d_tcp"),
                                "both_contact": res.get("both_contact")}}
            if record_frames and res.get("frames"):
                demo["n_frames"] = len(res["frames"])         # the demonstrated motion length (the trajectory)
            demos.append(demo)
    n_success = len(demos)
    return {"n_attempts": attempts, "n_success": n_success,
            "yield": round(n_success / max(1, attempts), 3),
            "augmentation_x": n_success,                       # demos produced from the ONE scripted skill
            "demos": demos}


def write_grasp_dataset(gene, out_path, *, n: int = 48, seed: int = 0, **kw) -> dict:
    """Generate the demo dataset and persist it as JSON; returns the summary (without the demo list)."""
    import json
    from pathlib import Path
    ds = generate_grasp_demos(gene, n=n, seed=seed, **kw)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ds, indent=2), encoding="utf-8")
    return {k: v for k, v in ds.items() if k != "demos"} | {"dataset_path": str(p), "n_demos": len(ds["demos"])}
