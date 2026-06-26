"""Vision-guided grasp demo: train the CV to LOCATE a box, then the arm grasps the LEARNED position (not premade).

    VIRTUROID_LLM_BACKEND=off python scripts/vision_grasp_demo.py

Proof it is actually TRAINING (not handed coordinates): with an UNTRAINED encoder the arm misses; after training
the encoder reads the box's (x,y) from an overhead camera to ~1 cm and the arm grasps + lifts. The arm always
aims at the network's pixels-in -> (x,y)-out prediction, while the box really sits at the true spot. Saves a
side-view montage of the arm grasping a CV-located box. See services/vision_grasp.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from virturoid.services.vision_encoder import TinyVisionEncoder, to_jax_params  # noqa: E402
from virturoid.services.vision_grasp import (  # noqa: E402
    GX0,
    GX1,
    GY0,
    GY1,
    build_arm,
    evaluate_vision_grasp,
    gen_box_dataset,
    make_box_locator,
    train_box_locator,
    vision_grasp_once,
)
from virturoid.services.vision_nav import save_vision_montage  # noqa: E402


def main() -> int:
    gene = build_arm()
    print("[1/3] Training the tiny CV encoder to LOCATE the box from an overhead camera...")
    x, y = gen_box_dataset(gene, n=400, seed=0)
    params, test_mae, baseline_mae = train_box_locator(x, y, epochs=120)
    print(f"      locator MAE {test_mae:.4f} vs {baseline_mae:.4f} untrained (normalized xy)")

    print("[2/3] Grasping with UNTRAINED vs TRAINED perception (same 10 boxes)...")
    untrained = {**to_jax_params(TinyVisionEncoder(seed=0)), "Wh": jnp.zeros((64, 2)), "bh": jnp.zeros((2,))}
    ev_un = evaluate_vision_grasp(gene, make_box_locator(untrained), n=10, seed=1)
    ev_tr = evaluate_vision_grasp(gene, make_box_locator(params), n=10, seed=1)
    print(f"      UNTRAINED CV -> grasp {ev_un['grasp_success_rate']:.0%}  "
          f"(box-location error {ev_un['mean_loc_err_m'] * 100:.1f} cm)")
    print(f"      TRAINED   CV -> grasp {ev_tr['grasp_success_rate']:.0%}  "
          f"(box-location error {ev_tr['mean_loc_err_m'] * 100:.1f} cm)")

    print("[3/3] Rendering the arm grasping a CV-located box...")
    locate = make_box_locator(params)
    rng = np.random.default_rng(7)
    gx, gy = float(rng.uniform(GX0, GX1)), float(rng.uniform(GY0, GY1))
    g = vision_grasp_once(gene, locate, gx, gy, record_frames=True, frame_every=14)
    out = save_vision_montage(g["frames"], "build/vision_demo/vision_grasp_sequence.png", upscale=1, max_cols=8)
    print(f"      box at {g['true']} -> CV predicted {g['pred']} (err {g['loc_err_m'] * 100:.1f} cm); "
          f"grasped={g['success']} lifted={g['lifted']}m")
    print(f"      montage: {out}")
    return 0 if (ev_tr["grasp_success_rate"] >= 0.8 and test_mae < 0.5 * baseline_mae) else 1


if __name__ == "__main__":
    raise SystemExit(main())
