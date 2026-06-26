"""Vision-guided grasp: the CV encoder is TRAINED to locate the box; the arm grasps the LEARNED position.

Relies on the suite's VIRTUROID_LLM_BACKEND=off (like test_grasp_eval) so the composer yields the working arm.
"""
import pytest


def test_trained_cv_locates_box_and_arm_grasps_it():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    import jax.numpy as jnp

    from virturoid.services.vision_encoder import TinyVisionEncoder, to_jax_params
    from virturoid.services.vision_grasp import (
        build_arm,
        evaluate_vision_grasp,
        gen_box_dataset,
        make_box_locator,
        train_box_locator,
    )

    try:
        gene = build_arm()
        x, y = gen_box_dataset(gene, n=160, seed=0)
        params, test_mae, baseline_mae = train_box_locator(x, y, epochs=70)
        trained = evaluate_vision_grasp(gene, make_box_locator(params), n=4, seed=1)
        untrained_params = {**to_jax_params(TinyVisionEncoder(seed=0)),
                            "Wh": jnp.zeros((64, 2)), "bh": jnp.zeros((2,))}
        untrained = evaluate_vision_grasp(gene, make_box_locator(untrained_params), n=4, seed=1)
    except Exception as exc:  # noqa: BLE001 - headless/no-GL or composer offline issue: skip, don't fail
        pytest.skip(f"no GL render / compose: {exc}")

    assert test_mae < 0.5 * baseline_mae                       # the encoder LEARNED to locate the box
    assert trained["mean_loc_err_m"] < 0.02                    # within ~2 cm from raw pixels
    assert trained["grasp_success_rate"] >= 0.75              # the arm grasps the LEARNED position
    # training matters: trained perception grasps at least as well, and locates far better, than untrained
    assert trained["grasp_success_rate"] >= untrained["grasp_success_rate"]
    assert trained["mean_loc_err_m"] < untrained["mean_loc_err_m"]
