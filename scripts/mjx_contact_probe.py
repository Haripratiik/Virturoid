"""Probe: does the CONTACT pipeline work in MJX on this box? (M2 prerequisite.)

The reach trainer (mjx_ppo_min.py) DISABLES contacts — free-space reach only. A contact skill
(push / grasp) needs them ON, which is the fragile part on a consumer GPU under the WSL2 TDR
watchdog and MJX's fixed-size contact buffers. This compiles the arm gene + a real pick-place
scene (table + a free box) WITH contacts, puts it on the GPU, batches N envs, steps, and reports:
the contact-cap API (naconmax/nconmax/njmax), per-step timing, and whether state stays finite.
Run it before investing in the trainer.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_contact_probe.py
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    print("jax", jax.__version__, "| backend", jax.default_backend(), jax.devices())
    print("mujoco", mujoco.__version__)

    gene = tabletop_arm_gene()
    spec = select_task_spec("place the box on the target")
    scenes = generate_task_scenes(gene, spec, count=1)
    xml = compile_gene_with_scene(gene, scenes[0].objects)
    mj = mujoco.MjModel.from_xml_string(xml)  # contacts ON (NOT disabled like the reach script)
    print(f"model: ngeom={mj.ngeom} nq={mj.nq} nv={mj.nv} nu={mj.nu}")

    mx = mjx.put_model(mj)
    print("make_data signature:", str(inspect.signature(mjx.make_data)))

    # Discover which contact-buffer cap kwargs this MJX build accepts.
    N = 64
    cap_kwargs = {}
    for trial in ({"naconmax": N * 16, "njmax": N * 16}, {"nconmax": N * 16, "njmax": N * 16}):
        try:
            mjx.make_data(mx, **trial)
            cap_kwargs = trial
            print("make_data caps accepted:", list(trial))
            break
        except TypeError:
            continue
    if not cap_kwargs:
        print("make_data caps: none accepted -> auto-sized buffers")

    data = jax.vmap(lambda _: mjx.make_data(mx))(jp.arange(N))
    data = jax.vmap(lambda d: mjx.forward(mx, d))(data)
    step_v = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))  # single-step kernel = TDR-safe
    ctrl0 = jp.zeros((N, mj.nu))

    # Warm up (compile), then time 40 steps of gravity settling under contacts.
    data = step_v(mx, data.replace(ctrl=ctrl0))
    jax.block_until_ready(data.qpos)
    t0 = time.time()
    for _ in range(40):
        data = step_v(mx, data.replace(ctrl=ctrl0))
    jax.block_until_ready(data.qpos)
    dt = time.time() - t0
    finite = bool(jp.all(jp.isfinite(data.qpos)))
    sps = 40 * N / dt
    print(f"stepped 40 x {N} envs WITH contacts in {dt:.2f}s ({sps:,.0f} env-steps/s), finite={finite}")
    print("=== CONTACT PIPELINE OK ===" if finite else "=== DIVERGED (NaN) ===")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
