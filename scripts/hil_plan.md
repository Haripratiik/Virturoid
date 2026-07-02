# Hardware-in-the-loop (HIL) sim2real plan — costed (plan v3 M5)

Pure-sim work measures the reality **gap** (`services/sim2real.sim2real_transfer_report`: nominal vs a held-out,
wider-than-training DR distribution → `transfer_survival` / `forward_retention` / `reality_gap_m`). Closing the
gap on real hardware is a separate, costed track — scoped here so it is concrete, not hand-waved. **Full hardware
is explicitly out of scope for the sim platform; this is the plan a hardware track would execute.**

## Target platform (cheapest credible quad)

| Item | Choice | ~Cost (USD) |
|---|---|---|
| Actuators | 12× hobby serial-bus servos (e.g. Feetech STS3215, 30 kg·cm, position+load feedback) | ~$220 |
| Chassis | 3D-printed quad frame (our CAD export → STL), M3 hardware | ~$40 |
| Compute | Raspberry Pi 5 (8 GB) — runs the CPU `recipe_rollout_morph` policy at 50 Hz | ~$80 |
| IMU | BNO085 (fused orientation, for the upright/height estimate) | ~$25 |
| Power | 2S/3S LiPo + UBEC | ~$45 |
| **Total** | | **~$410** |

The design maps 1:1 onto our 8–12 DOF `steerable_quadruped`; the servos are position-controlled, which matches
the deployed **PD-to-default-pose + residual** control law exactly (no torque estimation needed on hardware).

## The sim2real bridge (already in hand)

The deploy-gap fixes are the bridge — deploy the policy on hardware the same way it trained:
- **50 Hz control** (`decimation=10`) — the servo command rate; matches training.
- **Action LPF** (`action_lpf`) — smooths servo jitter, matches training.
- **Sphere feet** (manifold-invariant contact) — the sim foot model closest to a real rubber foot.
- **Contact-DR** was a *train-time* aug; on hardware the real contact replaces it.
- Select the checkpoint by **deploy-sim CPU forward** (`best_checkpoint_by_deploy`), not MJX reward.

## Protocol (staged, kill criteria)

1. **Bench (no locomotion):** command the default stance; verify each joint tracks its target ±X°. *Kill:* if
   servo tracking error > 15°, the gains/backlash break the PD assumption → recalibrate before walking.
2. **Tethered stand:** run the policy on a gantry; confirm it holds the stance under a light push. *Kill:* falls
   immediately → the reality gap on *balance* exceeds the held-out-DR band; widen training DR to that band.
3. **Free walk (short):** 2 m straight-line walk; measure real forward vs the sim `forward_m`. The **measured
   reality gap** = `sim_forward − real_forward`; compare to the sim `reality_gap_m` prediction.
4. **Robustness:** walk over a foam mat / small incline; the survival rate is the real analog of
   `transfer_survival`.

## What "success" means

Not "zero gap" — a *predicted* gap. If the sim `reality_gap_m` (from `sim2real_transfer_report`) brackets the
measured bench-to-hardware drop, the sim is **calibrated** and the platform can be trusted to pre-screen designs
before any metal is cut. That is the honest sim2real claim: *the sim predicts the gap*, not *there is no gap*.

## Effort / cost

- Parts: **~$410**, ~1 week lead time.
- Assembly + firmware (servo bus + Pi policy runner reading the banked npz): ~3–5 engineer-days.
- The policy runner is a thin loop: load `MorphPolicy.from_npz`, read IMU+servo feedback into the same obs the
  sim builds, emit position targets at 50 Hz — the CPU `recipe_rollout_morph` control law, ported to hardware I/O.
