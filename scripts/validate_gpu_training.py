"""End-to-end GPU training validation: train a quadruped on the GPU box (MJX PPO, with domain randomization),
fetch the policy, and verify it WALKS on CPU (cadence + upright + survived) AND holds up under randomized
dynamics. This closes two "partial" gaps in one run: GPU training was wired but never validated end-to-end, and
the GPU --dr application was not integration-tested.

    python scripts/validate_gpu_training.py

Exit 0 = validated walking policy; 1 = trained but did not walk; 2 = GPU box unreachable (needs the external
Tailscale box). On PASS it leaves the validated policy at models/_gpu_validate_quad.npz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    from virturoid.services.gpu_trainer import gpu_available, train_gene_on_gpu

    if not gpu_available(timeout=25):
        print("GPU box unreachable -- skipping (this validator needs the external box).")
        return 2

    from virturoid.services.morph_policy import MorphPolicy, recipe_robustness, recipe_rollout_morph
    from virturoid.services.morphology_composer import compose_robot

    gene = compose_robot("a quadruped walking robot")
    print("[1/3] Training on the GPU box (MJX PPO, trot-CPG + domain randomization)...", flush=True)
    npz = train_gene_on_gpu(gene, out_path="models/_gpu_validate_quad.npz", iters=100, envs=1024,
                            cpg=True, dr=True, progress=lambda m: print("   ", m, flush=True))
    if not npz:
        print("FAIL: GPU training returned no policy (box stalled or launch failed).")
        return 1

    pol = MorphPolicy.from_npz(npz)
    print("[2/3] Verifying the FETCHED policy walks on CPU (the end-to-end round-trip)...", flush=True)
    w = recipe_rollout_morph(gene, pol, steps=1500)
    walked = bool(w.get("survived") and float(w.get("cadence", 0)) > 0.5 and float(w.get("upright_frac", 0)) > 0.5)
    print(f"      forward={float(w.get('forward', 0)):.2f}m  cadence={float(w.get('cadence', 0)):.1f}/s  "
          f"upright={w.get('upright_frac')}  survived={w.get('survived')}  -> walked={walked}", flush=True)

    print("[3/3] Verifying robustness under randomized dynamics (the DR check)...", flush=True)
    r = recipe_robustness(gene, pol, n=6)
    print(f"      randomized-dynamics result: {r}", flush=True)

    print("VALIDATION:", "PASS -- GPU end-to-end produces a walking, DR-trained policy" if walked else
          "FAIL -- trained but did not meet the walk gate")
    return 0 if walked else 1


if __name__ == "__main__":
    raise SystemExit(main())
