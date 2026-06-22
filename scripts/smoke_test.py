"""Fast demo-machine smoke check (~30-60s, fully offline) — run this before recording the demo to confirm the
machine is green end-to-end without waiting ~16 min for the full test suite.

    python scripts/smoke_test.py        # exits 0 (all green) or 1 (something is broken)

Verifies the real demo path: deps import -> AI body composes offline -> legged build writes a replayable
package WITH an honest readiness ledger -> the gait replays -> the walking recipe is reachable.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VIRTUROID_LLM_BACKEND", "off")        # never hit the network on a demo machine
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_STATE: dict = {}


def _check(name, fn):
    t = time.time()
    try:
        fn()
        print(f"  [ OK ]  {name}  ({time.time() - t:.1f}s)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL]  {name}: {type(exc).__name__}: {exc}", flush=True)
        return False


def _deps():
    import mujoco  # noqa: F401
    import numpy  # noqa: F401
    from virturoid.services.morphology_composer import compose_robot  # noqa: F401


def _compose():
    from virturoid.services.morphology_composer import compose_robot
    g = _STATE["gene"] = compose_robot("a robot dog that walks", llm=None)
    assert g.robot_class, "no robot_class on composed gene"
    assert g.actuated_joints(), "composed body has no actuated joints"


def _build_and_ledger():
    from virturoid.services.gene_build import build_gene_package
    d = _STATE["pkg"] = Path(tempfile.mkdtemp(prefix="vsmoke_"))
    summary = build_gene_package(_STATE["gene"], "a robot dog that walks", d)
    assert summary["task_type"] == "locomotion", summary["task_type"]
    assert (d / "simulation" / "scene_set.json").exists(), "no scene_set.json (replay would crash)"
    led = d / "reports" / "product_readiness_ledger.json"
    assert led.exists(), "no honest product_readiness_ledger.json"
    import json
    stages = json.loads(led.read_text(encoding="utf-8")).get("stages")
    assert stages, "ledger has no stages"


def _replay():
    from virturoid.services.viewer_sim import simulate_episode_for_viewer
    view = simulate_episode_for_viewer(_STATE["pkg"])
    assert view["task"] == "locomotion", view["task"]
    assert view["frame_count"] > 0, "no replay frames"


def _recipe_reachable():
    from virturoid.services.morph_policy import recipe_rollout_morph
    from virturoid.services.morph_trainer import recipe_forward_score  # noqa: F401
    r = recipe_rollout_morph(_STATE["gene"], steps=120)
    assert r["finite"], "recipe rollout diverged"


def main():
    print("Virturoid demo-machine smoke check (offline)\n", flush=True)
    checks = [("dependencies import", _deps), ("AI body composes (offline)", _compose),
              ("legged build + honest ledger", _build_and_ledger), ("gait replay records frames", _replay),
              ("walking recipe reachable", _recipe_reachable)]
    ok = all(_check(name, fn) for name, fn in checks)
    print("\n" + ("ALL GREEN - safe to demo." if ok else "SOMETHING IS BROKEN - see [FAIL] above."), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
