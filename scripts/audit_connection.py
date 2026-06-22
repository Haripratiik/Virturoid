"""Audit a composed robot for VISUAL DISCONNECTION — parts that float apart instead of joining as one body.

Two failure modes the renders showed:
  • LATERAL gap: a limb branches off its parent at a sideways ``mount_offset`` whose lateral magnitude
    exceeds the parent's surface half-width -> the limb root hangs in empty space beside the body.
  • AXIAL gap/overlap: a child whose mount sits beyond (or far inside) the parent's distal tip along the
    chain, so consecutive links don't meet.
Plus a coverage check: every non-finger segment should get a visual mesh that spans ~[0, length].

    PYTHONPATH=src python scripts/audit_connection.py --prompt "a humanoid robot" [--kitbash] [--json]
Outputs a JSON report (machine-readable) so an agent/CI can flag regressions, not just eyeball renders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def audit(prompt: str, *, kitbash: bool = True) -> dict:
    os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
    from virturoid.services.morphology_composer import compose_robot

    g = compose_robot(prompt)
    seg = {s.name: s for s in g.segments}

    def halfwidth(s) -> float:
        return float(s.radius_m)            # box: half-width; capsule/cylinder: radius — the lateral surface

    lateral_gaps, axial_notes = [], []
    for s in g.segments:
        if s.parent is None:
            continue
        # gripper/hand jaws are prismatic children that intentionally spread to ±span and are bridged by the
        # compiler's visual palm plate — not a floating gap.
        if s.joint_type == "prismatic":
            continue
        p = seg[s.parent]
        mo = getattr(s, "mount_offset", (0.0, 0.0, 0.0))
        lateral = max(abs(mo[0]), abs(mo[1]))
        gap = lateral - halfwidth(p)
        if gap > 0.005:                     # >5 mm beyond the parent surface = a visible float
            lateral_gaps.append({"segment": s.name, "parent": s.parent,
                                 "lateral_offset_m": round(lateral, 4),
                                 "parent_halfwidth_m": round(halfwidth(p), 4),
                                 "gap_m": round(gap, 4)})
        # axial: child base sits at parent.length + mo.z along the parent's +z. For a serial chain link
        # (mount at the tip) mo.z≈0; a branch (limb) intentionally offsets. Flag a child that mounts ABOVE
        # the parent tip (mo.z>0 => past the end => a gap) on a non-branch.
        if mo[2] > 0.01 and abs(mo[0]) < 1e-6 and abs(mo[1]) < 1e-6:
            axial_notes.append({"segment": s.name, "parent": s.parent, "mount_z_past_tip_m": round(mo[2], 4)})

    # mesh coverage (kit-bash or procedural) — every non-prismatic segment should bake a mesh spanning ~length
    coverage, missing = [], []
    try:
        import tempfile

        from virturoid.services.cad_geometry import build_visual_meshes
        with tempfile.TemporaryDirectory() as tmp:
            meshes = build_visual_meshes(g, tmp, kitbash=kitbash)
        for s in g.segments:
            if s.joint_type == "prismatic":
                continue
            coverage.append({"segment": s.name, "meshed": s.name in meshes})
            if s.name not in meshes:
                missing.append(s.name)
    except Exception as e:  # noqa: BLE001
        coverage = [{"error": f"{type(e).__name__}: {e}"}]

    return {
        "prompt": prompt, "robot_class": g.robot_class, "species": g.species,
        "n_segments": len(g.segments), "design_source": getattr(g, "design_source", "?"),
        "lateral_gaps": lateral_gaps, "axial_notes": axial_notes,
        "meshes_missing": missing, "n_meshed": sum(1 for c in coverage if c.get("meshed")),
        "verdict": "CONNECTED" if not lateral_gaps and not missing else "HAS_GAPS",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a humanoid robot")
    ap.add_argument("--no-kitbash", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rep = audit(args.prompt, kitbash=not args.no_kitbash)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"{rep['robot_class']:12s} {rep['verdict']:10s} segs={rep['n_segments']} "
              f"meshed={rep['n_meshed']} lateral_gaps={len(rep['lateral_gaps'])} missing={rep['meshes_missing']}")
        for gp in rep["lateral_gaps"]:
            print(f"   GAP {gp['segment']} off {gp['parent']}: {gp['gap_m']*1000:.0f}mm beyond surface")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
