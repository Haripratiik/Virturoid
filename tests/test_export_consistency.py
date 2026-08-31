"""B3c + M3 (2026-07-24 audit): the two export doors must ship the SAME buildable robot.

A prompt can leave the studio two ways: the package builder (``build_gene_package``) or the agent's
``export_held`` tool. Before this fix they diverged -- ``export_held`` emitted the ungrounded ~3.5 kg fantasy
body for the URDF/sim while the builder grounded to ~8 kg with real actuator masses -- so "the same robot"
weighed twice as much depending on which button you pressed. Both now ground through the single shared
``ground_and_repair`` helper, so URDF mass + actuators match to floating point.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="export round-trip needs MuJoCo")


def _urdf_total_mass(urdf_path: Path) -> float:
    root = ET.parse(urdf_path).getroot()
    return round(sum(float(m.get("value", 0.0)) for m in root.iter("mass")), 3)


def _find_urdf(root: Path) -> Path:
    hits = list(root.rglob("robot.urdf")) or list(root.rglob("*.urdf"))
    assert hits, f"no URDF written under {root}"
    return hits[0]


def test_both_export_doors_ship_the_same_grounded_body():
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gene_build import build_gene_package
    from virturoid.services.morphology_composer import compose_robot

    os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="cmp_sess_")
    gene = compose_robot("a quadruped robot dog", llm=None)

    # Door A: the package builder grounds via ground_and_repair.
    out_a = Path(tempfile.mkdtemp(prefix="doorA_"))
    build_gene_package(copy.deepcopy(gene), "a quadruped robot dog", out_a, scene_count=2)
    mass_a = _urdf_total_mass(_find_urdf(out_a))

    # Door B: export_held must ground the SAME way (was: exported the ungrounded held gene).
    rid = S.put_robot(copy.deepcopy(gene), robot_id="cmp_dog")
    res = call_tool("export_held", {"robot_id": rid, "formats": ["urdf", "bom"]}).get("result", {})
    urdf_b = Path(res["artifacts"]["urdf"]).parent.parent / "robot" / "robot.urdf"
    mass_b = _urdf_total_mass(urdf_b)

    assert mass_a > 0 and mass_b > 0
    rel = abs(mass_a - mass_b) / max(0.1, mass_a)
    assert rel < 0.02, f"export doors diverge: build={mass_a} kg vs export_held={mass_b} kg (rel {rel:.3f})"


def test_export_held_grounds_without_mutating_the_session_gene():
    """The held session gene stays exactly as the user holds it -- export grounds a COPY."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.morphology_composer import compose_robot

    os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="cmp_sess2_")
    gene = compose_robot("a quadruped robot dog", llm=None)
    rid = S.put_robot(gene, robot_id="hold_dog")
    before = sum(float(getattr(s, "mass_kg", 0.0) or 0.0) for s in S.get_robot(rid).segments)
    call_tool("export_held", {"robot_id": rid, "formats": ["urdf"]})
    after = sum(float(getattr(s, "mass_kg", 0.0) or 0.0) for s in S.get_robot(rid).segments)
    assert abs(before - after) < 1e-6, f"export_held mutated the held gene: {before} -> {after}"
