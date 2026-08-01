"""An imported robot's CLASS must survive ingestion.

`robot_import._infer_class` was fixed (#244, tests/test_imported_robot_class.py) to tell a humanoid from a
quadruped by WHAT CARRIES THE BODY, and it gets the whole Menagerie corpus right. `_ingest_project` then threw
that answer away. A reconciliation added for #214 -- a fixed-base legged import the string classifier had called
an arm -- fired for EVERY class outside ("quadruped", "legged", "hexapod") and always wrote the literal
"quadruped", so an imported HUMANOID was renamed a quadruped on the way in.

Measured at HEAD on ~/.cache/robot_descriptions/mujoco_menagerie/unitree_g1 (2026-08-01):

    import.robot_class      : humanoid        <- _infer_class got it right
    HELD gene.robot_class   : quadruped       <- the ingest overwrote it
    NOTE: reclassified the import from 'humanoid' to a legged body (it has multiple symmetric limbs ...)
    NOTE: the imported quadruped does not walk credibly as imported (0.0 m under the scripted gait) ...
    walkable_template_offer : {'available': True, 'template_distance_m': 0.681, ...}

0.681 m is the GENERIC QUAD template's distance -- byte-identical to the number offered for a Go2 -- so the
offer was proposing to turn a biped into a quadruped. bom.json and spec_sheet.json read `robot_class:
quadruped` off the same field.

This is NOT #244 resurfacing. `_infer_class` is correct here and stays correct; the defect is a SECOND, coarser
classifier sitting downstream of it, with a hard-coded target. The tests below pin both halves: the humanoid
keeps its class, and the #214 case it was written for still reconciles.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="ingest needs MuJoCo")

_LEGGED_FAMILY = {"quadruped", "hexapod", "octopod", "legged", "humanoid", "biped"}


def _menagerie_project(tmp_path, model_dir: str, model_file: str) -> str:
    """A customer 'project folder' holding ONE real Menagerie model + its meshes. README/extra variants are left
    out so the description under test is the only text the NLP layer sees."""
    src = _MEN / model_dir
    if not (src / model_file).is_file():
        pytest.skip(f"{model_dir} is not cached locally (robot_descriptions fetches on demand)")
    dst = Path(tmp_path) / model_dir
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / model_file, dst / model_file)
    for sub in ("assets", "meshes"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    return str(dst)


def _gene(rel: str):
    """The editable gene twin of a real Menagerie model."""
    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot
    return import_robot(str(src), robot_id="t")["gene"]


def test_ingesting_a_real_humanoid_does_not_rename_it_a_quadruped(tmp_path):
    """The G1 is a biped. Every artefact the ingest produces has to say so."""
    from virturoid.services import session_state as S
    from virturoid.services.input_training_tools import _ingest_project

    os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="ingest_cls_")
    proj = _menagerie_project(tmp_path, "unitree_g1", "g1.xml")
    r = _ingest_project({"project_path": proj,
                         "description": "Our Unitree G1 humanoid, 1.3 m tall, 35 kg, 29 joints."})

    assert (r.get("import") or {}).get("robot_class") == "humanoid", (
        f"precondition: _infer_class must still classify the G1 a humanoid, got {r.get('import')}")

    held = S.get_robot(r["robot_id"])
    assert held is not None, r.get("warnings")
    assert held.robot_class == "humanoid", (
        f"the held twin of an imported humanoid is classed {held.robot_class!r}; the ingest overwrote the "
        f"structural answer. notes={r.get('notes')}")

    notes = " ".join(r.get("notes") or []).lower()
    assert "to a legged body" not in notes and "'humanoid' to" not in notes, (
        f"a humanoid must not be 'reclassified' at all: {r.get('notes')}")
    # the honesty ledger names the body by what it IS
    walk_notes = [n for n in (r.get("notes") or []) if "does not walk credibly" in n]
    for n in walk_notes:
        assert "quadruped" not in n.lower(), f"the honesty ledger calls the biped a quadruped: {n}"

    # the walkable reference template is a QUADRUPED fan/crawl recipe -- never offered as an upgrade to a biped
    offer = r.get("walkable_template_offer") or {}
    assert not offer.get("available"), (
        f"offered a quadruped template to a humanoid ({offer.get('template_distance_m')} m): {offer}")

    # the same measured ingest also proves the payload half: 35 kg is the G1's OWN mass, not a load it carries
    assert r.get("payload_kg") is None, (
        f"the ingest applied the robot's own mass as a {r.get('payload_kg')} kg payload, silently upsizing "
        f"the customer's actuators")


def test_the_fixed_base_legged_import_214_was_written_for_still_reconciles():
    """Guard against over-correcting: a 4-limb fixed-base URDF really is mislabelled 'manipulator' by the string
    classifier, and must still come out of the ingest as a legged body so verify uses the legged rubric."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_import_verify_honesty import _fixed_base_quad_urdf

    from virturoid.services import session_state as S
    from virturoid.services.input_training_tools import _ingest_project
    from virturoid.services.robot_import import import_robot

    os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="ingest_cls214_")
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    assert import_robot(os.path.join(tmp, "q.urdf"))["robot_class"] == "manipulator", (
        "precondition for #214: the string classifier calls this fixed-base quad an arm")

    r = _ingest_project({"project_path": tmp, "description": "my patrol quadruped"})
    held = S.get_robot(r["robot_id"])
    assert held.robot_class in _LEGGED_FAMILY, f"#214 regression: reconciled to {held.robot_class!r}"
    assert held.robot_class == "quadruped", (
        f"it stands on four legs; reconciled to {held.robot_class!r}")


def test_the_reconciliation_never_invents_a_family_it_did_not_measure():
    """The rule that caused the bug: 'structurally legged' was turned into the literal string 'quadruped'.

    Two halves, both on real robots: a body whose class ALREADY names a legged family is not the #214 case at
    all, and when the remap does fire the family has to come from the limbs that carry the body."""
    from virturoid.services.input_training_tools import _legged_family, _needs_legged_reconciliation

    g1, go2 = _gene("unitree_g1/g1.xml"), _gene("unitree_go2/go2.xml")

    assert not _needs_legged_reconciliation(g1), (
        "an imported humanoid is not the #214 'the string classifier called my quadruped an arm' case")
    assert not _needs_legged_reconciliation(go2)

    assert _legged_family(g1) == "humanoid", "two legs carry a G1"
    assert _legged_family(go2) == "quadruped", "four legs carry a Go2"
