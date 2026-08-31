"""WHICH SENSORS DID WE READ OFF THE CUSTOMER'S MODEL, AND WHICH DID WE ASSUME? One answer, read by every
surface that reports or bills perception.

MEASURED DEFECT (2026-08-10, real MuJoCo Menagerie Go2 through ``agent_tools.call_tool``). The customer's file
compiles to ``ncam 0, nsensor 0``. On the robot imported from it we answered:

    verify_robot.vision -> {"camera_part": "Intel RealSense D435i", "fovy_deg": 58.0, "sees": true}
    bom.json            -> one Intel RealSense D435i, $334

Neither surface was buggy on its own. ``bom_builder._sensor_suite`` maps ``robot_class`` to a parts list, and
"a quadruped carries a forward depth camera" is a fine DEFAULT for a robot we are designing. The defect is that
nothing ever asked whether this robot was ours to design: the class default was applied to a machine the
customer already owns, ``sensor_geometry`` then emitted a functional ``<camera name="robot_cam">`` from it,
``camera_perception`` rendered through that camera, and the render came back with a target in frame -- so
``sees: true`` was a real measurement of a camera we had invented, and the BOM turned it into $334.

THE RULE THIS MODULE ENFORCES: a sensor the customer's model does not declare is a PROPOSAL, never a fact. It
may be recommended, costed as an addition and drawn in a design view; it may not be reported as fitted, billed
inside the as-imported machine, or rendered through and measured.

Two things it deliberately does NOT do:

* it does not map a declared ``<camera>`` to a part number. An MJCF camera is a render viewpoint: 19 of the 63
  Menagerie packages declare one and most are tracking or cinematic cams (``track@trunk`` on the Go1). "Their
  file has a camera element" is not "their robot has a $334 RealSense", and inventing the second from the first
  would be the same defect wearing a better disguise.
* it does not silence perception on robots WE composed. There the sensor suite is a design proposal by
  construction -- the robot does not exist yet -- so the camera, its FOV and its render are all legitimate.
"""

from __future__ import annotations

#: BOM categories that are PERCEPTION HARDWARE — the lines this module rules on.
SENSOR_CATEGORIES = ("camera", "lidar", "imu", "force_torque", "gps", "thermal", "microphone", "encoder")

#: MuJoCo ``<sensor>`` types that ARE evidence of a physical instrument, by the BOM category they evidence.
#:
#: This is the one inference the module permits in the customer's FAVOUR, and it is allowed because a
#: ``<sensor>`` is declared instrumentation in a way a ``<camera>`` is not: an author writes ``<gyro>`` because
#: the machine has a gyro, whereas a ``<camera>`` is a render viewpoint (see the module docstring). Refusing it
#: would make the table wrong in the other direction — quoting a $35 IMU as an ADDITION to a robot whose own
#: file declares gyro + accelerometer, which is the same failure to read their model, just cheaper.
#:
#: Deliberately excluded: ``mjSENS_FRAMEQUAT`` / ``mjSENS_FRAMELINVEL`` and friends are simulator ground-truth
#: taps on a body frame, not signals a physical part produces, so they evidence nothing about the hardware.
#: There is no camera entry at all — MJCF has no camera SENSOR, only viewpoints.
_TYPE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "imu": ("mjSENS_GYRO", "mjSENS_ACCELEROMETER", "mjSENS_MAGNETOMETER"),
    "force_torque": ("mjSENS_FORCE", "mjSENS_TORQUE"),
    "encoder": ("mjSENS_JOINTPOS", "mjSENS_JOINTVEL", "mjSENS_ACTUATORPOS", "mjSENS_ACTUATORVEL"),
    "lidar": ("mjSENS_RANGEFINDER",),
}


def category_declared(gene, category: str) -> tuple[bool, str]:
    """Does the customer's own model declare an instrument of this BOM ``category``? ``(yes, evidence)``.

    ``evidence`` names the ``<sensor>`` types it was read off, so the answer can be checked against their file.
    Always ``(False, "")`` for a body we composed (there is no customer model to read) and for ``camera``.
    """
    inv = declared(gene) if is_imported(gene) else None
    types = (inv or {}).get("sensor_types") or {}
    hits = [f"{t} x{types[t]}" for t in _TYPE_EVIDENCE.get(category, ()) if types.get(t)]
    return (bool(hits), ", ".join(hits))


def is_imported(gene) -> bool:
    """True when this body came in through the import path — i.e. it is the CUSTOMER'S machine, not ours.

    Keyed on ``metadata['imported_from']``, the same marker ``ai_native_tools._import_provenance`` and
    ``gene_build`` recognise an imported body by, so perception cannot disagree with the verdict about whose
    robot it is.
    """
    return bool(str(((getattr(gene, "metadata", None) or {}).get("imported_from")) or ""))


def declared(gene) -> dict | None:
    """The sensing the customer's own model declares, read verbatim at import (or ``None`` — see below).

    ``None`` means UNKNOWN, and unknown is not the same as none: a body we composed has nothing to declare, and
    a body imported before ``robot_import`` recorded the inventory was never asked. Both are handled by refusing
    to claim, never by assuming.
    """
    meta = getattr(gene, "metadata", None) or {}
    inv = meta.get("source_sensors")
    return dict(inv) if isinstance(inv, dict) else None


def declared_cameras(gene) -> list:
    """The ``<camera>`` elements the customer's model declares (name / mount body / fovy). Possibly empty."""
    inv = declared(gene) or {}
    cams = inv.get("cameras")
    return list(cams) if isinstance(cams, list) else []


def carries_our_proposed_sensors(gene) -> bool:
    """True when a sensor suite on this body would be a PROPOSAL rather than a report of fitted hardware.

    That is exactly "this is the customer's machine". For a body we composed the suite is part of the design we
    are handing them, so it is legitimately reported as what the robot carries.
    """
    return is_imported(gene)


def camera_is_ours_to_add(gene) -> tuple[bool, str]:
    """May we mount a functional camera on this body and report what it sees? ``(allowed, reason)``.

    Allowed for anything we composed. Refused for an imported machine, and the reason names the number the
    refusal rests on so it can be checked against the customer's own file.
    """
    if not is_imported(gene):
        return True, "this robot is a design of ours; its sensor suite is part of the design"
    inv = declared(gene)
    if inv is None:
        return False, ("this robot was imported from your model and we have no record of what sensing that model "
                       "declares, so we will not fit a camera to it and report what it sees")
    n_cam, n_sen = int(inv.get("ncam", 0) or 0), int(inv.get("nsensor", 0) or 0)
    if n_cam == 0:
        return False, (f"your model declares no camera (mjModel.ncam {n_cam}, nsensor {n_sen}). We will not "
                       f"mount one on your machine and report what it sees")
    names = ", ".join(f"{c.get('name')}@{c.get('mount_body')} (fovy {c.get('fovy_deg')})"
                      for c in declared_cameras(gene)[:6])
    return False, (f"your model declares {n_cam} <camera> element(s) — {names} — but an MJCF camera is a render "
                   f"viewpoint, not a declared piece of hardware, and our editable twin re-derives your links, "
                   f"so we cannot place it faithfully enough to report what YOUR camera sees. What your model "
                   f"declares is quoted above rather than converted into a part")


def sensor_reality_table(gene, suite) -> dict:
    """The deliverable: for this robot, which perception is READ from the customer's model and which is ASSUMED.

    ``suite`` is ``bom_builder._sensor_suite``'s output — ``[(part_name, qty, mount), ...]``. Every entry in it
    is, by construction, an inference from ``robot_class`` + task keywords + mass; the table says so line by
    line and sets it against what the model actually declares.
    """
    from virturoid.services.component_catalog import component

    inv = declared(gene)
    imported = is_imported(gene)
    assumed = []
    for name, qty, mount in suite or ():
        c = component(name)
        cat = getattr(c, "category", None) or "sensor"
        # WHICH PART is always ours to pick; whether the INSTRUMENT is on the robot can sometimes be read off
        # their file. Both halves are stated per line, so the table never has to be taken on trust.
        has, evidence = category_declared(gene, cat)
        row = {"part": name, "category": cat, "qty": int(qty),
               "why": f"assumed from robot_class={gene.robot_class!r} + task keywords ({mount})",
               "declared_by_your_model": bool(has)}
        if has:
            row["why"] = (f"the PART is our pick; your model declares a {cat} ({evidence}), so this is an "
                          f"equivalent for hardware you already have, not an addition")
            row["read_from_your_model"] = evidence
        assumed.append(row)
    out = {
        "robot_is": "your imported machine" if imported else "a design of ours",
        "read_from_your_model": (
            {"ncam": inv.get("ncam"), "nsensor": inv.get("nsensor"),
             "cameras": inv.get("cameras"), "sensor_types": inv.get("sensor_types")}
            if inv is not None else None),
        # NOT "assumed_by_us": a row whose instrument the model evidences is not an assumption, and a key that
        # said otherwise would make the table wrong about itself. Each row carries its own verdict.
        "perception_lines": assumed,
        "rule": ("a sensor your model does not declare is a PROPOSED ADDITION, priced separately and never "
                 "reported as fitted" if imported else
                 "this robot does not exist yet, so its whole sensor suite is a proposal we are making to you"),
    }
    if inv is None and imported:
        out["read_from_your_model"] = "not recorded at import (re-import to have your model's sensors read)"
    return out
