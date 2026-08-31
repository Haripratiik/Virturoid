"""The export door's simulability gate — which artifacts a twin that cannot be STEPPED is still entitled to.

Commit 9fbdcdc added ``robot_import._simulability_probe`` and both ``robot_import`` and ``robot_import_report``
began printing the sentence "*no verdict, certificate, BOM, spec sheet or calibration number can be produced
from this twin*". Nothing enforced it. MEASURED on MuJoCo Menagerie's ``flybody`` (the one package in 63 whose
editable twin genuinely does not compile — ``mass and inertia of moving bodies must be larger than mjMINVAL``),
``export_held`` returned ``ok: True`` and wrote ``bom.json`` quoting **66 Dynamixel actuators, $5,598.00,
5.146 kg, 213.6 W**, closing with "verify exact specs against the live datasheet before procurement", with no
occurrence of "simulab", "unverified", "warning" or "cannot" anywhere in the file. The body those motors were
sized for weighs **0.000985 kg**. A customer could procure from that.

WHERE THE LINE IS DRAWN, and why it is not "refuse everything":

* **The customer's geometry is still theirs.** ``mjcf``, ``urdf``, ``ros2``, ``cad``, ``usd`` and ``isaac_lab``
  TRANSCRIBE a body they gave us into another representation. They assert nothing we measured, and they are
  precisely the files an engineer needs in order to find and fix the degenerate link. Withholding them would
  be withholding the customer's own data at the moment it is most useful. They ship — carrying the refusal
  INSIDE the file, because an artifact that leaves the package stops travelling with its envelope.
* **A number asserted about physics that was never simulated does not ship.** ``bom`` sizes real motors,
  quotes a mass and prints a price from loads read off this body; ``spec`` leads with a mass, a peak torque
  and a verdict; ``certificate`` exists solely to say a rollout happened and what it measured. Each is refused
  and replaced by a ``*.REFUSED.json`` + ``.md`` at the place the artifact would have been.

The refusal documents are deliberately NOT written under the real artifact names. ``spec_sheet.build_spec_sheet``
globs for ``bom.json`` / ``verification_certificate.json`` and reports ``sources.verification_certificate: true``
for anything it finds there; a refusal parked under an evidence filename becomes evidence one directory later.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Formats that ASSERT a number about physics nobody simulated. Refused when the twin cannot be stepped.
GATED_FORMATS = ("bom", "spec", "certificate")

#: Formats that TRANSCRIBE the customer's own geometry. Still shipped, stamped with the refusal.
GEOMETRY_FORMATS = ("mjcf", "cad", "urdf", "ros2", "usd", "isaac_lab")

#: Per-format justification. Printed in the refusal document so the line is arguable, not merely enforced.
REFUSAL_REASON = {
    "bom": ("a bill of materials picks a REAL motor per joint from the torque this body demands, totals a mass, "
            "and prints a price. On a body that cannot be stepped all three are fabricated — and they arrive "
            "with a part number a buyer can order against."),
    "spec": ("a spec sheet's headline IS physics: as-built mass, peak joint torque, power draw, and a task "
             "verdict. There is no physics in this package for it to summarise."),
    "certificate": ("a verification certificate exists to state that a rollout happened and what it measured. "
                    "No rollout can happen on a model that does not compile, so there is nothing to certify — "
                    "not even negatively."),
}

_HEADLINE = "VIRTUROID EXPORT — THIS TWIN IS NOT SIMULABLE"


def check_simulable(gene) -> dict:
    """Can the body THIS PACKAGE SHIPS be stepped? Returns the probe verdict (``ok``/``checked``/``reason``).

    Probed here rather than trusted from import, for two reasons: the export door grounds and repairs a copy of
    the held gene before writing anything, so the body in the package is not necessarily the body that was
    imported; and a gene can be edited into a state its import-time marker no longer describes. A recorded
    ``simulable: False`` on the gene still counts — a twin that was rejected at the door stays rejected — but a
    recorded True never overrides a live failure.

    "We could not look" is not evidence of a broken twin: if the probe itself is unavailable this returns ``ok``
    with ``checked: False``, exactly as ``_simulability_probe`` does when MuJoCo is missing.
    """
    meta = getattr(gene, "metadata", None) or {}
    recorded = meta.get("simulable")
    try:
        from virturoid.services.robot_import import _simulability_probe
        sim = dict(_simulability_probe(gene))
    except Exception as exc:  # noqa: BLE001 - the gate must never be the reason an export fails
        sim = {"ok": True, "checked": False,
               "reason": f"the simulability probe could not run ({type(exc).__name__}: {exc})"[:300]}
    sim["probed_at"] = "export"
    if recorded is False and sim.get("ok"):
        # The import already ruled against this twin. A later probe that happens to pass does not overturn it.
        sim = {**sim, "ok": False, "recorded_at_import": False,
               "reason": (str((meta.get("simulation_check") or {}).get("reason")
                              or "the import marked this twin NOT SIMULABLE")
                          + " (recorded at import; the export-time probe did not reproduce it)")}
    elif recorded is not None:
        sim["recorded_at_import"] = bool(recorded)
    return sim


def split_formats(fmts) -> tuple[list[str], list[str]]:
    """``(refused, still_shipped)`` for a requested format list, in the caller's own order."""
    return ([f for f in fmts if f in GATED_FORMATS],
            [f for f in fmts if f not in GATED_FORMATS])


def build_refusal(sim: dict, fmts) -> dict:
    """The machine-readable refusal: what was withheld, what still shipped, and why — one dict, reused by
    every artifact stamp so the package cannot contradict itself about its own refusal."""
    refused, shipped = split_formats(fmts)
    return {
        "artifact": "virturoid_export_refusal",
        "version": 1,
        "simulable": False,
        "reason": sim.get("reason"),
        "simulation_check": sim,
        "refused": {f: REFUSAL_REASON[f] for f in refused},
        "still_shipped": shipped,
        "rule": ("An export may transcribe the customer's own geometry into another format. It may not assert a "
                 "number about physics that was never simulated. Everything withheld here is on the wrong side "
                 "of that line."),
        "do_not": ["order parts against this package", "quote a mass, cost or power figure from it",
                   "treat any file in it as a verification result"],
        "next_step": ("Fix the link named in the reason above and re-import, or simulate with the FAITHFUL "
                      "native lane (model_import), which runs the customer's model as-is and is unaffected by "
                      "the editable twin's defect."),
    }


def refusal_banner(refusal: dict) -> str:
    """The human paragraph, plain text — the same words in the MJCF comment, the READMEs and the notice."""
    refused = ", ".join(refusal.get("refused") or {}) or "(none)"
    shipped = ", ".join(refusal.get("still_shipped") or []) or "(none)"
    return (f"{_HEADLINE}\n\n"
            f"This model could not be stepped: {refusal.get('reason')}\n\n"
            f"WITHHELD from this package: {refused}. Every one of those asserts a number measured by running "
            f"this body, and this body does not run.\n"
            f"STILL SHIPPED: {shipped}. That is YOUR OWN geometry, transcribed — shipped so you can inspect "
            f"and repair it, NOT because anything about it was verified.\n\n"
            f"Do not size motors, quote a mass or a cost, or order parts from this package. "
            f"{refusal.get('next_step')}\n"
            f"Machine-readable detail: export_refusal.json at the package root.")


# --------------------------------------------------------------------------- stamping the shipped artifacts
def _xml_comment(text: str) -> str:
    """An XML comment that is legal whatever the reason string contains. ``--`` may not appear inside one, and
    a MuJoCo compile error legitimately can (`` -- ``), so it is broken up rather than dropped."""
    body = text.replace("--", "- -")
    while body.endswith("-"):
        body = body[:-1] + "–"          # a trailing '-' would close as '--->'
    return "<!--\n" + body + "\n-->\n"


def stamp_file(path, banner: str) -> bool:
    """Write the refusal INTO one shipped artifact, in that file's own comment syntax. Returns True if stamped.

    An artifact leaves the package; the envelope does not travel with it. A customer who opens ``robot.xml`` in
    MuJoCo, or hands the URDF to a colleague, must read the refusal there — which is the whole difference
    between this and the ``ok: False`` that was already available and already ignored.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return False
        text = p.read_text(encoding="utf-8", errors="replace")
        suffix = p.suffix.lower()
        if suffix in (".xml", ".urdf", ".sdf", ".usda"):
            block = _xml_comment(banner)
            if suffix == ".usda":
                # a .usda MUST open with its `#usda 1.0` magic line; the comment goes immediately after it.
                head, _, rest = text.partition("\n")
                p.write_text(head + "\n" + "".join(f"# {ln}\n" for ln in banner.splitlines()) + rest,
                             encoding="utf-8")
                return True
            if text.lstrip().startswith("<?xml"):
                decl, _, rest = text.partition("?>")
                p.write_text(decl + "?>\n" + block + rest.lstrip("\n"), encoding="utf-8")
            else:
                p.write_text(block + text, encoding="utf-8")
            return True
        if suffix in (".md", ".txt", ""):
            p.write_text("> **" + _HEADLINE + "**\n>\n"
                         + "".join(f"> {ln}\n" for ln in banner.splitlines()[2:]) + "\n" + text,
                         encoding="utf-8")
            return True
    except OSError:
        return False
    return False


def write_refusal_package(out_dir, refusal: dict, *, extra_targets=()) -> dict:
    """Write the package-level notice + one ``*.REFUSED.json``/``.md`` per withheld format, and stamp every
    shipped artifact we can name. Returns ``{"notice", "refusal_json", "refused_docs", "stamped"}``.

    ``*.REFUSED.json`` rather than the artifact's real name is deliberate: see the module docstring.
    """
    out = Path(out_dir)
    banner = refusal_banner(refusal)
    written: dict = {"refused_docs": {}, "stamped": []}

    (out / "export_refusal.json").write_text(json.dumps(refusal, indent=2, default=str), encoding="utf-8")
    written["refusal_json"] = str(out / "export_refusal.json")
    (out / "NOT_SIMULABLE.md").write_text(
        f"# {_HEADLINE}\n\n" + "\n".join(banner.splitlines()[2:]) + "\n\n## Withheld, and why\n\n"
        + "".join(f"- **{f}** — {why}\n" for f, why in (refusal.get("refused") or {}).items())
        + "\n## Do not\n\n" + "".join(f"- {d}\n" for d in refusal.get("do_not") or [])
        + f"\n_{refusal.get('rule')}_\n", encoding="utf-8")
    written["notice"] = str(out / "NOT_SIMULABLE.md")

    where = {"bom": out / "bom", "spec": out / "reports" / "spec_sheet",
             "certificate": out / "verification_certificate"}
    for fmt in refusal.get("refused") or {}:
        stem = where.get(fmt)
        if stem is None:
            continue
        stem.parent.mkdir(parents=True, exist_ok=True)
        doc = {**refusal, "withheld_artifact": fmt, "would_have_been": str(stem) + ".json",
               "why_this_one": REFUSAL_REASON[fmt]}
        (stem.parent / f"{stem.name}.REFUSED.json").write_text(json.dumps(doc, indent=2, default=str),
                                                               encoding="utf-8")
        (stem.parent / f"{stem.name}.REFUSED.md").write_text(
            f"# {stem.name}: REFUSED\n\n{REFUSAL_REASON[fmt]}\n\n" + "\n".join(banner.splitlines()[2:]) + "\n",
            encoding="utf-8")
        written["refused_docs"][fmt] = str(stem.parent / f"{stem.name}.REFUSED.json")

    for target in (out / "robot.xml", out / "robot" / "robot.urdf",
                   *(Path(t) for t in extra_targets if t)):
        if stamp_file(target, banner):
            written["stamped"].append(str(target))
    for extra in (sorted((out / "export").rglob("*.urdf")) + sorted((out / "export").rglob("README*"))
                  + sorted((out / "isaac_lab").rglob("README*"))):
        if stamp_file(extra, banner):
            written["stamped"].append(str(extra))
    cad = out / "cad"
    if cad.is_dir():
        (cad / "NOT_SIMULABLE.md").write_text(
            f"# {_HEADLINE}\n\nThese meshes are your own geometry, transcribed. They were NOT verified: "
            f"{refusal.get('reason')}\n\nSee ../NOT_SIMULABLE.md.\n", encoding="utf-8")
        written["stamped"].append(str(cad / "NOT_SIMULABLE.md"))
    return written
