"""Inbound BOM importer — Input Ingestion plan, Phase 6.

Parses a user-supplied Bill of Materials (CSV / JSON / YAML / XLSX-if-openpyxl) into a typed
:class:`BomImportResult`: a column classifier maps arbitrary headers to canonical fields, masses/prices are
unit-normalized (g->kg, lb->kg), duplicate part rows are merged, and :func:`reconcile_with_generated` overlays
the imported lines onto a generated BOM with explicit provenance. Deterministic and standard-library only
(YAML/XLSX are optional, degraded gracefully). No network (the plan's local-only import rule).
"""

from __future__ import annotations

import csv
import json
import os
import re

from virturoid.schemas.bom_import import BomImportItem, BomImportResult

# canonical field <- header-substring rules. Ordered: earlier rules win (part_number before part).
_COLUMN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("part_number", ("part_number", "partnumber", "part number", "part num", "part no", "part#", "mpn", "sku")),
    ("part", ("part", "component", "name", "description", "item", "model")),
    ("category", ("category", "type", "class", "kind")),
    ("qty", ("qty", "quantity", "count", "amount", "number")),
    ("unit_price_usd", ("unit_price", "price", "cost", "usd", "$")),
    ("unit_mass_kg", ("mass", "weight")),
    ("torque_nm", ("torque",)),
    ("voltage_v", ("voltage", "volt")),
    ("detail", ("detail", "notes", "note", "spec", "remark", "comment")),
)


def classify_columns(headers: list[str]) -> dict[str, str]:
    """Map each raw header to a canonical field (first matching rule wins; unmatched headers are dropped)."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for header in headers:
        lowered = (header or "").strip().lower()
        if not lowered:
            continue
        for canonical, needles in _COLUMN_RULES:
            if canonical in used:
                continue
            if any(n in lowered for n in needles):
                mapping[header] = canonical
                used.add(canonical)
                break
    return mapping


def _mass_factor(header: str) -> float:
    """kg per unit for a mass column, inferred from its header (defaults to kg)."""
    h = header.lower()
    if "kg" in h:
        return 1.0
    if "lb" in h or "pound" in h:
        return 0.453592
    if re.search(r"\(g\)|_g\b|\bg\b|gram|grams|\bmg\b", h):
        return 0.001 if "mg" not in h else 1e-6
    return 1.0


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = re.sub(r"[^0-9.\-eE]", "", str(value))
    try:
        return float(s) if s not in ("", "-", ".") else None
    except ValueError:
        return None


def _to_int(value, default=1):
    f = _to_float(value)
    return int(round(f)) if f is not None else default


def parse_bom_rows(rows: list[dict], *, source: str = "bom", bom_id: str | None = None) -> BomImportResult:
    """Core: classify + normalize a list of header->value row dicts into a :class:`BomImportResult`.

    Duplicate part rows (same normalized part+part_number) are merged by summing quantity. A mass column whose
    header lacks a unit is assumed to be kilograms, with a warning.
    """
    warnings: list[str] = []
    if not rows:
        return BomImportResult(id=bom_id or "bom_import", source=source, items=[], warnings=["empty BOM"])

    headers = list(rows[0].keys())
    mapping = classify_columns(headers)
    if "part" not in mapping.values():
        warnings.append("no part/name column recognized; rows may be unusable.")

    mass_header = next((h for h, c in mapping.items() if c == "unit_mass_kg"), None)
    mass_factor = _mass_factor(mass_header) if mass_header else 1.0
    if mass_header and "kg" not in mass_header.lower() and mass_factor == 1.0:
        warnings.append(f"mass column '{mass_header}' has no unit; assuming kilograms.")

    by_key: dict[str, BomImportItem] = {}
    order: list[str] = []
    for i, row in enumerate(rows):
        canon: dict = {}
        for header, value in row.items():
            field_name = mapping.get(header)
            if field_name:
                canon[field_name] = value
        part = str(canon.get("part", "")).strip()
        if not part:
            warnings.append(f"row {i} has no part name; skipped.")
            continue
        item = BomImportItem(
            part=part,
            category=str(canon.get("category", "unknown")).strip() or "unknown",
            qty=_to_int(canon.get("qty"), 1),
            unit_mass_kg=(None if canon.get("unit_mass_kg") in (None, "")
                          else round((_to_float(canon.get("unit_mass_kg")) or 0.0) * mass_factor, 6)),
            unit_price_usd=_to_float(canon.get("unit_price_usd")),
            part_number=(str(canon["part_number"]).strip() if canon.get("part_number") else None),
            torque_nm=_to_float(canon.get("torque_nm")),
            voltage_v=_to_float(canon.get("voltage_v")),
            detail=str(canon.get("detail", "")).strip(),
            source_row=i,
        )
        key = f"{item.part.lower()}|{(item.part_number or '').lower()}"
        if key in by_key:                                       # duplicate/alternate -> merge quantities
            by_key[key].qty += item.qty
            warnings.append(f"merged duplicate part '{item.part}' (row {i}).")
        else:
            by_key[key] = item
            order.append(key)

    items = [by_key[k] for k in order]
    total_mass = round(sum((it.unit_mass_kg or 0.0) * it.qty for it in items), 4)
    total_price = round(sum((it.unit_price_usd or 0.0) * it.qty for it in items), 2)
    return BomImportResult(
        id=bom_id or "bom_import", source=source, items=items, column_mapping=mapping, warnings=warnings,
        totals={"line_items": len(items), "mass_kg": total_mass, "price_usd": total_price,
                "unit_count": sum(it.qty for it in items)},
    )


def parse_bom_file(path: str) -> BomImportResult:
    """Parse a BOM file by extension: .csv / .json / .yaml|.yml / .xlsx (openpyxl). Returns a BomImportResult."""
    ext = os.path.splitext(path.lower())[1]
    bom_id = "bom_" + re.sub(r"[^a-z0-9]+", "_", os.path.basename(path).lower()).strip("_")
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return parse_bom_rows(rows, source=path, bom_id=bom_id)
    if ext == ".json":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data if isinstance(data, list) else data.get("items") or data.get("lines") or []
        return parse_bom_rows(rows, source=path, bom_id=bom_id)
    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            return BomImportResult(id=bom_id, source=path, items=[], warnings=["PyYAML not installed"])
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        rows = data if isinstance(data, list) else (data or {}).get("items") or (data or {}).get("lines") or []
        return parse_bom_rows(rows, source=path, bom_id=bom_id)
    if ext == ".xlsx":
        return _parse_xlsx(path, bom_id)
    return BomImportResult(id=bom_id, source=path, items=[], warnings=[f"unsupported BOM extension '{ext}'"])


def _parse_xlsx(path: str, bom_id: str) -> BomImportResult:
    try:
        import openpyxl
    except ImportError:
        return BomImportResult(id=bom_id, source=path, items=[],
                               warnings=["openpyxl not installed; export the BOM as CSV/JSON instead."])
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        headers = [str(h) if h is not None else "" for h in next(rows_iter)]
    except StopIteration:
        return BomImportResult(id=bom_id, source=path, items=[], warnings=["empty spreadsheet"])
    rows = [dict(zip(headers, r)) for r in rows_iter]
    return parse_bom_rows(rows, source=path, bom_id=bom_id)


def reconcile_with_generated(generated_bom: dict, imported: BomImportResult) -> dict:
    """Overlay the imported BOM onto a generated one with provenance.

    A generated line whose part matches an imported item is OVERRIDDEN (mass/price/detail from the user BOM,
    marked ``provenance='user_bom'``); imported items with no generated match are AUGMENTED as new lines. Every
    output line carries ``provenance`` so downstream sizing/power/cost knows which numbers the user supplied.
    """
    gen_lines = [dict(ln) for ln in (generated_bom.get("lines") or [])]
    by_part = {str(ln.get("part", "")).lower(): ln for ln in gen_lines}
    matched: set[str] = set()                                  # lowercased keys (for the added-filter)
    overridden: list[str] = []                                 # original part names (for the summary)

    for it in imported.items:
        key = it.part.lower()
        target = by_part.get(key)
        if target is not None:
            matched.add(key)
            overridden.append(it.part)
            if it.unit_mass_kg is not None:
                target["unit_mass_kg"] = it.unit_mass_kg
                target["mass_kg"] = round(it.unit_mass_kg * (target.get("qty", it.qty) or it.qty), 6)
            if it.unit_price_usd is not None:
                target["unit_price_usd"] = it.unit_price_usd
                target["price_usd"] = round(it.unit_price_usd * (target.get("qty", it.qty) or it.qty), 2)
            if it.part_number:
                target["part_number"] = it.part_number
            if it.detail:
                target["detail"] = it.detail
            target["provenance"] = "user_bom"
        else:
            gen_lines.append({
                "part": it.part, "category": it.category, "qty": it.qty,
                "unit_mass_kg": it.unit_mass_kg, "unit_price_usd": it.unit_price_usd,
                "mass_kg": round((it.unit_mass_kg or 0.0) * it.qty, 6),
                "price_usd": round((it.unit_price_usd or 0.0) * it.qty, 2),
                "part_number": it.part_number, "detail": it.detail, "provenance": "user_bom_added",
            })

    for ln in gen_lines:
        ln.setdefault("provenance", "generated")
    total_mass = round(sum(ln.get("mass_kg", 0.0) or 0.0 for ln in gen_lines), 4)
    total_price = round(sum(ln.get("price_usd", 0.0) or 0.0 for ln in gen_lines), 2)
    out = dict(generated_bom)
    out["lines"] = gen_lines
    out["totals"] = {**generated_bom.get("totals", {}), "line_items": len(gen_lines),
                     "mass_kg": total_mass, "price_usd": total_price}
    out["bom_reconciliation"] = {
        "overridden": overridden, "added": [it.part for it in imported.items if it.part.lower() not in matched],
        "source": imported.source,
    }
    return out
