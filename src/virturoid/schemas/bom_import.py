"""BOM import schema — Input Ingestion plan, Phase 6 (BOM before CAD, per immediate-recommendation #9).

A robotics team's Bill of Materials is the easiest high-value artifact to normalize: it carries real part
numbers, masses, prices, torques, and voltages. This schema is the typed result of importing a user BOM
(CSV/JSON/YAML/XLSX) — every row classified into canonical fields with a column mapping and provenance — so it
can OVERRIDE or AUGMENT the generated BOM with explicit source tracking. Standard-library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class BomImportItem:
    """One imported BOM row, normalized to the generated-BOM line vocabulary (mass in kg, price in USD)."""

    part: str
    category: str = "unknown"
    qty: int = 1
    unit_mass_kg: float | None = None
    unit_price_usd: float | None = None
    part_number: str | None = None
    torque_nm: float | None = None
    voltage_v: float | None = None
    detail: str = ""
    source_row: int = -1
    provenance: str = "user_bom"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BomImportResult(VersionedEntity):
    source: str = ""
    items: list[BomImportItem] = field(default_factory=list)
    column_mapping: dict = field(default_factory=dict)   # original header -> canonical field
    warnings: list[str] = field(default_factory=list)
    totals: dict = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.items, "items", "An imported BOM must have at least one row.")
        for it in self.items:
            if not it.part:
                result.add("missing_part", f"Row {it.source_row} has no part name.", "items")
        return result

    def to_dict(self) -> dict:
        return {
            "id": self.id, "version": self.version, "source": self.source,
            "items": [it.to_dict() for it in self.items],
            "column_mapping": self.column_mapping, "warnings": self.warnings, "totals": self.totals,
        }
