"""Load the frozen, supplier-sourced component snapshot used by the runnable MVP.

The fixture catalog remains available to unit tests and offline examples, but the product path must not
quietly turn fictional ``VX-*`` parts into a procurement BOM. This adapter keeps the external-data boundary
outside schemas/services and validates provenance before a component can enter resolution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.components import ActuatorSpecs, Component, ComponentCategory, SensorSpecs


@dataclass(frozen=True)
class CatalogSnapshot:
    catalog_id: str
    version: str
    frozen_at: str
    sources: tuple[str, ...]
    components: tuple[Component, ...]

    def manifest(self) -> dict:
        return {
            "catalog_id": self.catalog_id,
            "version": self.version,
            "frozen_at": self.frozen_at,
            "mode": "frozen_supplier_snapshot",
            "sources": list(self.sources),
            "component_ids": [component.id for component in self.components],
            "component_versions": {component.id: component.version for component in self.components},
            "procurement_warning": (
                "Supplier data is frozen for reproducibility; re-check availability, price, revisions, and "
                "safety limits against the live manufacturer page before procurement."
            ),
        }


def load_reference_catalog() -> CatalogSnapshot:
    path = files("virturoid").joinpath("data/reference_components.v1.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    components = tuple(_component(item) for item in raw["components"])
    _validate_snapshot(raw, components)
    return CatalogSnapshot(
        catalog_id=raw["catalog_id"], version=raw["version"], frozen_at=raw["frozen_at"],
        sources=tuple(raw["sources"]), components=components,
    )


def reference_component_catalog() -> list[Component]:
    return list(load_reference_catalog().components)


def reference_catalog_manifest() -> dict:
    return load_reference_catalog().manifest()


def _component(item: dict) -> Component:
    actuator = ActuatorSpecs(**item["actuator"]) if item.get("actuator") else None
    sensor_raw = item.get("sensor")
    sensor = None
    if sensor_raw:
        sensor = SensorSpecs(
            **{**sensor_raw,
               "resolution": tuple(sensor_raw["resolution"]) if sensor_raw.get("resolution") else None,
               "field_of_view_deg": tuple(sensor_raw["field_of_view_deg"])
               if sensor_raw.get("field_of_view_deg") else None,
               "range_m": tuple(sensor_raw["range_m"]) if sensor_raw.get("range_m") else None}
        )
    return Component(
        id=item["id"], version=item["version"], manufacturer=item["manufacturer"],
        part_number=item["part_number"], normalized_name=item["normalized_name"],
        category=ComponentCategory(item["category"]), aliases=list(item.get("aliases", [])),
        datasheet=ArtifactRef(**item["datasheet"]),
        cad_assets=[ArtifactRef(**asset) for asset in item.get("cad_assets", [])],
        mass_kg=item.get("mass_kg"), voltage_range=tuple(item["voltage_range"])
        if item.get("voltage_range") else None, max_current_a=item.get("max_current_a"),
        actuator=actuator, sensor=sensor, spec_confidence=float(item["spec_confidence"]),
        source_urls=list(item["source_urls"]),
    )


def _validate_snapshot(raw: dict, components: tuple[Component, ...]) -> None:
    if not raw.get("version") or not raw.get("frozen_at"):
        raise ValueError("component snapshot requires a version and frozen_at date")
    if len({component.id for component in components}) != len(components):
        raise ValueError("component snapshot contains duplicate ids")
    for component in components:
        result = component.validate()
        if not result.ok:
            raise ValueError(f"invalid supplier component {component.id}: {[i.code for i in result.issues]}")
        if not component.part_number or not component.manufacturer or component.datasheet is None:
            raise ValueError(f"supplier component {component.id} is missing identity/provenance")
        if not component.source_urls or any(not url.startswith("https://") for url in component.source_urls):
            raise ValueError(f"supplier component {component.id} requires HTTPS manufacturer sources")
        if component.version != raw["version"]:
            raise ValueError(f"supplier component {component.id} is not pinned to snapshot {raw['version']}")
