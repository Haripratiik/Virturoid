"""The design cassette — record/replay of prompt→design (master_plan_v6 §8.1 / kickoff #2).

The llm layer records spend telemetry only, not the DESIGNS it produced — so a live LLM run's outputs were
unreproducible and CI had nothing deterministic to score. This module fixes that: a **cassette** is a versioned
JSON file mapping each battery prompt to the design the model authored for it (the canonical ``RobotGene`` dict,
plus the authored anatomy graph when one is available). Its three jobs, per the plan:

  1. **Determinism substrate for Design-Bench** — CI replays a committed cassette token-free and scores the exact
     designs, so ``verdict@1`` is a stable, gating number rather than a per-run dice roll.
  2. **The versioned battery artifact** — the cassette IS the record of "what the model designed for v1".
  3. **The WS-B repair-loop corpus** — every recorded prompt→design pair is training/repair material.

Recording routes through the production ``compose_robot`` (LLM-first; the offline heuristic fires only behind
``VIRTUROID_ALLOW_HEURISTIC_FALLBACK`` exactly as elsewhere), so re-recording with dev tokens captures the real
model and catches drift, while an offline record fills a deterministic CI fixture. Replay never generates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

CASSETTE_VERSION = "v1"

# The committed, token-free fixture CI replays. A dev re-record with live tokens can overwrite it explicitly.
DEFAULT_CASSETTE_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "design_cassette_v1.json"


class DesignCassette:
    """A load/record/replay handle over one cassette JSON file. Pure I/O + dict lookup — no design logic."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else DEFAULT_CASSETTE_PATH
        self._data: dict = {"artifact": "virturoid_design_cassette", "version": CASSETTE_VERSION,
                            "battery_version": None, "entries": {}}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._data.setdefault("entries", {})
            except Exception:  # noqa: BLE001 - a corrupt cassette starts empty rather than crashing the harness
                pass

    # -------------------------------------------------------------- read
    def has(self, prompt_id: str) -> bool:
        return prompt_id in self._data.get("entries", {})

    def prompt_ids(self) -> list[str]:
        return sorted(self._data.get("entries", {}))

    def __len__(self) -> int:
        return len(self._data.get("entries", {}))

    def entry(self, prompt_id: str) -> dict | None:
        return self._data.get("entries", {}).get(prompt_id)

    def get_gene(self, prompt_id: str):
        """Rehydrate the recorded design as a ``RobotGene`` (None if absent or unrehydratable)."""
        from virturoid.schemas.gene import RobotGene
        e = self.entry(prompt_id)
        if not e or not isinstance(e.get("gene"), dict):
            return None
        try:
            return RobotGene.from_dict(e["gene"])
        except Exception:  # noqa: BLE001
            return None

    # -------------------------------------------------------------- write
    def record(self, prompt_id: str, *, prompt: str, gene, graph: dict | None = None,
               source: str = "unknown", error: str | None = None) -> None:
        """Store (or overwrite) one prompt→design entry. ``gene`` may be None to record a design FAILURE (so the
        funnel's schema/compile denominator is honest — an attempt that produced nothing is still an attempt)."""
        entry = {"prompt": prompt, "source": source, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if gene is not None:
            entry["gene"] = gene.to_dict() if hasattr(gene, "to_dict") else gene
            entry["design_source"] = getattr(gene, "design_source", None)
        if graph is not None:
            entry["graph"] = graph
        if error is not None:
            entry["error"] = str(error)[:300]
        self._data.setdefault("entries", {})[prompt_id] = entry

    def save(self, *, battery_version: str | None = None) -> Path:
        if battery_version is not None:
            self._data["battery_version"] = battery_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
        return self.path

    def summary(self) -> dict:
        entries = self._data.get("entries", {})
        sources: dict[str, int] = {}
        for e in entries.values():
            sources[e.get("source", "unknown")] = sources.get(e.get("source", "unknown"), 0) + 1
        return {"version": self._data.get("version"), "battery_version": self._data.get("battery_version"),
                "n_entries": len(entries), "n_failures": sum(1 for e in entries.values() if "gene" not in e),
                "sources": sources, "path": str(self.path)}


def design_from_prompt(prompt: str, *, prompt_id: str, cassette: DesignCassette | None = None,
                       allow_generate: bool = False, record: bool = False, strict_llm: bool = False):
    """The single design entry point Design-Bench uses. Returns ``(gene_or_None, source)``.

    * If a cassette holds ``prompt_id`` → replay it (token-free, deterministic) — this is the CI path.
    * Else if ``allow_generate`` → author via the production ``compose_robot`` (LLM-first) and, if ``record``,
      write it back to the cassette. This is the (re-)record path. ``strict_llm`` forbids the heuristic fallback.

    A generation that raises is recorded as a design failure (gene=None) so the funnel's denominator stays honest.
    """
    if cassette is not None and cassette.has(prompt_id):
        return cassette.get_gene(prompt_id), "cassette"
    if not allow_generate:
        return None, "absent"
    from virturoid.services.morphology_composer import compose_robot
    try:
        gene = compose_robot(prompt, llm="auto", strict_llm=strict_llm)
        source = getattr(gene, "design_source", "generated") or "generated"
    except Exception as exc:  # noqa: BLE001 - a failed design is DATA (the schema/compile floor), not a crash
        if cassette is not None and record:
            cassette.record(prompt_id, prompt=prompt, gene=None, source="error", error=f"{type(exc).__name__}: {exc}")
        return None, f"error:{type(exc).__name__}"
    if cassette is not None and record:
        cassette.record(prompt_id, prompt=prompt, gene=gene, source=source)
    return gene, source
