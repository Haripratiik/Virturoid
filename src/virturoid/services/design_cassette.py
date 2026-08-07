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

PROVENANCE (#208, 2026-08-07). The cassette used to record only *which builder* produced the geometry
(``design_source``), and the bench's ``model`` label was typed by a CLI flag on the run rather than read off
the data. Those two facts together let a number measured on the OFFLINE deterministic builders be reported
under a live-model label. MEASURED on the committed fixture: **0 of 20 rows were authored by a model** — 14
``heuristic`` and 6 ``anatomy_generic``, every one of the latter carrying ``_fallback_gene``'s note "No usable
anatomy model was available", which is reachable only with ``llm is None``.

So provenance is now DERIVED FROM THE ROW, never asserted by the caller:

  * ``authored_by_model`` — is this geometry the model's work? True only for the design sources reachable
    exclusively with a live backend (``llm``, ``llm_anatomy_graph``, ``anatomy``). This is decidable for
    legacy rows too, which is why the committed fixture can be classified without re-recording it.
  * ``llm_calls`` — completions actually billed while designing this row, read off the ``llm_client`` spend
    ledger. Recorded going forward; absent (``None``) on legacy rows.

``DesignCassette.mode()`` aggregates those into ``live`` / ``offline`` / ``mixed``, and Design-Bench refuses a
run label that contradicts it. The rule is the one ``bank_gate`` uses for gait rows: the artifact carries the
evidence for its own provenance, and a claim that outruns the evidence is rejected rather than printed.

TOKEN BOUNDARY. Recording is a DESIGN-PATH activity — a developer re-recording the battery may spend design
tokens (``scripts/design_bench.py --live``). Replay is token-free by construction and is what CI runs. Neither
sits on the customer hot path: nothing in this module is reachable from ``create_robot``/``verify_robot``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

CASSETTE_VERSION = "v1"

# The committed, token-free fixture CI replays. A dev re-record with live tokens can overwrite it explicitly.
DEFAULT_CASSETTE_PATH = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "design_cassette_v1.json"

#: Design sources that ONLY a live backend can produce — every assignment site sits inside an ``llm is not
#: None`` branch of ``morphology_composer`` (``llm`` §propose_morphology_spec, ``llm_anatomy_graph``
#: §_ground_anatomy_graph, ``anatomy`` §_compose_robot_impl). Seeing one of these IS the evidence a model
#: authored the body; nothing else in the record is.
MODEL_AUTHORED_SOURCES = frozenset({"llm", "llm_anatomy_graph", "anatomy", "anatomy_graph"})

#: The deterministic builders. Note ``heuristic``/``anatomy_generic`` are also the landing spot when a live
#: model WAS called and its proposal failed grounding — so these do not prove zero tokens, only that the
#: shipped geometry is not the model's. That is the distinction the funnel actually needs.
BUILDER_AUTHORED_SOURCES = frozenset({"heuristic", "anatomy_generic", "archetype", "anthropometric",
                                      "intent", "offline_composer"})


def authored_by_model(design_source: str | None) -> bool:
    """Did a live model author this geometry? Decidable from the recorded ``design_source`` alone."""
    return str(design_source or "") in MODEL_AUTHORED_SOURCES


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
               source: str = "unknown", error: str | None = None,
               llm_calls: int | None = None, backend: str | None = None, model: str | None = None,
               strict_llm: bool | None = None) -> None:
        """Store (or overwrite) one prompt→design entry. ``gene`` may be None to record a design FAILURE (so the
        funnel's schema/compile denominator is honest — an attempt that produced nothing is still an attempt).

        ``llm_calls``/``backend``/``model``/``strict_llm`` are the measured provenance of THIS attempt (see the
        module docstring). They are written as given — the caller measures, it does not assert."""
        entry = {"prompt": prompt, "source": source, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if gene is not None:
            entry["gene"] = gene.to_dict() if hasattr(gene, "to_dict") else gene
            entry["design_source"] = getattr(gene, "design_source", None)
        if graph is not None:
            entry["graph"] = graph
        if error is not None:
            entry["error"] = str(error)[:300]
        prov = {k: v for k, v in (("llm_calls", llm_calls), ("backend", backend), ("model", model),
                                  ("strict_llm", strict_llm)) if v is not None}
        if prov:
            entry["provenance"] = prov
        self._data.setdefault("entries", {})[prompt_id] = entry

    # -------------------------------------------------------------- provenance (the honest-gate substrate)
    def entry_provenance(self, prompt_id: str) -> dict:
        """Per-row provenance, DERIVED not asserted. ``authored_by_model`` is decidable for every row (legacy
        included) from ``design_source``; ``llm_calls`` is None on rows recorded before it was measured.
        ``failed`` marks a recorded design FAILURE — an attempt that produced no body at all."""
        e = self.entry(prompt_id) or {}
        prov = e.get("provenance") or {}
        return {"prompt_id": prompt_id,
                "design_source": e.get("design_source"),
                "failed": "gene" not in e,
                "authored_by_model": authored_by_model(e.get("design_source")),
                "llm_calls": prov.get("llm_calls"),
                "backend": prov.get("backend"), "model": prov.get("model"),
                "strict_llm": prov.get("strict_llm")}

    def mode(self) -> str:
        """The cassette's measured mode: WHICH GENERATOR was measured, not how well it did.

        ``live`` (every design that exists was authored by a model), ``offline`` (none was), ``mixed`` (both
        generators appear), ``empty``. **Design FAILURES are excluded from the vote**, deliberately: under
        ``strict_llm`` the product refuses rather than substituting a template, so a live run that refuses is
        still a measurement OF THE LIVE PATH. Counting a refusal as evidence of the offline generator would
        relabel the honest strict-mode behaviour as an offline run. The refusals are not hidden — they are
        ``n_failed`` here and schema failures in the funnel's absolute denominator.
        """
        rows = [r for r in (self.entry_provenance(p) for p in self.prompt_ids()) if not r["failed"]]
        if not rows:
            return "empty"
        n_live = sum(1 for r in rows if r["authored_by_model"])
        return "live" if n_live == len(rows) else "offline" if n_live == 0 else "mixed"

    def provenance(self) -> dict:
        """The whole-cassette provenance block that Design-Bench stamps onto its funnel."""
        rows = [self.entry_provenance(p) for p in self.prompt_ids()]
        built = [r for r in rows if not r["failed"]]
        n_live = sum(1 for r in built if r["authored_by_model"])
        calls = [r["llm_calls"] for r in rows if r["llm_calls"] is not None]
        srcs: dict[str, int] = {}
        for r in rows:
            key = "(design failed)" if r["failed"] else str(r["design_source"] or "unknown")
            srcs[key] = srcs.get(key, 0) + 1
        return {"mode": self.mode(), "n_rows": len(rows),
                "n_built": len(built), "n_failed": len(rows) - len(built),
                "n_authored_by_model": n_live, "n_authored_by_builder": len(built) - n_live,
                "design_sources": srcs,
                "llm_calls_total": sum(calls) if calls else None,
                "n_rows_with_measured_calls": len(calls),
                "backends": sorted({r["backend"] for r in rows if r["backend"]}),
                "models": sorted({r["model"] for r in rows if r["model"]}),
                "evidence": ("authored_by_model is read off design_source; the listed sources are assigned only "
                             "inside llm-is-not-None branches of morphology_composer. Design failures are "
                             "counted but do not vote on the mode (see DesignCassette.mode). Rows without "
                             "llm_calls predate provenance recording (#208)."),
                "path": str(self.path)}

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
    import os

    from virturoid.services.llm_client import spend_snapshot
    from virturoid.services.morphology_composer import compose_robot

    def _calls() -> int:
        return int(spend_snapshot()["totals"]["internal_calls"])

    # MEASURE the spend across this one design instead of trusting a flag. The ledger is process-global and
    # counts every internal completion (llm_client._LedgerClient), so the delta is exactly what this attempt
    # billed -- including the calls the intent planner and anatomy designer make underneath compose_robot.
    before = _calls()

    def _prov() -> dict:
        # Read AFTER the attempt: get_llm()'s _load_local_env() is what populates these from the project .env,
        # so reading first would report "no backend" on a machine that has one configured.
        return {"backend": os.environ.get("VIRTUROID_LLM_BACKEND"),
                "model": os.environ.get("VIRTUROID_OPENAI_MODEL") or os.environ.get("VIRTUROID_CLAUDE_MODEL"),
                "strict_llm": bool(strict_llm)}
    try:
        gene = compose_robot(prompt, llm="auto", strict_llm=strict_llm)
        source = getattr(gene, "design_source", "generated") or "generated"
    except Exception as exc:  # noqa: BLE001 - a failed design is DATA (the schema/compile floor), not a crash
        if cassette is not None and record:
            cassette.record(prompt_id, prompt=prompt, gene=None, source="error",
                            error=f"{type(exc).__name__}: {exc}", llm_calls=_calls() - before, **_prov())
        return None, f"error:{type(exc).__name__}"
    if cassette is not None and record:
        cassette.record(prompt_id, prompt=prompt, gene=gene, source=source,
                        llm_calls=_calls() - before, **_prov())
    return gene, source
