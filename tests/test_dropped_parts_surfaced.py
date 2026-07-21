"""A part that produces no body must be REPORTED, never silently dropped (MVP red-team finding).

Measured before: a graph with a cycle -- or simply authored bottom-up, a part listed before its parent --
compiled to `ok: True` with `coercions: None` and the limbs missing. One probe returned a legless torso
(n_seg=1) and called it a success: the product shipped a different robot than the one authored, with no warning.
That is precisely the silent-substitution the honesty architecture exists to prevent.

`_emit_chain` skips any part whose parent is not built yet, so the drop is structural; the fix reports it.
"""
from __future__ import annotations

from virturoid.services.agent_design_tools import _dropped_part_coercions


class _Seg:
    def __init__(self, name): self.name = name


class _Gene:
    def __init__(self, names): self.segments = [_Seg(n) for n in names]


def _graph(*names_parents):
    return {"name": "g", "parts": [{"name": n, "role": "leg" if p else "body", "parent": p,
                                    "size": 0.2, "girth": 0.03} for n, p in names_parents]}


def test_parts_that_produced_no_segment_are_reported():
    g = _graph(("torso", None), ("armA", "armB"), ("armB", "armA"))   # cycle: neither arm can be emitted
    out = _dropped_part_coercions(g, _Gene(["torso"]))
    assert {c["part"] for c in out} == {"armA", "armB"}
    for c in out:
        assert c["from"] == "authored" and c["to"] == "omitted"
        assert "parent" in c["why"]                                  # tells the author how to fix it


def test_a_part_expanding_into_several_segments_is_not_a_drop():
    """A leg legitimately becomes leg_0/leg_1/leg_2 -- prefix matching must not flag that as omitted."""
    g = _graph(("torso", None), ("leg1", "torso"))
    assert _dropped_part_coercions(g, _Gene(["torso", "leg1_0", "leg1_1", "leg1_2"])) == []


def test_healthy_graph_reports_nothing():
    g = _graph(("torso", None), ("leg1", "torso"), ("leg2", "torso"))
    assert _dropped_part_coercions(g, _Gene(["torso", "leg1", "leg2"])) == []


def test_never_raises_on_a_malformed_graph():
    """Provenance is additive: it must not be able to sink an otherwise valid design."""
    assert _dropped_part_coercions({}, _Gene([])) == []
    assert _dropped_part_coercions({"parts": ["not-a-dict", {"no_name": 1}]}, _Gene([])) == []
    # A gene with no segments means the part really did produce nothing, so REPORTING it is the honest answer
    # here (not silence); the contract being pinned is only that odd input returns a list instead of raising.
    assert isinstance(_dropped_part_coercions({"parts": [{"name": "x"}]}, object()), list)
