"""Intent router for the build entry (fixes the #1 generality bottleneck: keyword lookup that discarded
the request's qualifiers/composition).

Given any request, an LLM picks the BEST-FIT real model from the Menagerie catalog *considering the
qualifiers* (reach, size, payload, type) — or routes to the procedural generative path when no real model
fits (novel morphology: snake/hexapod/tentacle/modular) or the request needs parts the catalog can't
supply as one model. Falls back to the deterministic keyword router when no LLM is available.
"""

from __future__ import annotations

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "base": {"type": "string", "enum": ["real", "procedural"]},
        "model_key": {"type": "string"},
        "attachments": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["base", "reason"],
}


def _catalog_text() -> str:
    from virturoid.services.real_model_library import _REGISTRY
    seen, lines = set(), []
    for key, (_mod, label, kind) in _REGISTRY.items():
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"  - {key}: {label} ({kind})")
    return "\n".join(lines)


def plan_build(prompt: str, llm=None) -> dict:
    """Return ``{base, model_key, attachments, reason, source}``. ``base`` is 'real' (use ``model_key`` from
    the catalog) or 'procedural' (generative). Uses ``llm`` if given; else a deterministic keyword fallback."""
    from virturoid.services.real_model_library import _REGISTRY, real_model_key

    if llm is not None:
        system = (
            "You route a robot-build request to the BEST option. Either pick ONE real model from this "
            "catalogue (choose by FIT — match the type AND qualifiers like reach, size, payload, dexterity) "
            "or choose the procedural generative path when NO catalogue model genuinely fits — i.e. a novel "
            "morphology (snake, hexapod, tentacle, modular, soft, wheeled-legged) OR a request that needs "
            "parts combined that no single model provides.\n"
            f"CATALOGUE (key: label (kind)):\n{_catalog_text()}\n"
            "Rules: prefer a longer-reach/bigger arm for 'long reach'/'heavy-duty'; a multi-finger HAND for "
            "'tentacle'/'dexterous', never a parallel-jaw 'gripper' for those; 'mobile manipulator' for "
            "base+arm. If you pick real, model_key MUST be a catalogue key. List 'attachments' (e.g. 'arm', "
            "'gripper') the user asked for that the base lacks. Output JSON only."
        )
        try:
            out = llm.complete_json(system, f"Request: {prompt}", _PLAN_SCHEMA, max_tokens=300) or {}
            base = out.get("base")
            key = (out.get("model_key") or "").strip()
            if base == "real" and key in _REGISTRY:
                return {"base": "real", "model_key": key, "attachments": out.get("attachments", []),
                        "reason": out.get("reason", ""), "source": "llm"}
            if base == "procedural":
                return {"base": "procedural", "model_key": None, "attachments": out.get("attachments", []),
                        "reason": out.get("reason", ""), "source": "llm"}
        except Exception:  # noqa: BLE001 - any LLM/parse failure -> deterministic fallback below
            pass

    key = real_model_key(prompt)
    if key:
        return {"base": "real", "model_key": key, "attachments": [], "reason": "keyword match", "source": "keyword"}
    return {"base": "procedural", "model_key": None, "attachments": [], "reason": "no catalogue match",
            "source": "keyword"}
