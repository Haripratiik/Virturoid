"""Unified robot build entry point.

The product's job is to CREATE ORIGINAL robot designs — so by default it always GENERATES a body via the
composer (``morphology_composer``), never copies a real production model. Copying a real robot (e.g. serving
a Unitree H1 for "a humanoid") would defeat the purpose; real models are kept only as an OPT-IN reference /
private benchmark to measure our generated designs against (``prefer_real=True``, e.g. an "import a reference
robot" action), not as the product output. A generated body is carried as a ``RobotGene``; an opt-in real
reference is carried as a portable MJCF string (same string-based viewport/import path as an imported model).
"""

from __future__ import annotations


def build_robot(prompt: str, *, prefer_real: bool = False, reach_m: float | None = None,
                payload_kg: float | None = None, llm="auto") -> dict:
    """Return ``{kind: 'procedural'|'real', prompt, ...}``. Default GENERATES an original design ('procedural':
    ``gene``). ``prefer_real=True`` is the opt-in reference path (returns 'real': ``mjcf``, ``label``, …) — used
    only when the user explicitly wants to import/benchmark a real production model, never for original design."""
    if prefer_real:
        from virturoid.services.build_planner import plan_build
        from virturoid.services.real_model_library import real_model_mjcf
        router_llm = None
        if llm == "auto":
            try:
                from virturoid.services.llm_client import get_llm
                router_llm = get_llm("scene")          # a fast LLM understands intent; keyword fallback if None
            except Exception:  # noqa: BLE001
                router_llm = None
        elif llm not in (None, "none"):
            router_llm = llm
        plan = plan_build(prompt, router_llm)
        if plan["base"] == "real" and plan["model_key"]:
            info = real_model_mjcf(robot_class=plan["model_key"])
            if info.get("ok"):
                return {"kind": "real", "prompt": prompt, "mjcf": info["mjcf"], "label": info["label"],
                        "robot_kind": info["kind"], "free_base": info.get("free_base", False),
                        "actuated": info["actuated"], "bodies": info["bodies"], "meshes": info["meshes"],
                        "attachments": plan.get("attachments", []), "route_reason": plan.get("reason", ""),
                        "route_source": plan.get("source", "")}
    from virturoid.services.gene_validation import validate_gene_design
    from virturoid.services.grounded_physics import ground_gene
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot(prompt, reach_m=reach_m, payload_kg=payload_kg, llm=llm)
    # Validate BEFORE grounding (on the design torques) so the actuator-margin checks aren't double-counted:
    # ground_gene rewrites each joint torque to its real actuator's stall, which would inflate a re-derived BOM.
    validation = validate_gene_design(gene, payload_kg=float(payload_kg or 0.0))
    grounding = ground_gene(gene)           # Stage 2: real masses (material+geometry) + real actuators + BOM
    build_plan = dict((gene.metadata or {}).get("build_plan", {}))
    return {"kind": "procedural", "prompt": prompt, "gene": gene, "grounding": grounding,
            "validation": validation, "intent": build_plan,
            "requires_clarification": bool(build_plan and not build_plan.get("buildable", True))}
