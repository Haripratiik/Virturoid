"""Import-any-robot CI matrix (breakthrough v5, B5): the clearest public 'general' signal. Run the SAME pipeline
— build -> ground (real parts) -> roll out under the shared recipe -> executable-on-BOM certificate — over a
diverse set of morphologies (quadruped / hexapod / octopod / anatomy-graph creature / imported MJCF) and emit one
comparable row per robot. A single harness that certifies a frog, a hexapod, and an imported arm without special-
casing is the honest evidence that the tool is general, not a demo tuned to one body.

Every robot is isolated: a build/rollout/cert failure on one becomes an error ROW, never a crashed matrix. Pure
CPU (recipe rollout + BOM cert); the GPU trainer is an optional per-robot upgrade the caller injects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RobotSpec:
    name: str
    factory: object                    # callable() -> RobotGene (or an object with .segments)
    robot_class: str = "legged"
    material: str = "carbon_fiber"
    fill: float = 0.25


def default_matrix() -> list[RobotSpec]:
    """A diverse, self-contained morphology set (no external assets needed). Covers leg-count generality + a
    distinct anatomy-graph topology so the matrix exercises the general compiler, not one template."""
    from virturoid.services.steerable_body import steerable_quadruped

    specs = [
        RobotSpec("quad4", lambda: steerable_quadruped(n_legs=4), "quadruped"),
        RobotSpec("hex6", lambda: steerable_quadruped(n_legs=6, bilateral=True), "hexapod"),
        RobotSpec("octo8", lambda: steerable_quadruped(n_legs=8, bilateral=True), "octopod"),
    ]
    try:                                                     # a genuinely different topology via the anatomy graph
        from virturoid.services.anatomy_compiler import build_from_anatomy
        graph = {"robot_class": "quadruped", "parts": [
            {"name": "torso", "role": "body", "size": 0.5, "girth": 0.4},
            {"name": "head", "role": "head", "parent": "torso", "attach": "front_top", "aim": "forward", "size": 0.12},
            {"name": "leg", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down",
             "size": 0.28, "segments": 3, "symmetry": "left_right"},
            {"name": "tail", "role": "tail", "parent": "torso", "attach": "rear_top", "aim": "back", "size": 0.2}]}
        specs.append(RobotSpec("anatomy_creature", lambda g=graph: build_from_anatomy(g), "quadruped"))
    except Exception:  # noqa: BLE001 - anatomy compiler optional
        pass
    return specs


def spec_from_mjcf(name: str, mjcf_path: str, robot_class: str = "imported") -> RobotSpec:
    """Import an external robot (MuJoCo Menagerie / a URDF-converted MJCF) as a CI row via the robot_mjcf
    passthrough, if available. Lets the matrix certify real third-party robots alongside generated ones."""
    def _factory():
        from virturoid.services.robot_mjcf import load_gene_from_mjcf  # optional passthrough
        return load_gene_from_mjcf(mjcf_path)
    return RobotSpec(name, _factory, robot_class)


def run_ci_matrix(specs: list[RobotSpec] | None = None, *, ground: bool = True, steps: int = 300,
                  n_seeds: int = 2, cpg: dict | None = None, out_path: str | None = None) -> dict:
    """Run every spec through build -> (ground) -> recipe rollout -> BOM certificate. Returns a matrix report:
    per-robot {built, n_segments, mass_kg, survived, cadence, cert_pass, cert_gates, error}. Robots are
    independent; one failure is an error row, not a crash."""
    from virturoid.services.morph_policy import CPG_DEFAULT, recipe_rollout_morph
    from virturoid.services.bom_certificate import certify_policy_on_bom

    specs = specs or default_matrix()
    cpg = cpg if cpg is not None else {**CPG_DEFAULT, "calf_phase": 0.0}
    rows: list[dict] = []
    for s in specs:
        row = {"name": s.name, "class": s.robot_class, "built": False, "error": ""}
        try:
            gene = s.factory()
            row["n_segments"] = len(getattr(gene, "segments", []))
            if ground:
                from virturoid.services.grounded_physics import ground_gene
                ground_gene(gene, material=s.material, fill=s.fill)
            row["mass_kg"] = round(sum(getattr(seg, "mass_kg", 0.0) or 0.0 for seg in gene.segments), 2)
            row["built"] = True
            r = recipe_rollout_morph(gene, None, steps=steps, seed=0, cpg=cpg)
            row["survived"] = bool(r.get("survived"))
            row["cadence"] = r.get("cadence")
            cert = certify_policy_on_bom(gene, None, steps=steps, n_seeds=n_seeds, cpg=cpg)
            row["cert_pass"] = cert["pass"]
            row["cert_gates"] = f"{cert['n_gates_pass']}/6"
        except Exception as e:  # noqa: BLE001 - isolate per-robot failure
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    n_built = sum(r["built"] for r in rows)
    n_cert = sum(1 for r in rows if r.get("cert_pass"))
    report = {"n_robots": len(rows), "n_built": n_built, "n_certified": n_cert, "rows": rows,
              "summary": f"{n_built}/{len(rows)} built, {n_cert}/{len(rows)} certified on the real BOM"}
    if out_path:
        import json
        import os
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report
