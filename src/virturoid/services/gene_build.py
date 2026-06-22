"""Gene-driven build + evaluation: run a real task on a robot generated from a gene.

This closes the keystone gap — it takes any ``RobotGene``, compiles it to MuJoCo with a
generated scene, and runs the **real** pick-and-place controller on it, returning honest
success. Because the controller is morphology-agnostic (it discovers actuated joints + the
``ee_site``), the *same* code evaluates a humanoid upper-body or a novel body exactly as the
tabletop arm — so "build a humanoid" now produces and runs a humanoid, not the templated arm.

It also writes a minimal gene-native package (compiled scenes + a genome derived from the
gene + a report) that the existing 3D viewer can replay via ``simulation/scene_set.json`` +
``simulation/mujoco/compiled_scene_index.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.gene import RobotGene
from virturoid.services.gene_compiler import compile_gene_with_scene


def default_controller_params(gene: RobotGene) -> dict | None:
    """Class-aware controller defaults: heavier humanoid limbs need stronger gains + more
    settle time than the light tabletop arm. None = use the controller's own defaults."""
    if gene.robot_class == "humanoid":
        # Co-designed controller (scripts/search_humanoid.py) paired with the tuned humanoid gene.
        return {"kp": 60.35, "kd": 2.36, "phase_steps": 352}
    return None


def evaluate_gene_pick_place(gene: RobotGene, scenes, params: dict | None = None,
                             assignments: dict | None = None) -> dict:
    """Run the real pick-place controller on a gene-compiled robot across scenes.

    Returns honest aggregate success plus per-scene outcomes. Requires MuJoCo. ``assignments`` is the
    task's object->target map; WITHOUT it the controller falls back to the hardcoded red/blue sort, so
    any other task (place_to_target / transport / spray) tracks NO objects and reports a vacuous 100% —
    pass the task spec's assignments so co-design optimizes the REAL task (§22 honest metrics).
    """
    import mujoco

    from virturoid.services.pick_place_controller import run_pick_place_episode

    if params is None:
        params = default_controller_params(gene)
    episodes = []
    for scene in scenes:
        xml = compile_gene_with_scene(gene, scene.objects)
        model = mujoco.MjModel.from_xml_string(xml)
        mujoco.mj_forward(model, mujoco.MjData(model))  # surfaces compile errors early
        objects = {o.name: tuple(o.pose_xyz_rpy[:3]) for o in scene.objects}
        task = {"objects": objects}
        if assignments:
            task["assignments"] = assignments
        out = run_pick_place_episode(model, task, params=params)
        episodes.append({"scene_id": scene.id, **{k: out[k] for k in ("status", "failure_label", "placed_count", "block_count")}})

    n = len(episodes)
    placed = sum(e["placed_count"] for e in episodes)
    total = sum(e["block_count"] for e in episodes)
    return {
        "gene_id": gene.id,
        "species": gene.species,
        "robot_class": gene.robot_class,
        "episodes": episodes,
        "success_rate": round(sum(1 for e in episodes if e["status"] == "success") / n, 3) if n else 0.0,
        "blocks_placed": placed,
        "blocks_total": total,
    }


def generate_pick_place_scenes(prompt: str, count: int = 6, purpose: str = "variation"):
    """Generate pick-and-place scenes (reusing the deterministic generator) for a gene eval."""
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.scene_generator import generate_scene_set
    from virturoid.services.task_builder import build_task_graph

    req = build_requirements_from_prompt(prompt)
    task = build_task_graph(req)
    return generate_scene_set(task, count=count, purpose=purpose).scenes


def robot_reachable_points(gene: RobotGene, *, z: float = 0.06, ring: int = 24, radial: int = 7,
                           tol: float = 0.05, grasp_frac: float = 0.72) -> list:
    """Sample the compiled robot's reliably-GRASPABLE workspace at grasp height.

    Returns (x, y) points the robot's end effector can reach (IK FK within ``tol``), so
    scenes can be matched to THIS robot rather than a fixed annulus tuned for the arm.
    This is what makes a torso-mounted humanoid get blocks placed where it can work.

    CRITICAL: IK-REACHABLE is not GRASPABLE. Near full extension the arm can touch a target but has no joint
    margin left to descend onto it and lift it (measured: grasp succeeds to ~0.72 of max reach, then fails
    'gripped_no_lift'/'no_grasp_contact'). So the outer ring is capped at ``grasp_frac`` of max reach — objects
    are placed only where the arm can actually pick them up. Sampling the full reachable annulus (to 0.95 reach)
    was the pick_place robustness bug: blocks at 0.78-0.95 reach -> missed_grasp, success_rate ~0.33.
    """
    import math

    import mujoco
    import numpy as np

    from virturoid.services.pick_place_controller import plan_joint_targets

    model = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, []))
    reach = max(0.05, sum(s.length_m for s in gene.actuated_joints()) + 0.05)
    inner = 0.36 * grasp_frac / 0.72                      # inner ring proportional to the graspable cap
    arc = math.radians(58)                                # reliable forward half-arc: a top-down grasp needs
    #                                                       the wrist pointing DOWN, which it can't at the far
    #                                                       sides — measured grasp fails past ~±60deg.
    points = []
    for ri in range(1, radial + 1):
        for ai in range(ring):
            ang = -arc + 2 * arc * ai / (ring - 1)          # forward graspable cone (not the full 180deg)
            side = abs(ang) / arc                           # 0 straight-ahead -> 1 at the cone edge
            cap = grasp_frac * (1.0 - 0.16 * side)          # graspable reach is a TEARDROP: shorter at the sides
            r = reach * (inner + (cap - inner) * ri / radial)
            x, y = round(r * math.cos(ang), 4), round(r * math.sin(ang), 4)
            if x < 0.12:                                   # keep targets in front, off the base
                continue
            _, dist = plan_joint_targets(model, np.array([x, y, z]), iterations=8, candidates=64, seed=ri * 100 + ai)
            if dist <= tol:
                points.append((x, y))
    return points


def evaluate_genes_batch(genes: list, scenes, params: dict | None = None) -> list[dict]:
    """Evaluate many genes (the vectorized-ready interface for scaled co-design, Phase B).

    Today this runs MuJoCo sequentially. A GPU-parallel backend (MuJoCo MJX / Brax) plugs in
    behind this same signature to evaluate thousands of candidate genes at once — the only
    change is the inner executor, not the callers (co-design, surrogate training).
    """
    return [evaluate_gene_pick_place(g, scenes, params=params) for g in genes]


def maybe_scene_envelope(prompt: str) -> dict | None:
    """Consult the Scene Agent for a randomization envelope, if an LLM backend is configured.

    Wires the (previously dead) Scene Agent into generation: its validated envelope tunes the
    robot-matched scenes (object count, mass range). Gated + offline-safe (None -> defaults).
    """
    try:
        from virturoid.services.llm_client import get_llm
        from virturoid.services.requirements_builder import build_requirements_from_prompt
        from virturoid.services.scene_agent import propose_scene_envelope

        llm = get_llm("scene")
        if llm is None:
            return None
        result = propose_scene_envelope(prompt, build_requirements_from_prompt(prompt), llm)
        return result["envelope"] if result and result.get("feasible") else None
    except Exception:  # noqa: BLE001 - advisory
        return None


def generate_reachable_scenes(gene: RobotGene, count: int = 6, *, seed: int = 0, envelope: dict | None = None):
    """Build pick-place scenes with blocks + bins placed in THIS robot's reachable workspace.

    Robot-matched scenes (Phase B): instead of a fixed annulus tuned for the tabletop arm,
    objects are sampled from where the compiled robot can actually reach — so a humanoid (or
    any gene) is evaluated on a task it can physically perform.
    """
    import random

    from virturoid.schemas.scenes import SceneGraph, SceneObject

    # Scene Agent envelope (gated, optional) tunes object mass + separation; defaults otherwise.
    mass_lo, mass_hi = (envelope or {}).get("mass_range_kg", [0.05, 0.05])
    min_sep = float((envelope or {}).get("min_separation_m", 0.16))

    pts = robot_reachable_points(gene)
    if len(pts) < 4:
        return generate_pick_place_scenes(gene.species, count=count)  # fallback

    def far_enough(p, chosen, d=min_sep):
        return all((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 >= d * d for c in chosen)

    scenes = []
    for i in range(count):
        rng = random.Random(seed + i)
        pool = pts[:]
        rng.shuffle(pool)
        chosen = []
        for p in pool:
            if far_enough(p, chosen):
                chosen.append(p)
            if len(chosen) == 4:
                break
        if len(chosen) < 4:
            chosen = pool[:4]
        (rbx, rby), (bbx, bby), (rnx, rny), (bnx, bny) = chosen[:4]
        objects = [
            SceneObject("red_block", "cube", (rbx, rby, 0.02, 0, 0, round(rng.uniform(-3.14, 3.14), 3)),
                        mass_kg=round(rng.uniform(mass_lo, mass_hi), 4), material="red_block", friction=1.0, scale=1.0),
            SceneObject("blue_block", "cube", (bbx, bby, 0.02, 0, 0, round(rng.uniform(-3.14, 3.14), 3)),
                        mass_kg=round(rng.uniform(mass_lo, mass_hi), 4), material="blue_block", friction=1.0, scale=1.0),
            SceneObject("red_bin", "container", (rnx, rny, 0.0, 0, 0, 0), material="red_bin"),
            SceneObject("blue_bin", "container", (bnx, bny, 0.0, 0, 0, 0), material="blue_bin"),
        ]
        scenes.append(SceneGraph(
            id=f"reach_{gene.id}_{i}", version="0.1.0", name=f"{gene.species} reachable scene {i + 1}",
            backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0), objects=objects,
            variation_parameters={"robot_matched": True}, requirement_trace=["reachable_workspace"]))
    return scenes


def _export_real_cad(gene: RobotGene, output_dir: Path) -> dict | None:
    """Write REAL B-rep CAD (per-link STEP+STL + assembly STEP) for the gene via build123d, plus a manifest.
    Returns the manifest, or ``None`` when build123d/the kernel is unavailable — in which case NO file is
    written and the readiness ledger honestly marks CAD 'scaffolded'. FAIL-CLOSED: we never emit a fake STEP
    here (that is exactly the 479-byte placeholder the product audit flagged). The gene path holds a real
    ``RobotGene`` with ``.segments``, so ``export_gene_cad`` (proven: 30-290 KB real STEPs) applies directly."""
    try:
        import json as _json

        from virturoid.services.cad_geometry import export_gene_cad
        manifest = export_gene_cad(gene, str(Path(output_dir) / "cad"))
        if manifest.get("part_count"):
            (Path(output_dir) / "cad" / "cad_manifest.json").write_text(
                _json.dumps(manifest, indent=2), encoding="utf-8")
            return manifest
        return None
    except Exception:  # noqa: BLE001 - missing build123d / kernel failure -> honest 'no real CAD', never fake
        return None


def build_gene_package(gene: RobotGene, prompt: str, output_dir: Path, scene_count: int = 6,
                       controller_params: dict | None = None, gated_export: bool = False,
                       export_tier_counts: dict | None = None) -> dict:
    """Compile a gene into a viewer-replayable package and evaluate it in real physics.

    Writes per-scene MJCF, a compiled-scene index, a scene_set.json, a genome-from-gene, and
    an evaluation report — the minimal artifacts the 3D viewer and reports consume. Returns
    the evaluation summary (honest success rate). ``controller_params`` (e.g. the controller a
    co-design step produced) is used for the evaluation so the built robot is scored with the SAME
    brain it was tuned with — otherwise build and co-design report inconsistent success (§22).

    ``gated_export`` (default OFF) additionally runs the §15.3 tiered baseline+randomized robustness gate
    (``gated_evaluation_run``) on a manipulator and persists ``reports/evaluation_run.json``, which the
    readiness ledger then ENFORCES (a robot that ran real physics but fails the tiered gate reads
    ``physics_evaluated=below_gate``, not export-ready). It is OPT-IN because the tiered loop is a heavier,
    rigorous check meant for an export-readiness verdict, not every fast iterative build. Operates on the REAL
    composed gene (no genome->gene reconstruction), so the gate scores exactly the robot that was built.
    """
    output_dir = Path(output_dir)
    # A legged body is scored on LOCOMOTION (forward walking distance), not pick-place — a quadruped has
    # no gripper and no reachable object workspace. Dispatch by structure so any free-base legged morphology
    # the composer/LLM produces is evaluated on the task it actually implies (see task_matched_eval).
    from virturoid.services.task_matched_eval import robot_kind

    if robot_kind(gene) == "legged":
        return _build_legged_package(gene, prompt, output_dir, controller_params)

    # Robot-matched scenes: objects in this robot's reachable workspace (Phase B), optionally
    # tuned by the Scene Agent's validated envelope (gated by VIRTUROID_LLM_BACKEND).
    # The task is SELECTED from the prompt (Phase 2): sort / place-to-target / transport, so a
    # "lift boxes" or "carry" request runs that task, not a hard-coded sort.
    from virturoid.services.task_runtime import evaluate_gene_on_task, generate_task_scenes, select_task_spec

    spec = select_task_spec(prompt)
    scenes = generate_task_scenes(gene, spec, count=scene_count)

    # Per-scene MJCF + compiled index (the viewer reads these).
    scene_dir = output_dir / "simulation" / "mujoco" / "scenes" / "variation"
    scene_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in scenes:
        rel = f"simulation/mujoco/scenes/variation/{scene.id}.xml"
        (output_dir / rel).write_text(compile_gene_with_scene(gene, scene.objects), encoding="utf-8")
        entries.append({"scene_set_id": f"sceneset_{gene.id}", "scene_id": scene.id,
                        "purpose": "variation", "mujoco_xml": rel, "object_count": len(scene.objects)})
    (output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").write_text(
        json.dumps({"id": f"compiled_{gene.id}", "robot_genome_id": gene.id, "backend": "mujoco",
                    "scene_count": len(entries), "scenes": entries}, indent=2), encoding="utf-8")

    # scene_set.json (the viewer's default entry point). Carries the task's object->target
    # assignments so the viewer replays the SAME task that was evaluated.
    (output_dir / "simulation").mkdir(parents=True, exist_ok=True)
    (output_dir / "simulation" / "scene_set.json").write_text(json.dumps({
        "id": f"sceneset_{gene.id}", "version": "0.1.0", "task_graph_id": f"task_{gene.id}",
        "purpose": "variation", "task_type": spec.task_type, "assignments": spec.assignments,
        "scenes": [_scene_to_dict(s) for s in scenes],
    }, indent=2), encoding="utf-8")

    # A genome-from-gene so reports/viewer have the structure (links/joints/ee from the gene).
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    (output_dir / "robot" / "robot_genome.json").write_text(json.dumps(_gene_to_genome(gene), indent=2), encoding="utf-8")

    cad = _export_real_cad(gene, output_dir)   # REAL B-rep STEP/STL (the gene path shipped no CAD before)
    summary = evaluate_gene_on_task(gene, spec, scenes, params=controller_params)
    summary["cad_real"] = bool(cad)
    summary["cad_part_count"] = (cad or {}).get("part_count", 0)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "gene_evaluation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # REAL contact-grasp certification: for a gripper/hand manipulator, run the no-pin friction grasp+lift
    # (grasp_eval.evaluate_grasp_lift closes the fingers and lifts the object by CONTACT FRICTION — not the
    # idealized pin the sort/transport loop uses) and write its honest report. The readiness gate reads
    # grasp_model="contact" from this and certifies a REAL grasp (physics_evaluated ATTAINED), while the sort
    # report stays disclosed as idealized_pin. Best-effort: a degenerate gripper just doesn't certify.
    if str(getattr(gene, "end_effector_type", "") or "").lower() in ("gripper", "hand"):
        try:
            from virturoid.services.grasp_eval import evaluate_grasp_lift
            gr = evaluate_grasp_lift(gene)
            (output_dir / "reports" / "grasp_evaluation_report.json").write_text(json.dumps({
                "task_type": "grasp_lift", "grasp_model": "contact",
                "success_rate": gr.get("success_rate"), "mean_lift_m": gr.get("mean_lift"),
                "max_lift_m": gr.get("max_lift"), "mean_tilt_deg": gr.get("mean_tilt_deg"),
                "attempts": gr.get("attempts"),
                "note": "Real friction grasp+lift: the fingers close on the object and hold it by CONTACT FRICTION "
                        "(no _pin_block teleport). The sort/transport loop separately uses an idealized pin.",
            }, indent=2), encoding="utf-8")
            summary["contact_grasp_success_rate"] = gr.get("success_rate")
        except Exception:  # noqa: BLE001 - certification is best-effort; the sort report still gates honestly
            pass
    summary["bom"] = _emit_bom(gene, output_dir, task=prompt)  # real per-joint actuators + sensors + materials
    # OPT-IN §15.3 export gate: run the tiered baseline+randomized robustness loop on the REAL gene and persist
    # reports/evaluation_run.json BEFORE the ledger, so _emit_readiness ENFORCES it (manipulator-only — the gate is
    # pick-place tiered; best-effort so it never breaks a build).
    if gated_export and robot_kind(gene) == "manipulator":
        try:
            from virturoid.services.evaluation_loop import gated_evaluation_run
            gr = gated_evaluation_run(gene, prompt=prompt, package_dir=output_dir,
                                      params=controller_params, tier_counts=export_tier_counts)
            summary["export_gate"] = {"export_ready": gr["decision"]["export_ready"],
                                      "tier_success": gr["decision"]["tier_success"],
                                      "blockers": gr["decision"]["export_blockers"]}
        except Exception:  # noqa: BLE001 - the export gate is best-effort; never block a build
            pass
    summary["readiness"] = _emit_readiness(gene, output_dir)   # honest truth-gate (this path had NONE before)
    return summary


# A from-scratch scripted gait that clears this distance is a "walking" baseline; the score normalizes
# against it so the as-built robot reads honestly (the trained MorphPolicy is what lifts it further).
_LOCOMOTION_TARGET_FORWARD_M = 1.0          # normalizer for the success_rate magnitude (NOT the 'walked' bar)
_WALK_MIN_FORWARD_M = 0.15                   # the 'actually walked' distance bar (shared with readiness_ledger)


def _build_legged_package(gene: RobotGene, prompt: str, output_dir: Path, controller_params: dict | None) -> dict:
    """Build + evaluate a LEGGED gene on locomotion (forward walking), the task its morphology implies.

    Unlike the pick-place path there are no object scenes — a walker is scored on how far it travels while
    staying upright (``task_matched_eval.evaluate_robot`` -> ``run_locomotion_episode``). We still write the
    genome and the evaluation report (the artifacts the report/viewer consume), and return a summary in the
    same shape callers read (``success_rate`` + ``task_type``), with ``task_type='locomotion'`` so the build
    message, report, and memory record all reflect a walker rather than a misleading pick-place score.
    """
    from virturoid.services.task_matched_eval import evaluate_robot

    output_dir = Path(output_dir)
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    (output_dir / "robot" / "robot_genome.json").write_text(json.dumps(_gene_to_genome(gene), indent=2), encoding="utf-8")

    # Write the LOCOMOTION scene (robot on flat ground) as a real compiled MJCF + index. A walker's
    # "simulation" is the locomotion rollout, not object scenes — but it IS a compiled, loadable scene, so
    # emitting it lets the readiness ledger's sim_compiled stage attain honestly (it was 'not_run' before,
    # which kept every legged package from ever reading as ready).
    try:
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        scene_dir = output_dir / "simulation" / "mujoco" / "scenes" / "locomotion"
        scene_dir.mkdir(parents=True, exist_ok=True)
        rel = f"simulation/mujoco/scenes/locomotion/{gene.id}.xml"
        (output_dir / rel).write_text(
            compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)), encoding="utf-8")
        (output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").write_text(json.dumps({
            "id": f"compiled_{gene.id}", "robot_genome_id": gene.id, "backend": "mujoco", "scene_count": 1,
            "scenes": [{"scene_set_id": f"sceneset_{gene.id}", "scene_id": f"locomotion_{gene.id}",
                        "purpose": "locomotion", "mujoco_xml": rel, "object_count": 0}]},
            indent=2), encoding="utf-8")
        # scene_set.json (the viewer's entry point). A walker has no object scenes, but the viewer/episode
        # replay still reads this file — without it simulate_episode_for_viewer raised FileNotFoundError and
        # the desktop viewport showed "3D render unavailable" for EVERY legged/humanoid build.
        (output_dir / "simulation" / "scene_set.json").write_text(json.dumps({
            "id": f"sceneset_{gene.id}", "version": "0.1.0", "task_graph_id": f"task_{gene.id}",
            "purpose": "locomotion", "task_type": "locomotion",
            "scenes": [{"id": f"locomotion_{gene.id}", "name": "locomotion (forward walk)", "objects": []}],
        }, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - scene emission is best-effort; the eval below is the real signal
        pass

    # Score the walker with the SAME controller the VIEWER replays (the banked recipe policy), so the build's
    # headline number can never contradict the 3D viewport — the audit caught the scripted-eval reporting a
    # BACKWARD "walked" while the viewport showed a FORWARD fall. Fall back to the scripted eval only when no
    # policy is banked yet (then both the report and the viewer use the scripted gait).
    from virturoid.services.learn_locomotion import banked_policy_for
    pol = banked_policy_for(gene, models_dir="models")
    cadence = upright_frac = None; n_feet = 0
    if pol is not None and getattr(pol, "obs_mean", None) is not None:
        from virturoid.services.morph_policy import recipe_rollout_morph
        rr = recipe_rollout_morph(gene, pol, steps=900)              # matches viewer_sim._locomotion_episode
        forward_m = float(rr.get("forward", 0.0))
        cadence = float(rr.get("cadence", 0.0)); upright_frac = float(rr.get("upright_frac", 0.0))
        n_feet = int(rr.get("n_feet", 0))
        # HONEST 'walked' gate (anti-Goodhart): genuine forward distance AND sustained-tall posture AND a real
        # stepping cadence — so an upright SLIDE (cadence 0) or a forward TOPPLE no longer certifies as a walk.
        upright = upright_frac >= 0.6
        stepping = cadence >= 1.0 or n_feet < 2                      # a legged body must lift feet; non-legged skip
        walked = forward_m > _WALK_MIN_FORWARD_M and upright and stepping
        status = "walked" if walked else ("upright" if upright_frac >= 0.5 else "fell")
        distance_m = abs(forward_m)
    else:
        res = evaluate_robot(gene, prompt=prompt, controller_params=controller_params)
        d = res.get("detail", {})
        forward_m = float(d.get("forward_m", 0.0)); upright = bool(d.get("upright", False))
        status = d.get("status", "unknown"); distance_m = float(d.get("distance_m", 0.0))
        walked = status == "walked"
    # success_rate is GATED on genuinely walking — a slide/topple earns 0, not partial credit for raw displacement.
    success_rate = round(min(1.0, max(0.0, forward_m) / _LOCOMOTION_TARGET_FORWARD_M), 3) if walked else 0.0
    summary = {
        "task_type": "locomotion",
        "species": gene.species,
        "robot_class": gene.robot_class,
        "success_rate": success_rate,
        "distance_m": distance_m,
        "forward_m": forward_m,
        "upright": upright,
        "status": status,
        "cadence_hz": cadence,
        "upright_frac": upright_frac,
    }
    cad = _export_real_cad(gene, output_dir)   # REAL B-rep STEP/STL for the walker too
    summary["cad_real"] = bool(cad)
    summary["cad_part_count"] = (cad or {}).get("part_count", 0)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "gene_evaluation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["bom"] = _emit_bom(gene, output_dir, task=prompt)  # real per-joint actuators + sensors + materials
    summary["readiness"] = _emit_readiness(gene, output_dir)
    return summary


def _emit_bom(gene: RobotGene, output_dir: Path, task: str = "") -> dict:
    """Write the robot's REAL bill of materials — per-joint actuators (sized to each joint's torque), link
    materials, the TASK-ADAPTIVE sensor suite (camera eyes, a navigator's LiDAR, an inspector's thermal cam),
    compute and power — as machine-readable JSON + a human BOM table, and stash it on the gene. Returns totals."""
    try:
        from virturoid.services.bom_builder import build_bom, format_bom_markdown
        bom = build_bom(gene, task=task)
        (output_dir / "robot").mkdir(parents=True, exist_ok=True)
        (output_dir / "robot" / "bill_of_materials.json").write_text(json.dumps(bom, indent=2), encoding="utf-8")
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports" / "bill_of_materials.md").write_text(format_bom_markdown(bom), encoding="utf-8")
        gene.metadata["bom"] = bom
        return bom.get("totals", {})
    except Exception:  # noqa: BLE001 - the BOM is value-add; never let it break a build
        return {}


def _emit_readiness(gene: RobotGene, output_dir: Path) -> dict:
    """Emit the honest Product Readiness Ledger for a gene package (EMIT-ONLY: reports the truth, does not yet
    block). The gene/autonomous path previously emitted NO validator/readiness at all — this gives it the same
    truth-gate the audit demanded, including the spawn/stability check it was missing."""
    try:
        from virturoid.services.readiness_ledger import write_product_readiness_ledger
        ledger = write_product_readiness_ledger(output_dir, robot_class=gene.robot_class, gene=gene,
                                                enforce=False)
        return {"safe_to_export": ledger.safe_to_export, "highest_attained": ledger.highest_attained}
    except Exception:  # noqa: BLE001 - readiness emission must never break a build
        return {}


def _gene_to_genome(gene: RobotGene) -> dict:
    return {
        "id": f"genome_{gene.id}",
        "name": gene.species,
        "species": gene.species,
        "robot_class": gene.robot_class,
        "links": [s.name for s in gene.segments],
        "joints": [
            {"name": f"{s.name}_joint", "joint_type": s.joint_type,
             "parent_link": s.parent, "child_link": s.name}
            for s in gene.segments if s.joint_type in ("revolute", "prismatic")
        ],
        "end_effectors": [s.name for s in gene.segments if s.is_end_effector],
        "source": "gene_compiler",
    }


def _scene_to_dict(scene) -> dict:
    return {
        "id": scene.id, "version": getattr(scene, "version", "0.1.0"), "name": scene.name,
        "backend_targets": list(scene.backend_targets),
        "robot_spawn_xyz_rpy": list(scene.robot_spawn_xyz_rpy or (0, 0, 0, 0, 0, 0)),
        "objects": [
            {"name": o.name, "object_type": o.object_type, "pose_xyz_rpy": list(o.pose_xyz_rpy),
             "mass_kg": o.mass_kg, "material": o.material, "friction": o.friction, "scale": o.scale}
            for o in scene.objects
        ],
        "variation_parameters": dict(getattr(scene, "variation_parameters", {})),
        "requirement_trace": list(getattr(scene, "requirement_trace", [])),
    }
