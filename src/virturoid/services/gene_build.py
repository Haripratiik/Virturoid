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
    except ImportError:  # build123d absent -> NON-SILENT: tell the user how to get real CAD (don't fake it)
        import warnings
        warnings.warn("build123d is not installed, so this build ships NO CAD (the readiness ledger will mark "
                      "CAD 'scaffolded', not real). Install it for real B-rep STEP/STL: pip install -e \".[sim]\" "
                      "(or \".[cad]\").", RuntimeWarning, stacklevel=2)
        return None
    except Exception:  # noqa: BLE001 - kernel failure -> honest 'no real CAD', never fake
        return None


# A limb must be able to HOUSE the actuator that drives it: the visual housing may exceed the limb
# radius by at most this factor. 1.5x keeps a believable "shell over motor" look while still letting
# the real datasheet envelope show; beyond it the render reads as a drum threaded on a pencil.
_MAX_HOUSING_TO_LIMB = 1.5


#: what ``ground_and_repair`` uses for a body that has NEVER been grounded (the validated-trainable B0 config).
DEFAULT_GROUNDING = {"material": "carbon_fiber", "fill": 0.25}


def grounding_config(gene: RobotGene) -> dict:
    """How this body's masses were derived -- so re-grounding REPRODUCES it instead of re-deriving it.

    ``metadata['mass_source'] == 'source_model'`` (an imported customer robot) means the per-link masses are
    the manufacturer's own and must never be replaced by our primitive-volume estimate; ``preserve_mass`` is
    then True and the material only sizes actuators. Otherwise the material/fill ``ground_gene`` last recorded
    wins, falling back to :data:`DEFAULT_GROUNDING`.

    ``preserve_geometry`` is the same rule for SHAPE, and it is keyed on ``metadata['imported_from']``: every
    link length and radius on an imported body is a measurement of a machine that already exists, so the two
    geometry-editing passes in :func:`ground_and_repair` (stress thickening, actuator-housing fit) must report
    what they find rather than silently re-cut the customer's robot. It stays True across an amend, because
    what a customer holds and verifies after their own edit is still what the export door has to ship
    unchanged; a deliberate edit moves the geometry through ``edit_operators``, which is the customer asking.
    """
    meta = getattr(gene, "metadata", None) or {}
    rec = meta.get("grounding") if isinstance(meta.get("grounding"), dict) else {}
    try:
        fill = float(rec.get("fill", DEFAULT_GROUNDING["fill"]))
    except (TypeError, ValueError):
        fill = float(DEFAULT_GROUNDING["fill"])
    return {
        "material": str(rec.get("material") or DEFAULT_GROUNDING["material"]),
        "fill": fill if 0.0 < fill <= 1.0 else float(DEFAULT_GROUNDING["fill"]),
        "preserve_mass": str(meta.get("mass_source") or "") == "source_model",
        "preserve_geometry": bool(str(meta.get("imported_from") or "")
                                  or str(meta.get("mass_source") or "") == "source_model"),
        "source": ("imported" if str(meta.get("mass_source") or "") == "source_model"
                   else ("recorded" if rec else "default")),
    }


def ground_and_repair(gene: RobotGene, *, task: str = "") -> dict:
    """GROUND a body to real actuators + material mass, then structurally repair under-margined links and
    re-ground so mass tracks the fix. This is the SINGLE grounding path both exit doors share -- the
    package builder (``build_gene_package``) and the agent's ``export_held`` -- so one prompt ships the SAME
    buildable robot (mass, actuators, safety factor) whichever door it leaves by. Before this was shared, the
    two doors diverged (export_held emitted the ungrounded 3.57 kg fantasy body while the builder grounded to
    ~8 kg with real motors).

    GROUND THE BODY WE WERE HANDED, NOT A DIFFERENT ONE. This used to hardcode ``carbon_fiber``/0.25, and
    ``ground_gene`` re-derives every link mass from geometry x density x fill -- so a body grounded anywhere
    else silently changed weight on its way out the door. Measured: a Go2 held and VERIFIED at 27.362 kg
    (aluminium, 2700 kg/m3) exported at 19.151 kg (carbon fibre, 1600) -- 1600/2700 = 0.593 exactly, on all 13
    links -- while the certificate travelling in the same package still claimed the verdict was signed by the
    body that deploys. An IMPORTED robot was worse: its manufacturer's real per-link masses were replaced by
    primitive-volume estimates (Menagerie Go2, 15.206 -> 13.235 kg, base 6.921 -> 6.107).

    So: honour the gene's OWN recorded grounding (``metadata['grounding']``, stamped by ``ground_gene``), and
    PRESERVE authoritative masses outright (``metadata['mass_source'] == 'source_model'`` -- an imported
    customer robot). carbon_fiber/0.25 remains the default for a body that has never been grounded, which is
    the case this function was written for and the validated-trainable config from the B0 GPU runs.

    Rationale (fidelity gap-closure §G3): ``ground_gene`` sizes real actuators on RATED torque and sets link
    mass = structure + motor, so sim, eval, spec sheet and BOM all describe one robot. §G3b: the added actuator
    mass can drop a link below the SF>=2.0 structural target the readiness ledger fail-closes on, so thicken
    only the under-margined links (SF ~ r^3, +5% headroom) and re-ground -- walking legs stay inside the
    slenderness band (length/diameter >= 2.25) so the fix never re-creates stub legs. Fail-open: a grounding
    error is reported, never raised.

    ...AND THE SAME PROTECTION FOR SHAPE AS FOR MASS. Both of those passes RE-CUT the body, which is right for
    a body we drew and wrong for one the customer handed us: on an imported robot every radius is a measurement
    of hardware that exists. Measured on a Menagerie Go2 once the importer started reading the customer's real
    torque limits (23.7/23.7/45.43 N.m instead of a 0.7-5.1 N.m structural guess): the housing-fit pass sized
    the catalog equivalent of their knee motor at 110 mm and grew all four calves 0.02903 -> 0.03667 m (+26%)
    and all four hips 0.02674 -> 0.02733 m, at unchanged mass -- so the shipped URDF had a different inertia
    than the body the verdict was earned on, and ``export_held`` correctly reported ``same_as_verified: false``
    on a robot nobody had edited. The premise does not even hold on real hardware: a Go2's knee actuator sits
    at the joint, not inside the shin, and a one-radius capsule cannot say so. So for an imported body both
    passes REPORT (``housing_over_limb``, ``understrength_links``) instead of editing, and the parts list still
    names the part that covers the joint -- what changes is that we stop reshaping their robot to fit it."""
    _g = grounding_config(gene)
    _mat, _fill, _keep = _g["material"], _g["fill"], _g["preserve_mass"]
    _hold_shape = bool(_g["preserve_geometry"])
    _findings: dict = {}
    try:
        from virturoid.services.grounded_physics import ground_gene
        report = ground_gene(gene, material=_mat, fill=_fill, preserve_mass=_keep)
        from virturoid.services.structural_validation import validate_structure
        for _ in range(3):
            _st = validate_structure(gene)
            if _st.get("ok", False):
                break
            if _hold_shape:                       # their shape, their call: name the weak links, don't re-cut them
                _findings["understrength_links"] = [
                    {"link": str(_l.get("name")), "safety_factor": round(float(_l.get("safety_factor") or 0.0), 2)}
                    for _l in _st.get("links", []) if not _l.get("feasible")][:12]
                break
            _by_name = {s.name: s for s in gene.segments}
            for _link in _st.get("links", []):
                _s = _by_name.get(str(_link.get("name")))
                _sf = float(_link.get("safety_factor") or 0.0)
                if _link.get("feasible") or _s is None or _sf <= 0.0 or float(_s.radius_m) <= 0.0:
                    continue
                _f = min(1.6, 1.05 * (2.0 / _sf) ** (1.0 / 3.0))
                _r = float(_s.radius_m) * _f
                if ("leg" in _s.name.lower() and (_s.joint_type or "").lower() == "revolute"
                        and float(_s.length_m) > 0):
                    _r = min(_r, float(_s.length_m) / 4.5)
                _s.radius_m = round(max(float(_s.radius_m), _r), 5)
            report = ground_gene(gene, material=_mat, fill=_fill, preserve_mass=_keep)
        # A LINK MUST BE BIG ENOUGH TO HOUSE THE MOTOR THAT DRIVES IT (visual-fidelity defect T1). Actuator
        # housings are drawn at their true datasheet envelope, which is right -- but nothing forced the LIMB to
        # be able to carry that part, so the compiler emitted a 5.6 mm rod mounting a 98 mm-diameter T-Motor
        # AK10-9 (housing/limb = 8.86x on a hexapod hip; 3.23x worst on the dog). That is both physically
        # impossible and the single most damaging "toy" cue in every render: fat drums threaded on pencil
        # sticks. Real limbs enclose their actuator, so GROW THE LIMB rather than shrink an honest datasheet.
        # Bounded by the same slenderness rule the loop above uses, so a leg never becomes a stub, and followed
        # by a re-ground so mass/inertia track the new geometry.
        #
        # ON AN IMPORTED BODY THE CONCLUSION FLIPS. There the datasheet part is a catalog EQUIVALENT for a motor
        # the customer already owns and already fitted, so a limb thinner than our stand-in is evidence our
        # stand-in is bigger than their motor -- not evidence their limb is wrong. Record the ratio and leave
        # the measurement alone; ``spec_sheet``/``ingest_project`` surface it.
        try:
            from virturoid.services.component_geometry import actuator_for_joint
            _grew, _over = False, []
            for _s in gene.segments:
                _ds = actuator_for_joint(_s)
                if _ds is None or float(_s.radius_m or 0.0) <= 0.0:
                    continue
                _env = _ds["envelope_m"]
                _need = (max(float(_env[0]), float(_env[1])) / 2.0) / _MAX_HOUSING_TO_LIMB
                # NB: no slenderness clamp here. That rule exists to stop the STRESS pass above from fattening a
                # long limb into a stub, but applied to physical fit it does the opposite harm: a SHORT segment
                # (a hip abduction block) would be pinned thinner than the motor bolted to it -- which is how the
                # hexapod ended up as a 5.6 mm rod under a 98 mm motor. A short, chunky joint housing is what a
                # real hip assembly looks like; fitting the part is the honest constraint, so let it win.
                if _need <= float(_s.radius_m):
                    continue
                if _hold_shape:
                    _over.append({"link": _s.name, "catalog_part": _ds.get("part"),
                                  "part_diameter_m": round(max(float(_env[0]), float(_env[1])), 4),
                                  "limb_diameter_m": round(2.0 * float(_s.radius_m), 4),
                                  "ratio": round(_need / float(_s.radius_m), 2)})
                    continue
                _s.radius_m = round(_need, 5)
                _grew = True
            if _over:
                _findings["housing_over_limb"] = _over[:12]
            if _grew:
                report = ground_gene(gene, material=_mat, fill=_fill, preserve_mass=_keep)
        except Exception:  # noqa: BLE001 - housing fit is a fidelity pass, never a build blocker
            pass
    except Exception as _ge:  # noqa: BLE001
        report = {"error": f"{type(_ge).__name__}: {_ge}"}
    # ...AND THEN BOLT ON THE PARTS THE PARTS LIST ALREADY SPECIFIES. Grounding gives a link its structure and
    # its motor; the battery, the compute and the sensors were still weightless, so the verdict, the actuator
    # sizing and the torque margins were all signed against a body lighter than the machine the customer would
    # build (MEASURED: an authored hexapod 15.043 kg simulated vs 17.894 kg of real parts, 19%). It runs LAST,
    # after every re-ground above, because a re-ground re-derives link masses and would drop what it added --
    # and it declines outright on an imported body, whose manufacturer masses already include their own
    # electronics. Fail-open, like everything else here: a parts list that cannot be built is reported.
    #
    # ``task`` MATTERS HERE and must be the one the shipped parts list is built with. The sensor suite is
    # task-adaptive -- a navigator gets a LiDAR, an inspector a thermal camera -- so embodying against a
    # task-neutral BOM and then shipping ``_emit_bom(task=prompt)`` puts different hardware in the two, and the
    # sim/BOM equality this exists to establish quietly fails by the difference (measured on a warehouse rover:
    # 0.655 kg of LiDAR and drive hardware the body never carried).
    try:
        from virturoid.services.grounded_physics import embody_component_masses
        _emb = embody_component_masses(gene, task=task)
        if isinstance(report, dict):
            report["component_embodiment"] = _emb
    except Exception:  # noqa: BLE001 - embodiment is a fidelity pass, never a build blocker
        pass
    if isinstance(report, dict):
        report["geometry_preserved"] = _hold_shape
        report.update(_findings)
    return report


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
    _ground_report = ground_and_repair(gene, task=prompt)
    try:                                                   # persist grounding + the executable-on-BOM certificate
        import json as _json
        _rep_dir = output_dir / "reports"
        _rep_dir.mkdir(parents=True, exist_ok=True)
        (_rep_dir / "grounding_report.json").write_text(_json.dumps(_ground_report, indent=2), encoding="utf-8")
        from virturoid.services.bom_certificate import certify_policy_on_bom
        certify_policy_on_bom(gene, None, steps=250, n_seeds=1,
                              out_path=str(_rep_dir / "bom_certificate.json"))
    except Exception:  # noqa: BLE001 - the certificate is a report, never a build blocker
        pass
    try:                                                   # G6: design-time morphology gate (report, ledger-visible)
        import json as _json
        from dataclasses import asdict as _asdict
        from virturoid.services.morphology_priors import validate_morphology
        (output_dir / "reports" / "morphology_report.json").write_text(
            _json.dumps(_asdict(validate_morphology(gene)), indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    # A legged body is scored on LOCOMOTION (forward walking distance), not pick-place — a quadruped has
    # no gripper and no reachable object workspace. Dispatch by structure so any free-base legged morphology
    # the composer/LLM produces is evaluated on the task it actually implies (see task_matched_eval).
    from virturoid.services.task_matched_eval import robot_kind

    kind = robot_kind(gene)
    if kind == "legged":
        return _build_legged_package(gene, prompt, output_dir, controller_params)
    if kind in ("aerial", "aquatic"):
        # a drone flies / a fish swims — score on that free-motion task, NOT pick-place (which reported a drone
        # 0% at generic_manipulation because it fell through to the manipulation path with no gripper).
        return _build_freemotion_package(gene, prompt, output_dir, controller_params)
    if kind == "mobile":
        # A wheeled mobile base navigates — it has no gripper, so it must NOT fall through to the pick-place
        # path. It builds its real floor course (navigation route or maze, chosen from the prompt by the
        # general scene generator) and is scored on navigation.
        return _build_navigation_package(gene, prompt, output_dir, controller_params, scene_count)

    # Robot-matched scenes: objects in this robot's reachable workspace (Phase B), optionally
    # tuned by the Scene Agent's validated envelope (gated by VIRTUROID_LLM_BACKEND).
    # The task is SELECTED from the prompt (Phase 2): sort / place-to-target / transport, so a
    # "lift boxes" or "carry" request runs that task, not a hard-coded sort.
    from virturoid.services.task_runtime import evaluate_gene_on_task, generate_task_scenes, select_task_spec

    spec = select_task_spec(prompt)
    # SORT keeps the reachability-matched generator + its tested spec; richer manipulation tasks (lift onto a
    # shelf, stack a tower, push to a target) render their OWN scene from the general generator, with the
    # object->target assignments DERIVED from that scene (no hard-coded layout).
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.scene_generator import generate_scene_set
    from virturoid.services.task_builder import build_task_graph
    _task = build_task_graph(build_requirements_from_prompt(prompt))
    if _task.task_type == "pick_place_sort":
        scenes = generate_task_scenes(gene, spec, count=scene_count)
    else:
        # Take the task STRUCTURE (objects/targets/assignments) from the general generator, then place it in the
        # robot's REACHABLE workspace (the dexterous zone the sort path uses) so the controller can actually reach
        # and place every object -- the arbitrary layout left non-sort targets out of reach (shelf dropped, stack
        # missed-grasp). Falls back to the arbitrary layout only if the reachable matcher can't fit the objects.
        spec = _spec_from_scene(spec, generate_scene_set(_task, count=1, purpose="variation").scenes[0])
        reachable = generate_task_scenes(gene, spec, count=scene_count)
        scenes = reachable or generate_scene_set(_task, count=scene_count, purpose="variation").scenes

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

    # REAL contact-grasp sort replay for the viewer (no _pin_block): record the gripper closing on each block by
    # CONTACT FRICTION, lifting, carrying it to its bin, releasing -- so the 3D demo shows actual physics, not the
    # idealized pin the live sort loop uses. The viewer (simulate_episode_for_viewer) prefers this file and falls
    # back to the pin only when it's absent (older packages) or the gripper is degenerate. Best-effort.
    _record_contact_replay(gene, scenes, spec, output_dir)

    # A genome-from-gene so reports/viewer have the structure (links/joints/ee from the gene).
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    _write_genome_and_urdf(gene, output_dir)

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
    _maybe_write_summaries(output_dir)                         # spec sheet + deployment guide (reports/)
    _maybe_write_spec_compliance(output_dir, gene, prompt)     # requested-vs-achieved spec report (§4.8E)
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
    _emit_mvp_artifacts(gene, output_dir, task_type=_task.task_type, package_type="gene_package")
    summary["readiness"] = _emit_readiness(gene, output_dir)   # honest truth-gate (this path had NONE before)
    return summary


def _record_contact_replay(gene: RobotGene, scenes, spec, output_dir: Path) -> bool:
    """Record a no-pin contact-grasp SORT episode on scene 0 to simulation/contact_episode.json so the viewer
    replays a REAL friction grasp (the gripper closes on each block and holds it by CONTACT), not the idealized-pin
    sort loop. Returns True iff a successful replay was written. Best-effort: only for a gripper/hand that actually
    places at least one block -- a missed/dropped grasp leaves no file so the viewer keeps the honest pin fallback."""
    if str(getattr(gene, "end_effector_type", "") or "").lower() not in ("gripper", "hand"):
        return False
    try:
        import mujoco

        from virturoid.services.pick_place_controller import run_contact_pick_place_episode
        from virturoid.services.viewer_sim import _geom_metadata

        # Find the first scene whose contact SORT cleanly places EVERY block: a reachability-matched layout can
        # still drop one, and a half-failed replay is worse demo than a clean pin. Bounded so the build stays fast.
        chosen = None
        for sc in scenes[: min(4, len(scenes))]:
            frames: list = []
            ep = run_contact_pick_place_episode(gene, sc.objects, spec.assignments, record_frames=frames, frame_every=12)
            if frames and ep.get("placed_count", 0) >= max(1, ep.get("block_count", 1)):
                chosen = (sc, frames, ep)
                break
        if chosen is None:
            return False
        sc, frames, ep = chosen
        # geoms from the SAME deterministic compile the episode ran on, so geom order matches the recorded frames.
        cmj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, sc.objects))
        (output_dir / "simulation").mkdir(parents=True, exist_ok=True)
        (output_dir / "simulation" / "contact_episode.json").write_text(json.dumps({
            "scene_id": sc.id, "grasp_model": "contact",
            "geoms": _geom_metadata(cmj), "frames": frames,
            "outcome": {"status": ep.get("status"), "failure_label": ep.get("failure_label"),
                        "placed_count": ep.get("placed_count"), "block_count": ep.get("block_count"),
                        "grasp_model": "contact", "metrics": ep.get("metrics")},
        }, indent=2), encoding="utf-8")
        # Make the contact-reliable scene the viewer's default (index 0) so SCENE mode and the EPISODE replay show the
        # same layout -- the per-scene MJCF + compiled index are keyed by scene_id, so only the display order changes.
        ss_path = output_dir / "simulation" / "scene_set.json"
        try:
            ss = json.loads(ss_path.read_text(encoding="utf-8"))
            ss["scenes"].sort(key=lambda s: 0 if s.get("id") == sc.id else 1)
            ss_path.write_text(json.dumps(ss, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 - reorder is cosmetic; the replay is keyed by scene_id regardless
            pass
        return True
    except Exception:  # noqa: BLE001 - replay is best-effort; the viewer falls back to the pin episode
        return False


# A from-scratch scripted gait that clears this distance is a "walking" baseline; the score normalizes
# against it so the as-built robot reads honestly (the trained MorphPolicy is what lifts it further).
_LOCOMOTION_TARGET_FORWARD_M = 1.0          # normalizer for the success_rate magnitude (NOT the 'walked' bar)
_WALK_MIN_FORWARD_M = 0.15                   # the 'actually walked' distance bar (shared with readiness_ledger)


def _robot_footprint_radius(gene: RobotGene) -> float:
    """Half the robot's xy footprint (a characteristic size) so floor scenes scale to THIS robot. Compiled
    table-less so only the robot's own geoms are measured; small default if MuJoCo is unavailable."""
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [], table=False))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        rmax = 0.0
        for gi in range(m.ngeom):
            x, y = float(d.geom_xpos[gi, 0]), float(d.geom_xpos[gi, 1])
            sz = float(max(m.geom_size[gi]))
            rmax = max(rmax, (x * x + y * y) ** 0.5 + sz)
        return max(0.1, min(rmax, 0.6))
    except Exception:  # noqa: BLE001
        return 0.2


def _spec_from_scene(fallback_spec, scene):
    """Derive object->target assignments from a GENERATED task scene so the as-built pick-place controller can
    attempt any manipulation scene (lift onto a shelf, stack a tower, push to a target) — no hard-coded layout.
    Each manipulable pairs with a same-coloured target when one exists, else the nearest; falls back to the
    tested spec if the scene has no usable manipulable/target pair."""
    from virturoid.services.task_runtime import PickPlaceTaskSpec

    def colour(o):
        m = (getattr(o, "material", "") or "").lower()
        for c in ("red", "blue", "green", "orange", "yellow"):
            if c in m:
                return c
        return "gray"

    manip = [o for o in scene.objects if o.object_type in ("cube", "box")]
    targets = [o for o in scene.objects if o.object_type in ("container", "zone", "platform")]
    if not manip or not targets:
        return fallback_spec
    materials = {o.name: colour(o) for o in scene.objects}
    assignments = {}
    for o in manip:
        same = [t for t in targets if materials.get(t.name) == materials.get(o.name)]
        if same:
            tgt = same[0]
        else:
            ox, oy = o.pose_xyz_rpy[0], o.pose_xyz_rpy[1]
            tgt = min(targets, key=lambda t: (t.pose_xyz_rpy[0] - ox) ** 2 + (t.pose_xyz_rpy[1] - oy) ** 2)
        assignments[o.name] = tgt.name
    task_type = scene.requirement_trace[0] if getattr(scene, "requirement_trace", None) else fallback_spec.task_type
    return PickPlaceTaskSpec(task_type, [o.name for o in manip], [t.name for t in targets], assignments, materials)


def _emit_mvp_artifacts(gene: RobotGene, output_dir: Path, *, task_type: str, package_type: str) -> None:
    """Write the shared package artifacts a gene build previously skipped — package contract, file-level
    validation, build summary, MVP readiness report, and MVP summary — so the Contract/Validation/Reports/
    Artifacts tabs populate and the package carries an HONEST readiness report. The MVP readiness gates score
    against the stated goal: the CV/world-model/training stages are optional and skipped (not failed) when this
    build did not run them. Best-effort: never crashes the build."""
    try:
        from virturoid.services.package_contract_builder import PACKAGE_CONTRACT_URI, write_robot_package_contract
        from virturoid.services.readiness_report import READINESS_REPORT_URI, write_mvp_readiness_report
        output_dir = Path(output_dir)
        artifacts = {
            "robot_genome": "robot/robot_genome.json",
            "scene_set": "simulation/scene_set.json",
            "compiled_scene_index": "simulation/mujoco/compiled_scene_index.json",
        }
        idx_path = output_dir / "simulation" / "mujoco" / "compiled_scene_index.json"
        if idx_path.exists():
            scenes = json.loads(idx_path.read_text(encoding="utf-8")).get("scenes", [])
            if scenes:
                artifacts["mujoco_xml"] = scenes[0].get("mujoco_xml", "")
        contract = write_robot_package_contract(
            output_dir, package_type=package_type, robot_class=gene.robot_class, species=gene.species,
            morphology_template_id=gene.id, task_type=task_type, artifacts=artifacts, training=None)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        checks = {k: (output_dir / v).exists() for k, v in artifacts.items()}
        (output_dir / "reports" / "package_validation_report.json").write_text(json.dumps(
            {"id": f"validation_{gene.id}", "ok": all(checks.values()) and contract.ok, "checks": checks,
             "note": "gene package file-level validation: core artifacts exist + parse"}, indent=2), encoding="utf-8")
        summary = {"selected_robot_class": gene.robot_class, "selected_species": gene.species,
                   "selected_morphology_template_id": gene.id, "package_type": package_type,
                   "package_valid": bool(contract.ok), "task_type": task_type,
                   "artifacts": {**artifacts, "robot_package_contract": PACKAGE_CONTRACT_URI,
                                 "mvp_readiness_report": READINESS_REPORT_URI}}
        (output_dir / "reports" / "autonomous_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_mvp_readiness_report(output_dir, summary=summary, artifacts=summary["artifacts"],
                                   training=None, requested_training=False)
        (output_dir / "reports" / "mvp_summary.md").write_text(
            f"# {gene.id}\n\n- **robot_class:** {gene.robot_class}\n- **species:** {gene.species}\n"
            f"- **task:** {task_type}\n- **package_type:** {package_type}\n\n"
            "Gene-composed package: an original generated morphology with task scenes, compiled MuJoCo, and real "
            "CAD. CV/perception and policy training are optional stages not run in this build.\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - artifact emission is best-effort; never crash the build
        pass


def _build_navigation_package(gene: RobotGene, prompt: str, output_dir: Path, controller_params: dict | None,
                              scene_count: int = 6) -> dict:
    """Build + evaluate a MOBILE (wheeled) gene on NAVIGATION.

    The scene the viewer renders is the task's REAL floor course — a navigation route or a MAZE, chosen from
    the prompt by the general scene generator (no hard-coded layout) and compiled FLOOR-aware (real ground +
    full-size walls, not a tabletop). The robot is scored on the navigation task its morphology implies
    (``task_matched_eval`` goal-reach), never on pick-place.
    """
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.scene_generator import generate_scene_set
    from virturoid.services.task_builder import build_task_graph
    from virturoid.services.task_matched_eval import evaluate_robot

    output_dir = Path(output_dir)
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    _write_genome_and_urdf(gene, output_dir)

    # Prompt-driven scene set (navigation course or maze) — the SAME general generator the gallery uses, so a
    # "navigate a maze" build renders an actual maze and "navigate to the goal" renders a course.
    task = build_task_graph(build_requirements_from_prompt(prompt))
    robot_radius = _robot_footprint_radius(gene)   # scale the course (corridors, obstacles) to THIS robot
    scenes = generate_scene_set(task, count=max(1, scene_count), purpose="variation", robot_radius=robot_radius).scenes
    scene_dir = output_dir / "simulation" / "mujoco" / "scenes" / "variation"
    scene_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in scenes:
        rel = f"simulation/mujoco/scenes/variation/{scene.id}.xml"
        (output_dir / rel).write_text(compile_gene_with_scene(gene, scene.objects), encoding="utf-8")
        entries.append({"scene_set_id": f"sceneset_{gene.id}", "scene_id": scene.id, "purpose": "variation",
                        "mujoco_xml": rel, "object_count": len(scene.objects)})
    (output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").write_text(json.dumps(
        {"id": f"compiled_{gene.id}", "robot_genome_id": gene.id, "backend": "mujoco",
         "scene_count": len(entries), "scenes": entries}, indent=2), encoding="utf-8")
    (output_dir / "simulation").mkdir(parents=True, exist_ok=True)
    (output_dir / "simulation" / "scene_set.json").write_text(json.dumps({
        "id": f"sceneset_{gene.id}", "version": "0.1.0", "task_graph_id": f"task_{gene.id}",
        "purpose": "variation", "task_type": task.task_type,
        "scenes": [_scene_to_dict(s) for s in scenes],
    }, indent=2), encoding="utf-8")

    # Score on the morphology-matched NAVIGATION task (goal-reach rate), not pick-place.
    try:
        res = evaluate_robot(gene, prompt=prompt, controller_params=controller_params)
        detail = res.get("detail", {})
        success_rate = float(res.get("value", 0.0))
        status = "reached_goals" if success_rate > 0 else "no_goal_reached"
    except Exception as exc:  # noqa: BLE001 - eval is best-effort; the compiled scene is the rendered artifact
        detail = {"error": str(exc)}
        success_rate = 0.0
        status = "eval_unavailable"
    is_maze = any(w in (prompt or "").lower() for w in ("maze", "labyrinth"))
    summary = {
        "task_type": "navigation",
        "scene_kind": "maze" if is_maze else "navigation_course",
        "species": gene.species,
        "robot_class": gene.robot_class,
        "success_rate": round(success_rate, 3),
        "status": status,
        "detail": detail,
    }
    # PERCEPTION: when the prompt asks for SENSED navigation of an UNKNOWN environment, also evaluate the
    # range-sensing reactive navigator -- the robot reaches goals using ONLY its rangefinder ring, no map.
    if any(w in (prompt or "").lower()
           for w in ("perception", "sensor", "unknown", "explore", "rangefinder", "sense", "lidar")):
        try:
            from virturoid.services.perception_nav import evaluate_perception_nav
            psum = evaluate_perception_nav(gene, n_scenes=4, seed=0)
            summary["perception"] = {"success_rate": psum["success_rate"], "reached": psum["reached"],
                                     "n_scenes": psum["n_scenes"], "sensed_only": True}
            summary["success_rate"] = round(max(summary["success_rate"], psum["success_rate"]), 3)
            (output_dir / "reports").mkdir(parents=True, exist_ok=True)
            (output_dir / "reports" / "perception_navigation_report.json").write_text(
                json.dumps(psum, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 - perception eval is best-effort; never breaks the build
            pass
    cad = _export_real_cad(gene, output_dir)   # REAL B-rep STEP/STL for the mobile base too
    summary["cad_real"] = bool(cad)
    summary["cad_part_count"] = (cad or {}).get("part_count", 0)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "gene_evaluation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["bom"] = _emit_bom(gene, output_dir, task=prompt)  # real per-joint actuators + sensors + materials
    _maybe_write_summaries(output_dir)                         # spec sheet + deployment guide (reports/)
    _maybe_write_spec_compliance(output_dir, gene, prompt)     # requested-vs-achieved spec report (§4.8E)
    _emit_mvp_artifacts(gene, output_dir, task_type=task.task_type, package_type="gene_navigation_package")
    summary["readiness"] = _emit_readiness(gene, output_dir)
    return summary


def _build_freemotion_package(gene: RobotGene, prompt: str, output_dir: Path, controller_params: dict | None) -> dict:
    """Build + evaluate an AERIAL (flight) or AQUATIC (swim) gene on the free-motion task its morphology implies
    — a quadcopter flies to waypoints, an undulator swims forward (``task_matched_eval.evaluate_robot`` ->
    _honest_fly / _honest_swim). No floor course + no pick-place: these bodies move through air/water, so the
    package must NOT fall through to the manipulation path (which scored a drone 0% at generic_manipulation)."""
    from virturoid.services.task_matched_eval import evaluate_robot
    output_dir = Path(output_dir)
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    _write_genome_and_urdf(gene, output_dir)
    # a single bare scene (the robot in free space) so the viewer has something to render
    rel = "simulation/mujoco/scenes/variation/free.xml"
    (output_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (output_dir / rel).write_text(compile_gene_with_scene(gene, []), encoding="utf-8")
    (output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").write_text(json.dumps(
        {"id": f"compiled_{gene.id}", "robot_genome_id": gene.id, "backend": "mujoco", "scene_count": 1,
         "scenes": [{"scene_set_id": f"sceneset_{gene.id}", "scene_id": "free", "purpose": "variation",
                     "mujoco_xml": rel, "object_count": 0}]}, indent=2), encoding="utf-8")
    try:
        res = evaluate_robot(gene, prompt=prompt, controller_params=controller_params)
        task_type = str(res.get("task") or "flight")
        success_rate = float(res.get("value", 0.0))
        detail = res.get("detail", {})
        status = "attained" if success_rate > 0 else "not_attained"
    except Exception as exc:  # noqa: BLE001 - eval is best-effort; the compiled scene is still the rendered artifact
        task_type, success_rate, detail, status = ("flight", 0.0, {"error": str(exc)}, "eval_unavailable")
    summary = {"task_type": task_type, "species": gene.species, "robot_class": gene.robot_class,
               "success_rate": round(success_rate, 3), "status": status, "detail": detail}
    cad = _export_real_cad(gene, output_dir)
    summary["cad_real"] = bool(cad)
    summary["cad_part_count"] = (cad or {}).get("part_count", 0)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "gene_evaluation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["bom"] = _emit_bom(gene, output_dir, task=prompt)
    _maybe_write_summaries(output_dir)
    _maybe_write_spec_compliance(output_dir, gene, prompt)
    _emit_mvp_artifacts(gene, output_dir, task_type=task_type, package_type="gene_freemotion_package")
    summary["readiness"] = _emit_readiness(gene, output_dir)
    return summary


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
    _write_genome_and_urdf(gene, output_dir)

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
    from virturoid.services.morph_policy import recipe_rollout_morph
    pol = banked_policy_for(gene, models_dir="models")
    # Score with the CPG-driven recipe rollout, EXACTLY like the viewer: the banked recipe policy if one exists,
    # else the BARE trot-CPG (pol=None). The old code fell back to a no-CPG scripted eval when no policy was banked,
    # so every un-banked composed walker (six-legged, a fresh quad) read 'fell'/0.0 here while the viewport walked it
    # on the bare CPG -- the headline contradicted the 3D replay. A genuinely unstable anatomy body still reads 'fell'.
    # "Is this a RECIPE policy?" comes from the artifact's EXPLICIT marker (``recipe_control``, meta[11]), falling
    # back VERBATIM to the legacy obs_mean inference when unmarked (pre-marker artifacts must behave as before).
    # This is not a controller choice -- both branches run recipe_rollout_morph -- it decides whether the BANKED
    # policy is used AT ALL, and pol=None there is NOT a zero-residual baseline: recipe_rollout_morph constructs a
    # RANDOM policy. So inferring from obs_mean discarded every GPU artifact (that trainer banks no normalizer) and
    # scored the species' headline -- and the qpos trace the viewer replays -- on a random stand-in instead.
    _pol_recipe = getattr(pol, "recipe_control", None) if pol is not None else None
    if _pol_recipe is None:
        _pol_recipe = getattr(pol, "obs_mean", None) is not None
    use_pol = pol if (pol is not None and _pol_recipe) else None
    rr = recipe_rollout_morph(gene, use_pol, steps=900, record_qpos=True)   # qpos -> classify's ROLL/PITCH gate
    # M1/#212 (2026-07-24 audit): the build must not report EXPORT-BLOCKED for a body that verify_robot / create_robot
    # certify as a CREDIBLE WALK. Those use the tuned CRAWL gait; the trot recipe above drifts a fresh composed quad
    # BACKWARD. So ALSO score the walkable crawl gait (the exact controller verify_robot uses, with the gene's cached
    # tuned params) and keep whichever is the better credible walk -- the build headline then agrees with the
    # product's own verdict instead of contradicting it. The winning rollout's frames drive the viewer replay below,
    # so headline == viewport still holds.
    try:
        from virturoid.services.gait_quality import classify as _classify
        from virturoid.services.morph_policy import crawl_gait_rollout
        rr_crawl = crawl_gait_rollout(gene, steps=900, record_qpos=True)
        _recipe_cred = _classify(rr).startswith("CREDIBLE")
        _crawl_cred = _classify(rr_crawl).startswith("CREDIBLE")
        _crawl_wins = (_crawl_cred and not _recipe_cred) or (
            _crawl_cred == _recipe_cred
            and abs(float(rr_crawl.get("forward", 0.0))) > abs(float(rr.get("forward", 0.0))))
        if _crawl_wins:
            rr = rr_crawl
    except Exception:  # noqa: BLE001 - the crawl cross-check is an honesty accelerant; a miss keeps the recipe score
        pass
    forward_m = float(rr.get("forward", 0.0))
    cadence = float(rr.get("cadence", 0.0)); upright_frac = float(rr.get("upright_frac", 0.0))
    n_feet = int(rr.get("n_feet", 0))
    # HONEST 'walked' gate (v7-A3, master_plan_v7 §3): the build HEADLINE now shares the product's ONE un-gameable
    # verdict -- classify() -- for legged bodies, so it cannot contradict verify_robot / the viewport. classify
    # requires survived + upright + cadence + genuine stepping SUPPORT + forward AND a LEVEL body (roll<35/pitch<20):
    # this is what closes the gap the old gate left OPEN -- a forward PITCH-DIVE lurch cleared forward+upright+cadence
    # and falsely certified 'walked'. A non-legged body (n_feet<2, e.g. a wheeled deck) keeps the raw forward gate.
    from virturoid.services.gait_quality import classify
    gait_verdict = classify(rr)
    walked = (gait_verdict.startswith("CREDIBLE") if n_feet >= 2
              else forward_m > _WALK_MIN_FORWARD_M and upright_frac >= 0.5)
    upright = upright_frac >= 0.5
    status = "walked" if walked else ("upright" if upright_frac >= 0.5 else "fell")
    distance_m = abs(forward_m)
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
        "gait_verdict": gait_verdict,               # v7-A3: the un-gameable classify() verdict behind `status`
    }
    # The physics/evaluation model stays primitive and deterministic; write a second, package-local visual model
    # for the replay so a user sees the generated CAD-like anatomy, not only its collision capsules.  Fail-open
    # preserves the primitive replay when build123d is unavailable.
    try:
        from virturoid.services.gene_compiler import write_packaged_visual_mjcf

        visual = write_packaged_visual_mjcf(
            gene, output_dir, include_floor=True, spawn_z=standing_spawn_z(gene), task="locomotion")
        if visual:
            summary["viewer_visual_model"] = visual
    except Exception:  # noqa: BLE001 - this is an optional presentation artifact
        pass
    cad = _export_real_cad(gene, output_dir)   # REAL B-rep STEP/STL for the walker too
    summary["cad_real"] = bool(cad)
    summary["cad_part_count"] = (cad or {}).get("part_count", 0)
    # M1/#212: persist the WINNING rollout's qpos trace so the viewer replays the SAME gait the headline scored
    # (when the walkable crawl beat the trot, the viewer must show that forward walk, not the trot's drift). The
    # viewer replays these qpos on the robot model and falls back to its live rollout if the shape doesn't match.
    try:
        _qf = rr.get("qpos_frames") or []
        if _qf and float(rr.get("forward", 0.0)) != 0.0:
            _stride = max(1, len(_qf) // 150)                  # ~150 frames is plenty for a smooth replay
            (output_dir / "simulation").mkdir(parents=True, exist_ok=True)
            (output_dir / "simulation" / "locomotion_qpos.json").write_text(json.dumps({
                "nq": len(_qf[0]), "frame_every": _stride,
                "qpos_frames": [[round(float(v), 6) for v in _qf[i]] for i in range(0, len(_qf), _stride)],
                "forward_m": round(float(rr.get("forward", 0.0)), 4),
            }), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the replay trace is a presentation aid; never break the build
        pass
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports" / "gene_evaluation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["bom"] = _emit_bom(gene, output_dir, task=prompt)  # real per-joint actuators + sensors + materials
    _maybe_write_summaries(output_dir)                         # spec sheet + deployment guide (reports/)
    _maybe_write_spec_compliance(output_dir, gene, prompt)     # requested-vs-achieved spec report (§4.8E)
    _emit_mvp_artifacts(gene, output_dir, task_type="locomotion", package_type="gene_locomotion_package")
    summary["readiness"] = _emit_readiness(gene, output_dir)
    return summary


def _maybe_write_spec_compliance(output_dir: Path, gene, prompt: str) -> None:
    """Write the build's HONEST reports: the BOM<->sim FIDELITY report (every build, §4.8A — surfaces the
    optimistic-sim mass gap) and, when the prompt carried quantitative constraints (height/weight/payload/
    pinned), the requested-vs-achieved SPEC-COMPLIANCE report (§4.8E). Best-effort; never breaks a build."""
    try:                                                       # §4.8A: BOM<->sim fidelity, written for every build
        from virturoid.services.fidelity_report import bom_sim_fidelity
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports" / "bom_sim_fidelity.json").write_text(
            json.dumps(bom_sim_fidelity(gene), indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the fidelity report is value-add; never block a build
        pass

    # Input Compiler Phase 0: write input/interpretation.json for EVERY prompt build (provenance for every
    # inferred vs stated field), so a downstream compliance report can point back to what the user actually asked.
    interpretation = None
    try:
        from virturoid.services.input_evidence import interpret_prompt, write_interpretation
        interpretation = interpret_prompt(prompt or "")
        write_interpretation(interpretation, str(output_dir))
    except Exception:  # noqa: BLE001 - the interpretation is value-add; never block a build
        interpretation = None

    try:
        from virturoid.services.spec_parser import parse_prompt_spec, spec_compliance_report
        spec = parse_prompt_spec(prompt or "")
        if not spec.any():
            return
        bom = None
        try:
            from virturoid.services.bom_builder import build_bom
            bom = build_bom(gene, task=prompt or "")
        except Exception:  # noqa: BLE001 - the report still covers height/weight without a BOM
            pass
        report = spec_compliance_report(gene, spec, bom=bom)
        if interpretation is not None:                          # link each constraint back to its input evidence
            try:
                from virturoid.services.input_evidence import link_compliance_to_evidence
                report = link_compliance_to_evidence(interpretation, report)
            except Exception:  # noqa: BLE001 - linking is a nicety; keep the raw report if it fails
                pass
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports" / "spec_compliance.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the compliance report is value-add; never block a build
        pass


def _maybe_write_summaries(output_dir: Path) -> None:
    """Best-effort post-build summaries: the capability + cost SPEC SHEET (reports/spec_sheet.*) and the
    'build it for real' DEPLOYMENT GUIDE (reports/deployment_guide.md), both aggregated from the genome + BOM +
    evaluation + the URDF/ROS2/control artifacts. Pure summaries -- never break the build."""
    try:
        from virturoid.services.spec_sheet import write_spec_sheet
        write_spec_sheet(output_dir)
    except Exception:  # noqa: BLE001 - the spec sheet is a summary nicety
        pass
    try:
        from virturoid.services.deployment_guide import write_deployment_guide
        write_deployment_guide(output_dir)  # reads the spec sheet, so it runs AFTER it
    except Exception:  # noqa: BLE001 - the deployment guide is a summary nicety
        pass


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
        # The sensor-fusion stack is DERIVED from these same BOM sensors: an EKF/AHRS/odometry config referencing
        # exactly the parts above, on the links they mount to. Colocated with the BOM so every build ships it.
        try:
            from virturoid.services.sensor_fusion_compiler import write_sensor_fusion
            write_sensor_fusion(gene, output_dir, task=task)   # -> output_dir/fusion/{config,launch}/...
        except Exception:  # noqa: BLE001 - fusion configs are value-add; never break a build
            pass
        # The operational control scripts (obs-assembler, safety filter clamped to these actuators' peak torque,
        # state machine, watchdog, teleop, calibration) -- each compile-checked + sim-dry-run before it ships.
        try:
            from virturoid.services.control_script_compiler import write_control_scripts
            write_control_scripts(gene, output_dir, task=task)  # -> output_dir/software/scripts/*.py
        except Exception:  # noqa: BLE001 - control scripts are value-add; never break a build
            pass
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
        # Unified Honesty Scorecard (§4.1) — readiness + spec-compliance + fidelity (+ sim2sim if present) in one
        # claim-vs-verdict view; the demo's honesty centerpiece (surfaces weak results next to their verdict).
        try:
            from virturoid.services.honesty_scorecard import scorecard_from_package
            (output_dir / "reports" / "honesty_scorecard.json").write_text(
                json.dumps(scorecard_from_package(output_dir), indent=2, default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001 - scorecard is value-add; never break a build
            pass
        return {"safe_to_export": ledger.safe_to_export, "highest_attained": ledger.highest_attained}
    except Exception:  # noqa: BLE001 - readiness emission must never break a build
        return {}


_GAIT_CONTROLLER_SOURCE = '''"""Standalone trot-gait controller exported by Virturoid.

Computes feed-forward joint POSITION TARGETS for a CPG trot gait. For joint j at time t (seconds):
    target[j] = default_pose[j] + amplitude[j] * sin(2*pi*frequency_hz*t + phase_offset[j])
clamped to the joint position limits. A downstream PD / ros2_control loop tracks these targets at the
low-level control frequency. Pure standard library; no MuJoCo or Virturoid imports, so it runs inside a
ROS2 node or a bare Python process.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


class GaitController:
    def __init__(self, params: dict):
        self.joint_names = params["joint_names"]
        self.default_pose = params["default_pose"]
        self.amplitude = params["amplitude"]
        self.phase_offset = params["phase_offset"]
        self.frequency_hz = float(params["frequency_hz"])
        self.limits = params["position_limits"]

    @classmethod
    def from_file(cls, path: str) -> "GaitController":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def infer(self, t_seconds: float) -> dict:
        """Return clamped joint position targets for the gait phase at time ``t_seconds``."""
        omega = 2.0 * math.pi * self.frequency_hz
        out = {}
        for name, q0, amp, ph, limit in zip(
            self.joint_names, self.default_pose, self.amplitude, self.phase_offset, self.limits
        ):
            value = q0 + amp * math.sin(omega * t_seconds + ph)
            out[name] = max(limit[0], min(limit[1], value))
        return out


if __name__ == "__main__":
    _here = Path(__file__).parent
    _params = _here / "policy_params.json"
    if not _params.exists():
        _params = _here / "control_program.json"
    controller = GaitController.from_file(str(_params))
    for _i in range(6):
        _t = _i * 0.1
        print(json.dumps({"t": round(_t, 2), "targets": controller.infer(_t)}))
'''


_CRAWL_CONTROLLER_SOURCE = '''"""Standalone frozen crawl-wave controller exported by Virturoid."""
from __future__ import annotations
import json
import math
from pathlib import Path


class GaitController:
    def __init__(self, params: dict):
        self.joint_names = params["joint_names"]
        self.default_pose = params["default_pose"]
        self.limits = params["position_limits"]
        self.frequency_hz = float(params["frequency_hz"])
        self.hip_amp = float(params["hip_amp"])
        self.knee_amp = float(params["knee_amp"])
        self.lift_duty = float(params["lift_duty"])
        self.legs = list(params["legs"])

    @classmethod
    def from_file(cls, path: str) -> "GaitController":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def infer(self, t_seconds: float) -> dict:
        target = list(self.default_pose)
        for leg in self.legs:
            u = (self.frequency_hz * float(t_seconds) + float(leg["phase"])) % 1.0
            if u < self.lift_duty:
                lift = 0.5 * (1.0 - math.cos(2.0 * math.pi * u / self.lift_duty))
                stride = -math.cos(math.pi * u / self.lift_duty)
            else:
                lift = 0.0
                stride = math.cos(math.pi * (u - self.lift_duty) / max(1e-3, 1.0 - self.lift_duty))
            knee = int(leg["knee_index"]); hip = int(leg["hip_index"])
            target[knee] = self.default_pose[knee] + self.knee_amp * lift
            target[hip] = self.default_pose[hip] + self.hip_amp * float(leg["hip_sign"]) * stride
        return {name: max(limit[0], min(limit[1], value))
                for name, value, limit in zip(self.joint_names, target, self.limits)}


if __name__ == "__main__":
    _here = Path(__file__).parent
    _params = _here / "policy_params.json"
    if not _params.exists():
        _params = _here / "control_program.json"
    controller = GaitController.from_file(str(_params))
    for _i in range(6):
        _t = _i * 0.1
        print(json.dumps({"t": round(_t, 2), "targets": controller.infer(_t)}))
'''


def _verify_exported_gait(gene) -> dict:
    """Run the EXPORTED bare feed-forward trot-CPG gait on the body (recipe rollout, ZERO learned residual = exactly
    the exported control law) and classify it — so the control program can state HONESTLY whether this gait walks
    the body, instead of unconditionally claiming it does. Same un-gameable principle as the flywheel bank: a
    slide/crouch/fall is not a walk. Best-effort: an unverifiable body reports credible=False (conservative)."""
    try:
        from virturoid.services.gait_quality import classify
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import (CPG_DEFAULT, MorphPolicy, compiled_model,
                                                     recipe_rollout_morph, robot_mjcf)
        graph = encode_robot(compiled_model(robot_mjcf(gene)))
        pol = MorphPolicy(graph.feature_dim, seed=0)
        pol.cpg = CPG_DEFAULT                                    # zero residual + CPG prior == the exported bare gait
        r = recipe_rollout_morph(gene, pol, steps=1200)
        verdict = classify(r)
        return {"credible": verdict.startswith("CREDIBLE"), "forward_m": abs(float(r.get("forward", 0.0))),
                "survived": bool(r.get("survived", False)), "verdict": verdict}
    except Exception:  # noqa: BLE001 - verification is best-effort; default to the conservative (non-credible) claim
        return {"credible": False, "forward_m": 0.0, "survived": False, "verdict": "unverified"}


def _write_gait_control_program(gene: RobotGene, genome: dict, output_dir: Path) -> None:
    """For LEGGED gene robots, export the deterministic trot gait as a standalone, runnable control program
    (software/control_program.json + software/gait_controller.py) -- the gene-path analogue of the legacy
    control_program.json, so a gene-built walker ships an actual deployable controller (a downstream PD loop tracks
    the joint targets), not just a robot description. No-op for non-legged bodies (extract_gait_params -> None).

    HONESTY: the program VERIFIES the exported gait in sim and reports ``verified_walk`` + the measured distance,
    instead of unconditionally claiming the gait walks -- so a customer building from the package is never told a
    body walks when the bare feed-forward gait only crouches/slides on it (many composed bodies walk via a learned
    residual or the crawl wave gait, NOT this bare trot-CPG)."""
    from virturoid.services.morph_policy import extract_crawl_gait_params, extract_gait_params

    try:
        gait = extract_crawl_gait_params(gene)
    except Exception:  # noqa: BLE001 - retain the explicit legacy starter if extraction cannot run
        gait = None
    controller_source = _CRAWL_CONTROLLER_SOURCE if gait else _GAIT_CONTROLLER_SOURCE
    if gait:
        v = {"credible": True, "forward_m": gait["sim_forward_m"], "verdict": gait["sim_verdict"]}
    else:
        gait = extract_gait_params(gene)
        if not gait:
            return
        v = _verify_exported_gait(gene)
    walks = bool(v["credible"])
    note = (f"Deterministic feed-forward trot gait from the Virturoid recipe CPG prior. VERIFIED: this bare gait "
            f"produced a CREDIBLE walk in simulation ({v['forward_m']:.2f} m forward, upright). A learned residual "
            f"policy (morph_policy npz) refines it further."
            if walks else
            f"Deterministic feed-forward trot gait from the Virturoid recipe CPG prior. NOTE: the bare feed-forward "
            f"gait did NOT achieve a credible walk on this body in simulation ({v['verdict']}, {v['forward_m']:.2f} m) "
            f"— it is a STARTING POINT. Train a closed-loop residual policy (Virturoid train) or tune the gait before "
            f"relying on it to walk.")
    if gait.get("policy_type") == "crawl_wave_gait":
        note = (f"Frozen crawl-wave gait VERIFIED in simulation ({v['forward_m']:.2f} m forward; {v['verdict']}). "
                "The exported controller uses this measured wave direction and structural joint map without a deployment-time probe.")
    program = {
        "id": f"gait_control_program_{genome.get('id', 'robot')}",
        "robot_genome_id": genome.get("id"),
        "entrypoint": "software/gait_controller.py",
        "control_law": ("frozen crawl-wave swing/stance targets over the exported structural leg map; downstream PD "
                        "tracks the targets" if gait.get("policy_type") == "crawl_wave_gait" else
                        "target[j] = default_pose[j] + amplitude[j]*sin(2*pi*frequency_hz*t + phase_offset[j]); "
                        "a downstream PD / ros2_control loop tracks these joint-position targets"),
        "control_frequency_hz": 20.0,
        "verified_walk": walks,                                 # did the EXPORTED gait credibly walk the body in sim?
        "sim_forward_m": round(float(v["forward_m"]), 3),
        "sim_verdict": v["verdict"],
        **gait,
        "notes": [note],
    }
    sw = output_dir / "software"
    sw.mkdir(parents=True, exist_ok=True)
    (sw / "control_program.json").write_text(json.dumps(program, indent=2), encoding="utf-8")
    (sw / "gait_controller.py").write_text(controller_source, encoding="utf-8")
    # Also drop the controller into software/controller/ so the ROS2 exporter embeds it and its node RUNS the
    # gait (policy_type "trot_cpg_gait") instead of publishing a neutral pose.
    bundle = sw / "controller"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "policy_params.json").write_text(json.dumps(program, indent=2), encoding="utf-8")
    (bundle / "controller.py").write_text(controller_source, encoding="utf-8")


def _write_genome_and_urdf(gene: RobotGene, output_dir: Path) -> None:
    """Write robot/robot_genome.json AND robot/robot.urdf for a gene-built package. The gene path shipped the genome
    + MJCF but no URDF, so gene-built robots (any creature) had no standard robot description for ROS/Gazebo and
    wouldn't render in the viewer's Robot mode. Best-effort URDF -- the MJCF replay never needs it."""
    (output_dir / "robot").mkdir(parents=True, exist_ok=True)
    genome = _gene_to_genome(gene)
    (output_dir / "robot" / "robot_genome.json").write_text(json.dumps(genome, indent=2), encoding="utf-8")
    try:
        # Correct-for-ANY-body URDF: transcribe the compiled MuJoCo model (legs spread to their real mounts), NOT
        # the arm-centric compute_arm_layout that stacked every leg vertically at the torso tip. Falls back to the
        # legacy genome->URDF only if the MuJoCo transcription is unavailable.
        from virturoid.services.gene_urdf import gene_to_urdf
        # mesh_dir = the package's robot/meshes/ so the URDF's <mesh filename="meshes/X.stl"> resolves next to
        # robot.urdf: the shipped body renders its real shape programs in Studio, not the collider primitive.
        (output_dir / "robot" / "robot.urdf").write_text(
            gene_to_urdf(gene, mesh_dir=str(output_dir / "robot" / "meshes")), encoding="utf-8")
    except Exception:  # noqa: BLE001 - URDF is a nicety; the MJCF replay works without it
        try:
            from virturoid.schemas.robot import RobotGenome
            from virturoid.services.urdf_exporter import write_robot_urdf
            write_robot_urdf(RobotGenome.from_dict(genome), output_dir)
        except Exception:  # noqa: BLE001
            pass
    try:
        # THESIS A corpus self-update: a body that just built with verified shape programs contributes one shape
        # WORD per role to the physics/geometry-gated corpus, so the next structurally-similar body recalls a proven
        # mantle / tentacle / leg instead of re-deriving it. Best-effort keep-best; never blocks the package write.
        from virturoid.services.shape_flywheel import auto_bank_body_shapes
        auto_bank_body_shapes(gene)
    except Exception:  # noqa: BLE001 - corpus growth is an accelerant, not a build dependency
        pass
    try:
        # Legged robots get a standalone, runnable trot control program (the gene-path control_program.json) AND a
        # software/controller/ bundle. Written BEFORE the ROS2 export so the ROS2 node embeds it and runs the gait.
        _write_gait_control_program(gene, genome, output_dir)
    except Exception:  # noqa: BLE001 - the gait control program is best-effort; non-legged bodies are skipped
        pass
    try:
        # Installable ROS2 (ament_python) harness from the genome + URDF + (for legged robots) the gait-controller
        # bundle, so gene-built robots are ROS2-deployable like the legacy path's. Best-effort.
        from virturoid.services.ros2_exporter import maybe_export_ros2_package
        maybe_export_ros2_package(output_dir)
    except Exception:  # noqa: BLE001 - the ROS2 package is best-effort; the core package works without it
        pass


def _gene_to_genome(gene: RobotGene) -> dict:
    return {
        "id": f"genome_{gene.id}",
        "name": gene.species,
        "species": gene.species,
        "robot_class": gene.robot_class,
        "links": [s.name for s in gene.segments],
        "joints": [
            {"name": f"{s.name}_joint", "joint_type": s.joint_type,
             "parent_link": s.parent, "child_link": s.name,
             # carry the REAL joint axis + limits so the exported URDF round-trips: without axis_xyz every joint
             # defaulted to +z, which flattened a quadruped's hip-abduction/knee axes and left a re-imported body
             # unable to walk. (effort = the sized actuator's peak torque; a real velocity ceiling for the servo.)
             "axis_xyz": list(s.joint_axis),
             "limit": {"lower": (s.joint_lower if s.joint_lower is not None else -3.14159),
                       "upper": (s.joint_upper if s.joint_upper is not None else 3.14159),
                       "effort": float(s.actuator_torque_nm or 20.0), "velocity": 10.0}}
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
