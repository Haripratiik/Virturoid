"""STAGE 1 — a library of REAL robot models (MuJoCo Menagerie via the ``robot_descriptions`` package).

These are production-grade descriptions with real meshes, real link inertias, and real actuators — the
opposite of the primitive stick-figures the procedural composer emits. "Build a quadruped" can now START
from a real Unitree Go1 instead of tubes, then be iterated/learned on through the existing pipeline.

Models download once (git-cloned to ``~/.cache/robot_descriptions``) and are cached thereafter. We keep
the MJCF *path* (not a flattened string) so MuJoCo resolves the model's mesh assets relative to it.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

# intent/class -> (robot_descriptions MJCF module, human label, kind). Broad coverage of MuJoCo Menagerie
# so most requests resolve to a REAL production robot; anything unmatched falls to the procedural path.
_REGISTRY: dict[str, tuple[str, str, str]] = {
    # quadrupeds
    "quadruped":   ("go1_mj_description", "Unitree Go1", "quadruped"),
    "go2":         ("go2_mj_description", "Unitree Go2", "quadruped"),
    "a1":          ("a1_mj_description", "Unitree A1", "quadruped"),
    "aliengo":     ("aliengo_mj_description", "Unitree Aliengo", "quadruped"),
    "anymal":      ("anymal_c_mj_description", "ANYbotics ANYmal C", "quadruped"),
    "spot":        ("spot_mj_description", "Boston Dynamics Spot", "quadruped"),
    "anymal_b":    ("anymal_b_mj_description", "ANYbotics ANYmal B", "quadruped"),
    "barkour":     ("barkour_vb_mj_description", "Google Barkour vB", "quadruped"),
    # humanoids / bipeds
    "humanoid":    ("h1_mj_description", "Unitree H1", "humanoid"),
    "h1_2":        ("h1_2_mj_description", "Unitree H1-2", "humanoid"),
    "g1":          ("g1_mj_description", "Unitree G1", "humanoid"),
    "apollo":      ("apollo_mj_description", "Apptronik Apollo", "humanoid"),
    "talos":       ("talos_mj_description", "PAL Talos", "humanoid"),
    "op3":         ("op3_mj_description", "Robotis OP3", "humanoid"),
    "booster":     ("booster_t1_mj_description", "Booster T1", "humanoid"),
    "toddlerbot":  ("toddlerbot_2xm_mj_description", "ToddlerBot", "humanoid"),
    "fourier":     ("n1_mj_description", "Fourier N1", "humanoid"),
    "adam":        ("adam_lite_mj_description", "PNDbotics Adam Lite", "humanoid"),
    "jvrc":        ("jvrc_mj_description", "JVRC-1", "humanoid"),
    "rby1":        ("rby1_mj_description", "Rainbow Robotics RB-Y1", "humanoid"),
    "biped":       ("cassie_mj_description", "Agility Cassie", "biped"),
    # arms
    "manipulator": ("panda_mj_description", "Franka Emika Panda", "arm"),
    "arm":         ("panda_mj_description", "Franka Emika Panda", "arm"),
    "fr3":         ("fr3_mj_description", "Franka FR3", "arm"),
    "ur5e":        ("ur5e_mj_description", "Universal Robots UR5e", "arm"),
    "ur10e":       ("ur10e_mj_description", "Universal Robots UR10e", "arm"),
    "iiwa":        ("iiwa14_mj_description", "KUKA LBR iiwa 14", "arm"),
    "kinova":      ("gen3_mj_description", "Kinova Gen3", "arm"),
    "xarm":        ("xarm7_mj_description", "UFACTORY xArm7", "arm"),
    "sawyer":      ("sawyer_mj_description", "Rethink Sawyer", "arm"),
    "rizon":       ("rizon4_mj_description", "Flexiv Rizon 4", "arm"),
    "piper":       ("piper_mj_description", "AgileX PiPER", "arm"),
    "z1":          ("z1_mj_description", "Unitree Z1", "arm"),
    "viperx":      ("viper_mj_description", "Trossen ViperX", "arm"),
    "widowx":      ("widow_mj_description", "Trossen WidowX", "arm"),
    "so_arm":      ("so_arm100_mj_description", "SO-ARM100 (low-cost)", "arm"),
    "yam":         ("yam_mj_description", "I2RT YAM", "arm"),
    "openarm":     ("openarm_v1_mj_description", "OpenArm v1", "arm"),
    # grippers / dexterous hands
    "gripper":     ("robotiq_2f85_mj_description", "Robotiq 2F-85", "gripper"),
    "hand":        ("shadow_hand_mj_description", "Shadow Hand", "hand"),
    "allegro":     ("allegro_hand_mj_description", "Wonik Allegro Hand", "hand"),
    "leap":        ("leap_hand_mj_description", "LEAP Hand", "hand"),
    "ability":     ("ability_hand_mj_description", "PSYONIC Ability Hand", "hand"),
    # mobile manipulator, aerial, bimanual
    "mobile_manipulator": ("stretch_mj_description", "Hello Robot Stretch", "mobile_manipulator"),
    "drone":       ("skydio_x2_mj_description", "Skydio X2", "drone"),
    "crazyflie":   ("cf2_mj_description", "Bitcraze Crazyflie 2", "drone"),
    "bimanual":    ("aloha_mj_description", "ALOHA bimanual", "bimanual"),
}

# keyword -> registry key, FIRST match wins, so order from most specific to most general.
_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("anymal b", "anymal-b"), "anymal_b"), (("anymal",), "anymal"),
    (("boston dynamics", "spot"), "spot"), (("barkour",), "barkour"), (("go2",), "go2"),
    (("aliengo",), "aliengo"), (("unitree a1", " a1 "), "a1"),
    (("cassie",), "biped"),
    (("h1-2", "h1_2", "h1 2"), "h1_2"), (("unitree g1", " g1 "), "g1"), (("apollo",), "apollo"),
    (("talos",), "talos"), (("op3",), "op3"), (("booster", " t1"), "booster"),
    (("toddler", "small humanoid", "child humanoid", "tiny humanoid"), "toddlerbot"),
    (("fourier", " n1"), "fourier"), (("adam",), "adam"), (("jvrc",), "jvrc"),
    (("rb-y1", "rby1", "rainbow robot"), "rby1"),
    (("humanoid", "biped", "two-leg", "two leg", "bipedal", "h1", "android", "human-like"), "humanoid"),
    (("quadruped", "four-leg", "four leg", "four-legged", "dog robot", "go1"), "quadruped"),
    (("crazyflie", "cf2", "nano drone", "tiny drone", "micro drone"), "crazyflie"),
    (("drone", "quadrotor", "quadcopter", "uav", "aerial", "flying robot"), "drone"),
    (("bimanual", "two-arm", "two arm", "dual-arm", "dual arm", "aloha"), "bimanual"),
    (("mobile manipulator", "mobile arm", "stretch"), "mobile_manipulator"),
    (("allegro",), "allegro"), (("leap hand", "leap_hand"), "leap"), (("ability hand",), "ability"),
    (("shadow", "dexterous", "five-finger", "five finger", "dexterity"), "hand"),
    (("gripper", "parallel jaw", "parallel-jaw", "2f85", "robotiq"), "gripper"),
    (("hand",), "hand"),
    (("fr3",), "fr3"), (("ur10",), "ur10e"), (("ur5", "ur-5"), "ur5e"), (("iiwa", "kuka"), "iiwa"),
    (("kinova", "gen3"), "kinova"), (("xarm",), "xarm"), (("sawyer",), "sawyer"),
    (("flexiv", "rizon"), "rizon"), (("piper", "agilex"), "piper"), (("unitree z1", "z1"), "z1"),
    (("viperx", "viper"), "viperx"), (("widowx", "widow"), "widowx"),
    (("so-arm", "so arm", "so100", "so101", "so-100", "so-101", "low-cost arm", "low cost arm"), "so_arm"),
    (("yam",), "yam"), (("openarm", "open arm"), "openarm"),
    (("arm", "manipulator", "panda", "franka", "pick", "place", "grasp", "pick-and-place"), "arm"),
]


def real_model_key(prompt: str = "", robot_class: str | None = None) -> str | None:
    """Pick the registry key for a prompt/class, or None if nothing real matches."""
    if robot_class and robot_class.lower() in _REGISTRY:
        return robot_class.lower()
    p = (prompt or "").lower()
    for words, key in _KEYWORDS:
        if any(w in p for w in words):
            return key
    return None


def available_real_models() -> list[dict]:
    return [{"key": k, "module": m, "label": lbl, "kind": kind}
            for k, (m, lbl, kind) in _REGISTRY.items()]


def _cached_mjcf_path(module: str) -> str | None:
    """Find a complete, already-downloaded Menagerie model without invoking Git.

    ``robot_descriptions`` imports call ``clone_to_cache`` and may fetch a pinned
    commit even when the requested model files are already present. That makes
    an offline/local reference load depend on network access and Git ownership
    configuration. Resolve exact model filenames in the package's documented
    cache first; importing remains the fallback that can populate a missing
    cache. The match is data-driven from the module name, not robot-class
    dispatch (``go1_mj_description`` -> ``go1.xml``).
    """
    suffix = "_mj_description"
    stem = module[:-len(suffix)] if module.endswith(suffix) else module
    cache = Path(os.path.expanduser(os.environ.get(
        "ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")))
    menagerie = cache / "mujoco_menagerie"
    if not menagerie.is_dir():
        return None
    candidates = sorted(
        (path for path in menagerie.glob(f"**/{stem}.xml") if path.name.lower() != "scene.xml"),
        key=lambda path: (len(path.parts), len(str(path)), str(path)),
    )
    return str(candidates[0]) if candidates else None


def load_real_model(prompt: str = "", robot_class: str | None = None) -> dict:
    """Resolve a prompt/class to a REAL model and load it. Returns ``{ok, path, label, kind, key, bodies,
    actuated, meshes, note}`` — ``path`` is the MJCF file (load via ``from_xml_path`` so meshes resolve)."""
    key = real_model_key(prompt, robot_class)
    if key is None:
        return {"ok": False, "note": "no real model matches this request"}
    module, label, kind = _REGISTRY[key]
    try:
        import mujoco

        path = _cached_mjcf_path(module)
        source = "local_cache"
        if path is None:
            mod = importlib.import_module(f"robot_descriptions.{module}")
            path = mod.MJCF_PATH
            source = "robot_descriptions"
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as exc:  # noqa: BLE001 - missing package/model -> caller falls back to procedural
        return {"ok": False, "note": f"could not load real model {module!r}: {exc}", "key": key}
    free = any(int(m.jnt_type[j]) == 0 for j in range(m.njnt))   # free joint -> can locomote
    return {"ok": True, "path": path, "label": label, "kind": kind, "key": key,
            "bodies": int(m.nbody) - 1, "actuated": int(m.nu), "meshes": int(m.nmesh),
            "free_base": bool(free), "source": source, "note": f"real model: {label} ({source})"}


def real_model_mjcf(prompt: str = "", robot_class: str | None = None) -> dict:
    """Load a real model as a PORTABLE MJCF string: rewrite ``meshdir`` to an absolute path (so the meshes
    resolve even when the string is moved around) and ensure a ground plane. This lets a real Menagerie
    robot flow through the SAME string-based pipeline (viewport / import / learners) as a composed gene."""
    import re
    from pathlib import Path

    info = load_real_model(prompt, robot_class)
    if not info.get("ok"):
        return info
    p = Path(info["path"])
    xml = p.read_text(encoding="utf-8")
    md = re.search(r'meshdir\s*=\s*"([^"]*)"', xml)
    absdir = str((p.parent / (md.group(1) if md else ".")).resolve()).replace("\\", "/")
    if md:
        xml = xml.replace(md.group(0), f'meshdir="{absdir}"')
    elif "<compiler" in xml:
        xml = xml.replace("<compiler ", f'<compiler meshdir="{absdir}" ', 1)
    else:
        xml = xml.replace("<mujoco model", f'<mujoco model', 1).replace(">", f'>\n  <compiler meshdir="{absdir}"/>', 1)
    if 'type="plane"' not in xml:
        floor = '    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.30 0.34 0.40 1"/>\n'
        xml = xml.replace("<worldbody>", "<worldbody>\n" + floor, 1)
    info["mjcf"] = xml
    return info


# --- Display normalization (the "renders dark" fix) ----------------------------------------------
# Real Menagerie models often ship near-black materials (Unitree H1 ~[0.1,0.1,0.1], Go1 ~[0.2,..]) and
# many geoms carry no material at all, so a straight render is an unreadable black silhouette. This lifts
# near-black surfaces to a visible neutral grey and sets bright neutral lighting — DISPLAY ONLY: mass,
# inertia, collision geometry and joints are untouched, so physics/training are unaffected. Apply it to any
# ``MjModel`` built for VIEWING (viewport, render/inspect tools), never to the physics/eval model.
_DISPLAY_FLOOR = 0.18      # any albedo channel below this is "display-black" and gets lifted
_DISPLAY_MIN = 0.45        # lifted channels land at this neutral grey


def _lift_rgb(rgb):
    """Lift a near-black colour to a visible neutral grey, preserving relative hue."""
    import numpy as np
    rgb = np.asarray(rgb, dtype=float)[:3].copy()
    peak = float(rgb.max())
    if peak < 1e-6:
        return np.array([_DISPLAY_MIN, _DISPLAY_MIN, _DISPLAY_MIN])
    if peak < _DISPLAY_FLOOR:
        return np.clip(rgb * (_DISPLAY_MIN / peak), 0.0, 1.0)
    return rgb


def normalize_display(model) -> dict:
    """In-place display normalization of a loaded ``mujoco.MjModel`` (geometry/dynamics untouched).

    Lifts near-black material + per-geom colours to a legible neutral grey, tames specular highlights,
    and sets a bright neutral headlight, so EVERY model (real or generated) renders clearly instead of as
    a black silhouette. Returns a small report ``{materials_lifted, geoms_lifted}``. Safe no-op if MuJoCo
    isn't importable. Use for VIEWING only — never on the model the task loop simulates/scores."""
    try:
        import mujoco
        import numpy as np
    except Exception:  # noqa: BLE001 - no MuJoCo -> nothing to normalize
        return {"materials_lifted": 0, "geoms_lifted": 0}
    changed_mats = changed_geoms = 0
    for i in range(model.nmat):
        old = model.mat_rgba[i][:3].copy()
        new = _lift_rgb(old)
        if not np.allclose(old, new):
            model.mat_rgba[i][:3] = new
            changed_mats += 1
        model.mat_specular[i] = min(float(model.mat_specular[i]), 0.3)
        model.mat_shininess[i] = min(float(model.mat_shininess[i]), 0.3)
        model.mat_reflectance[i] = min(float(model.mat_reflectance[i]), 0.1)
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) or int(model.geom_matid[g]) >= 0:
            continue
        old = model.geom_rgba[g][:3].copy()
        new = _lift_rgb(old)
        if not np.allclose(old, new):
            model.geom_rgba[g][:3] = new
            changed_geoms += 1
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
    model.vis.headlight.specular[:] = [0.10, 0.10, 0.10]
    return {"materials_lifted": changed_mats, "geoms_lifted": changed_geoms}
