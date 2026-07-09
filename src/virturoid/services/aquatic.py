"""Make aquatic robots actually SWIM — the body/controller co-design that closes the swim frontier on CPU.

The fluid sim was real but a "fish robot" composed a round-bodied quadruped (a dog), which barely moves in water.
An undulatory swimmer needs TWO things, both measured this session:
  * a SERIAL SPINE (not legs) that undulates laterally, and
  * a LATERALLY-COMPRESSED cross-section (thin sideways, tall vertically) so the undulation pushes water — drag
    anisotropy. MEASURED: a round undulator swims ~0.06 m; the flattened one ~0.25 m; a clean flat 8-link body
    ~0.75 m (near the idealised ~0.85 m). Round vs flat is the whole difference between DOES NOT SWIM and SWIMS.

So aquatic prompts are composed as a spine and reshaped here; ``_honest_swim`` drives it with the tuned wave.
"""
from __future__ import annotations

import re

_AQUATIC_WORDS = ("fish", "eel", "shark", "whale", "dolphin", "aquatic", "underwater", "submarine", "swim",
                  "swimming", "manta", "stingray", "salmon", "tuna", "serpent", "marine", "koi", "lamprey")
_AQUATIC_RE = re.compile(r"\b(" + "|".join(_AQUATIC_WORDS) + r")\b", re.I)

# Tuned travelling-wave swim controller (measured best on a flat undulator): larger amplitude, a gentle head->tail
# phase gradient, a faster beat, and a stiff PD so the body tracks the wave. Shared with ai_native_tools._honest_swim.
SWIM_WAVE = {"amp": 1.1, "wavenum": 0.8, "freq": 1.8, "kp": 12.0, "kd": 0.4}


def is_aquatic_prompt(prompt: str) -> bool:
    return bool(_AQUATIC_RE.search(prompt or ""))


def _is_serial_spine(gene) -> bool:
    """True if the actuated body is a single non-branching chain (a spine), not a legged/branched body."""
    kids: dict = {}
    for s in gene.segments:
        if (s.joint_type or "") == "revolute" and s.parent:
            kids[s.parent] = kids.get(s.parent, 0) + 1
    return bool(kids) and max(kids.values()) <= 1


def ensure_aquatic_body(gene):
    """Laterally-compress a spine body into a fish/eel cross-section so undulation generates real thrust. Only
    reshapes a serial spine (>=3 actuated links); a non-spine body is left unchanged (honest — it can't swim)."""
    chain = [s for s in gene.segments if (s.joint_type or "") == "revolute"]
    if len(chain) < 3 or not _is_serial_spine(gene):
        return gene
    for s in gene.segments:
        r = max(0.02, float(s.radius_m))
        s.shape = "box"
        s.cross_section = (round(0.45 * r, 4), round(2.3 * r, 4))   # (thin sideways, tall) -> anisotropic drag
        s.geometry = None                                          # the box cross-section IS the shape now
    gene.robot_class = "aquatic"
    gene.species = "aquatic.undulator"
    md = dict(getattr(gene, "metadata", None) or {}); md["aquatic"] = True
    gene.metadata = md
    return gene
