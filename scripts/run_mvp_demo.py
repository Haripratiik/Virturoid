"""One-command MVP demo gallery (plan #4): build a diverse battery of robots from plain text, SEE them, VERIFY
them with un-gameable verdicts, and (for legged bodies) show the learned-gait FLYWHEEL make them better --
emitted as ONE self-contained HTML page you open in a browser. No server, no build step: renders are embedded as
base64, verdicts are the real ones (a robot that falls says so), and the flywheel bars are measured, not staged.

    python scripts/run_mvp_demo.py                 # full battery -> build/demo/index.html  (~12 min)
    python scripts/run_mvp_demo.py --mini          # 2-robot quick pass (what the smoke test runs; 4-10 min)
    python scripts/run_mvp_demo.py --llm            # use the configured LLM design path instead of offline
    python scripts/run_mvp_demo.py --no-learn       # skip the flywheel pass (faster)
    python scripts/run_mvp_demo.py --no-compare     # skip the same-family differentiation panel (faster)
    python scripts/run_mvp_demo.py --out build/demo # choose the output dir

Honesty (see memory 'verify-renders-not-just-numbers'): the gallery shows the render AND the measured verdict for
every robot, including weak/flagged ones -- never a curated win. The flywheel delta is verify-before vs verify-after
on the SAME held body at the SAME horizon.

AND THAT INCLUDES BEING HONEST ABOUT WHAT IT COSTS YOU TO WATCH. This was documented as "~1 minute" while three
cold --mini runs measured 212 s, 336 s and 579 s -- and printed NOTHING for 202/321/574 of those seconds, because
one call (create_robot on the quadruped) owns almost the whole run: it grounds the body and then searches for an
operating point that fits THAT body. The search is the honest version of the step and is not trimmed here. So the
runner announces every stage, reports what it cost, and heart-beats every 15 s while a stage is still working --
see ``_Stage``. A first-run reader should never have to guess whether this hung.

...AND that includes being honest about WHICH BRAIN DESIGNED THE BODY. This page used to be titled "text -> robot"
while running with the project .env suppressed, which means get_llm() returned None and no language model ever saw
the prompt: the bodies came from the offline compositor. The renders, the physics and the verdicts were real; the
implied claim was not. So the run now RESOLVES the design path up front (``_design_path``), says so in a banner at
the top of the page rather than in a footnote, and stamps every card with the ``design_source`` the composer
actually stamped on that gene. Offline stays the DEFAULT -- it needs no API key, costs nothing and is what a
reviewer can reproduce on a fresh clone -- but it is now labelled instead of implied, and ``--llm`` runs the real
thing when a backend is configured.
"""
from __future__ import annotations

import argparse
import base64
import html
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Wall clock for the whole run, so every line carries the elapsed time of the RUN and not just of its stage.
_T0 = time.time()

#: How often a still-running stage says so. 15 s is short enough that a reader never wonders whether the process
#: died, long enough that the log stays readable.
_HEARTBEAT_S = 15.0

# A deliberately DIVERSE battery -- different morphologies exercise different verdict paths (legged gait, manipulator
# grasp, mobile drive) so the gallery proves breadth, not one lucky body. Order = what a reviewer should see first.
FULL_BATTERY = [
    {"prompt": "a quadruped robot dog", "note": "legged locomotion — walks by default, flywheel improves it"},
    {"prompt": "a six-legged hexapod robot", "note": "many-leg gait from structure (no per-species code)"},
    {"prompt": "a robot arm on a table that picks up a block", "note": "manipulation — real friction grasp + lift"},
    {"prompt": "a four-wheeled rover", "note": "wheeled mobile base — drives on its wheels"},
    {"prompt": "a humanoid robot that walks", "note": "hard case — reported honestly, walk not guaranteed"},
]
MINI_BATTERY = [
    {"prompt": "a quadruped robot dog", "note": "legged locomotion"},
    {"prompt": "a robot arm on a table that picks up a block", "note": "manipulation grasp"},
]

# SAME FAMILY, DIFFERENT BODIES. The battery above proves BREADTH (a leg, an arm, a wheel); it cannot prove
# DIFFERENTIATION, because every card is a different class and of course they look different. The question a
# reviewer actually asks about a text-to-robot product is the narrow one: give it three prompts from ONE family
# and do you get three robots, or one robot three times? Until recently the honest answer was "one robot three
# times" -- create_robot ran the walkability gate on an ungrounded, un-tuned body, every authored quadruped
# measured ~0 m at another robot's hand-tuned operating point, and all of them were replaced by the same fanned
# template (byte-identical: 13.500 kg, torso:leg 0.498). create_robot now grounds the body, fits an operating
# point TO it, and only then asks whether it can walk -- so these three stay themselves. That is the property
# this panel exists to make visible and measurable, and it is measured, not asserted: if a body IS replaced the
# table says SUBSTITUTED and the claim retracts itself.
FAMILY_COMPARE = [
    {"prompt": "a quadruped robot dog", "label": "dog"},
    {"prompt": "a robot cat", "label": "cat"},
    {"prompt": "a robot horse", "label": "horse"},
]

# What the composer stamps on a gene -> what a reader should understand by it. The point of showing this per card
# is that it is the COMPOSER's own record of which path built that body, not the runner's guess about it.
_SOURCE_LABEL = {
    "anatomy": "LLM anatomy graph",
    "llm_anatomy_graph": "LLM anatomy graph",
    "anatomy_graph": "LLM anatomy graph",
    "intent": "LLM design intent",
    "anatomy_generic": "offline anatomy compiler",
    "archetype": "offline archetype builder",
    "anthropometric": "offline anthropometric builder",
    "aerial_platform": "offline aerial builder",
    "heuristic": "offline keyword recipe",
}


def _ascii(s: object) -> str:
    """Console-safe text. Windows consoles default to cp1252, which cannot encode an em-dash -- the first line a
    new user read was 'Open it in a browser ? it's self-contained'. Verdict strings are interpolated into console
    output too, so sanitise at the print site rather than hand-checking every literal. The HTML keeps its real
    typography: that file is written encoding='utf-8' and read by a browser, not a console."""
    return (str(s).replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"')
            .encode("ascii", "replace").decode("ascii"))


def _clock(t: float | None = None) -> str:
    """``[m:ss]`` since the process started -- a moving clock on every line, so a reader can tell at a glance
    whether the run is progressing or wedged."""
    s = int((t if t is not None else time.time()) - _T0)
    return f"[{s // 60:d}:{s % 60:02d}]"


def _say(msg: str, indent: int = 0) -> None:
    """One console line, ASCII-safe, wall-clock stamped, flushed. Never buffered -- a progress line that arrives
    at the end of the run is not a progress line."""
    print(f"{_clock()} {' ' * indent}{_ascii(msg)}", flush=True)


class _Stage:
    """Announce a stage, HEARTBEAT while it runs, and report what it cost.

    THIS IS THE FIX FOR THE FIRST-RUN EXPERIENCE, and it is worth saying why it is telemetry rather than a
    speed-up. Measured on this checkout, ``--mini`` spends 321 of its 336 seconds inside ONE call --
    ``create_robot`` on the quadruped -- and prints nothing for all 321 of them. A reader who has been told
    "~1 minute" checks five times whether it hung, and closes the tab at minute two of their first command.

    The 321 s is not waste to be trimmed: ``create_robot`` grounds the body and then fits an operating point TO
    that body by bounded search against the deploy-horizon verdict (see ai_native_tools.create_robot's own cost
    note -- 124.7 s / 360 evaluations for the authored dog). Making that cheaper means shipping a body tuned at
    another robot's operating point, which is exactly the dishonesty the search exists to remove. So the run
    SAYS what it is doing, every 15 seconds, with the clock running.

    The heartbeat is a daemon thread that only prints; it touches no shared state, so it cannot affect a verdict.
    """

    def __init__(self, label: str, why: str = "", indent: int = 6) -> None:
        self.label, self.why, self.indent = label, why, indent
        self._stop = threading.Event()
        self._t0 = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "_Stage":
        self._t0 = time.time()
        _say(f"{self.label} ...{(' ' + self.why) if self.why else ''}", self.indent)
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(_HEARTBEAT_S):
            _say(f"... still running: {self.label} ({int(time.time() - self._t0)}s)", self.indent + 2)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self.seconds = round(time.time() - self._t0, 1)
        _say(f"{'x' if exc[0] else 'done'} {self.label} in {self.seconds}s", self.indent)


def _design_path() -> dict:
    """RESOLVE (do not assume) which brain will design the bodies in this run.

    Calls the same ``get_llm("morphology")`` the composer calls, so the banner reports what actually happens
    rather than what the flags requested: a configured backend that is unreachable at build time still falls back
    per body, and the per-card ``design_source`` records that. Never returns the backend URL."""
    try:
        from virturoid.services.llm_client import get_llm, unwrap_llm
        llm = get_llm("morphology")
    except Exception as exc:  # noqa: BLE001 - a probe must never be able to fail a build
        return {"llm": False, "backend": None, "error": f"{type(exc).__name__}: {exc}"}
    if llm is None:
        return {"llm": False, "backend": None}
    raw = unwrap_llm(llm)
    return {"llm": True, "backend": getattr(raw, "name", "?"), "model": getattr(raw, "model", None)}


def _b64(path: str | None) -> tuple[str, str] | None:
    """(mime, base64) for an image path, or None. Small enough to inline; renders are a few hundred KB."""
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = "image/gif" if ext == ".gif" else "image/png"
    try:
        return mime, base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _verdict_class(verdict: str, credible: bool | None) -> str:
    """Honest badge colour from the real verdict text."""
    v = (verdict or "").upper()
    if credible or any(k in v for k in ("CREDIBLE", "PICKS UP", "DRIVES", "ARTICULATES OK")):
        return "good"
    if any(k in v for k in ("NO ACTUATORS", "COULD NOT", "FLAGGED", "UNSUPPORTED", "FALLS", "CANNOT")):
        return "bad"
    return "warn"


def _body_metrics(robot_id: str) -> dict:
    """MEASURE the held body: grounded mass, per-limb leg + torso length, segment/DOF count, the operating point
    it was fitted with, and whether the walkability gate replaced it with the fanned template. Read off the gene
    itself, so the comparison table can never disagree with the robot that was rendered and verified."""
    import re

    out: dict = {}
    try:
        from virturoid.services import session_state as S
        gene = S.get_robot(robot_id)
        if gene is None:
            return out
        legs = [float(getattr(s, "length_m", 0.0) or 0.0) for s in gene.segments
                if re.search(r"leg", (s.name or "").lower())]
        torso = [float(getattr(s, "length_m", 0.0) or 0.0) for s in gene.segments
                 if re.search(r"torso|trunk|spine|chassis", (s.name or "").lower())]
        # per-LIMB length (not the summed skeleton) is the number that means something next to a torso length
        n_legs = len({m.group(0) for s in gene.segments
                      if (m := re.search(r"(?:(?:front|hind)_)?leg\d*_[lr]", (s.name or "").lower()))}) or 4
        leg_len = (sum(legs) / n_legs) if legs else 0.0
        torso_len = sum(torso)
        md = getattr(gene, "metadata", None) or {}
        fit = md.get("gait_fit") or {}
        out = {
            "mass_kg": round(sum(float(getattr(s, "mass_kg", 0.0) or 0.0) for s in gene.segments), 3),
            "leg_m": round(leg_len, 3), "torso_m": round(torso_len, 3),
            "torso_over_leg": round(torso_len / leg_len, 3) if leg_len else None,
            "segments": len(gene.segments), "dof": len(gene.actuated_joints()),
            "substituted": bool((md.get("walkability_fallback") or {}).get("applied")),
            # the per-body operating point create_robot fitted (or "shipped default" when the default already walked)
            "gait": (md.get("gait_params") or None),
            "gait_note": (fit.get("reason") if fit else None),
            # THE ERROR BAR THAT BELONGS NEXT TO THE NUMBER. Two bodies can print the same verdict and a similar
            # distance while one survives a 10% perturbation of every gait parameter and the other survives none
            # -- four orders of magnitude apart, historically shown as the same two words (task #267).
            "robustness_rel": (fit.get("robustness_rel") if fit else None),
            "fragile": bool(fit.get("fragile")) if fit else False,
        }
    except Exception:  # noqa: BLE001 - metrics are descriptive; never fail a build over them
        pass
    return out


def build_one(spec: dict, *, quick: bool, learn: bool, reuse: dict[str, dict] | None = None) -> dict:
    """create -> render -> verify (+ optional flywheel before/after for legged). Returns a card dict.

    ``reuse`` maps prompt -> a body the comparison panel already built, so a prompt shared between the two
    sections is composed ONCE and both sections describe the same robot at different points in its life."""
    from virturoid.services.agent_tools import call_tool

    prompt = spec["prompt"]
    t0 = time.time()
    card: dict = {"prompt": prompt, "note": spec.get("note", ""), "error": None,
                  "img": None, "gif": None, "flywheel": None}
    try:
        held = (reuse or {}).get(prompt)
        if held:
            cr = held
            _say(f"create_robot: reusing the body built for this prompt in the comparison panel", 6)
        else:
            with _Stage("create_robot", "(compose -> ground the mass -> fit an operating point to THIS body; "
                                       "the gait fit is the slow part, minutes on CPU for a legged body)"):
                cr = call_tool("create_robot", {"prompt": prompt})["result"]
        rid = cr["robot_id"]
        card["robot_id"] = rid
        # the COMPOSER's own record of which path built this gene -- so the page reports the design path per
        # robot instead of inferring it once from the flags (a configured LLM that 429s falls back per body)
        card["design_source"] = cr.get("design_source", "unknown")

        with _Stage("render_view", "(MuJoCo render of the body)"):
            rv = call_tool("render_view", {"robot_id": rid}).get("result", {})
        arts = rv.get("artifacts") or []
        card["img"] = _b64(arts[0]) if arts else None

        mode = "quick" if quick else "full"
        with _Stage(f"verify_robot ({mode})", "(real rollout -> the verdict on the card)"):
            vr = call_tool("verify_robot", {"robot_id": rid, "mode": mode}).get("result", {})
        card.update({"kind": vr.get("kind", "?"), "verdict": vr.get("verdict", "no verdict"),
                     "credible": vr.get("credible_walk"), "survived": vr.get("survived")})
        card["metrics"] = {k: vr.get(k) for k in ("forward_m", "reach_m", "success_rate", "swim_m", "cadence_hz")
                           if vr.get(k) is not None}
        # a GIF (legged) is the most convincing artifact -> prefer it over the still
        vgif = _b64((vr.get("artifacts") or [None])[0]) if vr.get("artifacts") else None
        if vgif:
            card["gif"] = vgif

        # FLYWHEEL: for a legged body, learn a gait and re-verify on the SAME body/horizon -> measured before/after
        if learn and vr.get("kind") == "legged":
            before = vr.get("forward_m")
            with _Stage("learn_gait", "(6 generations x 16 candidates -- the flywheel pass)"):
                lg = call_tool("learn_gait", {"robot_id": rid, "generations": 6, "pop": 16,
                                              "steps": 900}).get("result", {})
            with _Stage("verify_robot (full, after learning)", "(same body, same horizon -> the before/after bar)"):
                vr2 = call_tool("verify_robot", {"robot_id": rid, "mode": "full"}).get("result", {})
            after = vr2.get("forward_m")
            if before is not None and after is not None:
                card["flywheel"] = {"before": round(float(before), 3), "after": round(float(after), 3),
                                    "verdict": vr2.get("verdict", ""), "source": lg.get("gait_source", "learned")}
                if _b64((vr2.get("artifacts") or [None])[0]) if vr2.get("artifacts") else None:
                    card["gif"] = _b64(vr2["artifacts"][0])
                card["verdict"] = vr2.get("verdict", card["verdict"])
                card["credible"] = vr2.get("credible_walk", card["credible"])
    except Exception as exc:  # noqa: BLE001 - a broken robot is shown as an honest error card, never hidden
        card["error"] = f"{type(exc).__name__}: {exc}"
    card["seconds"] = round(time.time() - t0, 1)
    return card


def build_family(specs: list[dict]) -> list[dict]:
    """Build the same-family comparison row: create -> render -> verify each body at ONE shared horizon.

    APPLES TO APPLES IS THE WHOLE POINT. The gallery cards run a mix of horizons -- a legged card re-verifies after
    a flywheel training pass, which rewrites the body's operating point -- so lifting their numbers into one table
    would put a TRAINED dog next to two untrained animals and call the difference morphology. Every body here gets
    the same 800-step quick rollout and no flywheel pass, because the property on show is the operating point
    ``create_robot`` fits to each body at BUILD time, before any training.

    Runs BEFORE the battery so the dog it creates can be handed to ``build_one`` by robot_id (``run_gallery``
    passes it as ``reuse``): the dog in this table is then literally the dog in the card, measured earlier in its
    life rather than a second dog that happened to come out similar -- and it is built once, not twice."""
    from virturoid.services.agent_tools import call_tool

    rows: list[dict] = []
    for i, spec in enumerate(specs, 1):
        prompt = spec["prompt"]
        t0 = time.time()
        row: dict = {"prompt": prompt, "label": spec.get("label", prompt), "error": None, "img": None}
        _say(f"[compare {i}/{len(specs)}] {prompt}", 3)
        try:
            with _Stage("create_robot", "(compose -> ground -> fit an operating point to THIS body)"):
                cr = call_tool("create_robot", {"prompt": prompt})["result"]
            rid = cr["robot_id"]
            row["design_source"] = cr.get("design_source", "unknown")
            row["robot_id"] = rid
            with _Stage("render_view"):
                rv = call_tool("render_view", {"robot_id": rid}).get("result", {})
            arts = rv.get("artifacts") or []
            row["img"] = _b64(arts[0]) if arts else None
            with _Stage("verify_robot (quick)", "(800-step rollout, identical for all three bodies)"):
                vr = call_tool("verify_robot", {"robot_id": rid, "mode": "quick"}).get("result", {})
            row["verdict"] = vr.get("verdict", "no verdict")
            row["credible"] = vr.get("credible_walk")
            row["forward_m"] = vr.get("forward_m")
            row.update(_body_metrics(rid))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        _say(f"-> {row['label']}: {row.get('verdict') or row.get('error')} "
             f"({row.get('mass_kg')} kg, {row.get('seconds')}s)", 3)
    return rows


def _card_html(c: dict) -> str:
    if c.get("error"):
        return (f'<article class="card err"><h3>{html.escape(c["prompt"])}</h3>'
                f'<p class="note">{html.escape(c.get("note",""))}</p>'
                f'<p class="badge bad">BUILD ERROR</p><pre>{html.escape(c["error"])}</pre></article>')
    media = ""
    img = c.get("gif") or c.get("img")
    if img:
        media = f'<img src="data:{img[0]};base64,{img[1]}" alt="{html.escape(c["prompt"])}"/>'
    else:
        media = '<div class="noimg">no render</div>'
    badge_cls = _verdict_class(c.get("verdict", ""), c.get("credible"))
    metrics = " · ".join(f'<b>{html.escape(k)}</b> {v}' for k, v in (c.get("metrics") or {}).items())
    fly = ""
    if c.get("flywheel"):
        f = c["flywheel"]
        b, a = f["before"], f["after"]
        hi = max(b, a, 0.01)
        fly = (f'<div class="fly"><div class="flylab">flywheel — learned gait</div>'
               f'<div class="bar"><span class="b1" style="width:{int(100*b/hi)}%"></span>'
               f'<em>before {b} m</em></div>'
               f'<div class="bar"><span class="b2" style="width:{int(100*a/hi)}%"></span>'
               f'<em>after {a} m</em></div></div>')
    return (f'<article class="card"><div class="media">{media}</div>'
            f'<h3>{html.escape(c["prompt"])}</h3>'
            f'<p class="note">{html.escape(c.get("note",""))}</p>'
            f'<p class="badge {badge_cls}">{html.escape(str(c.get("verdict","")))}</p>'
            f'{f"<p class=metrics>{metrics}</p>" if metrics else ""}{fly}'
            f'<p class="foot">{html.escape(c.get("kind","?"))} · {c.get("seconds","?")}s'
            f' · designed by {html.escape(_source_label(c.get("design_source")))}</p></article>')


def _source_label(src: str | None) -> str:
    """Human-readable design path for a card, straight off the gene's own ``design_source`` stamp."""
    return _SOURCE_LABEL.get(str(src or "").lower(), str(src or "unknown"))


def _path_banner(path: dict, n_family: int, bodies: list[dict] | None = None) -> str:
    """The banner that stops this page over-claiming. Top of the document, same weight as the headline.

    This exists because the page previously said 'text -> robot' over a run in which no model read any text. The
    fix is not to hide the offline path (it is the right default -- no key, no spend, reproducible on a fresh
    clone); it is to name it, name what it costs, and name what the other path would add.

    RESOLVING a backend is not the same as USING one, which is the trap this banner would otherwise fall into in
    its LLM form: a configured model that rate-limits, times out or returns a graph that fails grounding falls
    back PER BODY, and the run would still be headed 'an LLM designed these bodies'. So the LLM branch counts the
    ``design_source`` stamps the composer actually left and reports the split when it is not unanimous."""
    if path.get("llm"):
        model = f' ({html.escape(str(path.get("model")))})' if path.get("model") else ""
        stamps = [str(b.get("design_source") or "").lower() for b in (bodies or []) if not b.get("error")]
        n_llm = sum(1 for s in stamps if s in ("anatomy", "llm_anatomy_graph", "anatomy_graph", "intent"))
        split = ""
        if stamps and n_llm == 0:
            split = ('<p><b>&hellip;except that none of them carry an LLM design stamp.</b> The backend was '
                     'resolved but every body fell back to the offline compositor — unavailable, rate-limited, or '
                     'its graph failed grounding. Read this page as an offline run.</p>')
        elif stamps and n_llm < len(stamps):
            split = (f'<p><b>It did not design all of them.</b> {n_llm} of {len(stamps)} bodies carry an LLM '
                     f'design stamp; the rest fell back to the offline compositor when the model was unavailable '
                     f'or its graph failed grounding. Per-card labels below say which is which.</p>')
        body = (f'<p><b>An LLM designed these bodies.</b> Backend <code>{html.escape(str(path.get("backend")))}'
                f'</code>{model} was resolved for the <code>morphology</code> role and asked to author each body '
                f'as an anatomy graph — parts, topology, attachment, proportions — which the general compiler then '
                f'grounds into geometry. Each card is stamped with the path that actually produced it, so a body '
                f'the model declined or malformed shows its offline fallback instead of silently borrowing the '
                f'credit.</p>{split}')
    else:
        body = ('<p><b>No LLM was called in this run.</b> The demo runs with the project&rsquo;s <code>.env</code> '
                'suppressed (<code>VIRTUROID_NO_LOCAL_ENV=1</code>) and no backend configured in the environment '
                'either, so <code>get_llm()</code> returned nothing — that was checked at the start of the run, not '
                'assumed. Every body below was composed by the <b>offline compositor</b>: the prompt is routed by '
                'keyword to a body class, then built by whichever deterministic builder fits it — animal proportion '
                'priors and size words shape a creature, a requirements recipe (reach, payload) shapes an arm — and '
                'the geometry is grounded from there. That is a real generator, not a stored model. But its '
                'vocabulary is the words someone wrote down, so a creature nobody enumerated composes to the generic '
                'body plan, and each card below says which builder it came from.</p>'
                '<p><b>What the LLM path adds</b> is exactly that vocabulary: it authors the anatomy for a creature '
                'the keyword router has never heard of. Re-run with <code>--llm</code> and a configured backend to '
                'use it. <b>What does not change either way:</b> the physics, the renders and the verdicts. Those '
                'are MuJoCo rollouts of the body that was actually built, on both paths.</p>')
    extra = (f'<p>The <b>same-family comparison</b> further down is the load-bearing measurement on this page: '
             f'{n_family} prompts from one family, and whether they produce {n_family} different bodies or one body '
             f'{n_family} times — measured, with the numbers printed.</p>' if n_family else "")
    return f'<section class="honesty"><h2>How this run was built</h2>{body}{extra}</section>'


def _family_html(rows: list[dict]) -> str:
    """The differentiation panel: renders side by side + a table of MEASURED body numbers.

    Read this table adversarially, which is how it is built: if the walkability gate had replaced a body with the
    shared fanned template, the SUBSTITUTED column would say so and the mass/proportion columns would collapse to
    one repeated row -- which is precisely what they used to do (13.500 kg / 0.498 for every quadruped prompt)."""
    if not rows:
        return ""
    figs, trs = [], []
    for r in rows:
        if r.get("error"):
            figs.append(f'<figure class="fam err"><figcaption>{html.escape(str(r.get("label")))}</figcaption>'
                        f'<pre>{html.escape(r["error"])}</pre></figure>')
            continue
        img = r.get("img")
        media = (f'<img src="data:{img[0]};base64,{img[1]}" alt="{html.escape(str(r.get("prompt")))}"/>'
                 if img else '<div class="noimg">no render</div>')
        figs.append(f'<figure class="fam"><div class="media">{media}</div>'
                    f'<figcaption>&ldquo;{html.escape(str(r.get("prompt")))}&rdquo;</figcaption></figure>')
        sub = ('<span class="bad">SUBSTITUTED</span>' if r.get("substituted")
               else '<span class="ok">kept — its own body</span>')
        g = r.get("gait") or {}
        gait = (f'freq {round(float(g.get("freq", 0)), 2)} / kp {round(float(g.get("kp", 0)), 1)}' if g
                else 'shipped default')
        # the fitter's own sentence ("searched N gaits ... walks X m vs the default's Y m") on hover: too long for
        # a column, too good to throw away -- it is the receipt for the number in the cell
        gtip = f' title="{html.escape(str(r.get("gait_note")))}"' if r.get("gait_note") else ""
        # ...and the MARGIN in the cell itself, not only in the tooltip. An operating point no perturbed copy of
        # itself survives is one lucky float, and a reader comparing two rows cannot see that from freq/kp.
        _rel = r.get("robustness_rel")
        margin = ('' if not g else
                  f' &middot; survives &plusmn;{_rel:g}' if _rel is not None else
                  ' &middot; <span class="bad">FRAGILE</span>' if r.get("fragile") else '')
        vcls = _verdict_class(r.get("verdict", ""), r.get("credible"))
        trs.append(
            f'<tr><th>{html.escape(str(r.get("label")))}</th>'
            f'<td>{r.get("mass_kg")}</td><td>{r.get("leg_m")}</td><td>{r.get("torso_m")}</td>'
            f'<td>{r.get("torso_over_leg")}</td><td>{r.get("segments")}/{r.get("dof")}</td>'
            f'<td class="gait"{gtip}>{html.escape(gait)}{margin}</td><td>{sub}</td>'
            f'<td><span class="badge {vcls}">{html.escape(str(r.get("verdict","")))}</span> '
            f'{("" if r.get("forward_m") is None else str(r.get("forward_m")) + " m")}</td></tr>')
    dog_note = ('The dog card in the gallery above is <i>this same robot</i>, composed once and shared between the '
                'two sections: there it is measured after a flywheel training pass, here at build time with no '
                'training at all, which is why its two distances differ. '
                if any((r.get("label") or "") == "dog" for r in rows) else "")
    # SAY WHAT THE OPERATING-POINT COLUMN ACTUALLY SAYS. The build-time fitter only keeps a searched gait when it
    # beats the shipped default, so on a run where every body walks on the default the column reads "shipped
    # default" three times -- and a fixed caption crediting "their fitted operating points" would be describing a
    # different run. Which of the two happened is itself the interesting bit, so report it either way.
    n_own = sum(1 for r in rows if not r.get("error") and r.get("gait"))
    n_ok = sum(1 for r in rows if not r.get("error"))
    if n_own == 0:
        gait_note = ('Every one of them walked on the shipped default gait — none needed an operating point '
                     'searched for it, which is the cheap outcome, not a missing step: the fitter only keeps a '
                     'searched gait when it beats the default. ')
    elif n_own == n_ok:
        gait_note = (f'All {n_ok} needed an operating point searched for its own body — the shipped default did '
                     f'not walk them, and the column names what did. ')
    else:
        gait_note = (f'{n_own} of {n_ok} needed an operating point searched for its own body; the other '
                     f'{n_ok - n_own} already walked on the shipped default. ')
    # NAME THE CEILING, and only when the data shows it. The offline compositor re-proportions ONE skeleton, so
    # the segs/DOF column repeats -- that is the honest limit of an offline differentiation claim and the exact
    # thing the LLM path exists to lift. Conditional, because a mixed-topology family (a 6-leg prompt) would make
    # the sentence false, and a fixed caption that can go false is the failure mode this whole page is about.
    topo = {(r.get("segments"), r.get("dof")) for r in rows if not r.get("error") and r.get("segments")}
    topology_note = (
        '<p class="small"><b>And the limit of that claim, since the table shows it anyway:</b> the segs/DOF column '
        'is identical down the page. On this run the prompts differentiate <i>proportion, scale and the fitted '
        'operating point</i> — they do not author a different skeleton. Same body plan, re-proportioned, each one '
        're-tuned until it walks. Authoring a genuinely different anatomy for a creature nobody wrote down is what '
        'the LLM design path is for.</p>' if len(topo) == 1 and len(rows) > 1 else "")
    return f"""<section class="family">
  <h2>Same family, different bodies</h2>
  <p>Breadth (a leg, an arm, a wheel) is easy to fake with a handful of unrelated templates. The harder question:
     {len(rows)} prompts from <b>one</b> family. Until recently the honest answer was that they collapsed — the
     walkability gate judged every authored quadruped with another robot&rsquo;s hand-tuned operating point, all of
     them measured ~0&nbsp;m, and all of them were replaced by the same fanned template at a byte-identical
     13.500&nbsp;kg and torso:leg&nbsp;0.498. Building now grounds the body, fits an operating point <i>to</i> it,
     and only then asks whether it can walk.</p>
  <div class="famrow">{"".join(figs)}</div>
  {_family_headline(rows)}
  <div class="tablewrap"><table>
    <thead><tr><th></th><th>mass (kg)</th><th>leg (m)</th><th>torso (m)</th><th>torso:leg</th>
      <th>segs/DOF</th><th>operating point</th><th>body</th><th>verdict</th></tr></thead>
    <tbody>{"".join(trs)}</tbody>
  </table></div>
  <p class="small">Every body here was built and verified identically — same 800-step quick rollout, no flywheel
     pass — so what differs is the bodies. {gait_note}{dog_note}These are original generated bodies, not scans or
     scale models of the animals; the claim is that the prompts produce <i>different</i> robots that <i>each</i>
     walk, not that a robot horse is to scale.</p>
  {topology_note}
</section>"""


def _family_headline(rows: list[dict]) -> str:
    """COMPUTE the panel's claim from the panel's own numbers, so it can contradict itself in public.

    A comparison table is only evidence if the reader does not have to do the comparing. This counts distinct
    bodies by (mass, leg, torso) and prints what it finds -- so a run where the gate DOES collapse two prompts
    onto one template renders '3 prompts -> 2 distinct bodies' in the same red the failing verdicts use, and the
    section stops claiming differentiation without anyone editing the prose."""
    ok = [r for r in rows if not r.get("error") and r.get("mass_kg") is not None]
    if not ok:
        return ""
    distinct = {(r.get("mass_kg"), r.get("leg_m"), r.get("torso_m")) for r in ok}
    subs = sum(1 for r in ok if r.get("substituted"))
    walks = sum(1 for r in ok if _verdict_class(r.get("verdict", ""), r.get("credible")) == "good")
    cls = "good" if (len(distinct) == len(ok) and subs == 0) else "bad"
    masses = " / ".join(str(r.get("mass_kg")) for r in ok)
    return (f'<p class="headline"><span class="badge {cls}">{len(ok)} prompts &rarr; {len(distinct)} distinct '
            f'bodies</span> {masses} kg &middot; {subs} substituted &middot; {walks}/{len(ok)} credible walks</p>')


def render_html(cards: list[dict], *, quick: bool, path: dict | None = None,
                family: list[dict] | None = None) -> str:
    good = sum(1 for c in cards if not c.get("error") and _verdict_class(c.get("verdict",""), c.get("credible")) == "good")
    total = len(cards)
    cards_html = "\n".join(_card_html(c) for c in cards)
    mode = "quick" if quick else "full"
    path = path if path is not None else {"llm": False, "backend": None}
    banner = _path_banner(path, len(family or []), cards + (family or []))
    family_html = _family_html(family or [])
    # the design-path chip must track the RUN, not a slogan: 'zero internal LLM tokens' is a true and load-bearing
    # claim offline and a false one under --llm, and a page that gets that wrong has no standing to lecture
    # anyone about honest verdicts.
    chip = (f'designed by {html.escape(str(path.get("backend")))}' if path.get("llm")
            else 'offline compositor · zero internal LLM tokens')
    # only promise the reader a section this run actually produced (--mini has no comparison panel)
    pointer = ('     Two things worth reading before the pictures: <b>how this run was built</b>, immediately '
               'below, and the <b>same-family comparison</b> at the bottom, which is the measurement that '
               'matters most.' if family else
               '     Start with <b>how this run was built</b>, immediately below — it names which brain designed '
               'these bodies.')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Virturoid — text-to-robot gallery</title>
<style>
:root {{ --bg:#0b0e14; --panel:#141922; --line:#232a36; --ink:#e6edf3; --dim:#8b98a9;
        --good:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
header {{ padding:40px 28px 22px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:-.02em; }}
h1 span {{ color:var(--accent); }}
header p {{ margin:0; color:var(--dim); max-width:70ch; }}
.stat {{ display:inline-block; margin-top:14px; padding:6px 12px; border:1px solid var(--line);
         border-radius:20px; color:var(--dim); font-size:13px; }}
.stat b {{ color:var(--good); }}
main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:18px; padding:24px 28px 60px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; overflow:hidden;
         display:flex; flex-direction:column; }}
.media {{ background:#05070b; aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; }}
.media img {{ width:100%; height:100%; object-fit:contain; }}
.noimg {{ color:var(--dim); font-size:13px; }}
.card h3 {{ margin:14px 16px 2px; font-size:16px; }}
.note {{ margin:0 16px 10px; color:var(--dim); font-size:13px; }}
.badge {{ margin:0 16px 8px; padding:5px 10px; border-radius:7px; font-size:12.5px; font-weight:600;
          display:inline-block; align-self:flex-start; }}
.badge.good {{ background:rgba(63,185,80,.14); color:var(--good); border:1px solid rgba(63,185,80,.4); }}
.badge.warn {{ background:rgba(210,153,34,.14); color:var(--warn); border:1px solid rgba(210,153,34,.4); }}
.badge.bad  {{ background:rgba(248,81,73,.14); color:var(--bad); border:1px solid rgba(248,81,73,.4); }}
.metrics {{ margin:0 16px 8px; color:var(--dim); font-size:12.5px; }}
.metrics b {{ color:var(--ink); font-weight:600; }}
.fly {{ margin:6px 16px 8px; }}
.flylab {{ color:var(--dim); font-size:12px; margin-bottom:5px; }}
.bar {{ position:relative; height:20px; background:#0b0f16; border-radius:5px; margin-bottom:5px; overflow:hidden; }}
.bar span {{ position:absolute; inset:0 auto 0 0; border-radius:5px; }}
.bar .b1 {{ background:#30363d; }}
.bar .b2 {{ background:linear-gradient(90deg,#238636,#3fb950); }}
.bar em {{ position:relative; z-index:1; font-style:normal; font-size:11.5px; line-height:20px;
           padding-left:8px; color:var(--ink); }}
.foot {{ margin:auto 16px 14px; color:var(--dim); font-size:11.5px; border-top:1px solid var(--line); padding-top:10px; }}
.card.err {{ border-color:rgba(248,81,73,.4); }}
.card.err pre {{ margin:0 16px 14px; color:var(--bad); font-size:11.5px; white-space:pre-wrap; }}
footer {{ padding:20px 28px 50px; color:var(--dim); font-size:12.5px; border-top:1px solid var(--line); }}
code {{ background:#0b0f16; border:1px solid var(--line); border-radius:4px; padding:1px 5px; font-size:12.5px; }}
.honesty {{ margin:22px 28px 0; padding:18px 20px; background:#111722; border:1px solid var(--line);
            border-left:3px solid var(--accent); border-radius:10px; max-width:88ch; }}
.honesty h2 {{ margin:0 0 8px; font-size:15px; letter-spacing:.02em; text-transform:uppercase; color:var(--accent); }}
.honesty p {{ margin:0 0 10px; color:var(--dim); font-size:13.5px; }}
.honesty p:last-child {{ margin-bottom:0; }}
.honesty b {{ color:var(--ink); }}
.family {{ padding:6px 28px 54px; border-top:1px solid var(--line); }}
.family h2 {{ margin:26px 0 8px; font-size:20px; letter-spacing:-.01em; }}
.family p {{ margin:0 0 14px; color:var(--dim); max-width:88ch; font-size:13.5px; }}
.family b {{ color:var(--ink); }}
.family p.small {{ font-size:12.5px; margin-top:14px; }}
.family p.headline {{ color:var(--ink); font-size:13.5px; margin-bottom:12px; }}
.family p.headline .badge {{ margin:0 8px 0 0; padding:4px 10px; }}
.famrow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; margin-bottom:18px; }}
figure.fam {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
figure.fam figcaption {{ padding:10px 14px; font-size:13px; color:var(--dim); }}
figure.fam.err pre {{ margin:14px; color:var(--bad); font-size:11.5px; white-space:pre-wrap; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:13px; min-width:640px; }}
th, td {{ text-align:left; padding:8px 14px 8px 0; border-bottom:1px solid var(--line); white-space:nowrap; }}
thead th {{ color:var(--dim); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
tbody th {{ color:var(--ink); }}
td .badge {{ margin:0; padding:3px 8px; font-size:11.5px; }}
td.gait {{ color:var(--dim); }}
.ok {{ color:var(--good); }}
table .bad {{ color:var(--bad); font-weight:600; }}
</style></head><body>
<header>
  <h1>Virturoid <span>text → robot</span></h1>
  <p>Each robot below was generated from the plain-text prompt shown, then simulated and scored with an
     un-gameable verdict. Renders and metrics are the real ones — bodies that don't work say so.
{pointer}</p>
  <div class="stat"><b>{good}</b>/{total} credible · mode: {mode} · {chip}</div>
</header>
{banner}
<main>
{cards_html}
</main>
{family_html}
<footer>Generated by <code>scripts/run_mvp_demo.py</code>. Self-contained (renders embedded).
  Verdicts: forward displacement + cadence + upright for legged; friction grasp + lift for arms; wheel contact for
  mobile. The flywheel bars compare verify-before vs verify-after learning a gait on the same body.</footer>
</body></html>"""


def run_gallery(prompts: list[dict], out_dir: str, *, quick: bool = False, learn: bool = True,
                compare: list[dict] | None = None, offline: bool = True) -> str:
    """Build the battery and write a self-contained index.html. Returns the HTML path.

    ``offline`` (default) suppresses the project ``.env`` so no API key is needed and no model is called; the
    page says so. ``compare`` runs the same-family differentiation panel and, being built first, donates any
    body whose prompt the battery also uses."""
    if offline:
        # set HERE rather than at import, so --llm can decline it before the first virturoid import
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VIRTUROID_SESSION_DIR", str(out / "session"))
    _say(f"output dir: {out.resolve()}")
    path_info = _design_path()
    _say("design path: " + (f"LLM backend '{path_info.get('backend')}'" if path_info.get("llm")
                            else "OFFLINE compositor (no LLM enabled for this run)"))
    n_bodies = len({s["prompt"] for s in prompts} | {s["prompt"] for s in (compare or [])})
    _say(f"plan: {n_bodies} bodies"
         f"{f' ({len(compare)} in the comparison panel, {len(prompts)} in the gallery, the dog shared)' if compare else ''}"
         f"{'; each legged body also gets a flywheel learn pass' if learn else ''}")
    # SAY WHAT IT COSTS, BEFORE it costs it. Measured on this checkout (CPU, no GPU, cold memory bank):
    # --mini took 336 s, of which 321 s were ONE call -- create_robot on the quadruped -- and the run printed
    # nothing at all for those 321 s. The cost is real (create_robot fits an operating point to each body by
    # bounded search; see ai_native_tools.create_robot's cost note), so the honest fix is to name it up front
    # and then keep talking while it happens.
    _say("cost: minutes, not seconds -- most of it in create_robot, which grounds each body and then SEARCHES")
    _say("      for a gait that fits it (a legged body is ~2-6 min on CPU; an arm or a rover is seconds).")
    _say("      Every stage prints when it starts and what it cost; a '... still running' line every "
         f"{int(_HEARTBEAT_S)}s means it is working, not wedged.")
    family: list[dict] = []
    if compare:
        _say(f"[compare] same-family differentiation ({len(compare)} bodies)")
        family = build_family(compare)
    reuse = {r["prompt"]: r for r in family if r.get("robot_id") and not r.get("error")}
    cards = []
    for i, spec in enumerate(prompts, 1):
        _say(f"[{i}/{len(prompts)}] {spec['prompt']}", 3)
        c = build_one(spec, quick=quick, learn=learn, reuse=reuse)
        # _say, not a bare f-string: the verdicts are interpolated here and some carry typographic dashes
        # ("a scripted gait can't balance a walking biped - needs a learned policy"), which a cp1252 console
        # renders as a black diamond. Sanitising at the print site covers every string, not one known literal.
        _say(f"-> {c.get('verdict', c.get('error'))}  ({c.get('seconds')}s for this robot)", 3)
        cards.append(c)
    path = out / "index.html"
    path.write_text(render_html(cards, quick=quick, path=path_info, family=family), encoding="utf-8")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the self-contained text-to-robot demo gallery.")
    ap.add_argument("--out", default="build/demo", help="output dir (default build/demo)")
    ap.add_argument("--mini", action="store_true", help="2-robot quick battery (smoke)")
    ap.add_argument("--quick", action="store_true", help="quick verify mode (faster, less definitive)")
    ap.add_argument("--no-learn", action="store_true", help="skip the flywheel learn pass")
    ap.add_argument("--no-compare", action="store_true", help="skip the same-family differentiation panel")
    ap.add_argument("--llm", action="store_true",
                    help="use the configured LLM design path (reads the project .env) instead of the offline "
                         "compositor; the page reports which one actually ran")
    args = ap.parse_args()
    battery = MINI_BATTERY if args.mini else FULL_BATTERY
    quick = args.quick or args.mini
    learn = not args.no_learn and not args.mini
    # the comparison is the point of the full run; --mini stays a 2-card smoke pass (and a 2-card page has no
    # room for a differentiation claim it did not measure)
    compare = None if (args.mini or args.no_compare) else FAMILY_COMPARE
    t0 = time.time()
    _say(f"Virturoid demo gallery -- {'mini (2 robots)' if args.mini else 'full battery'}, "
         f"offline, CPU only, no API key.")
    path = run_gallery(battery, args.out, quick=quick, learn=learn, compare=compare, offline=not args.llm)
    n = len({s["prompt"] for s in battery} | {s["prompt"] for s in (compare or [])})   # the dog is shared, not doubled
    took = round(time.time() - t0, 1)
    _say(f"Gallery ({n} robots) written in {took}s ({int(took // 60)}m{int(took % 60):02d}s):")
    _say(f"  {path}")
    _say("Open it in a browser - it's self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
