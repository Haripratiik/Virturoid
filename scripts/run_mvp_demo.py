"""One-command MVP demo gallery (plan #4): build a diverse battery of robots from plain text, SEE them, VERIFY
them with un-gameable verdicts, and (for legged bodies) show the learned-gait FLYWHEEL make them better --
emitted as ONE self-contained HTML page you open in a browser. No server, no build step: renders are embedded as
base64, verdicts are the real ones (a robot that falls says so), and the flywheel bars are measured, not staged.

    python scripts/run_mvp_demo.py                 # full battery -> build/demo/index.html
    python scripts/run_mvp_demo.py --mini          # 2-robot quick pass (what the smoke test runs)
    python scripts/run_mvp_demo.py --no-learn       # skip the flywheel pass (faster)
    python scripts/run_mvp_demo.py --out build/demo # choose the output dir

Honesty (see memory 'verify-renders-not-just-numbers'): the gallery shows the render AND the measured verdict for
every robot, including weak/flagged ones -- never a curated win. The flywheel delta is verify-before vs verify-after
on the SAME held body at the SAME horizon.
"""
from __future__ import annotations

import argparse
import base64
import html
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

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


def build_one(spec: dict, *, quick: bool, learn: bool) -> dict:
    """create -> render -> verify (+ optional flywheel before/after for legged). Returns a card dict."""
    from virturoid.services.agent_tools import call_tool

    prompt = spec["prompt"]
    t0 = time.time()
    card: dict = {"prompt": prompt, "note": spec.get("note", ""), "error": None,
                  "img": None, "gif": None, "flywheel": None}
    try:
        rid = call_tool("create_robot", {"prompt": prompt})["result"]["robot_id"]
        card["robot_id"] = rid

        rv = call_tool("render_view", {"robot_id": rid}).get("result", {})
        arts = rv.get("artifacts") or []
        card["img"] = _b64(arts[0]) if arts else None

        mode = "quick" if quick else "full"
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
            lg = call_tool("learn_gait", {"robot_id": rid, "generations": 6, "pop": 16, "steps": 900}).get("result", {})
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
            f'<p class="foot">{html.escape(c.get("kind","?"))} · {c.get("seconds","?")}s</p></article>')


def render_html(cards: list[dict], *, quick: bool) -> str:
    good = sum(1 for c in cards if not c.get("error") and _verdict_class(c.get("verdict",""), c.get("credible")) == "good")
    total = len(cards)
    cards_html = "\n".join(_card_html(c) for c in cards)
    mode = "quick" if quick else "full"
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
</style></head><body>
<header>
  <h1>Virturoid <span>text → robot</span></h1>
  <p>Each robot below was generated from the plain-text prompt shown, then simulated and scored with an
     un-gameable verdict. Renders and metrics are the real ones — bodies that don't work say so.</p>
  <div class="stat"><b>{good}</b>/{total} credible · mode: {mode} · agent-first (zero internal LLM tokens)</div>
</header>
<main>
{cards_html}
</main>
<footer>Generated by <code>scripts/run_mvp_demo.py</code>. Self-contained (renders embedded).
  Verdicts: forward displacement + cadence + upright for legged; friction grasp + lift for arms; wheel contact for
  mobile. The flywheel bars compare verify-before vs verify-after learning a gait on the same body.</footer>
</body></html>"""


def run_gallery(prompts: list[dict], out_dir: str, *, quick: bool = False, learn: bool = True) -> str:
    """Build the battery and write a self-contained index.html. Returns the HTML path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VIRTUROID_SESSION_DIR", str(out / "session"))
    cards = []
    for i, spec in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {spec['prompt']} ...", flush=True)
        c = build_one(spec, quick=quick, learn=learn)
        print(f"      -> {c.get('verdict', c.get('error'))}  ({c.get('seconds')}s)", flush=True)
        cards.append(c)
    path = out / "index.html"
    path.write_text(render_html(cards, quick=quick), encoding="utf-8")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the self-contained text-to-robot demo gallery.")
    ap.add_argument("--out", default="build/demo", help="output dir (default build/demo)")
    ap.add_argument("--mini", action="store_true", help="2-robot quick battery (smoke)")
    ap.add_argument("--quick", action="store_true", help="quick verify mode (faster, less definitive)")
    ap.add_argument("--no-learn", action="store_true", help="skip the flywheel learn pass")
    args = ap.parse_args()
    battery = MINI_BATTERY if args.mini else FULL_BATTERY
    quick = args.quick or args.mini
    learn = not args.no_learn and not args.mini
    t0 = time.time()
    path = run_gallery(battery, args.out, quick=quick, learn=learn)
    print(f"\nGallery ({len(battery)} robots) written in {round(time.time()-t0,1)}s:\n  {path}\n"
          f"Open it in a browser — it's self-contained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
