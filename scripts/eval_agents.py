"""Offline-capable eval harness for Virturoid's LLM agents (plan §8, Phase E).

Measures the design layer the way a robotics platform should: not "did the model
say something", but "did it return a schema-valid, physics-feasible answer, and how
many self-repair rounds did it take". Run the same prompts against any backend
(Claude, your local Nemotron, or the deterministic mock) to decide whether a
fine-tuned local model is good enough to take over a role from Claude.

    # whatever VIRTUROID_LLM_BACKEND points at:
    python scripts/eval_agents.py

    # compare your local Nemotron head to head with Claude:
    python scripts/eval_agents.py --backend local  --url http://100.x.y.z:11434/v1 --model nemotron-3-nano
    python scripts/eval_agents.py --backend claude

    # only the part recommender, write a JSON report:
    python scripts/eval_agents.py --agents parts --out reports/agent_eval.json

Exit code is 0 if every selected agent met its pass threshold (default 0.9 valid).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Prompt suite: realistic tabletop/sorting tasks with a range of reach + payload.
_PROMPTS = [
    "Build a tabletop arm that sorts red and blue blocks into bins.",
    "Make a small desk robot that picks up light cubes within 40 cm and sorts by color.",
    "Design an arm that reaches 70 cm to sort heavier 200 g blocks into two bins.",
    "A compact pick-and-place arm for a 30 cm workspace sorting plastic chips.",
    "Build a sorting arm with an rgbd camera that reaches across a 60 cm table.",
]


def _eval_designer(llm) -> list[dict]:
    from virturoid.services.design_agent import propose_arm_design
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.task_builder import build_task_graph

    rows = []
    for p in _PROMPTS:
        req = build_requirements_from_prompt(p)
        task = build_task_graph(req)
        t0 = time.time()
        out = propose_arm_design(p, req, task, llm)
        rows.append({
            "prompt": p,
            "ok": bool(out and out.get("feasible")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": (out or {}).get("design") or (out or {}).get("issues"),
        })
    return rows


def _eval_parts(llm) -> list[dict]:
    from virturoid.fixtures.components import extended_component_library
    from virturoid.services.part_agent import recommend_parts
    from virturoid.services.requirements_builder import build_requirements_from_prompt

    cat = extended_component_library()
    rows = []
    for p in _PROMPTS:
        req = build_requirements_from_prompt(p, sensor="rgbd_camera" if "rgbd" in p or "camera" in p else None)
        t0 = time.time()
        out = recommend_parts(p, req, cat, llm)
        rows.append({
            "prompt": p,
            "ok": bool(out and out.get("feasible")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": (out or {}).get("selection") or (out or {}).get("issues"),
        })
    return rows


def _eval_failures(llm) -> list[dict]:
    from virturoid.services.failure_agent import analyze_failures

    suites = [
        [{"label": "missed_grasp", "count": 5}, {"label": "wrong_bin", "count": 1}],
        [{"label": "dropped", "count": 3}, {"label": "missed_grasp", "count": 2}],
        [{"label": "instability", "count": 4}],
    ]
    rows = []
    for clusters in suites:
        t0 = time.time()
        out = analyze_failures(clusters, llm)
        rows.append({
            "clusters": clusters,
            "ok": bool(out and out.get("valid")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": (out or {}).get("diagnosis") or (out or {}).get("issues"),
        })
    return rows


def _eval_tasks(llm) -> list[dict]:
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.task_agent import synthesize_task_graph

    rows = []
    for p in _PROMPTS:
        req = build_requirements_from_prompt(p)
        t0 = time.time()
        out = synthesize_task_graph(p, req, llm)
        rows.append({
            "prompt": p,
            "ok": bool(out and out.get("valid")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": ((out or {}).get("graph") or {}).get("task_type") or (out or {}).get("issues"),
        })
    return rows


def _eval_scenes(llm) -> list[dict]:
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.scene_agent import propose_scene_envelope

    rows = []
    for p in _PROMPTS:
        req = build_requirements_from_prompt(p)
        t0 = time.time()
        out = propose_scene_envelope(p, req, llm)
        rows.append({
            "prompt": p,
            "ok": bool(out and out.get("feasible")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": (out or {}).get("envelope") or (out or {}).get("issues"),
        })
    return rows


_SAMPLE_URDF = """<robot name="arm3">
  <link name="base_link"/><link name="upper_link"/><link name="forearm_link"/><link name="gripper">
  <joint name="shoulder" type="revolute"><parent link="base_link"/><child link="upper_link"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper_link"/><child link="forearm_link"/></joint>
  <joint name="wrist" type="revolute"><parent link="forearm_link"/><child link="gripper"/></joint>
</robot>"""


def _eval_parser(llm) -> list[dict]:
    from virturoid.services.robot_parser_agent import parse_robot_description

    rows = []
    for fmt in ("urdf", "urdf", "urdf"):
        t0 = time.time()
        out = parse_robot_description(_SAMPLE_URDF, fmt, llm)
        rows.append({
            "format": fmt,
            "ok": bool(out and out.get("valid")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": ((out or {}).get("draft") or {}).get("links") or (out or {}).get("issues"),
        })
    return rows


def _eval_transfer(llm) -> list[dict]:
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.transfer_agent import recommend_transfer

    candidates = [{
        "key": "manipulator__pick_place_sort", "robot_class": "manipulator",
        "task_type": "pick_place_sort", "success_rate": 0.95,
        "converged_design": {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12},
    }]
    rows = []
    for p in _PROMPTS[:3]:
        req = build_requirements_from_prompt(p)
        t0 = time.time()
        out = recommend_transfer(p, req, candidates, llm)
        rows.append({
            "prompt": p,
            "ok": bool(out and out.get("valid")),
            "attempts": (out or {}).get("attempts"),
            "latency_s": round(time.time() - t0, 2),
            "detail": (out or {}).get("decision") or (out or {}).get("issues"),
        })
    return rows


_AGENTS = {"design": _eval_designer, "parts": _eval_parts, "failures": _eval_failures,
           "tasks": _eval_tasks, "scenes": _eval_scenes, "parser": _eval_parser, "transfer": _eval_transfer}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Virturoid's LLM agents.")
    ap.add_argument("--backend", help="override VIRTUROID_LLM_BACKEND (claude|local|mock)")
    ap.add_argument("--url", help="override VIRTUROID_LOCAL_LLM_URL")
    ap.add_argument("--model", help="override the model id")
    ap.add_argument("--format", dest="fmt", help="local format mode: json_object|json_schema|text")
    ap.add_argument("--agents", default="design,parts,failures,tasks,scenes,parser,transfer",
                    help="comma list: design,parts,failures,tasks,scenes,parser,transfer")
    ap.add_argument("--threshold", type=float, default=0.9, help="min valid-rate to pass")
    ap.add_argument("--out", help="write a JSON report to this path")
    args = ap.parse_args(argv)

    if args.backend:
        os.environ["VIRTUROID_LLM_BACKEND"] = args.backend
    if args.url:
        os.environ["VIRTUROID_LOCAL_LLM_URL"] = args.url
    if args.fmt:
        os.environ["VIRTUROID_LOCAL_LLM_FORMAT"] = args.fmt
    if args.model:
        b = os.environ.get("VIRTUROID_LLM_BACKEND", "").lower()
        env = ("VIRTUROID_CLAUDE_MODEL" if b == "claude"
               else "VIRTUROID_OPENAI_MODEL" if b in ("openai", "chatgpt", "gpt")
               else "VIRTUROID_LOCAL_LLM_MODEL")
        os.environ[env] = args.model

    from virturoid.services.llm_client import _load_local_env, get_llm

    _load_local_env()  # fill from project-local .env (real env / CLI overrides win)
    backend = os.environ.get("VIRTUROID_LLM_BACKEND", "off")
    selected = [a.strip() for a in args.agents.split(",") if a.strip() in _AGENTS]
    print(f"backend: {backend} | agents: {selected} | pass threshold: {args.threshold:.0%}\n")

    report = {"backend": backend, "threshold": args.threshold, "agents": {}}
    all_passed = True
    for agent in selected:
        llm = get_llm({"design": "designer", "parts": "part_recommender",
                       "failures": "failure_analyst", "tasks": "task_graph", "scenes": "scene",
                       "parser": "scene", "transfer": "task_graph"}[agent])
        rows = _AGENTS[agent](llm)
        n = len(rows)
        ok = sum(1 for r in rows if r["ok"])
        valid_rate = ok / n if n else 0.0
        attempts = [r["attempts"] for r in rows if r.get("attempts")]
        avg_attempts = round(sum(attempts) / len(attempts), 2) if attempts else None
        avg_latency = round(sum(r["latency_s"] for r in rows) / n, 2) if n else 0.0
        passed = valid_rate >= args.threshold
        all_passed = all_passed and passed
        report["agents"][agent] = {
            "valid_rate": round(valid_rate, 3), "n": n, "avg_attempts": avg_attempts,
            "avg_latency_s": avg_latency, "passed": passed, "rows": rows,
        }
        flag = "PASS" if passed else "FAIL"
        backend_note = " (no backend: deterministic fallback returned None)" if llm is None else ""
        print(f"[{flag}] {agent:9s} valid={valid_rate:.0%} ({ok}/{n}) "
              f"avg_attempts={avg_attempts} avg_latency={avg_latency}s{backend_note}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport -> {args.out}")

    if backend.lower() in ("off", ""):
        print("\nNote: no backend configured, so agents returned None (offline fallback). "
              "Set --backend to actually exercise a model.")
        return 0
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
