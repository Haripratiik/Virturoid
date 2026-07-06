"""In-process job registry — the thin enabler that lets the UI watch the agent work live.

Generalizes the pattern already proven in ``virturoid.webapp`` (in-memory jobs dict + lock,
uuid ids, daemon worker threads, ``?since`` polling) into a reusable, stdlib-only service:

* ``create(kind, args, build_root)`` spawns a daemon worker and returns immediately.
* The worker wires ``autonomous_build``'s existing ``progress`` callback into the job's
  event list — the backend stages stay raw and honest; UI grouping happens client-side.
* ``request_cancel`` flips a flag the progress callback checks: the build aborts at its
  next emit, honestly leaving partial artifacts on disk (status ``cancelled``).
* ``max_concurrent=1`` for physics work: MuJoCo builds are CPU-heavy and not reentrant,
  so extra jobs queue behind a semaphore instead of thrashing.

Purely additive: nothing else in the server changes behavior because this module exists.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path


class JobCancelled(Exception):
    """Raised inside a worker when the job's cancel flag is set."""


_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_ORDER: list[str] = []
# MuJoCo physics is CPU-heavy and not safely reentrant -- run one heavy job at a time.
_HEAVY = threading.Semaphore(1)

JOB_KINDS = ("autonomous_build", "tool", "train_gene")


def _now() -> float:
    return time.time()


def _slug(value: str) -> str:
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.lower()).strip("._-")
    return cleaned[:48] or "virturoid_build"


def job_view(job: dict) -> dict:
    """Public job shape (events excluded; fetch those via events_since)."""
    return {
        "id": job["id"],
        "kind": job["kind"],
        "args": job["args"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "result": job["result"],
        "error": job["error"],
        "n_events": len(job["events"]),
    }


def create(kind: str, args: dict, build_root: Path) -> dict:
    """Create a job and start its daemon worker. Returns the job view immediately."""
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind '{kind}'; expected one of {JOB_KINDS}")
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "kind": kind,
        "args": dict(args or {}),
        "status": "queued",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "events": [],
        "result": None,
        "error": None,
        "cancel_requested": False,
    }
    with _LOCK:
        _JOBS[job_id] = job
        _ORDER.append(job_id)
    threading.Thread(target=_run_job, args=(job, Path(build_root)), daemon=True).start()
    return job_view(job)


def get(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return job_view(job) if job else None


def list_jobs() -> list[dict]:
    with _LOCK:
        return [job_view(_JOBS[jid]) for jid in reversed(_ORDER)]


def events_since(job_id: str, since: int) -> tuple[dict, list[dict]] | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return job_view(job), list(job["events"][max(0, int(since)):])


def request_cancel(job_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None or job["status"] not in {"queued", "running"}:
            return False
        job["cancel_requested"] = True
        return True


def _emit(job: dict, stage: str, message: str, data: dict | None = None) -> None:
    with _LOCK:
        job["events"].append(
            {
                "seq": len(job["events"]),
                "ts": _now(),
                "stage": stage,
                "message": message,
                "data": data or {},
            }
        )


def _progress_callback(job: dict):
    """Bridge autonomous_build's {stage, message, data} progress dicts into job events,
    checking the cancel flag at every emit (cancellation for free)."""

    def cb(event: dict) -> None:
        if job["cancel_requested"]:
            raise JobCancelled()
        _emit(job, str(event.get("stage", "")), str(event.get("message", "")), event.get("data") or {})

    return cb


def _run_job(job: dict, build_root: Path) -> None:
    with _HEAVY:
        with _LOCK:
            if job["cancel_requested"]:
                job["status"] = "cancelled"
                job["finished_at"] = _now()
                return
            job["status"] = "running"
            job["started_at"] = _now()
        try:
            if job["kind"] == "autonomous_build":
                job["result"] = _run_autonomous_build(job, build_root)
            elif job["kind"] == "train_gene":                 # agent-first (G-C): train/optimize a HELD gene
                from virturoid.services.agent_design_tools import run_train_gene_job
                job["result"] = run_train_gene_job(job["args"], _progress_callback(job))
            else:
                job["result"] = _run_tool(job)
            with _LOCK:
                job["status"] = "succeeded"
        except JobCancelled:
            _emit(job, "cancelled", "Job cancelled; partial artifacts stay on disk (honest state).")
            with _LOCK:
                job["status"] = "cancelled"
        except Exception as exc:  # noqa: BLE001 - job surface: report, don't crash the server
            _emit(job, "error", f"{type(exc).__name__}: {exc}")
            with _LOCK:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with _LOCK:
                job["finished_at"] = _now()


def _run_autonomous_build(job: dict, build_root: Path) -> dict:
    from virturoid.services.autonomous_build import autonomous_build

    args = job["args"]
    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("autonomous_build requires a 'prompt'")
    output_name = str(args.get("output_name") or _slug(prompt))
    output_dir = Path(build_root) / output_name
    report = autonomous_build(
        prompt,
        output_dir,
        target_success_rate=float(args.get("target", 0.8)),
        train=bool(args.get("train", False)),
        memory_dir=Path(build_root) / "memory",
        progress=_progress_callback(job),
    )
    return {
        "output_name": output_name,
        "package_dir": str(output_dir),
        "robot_class": getattr(report, "robot_class", None),
        "species": getattr(report, "species", None),
        "task_type": getattr(report, "task_type", None),
        "succeeded": bool(getattr(report, "succeeded", False)),
        "final_success_rate": float(getattr(report, "final_success_rate", 0.0) or 0.0),
        "species_exact": bool(getattr(report, "species_exact", True)),
        "species_note": getattr(report, "species_note", None),
        "memory_reused": bool(getattr(report, "memory_reused", False)),
    }


def _run_tool(job: dict) -> dict:
    from virturoid.services.agent_tools import call_tool

    args = job["args"]
    tool = str(args.get("tool") or "")
    tool_args = args.get("args") or {}
    _emit(job, "start", f"Running tool {tool}...")
    res = call_tool(tool, tool_args)
    if not res.get("ok"):
        raise RuntimeError(str(res.get("error") or f"tool {tool} failed"))
    _emit(job, "done", f"Tool {tool} finished.")
    return res
