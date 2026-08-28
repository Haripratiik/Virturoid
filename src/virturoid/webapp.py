"""Virturoid browser debug surface over the build/evaluate/iterate services.

Virturoid Studio (``python -m virturoid.desktop``) is the primary product UI. This
FastAPI app remains as a local debugging/control surface for exercising the same
build, preview, viewer, memory, and artifact endpoints in a browser.

Run:  python -m virturoid.webapp   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from virturoid.services.agent import VirturoidAgent, parse_intent

_STATIC = Path(__file__).parent / "webstatic"


def create_app(workspace: Path) -> FastAPI:
    workspace = Path(workspace)
    agent = VirturoidAgent(workspace, target_success=0.8)
    project_dir = agent.project_dir
    jobs: dict[str, dict] = {}
    lock = threading.Lock()

    app = FastAPI(title="Virturoid Debug Surface")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/app.js", response_class=PlainTextResponse)
    def appjs():
        return PlainTextResponse((_STATIC / "app.js").read_text(encoding="utf-8"), media_type="text/javascript")

    @app.get("/app.css", response_class=PlainTextResponse)
    def appcss():
        return PlainTextResponse((_STATIC / "app.css").read_text(encoding="utf-8"), media_type="text/css")

    @app.get("/vendor/{name}", response_class=PlainTextResponse)
    def vendor(name: str):
        target = (_STATIC / "vendor" / name).resolve()
        if (_STATIC / "vendor").resolve() not in target.parents or not target.exists():
            return PlainTextResponse("not found", status_code=404)
        return PlainTextResponse(target.read_text(encoding="utf-8"), media_type="text/javascript")

    @app.post("/api/chat")
    async def chat(payload: dict):
        message = (payload or {}).get("message", "").strip()
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)
        intent, params = parse_intent(message)
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"events": [], "done": False, "result": None, "error": None, "intent": intent}
        threading.Thread(target=_run_job, args=(job_id, message, intent, params), daemon=True).start()
        return {"job_id": job_id, "intent": intent}

    def _run_job(job_id: str, message: str, intent: str, params: dict):
        job = jobs[job_id]

        def emit(event: dict):
            job["events"].append(event)

        try:
            if intent in {"build", "iterate", "adjust_and_rebuild"}:
                from virturoid.services.autonomous_build import autonomous_build

                if intent == "build":
                    agent.prompt = params.get("prompt", message)
                elif intent == "adjust_and_rebuild" and params.get("reach_m"):
                    agent.reach_m = params["reach_m"]
                    agent.prompt = f"{agent.prompt} (reach {agent.reach_m} m)"
                elif intent == "iterate":
                    agent.target_success = min(1.0, agent.target_success + 0.1)
                if not agent.prompt:
                    emit({"stage": "error", "message": "Tell me what to build first."})
                    job["result"] = {"message": "Tell me what to build first."}
                else:
                    report = autonomous_build(
                        agent.prompt, project_dir, target_success_rate=agent.target_success,
                        memory_dir=workspace / "memory", progress=emit,
                    )
                    agent.last = {"succeeded": report.succeeded, "final_success_rate": report.final_success_rate,
                                  "converged_design": report.converged_design, "compute": report.compute}
                    job["result"] = {"message": f"Build {'succeeded' if report.succeeded else 'did not reach target'}: "
                                                f"task success {report.final_success_rate:.0%}.",
                                     "report": _project_summary(project_dir)}
            else:
                emit({"stage": intent, "message": f"Working on: {message}"})
                response = agent.handle(message)
                emit({"stage": "done", "message": response.message})
                job["result"] = {"message": response.message, "report": _project_summary(project_dir)}
        except Exception as exc:  # noqa: BLE001
            job["error"] = f"{exc}"
            emit({"stage": "error", "message": f"Error: {exc}"})
            traceback.print_exc()
        finally:
            job["done"] = True

    @app.get("/api/job/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return {"events": job["events"], "done": job["done"], "result": job["result"], "error": job["error"]}

    @app.get("/api/project")
    def project():
        return _project_summary(project_dir)

    @app.post("/api/viewer")
    def viewer(payload: dict | None = None):
        from virturoid.services.mujoco_runner import mujoco_available

        if not _has_project(project_dir):
            return JSONResponse({"error": "no robot built yet"}, status_code=409)
        if not mujoco_available():
            return JSONResponse({"error": "mujoco not installed"}, status_code=409)
        from virturoid.services.viewer_sim import simulate_episode_for_viewer

        scene_index = int((payload or {}).get("scene_index", 0))
        with lock:  # mujoco stepping is not reentrant across threads
            data = simulate_episode_for_viewer(project_dir, scene_index=scene_index)
        return data

    @app.post("/api/preview")
    def preview(payload: dict | None = None):
        """Compose a robot from the prompt and return one renderable pose.

        Mirrors the native desktop ComposeWorker path so browser preview does not drift away from Studio.
        """
        from virturoid.services.mujoco_runner import mujoco_available

        prompt = ((payload or {}).get("prompt") or "").strip()
        if not prompt:
            return JSONResponse({"error": "Describe a robot first."}, status_code=400)
        if not mujoco_available():
            return JSONResponse({"error": "MuJoCo is not installed."}, status_code=409)
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, write_packaged_visual_mjcf
        from virturoid.services.robot_factory import build_robot
        from virturoid.services.viewer_sim import _geom_metadata

        try:
            result = build_robot(prompt)
            gene = result["gene"]
            mesh_uris = {}
            # ONE DIRECTORY PER REQUEST. ``.preview`` used to be shared by every /api/preview call, which was
            # merely wasteful while the visual writer only ever ADDED files -- and became destructive the moment
            # it started removing the ones the current robot does not use. Measured with two concurrent
            # previews of different robots into the shared directory: one lost 7 of its own 7 mesh assets and
            # the other 5 of 18, so the URIs each response had just handed the browser 404'd. The assets have to
            # outlive the response (the browser fetches them afterwards through /api/artifact-binary), so a lock
            # around the write would not have helped; the requests need separate directories.
            rel_root = Path(".preview") / uuid.uuid4().hex[:12]
            preview_root = project_dir / rel_root
            visual = write_packaged_visual_mjcf(gene, preview_root, include_floor=True)
            if visual:
                model = mujoco.MjModel.from_xml_path(str(preview_root / visual["model_uri"]))
                index = json.loads((preview_root / visual["mesh_index_uri"]).read_text(encoding="utf-8"))
                mesh_uris = {
                    name: {**meta, "uri": (rel_root / meta["uri"]).as_posix()}
                    for name, meta in index.get("meshes", {}).items()
                }
            else:
                model = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
            _reap_preview_dirs(project_dir)
        except Exception as exc:  # noqa: BLE001 - a bad/unbuildable prompt must not 500
            return JSONResponse({"error": f"Could not compose a robot: {exc}"}, status_code=400)

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        quat = np.zeros(4)
        frame = []
        for g in range(model.ngeom):
            pos = data.geom_xpos[g]
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g])      # [w, x, y, z]
            frame.append([float(pos[0]), float(pos[1]), float(pos[2]),
                          float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])
        ee = gene.end_effector_type
        return {
            "preview": True, "task": "preview",
            "geoms": _geom_metadata(model, mesh_uris),
            "frames": [frame], "frame_count": 1,
            "outcome": {"status": "preview", "placed_count": 0, "block_count": 0},
            "robot": {
                "name": gene.species or gene.id, "species": gene.species,
                "robot_class": gene.robot_class,
                "links": [s.name for s in gene.segments],
                "end_effectors": [] if (not ee or ee == "none") else [ee],
                "dof": len(gene.actuated_joints()),
                "valid": not gene.validate(),
                "design_source": getattr(gene, "design_source", "heuristic"),
                "composition_notes": list(getattr(gene, "composition_notes", [])),
                "intent": result.get("intent", {}),
                "requires_clarification": bool(result.get("requires_clarification", False)),
            },
            "scenes": [],
        }

    @app.get("/api/memory")
    def memory():
        """Virturoid Memory snapshot: what the AIs have learned across builds."""
        from virturoid.services.memory_db import MemoryDB

        db_path = workspace / "memory" / "virturoid_memory.db"
        if not db_path.exists():
            return {"exists": False, "stats": {"runs": 0, "designs": 0, "failures": 0, "transfers": 0, "species": 0}, "recent": []}
        with MemoryDB(db_path) as db:
            recent = db.conn.execute(
                "SELECT prompt, robot_class, task_type, success_rate, succeeded, backend, design_source, created_at "
                "FROM runs ORDER BY id DESC LIMIT 10"
            ).fetchall()
            return {
                "exists": True,
                "stats": db.stats(),
                "recent": [dict(r) for r in recent],
                "trainable_rows": len(db.training_dataset(min_success=0.8)),
            }

    @app.get("/api/artifact")
    def artifact(path: str):
        target = (project_dir / path).resolve()
        if project_dir.resolve() not in target.parents or not target.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return PlainTextResponse(target.read_text(encoding="utf-8"))

    @app.get("/api/artifact-binary")
    def artifact_binary(path: str):
        """Serve one package asset to the local Studio viewer.

        The containment check is deliberately identical to the text artifact endpoint;
        exposing an arbitrary local-file route for a convenience viewer would be unsafe.
        """
        target = (project_dir / path).resolve()
        if project_dir.resolve() not in target.parents or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(target, media_type="application/octet-stream")

    return app


#: preview scratch directories to keep, and how long one is guaranteed to live. Every /api/preview writes its
#: own; the browser fetches the meshes AFTER the response returns, so a directory is not free to go the moment
#: the next request starts.
_PREVIEW_KEEP = 8
_PREVIEW_MIN_AGE_S = 120.0
_PREVIEW_DIR_RE = re.compile(r"^[0-9a-f]{12}$")


def _reap_preview_dirs(project_dir: Path) -> None:
    """Drop old per-request preview scratch directories under ``.preview/``.

    The only things removed are directories THIS server minted: the name has to be one of our 12-hex request
    tokens AND the directory has to carry the visual-model index we write into it. That is the same rule the
    exporters' prune follows -- delete what we can show is ours, leave everything else -- and it matters here
    because ``project_dir`` is the customer's workspace. A directory is also never removed while it is one of
    the ``_PREVIEW_KEEP`` most recent or younger than ``_PREVIEW_MIN_AGE_S``, so a response that has just handed
    the browser a set of mesh URIs cannot have them swept out from under it.
    """
    import shutil
    import time

    root = Path(project_dir) / ".preview"
    if not root.is_dir():
        return
    try:
        mine = [(p.stat().st_mtime, p) for p in root.iterdir()
                if p.is_dir() and _PREVIEW_DIR_RE.match(p.name)
                and (p / "simulation" / "viewer_mesh_index.json").is_file()]
    except OSError:
        return
    now = time.time()
    for mtime, p in sorted(mine, key=lambda t: t[0], reverse=True)[_PREVIEW_KEEP:]:
        if now - mtime < _PREVIEW_MIN_AGE_S:
            continue
        shutil.rmtree(p, ignore_errors=True)


def _has_project(project_dir: Path) -> bool:
    return (project_dir / "robot" / "robot_genome.json").exists()


def _project_summary(project_dir: Path) -> dict:
    if not _has_project(project_dir):
        return {"built": False}
    summary: dict = {"built": True}
    for key, rel in [
        ("autonomy", "reports/autonomy_report.json"),
        ("evaluation", "reports/physics_evaluation_report.json"),
        ("design_validation", "reports/design_validation_report.json"),
        ("genome", "robot/robot_genome.json"),
        ("contract", "reports/robot_package_contract.json"),
    ]:
        p = project_dir / rel
        if p.exists():
            try:
                summary[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    # Mobile-base navigation: normalize into the shared "evaluation" shape for the UI.
    nav_path = project_dir / "reports" / "navigation_evaluation_report.json"
    if "evaluation" not in summary and nav_path.exists():
        try:
            nav = json.loads(nav_path.read_text(encoding="utf-8"))
            summary["evaluation"] = {
                "success_rate": nav.get("success_rate"),
                "blocks_placed": nav.get("reached"),
                "blocks_total": nav.get("total_episodes"),
                "failure_clusters": nav.get("failure_clusters", []),
                "task": "navigation",
            }
        except Exception:  # noqa: BLE001
            pass
    summary["artifacts"] = sorted(str(p.relative_to(project_dir)) for p in project_dir.rglob("*") if p.is_file())[:400]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Virturoid browser debug surface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workspace", default=str(Path("build") / "webapp"))
    args = parser.parse_args()

    import uvicorn

    app = create_app(Path(args.workspace))
    print(f"Virturoid app -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
