"""Conversational build agent (Phase 4).

Turns Virturoid from a set of commands into software you *talk to*: a persistent
session that maps natural-language requests to build / evaluate / iterate /
perceive / train / export actions on a robot project. The intent parser is
deterministic (rule-based) today; an LLM can replace it behind the same
``parse_intent`` boundary without changing the action layer.

Example session:
    agent.handle("build a tabletop arm that sorts red and blue blocks")
    agent.handle("evaluate it")
    agent.handle("it keeps missing grasps, make it better")
    agent.handle("give it more reach")
    agent.handle("train a policy and export it")
    agent.handle("status")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentResponse:
    intent: str
    message: str
    data: dict = field(default_factory=dict)


def parse_intent(message: str) -> tuple[str, dict]:
    """Map a natural-language message to (intent, params). Deterministic."""
    text = message.lower().strip()

    # Parameter adjustments (reach / payload) embedded in the message.
    params: dict = {}
    reach = re.search(r"(\d+(?:\.\d+)?)\s*(cm|centimet|m\b|meter)", text)
    if ("reach" in text or "further" in text or "farther" in text or "longer" in text) and reach:
        value = float(reach.group(1))
        params["reach_m"] = value / 100.0 if reach.group(2).startswith(("cm", "centimet")) else value

    if re.search(r"\b(build|create|make me|design)\b.*\b(robot|arm|base|gripper|sorter|bot)\b", text) or text.startswith("build "):
        return "build", {"prompt": message}
    if "status" in text or "summary" in text or "where are we" in text:
        return "status", {}
    if "export" in text or "give me the package" in text or "ship" in text:
        return "export", {}
    if any(k in text for k in ["train", "learn a policy", "behavior clon", "behaviour clon"]):
        return "train", {}
    if any(k in text for k in ["perception", "camera", "vision", "perceive", "see"]):
        return "perceive", {}
    if "evaluate" in text or "test it" in text or "how good" in text or "run it" in text:
        return "evaluate", {}
    if reach and ("reach" in text or "further" in text or "farther" in text or "longer" in text):
        return "adjust_and_rebuild", params
    if any(k in text for k in ["iterate", "improve", "make it better", "better", "fix", "failing", "missing", "drops", "drop", "optimi"]):
        return "iterate", {"feedback": message}
    return "unknown", {}


class VirturoidAgent:
    """Stateful build agent over a single robot project."""

    def __init__(self, workspace: Path, target_success: float = 0.8) -> None:
        self.workspace = Path(workspace)
        self.target_success = target_success
        self.prompt: str | None = None
        self.reach_m: float | None = None
        self.last: dict = {}
        self._project_dir: Path = self.workspace / "project"

    @property
    def project_dir(self) -> Path:
        """The directory of the robot this agent is working on RIGHT NOW.

        This used to be a constant, ``workspace/project``, so every robot the agent ever built landed on top of
        the last one. That was survivable while a package was only overwritten; it stopped being survivable when
        ``build_gene_package`` learned to REFUSE a directory that already holds a different class of robot,
        because the artifacts do not all get rewritten -- a manipulator build leaves the quadruped's
        ``policy_type: trot_cpg_gait`` and its controller.py in place, since those writers only run on the
        locomotion path. MEASURED: "build a quadruped" then "build an arm" raised MixedRobotPackageError and the
        second request returned an error instead of a robot.

        The refusal is right about the package and wrong about this flow, so the flow moved: a build whose class
        differs from what the current directory holds opens a NEW one. The first robot still lands in
        ``workspace/project`` so nothing that knew that path breaks, and a REBUILD of the same robot
        (``_do_adjust_and_rebuild``, and every same-class build) keeps its directory, which is what
        ``autonomous_build`` needs to iterate a design in place.
        """
        return self._project_dir

    def _fresh_project_dir(self) -> Path:
        """The next unused sibling of ``workspace/project``. Named plainly; the package inside says what it is."""
        n = 2
        while (self.workspace / f"project_{n}" / "robot" / "robot_genome.json").exists():
            n += 1
        return self.workspace / f"project_{n}"

    def handle(self, message: str) -> AgentResponse:
        intent, params = parse_intent(message)
        handler = getattr(self, f"_do_{intent}", None)
        if handler is None:
            return AgentResponse(
                "unknown",
                "I can build a robot, evaluate it, iterate/improve it, add perception, train a policy, "
                "adjust reach, export the package, or report status. What would you like?",
            )
        return handler(params)

    def _build(self, reach_m=None):
        from virturoid.services.autonomous_build import autonomous_build

        from virturoid.services.gene_build import MixedRobotPackageError

        def _run(where):
            return autonomous_build(self.prompt, where, target_success_rate=self.target_success,
                                    memory_dir=self.workspace / "memory")

        try:
            report = _run(self.project_dir)
        except MixedRobotPackageError:
            # ASK THE BUILD, DO NOT PREDICT. Only the builder knows which of its four paths a prompt lands on --
            # it classifies the composed GENE, not the words -- so the honest way to find out whether this robot
            # belongs in the current directory is to try, and to move if it says no. The guard runs as its first
            # statement, before anything is written, so the previous robot's package is untouched by the attempt.
            stale_dir = self._project_dir
            self._project_dir = self._fresh_project_dir()
            report = _run(self.project_dir)
            # The refused attempt wrote a conflict note into the OLD package. We resolved it by building
            # elsewhere, so that package is once again a correct, single-robot package and must not keep
            # a file saying otherwise. (A conflict nobody resolved keeps its note -- that is the guard's.)
            for rel in ("PACKAGE_CONFLICT.md", "reports/package_conflict.json"):
                try:
                    (stale_dir / rel).unlink(missing_ok=True)
                except OSError:
                    pass
        self.last = {
            "succeeded": report.succeeded,
            "final_success_rate": report.final_success_rate,
            "converged_design": report.converged_design,
            "compute": report.compute,
        }
        return report

    def _do_build(self, params) -> AgentResponse:
        self.prompt = params["prompt"]
        report = self._build()
        verdict = "succeeded" if report.succeeded else "did not reach target yet"
        return AgentResponse(
            "build",
            f"Built and validated a {report.robot_class} ({report.species}). Autonomous build {verdict}: "
            f"task success {report.final_success_rate:.0%} (target {self.target_success:.0%}). "
            f"Body: {report.converged_design}. Package at {self.project_dir}.",
            self.last,
        )

    def _do_adjust_and_rebuild(self, params) -> AgentResponse:
        if not self.prompt:
            return AgentResponse("adjust_and_rebuild", "Tell me what to build first.")
        self.reach_m = params.get("reach_m", self.reach_m)
        self.prompt = f"{self.prompt} (reach {self.reach_m} m)" if self.reach_m else self.prompt
        report = self._build()
        return AgentResponse(
            "adjust_and_rebuild",
            f"Updated reach to {self.reach_m} m and rebuilt. Task success now {report.final_success_rate:.0%}.",
            self.last,
        )

    def _do_iterate(self, params) -> AgentResponse:
        if not self.prompt:
            return AgentResponse("iterate", "Tell me what to build first, then I can iterate on it.")
        # Raise the bar and re-run the closed loop (hardware + controller co-optimization).
        self.target_success = min(1.0, self.target_success + 0.1)
        report = self._build()
        return AgentResponse(
            "iterate",
            f"Iterated on your feedback ('{params.get('feedback', '')[:60]}'). "
            f"Re-ran the build/evaluate/co-optimize loop; task success {report.final_success_rate:.0%}.",
            self.last,
        )

    def _do_evaluate(self, params) -> AgentResponse:
        from virturoid.services.mujoco_runner import mujoco_available

        if not self._has_project():
            return AgentResponse("evaluate", "No robot built yet — ask me to build one first.")
        if not mujoco_available():
            return AgentResponse("evaluate", "MuJoCo isn't installed, so I can't run the real task evaluation.")
        from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

        report = run_physics_pick_place_evaluation(self.project_dir)
        clusters = ", ".join(f"{c.failure_label}×{c.count}" for c in report.failure_clusters) or "none"
        return AgentResponse(
            "evaluate",
            f"Ran real pick-and-place: {report.success_rate:.0%} success, "
            f"{report.blocks_placed}/{report.blocks_total} blocks placed. Failures: {clusters}.",
            {"success_rate": report.success_rate, "clusters": clusters},
        )

    def _do_perceive(self, params) -> AgentResponse:
        from virturoid.services.mujoco_runner import mujoco_available

        if not self._has_project():
            return AgentResponse("perceive", "Build a robot first, then I can run it with perception.")
        if not mujoco_available():
            return AgentResponse("perceive", "MuJoCo isn't installed, so I can't render/perceive.")
        from virturoid.services.perception_adapter import default_perception
        from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

        report = run_physics_pick_place_evaluation(self.project_dir, perception=default_perception(seed=7))
        return AgentResponse(
            "perceive",
            f"Ran the task with perception in the loop ({report.controller}): {report.success_rate:.0%} success "
            f"using camera-estimated object poses instead of privileged state.",
            {"success_rate": report.success_rate},
        )

    def _do_train(self, params) -> AgentResponse:
        from virturoid.services.mujoco_runner import mujoco_available

        if not self._has_project():
            return AgentResponse("train", "Build a robot first, then I can train a policy for it.")
        if not mujoco_available():
            return AgentResponse("train", "MuJoCo isn't installed, so I can't train against physics.")
        from virturoid.services.behavior_cloning import train_bc_policy

        result = train_bc_policy(self.project_dir)
        return AgentResponse(
            "train",
            f"Trained a behavior-cloning policy from {result['demonstrations']} expert demonstrations "
            f"(joint residual {result['residual_rad']} rad) and exported it with a policy card.",
            {"demonstrations": result["demonstrations"]},
        )

    def _do_export(self, params) -> AgentResponse:
        if not self._has_project():
            return AgentResponse("export", "Nothing built yet to export.")
        from virturoid.services.ros2_exporter import export_ros2_package

        ros2_root = export_ros2_package(self.project_dir)
        artifacts = sorted(str(p.relative_to(self.project_dir)) for p in self.project_dir.rglob("*") if p.is_file())
        return AgentResponse(
            "export",
            f"The buildable package is at {self.project_dir} ({len(artifacts)} files): Robot Genome, URDF, "
            f"MuJoCo model, scenes, BOM/CAD, control program, reports — plus a ROS2 package at "
            f"{ros2_root.relative_to(self.project_dir)}.",
            {"file_count": len(artifacts), "ros2_package": str(ros2_root)},
        )

    def _do_status(self, params) -> AgentResponse:
        if not self.prompt:
            return AgentResponse("status", "No project yet. Ask me to build a robot.")
        return AgentResponse(
            "status",
            f"Project: '{self.prompt}'. Target success {self.target_success:.0%}. "
            f"Last result: {json.dumps(self.last) if self.last else 'not yet evaluated'}.",
            self.last,
        )

    def _has_project(self) -> bool:
        return (self.project_dir / "robot" / "robot_genome.json").exists()
