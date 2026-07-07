"""Inbound PolicyImporter (Tiers P0/P1) — Phase 4 of the Input Ingestion Research Plan.

Closes the "no inbound PolicyImporter" gap for the two *safe* tiers that never execute user code:
  * P0 metadata: classify a checkpoint/config by extension.
  * P1 static parse: AST-parse a Python controller to recover imports, an entrypoint, ROS topic literals, a
    best-effort action dimension, and observation keys accessed as ``obs["..."]`` — with zero execution.
Then negotiate the interface against a target: the plan's acceptance mismatch is "policy outputs N actions but
the model has M actuated joints". Sandbox (P2) and native adapters (P3) are deferred. Standard-library only.
"""

from __future__ import annotations

import ast
import os
import re

from virturoid.schemas.policy_import import PolicyFramework, PolicyImportSpec, PolicyTier

_EXT_FRAMEWORK = {
    ".npz": PolicyFramework.NUMPY_NPZ,
    ".onnx": PolicyFramework.ONNX,
    ".pt": PolicyFramework.TORCH, ".pth": PolicyFramework.TORCH, ".jit": PolicyFramework.TORCH,
    ".ckpt": PolicyFramework.TORCH,
    ".yaml": PolicyFramework.YAML_CONFIG, ".yml": PolicyFramework.YAML_CONFIG,
    ".py": PolicyFramework.PYTHON_SCRIPT,
}

_ENTRYPOINT_NAMES = (
    "act", "policy", "predict", "forward", "compute_control", "control",
    "step", "__call__", "get_action", "main",
)
_ACTION_DIM_NAMES = {"action_dim", "n_actions", "num_actions", "act_dim"}
_OBS_BASES = {"obs", "observation", "observations", "state", "inputs"}


def sniff_policy_metadata(name_or_path: str) -> PolicyImportSpec:
    """Tier P0: classify a policy artifact by extension only. No execution, no parse."""
    base = os.path.basename(name_or_path)
    ext = os.path.splitext(base.lower())[1]
    framework = _EXT_FRAMEWORK.get(ext, PolicyFramework.UNKNOWN)
    warnings: list[str] = []
    if framework is PolicyFramework.UNKNOWN:
        warnings.append(f"Unrecognized policy extension '{ext or '(none)'}'.")
    slug = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_") or "artifact"
    return PolicyImportSpec(
        id=f"policy_{slug}", source_ref=name_or_path, framework=framework,
        tier=PolicyTier.P0_METADATA, validation_status="metadata_only", warnings=warnings,
    )


class _ControllerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.defs: list[str] = []
        self.ros_topics: set[str] = set()
        self.action_dim: int | None = None
        self.obs_keys: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.add(node.module.split(".")[0])
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defs.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.defs.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defs.append(node.name)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and re.match(r"^/[A-Za-z][\w/]*$", node.value):
            self.ros_topics.add(node.value)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.lower() in _ACTION_DIM_NAMES:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                    self.action_dim = node.value.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if fname in {"zeros", "empty", "ones"} and node.args and self.action_dim is None:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                self.action_dim = arg.value
            elif isinstance(arg, ast.Tuple) and arg.elts and isinstance(arg.elts[0], ast.Constant) \
                    and isinstance(arg.elts[0].value, int):
                self.action_dim = arg.elts[0].value
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        base = getattr(node.value, "id", "") or getattr(node.value, "attr", "")
        if base.lower() in _OBS_BASES and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            self.obs_keys.add(node.slice.value)
        self.generic_visit(node)


def static_parse_python(source: str, *, source_ref: str = "controller.py") -> PolicyImportSpec:
    """Tier P1: AST-parse a Python controller string (NO execution)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return PolicyImportSpec(
            id="policy_unparsable", source_ref=source_ref, framework=PolicyFramework.PYTHON_SCRIPT,
            tier=PolicyTier.P1_STATIC_PARSE, validation_status="rejected",
            warnings=[f"Python controller failed to parse: {exc.msg} (line {exc.lineno})."],
        )
    visitor = _ControllerVisitor()
    visitor.visit(tree)

    entrypoint = next((name for name in _ENTRYPOINT_NAMES if name in visitor.defs), None)
    if entrypoint is None and visitor.defs:
        entrypoint = visitor.defs[0]

    warnings: list[str] = []
    if entrypoint is None:
        warnings.append("No function/class entrypoint found in the controller.")
    if visitor.action_dim is None:
        warnings.append("Could not statically infer the action dimension; confirm before sandbox execution.")

    framework = PolicyFramework.PYTHON_SCRIPT
    if "torch" in visitor.imports:
        framework = PolicyFramework.TORCH
    elif "jax" in visitor.imports or "flax" in visitor.imports:
        framework = PolicyFramework.JAX
    elif "onnxruntime" in visitor.imports or "onnx" in visitor.imports:
        framework = PolicyFramework.ONNX
    elif "stable_baselines3" in visitor.imports:
        framework = PolicyFramework.STABLE_BASELINES

    return PolicyImportSpec(
        id="policy_static", source_ref=source_ref, framework=framework, tier=PolicyTier.P1_STATIC_PARSE,
        entrypoint=entrypoint, dependencies=sorted(visitor.imports), ros_topics=sorted(visitor.ros_topics),
        action_dim=visitor.action_dim, expected_inputs=sorted(visitor.obs_keys),
        validation_status="static_parsed", warnings=warnings,
    )


def check_action_dim(spec: PolicyImportSpec, actuated_joint_count: int) -> list[str]:
    """The plan's acceptance mismatch: "policy outputs N actions but the model has M actuated joints"."""
    if spec.action_dim is None:
        return ["Policy action dimension is unknown; cannot map to actuators yet."]
    if spec.action_dim != actuated_joint_count:
        return [
            f"Policy outputs {spec.action_dim} actions but the model has {actuated_joint_count} actuated joints; "
            "map or ignore the extra outputs (e.g. gripper mimic joints) before running."
        ]
    return []


def negotiate_observations(spec: PolicyImportSpec, policy_observation_keys: list[str]) -> list[str]:
    """Report policy-expected observation keys that the target observation contract does not provide."""
    provided = set(policy_observation_keys)
    return [
        f"Policy expects observation '{key}' which is not in the target observation contract."
        for key in spec.expected_inputs if key not in provided
    ]
