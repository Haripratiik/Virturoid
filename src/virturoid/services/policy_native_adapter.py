"""PolicyImporter Tier P3 — native ONNX adapter (Input Ingestion plan, Phase 4).

P2 sandboxes a black-box PYTHON controller; P3 loads a serialized policy through a KNOWN adapter and validates it
against the observation/action contract. ONNX is the safe first native format: an ``.onnx`` file is a computation
GRAPH executed by onnxruntime, not arbitrary Python (unlike a torch pickle, whose ``torch.load`` can run code) —
so an untrusted ONNX policy can be inspected and run without the OS-process isolation P2 needs. This adapter reads
the model's declared input/output tensors, runs one inference against a simulated observation, and validates the
action's dimension, finiteness, and safety limits — the same acceptance contract as P2, one tier up.

onnxruntime is optional: absent, the adapter reports that honestly rather than failing the import.
"""

from __future__ import annotations

import math

from virturoid.schemas.policy_import import PolicyFramework, PolicyTier
from virturoid.services.policy_sandbox import _within_limits


def _onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        return None


def inspect_onnx(path: str) -> dict:
    """Read an ONNX model's declared inputs/outputs (names, shapes, dtypes) without running it."""
    ort = _onnxruntime()
    if ort is None:
        return {"available": False, "warnings": ["onnxruntime not installed"]}
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "ok": False, "warnings": [f"could not load ONNX model: {exc}"]}
    inputs = [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()]
    outputs = [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()]
    return {"available": True, "ok": True, "inputs": inputs, "outputs": outputs, "warnings": []}


def _target_shape(decl_shape, obs_len: int) -> list[int]:
    """Resolve an ONNX input shape (with None/str dynamic dims) to concrete ints for a flat observation."""
    shape = []
    for dim in decl_shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(1)                                    # dynamic/batch dim -> 1
    # if the declared shape has a single "feature" slot, put the whole observation there.
    prod_fixed = 1
    for d in shape:
        prod_fixed *= d
    if prod_fixed != obs_len and obs_len % max(1, prod_fixed // (shape[-1] or 1)) == 0:
        shape[-1] = obs_len // max(1, prod_fixed // (shape[-1] or 1))
    return shape


def run_onnx_policy(path: str, observation, *, action_dim: int | None = None,
                    safety_limits: dict | None = None) -> dict:
    """Load an ONNX policy and run ONE inference against ``observation``; validate the action (P3).

    Returns the same acceptance report shape as the P2 sandbox: ``{ran, action, action_len, action_dim_ok,
    finite, within_limits, validation_status, warnings, reason, tier, framework}``. Fail-closed: any load/shape/
    inference error or a failed check yields ``validation_status='rejected'``.
    """
    result = {
        "ran": False, "action": None, "action_len": 0, "action_dim_ok": None, "finite": None,
        "within_limits": None, "validation_status": "rejected", "warnings": [], "reason": "",
        "tier": PolicyTier.P3_NATIVE_ADAPTER.value, "framework": PolicyFramework.ONNX.value,
    }
    ort = _onnxruntime()
    if ort is None:
        result["reason"] = "onnxruntime not installed; cannot run the ONNX policy."
        return result
    try:
        import numpy as np
    except ImportError:
        result["reason"] = "numpy is required for the ONNX adapter."
        return result

    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"could not load ONNX model: {exc}"
        return result

    inp = sess.get_inputs()[0]
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    dtype = np.float32 if "float" in (inp.type or "float") else np.float32
    try:
        target = _target_shape(inp.shape, int(obs.size))
        feed = {inp.name: obs.astype(dtype).reshape(target)}
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"observation shape {obs.shape} does not fit model input {inp.shape}: {exc}"
        return result

    try:
        outputs = sess.run(None, feed)
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"ONNX inference failed: {exc}"
        return result

    action = np.asarray(outputs[0], dtype=np.float64).reshape(-1).tolist()
    result["ran"] = True
    result["action"] = action
    result["action_len"] = len(action)
    result["finite"] = all(math.isfinite(x) for x in action)
    result["action_dim_ok"] = (action_dim is None) or (len(action) == action_dim)
    within, warns = _within_limits(action, safety_limits)
    result["within_limits"] = within
    result["warnings"].extend(warns)

    if not result["finite"]:
        result["reason"] = "action contains non-finite values."
    elif action_dim is not None and not result["action_dim_ok"]:
        result["reason"] = f"policy returned {len(action)} actions but the model has {action_dim} actuators."
    elif not within:
        result["reason"] = "action violates the declared safety limits."
    else:
        result["validation_status"] = "native_validated"
        result["reason"] = "one-step ONNX action is dimension-correct, finite, and within limits."
    return result
