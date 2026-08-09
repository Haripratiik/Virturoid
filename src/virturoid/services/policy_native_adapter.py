"""PolicyImporter Tier P3 — native ONNX adapter (Input Ingestion plan, Phase 4).

P2 sandboxes a black-box PYTHON controller; P3 loads a serialized policy through a KNOWN adapter and validates it
against the observation/action contract. ONNX is the safe first native format: an ``.onnx`` file is a computation
GRAPH executed by onnxruntime, not arbitrary Python (unlike a torch pickle, whose ``torch.load`` can run code) —
so an untrusted ONNX policy can be inspected and run without the OS-process isolation P2 needs. This adapter reads
the model's declared input/output tensors, runs one inference against a simulated observation, and validates the
action's dimension, finiteness, and safety limits — the same acceptance contract as P2, one tier up.

onnxruntime is optional: absent, the adapter reports that honestly rather than failing the import.

**WHAT "IMPORT" DOES NOT MEAN HERE, said out loud.** This is INSPECT + VALIDATE. It is not deploy. Nothing in
the repo can make an imported ONNX policy a held robot's deployed controller, and until 2026-08-09 nothing said
so: the tool never claimed it, but the word "import" did, ``verify_robot``'s own ``to_get_a_real_verdict`` told
the customer to "import_onnx_policy ... then verify_robot again; the verdict then describes YOUR controller on
YOUR body" — which was false — and "bring your own trained policy" is a stated priority use case. See
:data:`DEPLOYMENT` for the disclosure every result now carries, and the reasoning for why deploying blind would
be worse than declining.
"""

from __future__ import annotations

import math

from virturoid.schemas.policy_import import PolicyFramework, PolicyTier
from virturoid.services.policy_sandbox import _within_limits

#: THE HONEST ANSWER TO "can you run my policy on my robot?", attached to every P3 result.
#:
#: WHY WE DECLINE TO DEPLOY rather than building the path. Our deployed-controller channel is the policy bank
#: (``policy_flywheel.recall_morph_policy``), which holds ``MorphPolicy`` ``.npz`` — attention over per-limb
#: morphology tokens whose feature layout WE define (``morph_graph.encode_robot``). An ONNX file declares tensor
#: NAMES, SHAPES and DTYPES and nothing else. To close the loop on the customer's policy we would have to know,
#: and cannot read from the file: which observation index is base linear velocity vs angular velocity vs
#: projected gravity vs the velocity command; the JOINT ORDER behind the joint-position and joint-velocity
#: blocks; whether the previous action is fed back and where; whether the outputs are position targets, position
#: DELTAS about a default pose, or torques; the action scale; and the control decimation. Guessing any one of
#: those produces a rollout that is not the customer's policy, reported under their robot's name — a
#: verdict-shaped lie, which is the exact defect class this disclosure exists to close. A 12-actuator quadruped
#: has 479,001,600 joint orderings; the file pins down none of them.
#:
#: WHAT WOULD MAKE IT BUILDABLE (and is not built): an EXPLICIT, customer-declared observation/action contract.
#: ``import_controller_interface`` already recovers half of it — joint order, command interfaces, limits and
#: control rate — from a ``<ros2_control>`` block. The observation layout has no such source and would have to be
#: stated. That is a real feature; it is not a text change, and shipping it half-done would resurrect the claim
#: this replaces.
DEPLOYMENT: dict = {
    "deployable": False,
    "what_this_tool_does": "inspects the IO contract and VALIDATES one inference (dimension, finiteness, "
                           "safety limits). It does not run your policy in our simulator and does not make it "
                           "your robot's controller.",
    "verify_robot_still_measures": "our generic scripted controller on your body — importing here does not "
                                   "change what verify_robot deploys or what its verdict is about",
    "why_not": "an .onnx file declares tensor names, shapes and dtypes only. Deploying it closed-loop needs the "
               "observation LAYOUT (which index is base angular velocity vs projected gravity vs the command), "
               "the JOINT ORDER, whether the outputs are torques or position targets about a default pose, the "
               "action scale and the control decimation. None of that is in the file, and a wrong guess yields a "
               "rollout that is not your policy reported under your robot's name.",
    "instead": [
        {"if": "you want a controller that DOES deploy on this body",
         "do": "train_held with mode='gpu_rl', or train_reward with train_backend='gpu' — MJX PPO produces a "
               "native MorphPolicy .npz that is re-rolled on your body and banked, after which verify_robot "
               "deploys it and reports gait_source 'learned_policy'"},
        {"if": "your controller is parameterised (gains, gait scalars) rather than a network",
         "do": "adopt_control_script {robot_id, params | params_path} — that IS the deploying door: it runs your "
               "parameters on your body, fits an improvement warm-started from them, and commits it (undoable)"},
        {"if": "you need the joint order / command interfaces your policy assumes",
         "do": "import_controller_interface (a <ros2_control> URDF/xacro) — the contract half we can read"},
        {"if": "you want to keep running the policy in YOUR stack",
         "do": "export_held / export_isaac ships the body; the policy stays yours and runs where it already runs"},
    ],
}


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
        return {"available": False, "warnings": ["onnxruntime not installed"], "deployment": dict(DEPLOYMENT)}
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "ok": False, "warnings": [f"could not load ONNX model: {exc}"],
                "deployment": dict(DEPLOYMENT)}
    inputs = [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()]
    outputs = [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()]
    return {"available": True, "ok": True, "inputs": inputs, "outputs": outputs, "warnings": [],
            "deployment": dict(DEPLOYMENT)}


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
        # ``native_validated`` below means "this one action is well-formed", NOT "this policy is deployed".
        # The distinction rides on every result because the tool's NAME implies the stronger claim.
        "deployment": dict(DEPLOYMENT),
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
        result["reason"] = ("one-step ONNX action is dimension-correct, finite, and within limits. That is a "
                            "check on ONE inference, not a deployment: see `deployment` for what this does and "
                            "does not put on your robot.")
    return result
