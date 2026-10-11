"""Failure-type classifier — the dossier's "small local robotics model" that closes the metrics->failure->repair loop.

``failure_curriculum`` maps a failure LABEL to concrete repairs, but something has to PRODUCE the label from raw
episode metrics. This module does that two ways, honestly:

  * :func:`rule_failure_label` — a deterministic labeler from metrics (survived / height / contacts / lift /
    forward), immediately useful at cold start with zero training data;
  * :class:`FailureClassifier` — a small sklearn model (Bet: "train small local robotics models, not a giant
    LLM") that LEARNS to label from accumulated episodes and refines the rules as the corpus grows.

The predicted label feeds straight into ``failure_curriculum.curriculum_updates_for_failure``. sklearn is
optional: absent, the learned path degrades to the rules. CPU-only, no GPU.
"""

from __future__ import annotations

# fixed feature order extracted from an episode-metrics dict (missing keys -> 0.0).
FEATURE_KEYS = ("forward", "height_ratio", "survived", "cadence", "contacts", "lifted", "d_tcp", "success")

# the non-success failure labels this classifier emits (a subset of failure_curriculum's keys).
LABELS = ("fell_over", "upright_but_slow", "no_contact", "gripped_no_lift", "lifted_then_dropped", "success")


#: The same quantities under the names other tools in this product report them. ``verify_robot`` says
#: ``forward_m``, and a classifier that read only ``forward`` scored a verify result as no travel at all: MEASURED
#: 2026-10-11 through the MCP server, a 10.5 m CREDIBLE WALK passed straight from verify_robot came back
#: "upright_but_slow" at confidence 1.0, with a speed curriculum prescribed for it.
_ALIASES = {"forward_m": "forward"}


def _canonical(metrics: dict) -> dict:
    """``metrics`` with every alias in :data:`_ALIASES` filled in under the name this module reads."""
    out = dict(metrics or {})
    for alias, key in _ALIASES.items():
        if key not in out and alias in out:
            out[key] = out[alias]
    return out


def features_from_metrics(metrics: dict) -> list[float]:
    """Extract the fixed FEATURE_KEYS vector from an episode-metrics dict (booleans -> 0/1, missing -> 0.0)."""
    metrics = _canonical(metrics)
    out: list[float] = []
    for key in FEATURE_KEYS:
        val = metrics.get(key, 0.0)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        out.append(float(val) if isinstance(val, (int, float)) else 0.0)
    return out


def rule_failure_label(metrics: dict, *, domain: str = "auto", forward_gate: float = 0.05,
                       lift_gate: float = 0.05) -> str:
    """Deterministic failure label from metrics — the cold-start labeler (no training data needed).

    ``domain`` 'legged'|'manipulator'|'auto'. In 'auto' the presence of ``contacts``/``lifted`` implies a
    manipulation episode. Returns one of :data:`LABELS` ('success' when no failure mode is detected).
    """
    metrics = _canonical(metrics)
    survived = bool(metrics.get("survived", True))
    hr = float(metrics.get("height_ratio", 1.0) or 1.0)
    forward = float(metrics.get("forward", 0.0) or 0.0)
    is_manip = domain == "manipulator" or (domain == "auto" and ("contacts" in metrics or "lifted" in metrics))

    if is_manip:
        contacts = float(metrics.get("contacts", 0) or 0)
        lifted = float(metrics.get("lifted", 0.0) or 0.0)
        if bool(metrics.get("success")):
            return "success"
        if contacts < 2:
            return "no_contact"
        if lifted <= lift_gate:
            return "gripped_no_lift"
        return "lifted_then_dropped"

    # legged / locomotion. The product's OWN verdict, when the caller passes it, outranks these thresholds:
    # ``gait_quality.classify`` judged the whole rollout, and a label that contradicted it would be a second opinion
    # from less evidence.
    verdict = str(metrics.get("verdict") or "")
    if verdict.startswith("CREDIBLE"):
        return "success"
    if verdict.startswith("FELL"):
        return "fell_over"
    if not survived or hr < 0.5:
        return "fell_over"
    if hr >= 0.6 and forward < forward_gate:
        return "upright_but_slow"
    return "success"


class FailureClassifier:
    """Small sklearn classifier over metric features. Falls back to the rule labeler when sklearn is absent."""

    def __init__(self, model=None, labels: tuple[str, ...] = LABELS):
        self._model = model
        self.labels = labels
        self.trained = model is not None

    def fit(self, examples: list[dict]) -> "FailureClassifier":
        """Fit on ``[{"metrics": {...}, "label": str}, ...]``. Needs >=2 classes; else stays rule-only."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            self._model = None
            self.trained = False
            return self
        X = [features_from_metrics(e["metrics"]) for e in examples]
        y = [e["label"] for e in examples]
        if len(set(y)) < 2:
            self._model = None
            self.trained = False
            return self
        clf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=0)
        clf.fit(X, y)
        self._model = clf
        self.trained = True
        return self

    def predict(self, metrics: dict, *, domain: str = "auto") -> dict:
        """Return ``{label, confidence, source}``. Uses the learned model if trained, else the rules."""
        if self._model is not None:
            import numpy as np
            x = [features_from_metrics(metrics)]
            proba = self._model.predict_proba(x)[0]
            idx = int(np.argmax(proba))
            return {"label": str(self._model.classes_[idx]), "confidence": round(float(proba[idx]), 4),
                    "source": "learned"}
        return {"label": rule_failure_label(metrics, domain=domain), "confidence": 1.0, "source": "rule"}


def repairs_for_metrics(metrics: dict, *, classifier: FailureClassifier | None = None, domain: str = "auto") -> dict:
    """End-to-end: metrics -> failure label -> concrete curriculum repairs (closes the R7 loop)."""
    from virturoid.services.failure_curriculum import curriculum_updates_for_failure
    pred = classifier.predict(metrics, domain=domain) if classifier is not None else {
        "label": rule_failure_label(metrics, domain=domain), "confidence": 1.0, "source": "rule"}
    label = pred["label"]
    repairs = [] if label == "success" else curriculum_updates_for_failure(label)
    return {"label": label, "confidence": pred["confidence"], "source": pred["source"], "repairs": repairs}
