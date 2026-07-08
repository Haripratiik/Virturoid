"""Failure-type classifier: rules (cold start) + learned refinement + the metrics->failure->repair loop.

The dossier's "small local robotics model" — CPU-only. Pure/offline (AGENTS.md).
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.failure_classifier import (  # noqa: E402
    FailureClassifier,
    repairs_for_metrics,
    rule_failure_label,
)

_SKLEARN = importlib.util.find_spec("sklearn") is not None


class RuleLabelTests(unittest.TestCase):
    def test_legged_labels(self):
        self.assertEqual(rule_failure_label({"survived": False, "height_ratio": 0.3, "forward": 0.0}), "fell_over")
        self.assertEqual(rule_failure_label({"survived": True, "height_ratio": 0.85, "forward": 0.01}),
                         "upright_but_slow")
        self.assertEqual(rule_failure_label({"survived": True, "height_ratio": 0.85, "forward": 0.5}), "success")

    def test_manipulation_labels(self):
        self.assertEqual(rule_failure_label({"contacts": 0, "lifted": 0.0}), "no_contact")
        self.assertEqual(rule_failure_label({"contacts": 2, "lifted": 0.01}), "gripped_no_lift")
        self.assertEqual(rule_failure_label({"contacts": 2, "lifted": 0.12, "success": False}),
                         "lifted_then_dropped")
        self.assertEqual(rule_failure_label({"contacts": 2, "lifted": 0.12, "success": True}), "success")


class RepairLoopTests(unittest.TestCase):
    def test_metrics_to_repairs(self):
        out = repairs_for_metrics({"survived": False, "height_ratio": 0.2})
        self.assertEqual(out["label"], "fell_over")
        self.assertIn("stability_prior", out["repairs"])         # closes metrics -> failure -> repair
        self.assertEqual(out["source"], "rule")

    def test_success_has_no_repairs(self):
        out = repairs_for_metrics({"survived": True, "height_ratio": 0.9, "forward": 0.6})
        self.assertEqual(out["label"], "success")
        self.assertEqual(out["repairs"], [])


class ClassifierFallbackTests(unittest.TestCase):
    def test_untrained_falls_back_to_rules(self):
        clf = FailureClassifier()
        self.assertFalse(clf.trained)
        pred = clf.predict({"survived": False, "height_ratio": 0.3})
        self.assertEqual(pred["source"], "rule")
        self.assertEqual(pred["label"], "fell_over")

    def test_single_class_data_stays_rule_only(self):
        clf = FailureClassifier().fit([{"metrics": {"survived": False}, "label": "fell_over"}] * 5)
        self.assertFalse(clf.trained)                            # <2 classes -> no learned model


@unittest.skipUnless(_SKLEARN, "needs sklearn for the learned classifier")
class LearnedClassifierTests(unittest.TestCase):
    def _dataset(self, n=240):
        import numpy as np
        rng = np.random.default_rng(0)
        examples = []
        for _ in range(n):
            kind = rng.integers(0, 6)
            if kind == 0:
                m = {"survived": False, "height_ratio": float(rng.uniform(0.1, 0.45)), "forward": 0.0}
            elif kind == 1:
                m = {"survived": True, "height_ratio": float(rng.uniform(0.6, 0.95)),
                     "forward": float(rng.uniform(0.0, 0.03))}
            elif kind == 2:
                m = {"survived": True, "height_ratio": float(rng.uniform(0.7, 0.95)),
                     "forward": float(rng.uniform(0.2, 0.8))}
            elif kind == 3:
                m = {"contacts": 0, "lifted": 0.0}
            elif kind == 4:
                m = {"contacts": 2, "lifted": float(rng.uniform(0.0, 0.03))}
            else:
                m = {"contacts": 2, "lifted": float(rng.uniform(0.08, 0.15)), "success": False}
            examples.append({"metrics": m, "label": rule_failure_label(m)})
        return examples

    def test_learns_to_label_from_metrics(self):
        data = self._dataset()
        train, test = data[:190], data[190:]
        clf = FailureClassifier().fit(train)
        self.assertTrue(clf.trained)
        correct = sum(clf.predict(e["metrics"])["label"] == e["label"] for e in test)
        acc = correct / len(test)
        self.assertGreater(acc, 0.85, f"learned classifier accuracy {acc:.2f} too low")
        self.assertEqual(clf.predict(test[0]["metrics"])["source"], "learned")


if __name__ == "__main__":
    unittest.main()
