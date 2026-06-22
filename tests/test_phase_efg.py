"""Fast (no-MuJoCo) tests for the Phase E (distillation) and Phase F (residual) interfaces."""

import json
import unittest

from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.services.proposer_distill import (
    build_sft_dataset,
    gene_to_target,
    sft_example,
)
from virturoid.services.residual_physics import (
    IdentityResidual,
    NeuralOperatorResidual,
    PINNResidual,
    get_residual_model,
)


class ProposerDistillTests(unittest.TestCase):
    def test_gene_target_is_schema_faithful_json(self):
        t = gene_to_target(tabletop_arm_gene())
        self.assertEqual("manipulator", t["robot_class"])
        self.assertTrue(any(s.get("is_end_effector") for s in t["segments"]))
        json.dumps(t)  # serializable

    def test_sft_example_has_chat_roles(self):
        ex = sft_example("build a sorting arm", tabletop_arm_gene())
        roles = [m["role"] for m in ex["messages"]]
        self.assertEqual(["system", "user", "assistant"], roles)
        json.loads(ex["messages"][2]["content"])  # assistant target is valid JSON

    def test_build_dataset_filters_by_success(self):
        pairs = [("p1", tabletop_arm_gene()), ("p2", tabletop_arm_gene())]
        examples = build_sft_dataset(pairs, min_success=0.8, successes=[0.95, 0.4])
        self.assertEqual(1, len(examples))  # only the high-success build is kept


class ResidualPhysicsTests(unittest.TestCase):
    def test_rigid_body_uses_identity_no_correction(self):
        m = get_residual_model("rigid_body")
        self.assertIsInstance(m, IdentityResidual)
        self.assertEqual([1, 2, 3], m.correct(None, [1, 2, 3]))  # MuJoCo prediction unchanged

    def test_soft_and_field_domains_are_deferred_not_faked(self):
        self.assertIsInstance(get_residual_model("soft_body"), PINNResidual)
        self.assertIsInstance(get_residual_model("fluid"), NeuralOperatorResidual)
        with self.assertRaises(NotImplementedError):
            get_residual_model("soft_body").correct(None, None)
        with self.assertRaises(NotImplementedError):
            get_residual_model("aero").correct(None, None)

    def test_unknown_domain_defaults_to_identity(self):
        self.assertIsInstance(get_residual_model("whatever"), IdentityResidual)


if __name__ == "__main__":
    unittest.main()
