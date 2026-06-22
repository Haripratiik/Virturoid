import unittest
from unittest import mock

from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.services.llm_client import MockLLM
from virturoid.services.morphology_selector import select_morphology_template
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.species_tree import is_buildable, resolve_buildable
from virturoid.services.task_builder import build_task_graph


def _select(prompt):
    """Return (requested_template, catalog).

    select_morphology_template now resolves selected_template to the nearest *buildable*
    species, keeping the raw request in requested_template — so we trace from the request.
    """
    cat = morphology_template_catalog()
    req = build_requirements_from_prompt(prompt)
    task = build_task_graph(req)
    return select_morphology_template(req, task, cat).requested_template, cat


class SpeciesTreeResolveTests(unittest.TestCase):
    def test_manipulator_and_mobile_are_buildable_humanoid_is_not(self):
        by_class = {t.robot_class: t for t in morphology_template_catalog()}
        self.assertTrue(is_buildable(by_class["manipulator"]))
        self.assertTrue(is_buildable(by_class["mobile_base"]))
        self.assertFalse(is_buildable(by_class["humanoid"]))
        self.assertFalse(is_buildable(by_class["quadruped"]))

    def test_explicit_arm_request_is_exact(self):
        sel, cat = _select("build a tabletop arm that sorts red and blue blocks")
        res = resolve_buildable(sel, cat)
        self.assertEqual("manipulator", sel.robot_class)
        self.assertTrue(res.exact)

    def test_humanoid_arm_recognized_then_traced_to_nearest_buildable(self):
        # The key regression: "humanoid robot arm" must be recognized as humanoid,
        # not silently flattened into a tabletop arm.
        sel, cat = _select("build a humanoid robot arm")
        self.assertEqual("humanoid", sel.robot_class)
        res = resolve_buildable(sel, cat)
        self.assertFalse(res.exact)
        self.assertEqual("humanoid.upper_body.bimanual", res.requested_species)
        self.assertEqual("manipulator", res.template.robot_class)  # nearest buildable
        self.assertIn("nearest buildable", res.note)

    def test_quadruped_traced_honestly(self):
        sel, cat = _select("a quadruped dog robot that walks")
        self.assertEqual("quadruped", sel.robot_class)
        res = resolve_buildable(sel, cat)
        self.assertFalse(res.exact)
        self.assertTrue(is_buildable(res.template))

    def test_selector_always_returns_a_buildable_template(self):
        # The contract every caller (mvp, autonomous_builder, mobile_base) relies on:
        # selected_template is ALWAYS buildable, so dispatch can never hit a planned builder.
        cat = morphology_template_catalog()
        for prompt in ("build a humanoid robot arm", "a quadruped dog robot", "a mobile robot", "a tabletop sorting arm"):
            req = build_requirements_from_prompt(prompt)
            sel = select_morphology_template(req, build_task_graph(req), cat)
            self.assertTrue(is_buildable(sel.selected_template), f"{prompt!r} -> {sel.selected_template.id} not buildable")

    def test_humanoid_request_reports_requested_vs_built(self):
        cat = morphology_template_catalog()
        req = build_requirements_from_prompt("build a humanoid robot arm")
        sel = select_morphology_template(req, build_task_graph(req), cat)
        self.assertEqual("humanoid", sel.requested_robot_class)
        self.assertEqual("manipulator", sel.selected_template.robot_class)
        self.assertFalse(sel.species_exact)
        self.assertIn("nearest buildable", sel.species_note)


class SpeciesAgentTests(unittest.TestCase):
    def _ctx(self):
        nodes = [
            {"species_pattern": "fixed_arm.three_dof.tabletop", "robot_class": "manipulator",
             "parent_species": "fixed_arm", "morphology_tags": ["serial_chain"], "buildable": True},
        ]
        return build_requirements_from_prompt("build a humanoid robot arm"), nodes, ["fixed_arm", "humanoid", "mobile_robot"]

    def test_no_backend_returns_none(self):
        from virturoid.services.species_agent import classify_or_create_species

        req, nodes, roots = self._ctx()
        self.assertIsNone(classify_or_create_species("x", req, nodes, roots, None))

    def test_classify_into_existing(self):
        from virturoid.services.species_agent import classify_or_create_species

        req, nodes, roots = self._ctx()
        out = classify_or_create_species("a tabletop arm", req, nodes, roots, MockLLM(fixed={
            "decision": "existing", "species_pattern": "fixed_arm.three_dof.tabletop",
            "robot_class": "manipulator",
        }), max_repairs=0)
        self.assertTrue(out["valid"])
        self.assertEqual("existing", out["species"]["decision"])

    def test_create_new_species_with_valid_gene(self):
        from virturoid.services.species_agent import classify_or_create_species

        req, nodes, roots = self._ctx()
        out = classify_or_create_species("humanoid arm", req, nodes, roots, MockLLM(fixed={
            "decision": "new", "species_pattern": "humanoid.arm.single", "parent_species": "humanoid",
            "robot_class": "humanoid", "morphology_tags": ["humanoid", "single_arm"],
            "genes": {
                "links": ["torso", "shoulder", "upper", "forearm", "hand"],
                "joints": [
                    {"name": "j0", "joint_type": "revolute", "parent_link": "torso", "child_link": "shoulder"},
                    {"name": "j1", "joint_type": "revolute", "parent_link": "shoulder", "child_link": "upper"},
                    {"name": "j2", "joint_type": "revolute", "parent_link": "upper", "child_link": "forearm"},
                    {"name": "j3", "joint_type": "revolute", "parent_link": "forearm", "child_link": "hand"},
                ],
                "end_effectors": ["hand"],
            },
        }), max_repairs=0)
        self.assertTrue(out["valid"], out.get("issues"))
        self.assertEqual("humanoid.arm.single", out["species"]["species_pattern"])

    def test_unknown_parent_is_rejected(self):
        from virturoid.services.species_agent import classify_or_create_species

        req, nodes, roots = self._ctx()
        out = classify_or_create_species("x", req, nodes, roots, MockLLM(fixed={
            "decision": "new", "species_pattern": "weird.thing", "parent_species": "ghost_root",
            "robot_class": "manipulator", "morphology_tags": ["x"],
        }), max_repairs=0)
        self.assertFalse(out["valid"])
        self.assertTrue(any("parent" in i for i in out["issues"]))

    def test_bad_taxonomy_id_and_cyclic_gene_rejected(self):
        from virturoid.services.species_agent import classify_or_create_species

        req, nodes, roots = self._ctx()
        out = classify_or_create_species("x", req, nodes, roots, MockLLM(fixed={
            "decision": "new", "species_pattern": "Bad ID!", "parent_species": "humanoid",
            "robot_class": "humanoid", "morphology_tags": ["x"],
            "genes": {"links": ["a", "b"], "joints": [
                {"name": "j1", "joint_type": "revolute", "parent_link": "a", "child_link": "b"},
                {"name": "j2", "joint_type": "revolute", "parent_link": "b", "child_link": "a"},
            ], "end_effectors": ["b"]},
        }), max_repairs=0)
        self.assertFalse(out["valid"])


if __name__ == "__main__":
    unittest.main()
