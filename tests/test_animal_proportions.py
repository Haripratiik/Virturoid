"""P4 — distinct animals must compose to STRUCTURALLY distinct bodies (was: horse==gecko==turtle==generic quad,
embedding cosine 1.0). The proportion priors drive leg length / stance / mass, while generic/dog stays the pinned
baseline. Word-boundary matching so 'ox' never hits 'box'."""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class AnimalProportionTests(unittest.TestCase):
    def test_table_is_word_boundary_and_defaults_to_unit(self):
        from virturoid.services.animal_proportions import animal_proportions
        self.assertEqual(animal_proportions("a quadruped robot"), {"leg": 1.0, "stance": 1.0, "torso": 1.0, "mass": 1.0})
        self.assertEqual(animal_proportions("a robot dog"), {"leg": 1.0, "stance": 1.0, "torso": 1.0, "mass": 1.0})
        self.assertGreater(animal_proportions("a horse robot")["leg"], 1.2)      # long-legged
        self.assertLess(animal_proportions("a gecko robot")["leg"], 0.8)         # short-legged
        # word boundary: a box-carrying robot must NOT trigger 'ox'
        self.assertEqual(animal_proportions("a box carrying robot"), {"leg": 1.0, "stance": 1.0, "torso": 1.0, "mass": 1.0})

    def test_composed_animals_differ_in_leg_length_but_dog_is_baseline(self):
        from virturoid.services.animal_proportions import animal_proportions
        from virturoid.services.morphology_composer import compose_robot

        def leg_len(prompt):                                     # sum any leg segment (both builder paths)
            g = compose_robot(prompt)
            return sum(s.length_m for s in g.segments if "leg" in s.name.lower())

        dog, horse, gecko, giraffe = (leg_len(f"a {a} robot") for a in ("dog", "horse", "gecko", "giraffe"))
        self.assertGreater(dog, 0.0)
        self.assertGreater(horse, dog * 1.2)                    # horse legs materially longer than the dog baseline
        self.assertGreater(giraffe, horse)                     # giraffe the longest
        self.assertLess(gecko, dog * 0.8)                      # gecko materially shorter
        # dog carries NO proportion prior (all-1.0) -> the gait-pinned baseline body is untouched
        self.assertEqual(animal_proportions("a dog robot"), {"leg": 1.0, "stance": 1.0, "torso": 1.0, "mass": 1.0})

    def test_embedding_separates_animals_that_were_identical(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.morphology_embedding import cosine_similarity, embed_gene
        horse, gecko = embed_gene(compose_robot("a horse robot")), embed_gene(compose_robot("a gecko robot"))
        self.assertLess(cosine_similarity(horse, gecko), 0.999)  # no longer the identical 1.0 point


if __name__ == "__main__":
    unittest.main()
