import importlib.util
import unittest

_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_TORCH, "torch not installed.")
class GeneGNNTests(unittest.TestCase):
    def test_gnn_learns_to_rank_genes_by_success(self):
        import dataclasses

        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.gene_surrogate_nn import GeneGNN, gene_graph

        base = tabletop_arm_gene()
        # Synthetic flywheel: success rises with forearm length (a learnable graph signal).
        genes, ys = [], []
        for length in (0.20, 0.28, 0.357, 0.42, 0.48):
            g = dataclasses.replace(base, id=f"v{length}", segments=[
                dataclasses.replace(s, length_m=length) if s.name == "forearm" else s for s in base.segments])
            genes.append(g)
            ys.append(min(1.0, max(0.0, (length - 0.20) / 0.28)))

        model = GeneGNN(hidden=24, device="cpu")
        metrics = model.fit(genes, ys, epochs=300)
        self.assertLess(metrics["final_loss"], metrics["first_loss"])   # it actually learned
        self.assertLess(metrics["final_loss"], 0.05)                    # fits the small set
        # Learned the monotonic trend: longer forearm -> higher predicted success.
        self.assertGreater(model.predict(genes[-1]), model.predict(genes[0]))

    def test_graph_encodes_topology(self):
        from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
        from virturoid.services.gene_surrogate_nn import gene_graph

        xa, ea = gene_graph(tabletop_arm_gene())
        xh, eh = gene_graph(humanoid_upper_body_gene())
        self.assertEqual(5, xa.shape[0])          # arm: 5 segments
        self.assertEqual(6, xh.shape[0])          # humanoid: 6 segments
        self.assertGreater(eh.shape[1], ea.shape[1])  # humanoid has more edges

    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.gene_surrogate_nn import GeneGNN

        g = tabletop_arm_gene()
        m = GeneGNN(hidden=16, device="cpu")
        m.fit([g], [0.7], epochs=20)
        before = m.predict(g)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gnn.pt"
            m.save(p)
            m2 = GeneGNN.load(p, device="cpu")
        self.assertAlmostEqual(before, m2.predict(g), places=5)


if __name__ == "__main__":
    unittest.main()
