"""Learned physics surrogate (plan Phase C): the "our own AI" custom asset.

Per docs/ai_architecture_plan.md, the defensible learned-physics asset is a model trained
on OUR MuJoCo rollouts that *screens* candidate genes cheaply — ranking them before the
expensive ground-truth sim, never replacing it. This is the offline-trainable, dependency-
free core of that: a ridge-regression fitness surrogate over morphology features, trained on
real (gene → measured success) outcomes from the flywheel. It plugs into the co-design inner
loop as a fast filter.

The architecture plan's full version is a graph-network dynamics simulator (GNS/MeshGraphNets)
that ingests the gene graph and predicts trajectories; that needs GPU-scale training data and
is the next step. The interface here (features -> predicted success, used to rank/screen) is
the same contract, so the GNN drops in behind it later. Trained on the same rollouts; always
gated against MuJoCo ground truth.
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene

# Feature names kept explicit so the surrogate is inspectable and the GNN can extend them.
FEATURE_NAMES = ["num_segments", "num_dof", "total_reach_m", "total_mass_kg",
                 "mean_torque_nm", "max_link_m", "base_height"]
_BASE_Z = {"table": 0.025, "floor": 0.0, "torso": 0.0}


def gene_features(gene: RobotGene) -> list[float]:
    """A fixed-length morphology feature vector for the surrogate (and the GNN later)."""
    acts = gene.actuated_joints()
    torques = [s.actuator_torque_nm or 0.0 for s in acts]
    lengths = [s.length_m for s in gene.segments]
    chain = [s.length_m for s in gene.segments if s.parent is not None]
    return [
        float(len(gene.segments)),
        float(len(acts)),
        round(sum(chain), 4),                                   # rough reach
        round(sum(s.mass_kg for s in gene.segments), 4),
        round(sum(torques) / len(torques), 4) if torques else 0.0,
        round(max(lengths), 4) if lengths else 0.0,
        _BASE_Z.get(gene.base_mount, 0.025),
    ]


class GeneFitnessSurrogate:
    """Ridge-regression success predictor over gene features. Numpy-only, no heavy deps."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = l2
        self._w = None
        self._mu = None
        self._sd = None
        self.trained = False

    def fit(self, X: list[list[float]], y: list[float]) -> "GeneFitnessSurrogate":
        import numpy as np

        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        self._mu = Xa.mean(axis=0)
        self._sd = Xa.std(axis=0) + 1e-8
        Xn = (Xa - self._mu) / self._sd
        Xb = np.hstack([Xn, np.ones((Xn.shape[0], 1))])         # bias column
        d = Xb.shape[1]
        reg = self.l2 * np.eye(d)
        reg[-1, -1] = 0.0                                       # don't regularize bias
        self._w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ ya)
        self.trained = True
        return self

    def predict(self, gene_or_features) -> float:
        import numpy as np

        if self._w is None:
            raise RuntimeError("surrogate not trained")
        feats = gene_features(gene_or_features) if isinstance(gene_or_features, RobotGene) else list(gene_or_features)
        xn = (np.asarray(feats, dtype=float) - self._mu) / self._sd
        return float(np.dot(np.append(xn, 1.0), self._w))

    def rank(self, genes: list[RobotGene]) -> list[tuple[RobotGene, float]]:
        """Cheap screen: rank candidate genes by predicted success (high to low)."""
        scored = [(g, self.predict(g)) for g in genes]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


def train_from_rows(rows: list[dict]) -> GeneFitnessSurrogate:
    """Train the surrogate from flywheel rows of {features:[...], success_rate:float}."""
    X = [r["features"] for r in rows]
    y = [float(r["success_rate"]) for r in rows]
    return GeneFitnessSurrogate().fit(X, y)


def train_from_genes(genes: list[RobotGene], scenes) -> tuple[GeneFitnessSurrogate, list[dict]]:
    """Evaluate each gene in REAL MuJoCo, then train the surrogate on (features → success).

    This is the genuine flywheel step — labels come from the ground-truth simulator. Returns
    the trained surrogate and the rows it learned from (which can be persisted to memory).
    """
    from virturoid.services.gene_build import evaluate_gene_pick_place

    rows = []
    for g in genes:
        r = evaluate_gene_pick_place(g, scenes)
        rows.append({"gene_id": g.id, "species": g.species,
                     "features": gene_features(g), "success_rate": r["success_rate"]})
    return train_from_rows(rows), rows
