"""Graph-network fitness surrogate (plan Phase C, the real GNN — runs on the GPU).

This is the upgrade from the dependency-free ridge surrogate (``gene_surrogate.py``) to the
graph-network model the architecture plan calls for: the gene's kinematic tree IS the graph
(segments = nodes, joints = edges), and a small message-passing GNN predicts task success.
It trains on real (gene -> measured MuJoCo success) rows and is used as a fast inner-loop
SCREEN to rank candidate genes before expensive ground-truth sim — never replacing MuJoCo.

PyTorch only (no torch_geometric dep): message passing is hand-rolled over per-gene adjacency,
which keeps small kinematic graphs cheap and portable. CUDA is used when available (the RTX
3060); it falls back to CPU for tests. Trained model + feature scaler are saved together.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.gene import RobotGene

# Per-node (segment) features: 9 kinematic + 4 TopoPE. Kept explicit so the model is inspectable.
_NODE_DIM = 13


def gene_graph(gene: RobotGene):
    """(node_features [N,_NODE_DIM], edge_index [2,E] parent<->child) for a gene.

    Node features = 9 kinematic (length / radius / mass / actuated / torque / joint-axis / is-EE) + 4 TopoPE
    (Topology Positional Encoding: the edit-invariant root->node path-hash, ``topo_pe.topo_path_vector``). TopoPE
    lets the message-passing GNN tell structurally-different bodies with identical link params apart (the GNN
    symmetry blind spot; BodyGen arXiv:2503.00533) — so an unseen morphology gets a STRUCTURALLY-sensible latent
    instead of an arbitrary one (the fix for the OOD z_body gap), aligned across bodies for transfer.
    """
    import torch

    from virturoid.services.topo_pe import topo_path_vector

    idx = {s.name: i for i, s in enumerate(gene.segments)}
    by_name = {s.name: s for s in gene.segments}
    # Edit-invariant TopoPE (BodyGen path-hash), UNIFIED with the tokenizer via topo_pe.topo_path_vector:
    # each node's positional code depends only on its root->node child-index path, so adding a limb elsewhere
    # doesn't renumber existing nodes and structurally-similar limbs across different robots get aligned codes
    # (replaces the fragile depth-based PE) — the property that lets the design latent transfer across bodies.
    kids_order: dict = {}
    for s in gene.segments:
        kids_order.setdefault(s.parent, []).append(s.name)

    def _topo_path(seg) -> list:
        chain, cur, guard = [], seg, 0
        while cur.parent is not None and cur.parent in by_name and guard <= len(gene.segments):
            sibs = kids_order.get(cur.parent, [])
            chain.append(sibs.index(cur.name) if cur.name in sibs else 0)
            cur = by_name[cur.parent]
            guard += 1
        chain.reverse()
        return chain

    feats = []
    for s in gene.segments:
        act = 1.0 if s.joint_type in ("revolute", "prismatic") else 0.0
        ax = s.joint_axis if act else (0.0, 0.0, 0.0)
        pe = topo_path_vector(_topo_path(s), dim=4)
        feats.append([
            s.length_m, s.radius_m, s.mass_kg, act,
            (s.actuator_torque_nm or 0.0) / 50.0,   # scaled
            float(ax[0]), float(ax[1]), float(ax[2]),
            1.0 if s.is_end_effector else 0.0,
            pe[0], pe[1], pe[2], pe[3],             # TopoPE (edit-invariant path-hash)
        ])
    src, dst = [], []
    for s in gene.segments:
        if s.parent is not None and s.parent in idx:
            a, b = idx[s.parent], idx[s.name]
            src += [a, b]      # undirected message passing
            dst += [b, a]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(feats, dtype=torch.float32), edge_index


class GeneGNN:
    """Small 2-layer message-passing GNN that predicts pick-place success in [0,1]."""

    def __init__(self, hidden: int = 32, device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden = hidden
        self._build()

    def _build(self):
        import torch.nn as nn

        h = self.hidden
        self.enc = nn.Linear(_NODE_DIM, h)
        self.msg1 = nn.Linear(h, h)
        self.msg2 = nn.Linear(h, h)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))  # 2h: mean|max pool
        # graph-autoencoder decoder: reconstruct node features from per-node latents -> the objective that
        # shapes the pooled latent into a reusable DESIGN manifold (GLSO), not just a success scalar.
        self.dec = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, _NODE_DIM))
        self.module = nn.ModuleList([self.enc, self.msg1, self.msg2, self.head, self.dec]).to(self.device)

    def _message_pass(self, x, edge_index):
        """Message-pass -> per-NODE hidden states h [N, hidden] (pre-pool)."""
        import torch
        import torch.nn.functional as F

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        h = F.relu(self.enc(x))
        for msg in (self.msg1, self.msg2):
            if edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                agg = torch.zeros_like(h).index_add_(0, dst, h[src])
                deg = torch.zeros(h.shape[0], device=self.device).index_add_(
                    0, dst, torch.ones(dst.shape[0], device=self.device)).clamp(min=1).unsqueeze(1)
                h = F.relu(h + msg(agg / deg))
            else:
                h = F.relu(h + msg(h))
        return h

    def _encode_one(self, x, edge_index):
        """Message-pass + mean|max graph pooling -> the pooled graph latent (``z_body``), pre-head.

        Concatenating mean AND max over the node hidden states keeps COARSE structure that mean-pooling
        alone washes out: e.g. max over the TopoPE child-count signal separates a BRANCHED quad (torso has
        4 children) from a SERIAL arm (max 1 child) — the ``arm-near-quad`` failure of pure mean-pooling."""
        import torch

        h = self._message_pass(x, edge_index)
        return torch.cat([h.mean(dim=0), h.max(dim=0).values])

    def _forward_one(self, x, edge_index):
        import torch

        return torch.sigmoid(self.head(self._encode_one(x, edge_index)))  # -> success

    def _recon_one(self, x, edge_index):
        """Reconstruct node features from per-node latents (the graph-autoencoder objective)."""
        return self.dec(self._message_pass(x, edge_index))

    def embed(self, gene: RobotGene) -> list[float]:
        """The pooled graph latent for a gene — the learned ``z_body`` for the robotics vector memory
        (``robotics_vector_memory.embed_body(gene, latent=gnn.embed(gene))``).

        This is the seam the embeddings research calls for: the encoder trunk already exists here, it
        just fed only the success head. Untrained weights give a deterministic random projection; after
        ``fit()``/``load()`` it is the morphology representation the success head is trained to read.
        """
        import torch

        self.module.eval()
        with torch.no_grad():
            x, ei = gene_graph(gene)
            return self._encode_one(x, ei).cpu().tolist()

    def predict(self, gene: RobotGene) -> float:
        import torch

        self.module.eval()
        with torch.no_grad():
            x, ei = gene_graph(gene)
            return float(self._forward_one(x, ei).item())

    def rank(self, genes: list[RobotGene]) -> list[tuple[RobotGene, float]]:
        scored = [(g, self.predict(g)) for g in genes]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def fit(self, genes: list[RobotGene], successes: list[float], *, epochs: int = 200, lr: float = 1e-2,
            recon_weight: float = 0.3):
        """Multi-task fit: success-prediction MSE + ``recon_weight`` * node-reconstruction MSE (the
        autoencoder term that makes the latent a reusable design manifold). ``final_loss``/``first_loss``
        report the SUCCESS loss (the ranking objective); ``recon_loss`` is the auxiliary reconstruction."""
        import torch

        graphs = [gene_graph(g) for g in genes]
        y = torch.tensor(successes, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.module.parameters(), lr=lr)
        lossfn = torch.nn.MSELoss()
        self.module.train()
        history = []
        recon_last = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            preds = torch.stack([self._forward_one(x, ei).squeeze() for x, ei in graphs])
            s_loss = lossfn(preds, y)
            r_loss = torch.stack([lossfn(self._recon_one(x, ei), x.to(self.device)) for x, ei in graphs]).mean()
            (s_loss + recon_weight * r_loss).backward()
            opt.step()
            history.append(float(s_loss.item()))
            recon_last = float(r_loss.item())
        return {"final_loss": history[-1], "first_loss": history[0], "recon_loss": recon_last,
                "device": self.device, "epochs": epochs}

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.module.state_dict(), "hidden": self.hidden}, str(path))

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "GeneGNN":
        import torch

        blob = torch.load(str(path), map_location=device or "cpu")
        m = cls(hidden=int(blob["hidden"]), device=device)
        m.module.load_state_dict(blob["state"])
        return m


def train_from_rows_jsonl(path: Path, *, epochs: int = 300) -> tuple[GeneGNN, dict]:
    """Train the GNN from a JSONL of {gene:{...}, success_rate:float} (the flywheel/data-gen)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    genes = [RobotGene.from_dict(r["gene"]) for r in rows]
    ys = [float(r["success_rate"]) for r in rows]
    model = GeneGNN()
    metrics = model.fit(genes, ys, epochs=epochs)
    return model, metrics
