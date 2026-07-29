"""Robotics vector memory (Pillar 2 core): ONE retrieval index for bodies, skills, episodes, tasks.

This is the vector-store upgrade of the retrieval seams the memory layer named for itself:
``memory_db.similar_runs`` ("swap the scoring for embeddings/pgvector later") and
``species_discovery.nearest_species`` (linear-scan over a hand-crafted feature vector). Instead of
recomputing a distance over every candidate on each query — and instead of token-Jaccard string
matching — every robotics object gets a persistent embedding in a SQLite-backed store, and "nearest
in the vector" means "reusable for warm-start". Four sub-spaces (body / skill / episode / task)
share one table, namespaced by ``obj_type`` so bodies are compared to bodies, skills to skills.

Design honesty (see docs/usp_and_two_pillar_plan.md, Pillar 2):

* **The STORE is the new infrastructure.** Cosine kNN + provenance edges are real and CPU-only.
* **The embeddings it indexes are, today, deterministic and dependency-free** — the existing
  ``morphology_embedding.embed_gene`` for bodies, and a feature-hashing embedding for text. The
  *learned* graph latent (``GeneGNN.embed``) and a frozen-LM sentence encoder slot in behind the
  SAME function signatures later (the GPU phase) without changing any caller.
* **Provenance is first-class** so warm-start compounding is *proven, not asserted* (what seeded
  this build, and the measured success delta) — the moat metric the embeddings research flagged as
  the thing competitors can't fake.

Pure-Python and dependency-free (lives beside the SQLite memory layer); no numpy/faiss/torch on the
default path, so it runs anywhere the memory DB does.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time

from virturoid.schemas.gene import RobotGene
from virturoid.services.morphology_embedding import FEATURE_NAMES, embed_gene

_TEXT_DIM = 64
BODY_DIM = len(FEATURE_NAMES)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# object-type namespaces (each a separate kNN sub-space in the shared table)
BODY, SKILL, EPISODE, TASK, RUN = "body", "skill", "episode", "task", "run"
TIP, LESSON = "tip", "lesson"   # knowledge sub-spaces: semantic tips + procedural lessons, keyed by z_body

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    obj_type   TEXT NOT NULL,          -- body | skill | episode | task | run
    obj_id     TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        TEXT NOT NULL,          -- JSON list[float], L2-normalized on write
    meta       TEXT,                   -- JSON (robot_class, task_type, ...)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (obj_type, obj_id)
);
CREATE INDEX IF NOT EXISTS idx_vectors_type ON vectors(obj_type);
CREATE TABLE IF NOT EXISTS provenance (
    -- A warm-start edge: this child object was seeded by that parent (a prior body/skill/reward),
    -- with the measured success delta attributable to the reuse. The compounding proof.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    child_type  TEXT NOT NULL,
    child_id    TEXT NOT NULL,
    parent_type TEXT,
    parent_id   TEXT,
    kind        TEXT,                  -- warm_start | amend | transfer | reward_reuse
    delta       REAL,                  -- success delta vs the cold-start / prior baseline
    meta        TEXT,                  -- JSON
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prov_child  ON provenance(child_type, child_id);
CREATE INDEX IF NOT EXISTS idx_prov_parent ON provenance(parent_type, parent_id);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- embeddings
def _l2(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _hash_bucket(token: str, dim: int) -> tuple[int, float]:
    """Stable (process-independent) feature-hashing bucket + sign for a token.

    Uses md5, NOT Python's built-in ``hash`` (which is salted per-process for strings), so the same
    text always embeds to the same vector across runs and machines — required for a persisted index.
    """
    h = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:4], "big") % dim
    sign = 1.0 if (h[4] & 1) else -1.0
    return bucket, sign


def embed_text(text: str, dim: int = _TEXT_DIM) -> list[float]:
    """Deterministic, dependency-free text embedding (feature hashing over tokens + char trigrams).

    A stand-in for a frozen sentence-embedding model that needs no download: token unigrams capture
    word overlap, char trigrams capture sub-word similarity ("grasp" ~ "grasping") that a pure token
    set misses. L2-normalized so cosine is a dot product. The learned LM encoder slots in behind this
    same signature later.
    """
    vec = [0.0] * dim
    toks = _TOKEN_RE.findall((text or "").lower())
    grams: list[str] = list(toks)
    for t in toks:
        pad = f"#{t}#"
        grams += [pad[i:i + 3] for i in range(len(pad) - 2)]
    if not grams:
        return vec
    for g in grams:
        b, s = _hash_bucket(g, dim)
        vec[b] += s
    return _l2(vec)


def embed_task(prompt: str, task_type: str | None = None, robot_class: str | None = None) -> list[float]:
    """Task/query embedding: the natural-language prompt + structured task/class tokens."""
    return embed_text(" ".join(p for p in (prompt, task_type, robot_class) if p))


def embed_body(gene: RobotGene, latent: list[float] | None = None) -> list[float]:
    """Body embedding (``z_body``). Defaults to the deterministic 29-D morphology vector, optionally reweighted by
    a LEARNED diagonal transfer metric (``body_metric``) when one is fitted AND proven to beat the baseline on
    held-out transfer ranking — that reweighting, not unsupervised whitening, is what makes distance predict
    transfer (measured, not asserted; a null metric is identity so this never regresses). Pass a learned graph
    latent (``GeneGNN.embed``) to use it instead — same downstream index."""
    if latent is not None:
        return _l2(list(latent))
    from virturoid.services.body_metric import embed_metric
    return _l2(embed_metric(gene))


def embed_skill(task_text: str, gene: RobotGene | None = None, *, success_rate: float | None = None,
                latent: list[float] | None = None) -> list[float]:
    """Skill embedding (``z_skill`` = z_body ⊕ z_task ⊕ a small policy fingerprint), fixed-length so
    every banked skill lives in one comparable sub-space (body block is zeros when the gene is absent).

    The body block here is the DIMENSION-STABLE baseline morphology vector, NOT the metric-aware ``embed_body``:
    if a learned metric were later adopted it would change ``embed_body``'s dimension, silently desyncing the
    skill index (a gene-present skill vs a gene-absent zero-block of a different length). Keeping the skill space
    on the fixed 29-D baseline makes it stable across metric adoptions; the metric improves the BODY sub-space
    (re-indexed on adoption). Byte-identical today (the metric is gated off, so ``embed_body`` == this)."""
    body = (_l2(list(latent)) if latent is not None else _l2(embed_gene(gene))) if gene is not None \
        else [0.0] * BODY_DIM
    task = embed_text(task_text)
    fingerprint = [float(success_rate or 0.0)]
    return _l2(body + task + fingerprint)


def embed_episode(features: dict[str, float]) -> list[float]:
    """Episode embedding (``z_epi``) from a proprioception/behavior feature dict (mean/std joint
    velocity, cadence, upright fraction, forward distance, contacts, ...). Deterministic feature
    hashing so arbitrary named features map to a stable vector. The frozen-vision (DINOv2) path for
    camera rollouts slots in behind this same signature on the GPU phase."""
    vec = [0.0] * _TEXT_DIM
    for key, value in sorted((features or {}).items()):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        b, s = _hash_bucket(str(key), _TEXT_DIM)
        vec[b] += s * v
    return _l2(vec)


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity (bigger = more similar). Robust to unequal lengths and zero vectors."""
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return sum(a[i] * b[i] for i in range(m)) / (na * nb)


# --------------------------------------------------------------------------- learned z_body (gated)
# The learned graph latent (GeneGNN.embed) is used for the body sub-space ONLY when a trained model exists
# AND its latent agrees with the deterministic morphology space on obvious cases — so a learned z_body can
# never be worse than the shipped hand-crafted vector (the same "never worse" discipline as similar_runs).
_GNN_MODEL_PATH = "build/models/gene_gnn.pt"
_TRUSTED_GNN = None   # None = untried; False = absent/untrusted; else a loaded GeneGNN


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no numpy). 0.0 for <2 points."""
    n = len(a)
    if n < 2:
        return 0.0

    def _rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, idx in enumerate(order):
            r[idx] = pos
        return r

    ra, rb = _rank(a), _rank(b)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _load_trusted_gnn():
    """The trained gene GNN IF ``build/models/gene_gnn.pt`` exists AND its pooled latent rank-agrees with the
    deterministic morphology space on the fixture bodies (Spearman ≥ 0.5 over pairwise distances). Cached once
    per process. Returns the model or ``None`` (→ callers use the deterministic ``embed_gene``). This gate is
    what makes the learned z_body opt-in-by-quality: a narrow/degenerate model that disagrees with obvious
    morphology structure (arm≈arm, arm≠quad) is distrusted and never shipped."""
    global _TRUSTED_GNN
    if _TRUSTED_GNN is not None:
        return _TRUSTED_GNN or None
    _TRUSTED_GNN = False
    try:
        import itertools
        from pathlib import Path
        if not Path(_GNN_MODEL_PATH).exists():
            return None
        from virturoid.fixtures.gene_library import (humanoid_upper_body_gene, quadruped_gene,
                                                     tabletop_arm_gene)
        from virturoid.services.design_critic import add_parallel_gripper
        from virturoid.services.gene_surrogate_nn import GeneGNN
        gnn = GeneGNN.load(Path(_GNN_MODEL_PATH), device="cpu")
        fixtures = [tabletop_arm_gene(), add_parallel_gripper(tabletop_arm_gene()),
                    humanoid_upper_body_gene(), quadruped_gene()]
        lat = [gnn.embed(g) for g in fixtures]
        det = [embed_gene(g) for g in fixtures]
        pairs = list(itertools.combinations(range(len(fixtures)), 2))
        dl = [math.sqrt(sum((lat[i][k] - lat[j][k]) ** 2 for k in range(len(lat[i])))) for i, j in pairs]
        dd = [math.sqrt(sum((det[i][k] - det[j][k]) ** 2 for k in range(len(det[i])))) for i, j in pairs]
        if _spearman(dl, dd) >= 0.5:
            _TRUSTED_GNN = gnn
    except Exception:  # noqa: BLE001 - any failure -> deterministic embedding (never worse)
        _TRUSTED_GNN = False
    return _TRUSTED_GNN or None


def _body_latent(gene: RobotGene) -> list[float] | None:
    """The learned z_body (GeneGNN pooled latent) when a TRUSTED model exists, else ``None`` (→ the caller
    falls back to the deterministic morphology vector via ``embed_body``)."""
    gnn = _load_trusted_gnn()
    if gnn is None:
        return None
    try:
        return gnn.embed(gene)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- the store
class RoboticsVectorMemory:
    """SQLite-backed cosine-kNN store over the four robotics sub-spaces + a provenance ledger.

    Share the memory DB's connection (``RoboticsVectorMemory(db)`` where ``db`` is a ``MemoryDB`` or a
    ``sqlite3.Connection``) so bodies/skills/runs live in the one file the plan mandates (§33: "do not
    start with too many databases"; Postgres+pgvector is the drop-in scale-out behind this interface).
    """

    def __init__(self, db) -> None:
        if hasattr(db, "conn"):
            self.conn = db.conn
        elif isinstance(db, sqlite3.Connection):
            self.conn = db
        else:  # a path
            self.conn = sqlite3.connect(str(db))
            self.conn.row_factory = sqlite3.Row
        if self.conn.row_factory is None:
            self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ----------------------------------------------------------------- writes
    def upsert(self, obj_type: str, obj_id: str, vec: list[float], meta: dict | None = None) -> None:
        """Store (and L2-normalize) a vector under ``obj_type``/``obj_id``. Idempotent."""
        v = _l2(list(vec))
        now = _now()
        self.conn.execute(
            """INSERT INTO vectors (obj_type, obj_id, dim, vec, meta, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(obj_type, obj_id) DO UPDATE SET
                   dim=excluded.dim, vec=excluded.vec, meta=excluded.meta, updated_at=excluded.updated_at""",
            (obj_type, str(obj_id), len(v), json.dumps(v), json.dumps(meta or {}), now, now),
        )
        self.conn.commit()

    def record_provenance(self, child_type: str, child_id: str, *, parent_type: str | None = None,
                          parent_id: str | None = None, kind: str = "warm_start",
                          delta: float | None = None, meta: dict | None = None) -> None:
        """Record a warm-start edge (what seeded this build + the measured success delta)."""
        self.conn.execute(
            """INSERT INTO provenance (child_type, child_id, parent_type, parent_id, kind, delta, meta, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (child_type, str(child_id), parent_type, None if parent_id is None else str(parent_id),
             kind, delta, json.dumps(meta or {}), _now()),
        )
        self.conn.commit()

    # ----------------------------------------------------------------- reads
    def get(self, obj_type: str, obj_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM vectors WHERE obj_type=? AND obj_id=?", (obj_type, str(obj_id))
        ).fetchone()
        if row is None:
            return None
        return {"obj_id": row["obj_id"], "vec": json.loads(row["vec"]),
                "meta": json.loads(row["meta"] or "{}")}

    def nearest(self, obj_type: str, vec: list[float], k: int = 5, *, exclude_id: str | None = None,
                min_sim: float | None = None) -> list[dict]:
        """Top-``k`` most similar objects in one sub-space (cosine, bigger = closer).

        Returns ``[{obj_id, similarity, meta}]`` sorted descending. ``min_sim`` filters weak matches;
        ``exclude_id`` drops the query object itself (useful when querying with a stored body's vec).
        """
        q = _l2(list(vec))
        rows = self.conn.execute(
            "SELECT obj_id, vec, meta FROM vectors WHERE obj_type=?", (obj_type,)
        ).fetchall()
        scored = []
        for r in rows:
            if exclude_id is not None and r["obj_id"] == str(exclude_id):
                continue
            sv = json.loads(r["vec"])
            # DIM GUARD (Wave 4): never SILENTLY zip-truncate a mismatched vector into a bogus similarity. A
            # stored vector of a different length is from another embedding version (e.g. the index predates a
            # metric adoption that changed the dim) — skip it so it triggers a re-index, not a wrong neighbour.
            if len(sv) != len(q):
                continue
            sim = sum(a * b for a, b in zip(q, sv))                    # both normalized -> dot
            if min_sim is not None and sim < min_sim:
                continue
            scored.append({"obj_id": r["obj_id"], "similarity": round(sim, 6),
                           "meta": json.loads(r["meta"] or "{}")})
        scored.sort(key=lambda d: d["similarity"], reverse=True)
        return scored[:k]

    def count(self, obj_type: str | None = None) -> int:
        if obj_type is None:
            return int(self.conn.execute("SELECT COUNT(*) AS c FROM vectors").fetchone()["c"])
        return int(self.conn.execute(
            "SELECT COUNT(*) AS c FROM vectors WHERE obj_type=?", (obj_type,)).fetchone()["c"])

    def provenance_for(self, child_type: str, child_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM provenance WHERE child_type=? AND child_id=? ORDER BY id",
            (child_type, str(child_id)),
        ).fetchall()
        return [dict(r) for r in rows]

    def compounding_summary(self) -> dict:
        """Is warm-start reuse actually paying off? Aggregate the provenance ledger into the moat
        metric: how many builds were seeded, and the mean measured success delta of those seedings."""
        rows = self.conn.execute("SELECT delta FROM provenance").fetchall()
        deltas = [r["delta"] for r in rows if r["delta"] is not None]
        edges = int(self.conn.execute("SELECT COUNT(*) AS c FROM provenance").fetchone()["c"])
        seeded = int(self.conn.execute(
            "SELECT COUNT(DISTINCT child_id) AS c FROM provenance").fetchone()["c"])
        mean_delta = round(sum(deltas) / len(deltas), 6) if deltas else None
        positive = round(sum(1 for d in deltas if d > 0) / len(deltas), 4) if deltas else None
        return {"edges": edges, "seeded_builds": seeded, "measured_deltas": len(deltas),
                "mean_delta": mean_delta, "positive_fraction": positive}

    def hint_reuse_summary(self) -> dict:
        """Measured flywheel-hint deployments, with failures in the denominator as well as wins."""
        rows = self.conn.execute(
            "SELECT delta FROM provenance WHERE kind='gait_hint_deploy' AND delta IS NOT NULL"
        ).fetchall()
        deltas = [float(r["delta"]) for r in rows]
        wins = sum(1 for d in deltas if d > 1e-9)
        losses = sum(1 for d in deltas if d < -1e-9)
        ties = len(deltas) - wins - losses
        return {"trials": len(deltas), "wins": wins, "losses": losses, "ties": ties,
                "hit_rate": round(wins / len(deltas), 4) if deltas else None,
                "mean_delta_m": round(sum(deltas) / len(deltas), 6) if deltas else None}

    # ----------------------------------------------------------------- integrators (backfill)
    def index_runs(self) -> int:
        """Embed every run's (prompt + task_type) into the ``run`` sub-space; skip already-indexed.

        Cheap and incremental — only new runs are embedded — so it can be called before each semantic
        query to keep the index fresh. Returns the number newly indexed.
        """
        existing = {r["obj_id"] for r in self.conn.execute(
            "SELECT obj_id FROM vectors WHERE obj_type=?", (RUN,)).fetchall()}
        rows = self.conn.execute(
            "SELECT id, prompt, robot_class, task_type FROM runs").fetchall()
        added = 0
        for r in rows:
            oid = str(r["id"])
            if oid in existing:
                continue
            self.upsert(RUN, oid, embed_task(r["prompt"], r["task_type"], r["robot_class"]),
                        {"robot_class": r["robot_class"], "task_type": r["task_type"]})
            added += 1
        return added

    def index_species_bodies(self) -> int:
        """Embed every species-tree node that carries a full gene into the ``body`` sub-space.

        Powers cross-body warm-start retrieval (a new prompt's candidate body -> nearest prior bodies)
        without recomputing embeddings per query. Returns the number of body vectors written.
        """
        rows = self.conn.execute(
            "SELECT species_pattern, robot_class, genes, buildable FROM species_tree "
            "WHERE genes IS NOT NULL AND merged_into IS NULL"
        ).fetchall()
        written = 0
        for r in rows:
            try:
                g = json.loads(r["genes"])
                if not (isinstance(g, dict) and g.get("segments") and isinstance(g["segments"][0], dict)):
                    continue
                gene = RobotGene.from_dict(g)
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue
            self.upsert(BODY, r["species_pattern"], embed_body(gene, latent=_body_latent(gene)),
                        {"robot_class": r["robot_class"], "buildable": bool(r["buildable"])})
            written += 1
        return written

    def _species_genes(self) -> dict:
        """species_pattern -> RobotGene for nodes carrying a full gene (for skill body embeddings)."""
        out: dict = {}
        for r in self.conn.execute(
                "SELECT species_pattern, genes FROM species_tree WHERE genes IS NOT NULL").fetchall():
            try:
                g = json.loads(r["genes"])
                if isinstance(g, dict) and g.get("segments") and isinstance(g["segments"][0], dict):
                    out[r["species_pattern"]] = RobotGene.from_dict(g)
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue
        return out

    def index_skills(self) -> int:
        """Embed every banked skill into the ``skill`` sub-space (z_skill = body ⊕ task ⊕ success),
        keyed by skill_id, so a warm-start search is a cosine kNN that also retrieves a NEAR-morphology
        skill the exact class+task string match misses (a push skill on a similar arm seeds a new
        transport arm — the skill-transfer moat, made semantic). Bodies are recovered from each skill's
        species gene when available (zeros otherwise). Returns the number of skill vectors written.
        """
        genes = self._species_genes()
        rows = self.conn.execute(
            "SELECT skill_id, robot_class, species, task_type, success_rate FROM skills").fetchall()
        written = 0
        for r in rows:
            gene = genes.get(r["species"])
            task_text = " ".join(p for p in (r["task_type"], r["robot_class"], r["species"]) if p)
            vec = embed_skill(task_text, gene, success_rate=r["success_rate"],
                              latent=_body_latent(gene) if gene is not None else None)
            self.upsert(SKILL, r["skill_id"], vec,
                        {"robot_class": r["robot_class"], "task_type": r["task_type"], "species": r["species"]})
            written += 1
        return written

    def nearest_skills(self, gene: RobotGene | None, task_type: str, *, robot_class: str | None = None,
                       k: int = 5, min_sim: float | None = None) -> list[dict]:
        """Nearest banked skills to a (body, task) in the ``skill`` sub-space — cross-body warm-start
        candidates ranked by morphology+task similarity (call ``index_skills`` first to populate)."""
        q = embed_skill(" ".join(p for p in (task_type, robot_class) if p), gene, success_rate=1.0,
                        latent=_body_latent(gene) if gene is not None else None)
        return self.nearest(SKILL, q, k=k, min_sim=min_sim)

    def nearest_bodies(self, gene: RobotGene, *, k: int = 5, min_sim: float | None = None,
                       exclude_id: str | None = None) -> list[dict]:
        """Nearest prior bodies to a gene in the ``body`` sub-space (call ``index_species_bodies`` first).

        The morphology-retrieval primitive the understanding layer exposes ("what have we built that's
        shaped like this?") — the read that ``nearest(BODY, ...)`` had no production caller for (audit gap).
        """
        return self.nearest(BODY, embed_body(gene, latent=_body_latent(gene)), k=k, min_sim=min_sim,
                            exclude_id=exclude_id)

    # ----------------------------------------------------------------- knowledge (tips + lessons), z_body-keyed
    def index_tips(self, *, incremental: bool = True) -> int:
        """Embed every species **tip** into the ``tip`` sub-space, keyed by the ``z_body`` of the species it
        was written for — so a tip written for one body is retrievable for a NEAR-morphology body (this
        un-dead-ends the tips: they were write-only and string-keyed to one exact species, read by no LLM).
        The tip text lives in ``meta``. Incremental: only newly-seen ``species#i`` tips are embedded."""
        existing = ({r["obj_id"] for r in self.conn.execute(
            "SELECT obj_id FROM vectors WHERE obj_type=?", (TIP,)).fetchall()} if incremental else set())
        rows = self.conn.execute(
            "SELECT species_pattern, robot_class, genes, tips FROM species_tree "
            "WHERE tips IS NOT NULL AND genes IS NOT NULL AND merged_into IS NULL"
        ).fetchall()
        added = 0
        for r in rows:
            try:
                tips = json.loads(r["tips"]) or []
                g = json.loads(r["genes"])
                if not (isinstance(g, dict) and g.get("segments") and isinstance(g["segments"][0], dict)):
                    continue
                gene = RobotGene.from_dict(g)
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue
            zb = embed_body(gene, latent=_body_latent(gene))
            for i, t in enumerate(tips):
                oid = f"{r['species_pattern']}#{i}"
                if oid in existing:
                    continue
                self.upsert(TIP, oid, zb, {"species": r["species_pattern"], "robot_class": r["robot_class"],
                                          "audience": t.get("audience"), "tip": t.get("tip")})
                added += 1
        return added

    def index_lessons(self, *, incremental: bool = True) -> int:
        """Embed every proven **lesson** (diagnosed failure -> the fix) into the ``lesson`` sub-space, keyed
        by ``z_body(class-representative body) ⊕ embed_text(failure_code+task+class)`` — so a lesson learned
        on one body is retrievable for a near-morphology body hitting a near failure (it was EXACT
        class+failure_code string match only, via ``recall_lesson``). The fix lives in ``meta``. Incremental."""
        existing = ({r["obj_id"] for r in self.conn.execute(
            "SELECT obj_id FROM vectors WHERE obj_type=?", (LESSON,)).fetchall()} if incremental else set())
        genes_by_class: dict = {}
        for gene in self._species_genes().values():
            genes_by_class.setdefault(gene.robot_class, gene)
        rows = self.conn.execute(
            "SELECT id, robot_class, task_type, failure_code, root_cause, operator, params, improvement, "
            "times_applied FROM lessons WHERE improvement > 0"
        ).fetchall()
        added = 0
        for r in rows:
            oid = str(r["id"])
            if oid in existing:
                continue
            rep = genes_by_class.get(r["robot_class"])
            body = embed_body(rep) if rep is not None else [0.0] * BODY_DIM
            key_text = " ".join(p for p in (r["failure_code"], r["task_type"], r["robot_class"]) if p)
            self.upsert(LESSON, oid, _l2(body + embed_text(key_text)),
                        {"robot_class": r["robot_class"], "task_type": r["task_type"],
                         "failure_code": r["failure_code"], "root_cause": r["root_cause"],
                         "operator": r["operator"], "params": r["params"],
                         "improvement": r["improvement"], "times_applied": r["times_applied"]})
            added += 1
        return added

    def recall_knowledge(self, gene: RobotGene, task_type: str | None = None, *,
                         failure_code: str | None = None, k: int = 3, refresh: bool = True,
                         behavior_features: dict | None = None) -> dict:
        """The **recall organ**: retrieve prior tips + lessons + skills for THIS body, keyed by its
        morphology embedding — so an LLM role reasons with "what worked on bodies like this", not blind.

        Returns ``{"tips":[...], "lessons":[...], "skills":[...], "episodes":[...]}``, each
        ``[{obj_id, similarity, meta}]``. ``behavior_features`` (a just-run rollout's cadence/upright/forward/…)
        adds the EPISODE channel — bodies that *behaved* like this — the 4th sub-space that was indexed on every
        build but had no reader (an observability gap). Honest scope: episode neighbours are behavioural CONTEXT,
        not a fine-transfer predictor (measured: within-class control transfer is near chance short of running it).
        ``refresh`` re-indexes the append-mostly tip/lesson spaces + populates body/skill spaces if empty.
        """
        if refresh:
            self.index_tips()
            self.index_lessons()
            if self.count(BODY) == 0:
                self.index_species_bodies()
            if self.count(SKILL) == 0:
                self.index_skills()
        zb = embed_body(gene, latent=_body_latent(gene))
        lesson_q = _l2(embed_body(gene) + embed_text(
            " ".join(p for p in (failure_code, task_type, gene.robot_class) if p)))
        return {
            "tips": self.nearest(TIP, zb, k=k),
            "lessons": self.nearest(LESSON, lesson_q, k=k),
            "skills": self.nearest_skills(gene, task_type or "", robot_class=gene.robot_class, k=k),
            "episodes": (self.nearest_episodes(behavior_features, k=k) if behavior_features else []),
        }

    def index_episode(self, obj_id: str, features: dict, meta: dict | None = None) -> None:
        """Embed one episode's behavior features (a walk's cadence/upright/forward, a grasp's success/…) into
        the ``episode`` sub-space, so "find episodes that behaved like this" is a cosine kNN — the 4th sub-space
        of the unified index, populated from the honest rollout stats every build already computes."""
        self.upsert(EPISODE, str(obj_id), embed_episode(features or {}), meta or {})

    def nearest_episodes(self, features: dict, *, k: int = 5, min_sim: float | None = None) -> list[dict]:
        """Nearest prior episodes to a behavior-feature query (cosine kNN over the ``episode`` sub-space)."""
        return self.nearest(EPISODE, embed_episode(features or {}), k=k, min_sim=min_sim)
