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
    """Body embedding (``z_body``). Defaults to the deterministic morphology feature vector; pass a
    learned graph latent (e.g. ``GeneGNN.embed(gene)``) to use it instead — same downstream index."""
    return _l2(list(latent) if latent is not None else embed_gene(gene))


def embed_skill(task_text: str, gene: RobotGene | None = None, *, success_rate: float | None = None,
                latent: list[float] | None = None) -> list[float]:
    """Skill embedding (``z_skill`` = z_body ⊕ z_task ⊕ a small policy fingerprint), fixed-length so
    every banked skill lives in one comparable sub-space (body block is zeros when the gene is absent)."""
    body = embed_body(gene, latent) if gene is not None else [0.0] * BODY_DIM
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
            sim = sum(a * b for a, b in zip(q, json.loads(r["vec"])))  # both normalized -> dot
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
            self.upsert(BODY, r["species_pattern"], embed_body(gene),
                        {"robot_class": r["robot_class"], "buildable": bool(r["buildable"])})
            written += 1
        return written
