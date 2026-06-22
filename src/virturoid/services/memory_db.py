"""Virturoid Memory database (startup plan §4, §13, §33).

The plan calls memory "the core product": Virturoid must remember what a robot is,
what tasks it attempted, what scenes/policies worked, what failed, what fixed it,
and what transfers to similar robots — then *retrieve* that for new builds so the AI
agents learn and improve instead of starting from scratch every time.

This is the canonical store for that. Per §33 ("Postgres + pgvector later, but do not
start with too many databases"), the first implementation is a single embedded
**SQLite** database — zero dependencies, offline, file-backed, fully queryable — with
a schema that covers the §13 memory types (run/robot, task+design, failure, transfer,
species) and the §13.4 privacy fields. Retrieval is exact (keys/metrics) plus a
dependency-free text similarity (token Jaccard) for "similar tasks"; the
``similar_runs`` interface is the seam where a real embedding/pgvector backend slots in
later without changing callers.

The AI agents use this two ways:
- **read** — warm-start designs (`best_design`), pick transfer candidates
  (`transfer_candidates`), avoid known failures (`known_failures`), find similar prior
  work (`similar_runs`).
- **write/learn** — every real build records a run + its failures + any transfer via
  `record_run`; high-success rows accumulate into a supervised
  `training_dataset()` the local model is fine-tuned on (the data flywheel).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path("build") / "memory" / "virturoid_memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    robot_class TEXT NOT NULL,
    species TEXT,
    task_type TEXT NOT NULL,
    converged_design TEXT,          -- JSON
    success_rate REAL,
    target_success_rate REAL,
    succeeded INTEGER,              -- 0/1
    backend TEXT,                   -- which LLM brain (openai/claude/local/off)
    design_source TEXT,             -- as_built / physics_codesign / ai_designer+codesign
    created_at TEXT NOT NULL,
    -- §13.4 privacy controls
    owner TEXT DEFAULT 'local',
    visibility TEXT DEFAULT 'private',
    training_permission TEXT DEFAULT 'full_opt_in'
);
CREATE TABLE IF NOT EXISTS designs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    prompt TEXT NOT NULL,
    robot_class TEXT NOT NULL,
    task_type TEXT NOT NULL,
    design TEXT NOT NULL,           -- JSON
    success_rate REAL,
    source TEXT,
    training_permission TEXT DEFAULT 'full_opt_in',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    robot_class TEXT,
    task_type TEXT,
    label TEXT NOT NULL,
    count INTEGER DEFAULT 1,
    root_cause TEXT,
    recommended_action TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transfers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    source_key TEXT,
    robot_class TEXT,
    assets TEXT,                    -- JSON list
    reused INTEGER,                 -- 0/1
    success_rate REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS species (
    name TEXT PRIMARY KEY,
    robot_class TEXT,
    best_success_rate REAL,
    best_design TEXT,               -- JSON
    runs INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    -- The reasoned-redesign flywheel: a diagnosed body limit -> the proven fix.
    -- Customer 1's box-lifter discovers "reach_limited -> lengthen the upper arm";
    -- customer 2's humanoid that hits the SAME diagnosis retrieves this fix instantly
    -- instead of re-deriving it. This is the cross-customer moat made queryable.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    robot_class TEXT NOT NULL,
    task_type TEXT,
    failure_code TEXT NOT NULL,     -- diagnosed failure mode, e.g. reach_limited
    root_cause TEXT,                -- physical, human-readable ("ee short by 0.12 m")
    operator TEXT NOT NULL,         -- amendment operator that fixed it (lengthen_reach, ...)
    params TEXT,                    -- JSON kwargs for the operator
    improvement REAL,               -- best observed success delta when applied
    source_gene TEXT,               -- provenance: gene id that first proved this fix
    times_applied INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skills (
    -- The SKILL flywheel (M3 G4): a trained control policy banked with everything needed to reuse it.
    -- Customer 1 trains a tabletop gripper to grasp; customer 2's related body/task retrieves this
    -- skill and WARM-STARTS its policy from it instead of learning from scratch. This is the flywheel
    -- applied to policies (not just designs/lessons) — the moat for learned control.
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT UNIQUE,           -- stable id, e.g. grasp__manipulator.tabletop.gripper
    robot_class TEXT NOT NULL,
    species TEXT,
    task_type TEXT NOT NULL,        -- push | grasp | transport | ...
    gene_id TEXT,                   -- provenance: gene this skill was trained on
    reward_spec TEXT,               -- JSON: the process-aware reward terms (so the next build reuses the recipe)
    region TEXT,                    -- JSON: the body<->task-matched spawn region the skill is valid in
    base_config TEXT,               -- JSON: base-controller params (kpj/kdj/alpha/phases/...) to reproduce + warm-start
    success_rate REAL,              -- held-out sustained success the skill achieved
    params_path TEXT,               -- path to the saved policy params (npz)
    obs_dim INTEGER,                -- policy IO dims (warm-start compatibility)
    act_dim INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS species_tree (
    species_pattern TEXT PRIMARY KEY,   -- dotted taxonomy id, e.g. humanoid.arm.single
    parent_species TEXT,                -- the parent node in the tree (NULL = kingdom root)
    robot_class TEXT,
    morphology_tags TEXT,               -- JSON list
    genes TEXT,                         -- JSON: parametric kinematic encoding (links/joints/...)
    buildable INTEGER DEFAULT 0,        -- 0/1: has an implemented builder
    source TEXT,                        -- 'catalog' | 'species_agent' | 'requested'
    tips TEXT,                          -- JSON list of {audience, tip, created_at} for builder/trainer agents
    merged_into TEXT,                   -- if set, this node was merged into another (tombstone)
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_class_task ON runs(robot_class, task_type);
CREATE INDEX IF NOT EXISTS idx_failures_class_task ON failures(robot_class, task_type);
CREATE INDEX IF NOT EXISTS idx_species_parent ON species_tree(parent_species);
CREATE INDEX IF NOT EXISTS idx_lessons_class_code ON lessons(robot_class, failure_code);
CREATE INDEX IF NOT EXISTS idx_skills_class_task ON skills(robot_class, task_type);
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MemoryDB:
    """Embedded SQLite-backed Virturoid Memory. Use as a context manager or call close()."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def __enter__(self) -> "MemoryDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------- writes (learn)
    def record_run(
        self,
        prompt: str,
        robot_class: str,
        task_type: str,
        converged_design: dict | None,
        success_rate: float,
        *,
        species: str | None = None,
        target_success_rate: float | None = None,
        succeeded: bool | None = None,
        backend: str | None = None,
        design_source: str | None = None,
        training_permission: str = "full_opt_in",
    ) -> int:
        """Record one build run (§13.3 step 1) and feed the design flywheel + species memory."""
        now = _now()
        cur = self.conn.execute(
            """INSERT INTO runs (prompt, robot_class, species, task_type, converged_design,
                   success_rate, target_success_rate, succeeded, backend, design_source,
                   created_at, training_permission)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (prompt, robot_class, species, task_type,
             json.dumps(converged_design) if converged_design else None,
             success_rate, target_success_rate,
             None if succeeded is None else int(succeeded),
             backend, design_source, now, training_permission),
        )
        run_id = int(cur.lastrowid)
        # Flywheel training row (only when there is a usable design).
        if converged_design:
            self.conn.execute(
                """INSERT INTO designs (run_id, prompt, robot_class, task_type, design,
                       success_rate, source, training_permission, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (run_id, prompt, robot_class, task_type, json.dumps(converged_design),
                 success_rate, design_source, training_permission, now),
            )
            if species:
                self._upsert_species(species, robot_class, success_rate, converged_design, now)
        self.conn.commit()
        return run_id

    def record_failures(self, run_id: int, robot_class: str, task_type: str,
                        clusters: list, root_cause: str | None = None,
                        recommended_action: str | None = None) -> None:
        """Record failure memory (§13.3 step 2). ``clusters`` may be dicts or objects."""
        now = _now()
        for c in clusters or []:
            label = _get(c, "label") or _get(c, "failure_label")
            if not label:
                continue
            self.conn.execute(
                """INSERT INTO failures (run_id, robot_class, task_type, label, count,
                       root_cause, recommended_action, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, robot_class, task_type, label, int(_get(c, "count") or 1),
                 root_cause, recommended_action, now),
            )
        self.conn.commit()

    def record_lesson(self, robot_class: str, failure_code: str, operator: str, *,
                      params: dict | None = None, root_cause: str | None = None,
                      improvement: float | None = None, task_type: str | None = None,
                      source_gene: str | None = None) -> None:
        """Record a reasoned-redesign lesson: a diagnosed failure -> the fix that worked.

        Idempotent on (robot_class, failure_code, operator): a repeat keeps the BEST observed
        improvement and bumps ``times_applied`` (so a fix proven on many robots ranks highest).
        This is the write side of the cross-customer flywheel — what makes the second customer's
        same problem cheap to solve.
        """
        now = _now()
        row = self.conn.execute(
            "SELECT id, improvement, times_applied FROM lessons "
            "WHERE robot_class=? AND failure_code=? AND operator=?",
            (robot_class, failure_code, operator),
        ).fetchone()
        if row is None:
            self.conn.execute(
                """INSERT INTO lessons (robot_class, task_type, failure_code, root_cause, operator,
                       params, improvement, source_gene, times_applied, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,1,?,?)""",
                (robot_class, task_type, failure_code, root_cause, operator,
                 json.dumps(params or {}), improvement, source_gene, now, now),
            )
        else:
            best = max(row["improvement"] or -1e9, improvement if improvement is not None else -1e9)
            keep_params = json.dumps(params or {}) if (improvement is not None and improvement >= (row["improvement"] or -1e9)) else None
            self.conn.execute(
                """UPDATE lessons SET improvement=?, times_applied=times_applied+1, updated_at=?,
                       params=COALESCE(?, params), root_cause=COALESCE(?, root_cause)
                   WHERE id=?""",
                (best, now, keep_params, root_cause, row["id"]),
            )
        self.conn.commit()

    def recall_lesson(self, robot_class: str, failure_code: str) -> dict | None:
        """Best proven fix for this class + diagnosed failure (the flywheel read).

        Ranks by observed improvement, then by how many robots it has worked on. Returns
        ``None`` when nothing has been learned yet (the loop then reasons from scratch).
        """
        row = self.conn.execute(
            "SELECT * FROM lessons WHERE robot_class=? AND failure_code=? AND improvement > 0 "
            "ORDER BY improvement DESC, times_applied DESC LIMIT 1",
            (robot_class, failure_code),
        ).fetchone()
        return _lesson_to_dict(row) if row else None

    def lessons_for_class(self, robot_class: str, limit: int = 20) -> list[dict]:
        """All proven fixes learned for a robot class (input for the Designer agent / reports)."""
        rows = self.conn.execute(
            "SELECT * FROM lessons WHERE robot_class=? ORDER BY improvement DESC, times_applied DESC LIMIT ?",
            (robot_class, limit),
        ).fetchall()
        return [_lesson_to_dict(r) for r in rows]

    def record_skill(self, skill_id: str, robot_class: str, task_type: str, *,
                     success_rate: float, params_path: str | None = None, species: str | None = None,
                     gene_id: str | None = None, reward_spec: dict | None = None,
                     region: dict | None = None, base_config: dict | None = None,
                     obs_dim: int | None = None, act_dim: int | None = None,
                     notes: str | None = None) -> None:
        """Bank a trained control policy + everything needed to reuse it (M3 G4 write side).

        Idempotent on ``skill_id``: a repeat keeps the BEST observed success (and that run's params /
        config / region), so the bank always offers the strongest known skill for warm-starting.
        """
        now = _now()
        row = self.conn.execute("SELECT id, success_rate FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
        rs = json.dumps(reward_spec or {}); rg = json.dumps(region or {}); bc = json.dumps(base_config or {})
        if row is None:
            self.conn.execute(
                """INSERT INTO skills (skill_id, robot_class, species, task_type, gene_id, reward_spec,
                       region, base_config, success_rate, params_path, obs_dim, act_dim, notes,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (skill_id, robot_class, species, task_type, gene_id, rs, rg, bc, success_rate,
                 params_path, obs_dim, act_dim, notes, now, now),
            )
        elif success_rate >= (row["success_rate"] or -1):   # only overwrite when this run is at least as good
            self.conn.execute(
                """UPDATE skills SET robot_class=?, species=?, task_type=?, gene_id=?, reward_spec=?,
                       region=?, base_config=?, success_rate=?, params_path=?, obs_dim=?, act_dim=?,
                       notes=?, updated_at=? WHERE id=?""",
                (robot_class, species, task_type, gene_id, rs, rg, bc, success_rate, params_path,
                 obs_dim, act_dim, notes, now, row["id"]),
            )
        self.conn.commit()

    def recall_skill(self, robot_class: str, task_type: str, *, species: str | None = None,
                     obs_dim: int | None = None, act_dim: int | None = None) -> dict | None:
        """Best banked skill to WARM-START a new (class, task) policy (the skill-flywheel read).

        Prefers an exact class+task match (optionally IO-compatible for direct param reuse), ranked by
        success. Returns ``None`` when nothing is banked yet — the trainer then cold-starts.
        """
        q = "SELECT * FROM skills WHERE robot_class=? AND task_type=?"
        params: list = [robot_class, task_type]
        if obs_dim is not None and act_dim is not None:
            q += " AND obs_dim=? AND act_dim=?"; params += [obs_dim, act_dim]
        rows = self.conn.execute(q + " ORDER BY success_rate DESC, updated_at DESC", params).fetchall()
        if not rows:
            return None
        if species:   # prefer the closest species among the top matches
            rows = sorted(rows, key=lambda r: (-(r["success_rate"] or 0),
                                               -_jaccard(_tokens(species), _tokens(r["species"] or ""))))
        return _skill_to_dict(rows[0])

    def similar_skills(self, task_type: str, species: str, *, robot_class: str | None = None,
                       limit: int = 5) -> list[dict]:
        """Banked skills ranked by relevance to a new (task, species) — cross-body transfer candidates.

        Used when there is no exact class+task skill: score by task match + species token similarity,
        so a related body's grasp skill can still seed a new one (skill-library transfer research).
        """
        q = "SELECT * FROM skills"
        params: tuple = ()
        if robot_class:
            q += " WHERE robot_class=?"; params = (robot_class,)
        rows = self.conn.execute(q, params).fetchall()
        st = _tokens(species)
        scored = []
        for r in rows:
            score = (1.0 if r["task_type"] == task_type else 0.0) + _jaccard(st, _tokens(r["species"] or ""))
            if score > 0:
                d = _skill_to_dict(r); d["relevance"] = round(score, 4)
                scored.append(d)
        scored.sort(key=lambda d: (d["relevance"], d.get("success_rate") or 0), reverse=True)
        return scored[:limit]

    def skills_for_class(self, robot_class: str, limit: int = 20) -> list[dict]:
        """All banked skills for a robot class (input for the Trainer agent / reports)."""
        rows = self.conn.execute(
            "SELECT * FROM skills WHERE robot_class=? ORDER BY success_rate DESC, updated_at DESC LIMIT ?",
            (robot_class, limit),
        ).fetchall()
        return [_skill_to_dict(r) for r in rows]

    def record_transfer(self, run_id: int, robot_class: str, decision: dict,
                        success_rate: float | None = None) -> None:
        """Record a transfer decision (§13.3 step 7)."""
        self.conn.execute(
            """INSERT INTO transfers (run_id, source_key, robot_class, assets, reused,
                   success_rate, created_at) VALUES (?,?,?,?,?,?,?)""",
            (run_id, decision.get("source_key"), robot_class,
             json.dumps(decision.get("assets", [])), int(bool(decision.get("reuse"))),
             success_rate, _now()),
        )
        self.conn.commit()

    def upsert_species_node(self, species_pattern: str, *, parent_species: str | None = None,
                           robot_class: str | None = None, morphology_tags: list | None = None,
                           genes: dict | None = None, buildable: bool = False,
                           source: str = "species_agent") -> None:
        """Add or update a node in the robot species tree (the growing taxonomy).

        Idempotent on ``species_pattern``. Used both to seed the catalog tree and to
        record new species the Species Agent proposes — so the tree grows over builds.
        """
        now = _now()
        existing = self.conn.execute(
            "SELECT species_pattern FROM species_tree WHERE species_pattern=?", (species_pattern,)
        ).fetchone()
        if existing is None:
            self.conn.execute(
                """INSERT INTO species_tree (species_pattern, parent_species, robot_class,
                       morphology_tags, genes, buildable, source, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (species_pattern, parent_species, robot_class,
                 json.dumps(morphology_tags or []), json.dumps(genes) if genes else None,
                 int(buildable), source, now, now),
            )
        else:
            self.conn.execute(
                """UPDATE species_tree SET parent_species=COALESCE(?, parent_species),
                       robot_class=COALESCE(?, robot_class), morphology_tags=?,
                       genes=COALESCE(?, genes), buildable=?, updated_at=?
                   WHERE species_pattern=?""",
                (parent_species, robot_class, json.dumps(morphology_tags or []),
                 json.dumps(genes) if genes else None, int(buildable), now, species_pattern),
            )
        self.conn.commit()

    def species_tree_nodes(self, include_merged: bool = False) -> list[dict]:
        """All live species nodes (the taxonomy). Merged-away tombstones excluded by default."""
        q = "SELECT * FROM species_tree"
        if not include_merged:
            q += " WHERE merged_into IS NULL"
        rows = self.conn.execute(q + " ORDER BY species_pattern").fetchall()
        return [_species_to_dict(r) for r in rows]

    def species_node(self, species_pattern: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM species_tree WHERE species_pattern=?", (species_pattern,)
        ).fetchone()
        return _species_to_dict(row) if row else None

    def find_gene_for_class(self, robot_class: str) -> dict | None:
        """The best stored full gene for a robot class (flywheel reuse: amend a prior build).

        Returns the gene dict from the highest-success buildable species node of this class
        that has a full gene stored, or None. Lets a new customer reuse/amend a prior gene
        instead of starting from the seed.
        """
        rows = self.conn.execute(
            "SELECT species_pattern, genes FROM species_tree "
            "WHERE robot_class=? AND buildable=1 AND genes IS NOT NULL AND merged_into IS NULL "
            "ORDER BY updated_at DESC",
            (robot_class,),
        ).fetchall()
        for r in rows:
            try:
                g = json.loads(r["genes"])
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(g, dict) and g.get("segments") and isinstance(g["segments"][0], dict):
                return g  # a full serialized gene (not just a name list)
        return None

    def add_species_tip(self, species_pattern: str, tip: str, audience: str = "builder") -> bool:
        """Append a tip an agent learned to a species node (for builder/trainer agents).

        Tips are deduped on text. Returns False if the node doesn't exist.
        """
        node = self.species_node(species_pattern)
        if node is None:
            return False
        tips = node.get("tips") or []
        if any(t.get("tip") == tip for t in tips):
            return True
        tips.append({"audience": audience, "tip": tip, "created_at": _now()})
        self.conn.execute(
            "UPDATE species_tree SET tips=?, updated_at=? WHERE species_pattern=?",
            (json.dumps(tips), _now(), species_pattern),
        )
        self.conn.commit()
        return True

    def species_children(self, species_pattern: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM species_tree WHERE parent_species=? AND merged_into IS NULL ORDER BY species_pattern",
            (species_pattern,),
        ).fetchall()
        return [_species_to_dict(r) for r in rows]

    def merge_species(self, drop: str, keep: str) -> bool:
        """Merge species ``drop`` into ``keep``: repoint its children, carry its tips, tombstone it.

        Caller is responsible for the safety gates (same robot_class, no live runs on drop,
        etc.) — see ``species_maintenance``. Returns False if either node is missing.
        """
        dn, kn = self.species_node(drop), self.species_node(keep)
        if dn is None or kn is None or drop == keep:
            return False
        # Repoint children of drop -> keep.
        self.conn.execute(
            "UPDATE species_tree SET parent_species=?, updated_at=? WHERE parent_species=?",
            (keep, _now(), drop),
        )
        # Carry tips over.
        merged_tips = (kn.get("tips") or []) + (dn.get("tips") or [])
        seen, deduped = set(), []
        for t in merged_tips:
            if t.get("tip") not in seen:
                seen.add(t.get("tip"))
                deduped.append(t)
        self.conn.execute(
            "UPDATE species_tree SET tips=?, updated_at=? WHERE species_pattern=?",
            (json.dumps(deduped), _now(), keep),
        )
        # Tombstone the dropped node.
        self.conn.execute(
            "UPDATE species_tree SET merged_into=?, updated_at=? WHERE species_pattern=?",
            (keep, _now(), drop),
        )
        self.conn.commit()
        return True

    def trim_species(self, species_pattern: str) -> bool:
        """Hard-delete a redundant leaf node. Caller enforces the safety gates."""
        if self.species_children(species_pattern):
            return False
        self.conn.execute("DELETE FROM species_tree WHERE species_pattern=?", (species_pattern,))
        self.conn.commit()
        return True

    def species_usage(self, species_pattern: str) -> int:
        """How many recorded runs used this species (a node with builds is never trimmed)."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM runs WHERE species=?", (species_pattern,)
        ).fetchone()
        return int(row["c"])

    def _upsert_species(self, name: str, robot_class: str, success_rate: float,
                       design: dict, now: str) -> None:
        row = self.conn.execute("SELECT best_success_rate, runs FROM species WHERE name=?", (name,)).fetchone()
        if row is None:
            self.conn.execute(
                """INSERT INTO species (name, robot_class, best_success_rate, best_design, runs, updated_at)
                   VALUES (?,?,?,?,1,?)""",
                (name, robot_class, success_rate, json.dumps(design), now),
            )
        else:
            best = max(row["best_success_rate"] or -1, success_rate)
            keep_design = json.dumps(design) if success_rate >= (row["best_success_rate"] or -1) else None
            if keep_design is not None:
                self.conn.execute(
                    "UPDATE species SET best_success_rate=?, best_design=?, runs=runs+1, updated_at=? WHERE name=?",
                    (best, keep_design, now, name),
                )
            else:
                self.conn.execute(
                    "UPDATE species SET runs=runs+1, updated_at=? WHERE name=?", (now, name),
                )

    # ----------------------------------------------------------------- reads (retrieve)
    def best_design(self, robot_class: str, task_type: str) -> dict | None:
        """Highest-success prior design for this exact class+task (warm-start)."""
        row = self.conn.execute(
            """SELECT * FROM runs WHERE robot_class=? AND task_type=? AND converged_design IS NOT NULL
               ORDER BY success_rate DESC, id DESC LIMIT 1""",
            (robot_class, task_type),
        ).fetchone()
        return _run_to_dict(row) if row else None

    def transfer_candidates(self, robot_class: str, limit: int = 5) -> list[dict]:
        """Best prior designs for a robot class across tasks — input for the Transfer Agent."""
        rows = self.conn.execute(
            """SELECT *, MAX(success_rate) FROM runs
               WHERE robot_class=? AND converged_design IS NOT NULL
               GROUP BY task_type ORDER BY success_rate DESC LIMIT ?""",
            (robot_class, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = _run_to_dict(r)
            d["key"] = f"{d['robot_class']}__{d['task_type']}"
            out.append(d)
        return out

    def known_failures(self, robot_class: str, task_type: str, limit: int = 10) -> list[dict]:
        """Aggregated failure memory for this class+task (so the build can pre-empt them)."""
        rows = self.conn.execute(
            """SELECT label, SUM(count) AS total, MAX(root_cause) AS root_cause,
                      MAX(recommended_action) AS recommended_action
               FROM failures WHERE robot_class=? AND task_type=?
               GROUP BY label ORDER BY total DESC LIMIT ?""",
            (robot_class, task_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def similar_runs(self, prompt: str, robot_class: str | None = None, limit: int = 5) -> list[dict]:
        """Most similar prior runs by task text (token Jaccard over prompt + task_type).

        This is the retrieval seam: swap the scoring for embeddings/pgvector later
        without changing callers.
        """
        q = "SELECT * FROM runs"
        params: tuple = ()
        if robot_class:
            q += " WHERE robot_class=?"
            params = (robot_class,)
        rows = self.conn.execute(q, params).fetchall()
        qt = _tokens(prompt)
        scored = []
        for r in rows:
            score = _jaccard(qt, _tokens(f"{r['prompt']} {r['task_type']}"))
            if score > 0:
                d = _run_to_dict(r)
                d["similarity"] = round(score, 4)
                scored.append(d)
        scored.sort(key=lambda d: d["similarity"], reverse=True)
        return scored[:limit]

    # ----------------------------------------------------------------- learning export
    def training_dataset(self, min_success: float = 0.0, require_opt_in: bool = True) -> list[dict]:
        """The supervised (prompt -> physics-optimized design) rows for fine-tuning.

        Respects §13.4 training_permission: rows marked ``no_training`` are excluded
        when ``require_opt_in`` is set.
        """
        q = "SELECT prompt, robot_class, task_type, design, success_rate, source FROM designs WHERE success_rate >= ?"
        params: list = [min_success]
        if require_opt_in:
            q += " AND training_permission != 'no_training'"
        rows = self.conn.execute(q + " ORDER BY success_rate DESC", params).fetchall()
        return [
            {**{k: r[k] for k in ("prompt", "robot_class", "task_type", "success_rate", "source")},
             "design": json.loads(r["design"])}
            for r in rows
        ]

    def export_training_jsonl(self, path: Path, min_success: float = 0.0) -> int:
        """Write the training dataset to JSONL (LoRA-ready). Returns row count."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.training_dataset(min_success=min_success)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return len(rows)

    def stats(self) -> dict:
        """Counts per table — quick health/observability snapshot."""
        def n(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])

        return {t: n(t) for t in ("runs", "designs", "failures", "transfers", "species", "species_tree", "lessons", "skills")}


def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _run_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("converged_design"):
        try:
            d["converged_design"] = json.loads(d["converged_design"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _lesson_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("params"):
        try:
            d["params"] = json.loads(d["params"])
        except (json.JSONDecodeError, TypeError):
            d["params"] = {}
    return d


def _skill_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("reward_spec", "region", "base_config"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = {}
    return d


def _species_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("morphology_tags", "genes", "tips"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    d["buildable"] = bool(d.get("buildable"))
    return d
