"""The Memory tab's read model must be able to report a memory that is NOT paying off.

The panel exists to make the moat visible, and on the live bank the moat's headline number is negative: the
dominant recall kind (``gait_hint_deploy``) has a mean delta below zero. A panel that can only render wins
would be a marketing surface, so the tests below are mostly about the unflattering cases --

  * a losing kind must come back ``direction == "hurts"`` and must drive the headline when it is the largest;
  * ties must stay out of the win-rate denominator, because folding 1634 ties into 246 wins turns a flat
    memory into a 90% one;
  * a robot with no recorded recall must say so, distinctly from "recall ran and did nothing";
  * the panel must not attribute one body's recall events to another (``anatomy_creature`` and
    ``anatomy_creature_91b931bf`` are both real, distinct ids in the live ledger);
  * and it must not be able to WRITE to the bank it reports on -- a status surface that migrates the schema of
    the developer's real memory every time a UI panel polls is the same class of mistake as the suite banking
    into the bank it measures.
"""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path

from virturoid.services.moat_panel import _gene_id_candidates, bank_census, moat_panel, recall_ledger


def _make_bank(path: Path, *, skills: list[dict], provenance: list[dict]) -> None:
    """A minimal bank with the two tables the panel reads. Deliberately NOT built through MemoryDB: the panel
    must cope with any schema that carries these columns, including one written by an older release."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE skills (id INTEGER PRIMARY KEY, skill_id TEXT, robot_class TEXT, species TEXT, "
                "task_type TEXT, gene_id TEXT, base_config TEXT)")
    con.execute("CREATE TABLE provenance (id INTEGER PRIMARY KEY, child_type TEXT, child_id TEXT, "
                "parent_type TEXT, parent_id TEXT, kind TEXT, delta REAL, meta TEXT, created_at TEXT)")
    for s in skills:
        con.execute("INSERT INTO skills (skill_id, robot_class, species, task_type, gene_id, base_config) "
                    "VALUES (?,?,?,?,?,?)",
                    (s.get("skill_id", "s"), s.get("robot_class", "quadruped"), s.get("species", "sp"),
                     s.get("task_type", "locomotion"), s.get("gene_id"), json.dumps(s.get("base_config", {}))))
    for p in provenance:
        con.execute("INSERT INTO provenance (child_type, child_id, parent_type, parent_id, kind, delta, meta, "
                    "created_at) VALUES (?,?,?,?,?,?,?,?)",
                    ("gene", p.get("child_id", "g"), "gait_hint_region", p.get("parent_id", "r"),
                     p.get("kind", "gait_hint_deploy"), p.get("delta"), json.dumps(p.get("meta", {})),
                     p.get("created_at", "2026-08-07T00:00:00")))
    con.commit()
    con.close()


class MoatPanelTests(unittest.TestCase):
    def _bank(self, **kw) -> Path:
        from tempfile import mkdtemp
        d = Path(mkdtemp(prefix="moat-"))
        _make_bank(d / "virturoid_memory.db", **kw)
        return d

    def test_a_missing_database_reads_as_an_honest_empty_not_an_error(self):
        out = moat_panel(Path("no") / "such" / "dir")
        self.assertFalse(out["db_present"])
        self.assertEqual(out["bank"]["rows"], 0)
        self.assertIn("nothing has been banked", out["recall"]["headline"].lower())
        self.assertFalse(out["this_build"]["matched"])

    def test_the_headline_reports_a_LOSING_dominant_kind_as_losing(self):
        """The load-bearing case. The biggest kind is negative; the panel must say BEHIND, not pick a winner."""
        d = self._bank(
            skills=[{"base_config": {"bank_gate": "ungated"}}],
            provenance=(
                [{"kind": "gait_hint_deploy", "delta": -0.4} for _ in range(8)]
                + [{"kind": "gait_hint_deploy", "delta": 0.1} for _ in range(2)]
                + [{"kind": "gait_warm_start", "delta": 2.0} for _ in range(3)]
            ),
        )
        out = moat_panel(d)
        dominant = out["recall"]["kinds"][0]
        self.assertEqual(dominant["kind"], "gait_hint_deploy")
        self.assertEqual(dominant["direction"], "hurts")
        self.assertLess(dominant["mean_delta_m"], 0)
        self.assertIn("BEHIND", out["recall"]["headline"])
        # The winning kind is still reported -- honesty is not pessimism.
        warm = next(k for k in out["recall"]["kinds"] if k["kind"] == "gait_warm_start")
        self.assertEqual(warm["direction"], "helps")

    def test_ties_are_excluded_from_the_win_rate_and_counted_separately(self):
        d = self._bank(skills=[{}], provenance=(
            [{"delta": 0.5}] + [{"delta": -0.5}] + [{"delta": 0.0} for _ in range(98)]))
        k = moat_panel(d)["recall"]["kinds"][0]
        self.assertEqual((k["wins"], k["losses"], k["ties"]), (1, 1, 98))
        self.assertAlmostEqual(k["decided_win_rate"], 0.5)   # NOT 1/100 and NOT 99/100
        self.assertEqual(k["edges"], 100)

    def test_the_census_separates_gated_rows_from_rows_that_merely_lack_a_stamp(self):
        d = self._bank(skills=[
            {"gene_id": "a", "base_config": {"bank_gate": "fragility_v1", "row_source": "real",
                                             "bank_door": "learn_gait_flywheel"}},
            {"gene_id": "a", "base_config": {"bank_gate": "ungated_declared", "row_source": "suite"}},
            {"gene_id": "b", "base_config": {}},
        ], provenance=[])
        con = sqlite3.connect(d / "virturoid_memory.db")
        con.row_factory = sqlite3.Row
        c = bank_census(con)
        con.close()
        self.assertEqual(c["rows"], 3)
        self.assertEqual(c["gated_rows"], 1)                       # only fragility_v1 counts as gated
        self.assertEqual(c["by_gate"]["ungated_declared"], 1)      # declared-unmeasured keeps its own word
        self.assertEqual(c["by_gate"]["ungated"], 1)               # no stamp at all is a THIRD state
        self.assertEqual(c["by_source"]["unstamped"], 1)           # never silently called "unattributed"
        self.assertEqual(c["by_door"]["unnamed"], 2)

    def test_body_concentration_is_reported_because_rows_are_not_independent(self):
        d = self._bank(skills=[{"gene_id": "dog"} for _ in range(9)] + [{"gene_id": "cat"}], provenance=[])
        out = moat_panel(d)
        self.assertEqual(out["bank"]["bodies"]["distinct"], 2)
        self.assertEqual(out["bank"]["bodies"]["largest_share_body"], "dog")
        self.assertAlmostEqual(out["bank"]["bodies"]["largest_share_fraction"], 0.9)
        self.assertTrue(any("pseudo-replicated" in n for n in out["notes"]))

    def test_a_robot_with_no_recorded_recall_says_so_rather_than_showing_an_empty_table(self):
        d = self._bank(skills=[{}], provenance=[{"child_id": "someone_else", "delta": 1.0}])
        pkg = d / "pkg"
        (pkg / "robot").mkdir(parents=True)
        (pkg / "robot" / "robot_genome.json").write_text(json.dumps({"id": "genome_untouched_v"}), encoding="utf-8")
        tb = moat_panel(d, package_dir=pkg)["this_build"]
        self.assertFalse(tb["matched"])
        self.assertIn("No recall event is recorded", tb["summary"])
        self.assertEqual(tb["events"], [])

    def test_a_package_id_is_peeled_back_to_the_id_the_ledger_recorded(self):
        self.assertEqual(_gene_id_candidates("genome_built_quadruped_18seg_v_v"),
                         ["genome_built_quadruped_18seg_v_v", "built_quadruped_18seg_v_v",
                          "built_quadruped_18seg_v", "built_quadruped_18seg"])

    def test_one_bodys_recall_events_are_never_credited_to_a_similarly_named_body(self):
        """Both ids below exist in the live ledger. A prefix match would hand `anatomy_creature`'s wins to
        `anatomy_creature_91b931bf`, i.e. the panel would fabricate the evidence it exists to audit."""
        d = self._bank(skills=[{}], provenance=[
            {"child_id": "anatomy_creature", "delta": 5.0, "meta": {"selected": "hint"}},
            {"child_id": "anatomy_creature_91b931bf", "delta": -1.0, "meta": {"selected": "default"}},
        ])
        pkg = d / "pkg"
        (pkg / "robot").mkdir(parents=True)
        (pkg / "robot" / "robot_genome.json").write_text(
            json.dumps({"id": "genome_anatomy_creature_91b931bf"}), encoding="utf-8")
        tb = moat_panel(d, package_dir=pkg)["this_build"]
        self.assertTrue(tb["matched"])
        self.assertEqual([e["gene_id"] for e in tb["events"]], ["anatomy_creature_91b931bf"])
        self.assertAlmostEqual(tb["mean_delta_m"], -1.0)          # not +5.0, and not +2.0

    def test_reading_the_panel_cannot_modify_the_bank(self):
        """Opened read-only on purpose: a UI poll must not migrate or touch the developer's real memory."""
        d = self._bank(skills=[{"gene_id": "a"}], provenance=[{"delta": 0.1}])
        db = d / "virturoid_memory.db"
        before = (db.stat().st_size, db.stat().st_mtime_ns, db.read_bytes())
        for _ in range(3):
            moat_panel(d)
        after = (db.stat().st_size, db.stat().st_mtime_ns, db.read_bytes())
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[2], after[2], "reading the moat panel rewrote the database file")
        # and no sidecar journal survived the read
        self.assertFalse((d / "virturoid_memory.db-wal").exists())
        self.assertFalse((d / "virturoid_memory.db-journal").exists())

    def test_a_bank_with_no_provenance_table_still_renders(self):
        """An older bank predates the ledger entirely. That is an empty recall section, not a 500."""
        d = Path(__import__("tempfile").mkdtemp(prefix="moat-old-"))
        con = sqlite3.connect(d / "virturoid_memory.db")
        con.execute("CREATE TABLE skills (id INTEGER PRIMARY KEY, skill_id TEXT, robot_class TEXT, "
                    "species TEXT, task_type TEXT, gene_id TEXT, base_config TEXT)")
        con.execute("INSERT INTO skills (task_type, base_config) VALUES ('locomotion', '{}')")
        con.commit()
        con.close()
        out = moat_panel(d)
        self.assertTrue(out["db_present"])
        self.assertEqual(out["bank"]["rows"], 1)
        self.assertEqual(out["recall"]["kinds"], [])
        self.assertIn("Nothing has been recalled", out["recall"]["headline"])

    def test_recall_ledger_orders_by_edge_count_so_the_dominant_kind_is_first(self):
        d = self._bank(skills=[], provenance=(
            [{"kind": "small", "delta": 9.0}] + [{"kind": "big", "delta": -0.1} for _ in range(5)]))
        con = sqlite3.connect(d / "virturoid_memory.db")
        con.row_factory = sqlite3.Row
        led = recall_ledger(con)
        con.close()
        self.assertEqual(led["dominant_kind"], "big")
        self.assertEqual(led["kinds"][0]["kind"], "big")


class MoatRouteTests(unittest.TestCase):
    """The Studio route. The read model being right is not the same as the product reaching it -- the moat was
    invisible for months while every number below already existed in the database."""

    def _serve(self, build_root: Path, query: str = "") -> dict:
        import threading
        from urllib.request import urlopen

        from virturoid.ui_server import create_server
        server = create_server("127.0.0.1", 0, build_root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            raw = urlopen(f"http://{host}:{port}/api/moat{query}", timeout=10).read().decode("utf-8")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        return json.loads(raw)

    def test_the_route_reads_the_build_roots_own_memory_directory(self):
        from tempfile import mkdtemp
        root = Path(mkdtemp(prefix="moat-root-"))
        _make_bank(root / "memory" / "virturoid_memory.db",
                   skills=[{"gene_id": "dog", "base_config": {"bank_gate": "fragility_v1"}}],
                   provenance=[{"kind": "gait_hint_deploy", "delta": -0.5}])
        out = self._serve(root)
        self.assertTrue(out["db_present"])
        self.assertEqual(out["bank"]["rows"], 1)
        self.assertEqual(out["recall"]["kinds"][0]["direction"], "hurts")
        self.assertIn("BEHIND", out["recall"]["headline"])

    def test_a_build_root_with_no_bank_answers_honestly_instead_of_erroring(self):
        from tempfile import mkdtemp
        out = self._serve(Path(mkdtemp(prefix="moat-empty-")))
        self.assertFalse(out["db_present"])
        self.assertNotIn("error", out)
        self.assertEqual(out["bank"]["rows"], 0)

    def test_a_package_name_cannot_escape_the_build_root(self):
        from tempfile import mkdtemp
        root = Path(mkdtemp(prefix="moat-esc-"))
        _make_bank(root / "memory" / "virturoid_memory.db", skills=[{}], provenance=[])
        out = self._serve(root, "?package=..%2F..%2Fetc")
        self.assertTrue(out["db_present"])
        self.assertFalse(out["this_build"]["matched"])


if __name__ == "__main__":
    unittest.main()
