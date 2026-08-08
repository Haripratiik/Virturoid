"""THE SWEEP: no destination may read like a constant and be a function of the process's CWD.

WHY THIS TEST EXISTS -- DO NOT DELETE IT AS PEDANTRY. One bug shape shipped THREE TIMES in one week,
and every instance was found by accident, after it had already done damage:

  1. ``memory_db.DEFAULT_DB_PATH = Path("build")/"memory"/...`` -- the test suite wrote the developer's
     REAL gait bank, growing it 97 -> 101 locomotion rows, four of them classed
     ``totally_made_up_xyz`` from ``tests/test_structural_dispatch.py``. The bank is gitignored, has no
     backup, and rows are never deleted, so fixture rows became permanent evidence in every later
     analysis of whether the flywheel has signal.                                     (fixed f77d092)
  2. ``agent_tools.safe_build_path(None, "memory")`` -> ``<cwd>/build/memory`` -- a SECOND default rule
     for the same destination that never read ``VIRTUROID_MEMORY_DIR``. 231 ``runs`` rows leaked AFTER
     incident 1's fix had been checked and declared verified end to end; the table that was checked was
     the one that had been fixed.                                                     (fixed f77d092)
  3. ``ui_server._resolve_build_root`` kept a bare relative ``build/ui_verify`` candidate, so the same
     install showed DIFFERENT ROBOTS from different directories. Caught only because an agent wrote a
     test it had not yet satisfied.                                                   (fixed 248afca)

The shape: a module-level constant, class attribute or default argument holding a relative path that is
later resolved against whatever directory the process happens to have been started in. It never raises.
It writes to the wrong place, or reads nothing and reports "not found" as "does not exist" -- and the
product degrades silently, which is why all three shipped.

The 2026-08-08 sweep that added this test found FOUR more (``session_state`` -- the cross-process session
store, which is the worst possible one to have here; ``mcp_server._ALLOWED_READ_ROOTS`` -- a security
confinement boundary re-deriving three roots other modules own; the policy bank ``models/``, which is
tracked in git; and five module-level ``build/...`` moat artifacts that silently fall back to unlearned
defaults). Assume there will be more. This test is the thing that makes the next one fail loudly.

WHAT COUNTS AS AN OFFENDER, AND WHAT DOES NOT
  * OFFENDER: a relative literal whose first segment is a directory this repo WRITES to
    (``build/``, ``models/``, ``dist/``, ``logs/``...), in a constant-like position.
  * NOT an offender: a URI *inside a built robot package* (``reports/spec_sheet.json``,
    ``simulation/scene_set.json``, ``training/...``). Those are joined to a package root by their caller
    and are portable by design -- that is what makes an exported package relocatable.
  * NOT an offender: an ``argparse`` default. The user sees it in ``--help`` and can override it, and
    ``argparse`` defaults live inside functions, not in constant-like positions, so they never match.

THE FIX when this test fails: import ``services.install_paths`` and wrap the literal --
``anchored("build/thing")`` for an install-owned destination, ``policy_bank_dir(...)`` for the policy
bank -- or, better, ASK THE MODULE THAT OWNS THE DESTINATION instead of re-deriving it. If the path is
legitimately relative, add it to ``ALLOWED`` below WITH THE REASON, so the next sweep does not re-flag it.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "virturoid"

#: Trees the check scans. ``scripts/`` is in scope because incident 3's destination -- the GIT-TRACKED
#: ``build/ui_verify`` demo set -- is regenerated from there: ``rebake_demo_set.py`` run from the wrong
#: directory silently rebaked a stray tree while ``git status`` stayed clean, which reads as "already up to
#: date". Underscore-prefixed scratch scripts (``scripts/_*.py``) are one-off probes, not shipped surface,
#: and are skipped -- they are also the only files where a bare ``Path("build/...")`` is honest.
SCAN_ROOTS = (SRC, REPO / "scripts")


def rel_to_abs(rel: str) -> Path:
    """A reported key back to a real file: bare names live under src/virturoid, 'scripts/...' at the repo root."""
    return (REPO / rel) if rel.startswith("scripts/") else (SRC / rel)


def _scan_files():
    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or path.name.startswith("_"):
                continue
            yield path

#: First path segments that name a directory this repo WRITES to. A relative literal starting with one of
#: these is resolved against the CWD by whoever opens it. Package-internal URI roots (``reports``,
#: ``simulation``, ``training``, ``software``, ``datasets``, ``robot``, ``cad``) are deliberately absent.
DEST_ROOTS = frozenset({"build", "dist", "models", "logs", "out", "cache", "site", "webui"})

#: Sites that are relative ON PURPOSE. Key: "<path>:<name>". Value: the reason, which must be a real one.
#: Adding an entry here is a claim you are making; make it true.
ALLOWED: dict[str, str] = {
    "services/agent_tools.py:safe_build_path": (
        "The confinement root for AGENT-SUPPLIED dir args, and the one place anchoring makes things worse: "
        "memory_db classifies destinations against conventional_memory_dir(), pinned to the INITIAL cwd, so "
        "an anchored safe_build_path would start handing the chokepoint the real bank as an 'explicit' "
        "destination from any other directory. Reasoned out at the site."),
    "services/llm_client.py:_load_local_env": (
        "Searches BOTH the repo root and Path.cwd() for a .env, on purpose -- 'the project I am standing in' "
        "is exactly the intended semantics for a dotenv, and the repo root is already checked first."),
    "services/memory_db.py:_INITIAL_CWD": (
        "Captures the process's initial cwd DELIBERATELY, as the definition of 'the bank under the older "
        "rule'. It is the input to the destination policy, not a destination."),
    "services/install_paths.py:policy_bank_dir": (
        "The anchoring helper itself: its default argument is the conventional relative name it exists to "
        "resolve."),
    "scripts/bank_grasp_skill.py:CARD": (
        "'build/skills/grasp_tabletop_arm.npz' is DATA inside a skill card -- a record of where the artifact "
        "lives ON THE GPU BOX where warm-start runs, not a path this process opens. Anchoring it to this "
        "checkout would make the recorded location wrong."),
    "services/urdf_exporter.py:robot_genome_to_urdf": (
        "'../cad/mesh/visual' is a URDF-INTERNAL mesh reference, resolved by the consumer relative to the "
        "URDF file, not by us relative to a CWD. Making it absolute would break the exported package the "
        "moment the customer moved it -- the same class as the reports/ and simulation/ package URIs."),
}

#: The POLICY BANK is exempt as a family, not site by site. ``models_dir="models"`` is the conventional NAME
#: of the bank; every place that turns that name into a real directory goes through
#: ``install_paths.policy_bank_dir``, which anchors it. Twelve near-identical allowlist entries would hide the
#: one that mattered, so the exemption is backed by ``test_policy_bank_resolution_stays_centralised`` below --
#: if anyone reintroduces a second resolution rule, THAT test fails, which is the failure we actually want.
BANK_ARGS = ("models_dir",)

#: The MEMORY bank is exempt for the OPPOSITE reason, and this one is counter-intuitive enough to be worth
#: measuring rather than asserting. ``memory_dir="build/memory"`` flows into ``MemoryDB.__init__``, which is the
#: chokepoint incident 2 installed: it recognises the conventional bank dir and REWRITES it to
#: ``VIRTUROID_MEMORY_DIR`` when a redirect is in force. Anchoring these defaults would hand the chokepoint an
#: absolute path it can only classify as an EXPLICIT destination -- which bypasses the redirect entirely.
#: MEASURED 2026-08-08, cwd outside the checkout, VIRTUROID_MEMORY_DIR pointed at a scratch dir:
#:     MemoryDB("build/memory/virturoid_memory.db")            -> <scratch>/virturoid_memory.db   (redirected)
#:     MemoryDB("<checkout>/build/memory/virturoid_memory.db") -> <checkout>/build/memory/...     (REAL BANK)
#: So "just anchor it" here re-creates incident 1 -- the suite writing the developer's real bank. The relative
#: literal is what KEEPS the redirect working. Leave it.
MEMORY_ARGS = ("memory_dir",)


#: Names that make an expression ABSOLUTE, so a relative segment inside it is just a segment.
#: ``Path(__file__).parents[2] / "webui"`` is anchored to the module, not to the CWD, and must not be flagged.
ANCHORS = ("__file__", "anchored(", "policy_bank_dir(", "checkout_root()", "default_memory_dir(",
           "default_build_root(", "_INITIAL_CWD", "REPO_ROOT", "_REPO_ROOT", "_PKG",
           "ROOT /")   # scripts/overhaul*: ROOT = Path(__file__).resolve().parents[2]


def _literals(node: ast.AST, text: str) -> list[str]:
    """Every string constant inside an expression -- catches ``Path("build") / "x"`` as well as "build/x".

    Returns nothing when the expression is rooted in an ABSOLUTE anchor: the whole point of this test is
    paths resolved against the CWD, and a segment of an ``__file__``-anchored expression is not one.
    """
    try:
        src = ast.get_source_segment(text, node) or ""
    except Exception:                                        # noqa: BLE001 - fall through to flagging
        src = ""
    if any(a in src for a in ANCHORS):
        return []
    keys = {id(k) for n in ast.walk(node) if isinstance(n, ast.Dict) for k in n.keys if k is not None}
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in keys]


#: A name is path-like if it says so. Without this, the single word "out" in an aim-direction vocabulary and
#: "build" in a stop-word list read as destinations -- ``DEST_ROOTS`` members are ordinary English words too.
_PATHY = ("_dir", "_path", "_root", "_pool", "_file", "_cache", "prefix", "uri")


def _path_like_name(name: str) -> bool:
    n = name.lower().split("(")[-1].rstrip(")").strip() or name.lower()
    return any(k in n for k in _PATHY) or any(k in name.lower() for k in _PATHY)


def _offending(lits: list[str], name: str) -> str | None:
    for s in lits:
        if not s or len(s) > 200 or s.startswith(("http", "urn:", "mailto:", "/", "\\")):
            continue
        if len(s) > 1 and s[1] == ":":                       # absolute Windows path
            continue
        if s.startswith(("./", ".\\", "../", "..\\")):
            return s
        parts = s.replace("\\", "/").split("/")
        if parts[0] not in DEST_ROOTS:
            continue
        # A multi-segment literal ("build/memory") is unambiguously a path. A BARE root word ("models",
        # "build") only counts when the thing holding it is named like a path.
        if len(parts) > 1 or _path_like_name(name):
            return s
    return None


def _constant_positions(tree: ast.Module):
    """Yield (name, lineno, value_node) for module constants, class attributes and default arguments."""
    for node in tree.body:                                    # module-level assignment
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                if isinstance(t, ast.Name):
                    yield t.id, node.lineno, node.value
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):                    # class attribute
            for st in node.body:
                if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
                    tgts = st.targets if isinstance(st, ast.Assign) else [st.target]
                    for t in tgts:
                        if isinstance(t, ast.Name):
                            yield f"{node.name}.{t.id}", st.lineno, st.value
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):   # default argument
            a = node.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults))
            pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, d in pairs:
                yield f"{node.name}({arg.arg})", node.lineno, d


def _string_lines(tree: ast.Module) -> set[int]:
    """Line numbers occupied by string literals — docstrings EXPLAIN this bug, they must not trip on it."""
    out: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.update(range(n.lineno, (getattr(n, "end_lineno", None) or n.lineno) + 1))
    return out


def _cwd_calls(text: str, tree: ast.Module):
    skip = _string_lines(tree)
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or i in skip:
            continue
        if "Path.cwd()" in line or "os.getcwd()" in line:
            yield i, stripped


class NoCwdRelativeDestinations(unittest.TestCase):

    def _scan(self):
        """Return [(rel_path, key, lineno, literal, kind)] for every unallowed offender under src/virturoid."""
        found = []
        for path in _scan_files():
            rel = path.relative_to(REPO).as_posix().replace("src/virturoid/", "")
            text = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=str(path))

            for name, lineno, value in _constant_positions(tree):
                lit = _offending(_literals(value, text), name)
                arg = name[name.find("("):]
                if lit == "models" and any(f"({a})" == arg for a in BANK_ARGS):
                    continue                                  # policy bank: resolved centrally, see BANK_ARGS
                if lit == "build/memory" and any(f"({a})" == arg for a in MEMORY_ARGS):
                    continue                                  # memory bank: relative KEEPS the redirect working
                if lit and f"{rel}:{name}" not in ALLOWED and f"{rel}:{name.split('(')[0]}" not in ALLOWED:
                    found.append((rel, name, lineno, lit, "relative destination literal"))

            for lineno, src_line in _cwd_calls(text, tree):
                fn = _enclosing(tree, lineno)
                if f"{rel}:{fn}" in ALLOWED or f"{rel}:{_module_const_at(tree, lineno)}" in ALLOWED:
                    continue
                found.append((rel, fn, lineno, src_line, "CWD-resolved path"))
        return found

    def test_no_new_cwd_relative_destination(self):
        found = self._scan()
        if not found:
            return
        lines = ["", "A path that reads like a constant is a function of the process's working directory.",
                 "This shipped three times (memory bank 97->101 rows, 231 leaked runs rows, the Robot Library",
                 "showing different robots from different directories). Read this test's docstring.", ""]
        for rel, name, lineno, lit, kind in found:
            lines += [
                f"  {rel_to_abs(rel)}:{lineno}",
                f"      {name}  ->  {kind}: {lit!r}",
                f"      FIX: from virturoid.services.install_paths import anchored"
                f"  ->  anchored({lit!r})",
                "      OR ask the module that owns this destination for it instead of re-deriving it.",
                "      OR, if it is legitimately relative, add"
                f' "{rel}:{name.split("(")[0]}" to ALLOWED with the reason.',
                "",
            ]
        self.fail("\n".join(lines))

    def test_the_three_originals_stay_fixed(self):
        """Regression pins for the incidents themselves — the BEHAVIOUR, not the spelling.

        For the bank this deliberately does NOT assert "the default is absolute". It is not, and it must not
        be: ``memory_db`` arbitrates by comparing against the conventional relative dir, and an absolute
        default would be classified as an explicit destination and bypass the redirect. The contract that
        actually failed in incidents 1 and 2 is *"does the redirect reach every writer"*, so that is what is
        pinned here — including through the ``safe_build_path`` spelling that was the second rule.
        """
        import os
        import tempfile
        from virturoid.services import memory_db, session_state
        from virturoid import ui_server

        # 1 + 2: EVERY spelling of the bank's location honours VIRTUROID_MEMORY_DIR.
        redirect = tempfile.mkdtemp(prefix="sweep_bank_")
        prev = os.environ.get("VIRTUROID_MEMORY_DIR")
        try:
            os.environ["VIRTUROID_MEMORY_DIR"] = redirect
            self.assertEqual(Path(memory_db.default_memory_dir()).resolve(), Path(redirect).resolve(),
                             "default_memory_dir() stopped reading VIRTUROID_MEMORY_DIR (incident 1)")
            for spelling in ("build/memory/virturoid_memory.db",                     # the literal default
                             str(memory_db.conventional_memory_dir() / "x.db")):     # safe_build_path's answer
                got = Path(memory_db.MemoryDB(spelling).path).resolve()
                self.assertEqual(got.parent, Path(redirect).resolve(),
                                 f"MemoryDB({spelling!r}) escaped the redirect — that is incident 2's shape: "
                                 f"a second rule for a destination that already has one")
        finally:
            if prev is None:
                os.environ.pop("VIRTUROID_MEMORY_DIR", None)
            else:
                os.environ["VIRTUROID_MEMORY_DIR"] = prev
        # 3: the build root and the demo-set fallback are properties of the install.
        self.assertTrue(Path(ui_server.default_build_root()).is_absolute(),
                        "ui_server.default_build_root() is CWD-relative again (incident 3)")
        # the 2026-08-08 sweep's worst find: the cross-process session store.
        self.assertTrue(Path(session_state._dir()).is_absolute(),
                        "session_state._dir() is CWD-relative again — the MCP server and the viewer are "
                        "different processes and will silently disagree about which robots exist")

    def test_policy_bank_resolution_stays_centralised(self):
        """``models_dir`` is exempt from the literal check ONLY because ONE function resolves it.

        This is the test that earns that exemption. Incident 2 was a SECOND rule for a destination that
        already had one; the defence against a third is not to police the literal (there are twelve of them
        and they are all just the bank's conventional name) but to police the number of places that turn the
        name into a directory. Every leaf must go through ``install_paths.policy_bank_dir``.
        """
        offenders = []
        for path in _scan_files():
            if path.name == "install_paths.py":
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if "Path(models_dir)" in line and not line.strip().startswith("#"):
                    offenders.append(f"{path.as_posix()}:{i}: {line.strip()}")
        self.assertEqual(offenders, [], "\n".join([
            "", "A SECOND rule for resolving the policy bank has appeared. That is incident 2's shape.",
            "``Path(models_dir)`` resolves against the CWD; use policy_bank_dir(models_dir) instead.", "",
            *offenders, ""]))

    def test_allowlist_entries_are_real(self):
        """An ALLOWED key must point at something that still exists, so the list cannot rot into cover."""
        for key in ALLOWED:
            rel, _, name = key.partition(":")
            path = rel_to_abs(rel)
            self.assertTrue(path.is_file(), f"ALLOWED names a file that no longer exists: {key}")
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIn(name.split(".")[-1], text, f"ALLOWED names a symbol that no longer exists: {key}")
            self.assertTrue(len(ALLOWED[key]) > 40, f"ALLOWED entry needs a real reason, not a shrug: {key}")


def _enclosing(tree: ast.Module, lineno: int) -> str:
    """Innermost function/class containing ``lineno``, for a readable failure message."""
    best, best_span = "<module>", None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno <= lineno <= end and (best_span is None or (end - node.lineno) < best_span):
                best, best_span = node.name, end - node.lineno
    return best


def _module_const_at(tree: ast.Module, lineno: int) -> str:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno == lineno:
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in tgts:
                if isinstance(t, ast.Name):
                    return t.id
    return ""


if __name__ == "__main__":
    unittest.main()
