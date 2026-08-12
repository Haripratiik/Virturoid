"""The customer's source folder is READ-ONLY to us, and that is enforced rather than intended.

WHY THIS MODULE EXISTS. An ingest of a real Menagerie Unitree Go2 wrote **two files into the customer's
own directory**, measured through ``agent_tools.call_tool`` on
``~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2``::

    go2.__mjcfprep__.xml    model_import.import_model      a prep copy, compiled then unlinked
    ingestion_report.json   input_training_tools           our report, left behind

Only the second one shows up in a before/after mtime diff, because the first is deleted on the way out.
That is exactly why an mtime diff is not the check: a write that cleans up after itself is still a write
into a folder that is not ours, and it still fails on a read-only mount, a network share, a git worktree
the customer expects to stay clean, or a CI checkout with a dirty-tree gate.

Neither write was malicious and neither was careless -- both had a real reason (MuJoCo resolves
``<include>``, ``meshdir`` and every asset relative to the XML's OWN directory, so a modified copy has to
sit beside the original to load; and a report is most useful next to the thing it describes). The reasons
are why "just remember not to" does not hold: the next person with a good reason writes there too. So the
rule is enforced at the boundary, where a violation cannot be forgotten:

    with source_guard.read_only(project_dir) as guard:
        ...                                   # the whole ingest
    guard.blocked                             # every write we ATTEMPTED, with its traceback

WHAT IT COVERS. Every Python-level path that can create, truncate, append to, rename, delete or even
re-STAMP a file (``os.utime`` -- ``shutil.copy2`` changes mtimes without writing bytes, and an mtime-based
check is precisely what would miss it). ``builtins.open`` and ``io.open`` are patched separately because
they are two names for one function object and rebinding one does not rebind the other -- ``pathlib``
reaches for ``io.open``, which is how the MJCF prep write got in.

WHAT IT DOES NOT COVER, stated because a guarantee with an unstated hole is worse than no guarantee: a
write issued from C (a native library calling ``fopen`` directly) never passes through Python and cannot
be seen here. MuJoCo only ever READS the customer's files, and its one writer (``mj_saveLastXML``) is
handed a path we choose, so nothing in the ingest path relies on the hole; a native writer added later
would need its own containment.

REFUSE, DO NOT REDIRECT. A blocked write raises ``PermissionError``, and callers that treat their write as
best-effort (the report writer did) would swallow it silently -- so the attempt is RECORDED and the ingest
result names it. A guard whose violations are invisible is a guard nobody knows has fired.
"""
from __future__ import annotations

import builtins
import io
import os
import shutil
import threading
import traceback
from pathlib import Path

_LOCK = threading.RLock()
#: ``_ACTIVE``: normcased absolute root -> its violation ledger.
#: ``_HOLDS``:  the same root -> how many live guards are holding it.
#:
#: The count is per ROOT, not just global: two nested guards on the SAME folder (an ingest that re-enters,
#: or a sub-project ingest under a parent already being ingested) share one ledger, and the inner one exiting
#: must not un-protect the folder the outer one is still inside. The global `_DEPTH` alone would keep the
#: patches installed while `_ACTIVE` had already forgotten the root — armed, and guarding nothing.
_ACTIVE: dict[str, list] = {}
_HOLDS: dict[str, int] = {}
_DEPTH = {"n": 0}
_ORIGINALS: dict[str, object] = {}

#: ``open``/``io.open`` modes that can modify a file. ``r`` alone is the only safe one; ``r+`` is not.
_WRITE_CHARS = frozenset("wxa+")


def _norm(path) -> str | None:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError):
        return None


def _protecting(path) -> list | None:
    """The violation ledger guarding ``path``, or None. Fast no-op when nothing is armed."""
    if not _ACTIVE:
        return None
    p = _norm(path)
    if p is None:
        return None
    for root, ledger in _ACTIVE.items():
        # `p == root` catches an attempt to replace/remove the directory itself; the sep guards against
        # `.../unitree_go2_backup` matching the root `.../unitree_go2`.
        if p == root or p.startswith(root + os.sep):
            return ledger
    return None


class SourceWriteBlocked(PermissionError):
    """A write into the customer's source folder. Not an error in their project — one in ours."""


def _refuse(ledger: list, op: str, path) -> None:
    entry = {"operation": op, "path": str(path),
             # the frame that ASKED for the write, not this module's own plumbing
             "called_from": [ln.strip() for ln in traceback.format_stack()[:-2]
                             if "virturoid" in ln and "source_guard.py" not in ln][-3:]}
    ledger.append(entry)
    raise SourceWriteBlocked(
        f"refusing to {op} inside the customer's source folder: {path}. That directory is READ-ONLY to "
        f"Virturoid — a folder a customer points us at is theirs, and our artifacts belong under build/. "
        f"Write to a staging directory (source_guard.staging_dir) instead.")


def _guarded_open(original, label):
    def opener(file, mode="r", *args, **kwargs):
        if _WRITE_CHARS & set(str(mode)):
            ledger = _protecting(file)
            if ledger is not None:
                _refuse(ledger, label, file)
        return original(file, mode, *args, **kwargs)
    return opener


def _guarded_os_open(original):
    writing = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def opener(path, flags, *args, **kwargs):
        if flags & writing:
            ledger = _protecting(path)
            if ledger is not None:
                _refuse(ledger, "os.open(write)", path)
        return original(path, flags, *args, **kwargs)
    return opener


def _guarded_one(original, label):
    def fn(path, *args, **kwargs):
        ledger = _protecting(path)
        if ledger is not None:
            _refuse(ledger, label, path)
        return original(path, *args, **kwargs)
    return fn


def _guarded_two(original, label):
    """Two-path operations (rename/replace/copy/move): EITHER end inside the root is a violation.

    The source end counts because ``os.rename(src, elsewhere)`` removes the customer's file just as
    surely as deleting it.
    """
    def fn(src, dst, *args, **kwargs):
        for candidate in (dst, src):
            ledger = _protecting(candidate)
            if ledger is not None:
                _refuse(ledger, label, candidate)
        return original(src, dst, *args, **kwargs)
    return fn


#: (module, attribute, wrapper-factory). ``builtins.open`` and ``io.open`` are the SAME function object
#: reached through two names — patching one leaves the other live, and `pathlib` uses `io.open`.
_PATCHES = (
    (builtins, "open", lambda f: _guarded_open(f, "open(write)")),
    (io, "open", lambda f: _guarded_open(f, "io.open(write)")),
    (os, "open", _guarded_os_open),
    (os, "mkdir", lambda f: _guarded_one(f, "os.mkdir")),
    (os, "makedirs", lambda f: _guarded_one(f, "os.makedirs")),
    (os, "remove", lambda f: _guarded_one(f, "os.remove")),
    (os, "unlink", lambda f: _guarded_one(f, "os.unlink")),
    (os, "rmdir", lambda f: _guarded_one(f, "os.rmdir")),
    (os, "truncate", lambda f: _guarded_one(f, "os.truncate")),
    # mtime/permission stamps change the folder WITHOUT writing bytes — and an mtime diff is the very
    # check that would miss `shutil.copy2` restamping a file it copied out.
    (os, "utime", lambda f: _guarded_one(f, "os.utime")),
    (os, "chmod", lambda f: _guarded_one(f, "os.chmod")),
    (os, "rename", lambda f: _guarded_two(f, "os.rename")),
    (os, "replace", lambda f: _guarded_two(f, "os.replace")),
    (shutil, "rmtree", lambda f: _guarded_one(f, "shutil.rmtree")),
    (shutil, "move", lambda f: _guarded_two(f, "shutil.move")),
    (shutil, "copytree", lambda f: _guarded_two(f, "shutil.copytree")),
)


class Guard:
    """Handle on one protected root. ``blocked`` is the list of writes we tried and were refused."""

    __slots__ = ("root", "blocked")

    def __init__(self, root: str, blocked: list):
        self.root, self.blocked = root, blocked

    def report(self) -> dict | None:
        """The disclosure an ingest result carries — None when we behaved, which is the normal case."""
        if not self.blocked:
            return None
        return {
            "source_folder": self.root,
            "n_blocked_writes": len(self.blocked),
            "attempts": self.blocked[:10],
            "what_this_means": "Virturoid tried to write inside YOUR project folder and was stopped. Your "
                               "files are unchanged. This is a defect in Virturoid, not in your project — "
                               "please report it; our artifacts belong under build/.",
        }


class _Disarmed:
    """The context manager's null form: nothing to protect, so nothing is patched."""

    __slots__ = ("root", "blocked")

    def __init__(self):
        self.root, self.blocked = None, []

    def report(self):
        return None


def read_only(root):
    """Make ``root`` read-only to this process for the duration of the ``with`` block.

    Nesting is reference-counted, so an inner ingest does not un-patch the outer one. A root that does
    not exist (a .zip path, a bare description ingest) disarms rather than pretending to protect.
    """
    return _ReadOnly(root)


class _ReadOnly:
    __slots__ = ("_root", "_guard", "_armed")

    def __init__(self, root):
        # A directory protects its whole subtree; a FILE (a customer's .zip, a bare model path) protects
        # exactly itself -- protecting its parent instead would claim their Downloads folder.
        self._root = _norm(root) if root and os.path.exists(str(root)) else None
        self._guard = None
        self._armed = False

    def __enter__(self):
        if self._root is None:
            self._guard = _Disarmed()
            return self._guard
        with _LOCK:
            ledger = _ACTIVE.get(self._root)
            if ledger is None:
                ledger = _ACTIVE[self._root] = []
            _HOLDS[self._root] = _HOLDS.get(self._root, 0) + 1
            self._armed = True
            if _DEPTH["n"] == 0:
                for module, name, wrap in _PATCHES:
                    original = getattr(module, name)
                    _ORIGINALS[f"{module.__name__}.{name}"] = original
                    setattr(module, name, wrap(original))
            _DEPTH["n"] += 1
        self._guard = Guard(self._root, ledger)
        return self._guard

    def __exit__(self, *exc):
        if not self._armed:
            return False
        with _LOCK:
            _DEPTH["n"] -= 1
            # Only the LAST holder of this root releases it. Popping unconditionally would leave an outer
            # guard nominally armed over a folder `_protecting` no longer recognises.
            remaining = _HOLDS.get(self._root, 1) - 1
            if remaining > 0:
                _HOLDS[self._root] = remaining
            else:
                _HOLDS.pop(self._root, None)
                _ACTIVE.pop(self._root, None)
            if _DEPTH["n"] <= 0:
                _DEPTH["n"] = 0
                _ACTIVE.clear()
                _HOLDS.clear()
                for module, name, _ in _PATCHES:
                    original = _ORIGINALS.pop(f"{module.__name__}.{name}", None)
                    if original is not None:
                        setattr(module, name, original)
        return False


def staging_dir(source_root, *, kind: str = "ingest") -> Path:
    """Where a prep artifact for ``source_root`` goes instead: ``build/<kind>/<name>-<digest>/``.

    Keyed on a digest of the ABSOLUTE source path, so two folders with the same basename (every customer
    has a ``robot/``) never share a staging directory and the second import cannot read the first's
    prepped file. Anchored to the install, so ``cd`` cannot move it — the rule ``install_paths`` owns.
    """
    import hashlib

    from virturoid.services.install_paths import anchored
    src = str(source_root or "")
    key = _norm(src) or src
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    name = (Path(src).name or "project")[:40]
    out = Path(anchored(os.path.join("build", kind))) / f"{name}-{digest}"
    # Created through the ORIGINAL makedirs: `out` is under build/, but if a caller ever aims a staging
    # dir at the source root we want the guard to see it, so no bypass is granted here.
    out.mkdir(parents=True, exist_ok=True)
    return out
