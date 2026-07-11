"""VIRT-Bench-Transfer CLI (P6): print the standing distance-predicts-transfer scorecard and exit non-zero if the
regression gate fails — wire into CI as `python scripts/embedding_scorecard.py`. Uses the cached transfer corpus
(build/data/transfer_corpus.json) by default, or the committed fixture with --fixture for a fast, sim-free run.

  python scripts/embedding_scorecard.py            # cached corpus (built by embedding_transfer_corpus)
  python scripts/embedding_scorecard.py --fixture  # committed physics-real fixture (no MuJoCo)
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.embedding_scorecard import embedding_transfer_scorecard  # noqa: E402


def main() -> int:
    corpus = None
    if "--fixture" in sys.argv:
        corpus = Path("tests/fixtures/transfer_corpus_fixture.json")
    sc = embedding_transfer_scorecard(corpus)
    print(json.dumps(sc, indent=2))
    print("\n" + sc.get("headline", sc.get("note", "")))
    return 0 if sc.get("gate_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
