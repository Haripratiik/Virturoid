"""The GPU end-to-end validator. The real walk-gate proof is the live run (scripts/validate_gpu_training.py on
the box); here we certify the no-GPU paths so CI stays green without the external box: it skips gracefully when
the box is unreachable, and exits 1 (not crash) when training returns no policy."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_validator_skips_gracefully_without_a_gpu_box(monkeypatch):
    import validate_gpu_training as v
    import virturoid.services.gpu_trainer as gt

    monkeypatch.setattr(gt, "gpu_available", lambda **kw: False)
    assert v.main() == 2                       # box unreachable -> graceful skip (exit 2), never a crash


def test_remote_policy_read_retries_until_the_npz_flushes():
    import io
    import numpy as np

    from virturoid.services.gpu_trainer import _read_complete_npz

    buf = io.BytesIO()
    np.savez(buf, meta=np.asarray([1, 2]), weights=np.asarray([0.1, 0.2]))
    responses = iter([b"too-short", buf.getvalue()])
    payload = _read_complete_npz(lambda: next(responses), attempts=2, wait_s=0)
    assert payload == buf.getvalue()


def test_validator_reports_failure_when_training_returns_no_policy(monkeypatch):
    import validate_gpu_training as v
    import virturoid.services.gpu_trainer as gt

    monkeypatch.setattr(gt, "gpu_available", lambda **kw: True)
    monkeypatch.setattr(gt, "train_gene_on_gpu", lambda *a, **kw: None)   # box stalled / launch failed
    pytest.importorskip("mujoco")              # compose_robot needs it
    import os
    os.environ.setdefault("VIRTUROID_LLM_BACKEND", "off")
    assert v.main() == 1                        # trained-but-nothing -> honest failure, not a false PASS
