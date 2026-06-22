"""Test session setup.

Keep the suite hermetic: never read the developer's project-local ``.env`` during
tests, so a configured OpenAI/Claude/local backend can't leak into the offline,
deterministic test expectations. Set before any virturoid import resolves a backend.
"""

import os

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
