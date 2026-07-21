"""Test session setup.

Keep the suite hermetic: never read the developer's project-local ``.env`` during
tests, so a configured OpenAI/Claude/local backend can't leak into the offline,
deterministic test expectations. Set before any virturoid import resolves a backend.
"""

import os

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
# Production builds are LLM-first and fail closed when no design model is
# configured.  Tests exercise the explicitly supported offline compatibility
# lane instead of silently depending on that production fallback.
os.environ.setdefault("VIRTUROID_ALLOW_HEURISTIC_FALLBACK", "1")
