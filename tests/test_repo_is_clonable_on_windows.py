"""A repository nobody can CLONE is not shippable, however good the code inside it is.

Windows' default MAX_PATH is 260 characters, and it applies to the fully-resolved path -- clone root plus the
path inside the repo. A fresh-clone audit hit exactly this: the scene generator embedded a ~57-char slug of the
user's prompt in every scene id, that id became the filename, and those files sit six directories deep:

    build/ui_verify/arm_sort/simulation/mujoco/scenes/regression/
        regression_scene_task_req_a_tabletop_robot_arm_that_sorts_red_and_blue_blo_stress_002.xml   (150 chars)

42 tracked paths were in that shape, so any clone root deeper than ~110 characters -- an ordinary
C:\\Users\\<name>\\source\\repos\\... -- could not check the repository out AT ALL. Git reports it as a checkout
failure, not as "your path is too long", so it reads like a corrupt repo.

These two tests are the standing guard: the budget on TRACKED paths, and the scene ids that broke it. The
second one matters on its own -- the first only sees ids that were committed, and a generator that starts
emitting long names again would ship the defect to every user package long before anything is tracked.
"""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

# 260 (MAX_PATH) minus a realistic clone root ("C:\\Users\\<name>\\source\\repos\\Virturoid\\" and friends),
# minus git's own worktree temp suffixes. 100 leaves ~150 characters of room for the user's directory.
_MAX_TRACKED_PATH = 100
_REPO = Path(__file__).resolve().parents[1]


class TrackedPathLengthTests(unittest.TestCase):
    def test_no_tracked_path_exceeds_the_windows_budget(self):
        try:
            out = subprocess.run(["git", "ls-files"], cwd=_REPO, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:      # no git on PATH -> nothing to assert
            self.skipTest(f"git unavailable: {exc}")
        if out.returncode != 0:
            self.skipTest("not a git checkout")
        long_paths = sorted((p for p in out.stdout.splitlines() if len(p) > _MAX_TRACKED_PATH),
                            key=len, reverse=True)
        self.assertEqual(
            [], long_paths[:10],
            f"{len(long_paths)} tracked path(s) exceed {_MAX_TRACKED_PATH} chars; a Windows clone into a "
            f"normally-deep directory will FAIL to check the repo out. Longest: {len(long_paths[0]) if long_paths else 0}")


class SceneIdLengthTests(unittest.TestCase):
    """Scene ids ARE filenames (simulation/mujoco/scenes/<purpose>/<scene_id>.xml), so they carry a budget."""

    def _task(self, prompt):
        from virturoid.services.requirements_builder import build_requirements_from_prompt
        from virturoid.services.task_builder import build_task_graph
        return build_task_graph(build_requirements_from_prompt(prompt))

    def test_scene_ids_stay_short_for_a_long_prompt(self):
        from virturoid.services.scene_generator import generate_scene_set
        # the exact prompt whose 57-char task id produced the 150-char tracked paths
        task = self._task("a tabletop robot arm that sorts red and blue blocks into matching bins")
        for purpose in ("baseline", "variation", "edge_case", "holdout", "stress"):
            for scene in generate_scene_set(task, count=2, purpose=purpose).scenes:
                self.assertLessEqual(len(scene.id), 40, f"scene id is a filename: {scene.id}")
                self.assertNotIn("tabletop", scene.id, "the prompt slug must not be re-embedded in filenames")

    def test_scene_ids_are_deterministic_and_distinct_per_task(self):
        from virturoid.services.scene_generator import generate_scene_set
        a = self._task("a tabletop robot arm that sorts red and blue blocks into matching bins")
        b = self._task("build a mobile base that delivers parts indoors")
        ids_a = [s.id for s in generate_scene_set(a, count=3, purpose="baseline").scenes]
        again = [s.id for s in generate_scene_set(a, count=3, purpose="baseline").scenes]
        ids_b = [s.id for s in generate_scene_set(b, count=3, purpose="baseline").scenes]
        self.assertEqual(ids_a, again, "same task+purpose+index must give the same id (reproducible runs)")
        self.assertEqual(3, len(set(ids_a)), "ids within a set must stay distinct")
        self.assertFalse(set(ids_a) & set(ids_b), "different tasks must not collide on scene ids")

    def test_regression_scene_ids_fit_under_their_deeper_directory(self):
        from virturoid.schemas.runs import FailureRecord
        from virturoid.services.scene_generator import generate_regression_scene_set, generate_scene_set
        task = self._task("a tabletop robot arm that sorts red and blue blocks into matching bins")
        source = generate_scene_set(task, count=3, purpose="stress")
        failures = [FailureRecord(id=f"failure_ep_{i}", episode_id=f"ep_{i}", failure_type="collision",
                                  severity="medium", summary="s",
                                  regression_scene_id=f"regr_{source.scenes[i].id}") for i in range(3)]
        regr = generate_regression_scene_set(task, failures, source)
        prefix = "build/ui_verify/arm_sort/simulation/mujoco/scenes/regression/"
        for scene in regr.scenes:
            self.assertLessEqual(len(prefix) + len(scene.id) + len(".xml"), _MAX_TRACKED_PATH, scene.id)


if __name__ == "__main__":
    unittest.main()
