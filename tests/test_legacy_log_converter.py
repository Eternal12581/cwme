"""Tests for legacy text-log trajectory reconstruction."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_legacy_logs_to_trajectories import parse_legacy_logs


class LegacyLogConverterTests(unittest.TestCase):
    def test_ignores_plan_dump_and_reconstructs_scores(self):
        content = "\n".join(
            [
                "Episode start: task=boil variation=21 ep=1/1 pattern_lib_size=0",
                "<Action> task - <Response>Execute task toward the task goal.",
                "t = 1: Exectued action `['look around']` | received response `['old diagnostic']`",
                "INFO - t = 2: Exectued action `['go to kitchen']` | received response `['You move to the kitchen.']` | number of token sent so far: 0",
                "INFO - t = 3: Exectued action `open oven` | received response `The oven is now open. | Total Score 25 | number of token sent so far: 0`",
                "FINISHED RUN!",
                "Finished task=boil variation=21 ep=1/1 score=25 pattern_lib_size=1",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "result.log"
            log.write_text(content, encoding="utf-8")
            rows, stats = parse_legacy_logs([str(log)])

        self.assertEqual(stats["episodes_emitted"], 1)
        self.assertEqual(stats["executed_action_records"], 1)
        self.assertEqual(len(rows[0]["steps"]), 1)
        step = rows[0]["steps"][0]
        self.assertEqual(step["action"], "open oven")
        self.assertEqual(step["state_after"], "The oven is now open.")
        self.assertEqual(step["score_before"], 0.0)
        self.assertEqual(step["score_after"], 25.0)
        self.assertTrue(step["state_before_missing"])

    def test_skips_unfinished_episode_by_default(self):
        content = (
            "Episode start: task=boil variation=22 ep=1/1\n"
            "INFO - t = 1: Exectued action `look around` | received response `ok | Total Score 0`\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "result.log"
            log.write_text(content, encoding="utf-8")
            rows, stats = parse_legacy_logs([str(log)])

        self.assertEqual(rows, [])
        self.assertEqual(stats["episodes_skipped_incomplete"], 1)

    def test_filters_task_and_variation(self):
        content = "\n".join(
            [
                "Episode start: task=boil variation=1 ep=1/1",
                "INFO - t = 1: Exectued action `look around` | received response `ok | Total Score 0`",
                "Finished task=boil variation=1 ep=1/1 score=0",
                "Episode start: task=melt variation=2 ep=1/1",
                "INFO - t = 2: Exectued action `look around` | received response `ok | Total Score 0`",
                "Finished task=melt variation=2 ep=1/1 score=0",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "result.log"
            log.write_text(content, encoding="utf-8")
            rows, stats = parse_legacy_logs(
                [str(log)], tasks={"boil"}, variations={"1"}
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task"], "boil")
        self.assertEqual(stats["episodes_filtered"], 1)


if __name__ == "__main__":
    unittest.main()
