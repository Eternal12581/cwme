from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestTwoVariants(unittest.TestCase):
    def test_variant_configs_have_disjoint_namespaces(self):
        config_text = {
            name: (ROOT / "config" / "experiments" / name).read_text(encoding="utf-8")
            for name in ("self_evolve.yml", "gt_hkg_ablation.yml")
        }
        get = lambda text, key: next(
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith(f"  {key}:")
        )
        namespaces = {get(text, "ARTIFACT_NAMESPACE") for text in config_text.values()}
        self.assertEqual(
            namespaces,
            {"artifacts/variants/self_evolve", "artifacts/variants/gt_hkg"},
        )
        paths = []
        for text in config_text.values():
            paths.extend([
                get(text, "TRAJECTORY_OUTPUT"),
                get(text, "historical_kg_path"),
                get(text, "persist_path"),
                get(text, "MANIFEST_PATH"),
                get(text, "LOG_ROOT"),
            ])
            self.assertIn("  fast_path_enabled: false", text)
            self.assertIn("  learned_reuse_enabled: true", text)
            self.assertIn("  task_heuristics_enabled: false", text)
            self.assertIn("  legacy_heuristics_enabled: false", text)
        self.assertEqual(len(paths), len(set(paths)))

    def test_state_action_guard_blocks_only_the_failed_state(self):
        from core.evolution.episode_guard import EpisodeGuard

        guard = EpisodeGuard()
        guard.note_transition("agent.location=hall", "go to kitchen", progress=False)
        self.assertTrue(
            guard.is_blocked("go to kitchen", state_signature="agent.location=hall")
        )
        self.assertFalse(
            guard.is_blocked("go to kitchen", state_signature="agent.location=lab")
        )

    def test_collector_cli_exposes_task_override(self):
        source = (ROOT / "scripts" / "collect_training_trajectories.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parser.add_argument(\n        "--tasks"', source)
        self.assertIn('exp["TASKS"] = list(args.tasks)', source)
        self.assertIn('exp["VARIATIONS"] = list(args.variations)', source)

    def test_gt_hkg_drops_ineligible_episodes_and_noops(self):
        from scripts.build_gt_hkg import prepare
        from core.world_model.historical_kg import HistoricalKG

        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source = temp_path / "gt.jsonl"
            rows = [
                {
                    "episode_id": 0,
                    "task": "boil",
                    "task_id": "boil",
                    "final_score": 100,
                    "success": True,
                    "steps": [
                        {
                            "step": 0,
                            "state_before": "agent.location=hall",
                            "action": "go to kitchen",
                            "state_after": "agent.location=kitchen",
                            "score_before": 0,
                            "score_after": 0,
                            "valid": True,
                            "meaningful_change": True,
                        },
                        {
                            "step": 1,
                            "state_before": "agent.location=kitchen",
                            "action": "look around",
                            "state_after": "agent.location=kitchen",
                            "score_before": 0,
                            "score_after": 0,
                            "valid": True,
                            "meaningful_change": False,
                        },
                    ],
                },
                {
                    "episode_id": 1,
                    "task": "boil",
                    "task_id": "boil",
                    "final_score": 0,
                    "success": False,
                    "steps": [],
                },
            ]
            with source.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            output = temp_path / "hkg" / "historical_world_model.json"
            report = prepare(str(source), str(output))
            self.assertEqual(report["retained_episode_count"], 1)
            self.assertEqual(report["dropped_incomplete_episode_count"], 1)
            self.assertEqual(report["positive_transition_count"], 1)
            kg = HistoricalKG.load(str(output.parent), read_only=True)
            self.assertEqual(kg.metadata["source"], "ground_truth_filtered")
            self.assertEqual(kg.metadata["positive_transition_count"], 1)
            self.assertGreaterEqual(len(kg), 1)

    def test_merge_renumbers_and_preserves_worker_provenance(self):
        from scripts.merge_variant_artifacts import merge_trajectories

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "worker_01.jsonl"
            second = root / "worker_02.jsonl"
            output = root / "merged" / "training_trajectories.jsonl"
            first.write_text(json.dumps({
                "episode_id": 0,
                "task": "boil",
                "task_id": "boil",
                "variation_id": "0",
                "worker_id": "worker_01",
                "steps": [],
            }) + "\n", encoding="utf-8")
            second.write_text(json.dumps({
                "episode_id": 0,
                "task": "melt",
                "task_id": "melt",
                "variation_id": "0",
                "worker_id": "worker_02",
                "steps": [],
            }) + "\n", encoding="utf-8")

            rows, report = merge_trajectories([first, second], output)
            self.assertEqual([row["episode_id"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["source"]["merge_source_worker_id"], "worker_01")
            self.assertEqual(rows[1]["source"]["merge_source_episode_id"], "0")
            self.assertEqual(report["episode_count"], 2)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_merge_rejects_overlapping_task_variation_slots(self):
        from scripts.merge_variant_artifacts import merge_trajectories

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = json.dumps({
                "episode_id": 0,
                "task": "boil",
                "task_id": "boil",
                "variation_id": "0",
                "steps": [],
            }) + "\n"
            first = root / "worker_01.jsonl"
            second = root / "worker_02.jsonl"
            first.write_text(rows, encoding="utf-8")
            second.write_text(rows, encoding="utf-8")
            with self.assertRaises(ValueError):
                merge_trajectories([first, second], root / "merged.jsonl")

    def test_sharded_config_isolates_outputs_and_forces_one_agent(self):
        from scripts.run_sharded_collection import _build_shards, _worker_config

        with tempfile.TemporaryDirectory() as temp:
            worker_dir = Path(temp) / "worker_01"
            config = {
                "EXPERIMENT": {
                    "ARTIFACT_NAMESPACE": str(Path(temp) / "variant"),
                    "LOG_ROOT": str(Path(temp) / "logs"),
                    "NUM_AGENTS": 4,
                },
                "WORLD_MODEL": {"historical_kg": False},
                "PROCEDURAL_MEMORY": {},
            }
            result = _worker_config(
                config,
                worker_dir=worker_dir,
                tasks=["boil"],
                worker_id="worker_01",
                variations=[0, 2, 4],
                hkg_input=str(Path(temp) / "gt_hkg.json"),
            )
            exp = result["EXPERIMENT"]
            self.assertEqual(exp["NUM_AGENTS"], 1)
            self.assertEqual(exp["TASKS"], ["boil"])
            self.assertEqual(exp["VARIATIONS"], [0, 2, 4])
            self.assertTrue(exp["TRAJECTORY_OUTPUT"].startswith(str(worker_dir)))
            self.assertTrue(result["WORLD_MODEL"]["trajectory_path"].startswith(str(worker_dir)))
            shards = _build_shards(["boil"], ["0", "1", "2"], [0, 1, 2, 3, 4, 5])
            self.assertEqual([item[1] for item in shards], [[0, 1], [2, 3], [4, 5]])


if __name__ == "__main__":
    unittest.main()
