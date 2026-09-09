#!/usr/bin/env python3
"""Run one independent trajectory collector per GPU/task shard.

Each child receives one visible GPU and a private artifact namespace.  The
resulting worker JSONL files are intentionally not merged here; call
``merge_variant_artifacts.py`` after all children finish.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _chunks(values: list[str], count: int) -> list[list[str]]:
    if count > len(values):
        raise ValueError(f"GPU count ({count}) cannot exceed task count ({len(values)})")
    base, extra = divmod(len(values), count)
    result: list[list[str]] = []
    cursor = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        result.append(values[cursor : cursor + size])
        cursor += size
    return result


def _build_shards(
    tasks: list[str],
    gpus: list[str],
    variations: list[int] | None,
    task_variations: dict[str, list[int]] | None = None,
) -> list[tuple[list[str], list[int] | dict[str, list[int]] | None]]:
    """Shard tasks normally, or task-local variations for a single task."""
    if task_variations is not None:
        normalized = {
            str(key): [int(value) for value in values]
            for key, values in task_variations.items()
        }
        missing = [task for task in tasks if task not in normalized]
        if missing:
            raise ValueError(
                "--task-variations must specify every requested task; "
                f"missing={missing}"
            )
        if len(tasks) == 1:
            selected = normalized[tasks[0]]
            chunks = _chunks([str(value) for value in selected], len(gpus))
            return [
                (list(tasks), {tasks[0]: [int(value) for value in chunk]})
                for chunk in chunks
            ]
        task_chunks = _chunks(tasks, len(gpus))
        return [
            (chunk, {task: normalized[task] for task in chunk})
            for chunk in task_chunks
        ]
    if variations is not None and len(tasks) == 1:
        variation_chunks = _chunks([str(value) for value in variations], len(gpus))
        return [
            (list(tasks), [int(value) for value in chunk])
            for chunk in variation_chunks
        ]
    task_chunks = _chunks(tasks, len(gpus))
    return [(chunk, variations) for chunk in task_chunks]


def _copy_input_artifact(source_value: str, target: Path, *, required: bool) -> str:
    source = _resolve(source_value)
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Required input artifact not found: {source}")
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return str(target)
    shutil.copy2(source, target)
    return str(target)


def _worker_config(
    base: dict,
    *,
    worker_dir: Path,
    tasks: list[str],
    worker_id: str,
    variations: list[int] | dict[str, list[int]] | None = None,
    hkg_input: str = "",
    pattern_input: str = "",
) -> dict:
    cfg = copy.deepcopy(base)
    exp = cfg.setdefault("EXPERIMENT", {})
    wm = cfg.setdefault("WORLD_MODEL", {})
    pm = cfg.setdefault("PROCEDURAL_MEMORY", {})

    exp["TASKS"] = list(tasks)
    if variations is not None:
        exp["VARIATIONS"] = copy.deepcopy(variations)
    exp["NUM_AGENTS"] = 1
    exp["COLLECTION_WORKER_ID"] = worker_id
    exp["ARTIFACT_NAMESPACE"] = str(worker_dir)
    exp["TRAJECTORY_OUTPUT"] = str(worker_dir / "training_trajectories.jsonl")
    exp["MANIFEST_PATH"] = str(worker_dir / "manifests" / f"{worker_id}.json")
    # Logs belong to the worker namespace as well. Using the base LOG_ROOT /
    # worker_id would collide when machine_01 and machine_02 share storage.
    exp["LOG_ROOT"] = str(worker_dir / "logs")
    exp["RESULTS_PATH"] = str(worker_dir / "results")

    wm["trajectory_path"] = exp["TRAJECTORY_OUTPUT"]
    hkg_source = str(hkg_input or wm.get("historical_kg_path") or "")
    hkg_enabled = bool(wm.get("historical_kg", False))
    hkg_target = worker_dir / "historical_kg" / "historical_world_model.json"
    wm["historical_kg_path"] = str(hkg_target)
    if hkg_source:
        _copy_input_artifact(
            hkg_source,
            hkg_target,
            required=hkg_enabled and not bool(wm.get("allow_missing_hkg", False)),
        )
    elif hkg_enabled and not bool(wm.get("allow_missing_hkg", True)):
        raise FileNotFoundError(f"Required HKG input not found: {hkg_source}")

    pm["persist_path"] = str(worker_dir / "procedural_memory" / "patterns.jsonl")
    load_source = str(
        pattern_input
        or pm.get("load_path")
        or pm.get("procedural_load_path")
        or ""
    )
    if load_source:
        load_target = worker_dir / "inputs" / Path(load_source).name
        pm["load_path"] = _copy_input_artifact(load_source, load_target, required=True)
        pm.pop("procedural_load_path", None)
    return cfg


def run_sharded(
    config_path: str,
    tasks: list[str],
    gpus: list[str],
    *,
    run_id: str = "",
    machine_id: str = "machine_01",
    variations: list[int] | None = None,
    task_variations: dict[str, list[int]] | None = None,
    hkg_input: str = "",
    pattern_input: str = "",
) -> int:
    import yaml

    config_file = _resolve(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config not found: {config_file}")
    with config_file.open(encoding="utf-8") as fh:
        base = yaml.load(fh, Loader=yaml.FullLoader) or {}

    exp = base.setdefault("EXPERIMENT", {})
    namespace = _resolve(str(exp.get("ARTIFACT_NAMESPACE") or f"artifacts/variants/{config_file.stem}"))
    run_id = run_id or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    machine_id = machine_id.replace("/", "_").replace("\\", "_").strip() or "machine_01"
    output_root = namespace / "distributed" / run_id / machine_id
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Worker output already exists: {output_root}. Use a new --run-id; refusing to overwrite."
        )
    if variations is not None and task_variations is not None:
        raise ValueError("Use either --variations or --task-variations, not both")
    shards = _build_shards(tasks, gpus, variations, task_variations)
    processes: list[tuple[str, subprocess.Popen, object]] = []
    collector = ROOT / "scripts" / "collect_training_trajectories.py"

    for index, (gpu, (shard_tasks, shard_variations)) in enumerate(zip(gpus, shards), 1):
        worker_id = f"worker_{index:02d}"
        worker_dir = output_root / worker_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        cfg = _worker_config(
            base,
            worker_dir=worker_dir,
            tasks=shard_tasks,
            worker_id=worker_id,
            variations=shard_variations,
            hkg_input=hkg_input,
            pattern_input=pattern_input,
        )
        worker_config = worker_dir / "worker_config.yml"
        with worker_config.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, sort_keys=False, allow_unicode=True)
        log_file = worker_dir / "launcher.log"
        log_handle = log_file.open("w", encoding="utf-8")
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [sys.executable, "-u", str(collector), str(worker_config)]
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=child_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, process, log_handle))
        print(
            f"{worker_id}: gpu={gpu} tasks={shard_tasks} "
            f"variations={shard_variations} pid={process.pid} log={log_file}"
        )

    failed = 0
    for worker_id, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        print(f"{worker_id}: exit={return_code}")
        if return_code != 0:
            failed += 1
    print(f"Worker artifacts: {output_root}")
    print("Merge after completion with scripts/merge_variant_artifacts.py")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent collector processes across GPUs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical GPU ids, one worker per id")
    parser.add_argument("--run-id", default="", help="Stable output run id; default is a timestamp")
    parser.add_argument("--machine-id", default="machine_01", help="Unique id when several machines share a filesystem")
    parser.add_argument(
        "--variations",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Run only these task-local train variation ids",
    )
    parser.add_argument(
        "--task-variations",
        nargs="+",
        default=None,
        metavar="TASK=ID,ID",
        help="Task-local variation selections, e.g. boil=0,1,2 melt=4,5",
    )
    parser.add_argument(
        "--hkg-input",
        default="",
        help="Override the source HKG file copied into each worker (mainly GT-HKG)",
    )
    parser.add_argument(
        "--pattern-input",
        default="",
        help="Override the read-only P_train procedural artifact copied into each worker",
    )
    args = parser.parse_args()
    task_variations = None
    if args.task_variations:
        task_variations = {}
        for spec in args.task_variations:
            if "=" not in spec:
                raise SystemExit(
                    f"Invalid --task-variations item {spec!r}; expected TASK=ID,ID"
                )
            task, raw_ids = spec.split("=", 1)
            task = task.strip()
            if not task or not raw_ids.strip():
                raise SystemExit(f"Invalid --task-variations item {spec!r}")
            try:
                ids = [int(value.strip()) for value in raw_ids.split(",") if value.strip()]
            except ValueError as exc:
                raise SystemExit(f"Invalid variation id in {spec!r}") from exc
            if not ids:
                raise SystemExit(f"No variation ids in {spec!r}")
            task_variations[task] = ids
    raise SystemExit(
        run_sharded(
            args.config,
            args.tasks,
            args.gpus,
            run_id=args.run_id,
            machine_id=args.machine_id,
            variations=args.variations,
            task_variations=task_variations,
            hkg_input=args.hkg_input,
            pattern_input=args.pattern_input,
        )
    )


if __name__ == "__main__":
    main()
