#!/usr/bin/env python3
"""
Run one experiment config across multiple random seeds.

Usage:
  python scripts/run_multi_seed.py config/experiments/e01_cwme_sci_main.yml
  python scripts/run_multi_seed.py config/experiments/e03_full.yml --seeds 4902 4903
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

# Run the first seed by default. Supplementary seeds must be requested
# explicitly so a normal experiment command does not multiply the workload.
DEFAULT_SEEDS = [4901]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment config with multiple seeds")
    parser.add_argument("config", help="Base experiment YAML")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds (default: 4901)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        base_cfg = yaml.load(fh, Loader=yaml.FullLoader)

    runner = os.path.join(_REPO_ROOT, "ExperimentRunner.py")
    base_name = os.path.splitext(os.path.basename(config_path))[0]
    base_log_root = (
        (base_cfg.get("EXPERIMENT") or {}).get("LOG_ROOT")
        or f"/mnt/workspace/outputs/{base_name}"
    ).rstrip("/")

    for seed in args.seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("AGENT", {})["SEED"] = int(seed)
        exp = cfg.setdefault("EXPERIMENT", {})
        log_root = (exp.get("LOG_ROOT") or f"/mnt/workspace/outputs/{base_name}").rstrip("/")
        exp["LOG_ROOT"] = f"{log_root}/seed_{seed}"
        exp_name = str(exp.get("EXPERIMENT_NAME", base_name))
        exp["EXPERIMENT_NAME"] = f"{exp_name}_seed{seed}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            prefix=f"{base_name}_seed{seed}_",
            delete=False,
            encoding="utf-8",
        ) as tf:
            yaml.dump(cfg, tf, sort_keys=False, allow_unicode=True)
            tmp_path = tf.name

        print(f"\n========== seed={seed} log={exp['LOG_ROOT']} ==========")
        try:
            rc = subprocess.call([sys.executable, runner, tmp_path], cwd=_REPO_ROOT)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if rc != 0:
            raise SystemExit(f"Experiment failed for seed={seed} (exit {rc})")

    print("\nMulti-seed run complete. Aggregating...")
    agg_script = os.path.join(_REPO_ROOT, "scripts", "aggregate_multiseed.py")
    subprocess.call([sys.executable, agg_script, base_log_root])


if __name__ == "__main__":
    main()
