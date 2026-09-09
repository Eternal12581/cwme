"""
Legacy oracle-action trainer (gold trajectory replay).

For CWME experiments use instead:
  python scripts/collect_training_trajectories.py
  python scripts/build_historical_kg.py
  python ExperimentRunner.py config/platform/ablation_cwme.yml
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import yaml
from scienceworld import ScienceWorldEnv

from ReasoningAgent import ReasoningAgent
from utils import get_connection_pool, load_config

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_INI = os.path.join(_REPO_ROOT, "config", "config.ini")
_CONFIG_YML = os.path.join(_REPO_ROOT, "config", "config.yml")


def setup_experiment_logger(log_file, agent_id=None):
    logger_name = f"experiment_logger_{agent_id}" if agent_id else "experiment_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train(agent):
    asyncio.run(agent.update(store=False))
    for action in agent.env.getGoldActionSequence():
        _, res, info, _ = asyncio.run(agent.step(action))
        agent.logger.info(
            "Executed action `%s` | received response `%s` | Total Score %s",
            action, res, info.get("score"),
        )


def run_agent(agent_config, tasks, variations, simplification, global_experiment_dir, agent_id):
    connection_pool = get_connection_pool(config)
    agent_model = agent_config["AGENT"]["AGENT_MODEL"]
    agent_kgmodel = agent_config["AGENT"].get("KNOWLEDGE_GRAPH_MODEL") or agent_model
    agent = ReasoningAgent(
        config, None, connection_pool, agent_id, 300,
        agent_config["AGENT"]["MAX_LOOK_AHEAD"],
        agent_config["AGENT"]["MAX_QUERY"],
        ScienceWorldEnv(), agent_model, agent_kgmodel,
        agent_config=agent_config,
    )

    for task in tasks:
        for variation in variations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(global_experiment_dir, f"log/{task}/variation_{variation}_t_{timestamp}/")
            os.makedirs(log_dir, exist_ok=True)
            logger = setup_experiment_logger(os.path.join(log_dir, "experiment.log"), agent_id)
            agent.logger = logger
            agent.env.load(
                task, variationIdx=variation,
                simplificationStr=simplification, generateGoldPath=True,
            )
            agent.env.reset()
            logger.info(agent.env.getTaskDescription())
            agent.reset()
            train(agent)
            global_logger.info("finished task: %s variation: %s", task, variation)


def divide_chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def run_parallel_experiments(num_agents, tasks, simplification, agent_config, global_experiment_dir, agent_id):
    ep_per_agent = max(1, len(tasks) // num_agents)
    task_chunks = list(divide_chunks(tasks, ep_per_agent))
    with ProcessPoolExecutor(max_workers=num_agents) as executor:
        futures = [
            executor.submit(
                run_agent, agent_config, chunk, [0, 1, 2, 3, 4],
                simplification, global_experiment_dir, agent_id,
            )
            for chunk in task_chunks
        ]
        return [f.result() for f in futures]


if __name__ == "__main__":
    config = load_config(_CONFIG_INI)
    if config is None:
        raise SystemExit(f"Missing or invalid config.ini: {_CONFIG_INI}")
    with open(_CONFIG_YML, encoding="utf-8") as fh:
        agent_config = yaml.load(fh, Loader=yaml.FullLoader)

    seed = int(agent_config["AGENT"]["SEED"])
    random.seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_experiment_dir = os.path.join(_REPO_ROOT, f"log/oracle_{timestamp}")
    os.makedirs(global_experiment_dir, exist_ok=True)
    global_logger = setup_experiment_logger(os.path.join(global_experiment_dir, "global_experiment.log"))
    agent_id = agent_config["AGENT"]["ID"]

    try:
        run_parallel_experiments(
            num_agents=1,
            tasks=["boil"],
            simplification="easy",
            agent_config=agent_config,
            global_experiment_dir=global_experiment_dir,
            agent_id=agent_id,
        )
    except Exception as exc:
        global_logger.error("Oracle training failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        gc.collect()
