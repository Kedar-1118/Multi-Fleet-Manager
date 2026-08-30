"""Hyperparameter tuning for PPO using Ray Tune.

Usage:
    python -m src.training.tune --config configs/tuning.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils.config import load_config, load_base_config
from src.utils.logger import setup_logger
from src.utils.seed import set_global_seed


def train_ppo_trial(config: dict[str, Any]) -> None:
    """Single Ray Tune trial training function.

    Args:
        config: Hyperparameter config from Ray Tune.
    """
    from ray import train as ray_train

    from src.environment.fleet_env import DynamicFleetEnv
    from src.agents.action_masking import ActionMaskingWrapper
    from src.agents.ppo_agent import PPOAgent
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    base_config = config.pop("__base_config__", {})
    training_timesteps = config.pop("__training_timesteps__", 50000)
    seed = config.pop("__seed__", 42)

    set_global_seed(seed)

    def mask_fn(env):
        return env.action_masks()

    def make_env():
        env = DynamicFleetEnv(base_config)
        env = ActionMaskingWrapper(env)
        env = ActionMasker(env, mask_fn)
        return env

    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Map Ray config to PPO config
    net_size = config.get("net_arch_size", 128)
    ppo_config = {
        "ppo": {
            "learning_rate": config.get("learning_rate", 3e-4),
            "gamma": config.get("gamma", 0.99),
            "gae_lambda": config.get("gae_lambda", 0.95),
            "clip_range": config.get("clip_range", 0.2),
            "ent_coef": config.get("ent_coef", 0.01),
            "batch_size": config.get("batch_size", 64),
            "n_steps": config.get("n_steps", 2048),
        },
        "policy": {
            "net_arch": {"pi": [net_size, net_size], "vf": [net_size, net_size]},
        },
    }

    agent = PPOAgent(config=ppo_config)
    agent.build_model(train_env)

    # Train in chunks and report
    chunk_size = min(10000, training_timesteps)
    total_trained = 0

    while total_trained < training_timesteps:
        agent.model.learn(total_timesteps=chunk_size, reset_num_timesteps=False)
        total_trained += chunk_size

        # Evaluate
        from src.training.evaluate import evaluate_dispatcher
        env_for_eval = DynamicFleetEnv(base_config)
        collector = evaluate_dispatcher(env_for_eval, agent, n_episodes=2, seed=seed)
        summary = collector.get_summary()

        ray_train.report({
            "eval/mean_reward": summary.get("total_reward_mean", 0.0),
            "eval/completion_rate": summary.get("completion_rate_mean", 0.0),
            "eval/sla_compliance": summary.get("sla_compliance_rate_mean", 0.0),
            "training_iteration": total_trained,
        })

    train_env.close()
    eval_env.close()


def main() -> None:
    """Run hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters")
    parser.add_argument("--config", type=str, default="configs/tuning.yaml")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger("tune")

    tuning_config = load_config(args.config) if os.path.exists(args.config) else {}
    base_config = load_config(args.base_config) if os.path.exists(args.base_config) else load_base_config()

    tuning_cfg = tuning_config.get("tuning", {})
    search_cfg = tuning_config.get("search_space", {})

    try:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
    except ImportError:
        logger.error("Ray Tune not installed. Install with: pip install 'ray[tune]'")
        return

    # Build search space
    param_space: dict[str, Any] = {}
    param_space["__base_config__"] = base_config
    param_space["__training_timesteps__"] = tuning_cfg.get("training_timesteps", 50000)
    param_space["__seed__"] = args.seed

    for param_name, param_cfg in search_cfg.items():
        param_type = param_cfg.get("type", "choice")
        if param_type == "loguniform":
            param_space[param_name] = tune.loguniform(param_cfg["low"], param_cfg["high"])
        elif param_type == "uniform":
            param_space[param_name] = tune.uniform(param_cfg["low"], param_cfg["high"])
        elif param_type == "choice":
            param_space[param_name] = tune.choice(param_cfg["values"])

    # Scheduler
    scheduler = ASHAScheduler(
        metric=tuning_cfg.get("metric", "eval/mean_reward"),
        mode=tuning_cfg.get("mode", "max"),
        grace_period=tuning_cfg.get("grace_period", 10000),
        reduction_factor=tuning_cfg.get("reduction_factor", 3),
    )

    logger.info(f"Starting tuning with {tuning_cfg.get('num_samples', 10)} trials")

    # Run tuning
    tuner = tune.Tuner(
        train_ppo_trial,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=tuning_cfg.get("num_samples", 10),
            max_concurrent_trials=tuning_cfg.get("max_concurrent_trials", 2),
            scheduler=scheduler,
        ),
    )

    results = tuner.fit()

    # Save best config
    best_result = results.get_best_result(
        metric=tuning_cfg.get("metric", "eval/mean_reward"),
        mode=tuning_cfg.get("mode", "max"),
    )

    artifacts_cfg = tuning_config.get("artifacts", {})
    results_dir = artifacts_cfg.get("results_dir", "artifacts/tuning")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Save best config
    best_config = {k: v for k, v in best_result.config.items()
                   if not k.startswith("__")}
    best_config_path = artifacts_cfg.get("best_config_path",
                                         os.path.join(results_dir, "best_config.yaml"))
    with open(best_config_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)

    logger.info(f"Best config saved to {best_config_path}")
    logger.info(f"Best reward: {best_result.metrics.get('eval/mean_reward', 'N/A')}")

    # Save results CSV
    results_df = results.get_dataframe()
    csv_path = artifacts_cfg.get("results_csv_path",
                                 os.path.join(results_dir, "tuning_results.csv"))
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
