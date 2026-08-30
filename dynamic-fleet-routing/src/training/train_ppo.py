"""Training pipeline for PPO agent.

Usage:
    python -m src.training.train_ppo --config configs/ppo.yaml
    python -m src.training.train_ppo --config configs/ppo.yaml --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from src.utils.config import load_config, load_base_config
from src.utils.seed import set_global_seed
from src.utils.logger import setup_logger


def main() -> None:
    """Run PPO training pipeline."""
    parser = argparse.ArgumentParser(description="Train PPO fleet routing agent")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml",
                        help="Path to PPO config file")
    parser.add_argument("--base-config", type=str, default="configs/base.yaml",
                        help="Path to base config file")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-dir", type=str, default="artifacts/models",
                        help="Model save directory")
    args = parser.parse_args()

    logger = setup_logger("train_ppo")

    # Load configs
    base_config = load_base_config() if not os.path.exists(args.base_config) else load_config(args.base_config)
    ppo_config = load_config(args.config) if os.path.exists(args.config) else {}

    # Set seed
    set_global_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Training parameters
    training_cfg = ppo_config.get("training", {})
    total_timesteps = args.total_timesteps or training_cfg.get("total_timesteps", 100000)
    n_envs = training_cfg.get("n_envs", 1)
    eval_freq = training_cfg.get("eval_freq", 10000)
    n_eval_episodes = training_cfg.get("n_eval_episodes", 3)

    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Vectorized envs: {n_envs}")

    # Create environments
    from src.environment.fleet_env import DynamicFleetEnv
    from src.agents.action_masking import ActionMaskingWrapper, make_masked_env
    from src.agents.ppo_agent import PPOAgent

    # Build vectorized environment using SubprocVecEnv or DummyVecEnv
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.utils import get_action_masks

    def mask_fn(env):
        return env.action_masks()

    def make_env(seed_offset: int = 0):
        def _init():
            env_config = dict(base_config)
            env_config["simulation"] = dict(env_config.get("simulation", {}))
            env_config["simulation"]["seed"] = args.seed + seed_offset
            env = DynamicFleetEnv(env_config)
            env = ActionMaskingWrapper(env)
            env = ActionMasker(env, mask_fn)
            return env
        return _init

    # Create training env
    if n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        try:
            train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        except Exception:
            from stable_baselines3.common.vec_env import DummyVecEnv
            train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env = DummyVecEnv([make_env(0)])

    # Create eval env
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([make_env(100)])

    logger.info("Environments created")

    # Build and train agent
    agent = PPOAgent(config=ppo_config)
    agent.build_model(train_env)
    logger.info("PPO model built")

    # MLflow tracking
    try:
        import mlflow
        mlflow.set_experiment("ppo_training")
        with mlflow.start_run(run_name=f"ppo_seed{args.seed}"):
            mlflow.log_params({
                "seed": args.seed,
                "total_timesteps": total_timesteps,
                "learning_rate": ppo_config.get("ppo", {}).get("learning_rate", 3e-4),
                "gamma": ppo_config.get("ppo", {}).get("gamma", 0.99),
                "n_envs": n_envs,
            })

            agent.train(
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                save_dir=args.save_dir,
            )

            # Log artifacts
            model_path = os.path.join(args.save_dir, "final_model.zip")
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path)

            logger.info("Training complete. Model saved and logged to MLflow.")
    except ImportError:
        logger.warning("MLflow not available, training without experiment tracking")
        agent.train(
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_dir=args.save_dir,
        )
        logger.info("Training complete.")

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
