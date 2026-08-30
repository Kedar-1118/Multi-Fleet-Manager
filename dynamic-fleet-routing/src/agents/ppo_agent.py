"""PPO agent wrapper for fleet routing.

Provides a unified interface for training, evaluating, and using
the PPO agent with Stable-Baselines3's MaskablePPO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.agents.action_masking import ActionMaskingWrapper


class PPOAgent:
    """PPO-based dispatch agent using sb3-contrib MaskablePPO.

    Wraps Stable-Baselines3's MaskablePPO with action masking support
    for the fleet routing environment.

    Args:
        env: The fleet environment (will be wrapped if needed).
        config: PPO configuration dictionary.
    """

    def __init__(
        self,
        env: Optional[DynamicFleetEnv] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the PPO agent.

        Args:
            env: Optional environment for training.
            config: PPO hyperparameters.
        """
        self.config = config or {}
        self.model = None
        self.name = "ppo"
        self._env = env

    def build_model(self, env: Any) -> None:
        """Build the MaskablePPO model.

        Args:
            env: The (vectorized) environment.
        """
        from sb3_contrib import MaskablePPO

        ppo_cfg = self.config.get("ppo", {})
        policy_cfg = self.config.get("policy", {})

        net_arch_cfg = policy_cfg.get("net_arch", {"pi": [256, 256], "vf": [256, 256]})
        net_arch = [dict(pi=net_arch_cfg.get("pi", [256, 256]),
                         vf=net_arch_cfg.get("vf", [256, 256]))]

        self.model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            verbose=1,
            policy_kwargs={"net_arch": net_arch},
        )

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_dir: str = "artifacts/models",
        log_dir: str = "artifacts/logs",
    ) -> None:
        """Train the PPO agent.

        Args:
            total_timesteps: Total training timesteps.
            eval_env: Optional evaluation environment.
            eval_freq: Steps between evaluations.
            n_eval_episodes: Number of evaluation episodes.
            save_dir: Directory for model checkpoints.
            log_dir: Directory for training logs.
        """
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
            CallbackList,
        )

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        callbacks = []

        # Checkpoint callback
        checkpoint_cb = CheckpointCallback(
            save_freq=max(total_timesteps // 10, 1000),
            save_path=save_dir,
            name_prefix="ppo_fleet",
        )
        callbacks.append(checkpoint_cb)

        # Eval callback
        if eval_env is not None:
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=save_dir,
                log_path=log_dir,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_cb)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )

        # Save final model
        final_path = os.path.join(save_dir, "final_model")
        self.model.save(final_path)

    def select_action(self, env: DynamicFleetEnv) -> int:
        """Select an action using the trained PPO policy.

        Args:
            env: The fleet environment.

        Returns:
            Selected action index.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() or build_model() first.")

        obs = env._get_observation()
        mask = env.get_action_mask()

        action, _ = self.model.predict(
            obs, deterministic=True, action_masks=mask
        )
        return int(action)

    def get_action_probabilities(
        self, obs: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get action probability distribution from the policy.

        Args:
            obs: Observation array.
            mask: Optional action mask.

        Returns:
            Action probabilities array.
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded.")

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(self.model.device)

        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy().flatten()

        if mask is not None:
            probs = probs * mask.astype(np.float32)
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs /= prob_sum
            else:
                # All masked — uniform over valid actions
                valid = mask.astype(np.float32)
                probs = valid / valid.sum()

        return probs

    def save(self, filepath: str) -> None:
        """Save the model to disk.

        Args:
            filepath: File path for the saved model.
        """
        if self.model is not None:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(filepath)

    def load(self, filepath: str, env: Optional[Any] = None) -> None:
        """Load a trained model from disk.

        Args:
            filepath: Path to the saved model.
            env: Optional environment for continued training.
        """
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(filepath, env=env)
