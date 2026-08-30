"""Action masking wrapper for Stable-Baselines3 compatibility.

Wraps DynamicFleetEnv to expose action masks through the SB3
ActionMasker interface, ensuring the PPO agent only selects valid actions.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environment.fleet_env import DynamicFleetEnv


class ActionMaskingWrapper(gym.Wrapper):
    """Gymnasium wrapper that adds action masking to DynamicFleetEnv.

    Exposes the action mask through the info dict and provides
    the `action_masks()` method required by sb3-contrib's MaskablePPO.

    This wrapper samples only from valid actions when the environment
    is used with random action selection (e.g., during evaluation).
    """

    def __init__(self, env: DynamicFleetEnv) -> None:
        """Initialize the action masking wrapper.

        Args:
            env: The underlying DynamicFleetEnv instance.
        """
        super().__init__(env)
        self._last_mask: Optional[np.ndarray] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and compute initial action mask.

        Args:
            seed: Optional random seed.
            options: Optional reset options.

        Returns:
            Tuple of (observation, info with action_mask).
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_mask = self.env.get_action_mask()
        info["action_mask"] = self._last_mask
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step and update the action mask.

        Args:
            action: The action to take.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_mask = self.env.get_action_mask()
        info["action_mask"] = self._last_mask
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask for sb3-contrib MaskablePPO.

        Returns:
            Boolean array where True indicates a valid action.
        """
        if self._last_mask is None:
            self._last_mask = self.env.get_action_mask()
        return self._last_mask

    def sample_valid_action(self) -> int:
        """Sample a random valid action using the current mask.

        Returns:
            A randomly selected valid action index.
        """
        mask = self.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return self.env.action_space.n - 1  # NO-OP fallback
        return int(np.random.choice(valid))


def make_masked_env(config: Optional[dict[str, Any]] = None, seed: int = 42):
    """Factory function to create a masked environment.

    Args:
        config: Environment configuration.
        seed: Random seed.

    Returns:
        A callable that creates the wrapped environment.
    """
    def _init():
        env = DynamicFleetEnv(config)
        env = ActionMaskingWrapper(env)
        return env
    return _init
