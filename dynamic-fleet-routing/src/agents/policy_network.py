"""Custom policy network for the PPO agent.

Defines the MLP architecture used by the PPO policy, with
configurable hidden layers and activation functions.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np


class FleetPolicyNetwork(nn.Module):
    """Multi-layer perceptron policy network for fleet routing.

    This network processes the fleet environment observation and outputs
    action logits and state values. It can be used standalone or as
    the feature extractor for Stable-Baselines3 policies.

    Args:
        obs_dim: Dimension of the observation vector.
        action_dim: Number of possible actions.
        hidden_sizes: List of hidden layer sizes.
        activation: Activation function class.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Optional[list[int]] = None,
        activation: type = nn.ReLU,
    ) -> None:
        """Initialize the policy network.

        Args:
            obs_dim: Input observation dimension.
            action_dim: Output action dimension.
            hidden_sizes: Sizes of hidden layers. Defaults to [256, 256].
            activation: Activation function class.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        # Shared feature extractor
        layers: list[nn.Module] = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        # Policy head (action logits)
        self.policy_head = nn.Linear(prev_size, action_dim)

        # Value head
        self.value_head = nn.Linear(prev_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller initialization for output heads
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Tuple of (action_logits, state_value).
        """
        features = self.feature_extractor(obs)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value

    def get_action_probabilities(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get action probabilities with optional masking.

        Args:
            obs: Observation tensor.
            action_mask: Boolean mask (True = valid action).

        Returns:
            Action probability distribution.
        """
        logits, _ = self.forward(obs)

        if action_mask is not None:
            # Set invalid action logits to very negative value
            invalid_mask = ~action_mask
            logits = logits.masked_fill(invalid_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return probs

    def export_torchscript(self, filepath: str, obs_dim: int) -> None:
        """Export the network to TorchScript format.

        Args:
            filepath: Output file path for the TorchScript model.
            obs_dim: Observation dimension for tracing.
        """
        self.eval()
        dummy_input = torch.randn(1, obs_dim)
        traced = torch.jit.trace(self, dummy_input)
        torch.jit.save(traced, filepath)
