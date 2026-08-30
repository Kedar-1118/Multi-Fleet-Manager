"""Reward function for the fleet routing environment.

Implements a multi-objective reward system with configurable weights
for delivery success, travel costs, fuel, SLA compliance, and utilization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RewardConfig:
    """Configuration for the reward function.

    Attributes:
        delivery_reward: Reward for each successful delivery.
        travel_penalty: Penalty per km traveled.
        fuel_penalty: Penalty per fuel unit consumed.
        sla_violation_penalty: Penalty for missing a deadline.
        idle_penalty: Penalty per minute of vehicle idle time.
        utilization_reward: Bonus for vehicle utilization.
        expiry_penalty: Penalty for a request expiring.
        normalize: Whether to apply reward normalization.
        normalization_window: Window size for running statistics.
    """
    delivery_reward: float = 10.0
    travel_penalty: float = 0.1
    fuel_penalty: float = 0.5
    sla_violation_penalty: float = 20.0
    idle_penalty: float = 0.05
    utilization_reward: float = 1.0
    expiry_penalty: float = 15.0
    normalize: bool = True
    normalization_window: int = 100


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components for logging.

    Attributes:
        delivery_reward: Reward from successful deliveries.
        travel_penalty: Penalty from distance traveled.
        fuel_penalty: Penalty from fuel consumed.
        sla_penalty: Penalty from SLA violations.
        idle_penalty: Penalty from idle vehicles.
        utilization_reward: Reward from fleet utilization.
        expiry_penalty: Penalty from expired requests.
        total_reward: Sum of all components.
    """
    delivery_reward: float = 0.0
    travel_penalty: float = 0.0
    fuel_penalty: float = 0.0
    sla_penalty: float = 0.0
    idle_penalty: float = 0.0
    utilization_reward: float = 0.0
    expiry_penalty: float = 0.0
    total_reward: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert breakdown to a dictionary for logging."""
        return {
            "delivery_reward": round(self.delivery_reward, 4),
            "travel_penalty": round(self.travel_penalty, 4),
            "fuel_penalty": round(self.fuel_penalty, 4),
            "sla_penalty": round(self.sla_penalty, 4),
            "idle_penalty": round(self.idle_penalty, 4),
            "utilization_reward": round(self.utilization_reward, 4),
            "expiry_penalty": round(self.expiry_penalty, 4),
            "total_reward": round(self.total_reward, 4),
        }


class RewardCalculator:
    """Calculates multi-objective rewards for fleet routing decisions.

    Combines multiple reward components with configurable weights and
    optional running-mean normalization to stabilize RL training.

    Attributes:
        config: Reward configuration.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        """Initialize the reward calculator.

        Args:
            config: Reward configuration. Uses defaults if None.
        """
        self.config = config or RewardConfig()
        self._reward_history: deque[float] = deque(
            maxlen=self.config.normalization_window
        )
        self._running_mean: float = 0.0
        self._running_var: float = 1.0

    def calculate(
        self,
        deliveries_completed: int = 0,
        distance_traveled: float = 0.0,
        fuel_consumed: float = 0.0,
        sla_violations: int = 0,
        idle_time: float = 0.0,
        fleet_utilization: float = 0.0,
        requests_expired: int = 0,
        on_time_deliveries: int = 0,
    ) -> RewardBreakdown:
        """Calculate the total reward and its components.

        Args:
            deliveries_completed: Number of deliveries this step.
            distance_traveled: Total distance traveled this step (km).
            fuel_consumed: Fuel consumed this step.
            sla_violations: Number of SLA violations this step.
            idle_time: Total idle time across fleet this step (minutes).
            fleet_utilization: Average fleet utilization (0-1).
            requests_expired: Number of requests expired this step.
            on_time_deliveries: Number of on-time deliveries this step.

        Returns:
            RewardBreakdown with all components and total.
        """
        breakdown = RewardBreakdown()

        # Positive rewards
        breakdown.delivery_reward = (
            self.config.delivery_reward * deliveries_completed
        )
        breakdown.utilization_reward = (
            self.config.utilization_reward * fleet_utilization
        )

        # Penalties (stored as negative values)
        breakdown.travel_penalty = -(
            self.config.travel_penalty * distance_traveled
        )
        breakdown.fuel_penalty = -(
            self.config.fuel_penalty * fuel_consumed
        )
        breakdown.sla_penalty = -(
            self.config.sla_violation_penalty * sla_violations
        )
        breakdown.idle_penalty = -(
            self.config.idle_penalty * idle_time
        )
        breakdown.expiry_penalty = -(
            self.config.expiry_penalty * requests_expired
        )

        # Total reward
        total = (
            breakdown.delivery_reward
            + breakdown.travel_penalty
            + breakdown.fuel_penalty
            + breakdown.sla_penalty
            + breakdown.idle_penalty
            + breakdown.utilization_reward
            + breakdown.expiry_penalty
        )

        # Apply normalization if enabled
        if self.config.normalize:
            total = self._normalize_reward(total)

        breakdown.total_reward = total
        return breakdown

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running mean and variance.

        Args:
            reward: Raw reward value.

        Returns:
            Normalized reward value.
        """
        self._reward_history.append(reward)

        if len(self._reward_history) < 2:
            return reward

        rewards_array = np.array(self._reward_history)
        self._running_mean = float(np.mean(rewards_array))
        self._running_var = float(np.var(rewards_array)) + 1e-8

        normalized = (reward - self._running_mean) / np.sqrt(self._running_var)
        # Clip to prevent extreme values
        return float(np.clip(normalized, -10.0, 10.0))

    def reset(self) -> None:
        """Reset the reward calculator's running statistics."""
        self._reward_history.clear()
        self._running_mean = 0.0
        self._running_var = 1.0
