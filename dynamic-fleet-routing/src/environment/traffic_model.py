"""Stochastic traffic model for the fleet routing simulation.

Models time-of-day traffic patterns, rush hours, random congestion events,
and provides travel time multipliers for the city graph edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TrafficState(Enum):
    """Discrete traffic conditions."""
    LOW_TRAFFIC = "LOW_TRAFFIC"
    NORMAL_TRAFFIC = "NORMAL_TRAFFIC"
    HEAVY_TRAFFIC = "HEAVY_TRAFFIC"
    CONGESTION = "CONGESTION"


# Multipliers applied to base travel time for each traffic state
TRAFFIC_MULTIPLIERS: dict[TrafficState, float] = {
    TrafficState.LOW_TRAFFIC: 0.8,
    TrafficState.NORMAL_TRAFFIC: 1.0,
    TrafficState.HEAVY_TRAFFIC: 1.5,
    TrafficState.CONGESTION: 2.5,
}


@dataclass
class RushHourWindow:
    """Defines a rush hour period with elevated traffic.

    Attributes:
        start: Start time in minutes from midnight.
        end: End time in minutes from midnight.
        multiplier: Additional traffic multiplier during this window.
    """
    start: float
    end: float
    multiplier: float = 2.0


@dataclass
class TrafficConfig:
    """Configuration for the traffic model.

    Attributes:
        rush_hour_windows: List of rush hour periods.
        congestion_probability: Probability of random congestion per update.
        noise_std: Standard deviation of travel time noise.
        traffic_update_interval: Minutes between traffic state updates.
        seed: Random seed for reproducibility.
    """
    rush_hour_windows: list[RushHourWindow] = field(default_factory=lambda: [
        RushHourWindow(start=480, end=600, multiplier=2.0),
        RushHourWindow(start=1020, end=1200, multiplier=2.5),
    ])
    congestion_probability: float = 0.05
    noise_std: float = 0.1
    traffic_update_interval: float = 30.0
    seed: int = 42


class TrafficModel:
    """Stochastic traffic model with time-of-day patterns and random events.

    The traffic model determines travel time multipliers based on:
    - Time of day (rush hours vs off-peak)
    - Random congestion events
    - Gaussian noise for travel time variability

    Attributes:
        config: Traffic model configuration.
        current_state: Current global traffic state.
        last_update_time: Simulation time of last traffic update.
    """

    def __init__(self, config: Optional[TrafficConfig] = None) -> None:
        """Initialize the traffic model.

        Args:
            config: Traffic configuration. Uses defaults if None.
        """
        self.config = config or TrafficConfig()
        self._rng = np.random.RandomState(self.config.seed)
        self.current_state: TrafficState = TrafficState.NORMAL_TRAFFIC
        self.last_update_time: float = 0.0
        # Per-edge congestion tracking (edge_key -> congestion_end_time)
        self._edge_congestion: dict[tuple[int, int], float] = {}

    def get_traffic_state(self, current_time: float) -> TrafficState:
        """Determine the traffic state for a given simulation time.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            The current traffic state.
        """
        time_of_day = current_time % 1440  # Wrap to 24-hour cycle

        # Check if we're in a rush hour window
        in_rush_hour = False
        max_rush_multiplier = 1.0
        for window in self.config.rush_hour_windows:
            if window.start <= time_of_day <= window.end:
                in_rush_hour = True
                max_rush_multiplier = max(max_rush_multiplier, window.multiplier)

        # Check for random congestion
        has_congestion = self._rng.random() < self.config.congestion_probability

        if has_congestion:
            return TrafficState.CONGESTION
        elif in_rush_hour and max_rush_multiplier >= 2.0:
            return TrafficState.HEAVY_TRAFFIC
        elif in_rush_hour:
            return TrafficState.HEAVY_TRAFFIC
        elif time_of_day < 360 or time_of_day > 1320:
            # Late night / early morning: low traffic
            return TrafficState.LOW_TRAFFIC
        else:
            return TrafficState.NORMAL_TRAFFIC

    def get_travel_time_multiplier(
        self,
        current_time: float,
        source: int = 0,
        target: int = 0,
    ) -> float:
        """Calculate the travel time multiplier for current conditions.

        Combines the global traffic state multiplier with Gaussian noise
        and any per-edge congestion effects.

        Args:
            current_time: Current simulation time in minutes.
            source: Source node ID (for edge-specific congestion).
            target: Target node ID (for edge-specific congestion).

        Returns:
            Multiplier to apply to base travel time. Always >= 0.5.
        """
        state = self.get_traffic_state(current_time)
        base_multiplier = TRAFFIC_MULTIPLIERS[state]

        # Add Gaussian noise
        noise = self._rng.normal(0, self.config.noise_std)
        multiplier = base_multiplier + noise

        # Check edge-specific congestion
        edge_key = (min(source, target), max(source, target))
        if edge_key in self._edge_congestion:
            if current_time < self._edge_congestion[edge_key]:
                multiplier *= 1.5  # Additional congestion on specific edge
            else:
                del self._edge_congestion[edge_key]

        # Clamp to reasonable range
        return max(0.5, min(multiplier, 4.0))

    def update(self, current_time: float) -> TrafficState:
        """Update the traffic model state.

        Should be called periodically at the configured update interval.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            Updated traffic state.
        """
        self.current_state = self.get_traffic_state(current_time)
        self.last_update_time = current_time

        # Randomly generate edge-specific congestion events
        if self._rng.random() < self.config.congestion_probability:
            # Congestion lasts 15-45 minutes
            duration = self._rng.uniform(15, 45)
            # Use random edge keys (will be matched if they exist in graph)
            edge = (
                self._rng.randint(0, 100),
                self._rng.randint(0, 100),
            )
            edge_key = (min(edge), max(edge))
            self._edge_congestion[edge_key] = current_time + duration

        return self.current_state

    def should_update(self, current_time: float) -> bool:
        """Check if the traffic model should be updated.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            True if enough time has passed since last update.
        """
        return (current_time - self.last_update_time) >= self.config.traffic_update_interval

    def get_state_name(self) -> str:
        """Return the current traffic state as a string."""
        return self.current_state.value

    def get_current_multiplier(self) -> float:
        """Return the base multiplier for the current traffic state."""
        return TRAFFIC_MULTIPLIERS[self.current_state]

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the traffic model to initial state.

        Args:
            seed: Optional new random seed.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.current_state = TrafficState.NORMAL_TRAFFIC
        self.last_update_time = 0.0
        self._edge_congestion.clear()

    def clone(self) -> TrafficModel:
        """Create a copy of the traffic model for MCTS planning.

        Returns:
            A new TrafficModel with copied state. Note: the RNG state
            is shared for efficiency, which is acceptable for MCTS
            since we only care about the deterministic state.
        """
        cloned = TrafficModel(self.config)
        cloned.current_state = self.current_state
        cloned.last_update_time = self.last_update_time
        cloned._edge_congestion = dict(self._edge_congestion)
        cloned._rng = np.random.RandomState(self._rng.randint(0, 2**31))
        return cloned
