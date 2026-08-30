"""Dynamic request generator for the fleet routing simulation.

Generates pickup/delivery requests according to a Poisson process
with configurable arrival rates and demand profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.environment.city_graph import CityGraph
from src.environment.request import Priority, Request


@dataclass
class RequestGeneratorConfig:
    """Configuration for the request generator.

    Attributes:
        lambda_rate: Average requests per arrival interval.
        arrival_interval: Minutes between arrival batches.
        min_deadline_minutes: Minimum SLA deadline from request time.
        max_deadline_minutes: Maximum SLA deadline from request time.
        min_package_size: Minimum package size in units.
        max_package_size: Maximum package size in units.
        priority_weights: Probability distribution for [LOW, MEDIUM, HIGH].
        seed: Random seed for reproducibility.
    """
    lambda_rate: int = 5
    arrival_interval: float = 30.0
    min_deadline_minutes: float = 30.0
    max_deadline_minutes: float = 120.0
    min_package_size: int = 1
    max_package_size: int = 3
    priority_weights: list[float] = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.priority_weights is None:
            self.priority_weights = [0.6, 0.3, 0.1]


class DemandProfile:
    """Models time-of-day demand variation.

    Scales the base Poisson rate based on the time of day to simulate
    realistic demand patterns with peaks during business hours.
    """

    # Demand multipliers by hour of day (0-23)
    HOURLY_MULTIPLIERS: list[float] = [
        0.1, 0.05, 0.05, 0.05, 0.1, 0.2,    # 00:00-05:59 (night)
        0.4, 0.6, 0.8, 1.0, 1.2, 1.5,        # 06:00-11:59 (morning ramp)
        1.8, 1.5, 1.3, 1.2, 1.4, 1.6,        # 12:00-17:59 (afternoon)
        1.8, 1.5, 1.0, 0.6, 0.3, 0.2,        # 18:00-23:59 (evening wind-down)
    ]

    @classmethod
    def get_demand_multiplier(cls, time_minutes: float) -> float:
        """Get the demand multiplier for a given time of day.

        Args:
            time_minutes: Time in minutes from midnight.

        Returns:
            Demand multiplier (0.05 to 1.8).
        """
        hour = int((time_minutes % 1440) / 60)
        hour = min(hour, 23)
        return cls.HOURLY_MULTIPLIERS[hour]


class RequestGenerator:
    """Generates dynamic delivery requests using a Poisson process.

    Produces requests at configurable rates with time-of-day demand
    variation, random pickup/dropoff locations, deadlines, and priorities.

    Attributes:
        config: Generator configuration.
        city_graph: City graph for selecting locations.
        next_request_id: Counter for generating unique request IDs.
        next_arrival_time: Simulation time of next request batch.
        total_generated: Total requests generated.
    """

    def __init__(
        self,
        city_graph: CityGraph,
        config: Optional[RequestGeneratorConfig] = None,
    ) -> None:
        """Initialize the request generator.

        Args:
            city_graph: City graph for selecting pickup/dropoff locations.
            config: Generator configuration. Uses defaults if None.
        """
        self.config = config or RequestGeneratorConfig()
        self.city_graph = city_graph
        self._rng = np.random.RandomState(self.config.seed)
        self.next_request_id: int = 0
        self.next_arrival_time: float = 0.0
        self.total_generated: int = 0

    def generate_batch(self, current_time: float) -> list[Request]:
        """Generate a batch of requests arriving at the current time.

        Uses a Poisson process scaled by time-of-day demand to determine
        the number of requests in this batch.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            List of newly generated Request objects.
        """
        # Scale lambda by time-of-day demand
        demand_multiplier = DemandProfile.get_demand_multiplier(current_time)
        effective_lambda = max(1, self.config.lambda_rate * demand_multiplier)

        # Draw number of arrivals from Poisson distribution
        num_requests = self._rng.poisson(effective_lambda)

        requests: list[Request] = []
        for _ in range(num_requests):
            request = self._create_request(current_time)
            requests.append(request)
            self.total_generated += 1

        return requests

    def _create_request(self, current_time: float) -> Request:
        """Create a single delivery request with random attributes.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            A new Request object.
        """
        # Select distinct pickup and dropoff locations
        pickup, dropoff = self.city_graph.get_random_node_pair(self._rng)

        # Generate deadline
        deadline_offset = self._rng.uniform(
            self.config.min_deadline_minutes,
            self.config.max_deadline_minutes,
        )
        deadline = current_time + deadline_offset

        # Generate priority
        priority_idx = self._rng.choice(
            len(self.config.priority_weights),
            p=self.config.priority_weights,
        )
        priority = Priority(priority_idx)

        # Generate package size
        package_size = self._rng.randint(
            self.config.min_package_size,
            self.config.max_package_size + 1,
        )

        request = Request(
            request_id=self.next_request_id,
            pickup_location=pickup,
            dropoff_location=dropoff,
            request_time=current_time,
            deadline=deadline,
            priority=priority,
            package_size=package_size,
        )

        self.next_request_id += 1
        return request

    def should_generate(self, current_time: float) -> bool:
        """Check if it's time to generate a new batch of requests.

        Args:
            current_time: Current simulation time in minutes.

        Returns:
            True if a new batch should be generated.
        """
        return current_time >= self.next_arrival_time

    def advance_arrival_time(self) -> float:
        """Advance the next arrival time by the configured interval.

        Returns:
            The new next arrival time.
        """
        self.next_arrival_time += self.config.arrival_interval
        return self.next_arrival_time

    def get_next_arrival_time(self) -> float:
        """Return the scheduled time for the next request batch."""
        return self.next_arrival_time

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the generator to initial state.

        Args:
            seed: Optional new random seed.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.next_request_id = 0
        self.next_arrival_time = 0.0
        self.total_generated = 0
