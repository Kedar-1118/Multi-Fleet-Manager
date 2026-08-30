"""Nearest vehicle baseline dispatcher.

Assigns each pending request to the nearest idle vehicle
by shortest-path distance in the city graph.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.environment.request import RequestStatus


class NearestVehicleDispatcher:
    """Baseline that assigns requests to the closest available vehicle.

    For each pending request, finds the idle vehicle with minimum
    travel distance to the pickup location and returns the corresponding
    encoded action.
    """

    def __init__(self) -> None:
        """Initialize the dispatcher."""
        self.name = "nearest_vehicle"

    def select_action(self, env: DynamicFleetEnv) -> int:
        """Select the best request-vehicle pair based on proximity.

        Args:
            env: The fleet environment.

        Returns:
            Encoded action index, or NO-OP if no valid assignment.
        """
        noop_action = env.action_space.n - 1
        mask = env.get_action_mask()

        # Get idle vehicles
        idle_vehicles = [
            (idx, v) for idx, v in enumerate(env.vehicles)
            if v.status == VehicleStatus.IDLE
        ]
        if not idle_vehicles:
            return noop_action

        # Get pending requests (top K)
        pending = env.pending_requests[:env.top_k_requests]
        if not pending:
            return noop_action

        best_action = noop_action
        best_distance = float("inf")

        for req_idx, req_id in enumerate(pending):
            request = env.requests[req_id]
            if request.status != RequestStatus.PENDING:
                continue
            if request.is_expired_at(env.current_time):
                continue

            for veh_idx, vehicle in idle_vehicles:
                action = req_idx * env.num_vehicles + veh_idx
                if action >= env.action_space.n - 1:
                    continue
                if not mask[action]:
                    continue

                distance = env.city_graph.get_shortest_distance(
                    vehicle.current_location, request.pickup_location
                )
                if distance < best_distance:
                    best_distance = distance
                    best_action = action

        return best_action
