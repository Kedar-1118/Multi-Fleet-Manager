"""Greedy dispatch baseline.

Scores candidate request-vehicle pairs using a weighted combination
of distance to pickup, estimated delivery time, and SLA risk,
then selects the lowest-score assignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.environment.request import RequestStatus


class GreedyDispatcher:
    """Greedy baseline that scores assignments by a heuristic cost function.

    Score = w_dist * distance_to_pickup
          + w_time * estimated_delivery_time
          + w_sla  * sla_urgency

    Lower scores are preferred.

    Args:
        w_dist: Weight for distance to pickup.
        w_time: Weight for estimated total delivery time.
        w_sla: Weight for SLA urgency (inverse of remaining time).
    """

    def __init__(
        self,
        w_dist: float = 1.0,
        w_time: float = 0.5,
        w_sla: float = 2.0,
    ) -> None:
        """Initialize the greedy dispatcher.

        Args:
            w_dist: Weight for pickup distance.
            w_time: Weight for delivery time estimate.
            w_sla: Weight for SLA urgency.
        """
        self.name = "greedy"
        self.w_dist = w_dist
        self.w_time = w_time
        self.w_sla = w_sla

    def select_action(self, env: DynamicFleetEnv) -> int:
        """Select the best action using greedy scoring.

        Args:
            env: The fleet environment.

        Returns:
            Encoded action index, or NO-OP if no valid assignment.
        """
        noop_action = env.action_space.n - 1
        mask = env.get_action_mask()

        idle_vehicles = [
            (idx, v) for idx, v in enumerate(env.vehicles)
            if v.status == VehicleStatus.IDLE
        ]
        if not idle_vehicles:
            return noop_action

        pending = env.pending_requests[:env.top_k_requests]
        if not pending:
            return noop_action

        best_action = noop_action
        best_score = float("inf")
        traffic_mult = env.traffic_model.get_current_multiplier()

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

                # Distance to pickup
                dist_to_pickup = env.city_graph.get_shortest_distance(
                    vehicle.current_location, request.pickup_location
                )

                # Estimated delivery time (pickup travel + service + dropoff travel)
                pickup_time = env.city_graph.get_travel_time(
                    vehicle.current_location, request.pickup_location, traffic_mult
                )
                dropoff_dist = env.city_graph.get_shortest_distance(
                    request.pickup_location, request.dropoff_location
                )
                dropoff_time = env.city_graph.get_travel_time(
                    request.pickup_location, request.dropoff_location, traffic_mult
                )
                est_delivery_time = pickup_time + 5.0 + dropoff_time  # 5 min service

                # SLA urgency: higher urgency for requests closer to deadline
                sla_remaining = request.sla_remaining_at(env.current_time)
                sla_urgency = 1.0 / max(sla_remaining, 1.0)

                # Priority boost: high priority requests get lower score
                priority_factor = 1.0 - (float(request.priority.value) * 0.2)

                score = (
                    self.w_dist * dist_to_pickup
                    + self.w_time * est_delivery_time
                    - self.w_sla * sla_urgency * 100.0  # Negative = prefer urgent
                ) * priority_factor

                if score < best_score:
                    best_score = score
                    best_action = action

        return best_action
