"""OR-Tools VRP baseline with rolling-horizon optimization.

Uses Google OR-Tools to solve a snapshot VRP for currently pending
requests, then executes the best assignment. Re-optimizes when
new requests arrive or vehicles become available.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.environment.request import RequestStatus


class ORToolsVRPDispatcher:
    """OR-Tools based rolling-horizon VRP solver.

    Collects pending requests and idle vehicles, builds a distance
    matrix, solves the assignment problem using OR-Tools routing
    library, and returns the best action.

    Args:
        time_limit_ms: Solver time limit in milliseconds.
    """

    def __init__(self, time_limit_ms: int = 1000) -> None:
        """Initialize the OR-Tools dispatcher.

        Args:
            time_limit_ms: Maximum solver time in milliseconds.
        """
        self.name = "ortools_vrp"
        self.time_limit_ms = time_limit_ms
        self._last_solve_time_ms: float = 0.0

    @property
    def last_solve_time_ms(self) -> float:
        """Return the last solve time in milliseconds."""
        return self._last_solve_time_ms

    def select_action(self, env: DynamicFleetEnv) -> int:
        """Select the best action using OR-Tools VRP solver.

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

        # Filter valid pending requests
        valid_requests = []
        for req_idx, req_id in enumerate(pending):
            request = env.requests[req_id]
            if request.status == RequestStatus.PENDING and not request.is_expired_at(env.current_time):
                valid_requests.append((req_idx, req_id, request))

        if not valid_requests:
            return noop_action

        start_time = time.perf_counter()

        try:
            action = self._solve_with_ortools(env, idle_vehicles, valid_requests, mask)
        except Exception:
            # Fallback to nearest-vehicle if OR-Tools fails
            action = self._fallback_nearest(env, idle_vehicles, valid_requests, mask)

        self._last_solve_time_ms = (time.perf_counter() - start_time) * 1000
        return action if action is not None else noop_action

    def _solve_with_ortools(
        self,
        env: DynamicFleetEnv,
        idle_vehicles: list[tuple[int, Any]],
        valid_requests: list[tuple[int, int, Any]],
        mask: np.ndarray,
    ) -> Optional[int]:
        """Solve the assignment problem using OR-Tools.

        Creates a cost matrix where cost[v][r] is the distance from
        vehicle v to request r's pickup, then solves for minimum cost.

        Args:
            env: Fleet environment.
            idle_vehicles: List of (index, vehicle) for idle vehicles.
            valid_requests: List of (req_idx, req_id, request) for valid requests.
            mask: Action mask.

        Returns:
            Best encoded action, or None.
        """
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        num_vehicles = len(idle_vehicles)
        num_requests = len(valid_requests)

        # Decision variables: assign[v][r] = 1 if vehicle v handles request r
        assign = {}
        for v in range(num_vehicles):
            for r in range(num_requests):
                assign[(v, r)] = model.NewBoolVar(f"assign_v{v}_r{r}")

        # Each request assigned to at most one vehicle
        for r in range(num_requests):
            model.Add(sum(assign[(v, r)] for v in range(num_vehicles)) <= 1)

        # Each vehicle assigned at most one request (single-step assignment)
        for v in range(num_vehicles):
            model.Add(sum(assign[(v, r)] for r in range(num_requests)) <= 1)

        # Build cost coefficients (scaled to integers for CP-SAT)
        SCALE = 1000
        costs = {}
        for v_pos, (veh_idx, vehicle) in enumerate(idle_vehicles):
            for r_pos, (req_idx, req_id, request) in enumerate(valid_requests):
                # Check action validity
                action = req_idx * env.num_vehicles + veh_idx
                if action >= env.action_space.n - 1 or not mask[action]:
                    # Invalid: force to 0
                    model.Add(assign[(v_pos, r_pos)] == 0)
                    costs[(v_pos, r_pos)] = 0
                    continue

                if not vehicle.can_accept_package(request.package_size):
                    model.Add(assign[(v_pos, r_pos)] == 0)
                    costs[(v_pos, r_pos)] = 0
                    continue

                # Cost = distance to pickup + pickup-to-dropoff distance
                dist_pickup = env.city_graph.get_shortest_distance(
                    vehicle.current_location, request.pickup_location
                )
                dist_delivery = env.city_graph.get_shortest_distance(
                    request.pickup_location, request.dropoff_location
                )

                # SLA bonus: reduce cost for urgent requests
                sla_remaining = request.sla_remaining_at(env.current_time)
                urgency_bonus = max(0, 50.0 - sla_remaining) * 0.1

                # Priority bonus
                priority_bonus = float(request.priority.value) * 2.0

                total_cost = dist_pickup + dist_delivery * 0.5 - urgency_bonus - priority_bonus
                costs[(v_pos, r_pos)] = int(max(total_cost, 0.1) * SCALE)

        # Objective: minimize total cost while maximizing assignments
        assignment_bonus = int(10.0 * SCALE)  # Encourage making assignments
        model.Minimize(
            sum(costs.get((v, r), 0) * assign[(v, r)]
                for v in range(num_vehicles)
                for r in range(num_requests))
            - assignment_bonus * sum(assign[(v, r)]
                                     for v in range(num_vehicles)
                                     for r in range(num_requests))
        )

        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_ms / 1000.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Find the first assignment
            best_cost = float("inf")
            best_action = None

            for v_pos, (veh_idx, vehicle) in enumerate(idle_vehicles):
                for r_pos, (req_idx, req_id, request) in enumerate(valid_requests):
                    if solver.Value(assign[(v_pos, r_pos)]) == 1:
                        action = req_idx * env.num_vehicles + veh_idx
                        cost = costs.get((v_pos, r_pos), float("inf"))
                        if cost < best_cost:
                            best_cost = cost
                            best_action = action

            return best_action

        return None

    def _fallback_nearest(
        self,
        env: DynamicFleetEnv,
        idle_vehicles: list[tuple[int, Any]],
        valid_requests: list[tuple[int, int, Any]],
        mask: np.ndarray,
    ) -> Optional[int]:
        """Fallback to nearest-vehicle heuristic.

        Args:
            env: Fleet environment.
            idle_vehicles: Idle vehicles.
            valid_requests: Valid pending requests.
            mask: Action mask.

        Returns:
            Best action by nearest distance, or None.
        """
        best_action = None
        best_distance = float("inf")

        for req_idx, req_id, request in valid_requests:
            for veh_idx, vehicle in idle_vehicles:
                action = req_idx * env.num_vehicles + veh_idx
                if action >= env.action_space.n - 1 or not mask[action]:
                    continue
                dist = env.city_graph.get_shortest_distance(
                    vehicle.current_location, request.pickup_location
                )
                if dist < best_distance:
                    best_distance = dist
                    best_action = action

        return best_action
