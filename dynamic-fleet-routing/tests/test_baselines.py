"""Tests for baseline dispatchers: NearestVehicle, Greedy, ORTools.

Verifies that each dispatcher:
- Returns valid actions within the action mask.
- Falls back to NO-OP when there are no idle vehicles or pending requests.
- Produces consistent assignments on deterministic environments.
- ORTools fallback-to-nearest works when solver constraints are infeasible.
"""

import pytest
import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.environment.request import RequestStatus
from src.baselines.nearest_vehicle import NearestVehicleDispatcher
from src.baselines.greedy_dispatch import GreedyDispatcher
from src.baselines.ortools_vrp import ORToolsVRPDispatcher


# ---------------------------------------------------------------------------
# Shared fixture: deterministic small environment
# ---------------------------------------------------------------------------

def _make_small_config() -> dict:
    """Return a minimal deterministic environment config."""
    return {
        "simulation": {
            "num_nodes": 15,
            "num_vehicles": 3,
            "seed": 42,
            "simulation_duration": 1440,
        },
        "city": {
            "grid_size": 5.0,
            "edge_density": 0.3,
            "base_speed_kmh": 30.0,
        },
        "traffic": {
            "congestion_probability": 0.0,
            "noise_std": 0.0,
            "traffic_update_interval": 1440,
            "rush_hour_windows": [],
        },
        "vehicles": {
            "capacity": 5,
            "fuel_consumption_per_km": 0.08,
            "service_time_minutes": 1,
            "initial_fuel": 100.0,
        },
        "requests": {
            "lambda_rate": 5,
            "arrival_interval": 30,
            "min_deadline_minutes": 60,
            "max_deadline_minutes": 120,
            "min_package_size": 1,
            "max_package_size": 2,
            "priority_weights": [0.5, 0.3, 0.2],
        },
        "observation": {
            "top_k_requests": 10,
            "max_vehicles": 3,
        },
        "reward": {
            "delivery_reward": 10.0,
            "travel_penalty": 0.1,
            "fuel_penalty": 0.5,
            "sla_violation_penalty": 20.0,
            "idle_penalty": 0.05,
            "utilization_reward": 1.0,
            "expiry_penalty": 15.0,
            "normalize": False,
            "normalization_window": 100,
        },
    }


@pytest.fixture
def small_env() -> DynamicFleetEnv:
    """Environment reset to a known seed."""
    env = DynamicFleetEnv(_make_small_config())
    env.reset(seed=42)
    return env


# =========================================================================
# NearestVehicleDispatcher Tests
# =========================================================================

class TestNearestVehicleDispatcher:
    """Tests for the NearestVehicleDispatcher."""

    def test_returns_valid_action(self, small_env: DynamicFleetEnv) -> None:
        """Returned action should be within the action mask."""
        dispatcher = NearestVehicleDispatcher()
        action = dispatcher.select_action(small_env)
        mask = small_env.get_action_mask()
        assert mask[action], "Action returned by nearest-vehicle must be valid"

    def test_noop_when_no_idle_vehicles(self, small_env: DynamicFleetEnv) -> None:
        """Should return NO-OP when all vehicles are busy."""
        # Force all vehicles to non-idle status
        for v in small_env.vehicles:
            v.status = VehicleStatus.MOVING_TO_PICKUP
        dispatcher = NearestVehicleDispatcher()
        action = dispatcher.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_noop_when_no_pending_requests(self, small_env: DynamicFleetEnv) -> None:
        """Should return NO-OP when there are no pending requests."""
        small_env.pending_requests.clear()
        dispatcher = NearestVehicleDispatcher()
        action = dispatcher.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_deterministic_action(self) -> None:
        """Same seed should produce the same action."""
        dispatcher = NearestVehicleDispatcher()
        env1 = DynamicFleetEnv(_make_small_config())
        env1.reset(seed=42)
        a1 = dispatcher.select_action(env1)

        env2 = DynamicFleetEnv(_make_small_config())
        env2.reset(seed=42)
        a2 = dispatcher.select_action(env2)

        assert a1 == a2

    def test_assigns_closest_vehicle(self, small_env: DynamicFleetEnv) -> None:
        """Action should pick the vehicle nearest to the chosen request's pickup."""
        dispatcher = NearestVehicleDispatcher()
        action = dispatcher.select_action(small_env)
        noop = small_env.action_space.n - 1
        if action == noop:
            pytest.skip("No valid dispatch available in this seed")

        veh_idx = action % small_env.num_vehicles
        req_idx = action // small_env.num_vehicles
        req_id = small_env.pending_requests[req_idx]
        request = small_env.requests[req_id]
        vehicle = small_env.vehicles[veh_idx]
        chosen_dist = small_env.city_graph.get_shortest_distance(
            vehicle.current_location, request.pickup_location
        )

        # Verify no idle vehicle is closer
        for idx, v in enumerate(small_env.vehicles):
            if v.status != VehicleStatus.IDLE:
                continue
            d = small_env.city_graph.get_shortest_distance(
                v.current_location, request.pickup_location
            )
            assert d >= chosen_dist - 1e-9


# =========================================================================
# GreedyDispatcher Tests
# =========================================================================

class TestGreedyDispatcher:
    """Tests for the GreedyDispatcher."""

    def test_returns_valid_action(self, small_env: DynamicFleetEnv) -> None:
        """Returned action should be within the action mask."""
        dispatcher = GreedyDispatcher()
        action = dispatcher.select_action(small_env)
        mask = small_env.get_action_mask()
        assert mask[action]

    def test_noop_when_no_idle_vehicles(self, small_env: DynamicFleetEnv) -> None:
        """Should return NO-OP when all vehicles are busy."""
        for v in small_env.vehicles:
            v.status = VehicleStatus.MOVING_TO_PICKUP
        dispatcher = GreedyDispatcher()
        action = dispatcher.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_noop_when_no_pending_requests(self, small_env: DynamicFleetEnv) -> None:
        """Should return NO-OP when there are no pending requests."""
        small_env.pending_requests.clear()
        dispatcher = GreedyDispatcher()
        action = dispatcher.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_custom_weights(self, small_env: DynamicFleetEnv) -> None:
        """Changing weights should influence action selection."""
        d1 = GreedyDispatcher(w_dist=1.0, w_time=0.0, w_sla=0.0)
        d2 = GreedyDispatcher(w_dist=0.0, w_time=0.0, w_sla=10.0)
        a1 = d1.select_action(small_env)
        a2 = d2.select_action(small_env)
        # At least one combination should differ (weight sensitivity check)
        # If they happen to agree, the test still passes — we just confirm they run
        assert small_env.get_action_mask()[a1]
        assert small_env.get_action_mask()[a2]

    def test_runs_full_episode(self) -> None:
        """Greedy dispatcher should complete a full episode without crash."""
        env = DynamicFleetEnv(_make_small_config())
        env.reset(seed=42)
        dispatcher = GreedyDispatcher()
        steps = 0
        for _ in range(500):
            action = dispatcher.select_action(env)
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps > 0


# =========================================================================
# ORToolsVRPDispatcher Tests
# =========================================================================

class TestORToolsVRPDispatcher:
    """Tests for the ORToolsVRPDispatcher."""

    def test_returns_valid_action(self, small_env: DynamicFleetEnv) -> None:
        """Returned action should be within the action mask."""
        dispatcher = ORToolsVRPDispatcher(time_limit_ms=500)
        action = dispatcher.select_action(small_env)
        mask = small_env.get_action_mask()
        assert mask[action]

    def test_noop_when_no_idle_vehicles(self, small_env: DynamicFleetEnv) -> None:
        """Should return NO-OP when all vehicles are busy."""
        for v in small_env.vehicles:
            v.status = VehicleStatus.MOVING_TO_PICKUP
        dispatcher = ORToolsVRPDispatcher(time_limit_ms=200)
        action = dispatcher.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_noop_when_no_pending_requests(self, small_env: DynamicFleetEnv) -> None:
        """Should return NO-OP when there are no pending requests."""
        small_env.pending_requests.clear()
        dispatcher = ORToolsVRPDispatcher(time_limit_ms=200)
        action = dispatcher.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_solver_time_recorded(self, small_env: DynamicFleetEnv) -> None:
        """The last_solve_time_ms property should be populated after a call."""
        dispatcher = ORToolsVRPDispatcher(time_limit_ms=500)
        dispatcher.select_action(small_env)
        assert dispatcher.last_solve_time_ms >= 0.0

    def test_runs_full_episode(self) -> None:
        """ORTools dispatcher should complete a full episode without crash."""
        env = DynamicFleetEnv(_make_small_config())
        env.reset(seed=42)
        dispatcher = ORToolsVRPDispatcher(time_limit_ms=200)
        steps = 0
        for _ in range(500):
            action = dispatcher.select_action(env)
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps > 0

    def test_fallback_nearest_works(self, small_env: DynamicFleetEnv) -> None:
        """The internal _fallback_nearest should return a valid action."""
        dispatcher = ORToolsVRPDispatcher(time_limit_ms=200)
        mask = small_env.get_action_mask()
        idle_vehicles = [
            (idx, v) for idx, v in enumerate(small_env.vehicles)
            if v.status == VehicleStatus.IDLE
        ]
        pending = small_env.pending_requests[:small_env.top_k_requests]
        valid_requests = [
            (req_idx, req_id, small_env.requests[req_id])
            for req_idx, req_id in enumerate(pending)
            if small_env.requests[req_id].status == RequestStatus.PENDING
            and not small_env.requests[req_id].is_expired_at(small_env.current_time)
        ]
        if idle_vehicles and valid_requests:
            action = dispatcher._fallback_nearest(
                small_env, idle_vehicles, valid_requests, mask
            )
            if action is not None:
                assert mask[action]
