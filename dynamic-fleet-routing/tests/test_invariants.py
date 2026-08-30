"""Invariant tests for the fleet routing simulation.

These property-based tests verify critical invariants that must hold
at all times during simulation, regardless of actions taken.
"""

import pytest
import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.environment.request import RequestStatus


class TestSimulationInvariants:
    """Tests for simulation invariants that must always hold."""

    @pytest.fixture
    def env(self) -> DynamicFleetEnv:
        """Create a test environment."""
        config = {
            "simulation": {"num_nodes": 15, "num_vehicles": 3, "seed": 42,
                           "simulation_duration": 300},
            "city": {"grid_size": 5.0, "edge_density": 0.3, "base_speed_kmh": 30.0},
            "traffic": {"congestion_probability": 0.0, "noise_std": 0.0,
                        "traffic_update_interval": 60,
                        "rush_hour_windows": []},
            "vehicles": {"capacity": 5, "fuel_consumption_per_km": 0.08,
                         "service_time_minutes": 2, "initial_fuel": 100.0},
            "requests": {"lambda_rate": 3, "arrival_interval": 20,
                         "min_deadline_minutes": 30, "max_deadline_minutes": 90,
                         "min_package_size": 1, "max_package_size": 2,
                         "priority_weights": [0.5, 0.3, 0.2]},
            "observation": {"top_k_requests": 10, "max_vehicles": 3},
            "reward": {"delivery_reward": 10.0, "travel_penalty": 0.1,
                       "fuel_penalty": 0.5, "sla_violation_penalty": 20.0,
                       "idle_penalty": 0.05, "utilization_reward": 1.0,
                       "expiry_penalty": 15.0, "normalize": False,
                       "normalization_window": 100},
        }
        return DynamicFleetEnv(config)

    def _run_episode(self, env: DynamicFleetEnv, max_steps: int = 200) -> None:
        """Run a full episode using random valid actions."""
        obs, info = env.reset(seed=42)
        rng = np.random.RandomState(42)

        for _ in range(max_steps):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    def test_vehicle_capacity_never_exceeded(self, env: DynamicFleetEnv) -> None:
        """Vehicle load must never exceed capacity."""
        obs, info = env.reset(seed=42)
        rng = np.random.RandomState(42)

        for _ in range(100):
            # Check invariant
            for v in env.vehicles:
                assert v.current_load <= v.capacity, (
                    f"Vehicle {v.vehicle_id} load {v.current_load} "
                    f"exceeds capacity {v.capacity}"
                )

            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

    def test_simulation_time_never_moves_backward(self, env: DynamicFleetEnv) -> None:
        """Simulation time must be monotonically non-decreasing."""
        obs, info = env.reset(seed=42)
        previous_time = env.current_time
        rng = np.random.RandomState(42)

        for _ in range(100):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)
            obs, reward, term, trunc, info = env.step(action)

            assert env.current_time >= previous_time, (
                f"Time went backward: {previous_time} -> {env.current_time}"
            )
            previous_time = env.current_time

            if term or trunc:
                break

    def test_delivered_requests_have_delivery_time(self, env: DynamicFleetEnv) -> None:
        """Delivered requests must have a delivery_time set."""
        self._run_episode(env)

        for req in env.requests.values():
            if req.status == RequestStatus.DELIVERED:
                assert req.delivery_time is not None, (
                    f"Request {req.request_id} is DELIVERED but has no delivery_time"
                )

    def test_no_duplicate_deliveries(self, env: DynamicFleetEnv) -> None:
        """Each request should be delivered at most once."""
        self._run_episode(env)

        delivered_ids = [
            req.request_id
            for req in env.requests.values()
            if req.status == RequestStatus.DELIVERED
        ]
        assert len(delivered_ids) == len(set(delivered_ids)), (
            "Duplicate deliveries detected"
        )

    def test_fuel_consumption_non_negative(self, env: DynamicFleetEnv) -> None:
        """Fuel consumption must be non-negative."""
        self._run_episode(env)

        for v in env.vehicles:
            assert v.fuel_consumed >= 0, (
                f"Vehicle {v.vehicle_id} has negative fuel consumed: {v.fuel_consumed}"
            )

    def test_expired_requests_not_delivered(self, env: DynamicFleetEnv) -> None:
        """Expired requests cannot be in DELIVERED state."""
        self._run_episode(env)

        for req in env.requests.values():
            # A request cannot be both expired and delivered
            assert not (req.status == RequestStatus.EXPIRED and req.delivery_time is not None), (
                f"Request {req.request_id} is EXPIRED but has delivery_time"
            )

    def test_state_clone_independence(self, env: DynamicFleetEnv) -> None:
        """Cloned state should be independent of original."""
        obs, info = env.reset(seed=42)

        # Take a few steps
        rng = np.random.RandomState(42)
        for _ in range(5):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)
            env.step(action)

        # Clone state
        state = env.clone_state()
        time_before = env.current_time
        deliveries_before = env._total_deliveries

        # Take more steps in original env
        for _ in range(10):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

        # Restore state
        env.restore_state(state)
        assert env.current_time == time_before
        assert env._total_deliveries == deliveries_before

    def test_observation_consistent_across_steps(self, env: DynamicFleetEnv) -> None:
        """Observations should always be finite and within bounds."""
        obs, info = env.reset(seed=42)
        rng = np.random.RandomState(42)

        for _ in range(50):
            assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
            assert obs.shape == env.observation_space.shape

            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

    def test_total_deliveries_consistent(self, env: DynamicFleetEnv) -> None:
        """Total deliveries in info should match request count."""
        self._run_episode(env)

        delivered_count = sum(
            1 for r in env.requests.values()
            if r.status == RequestStatus.DELIVERED
        )
        assert env._total_deliveries == delivered_count
