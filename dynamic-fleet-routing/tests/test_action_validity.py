"""Tests for action masking and validation in the fleet environment."""

import pytest
import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.environment.request import RequestStatus


class TestActionMasking:
    """Tests for action space masking in DynamicFleetEnv."""

    @pytest.fixture
    def small_env(self) -> DynamicFleetEnv:
        """Create a small environment for testing."""
        config = {
            "simulation": {"num_nodes": 10, "num_vehicles": 3, "seed": 42,
                           "simulation_duration": 1440},
            "city": {"grid_size": 5.0, "edge_density": 0.3, "base_speed_kmh": 30.0},
            "traffic": {"congestion_probability": 0.0, "noise_std": 0.0,
                        "traffic_update_interval": 1440,
                        "rush_hour_windows": []},
            "vehicles": {"capacity": 5, "fuel_consumption_per_km": 0.08,
                         "service_time_minutes": 1, "initial_fuel": 100.0},
            "requests": {"lambda_rate": 3, "arrival_interval": 30,
                         "min_deadline_minutes": 60, "max_deadline_minutes": 120,
                         "min_package_size": 1, "max_package_size": 2,
                         "priority_weights": [0.5, 0.3, 0.2]},
            "observation": {"top_k_requests": 10, "max_vehicles": 3},
            "reward": {"delivery_reward": 10.0, "travel_penalty": 0.1,
                       "fuel_penalty": 0.5, "sla_violation_penalty": 20.0,
                       "idle_penalty": 0.05, "utilization_reward": 1.0,
                       "expiry_penalty": 15.0, "normalize": False,
                       "normalization_window": 100},
        }
        env = DynamicFleetEnv(config)
        return env

    def test_action_mask_shape(self, small_env: DynamicFleetEnv) -> None:
        """Action mask should have correct shape."""
        small_env.reset(seed=42)
        mask = small_env.get_action_mask()
        assert mask.shape == (small_env.action_space.n,)
        assert mask.dtype == bool

    def test_noop_always_valid(self, small_env: DynamicFleetEnv) -> None:
        """NO-OP action should always be valid."""
        small_env.reset(seed=42)
        mask = small_env.get_action_mask()
        assert mask[-1] is np.bool_(True)

    def test_at_least_one_valid_action(self, small_env: DynamicFleetEnv) -> None:
        """There should always be at least one valid action (NO-OP)."""
        small_env.reset(seed=42)
        mask = small_env.get_action_mask()
        assert mask.any()

    def test_invalid_action_returns_penalty(self, small_env: DynamicFleetEnv) -> None:
        """Taking an invalid action should not crash but return penalty."""
        small_env.reset(seed=42)
        # Try an action with very high request index (likely invalid)
        invalid_action = small_env.action_space.n - 2  # Just before NO-OP
        mask = small_env.get_action_mask()
        if not mask[invalid_action]:
            obs, reward, term, trunc, info = small_env.step(invalid_action)
            assert info.get("invalid_action", False) is True

    def test_noop_advances_simulation(self, small_env: DynamicFleetEnv) -> None:
        """NO-OP should advance the simulation without crashes."""
        obs, info = small_env.reset(seed=42)
        noop = small_env.action_space.n - 1
        obs, reward, term, trunc, info = small_env.step(noop)
        assert not term or info["current_time"] >= 1440

    def test_valid_action_assigns_request(self, small_env: DynamicFleetEnv) -> None:
        """A valid action should assign a request to a vehicle."""
        small_env.reset(seed=42)
        mask = small_env.get_action_mask()

        # Find a valid non-NOOP action
        valid_actions = np.where(mask[:-1])[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            request_idx = action // small_env.num_vehicles
            vehicle_idx = action % small_env.num_vehicles

            req_id = small_env.pending_requests[request_idx]
            obs, reward, term, trunc, info = small_env.step(action)
            request = small_env.requests[req_id]
            assert request.status != RequestStatus.PENDING

    def test_observation_shape(self, small_env: DynamicFleetEnv) -> None:
        """Observation should match observation space shape."""
        obs, info = small_env.reset(seed=42)
        assert obs.shape == small_env.observation_space.shape

    def test_observation_bounds(self, small_env: DynamicFleetEnv) -> None:
        """Observation values should be within bounds."""
        obs, info = small_env.reset(seed=42)
        assert np.all(obs >= -1.0 - 1e-6)
        assert np.all(obs <= 1.0 + 1e-6)
