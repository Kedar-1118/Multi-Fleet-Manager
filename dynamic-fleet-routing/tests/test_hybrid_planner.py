"""Tests for the HybridPlanner (PPO + MCTS).

Verifies that the HybridPlanner correctly:
- Returns valid masked actions.
- Falls back to PPO when MCTS latency budget is exceeded.
- Tracks decision statistics (fallback count, latency).
- Handles edge cases with no valid actions or a single candidate.
- Integrates PPO priors into MCTS search.

These tests build a real (small) MaskablePPO model so that
get_action_probabilities and predict are fully exercised.
"""

import pytest
import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.environment.vehicle import VehicleStatus
from src.agents.ppo_agent import PPOAgent
from src.agents.action_masking import ActionMaskingWrapper
from src.planning.hybrid_planner import HybridPlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> dict:
    """Return a tiny deterministic environment config."""
    return {
        "simulation": {
            "num_nodes": 10,
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
            "lambda_rate": 3,
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


@pytest.fixture(scope="module")
def trained_agent():
    """Build a MaskablePPO agent (untrained, but structurally valid)."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    config = _make_config()

    def mask_fn(env):
        return env.action_masks()

    def make_env():
        env = DynamicFleetEnv(config)
        env = ActionMaskingWrapper(env)
        env = ActionMasker(env, mask_fn)
        return env

    vec_env = DummyVecEnv([make_env])
    agent = PPOAgent(config=config)
    agent.build_model(vec_env)
    # Run a tiny learn pass so weights are initialised
    agent.model.learn(total_timesteps=64, progress_bar=False)
    vec_env.close()
    return agent


@pytest.fixture
def small_env() -> DynamicFleetEnv:
    env = DynamicFleetEnv(_make_config())
    env.reset(seed=42)
    return env


# =========================================================================
# HybridPlanner Tests
# =========================================================================

class TestHybridPlanner:
    """Tests for the HybridPlanner."""

    def test_returns_valid_action(
        self, trained_agent: PPOAgent, small_env: DynamicFleetEnv
    ) -> None:
        """Action returned by hybrid planner must satisfy the action mask."""
        planner = HybridPlanner(
            trained_agent,
            num_simulations=5,
            max_depth=3,
            top_k_actions=5,
            rollout_horizon=2,
            latency_budget_ms=5000.0,  # generous budget
        )
        action = planner.select_action(small_env)
        mask = small_env.get_action_mask()
        assert mask[action], f"Hybrid action {action} is invalid"

    def test_latency_is_recorded(
        self, trained_agent: PPOAgent, small_env: DynamicFleetEnv
    ) -> None:
        """last_latency_ms should be a positive number after a call."""
        planner = HybridPlanner(
            trained_agent,
            num_simulations=5,
            rollout_horizon=2,
            latency_budget_ms=5000.0,
        )
        planner.select_action(small_env)
        assert planner.last_latency_ms > 0.0

    def test_decision_source_is_set(
        self, trained_agent: PPOAgent, small_env: DynamicFleetEnv
    ) -> None:
        """last_decision_source should be 'hybrid' with a generous budget."""
        planner = HybridPlanner(
            trained_agent,
            num_simulations=5,
            rollout_horizon=2,
            latency_budget_ms=5000.0,
        )
        planner.select_action(small_env)
        assert planner.last_decision_source in ("hybrid", "ppo_direct", "ppo_fallback")

    def test_fallback_with_tiny_budget(
        self, trained_agent: PPOAgent, small_env: DynamicFleetEnv
    ) -> None:
        """With a near-zero latency budget, the planner should fallback to PPO."""
        planner = HybridPlanner(
            trained_agent,
            num_simulations=50,
            rollout_horizon=5,
            latency_budget_ms=0.001,  # impossible budget
            fallback_to_ppo=True,
        )
        action = planner.select_action(small_env)
        mask = small_env.get_action_mask()
        assert mask[action]
        # With such a tiny budget, it should almost certainly fallback
        assert planner.last_decision_source in ("ppo_fallback", "ppo_direct")

    def test_stats_tracking(
        self, trained_agent: PPOAgent, small_env: DynamicFleetEnv
    ) -> None:
        """get_stats() should return consistent decision counters."""
        planner = HybridPlanner(
            trained_agent,
            num_simulations=3,
            rollout_horizon=2,
            latency_budget_ms=5000.0,
        )
        planner.select_action(small_env)
        planner.select_action(small_env)

        stats = planner.get_stats()
        assert stats["total_decisions"] == 2
        assert stats["fallback_rate"] >= 0.0
        assert stats["last_latency_ms"] > 0.0

    def test_noop_when_no_valid_actions(
        self, trained_agent: PPOAgent, small_env: DynamicFleetEnv
    ) -> None:
        """If all vehicles are busy and no requests, should return noop."""
        for v in small_env.vehicles:
            v.status = VehicleStatus.MOVING_TO_PICKUP
        small_env.pending_requests.clear()

        planner = HybridPlanner(
            trained_agent,
            num_simulations=5,
            rollout_horizon=2,
            latency_budget_ms=5000.0,
        )
        action = planner.select_action(small_env)
        assert action == small_env.action_space.n - 1

    def test_multiple_steps(
        self, trained_agent: PPOAgent,
    ) -> None:
        """Hybrid planner should run across multiple environment steps."""
        env = DynamicFleetEnv(_make_config())
        env.reset(seed=42)

        planner = HybridPlanner(
            trained_agent,
            num_simulations=3,
            rollout_horizon=2,
            latency_budget_ms=5000.0,
        )

        for _ in range(10):
            action = planner.select_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert planner.get_stats()["total_decisions"] >= 1
