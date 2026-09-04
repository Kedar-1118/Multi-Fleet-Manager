"""Smoke tests for the PPO training pipeline.

Verifies that the key training components initialise, run a tiny
training loop, save/load models, and that evaluation metrics are
collected — all without requiring long training.
"""

import os
import tempfile

import pytest
import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.action_masking import ActionMaskingWrapper
from src.training.evaluate import evaluate_dispatcher
from src.utils.config import (
    load_base_config,
    _default_config,
    deep_merge,
    get_nested,
)
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_config() -> dict:
    """Small config for fast tests."""
    return {
        "simulation": {
            "num_nodes": 10,
            "num_vehicles": 2,
            "seed": 42,
            "simulation_duration": 300,
        },
        "city": {
            "grid_size": 5.0,
            "edge_density": 0.3,
            "base_speed_kmh": 30.0,
        },
        "traffic": {
            "congestion_probability": 0.0,
            "noise_std": 0.0,
            "traffic_update_interval": 300,
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
            "max_vehicles": 2,
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


# =========================================================================
# Config utility tests
# =========================================================================

class TestConfigUtils:
    """Tests for config.py utilities."""

    def test_default_config_has_all_sections(self) -> None:
        cfg = _default_config()
        for section in ("simulation", "city", "traffic", "vehicles",
                        "requests", "observation", "reward"):
            assert section in cfg, f"Missing section: {section}"

    def test_deep_merge_override(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        over = {"a": {"y": 99}, "c": 4}
        merged = deep_merge(base, over)
        assert merged["a"]["x"] == 1
        assert merged["a"]["y"] == 99
        assert merged["b"] == 3
        assert merged["c"] == 4

    def test_deep_merge_does_not_mutate(self) -> None:
        base = {"a": {"x": 1}}
        over = {"a": {"x": 2}}
        deep_merge(base, over)
        assert base["a"]["x"] == 1  # original unchanged

    def test_get_nested(self) -> None:
        cfg = {"simulation": {"num_vehicles": 5}}
        assert get_nested(cfg, "simulation.num_vehicles") == 5
        assert get_nested(cfg, "simulation.missing", 42) == 42
        assert get_nested(cfg, "nonexistent.path", None) is None

    def test_load_base_config_returns_dict(self) -> None:
        cfg = load_base_config()
        assert isinstance(cfg, dict)
        assert "simulation" in cfg


# =========================================================================
# PPO Agent construction & mini-train tests
# =========================================================================

class TestPPOAgentSmoke:
    """Smoke tests for PPOAgent build, train, save, load cycle."""

    @pytest.fixture(scope="class")
    def vec_env(self):
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv

        config = _make_tiny_config()

        def mask_fn(env):
            return env.action_masks()

        def make():
            env = DynamicFleetEnv(config)
            env = ActionMaskingWrapper(env)
            env = ActionMasker(env, mask_fn)
            return env

        venv = DummyVecEnv([make])
        yield venv
        venv.close()

    def test_build_model(self, vec_env) -> None:
        """build_model should succeed and populate agent.model."""
        agent = PPOAgent(config=_make_tiny_config())
        agent.build_model(vec_env)
        assert agent.model is not None

    def test_mini_train(self, vec_env) -> None:
        """A 64-step train should complete without error."""
        agent = PPOAgent(config=_make_tiny_config())
        agent.build_model(vec_env)
        agent.model.learn(total_timesteps=64, progress_bar=False)

    def test_save_and_load(self, vec_env) -> None:
        """Save then load should produce a usable model."""
        agent = PPOAgent(config=_make_tiny_config())
        agent.build_model(vec_env)
        agent.model.learn(total_timesteps=64, progress_bar=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model")
            agent.save(path)
            assert os.path.exists(path + ".zip")

            agent2 = PPOAgent(config=_make_tiny_config())
            agent2.load(path)
            assert agent2.model is not None

    def test_select_action(self, vec_env) -> None:
        """Agent should return a masked-valid action from a live env."""
        agent = PPOAgent(config=_make_tiny_config())
        agent.build_model(vec_env)
        agent.model.learn(total_timesteps=64, progress_bar=False)

        env = DynamicFleetEnv(_make_tiny_config())
        env.reset(seed=42)
        action = agent.select_action(env)
        mask = env.get_action_mask()
        assert mask[action]

    def test_get_action_probabilities(self, vec_env) -> None:
        """Action probabilities should sum to ~1 over valid actions."""
        agent = PPOAgent(config=_make_tiny_config())
        agent.build_model(vec_env)
        agent.model.learn(total_timesteps=64, progress_bar=False)

        env = DynamicFleetEnv(_make_tiny_config())
        env.reset(seed=42)
        obs = env._get_observation()
        mask = env.get_action_mask()
        probs = agent.get_action_probabilities(obs, mask)

        assert probs.shape[0] == env.action_space.n
        assert abs(probs.sum() - 1.0) < 1e-4


# =========================================================================
# Evaluation pipeline smoke test
# =========================================================================

class TestEvaluationPipeline:
    """Smoke test for evaluate_dispatcher."""

    def test_evaluate_with_greedy(self) -> None:
        """evaluate_dispatcher should collect metrics for a greedy run."""
        from src.baselines.greedy_dispatch import GreedyDispatcher

        env = DynamicFleetEnv(_make_tiny_config())
        dispatcher = GreedyDispatcher()
        collector = evaluate_dispatcher(env, dispatcher, n_episodes=1, seed=42)

        assert collector.num_episodes == 1
        summary = collector.get_summary()
        assert "completion_rate_mean" in summary
        assert summary["total_requests_mean"] > 0


# =========================================================================
# Seed reproducibility
# =========================================================================

class TestSeedReproducibility:
    """Verify that global seed produces deterministic results."""

    def test_deterministic_env(self) -> None:
        """Same seed should yield identical observations."""
        set_global_seed(42)
        env1 = DynamicFleetEnv(_make_tiny_config())
        obs1, _ = env1.reset(seed=42)

        set_global_seed(42)
        env2 = DynamicFleetEnv(_make_tiny_config())
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)
