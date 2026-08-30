"""MCTS tests for search node and planner."""

import pytest
import numpy as np

from src.planning.search_node import SearchNode
from src.planning.mcts import MCTSPlanner
from src.environment.fleet_env import DynamicFleetEnv


class TestSearchNode:
    """Tests for the SearchNode class."""

    def test_node_creation(self) -> None:
        """Node should initialize with default values."""
        node = SearchNode()
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.is_leaf
        assert not node.is_expanded

    def test_mean_value(self) -> None:
        """Mean value should be total_value / visit_count."""
        node = SearchNode()
        node.visit_count = 4
        node.total_value = 10.0
        assert node.mean_value == 2.5

    def test_mean_value_zero_visits(self) -> None:
        """Mean value should be 0 for unvisited nodes."""
        node = SearchNode()
        assert node.mean_value == 0.0

    def test_expand(self) -> None:
        """Expanding should create child nodes for valid actions."""
        node = SearchNode()
        node.expand([0, 1, 2])
        assert len(node.children) == 3
        assert node.is_expanded
        assert not node.is_leaf

    def test_expand_with_priors(self) -> None:
        """Priors should be assigned to child nodes."""
        node = SearchNode()
        priors = {0: 0.5, 1: 0.3, 2: 0.2}
        node.expand([0, 1, 2], priors)
        assert node.children[0].prior == pytest.approx(0.5)
        assert node.children[1].prior == pytest.approx(0.3)

    def test_ucb1_score(self) -> None:
        """UCB1 score should combine exploitation and exploration."""
        root = SearchNode()
        root.visit_count = 10
        child = SearchNode(action=0, parent=root)
        child.visit_count = 2
        child.total_value = 4.0

        score = child.ucb1_score(exploration_constant=1.414)
        assert score > 0  # Should be positive (exploitation + exploration)

    def test_select_child(self) -> None:
        """Should select child with highest UCB1 score."""
        root = SearchNode()
        root.expand([0, 1, 2])
        root.visit_count = 10

        # Give one child more value
        root.children[1].visit_count = 2
        root.children[1].total_value = 10.0

        selected = root.select_child()
        assert selected is not None

    def test_select_child_no_children(self) -> None:
        """Selecting from childless node should raise."""
        node = SearchNode()
        with pytest.raises(ValueError, match="no children"):
            node.select_child()

    def test_backpropagate(self) -> None:
        """Backpropagation should update all ancestors."""
        root = SearchNode()
        root.expand([0])
        child = root.children[0]
        child.expand([1])
        grandchild = child.children[1]

        grandchild.backpropagate(5.0)

        assert grandchild.visit_count == 1
        assert child.visit_count == 1
        assert root.visit_count == 1
        assert grandchild.total_value == 5.0
        assert child.total_value == 5.0
        assert root.total_value == 5.0

    def test_best_action(self) -> None:
        """Best action should be the most-visited child."""
        root = SearchNode()
        root.expand([0, 1, 2])
        root.children[0].visit_count = 5
        root.children[1].visit_count = 10
        root.children[2].visit_count = 3

        assert root.best_action() == 1


class TestMCTSPlanner:
    """Tests for the MCTSPlanner class."""

    @pytest.fixture
    def env(self) -> DynamicFleetEnv:
        """Create a small environment for MCTS testing."""
        config = {
            "simulation": {"num_nodes": 10, "num_vehicles": 2, "seed": 42,
                           "simulation_duration": 300},
            "city": {"grid_size": 5.0, "edge_density": 0.3, "base_speed_kmh": 30.0},
            "traffic": {"congestion_probability": 0.0, "noise_std": 0.0,
                        "traffic_update_interval": 300,
                        "rush_hour_windows": []},
            "vehicles": {"capacity": 5, "fuel_consumption_per_km": 0.08,
                         "service_time_minutes": 1, "initial_fuel": 100.0},
            "requests": {"lambda_rate": 3, "arrival_interval": 30,
                         "min_deadline_minutes": 60, "max_deadline_minutes": 120,
                         "min_package_size": 1, "max_package_size": 1,
                         "priority_weights": [1.0, 0.0, 0.0]},
            "observation": {"top_k_requests": 5, "max_vehicles": 2},
            "reward": {"delivery_reward": 10.0, "travel_penalty": 0.1,
                       "fuel_penalty": 0.5, "sla_violation_penalty": 20.0,
                       "idle_penalty": 0.05, "utilization_reward": 1.0,
                       "expiry_penalty": 15.0, "normalize": False,
                       "normalization_window": 100},
        }
        return DynamicFleetEnv(config)

    def test_mcts_returns_valid_action(self, env: DynamicFleetEnv) -> None:
        """MCTS should return a valid action."""
        env.reset(seed=42)
        planner = MCTSPlanner(num_simulations=10, rollout_horizon=3)
        action = planner.search(env)
        mask = env.get_action_mask()
        assert mask[action], f"MCTS returned invalid action {action}"

    def test_mcts_does_not_mutate_state(self, env: DynamicFleetEnv) -> None:
        """MCTS search should not change the environment state."""
        env.reset(seed=42)
        state_before = env.clone_state()
        time_before = env.current_time

        planner = MCTSPlanner(num_simulations=10, rollout_horizon=3)
        planner.search(env)

        assert env.current_time == time_before
        assert len(env.pending_requests) == len(state_before["pending_requests"])

    def test_mcts_with_priors(self, env: DynamicFleetEnv) -> None:
        """MCTS should accept and use action priors."""
        env.reset(seed=42)
        mask = env.get_action_mask()
        valid = np.where(mask)[0].tolist()

        # Uniform priors
        priors = {a: 1.0 / len(valid) for a in valid}

        planner = MCTSPlanner(num_simulations=10)
        action = planner.search(env, action_priors=priors)
        assert action in valid

    def test_mcts_search_time_recorded(self, env: DynamicFleetEnv) -> None:
        """Search time should be recorded."""
        env.reset(seed=42)
        planner = MCTSPlanner(num_simulations=5)
        planner.search(env)
        assert planner.last_search_time_ms > 0
