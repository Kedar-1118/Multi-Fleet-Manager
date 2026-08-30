"""Monte Carlo Tree Search planner for fleet routing.

Implements a lightweight MCTS algorithm that explores dispatch
assignments using environment state cloning, UCB1 selection,
and bounded rollouts.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.planning.search_node import SearchNode


class MCTSPlanner:
    """Monte Carlo Tree Search planner for dispatch decisions.

    Uses environment state cloning to explore candidate actions
    without mutating the production environment. The planner
    performs selection, expansion, simulation (rollout), and
    backpropagation to find the best action.

    Args:
        num_simulations: Number of MCTS iterations.
        max_depth: Maximum tree depth.
        exploration_constant: UCB1 exploration parameter.
        rollout_horizon: Steps to simulate during rollout.
        discount_factor: Discount for future rewards.
    """

    def __init__(
        self,
        num_simulations: int = 50,
        max_depth: int = 10,
        exploration_constant: float = 1.414,
        rollout_horizon: int = 5,
        discount_factor: float = 0.99,
        prior_weight: float = 0.5,
    ) -> None:
        """Initialize the MCTS planner.

        Args:
            num_simulations: Number of MCTS simulation iterations.
            max_depth: Maximum tree search depth.
            exploration_constant: UCB1 exploration constant.
            rollout_horizon: Number of steps in rollout simulation.
            discount_factor: Discount factor for future rewards.
            prior_weight: Weight of policy prior in UCB score.
        """
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.rollout_horizon = rollout_horizon
        self.discount_factor = discount_factor
        self.prior_weight = prior_weight
        self.name = "mcts"
        self._last_search_time_ms: float = 0.0

    @property
    def last_search_time_ms(self) -> float:
        """Return the last search time in milliseconds."""
        return self._last_search_time_ms

    def search(
        self,
        env: DynamicFleetEnv,
        action_priors: Optional[dict[int, float]] = None,
    ) -> int:
        """Run MCTS from the current environment state.

        Args:
            env: The fleet environment (state will be cloned, not mutated).
            action_priors: Optional prior probabilities from a policy network.

        Returns:
            Best action determined by MCTS.
        """
        start_time = time.perf_counter()

        # Save environment state
        root_state = env.clone_state()

        # Create root node
        root = SearchNode()
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0].tolist()

        if not valid_actions:
            self._last_search_time_ms = (time.perf_counter() - start_time) * 1000
            return env.action_space.n - 1  # NO-OP

        if len(valid_actions) == 1:
            self._last_search_time_ms = (time.perf_counter() - start_time) * 1000
            return valid_actions[0]

        # Expand root with priors
        root.expand(valid_actions, action_priors)

        # Run simulations
        for _ in range(self.num_simulations):
            # Restore to root state
            env.restore_state(root_state)

            # Selection: traverse tree using UCB1
            node = root
            depth = 0

            while node.is_expanded and not node.is_leaf and depth < self.max_depth:
                node = node.select_child(
                    self.exploration_constant, self.prior_weight
                )
                depth += 1

                # Take the action in the cloned environment
                if node.action is not None:
                    obs, reward, terminated, truncated, info = env.step(node.action)
                    if terminated or truncated:
                        break

            # Expansion: if node is not expanded, expand it
            if not node.is_expanded and depth < self.max_depth:
                current_mask = env.get_action_mask()
                current_valid = np.where(current_mask)[0].tolist()
                if current_valid:
                    node.expand(current_valid)

            # Simulation (rollout): random rollout from current state
            rollout_value = self._rollout(env)

            # Backpropagation
            node.backpropagate(rollout_value)

        # Restore original state
        env.restore_state(root_state)

        self._last_search_time_ms = (time.perf_counter() - start_time) * 1000

        # Return action with most visits
        return root.best_action()

    def _rollout(self, env: DynamicFleetEnv) -> float:
        """Perform a random rollout from the current state.

        Takes random valid actions for `rollout_horizon` steps
        and accumulates discounted rewards.

        Args:
            env: The environment (in a cloned state).

        Returns:
            Total discounted reward from the rollout.
        """
        total_reward = 0.0
        discount = 1.0

        for step in range(self.rollout_horizon):
            mask = env.get_action_mask()
            valid_actions = np.where(mask)[0]

            if len(valid_actions) == 0:
                break

            # Random action selection during rollout
            action = int(np.random.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += discount * reward
            discount *= self.discount_factor

            if terminated or truncated:
                break

        return total_reward

    def select_action(self, env: DynamicFleetEnv) -> int:
        """Select an action (convenience wrapper for search).

        Args:
            env: The fleet environment.

        Returns:
            Best action from MCTS search.
        """
        return self.search(env)
