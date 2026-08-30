"""Hybrid PPO + MCTS planner.

Combines PPO policy priors with MCTS search refinement.
Uses PPO action probabilities to guide MCTS exploration,
with a latency budget fallback to pure PPO.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from src.environment.fleet_env import DynamicFleetEnv
from src.agents.ppo_agent import PPOAgent
from src.planning.mcts import MCTSPlanner


class HybridPlanner:
    """Hybrid PPO + MCTS dispatch planner.

    Workflow:
    1. Get action probabilities from trained PPO policy.
    2. Select top-K candidate actions by probability.
    3. Use PPO probabilities as MCTS priors.
    4. Run bounded MCTS over candidates.
    5. Return action with best visit/value score.
    6. Fallback to PPO action if MCTS exceeds latency budget.

    Args:
        ppo_agent: Trained PPO agent.
        num_simulations: MCTS simulation budget.
        max_depth: MCTS max tree depth.
        top_k_actions: Number of top PPO actions to consider.
        latency_budget_ms: Maximum allowed decision latency.
        exploration_constant: UCB1 exploration constant.
        rollout_horizon: MCTS rollout horizon.
        fallback_to_ppo: Whether to fall back to PPO on timeout.
    """

    def __init__(
        self,
        ppo_agent: PPOAgent,
        num_simulations: int = 50,
        max_depth: int = 10,
        top_k_actions: int = 10,
        latency_budget_ms: float = 45.0,
        exploration_constant: float = 1.414,
        rollout_horizon: int = 5,
        fallback_to_ppo: bool = True,
        prior_weight: float = 0.5,
    ) -> None:
        """Initialize the hybrid planner.

        Args:
            ppo_agent: Trained PPO agent for prior computation.
            num_simulations: Number of MCTS simulations.
            max_depth: Maximum MCTS tree depth.
            top_k_actions: Number of top candidates from PPO.
            latency_budget_ms: Maximum latency before fallback.
            exploration_constant: UCB1 exploration parameter.
            rollout_horizon: Steps per rollout.
            fallback_to_ppo: Enable PPO fallback on timeout.
            prior_weight: Weight of PPO prior in UCB score.
        """
        self.ppo_agent = ppo_agent
        self.top_k_actions = top_k_actions
        self.latency_budget_ms = latency_budget_ms
        self.fallback_to_ppo = fallback_to_ppo
        self.name = "ppo_mcts"

        self.mcts = MCTSPlanner(
            num_simulations=num_simulations,
            max_depth=max_depth,
            exploration_constant=exploration_constant,
            rollout_horizon=rollout_horizon,
            prior_weight=prior_weight,
        )

        self._last_latency_ms: float = 0.0
        self._last_decision_source: str = "hybrid"
        self._fallback_count: int = 0
        self._total_decisions: int = 0

    @property
    def last_latency_ms(self) -> float:
        """Return the latency of the last decision."""
        return self._last_latency_ms

    @property
    def last_decision_source(self) -> str:
        """Return the source of the last decision (hybrid or ppo_fallback)."""
        return self._last_decision_source

    @property
    def fallback_rate(self) -> float:
        """Return the fraction of decisions that fell back to PPO."""
        if self._total_decisions == 0:
            return 0.0
        return self._fallback_count / self._total_decisions

    def select_action(self, env: DynamicFleetEnv) -> int:
        """Select the best action using hybrid PPO + MCTS.

        Args:
            env: The fleet environment.

        Returns:
            Best action determined by hybrid planning.
        """
        start_time = time.perf_counter()
        self._total_decisions += 1

        # Step 1: Get PPO action probabilities
        obs = env._get_observation()
        mask = env.get_action_mask()
        probs = self.ppo_agent.get_action_probabilities(obs, mask)

        # Step 2: Select top-K candidates
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            self._last_latency_ms = (time.perf_counter() - start_time) * 1000
            self._last_decision_source = "ppo_fallback"
            return env.action_space.n - 1

        # Sort by probability and take top-K
        valid_probs = [(idx, probs[idx]) for idx in valid_indices]
        valid_probs.sort(key=lambda x: x[1], reverse=True)
        top_k = valid_probs[:self.top_k_actions]

        # If only one candidate, return it directly
        if len(top_k) == 1:
            self._last_latency_ms = (time.perf_counter() - start_time) * 1000
            self._last_decision_source = "ppo_direct"
            return top_k[0][0]

        # Step 3: Build prior dict for MCTS
        total_prob = sum(p for _, p in top_k)
        if total_prob > 0:
            action_priors = {
                action: prob / total_prob for action, prob in top_k
            }
        else:
            action_priors = {action: 1.0 / len(top_k) for action, _ in top_k}

        # Check remaining latency budget
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        remaining_budget = self.latency_budget_ms - elapsed_ms

        if remaining_budget < 5.0 and self.fallback_to_ppo:
            # Not enough budget for MCTS — fallback to PPO's top action
            self._last_latency_ms = (time.perf_counter() - start_time) * 1000
            self._last_decision_source = "ppo_fallback"
            self._fallback_count += 1
            return top_k[0][0]

        # Step 4: Run MCTS with PPO priors
        best_action = self.mcts.search(env, action_priors=action_priors)

        self._last_latency_ms = (time.perf_counter() - start_time) * 1000
        self._last_decision_source = "hybrid"

        # Post-hoc fallback check
        if self._last_latency_ms > self.latency_budget_ms and self.fallback_to_ppo:
            self._fallback_count += 1

        return best_action

    def get_stats(self) -> dict[str, Any]:
        """Get planner statistics.

        Returns:
            Dictionary with decision statistics.
        """
        return {
            "total_decisions": self._total_decisions,
            "fallback_count": self._fallback_count,
            "fallback_rate": round(self.fallback_rate, 4),
            "last_latency_ms": round(self._last_latency_ms, 4),
            "last_decision_source": self._last_decision_source,
        }
