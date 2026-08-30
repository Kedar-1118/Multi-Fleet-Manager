"""MCTS search tree node.

Defines the SearchNode class used in Monte Carlo Tree Search,
with UCB1 selection, visit tracking, and value backpropagation.
"""

from __future__ import annotations

import math
from typing import Optional


class SearchNode:
    """A node in the MCTS search tree.

    Each node represents a state-action pair, tracking visit counts,
    accumulated value, and a prior probability from the policy network.

    Attributes:
        action: The action that led to this node.
        parent: Parent node in the tree.
        prior: Prior probability from the policy network.
        visit_count: Number of times this node has been visited.
        total_value: Accumulated value from backpropagation.
        children: Child nodes keyed by action.
    """

    def __init__(
        self,
        action: Optional[int] = None,
        parent: Optional[SearchNode] = None,
        prior: float = 0.0,
    ) -> None:
        """Initialize a search node.

        Args:
            action: The action that led to this node.
            parent: Parent node.
            prior: Prior probability from policy network.
        """
        self.action = action
        self.parent = parent
        self.prior = prior
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.children: dict[int, SearchNode] = {}
        self._is_expanded: bool = False

    @property
    def mean_value(self) -> float:
        """Return the mean value (Q-value) of this node.

        Returns:
            Average accumulated value, or 0 if unvisited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """Check if this node has no children."""
        return len(self.children) == 0

    @property
    def is_expanded(self) -> bool:
        """Check if this node has been expanded."""
        return self._is_expanded

    def ucb1_score(
        self,
        exploration_constant: float = 1.414,
        prior_weight: float = 0.5,
    ) -> float:
        """Calculate the UCB1 score for node selection.

        Combines exploitation (mean value), exploration (visit count),
        and policy prior into a single selection score.

        UCB1 = Q(s,a) + c * P(a|s) * sqrt(N_parent) / (1 + N_child)

        Args:
            exploration_constant: Controls exploration vs exploitation.
            prior_weight: Weight given to the policy prior.

        Returns:
            UCB1 score for this node.
        """
        if self.parent is None:
            return 0.0

        parent_visits = max(self.parent.visit_count, 1)

        exploitation = self.mean_value

        exploration = (
            exploration_constant
            * (prior_weight * self.prior + (1 - prior_weight))
            * math.sqrt(parent_visits)
            / (1 + self.visit_count)
        )

        return exploitation + exploration

    def select_child(
        self,
        exploration_constant: float = 1.414,
        prior_weight: float = 0.5,
    ) -> SearchNode:
        """Select the child with highest UCB1 score.

        Args:
            exploration_constant: UCB1 exploration constant.
            prior_weight: Weight for policy prior.

        Returns:
            The child node with highest UCB1 score.

        Raises:
            ValueError: If node has no children.
        """
        if not self.children:
            raise ValueError("Cannot select child: node has no children")

        best_score = float("-inf")
        best_child = None

        for child in self.children.values():
            score = child.ucb1_score(exploration_constant, prior_weight)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(
        self,
        valid_actions: list[int],
        priors: Optional[dict[int, float]] = None,
    ) -> None:
        """Expand this node by creating child nodes for valid actions.

        Args:
            valid_actions: List of valid action indices.
            priors: Optional mapping from action to prior probability.
        """
        if priors is None:
            # Uniform prior
            uniform_prior = 1.0 / max(len(valid_actions), 1)
            priors = {a: uniform_prior for a in valid_actions}

        for action in valid_actions:
            if action not in self.children:
                prior = priors.get(action, 0.0)
                self.children[action] = SearchNode(
                    action=action, parent=self, prior=prior
                )

        self._is_expanded = True

    def backpropagate(self, value: float) -> None:
        """Backpropagate a value up the tree.

        Updates visit counts and accumulated values from this node
        up to the root.

        Args:
            value: The value to backpropagate.
        """
        node: Optional[SearchNode] = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    def best_action(self) -> int:
        """Return the action of the most-visited child.

        Returns:
            Action index of the most-visited child.

        Raises:
            ValueError: If node has no children.
        """
        if not self.children:
            raise ValueError("No children to select from")

        best_visits = -1
        best_action = -1

        for action, child in self.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action

        return best_action

    def best_child_by_value(self) -> SearchNode:
        """Return the child with highest mean value.

        Returns:
            Child node with highest Q-value.

        Raises:
            ValueError: If node has no children.
        """
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children.values(), key=lambda c: c.mean_value)

    def __repr__(self) -> str:
        return (
            f"SearchNode(action={self.action}, visits={self.visit_count}, "
            f"value={self.mean_value:.3f}, children={len(self.children)})"
        )
