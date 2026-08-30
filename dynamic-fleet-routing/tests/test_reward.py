"""Tests for the reward function."""

import pytest

from src.environment.reward import RewardCalculator, RewardConfig, RewardBreakdown


class TestRewardCalculator:
    """Tests for the RewardCalculator class."""

    def test_positive_delivery_reward(self) -> None:
        """Successful delivery should generate positive reward component."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(deliveries_completed=1)
        assert breakdown.delivery_reward > 0

    def test_travel_penalty_negative(self) -> None:
        """Travel distance should incur a penalty."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(distance_traveled=10.0)
        assert breakdown.travel_penalty < 0

    def test_fuel_penalty_negative(self) -> None:
        """Fuel consumption should incur a penalty."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(fuel_consumed=5.0)
        assert breakdown.fuel_penalty < 0

    def test_sla_violation_penalty(self) -> None:
        """SLA violations should incur a large penalty."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(sla_violations=1)
        assert breakdown.sla_penalty < 0
        assert abs(breakdown.sla_penalty) >= 20.0

    def test_idle_penalty(self) -> None:
        """Idle time should incur a penalty."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(idle_time=10.0)
        assert breakdown.idle_penalty < 0

    def test_utilization_reward(self) -> None:
        """Fleet utilization should generate a reward."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(fleet_utilization=0.8)
        assert breakdown.utilization_reward > 0

    def test_expiry_penalty(self) -> None:
        """Expired requests should incur a penalty."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(requests_expired=2)
        assert breakdown.expiry_penalty < 0

    def test_total_reward_is_sum(self) -> None:
        """Total reward should be sum of components (without normalization)."""
        calc = RewardCalculator(RewardConfig(normalize=False))
        breakdown = calc.calculate(
            deliveries_completed=1,
            distance_traveled=5.0,
            fuel_consumed=1.0,
            sla_violations=0,
            idle_time=2.0,
            fleet_utilization=0.5,
        )
        expected = (
            breakdown.delivery_reward
            + breakdown.travel_penalty
            + breakdown.fuel_penalty
            + breakdown.sla_penalty
            + breakdown.idle_penalty
            + breakdown.utilization_reward
            + breakdown.expiry_penalty
        )
        assert breakdown.total_reward == pytest.approx(expected, abs=1e-6)

    def test_configurable_weights(self) -> None:
        """Custom weights should affect reward components."""
        config = RewardConfig(delivery_reward=100.0, normalize=False)
        calc = RewardCalculator(config)
        breakdown = calc.calculate(deliveries_completed=1)
        assert breakdown.delivery_reward == 100.0

    def test_reward_normalization(self) -> None:
        """Normalized rewards should stabilize over time."""
        calc = RewardCalculator(RewardConfig(normalize=True, normalization_window=10))
        rewards = []
        for _ in range(20):
            breakdown = calc.calculate(deliveries_completed=1, distance_traveled=2.0)
            rewards.append(breakdown.total_reward)
        # After warmup, normalized rewards should be bounded
        assert all(-10.0 <= r <= 10.0 for r in rewards)

    def test_breakdown_to_dict(self) -> None:
        """Breakdown should convert to dict correctly."""
        breakdown = RewardBreakdown(delivery_reward=5.0, total_reward=3.0)
        d = breakdown.to_dict()
        assert d["delivery_reward"] == 5.0
        assert d["total_reward"] == 3.0
        assert "travel_penalty" in d

    def test_reset(self) -> None:
        """Reset should clear running statistics."""
        calc = RewardCalculator(RewardConfig(normalize=True))
        calc.calculate(deliveries_completed=1)
        calc.calculate(deliveries_completed=1)
        calc.reset()
        # After reset, first calculation should return unnormalized value
        breakdown = calc.calculate(deliveries_completed=1)
        # With single data point, normalization returns raw value
        assert breakdown.total_reward != 0.0
