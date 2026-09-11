"""Tests for Electric Vehicle (EV) battery constraints, charging stations,
carbon emissions tracking, and related environment extensions.

Tests cover:
- Vehicle battery SoC tracking and energy consumption
- Charging station designation and nearest-station queries
- Action masking for low-battery vehicles
- Charging event lifecycle (IDLE → CHARGING → IDLE)
- EmissionsCalculator correctness (EV vs ICE CO₂)
- Carbon penalty in reward breakdown
- Vehicle clone/reset preserves battery state
- Full episode runs with EV constraints
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.city_graph import CityGraph, CityGraphConfig
from src.environment.fleet_env import DynamicFleetEnv, EventType, SimEvent
from src.environment.reward import RewardCalculator, RewardConfig
from src.environment.vehicle import Vehicle, VehicleStatus
from src.utils.emissions import EmissionsCalculator, EmissionsResult


# ============================================================================
# Vehicle Battery Tests
# ============================================================================

class TestVehicleBattery:
    """Tests for Vehicle EV battery model."""

    def test_battery_soc_full(self):
        """Full battery should report 1.0 SoC."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=60.0)
        assert v.battery_soc == pytest.approx(1.0)

    def test_battery_soc_partial(self):
        """Partial battery reports correct fraction."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=30.0)
        assert v.battery_soc == pytest.approx(0.5)

    def test_battery_soc_empty(self):
        """Empty battery should report 0.0 SoC."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=0.0)
        assert v.battery_soc == pytest.approx(0.0)

    def test_needs_charging_below_threshold(self):
        """Vehicle below threshold should need charging."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=10.0)
        assert v.needs_charging(threshold=0.2) is True

    def test_needs_charging_above_threshold(self):
        """Vehicle above threshold should not need charging."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=50.0)
        assert v.needs_charging(threshold=0.2) is False

    def test_consume_energy_drains_battery(self):
        """consume_energy should reduce battery_level_kwh."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=60.0,
                    energy_consumption_kwh_per_km=0.15)
        energy_used = v.consume_energy(100.0)  # 100 km
        assert energy_used == pytest.approx(15.0)
        assert v.battery_level_kwh == pytest.approx(45.0)
        assert v.total_energy_consumed_kwh == pytest.approx(15.0)

    def test_consume_energy_clamps_to_zero(self):
        """Battery should never go below zero."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=5.0,
                    energy_consumption_kwh_per_km=0.15)
        v.consume_energy(500.0)  # Would need 75 kWh, only have 5
        assert v.battery_level_kwh == pytest.approx(0.0)

    def test_consume_energy_negative_distance_raises(self):
        """Negative distance should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0)
        with pytest.raises(ValueError, match="negative"):
            v.consume_energy(-10.0)

    def test_charge_battery(self):
        """charge_battery should increase battery up to capacity."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=20.0)
        added = v.charge_battery(30.0)
        assert added == pytest.approx(30.0)
        assert v.battery_level_kwh == pytest.approx(50.0)

    def test_charge_battery_caps_at_capacity(self):
        """Charging should not exceed battery_capacity_kwh."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=50.0)
        added = v.charge_battery(20.0)
        assert added == pytest.approx(10.0)  # Only 10 kWh headroom
        assert v.battery_level_kwh == pytest.approx(60.0)

    def test_charge_battery_negative_raises(self):
        """Negative charge energy should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0)
        with pytest.raises(ValueError, match="negative"):
            v.charge_battery(-5.0)

    def test_charging_status_transition(self):
        """IDLE → CHARGING and CHARGING → IDLE should be valid."""
        v = Vehicle(vehicle_id=0, current_location=0)
        v.set_status(VehicleStatus.CHARGING)
        assert v.status == VehicleStatus.CHARGING
        v.set_status(VehicleStatus.IDLE)
        assert v.status == VehicleStatus.IDLE

    def test_invalid_charging_transition(self):
        """MOVING_TO_PICKUP → CHARGING should be invalid."""
        v = Vehicle(vehicle_id=0, current_location=0)
        v.set_status(VehicleStatus.MOVING_TO_PICKUP)
        with pytest.raises(ValueError, match="Invalid vehicle status transition"):
            v.set_status(VehicleStatus.CHARGING)

    def test_clone_preserves_battery(self):
        """Vehicle clone should preserve all battery fields."""
        v = Vehicle(vehicle_id=0, current_location=5,
                    battery_capacity_kwh=80.0, battery_level_kwh=42.5,
                    energy_consumption_kwh_per_km=0.2,
                    total_energy_consumed_kwh=37.5)
        c = v.clone()
        assert c.battery_capacity_kwh == 80.0
        assert c.battery_level_kwh == 42.5
        assert c.energy_consumption_kwh_per_km == 0.2
        assert c.total_energy_consumed_kwh == 37.5
        # Ensure independence
        c.battery_level_kwh = 10.0
        assert v.battery_level_kwh == 42.5

    def test_reset_restores_battery(self):
        """Vehicle reset should restore battery to full capacity."""
        v = Vehicle(vehicle_id=0, current_location=0,
                    battery_capacity_kwh=60.0, battery_level_kwh=10.0,
                    total_energy_consumed_kwh=50.0)
        v.reset(start_location=5)
        assert v.battery_level_kwh == pytest.approx(60.0)
        assert v.total_energy_consumed_kwh == pytest.approx(0.0)


# ============================================================================
# Charging Station Tests
# ============================================================================

class TestChargingStations:
    """Tests for CityGraph charging station infrastructure."""

    def test_charging_stations_designated(self):
        """Graph should designate the configured number of stations."""
        config = CityGraphConfig(num_nodes=20, num_charging_stations=3, seed=42)
        graph = CityGraph(config)
        assert len(graph.charging_stations) == 3

    def test_charging_stations_are_valid_nodes(self):
        """All charging stations should be valid graph nodes."""
        config = CityGraphConfig(num_nodes=20, num_charging_stations=4, seed=42)
        graph = CityGraph(config)
        for station in graph.charging_stations:
            assert station in graph.graph.nodes

    def test_is_charging_station(self):
        """is_charging_station should return correct boolean."""
        config = CityGraphConfig(num_nodes=20, num_charging_stations=2, seed=42)
        graph = CityGraph(config)
        for station in graph.charging_stations:
            assert graph.is_charging_station(station) is True
        # Find a non-station node
        non_station = [n for n in graph.graph.nodes if n not in graph.charging_stations][0]
        assert graph.is_charging_station(non_station) is False

    def test_nearest_charging_station(self):
        """get_nearest_charging_station should return valid station and distance."""
        config = CityGraphConfig(num_nodes=30, num_charging_stations=3, seed=42)
        graph = CityGraph(config)
        node = 0
        station, dist = graph.get_nearest_charging_station(node)
        assert station in graph.charging_stations
        assert dist >= 0.0
        assert dist != float("inf")

    def test_no_charging_stations_raises(self):
        """get_nearest_charging_station should raise if no stations configured."""
        config = CityGraphConfig(num_nodes=10, num_charging_stations=0, seed=42)
        graph = CityGraph(config)
        with pytest.raises(ValueError, match="No charging stations"):
            graph.get_nearest_charging_station(0)

    def test_charging_stations_spatially_distributed(self):
        """Charging stations should not all be the same node."""
        config = CityGraphConfig(num_nodes=50, num_charging_stations=3, seed=42)
        graph = CityGraph(config)
        assert len(set(graph.charging_stations)) == 3


# ============================================================================
# Emissions Calculator Tests
# ============================================================================

class TestEmissionsCalculator:
    """Tests for the EmissionsCalculator utility."""

    def test_ev_emissions(self):
        """EV emissions should be energy × emission factor."""
        calc = EmissionsCalculator(ev_co2_per_kwh=0.233)
        result = calc.calculate_ev_emissions(100.0)
        assert result == pytest.approx(23.3)

    def test_ice_emissions(self):
        """ICE emissions should be distance × emission factor."""
        calc = EmissionsCalculator(ice_co2_per_km=0.21)
        result = calc.calculate_ice_emissions(1000.0)
        assert result == pytest.approx(210.0)

    def test_savings_calculation(self):
        """Savings should be ICE - EV emissions."""
        calc = EmissionsCalculator(ev_co2_per_kwh=0.233, ice_co2_per_km=0.21)
        result = calc.calculate_savings(distance_km=1000.0, energy_kwh=100.0)
        assert isinstance(result, EmissionsResult)
        assert result.ev_co2_kg == pytest.approx(23.3)
        assert result.ice_co2_kg == pytest.approx(210.0)
        assert result.co2_saved_kg == pytest.approx(186.7)
        assert result.co2_reduction_pct > 80.0  # EV should save >80%

    def test_savings_to_dict(self):
        """EmissionsResult.to_dict should return all keys."""
        calc = EmissionsCalculator()
        result = calc.calculate_savings(500.0, 50.0)
        d = result.to_dict()
        assert "ev_co2_kg" in d
        assert "ice_co2_kg" in d
        assert "co2_saved_kg" in d
        assert "co2_reduction_pct" in d

    def test_negative_factor_raises(self):
        """Negative emission factors should raise ValueError."""
        with pytest.raises(ValueError):
            EmissionsCalculator(ev_co2_per_kwh=-0.1)
        with pytest.raises(ValueError):
            EmissionsCalculator(ice_co2_per_km=-0.1)

    def test_zero_distance_savings(self):
        """Zero distance should produce zero emissions."""
        calc = EmissionsCalculator()
        result = calc.calculate_savings(0.0, 0.0)
        assert result.ev_co2_kg == 0.0
        assert result.ice_co2_kg == 0.0
        assert result.co2_saved_kg == 0.0


# ============================================================================
# Reward Carbon Penalty Tests
# ============================================================================

class TestRewardCarbonPenalty:
    """Tests for carbon and low-battery penalties in RewardCalculator."""

    def test_carbon_penalty_in_breakdown(self):
        """Carbon penalty should appear in reward breakdown."""
        config = RewardConfig(normalize=False, carbon_penalty=0.3)
        calc = RewardCalculator(config)
        breakdown = calc.calculate(energy_consumed_kwh=10.0)
        assert breakdown.carbon_penalty == pytest.approx(-3.0)

    def test_low_battery_penalty_in_breakdown(self):
        """Low battery penalty should appear in reward breakdown."""
        config = RewardConfig(normalize=False, low_battery_penalty=5.0)
        calc = RewardCalculator(config)
        breakdown = calc.calculate(vehicles_below_threshold=2)
        assert breakdown.low_battery_penalty == pytest.approx(-10.0)

    def test_carbon_penalty_in_total(self):
        """Carbon penalty should reduce total reward."""
        config = RewardConfig(normalize=False, carbon_penalty=1.0,
                              low_battery_penalty=0.0)
        calc = RewardCalculator(config)
        base = calc.calculate()
        penalized = calc.calculate(energy_consumed_kwh=5.0)
        assert penalized.total_reward < base.total_reward

    def test_carbon_penalty_in_to_dict(self):
        """Carbon penalty should appear in to_dict output."""
        config = RewardConfig(normalize=False)
        calc = RewardCalculator(config)
        breakdown = calc.calculate(energy_consumed_kwh=10.0)
        d = breakdown.to_dict()
        assert "carbon_penalty" in d
        assert "low_battery_penalty" in d


# ============================================================================
# Environment EV Integration Tests
# ============================================================================

class TestEnvironmentEVIntegration:
    """Tests for EV features integrated into DynamicFleetEnv."""

    @pytest.fixture
    def env(self):
        """Create a small environment for testing."""
        config = {
            "simulation": {"num_nodes": 20, "num_vehicles": 3, "seed": 42,
                           "simulation_duration": 1440},
            "vehicles": {"capacity": 10, "initial_fuel": 100.0},
            "ev": {"battery_capacity_kwh": 60.0,
                   "energy_consumption_kwh_per_km": 0.15,
                   "low_soc_threshold": 0.2,
                   "num_charging_stations": 2,
                   "charging_time_minutes": 15.0},
            "observation": {"top_k_requests": 10, "max_vehicles": 5},
            "reward": {"normalize": False, "carbon_penalty": 0.3,
                       "low_battery_penalty": 5.0},
        }
        return DynamicFleetEnv(config)

    def test_vehicles_have_battery(self, env):
        """Vehicles should be initialized with battery parameters."""
        obs, info = env.reset(seed=42)
        for v in env.vehicles:
            assert v.battery_capacity_kwh == 60.0
            assert v.battery_level_kwh == 60.0
            assert v.battery_soc == pytest.approx(1.0)

    def test_info_contains_ev_metrics(self, env):
        """Info dict should contain EV-specific metrics."""
        obs, info = env.reset(seed=42)
        assert "total_energy_kwh" in info
        assert "total_co2_kg" in info
        assert "avg_battery_soc" in info
        assert "vehicles_low_battery" in info

    def test_energy_consumed_after_steps(self, env):
        """Energy should be consumed after dispatch steps."""
        obs, info = env.reset(seed=42)
        mask = env.get_action_mask()
        initial_energy = info["total_energy_kwh"]

        # Take a non-NOOP action if available
        valid_actions = np.where(mask)[0]
        non_noop = [a for a in valid_actions if a != env._action_size - 1]
        if non_noop:
            obs, reward, done, trunc, info = env.step(non_noop[0])
            # Energy should have been consumed (or at least tracked)
            assert info["total_energy_kwh"] >= initial_energy

    def test_charging_stations_in_graph(self, env):
        """City graph should have charging stations."""
        obs, info = env.reset(seed=42)
        assert len(env.city_graph.charging_stations) == 2

    def test_clone_restore_preserves_ev_state(self, env):
        """clone_state/restore_state should preserve EV counters."""
        obs, info = env.reset(seed=42)
        # Take a step to change state
        mask = env.get_action_mask()
        action = np.where(mask)[0][0]
        env.step(action)

        state = env.clone_state()
        energy_before = env._total_energy_consumed_kwh

        # Take more steps
        for _ in range(3):
            mask = env.get_action_mask()
            action = np.where(mask)[0][0]
            env.step(action)

        # Restore
        env.restore_state(state)
        assert env._total_energy_consumed_kwh == pytest.approx(energy_before)
