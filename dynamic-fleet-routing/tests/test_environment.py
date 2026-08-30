"""Tests for the environment foundation: city graph, vehicle, and request models."""

import pytest
import numpy as np

from src.environment.city_graph import CityGraph, CityGraphConfig
from src.environment.vehicle import Vehicle, VehicleStatus
from src.environment.request import Request, RequestStatus, Priority
from src.environment.traffic_model import TrafficModel, TrafficConfig, TrafficState
from src.environment.request_generator import RequestGenerator, RequestGeneratorConfig


# ============================================================
# City Graph Tests
# ============================================================


class TestCityGraph:
    """Tests for the CityGraph class."""

    def test_graph_creation_default(self) -> None:
        """Graph should be created with default config."""
        graph = CityGraph()
        assert graph.num_nodes == 50
        assert graph.num_edges > 0

    def test_graph_creation_custom(self) -> None:
        """Graph should respect custom node count."""
        config = CityGraphConfig(num_nodes=20, seed=123)
        graph = CityGraph(config)
        assert graph.num_nodes == 20

    def test_graph_connectivity(self) -> None:
        """Graph must be connected."""
        import networkx as nx
        config = CityGraphConfig(num_nodes=30, edge_density=0.05, seed=99)
        graph = CityGraph(config)
        assert nx.is_connected(graph.graph)

    def test_node_positions(self) -> None:
        """All nodes should have valid positions."""
        config = CityGraphConfig(num_nodes=10, grid_size=10.0, seed=42)
        graph = CityGraph(config)
        for node_id in range(10):
            x, y = graph.get_node_position(node_id)
            assert 0 <= x <= 10.0
            assert 0 <= y <= 10.0

    def test_normalized_positions(self) -> None:
        """Normalized positions should be in [0, 1]."""
        config = CityGraphConfig(num_nodes=10, grid_size=10.0, seed=42)
        graph = CityGraph(config)
        for node_id in range(10):
            x, y = graph.get_normalized_position(node_id)
            assert 0 <= x <= 1.0
            assert 0 <= y <= 1.0

    def test_shortest_path_exists(self) -> None:
        """Shortest path should exist between any two nodes."""
        config = CityGraphConfig(num_nodes=15, seed=42)
        graph = CityGraph(config)
        path = graph.get_shortest_path(0, 14)
        assert len(path) >= 2
        assert path[0] == 0
        assert path[-1] == 14

    def test_shortest_distance_symmetric(self) -> None:
        """Distance should be the same in both directions."""
        config = CityGraphConfig(num_nodes=10, seed=42)
        graph = CityGraph(config)
        d1 = graph.get_shortest_distance(0, 5)
        d2 = graph.get_shortest_distance(5, 0)
        assert abs(d1 - d2) < 1e-6

    def test_travel_time_with_traffic(self) -> None:
        """Travel time should scale with traffic multiplier."""
        config = CityGraphConfig(num_nodes=10, seed=42)
        graph = CityGraph(config)
        t_normal = graph.get_travel_time(0, 5, traffic_multiplier=1.0)
        t_heavy = graph.get_travel_time(0, 5, traffic_multiplier=2.0)
        assert t_heavy == pytest.approx(t_normal * 2.0, rel=1e-6)

    def test_random_node_pair_distinct(self) -> None:
        """Random node pairs should be distinct."""
        config = CityGraphConfig(num_nodes=10, seed=42)
        graph = CityGraph(config)
        for _ in range(20):
            a, b = graph.get_random_node_pair()
            assert a != b

    def test_edge_weights_positive(self) -> None:
        """All edge weights should be positive."""
        config = CityGraphConfig(num_nodes=20, seed=42)
        graph = CityGraph(config)
        for u, v, data in graph.graph.edges(data=True):
            assert data["distance"] > 0
            assert data["base_travel_time"] > 0


# ============================================================
# Vehicle Tests
# ============================================================


class TestVehicle:
    """Tests for the Vehicle dataclass."""

    def test_vehicle_creation(self) -> None:
        """Vehicle should initialize with correct defaults."""
        v = Vehicle(vehicle_id=0, current_location=5)
        assert v.vehicle_id == 0
        assert v.current_location == 5
        assert v.capacity == 10
        assert v.current_load == 0
        assert v.status == VehicleStatus.IDLE
        assert v.is_available

    def test_capacity_remaining(self) -> None:
        """Capacity remaining should reflect current load."""
        v = Vehicle(vehicle_id=0, current_location=0, capacity=10)
        assert v.capacity_remaining == 10
        v.current_load = 3
        assert v.capacity_remaining == 7

    def test_load_package(self) -> None:
        """Loading a package should increase current load."""
        v = Vehicle(vehicle_id=0, current_location=0, capacity=5)
        v.load_package(2, request_id=1)
        assert v.current_load == 2
        assert 1 in v.assigned_requests

    def test_load_package_exceeds_capacity(self) -> None:
        """Loading beyond capacity should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0, capacity=3)
        v.load_package(2, request_id=1)
        with pytest.raises(ValueError, match="cannot load"):
            v.load_package(2, request_id=2)

    def test_load_package_invalid_size(self) -> None:
        """Loading zero or negative size should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0)
        with pytest.raises(ValueError, match="must be positive"):
            v.can_accept_package(0)

    def test_unload_package(self) -> None:
        """Unloading should decrease load and increment deliveries."""
        v = Vehicle(vehicle_id=0, current_location=0)
        v.load_package(2, request_id=1)
        v.unload_package(2, request_id=1)
        assert v.current_load == 0
        assert v.total_deliveries == 1
        assert 1 not in v.assigned_requests

    def test_unload_package_below_zero(self) -> None:
        """Unloading more than loaded should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0)
        with pytest.raises(ValueError, match="cannot unload"):
            v.unload_package(1, request_id=1)

    def test_fuel_consumption(self) -> None:
        """Fuel consumption should track correctly."""
        v = Vehicle(vehicle_id=0, current_location=0, fuel_remaining=100.0)
        v.consume_fuel(10.0, rate=0.08)
        assert v.fuel_consumed == pytest.approx(0.8)
        assert v.fuel_remaining == pytest.approx(99.2)
        assert v.total_distance == pytest.approx(10.0)

    def test_fuel_negative_distance(self) -> None:
        """Negative distance should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0)
        with pytest.raises(ValueError, match="cannot be negative"):
            v.consume_fuel(-5.0, rate=0.08)

    def test_status_transitions(self) -> None:
        """Valid status transitions should succeed."""
        v = Vehicle(vehicle_id=0, current_location=0)
        v.set_status(VehicleStatus.MOVING_TO_PICKUP)
        assert v.status == VehicleStatus.MOVING_TO_PICKUP
        v.set_status(VehicleStatus.SERVICING)
        assert v.status == VehicleStatus.SERVICING
        v.set_status(VehicleStatus.MOVING_TO_DROPOFF)
        assert v.status == VehicleStatus.MOVING_TO_DROPOFF
        v.set_status(VehicleStatus.IDLE)
        assert v.status == VehicleStatus.IDLE

    def test_invalid_status_transition(self) -> None:
        """Invalid status transitions should raise ValueError."""
        v = Vehicle(vehicle_id=0, current_location=0)
        with pytest.raises(ValueError, match="Invalid vehicle status"):
            v.set_status(VehicleStatus.MOVING_TO_DROPOFF)

    def test_utilization(self) -> None:
        """Utilization should be load / capacity."""
        v = Vehicle(vehicle_id=0, current_location=0, capacity=10)
        assert v.utilization == 0.0
        v.current_load = 5
        assert v.utilization == 0.5

    def test_clone(self) -> None:
        """Clone should create an independent copy."""
        v = Vehicle(vehicle_id=0, current_location=5, capacity=10)
        v.load_package(3, request_id=1)
        clone = v.clone()
        assert clone.vehicle_id == v.vehicle_id
        assert clone.current_load == v.current_load
        # Modifying clone should not affect original
        clone.current_load = 0
        assert v.current_load == 3

    def test_reset(self) -> None:
        """Reset should restore vehicle to initial state."""
        v = Vehicle(vehicle_id=0, current_location=5)
        v.load_package(2, request_id=1)
        v.consume_fuel(10.0, rate=0.1)
        v.reset(start_location=0, fuel=50.0)
        assert v.current_location == 0
        assert v.current_load == 0
        assert v.fuel_remaining == 50.0
        assert v.total_distance == 0.0


# ============================================================
# Traffic Model Tests
# ============================================================


class TestTrafficModel:
    """Tests for the TrafficModel class."""

    def test_traffic_state_rush_hour(self) -> None:
        """Rush hour should produce heavy traffic or worse."""
        model = TrafficModel(TrafficConfig(
            congestion_probability=0.0,  # Disable random congestion
            seed=42,
        ))
        state = model.get_traffic_state(500)  # 8:20 AM
        assert state in {TrafficState.HEAVY_TRAFFIC, TrafficState.CONGESTION}

    def test_traffic_state_night(self) -> None:
        """Late night should produce low traffic."""
        model = TrafficModel(TrafficConfig(
            congestion_probability=0.0,
            seed=42,
        ))
        state = model.get_traffic_state(180)  # 3:00 AM
        assert state == TrafficState.LOW_TRAFFIC

    def test_traffic_multiplier_range(self) -> None:
        """Multiplier should stay within bounds."""
        model = TrafficModel(TrafficConfig(seed=42))
        for t in range(0, 1440, 60):
            mult = model.get_travel_time_multiplier(float(t))
            assert 0.5 <= mult <= 4.0

    def test_traffic_update(self) -> None:
        """Update should change the traffic state."""
        model = TrafficModel(TrafficConfig(seed=42))
        model.update(500)  # Rush hour
        assert model.last_update_time == 500

    def test_traffic_should_update(self) -> None:
        """Should update returns true when interval has passed."""
        config = TrafficConfig(traffic_update_interval=30.0)
        model = TrafficModel(config)
        assert model.should_update(30.0)
        assert not model.should_update(15.0)

    def test_traffic_clone(self) -> None:
        """Clone should create independent copy."""
        model = TrafficModel(TrafficConfig(seed=42))
        model.update(500)
        clone = model.clone()
        assert clone.current_state == model.current_state
        clone.update(600)
        assert clone.last_update_time != model.last_update_time

    def test_traffic_reset(self) -> None:
        """Reset should restore initial state."""
        model = TrafficModel(TrafficConfig(seed=42))
        model.update(500)
        model.reset(seed=99)
        assert model.current_state == TrafficState.NORMAL_TRAFFIC
        assert model.last_update_time == 0.0
