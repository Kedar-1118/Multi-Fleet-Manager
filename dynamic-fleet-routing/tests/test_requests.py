"""Tests for request model lifecycle and request generator."""

import pytest

from src.environment.request import Request, RequestStatus, Priority
from src.environment.request_generator import (
    RequestGenerator,
    RequestGeneratorConfig,
    DemandProfile,
)
from src.environment.city_graph import CityGraph, CityGraphConfig


# ============================================================
# Request Lifecycle Tests
# ============================================================


class TestRequestLifecycle:
    """Tests for request state transitions and validation."""

    def _make_request(self, request_id: int = 0) -> Request:
        """Create a test request."""
        return Request(
            request_id=request_id,
            pickup_location=0,
            dropoff_location=5,
            request_time=100.0,
            deadline=200.0,
            priority=Priority.MEDIUM,
            package_size=2,
        )

    def test_request_creation(self) -> None:
        """Request should initialize with PENDING status."""
        r = self._make_request()
        assert r.status == RequestStatus.PENDING
        assert r.is_pending
        assert r.is_active
        assert not r.is_terminal

    def test_assign_request(self) -> None:
        """Assigning a pending request should transition to ASSIGNED."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        assert r.status == RequestStatus.ASSIGNED
        assert r.assigned_vehicle_id == 1
        assert r.assignment_time == 110.0
        assert r.is_assigned
        assert r.is_active

    def test_assign_non_pending_request(self) -> None:
        """Assigning a non-pending request should raise ValueError."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        with pytest.raises(ValueError, match="cannot be assigned"):
            r.assign_to(vehicle_id=2, current_time=120.0)

    def test_pickup_request(self) -> None:
        """Picking up an assigned request should transition to PICKED_UP."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        assert r.status == RequestStatus.PICKED_UP
        assert r.pickup_time == 120.0

    def test_pickup_non_assigned_request(self) -> None:
        """Picking up a pending request should raise ValueError."""
        r = self._make_request()
        with pytest.raises(ValueError, match="cannot be picked up"):
            r.mark_picked_up(current_time=110.0)

    def test_deliver_request(self) -> None:
        """Delivering a picked-up request should transition to DELIVERED."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        r.mark_delivered(current_time=150.0)
        assert r.status == RequestStatus.DELIVERED
        assert r.delivery_time == 150.0
        assert r.is_terminal
        assert not r.is_active

    def test_deliver_non_picked_up(self) -> None:
        """Delivering a non-picked-up request should raise ValueError."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        with pytest.raises(ValueError, match="cannot be delivered"):
            r.mark_delivered(current_time=150.0)

    def test_expire_pending_request(self) -> None:
        """Expiring a pending request should transition to EXPIRED."""
        r = self._make_request()
        r.mark_expired()
        assert r.status == RequestStatus.EXPIRED
        assert r.is_terminal

    def test_expire_assigned_request(self) -> None:
        """Expiring an assigned (but not picked up) request should work."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_expired()
        assert r.status == RequestStatus.EXPIRED

    def test_expire_picked_up_request(self) -> None:
        """Cannot expire a request that's already picked up."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        with pytest.raises(ValueError, match="already picked up"):
            r.mark_expired()

    def test_expire_delivered_request(self) -> None:
        """Cannot expire a request that's already delivered."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        r.mark_delivered(current_time=150.0)
        with pytest.raises(ValueError, match="already in terminal"):
            r.mark_expired()

    def test_turnaround_time(self) -> None:
        """Turnaround time should be delivery_time - request_time."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        r.mark_delivered(current_time=180.0)
        assert r.turnaround_time == 80.0

    def test_turnaround_time_none_when_not_delivered(self) -> None:
        """Turnaround time should be None if not delivered."""
        r = self._make_request()
        assert r.turnaround_time is None

    def test_sla_remaining(self) -> None:
        """SLA remaining should reflect time until deadline."""
        r = self._make_request()  # request_time=100, deadline=200
        assert r.sla_remaining_at(150.0) == 50.0
        assert r.sla_remaining_at(250.0) == -50.0

    def test_on_time_delivery(self) -> None:
        """On-time check should work correctly."""
        r = self._make_request()  # deadline=200
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        r.mark_delivered(current_time=190.0)
        assert r.was_delivered_on_time() is True

    def test_late_delivery(self) -> None:
        """Late delivery should be detected."""
        r = self._make_request()  # deadline=200
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        r.mark_delivered(current_time=210.0)
        assert r.was_delivered_on_time() is False

    def test_is_expired_at(self) -> None:
        """Expiry check should work with simulation time."""
        r = self._make_request()  # deadline=200
        assert not r.is_expired_at(190.0)
        assert r.is_expired_at(210.0)

    def test_picked_up_request_not_expired(self) -> None:
        """A picked-up request should never report as expired."""
        r = self._make_request()
        r.assign_to(vehicle_id=1, current_time=110.0)
        r.mark_picked_up(current_time=120.0)
        assert not r.is_expired_at(999.0)

    def test_clone_independence(self) -> None:
        """Clone should create an independent copy."""
        r = self._make_request()
        clone = r.clone()
        clone.assign_to(vehicle_id=1, current_time=110.0)
        assert r.status == RequestStatus.PENDING
        assert clone.status == RequestStatus.ASSIGNED

    def test_priority_ordering(self) -> None:
        """Priority should be orderable."""
        assert Priority.LOW < Priority.MEDIUM < Priority.HIGH


# ============================================================
# Request Generator Tests
# ============================================================


class TestRequestGenerator:
    """Tests for the RequestGenerator class."""

    @pytest.fixture
    def city_graph(self) -> CityGraph:
        """Create a small city graph for testing."""
        config = CityGraphConfig(num_nodes=20, seed=42)
        return CityGraph(config)

    def test_generate_batch(self, city_graph: CityGraph) -> None:
        """Generating a batch should produce Request objects."""
        config = RequestGeneratorConfig(lambda_rate=5, seed=42)
        gen = RequestGenerator(city_graph, config)
        batch = gen.generate_batch(current_time=0.0)
        assert len(batch) > 0
        for req in batch:
            assert isinstance(req, Request)
            assert req.status == RequestStatus.PENDING

    def test_request_ids_unique(self, city_graph: CityGraph) -> None:
        """Request IDs should be unique across batches."""
        gen = RequestGenerator(city_graph, RequestGeneratorConfig(seed=42))
        all_ids: set[int] = set()
        for t in range(0, 300, 30):
            batch = gen.generate_batch(float(t))
            for req in batch:
                assert req.request_id not in all_ids
                all_ids.add(req.request_id)

    def test_request_locations_valid(self, city_graph: CityGraph) -> None:
        """Request pickup/dropoff should be valid graph nodes."""
        gen = RequestGenerator(city_graph, RequestGeneratorConfig(seed=42))
        batch = gen.generate_batch(100.0)
        nodes = set(city_graph.graph.nodes)
        for req in batch:
            assert req.pickup_location in nodes
            assert req.dropoff_location in nodes
            assert req.pickup_location != req.dropoff_location

    def test_request_deadlines_valid(self, city_graph: CityGraph) -> None:
        """Deadlines should be after request time."""
        config = RequestGeneratorConfig(
            min_deadline_minutes=30,
            max_deadline_minutes=120,
            seed=42,
        )
        gen = RequestGenerator(city_graph, config)
        batch = gen.generate_batch(100.0)
        for req in batch:
            assert req.deadline > req.request_time
            assert req.deadline - req.request_time >= 30
            assert req.deadline - req.request_time <= 120

    def test_package_sizes_valid(self, city_graph: CityGraph) -> None:
        """Package sizes should be within bounds."""
        config = RequestGeneratorConfig(
            min_package_size=1, max_package_size=3, seed=42,
        )
        gen = RequestGenerator(city_graph, config)
        batch = gen.generate_batch(100.0)
        for req in batch:
            assert 1 <= req.package_size <= 3

    def test_demand_profile_variation(self) -> None:
        """Demand should vary by time of day."""
        night_demand = DemandProfile.get_demand_multiplier(120)  # 2 AM
        noon_demand = DemandProfile.get_demand_multiplier(720)    # 12 PM
        assert noon_demand > night_demand

    def test_arrival_time_tracking(self, city_graph: CityGraph) -> None:
        """Generator should track next arrival time."""
        config = RequestGeneratorConfig(arrival_interval=30.0, seed=42)
        gen = RequestGenerator(city_graph, config)
        assert gen.should_generate(0.0)
        gen.advance_arrival_time()
        assert gen.next_arrival_time == 30.0
        assert not gen.should_generate(15.0)
        assert gen.should_generate(30.0)

    def test_total_generated_count(self, city_graph: CityGraph) -> None:
        """Total generated count should track correctly."""
        gen = RequestGenerator(city_graph, RequestGeneratorConfig(seed=42))
        batch = gen.generate_batch(100.0)
        assert gen.total_generated == len(batch)

    def test_reset(self, city_graph: CityGraph) -> None:
        """Reset should clear all state."""
        gen = RequestGenerator(city_graph, RequestGeneratorConfig(seed=42))
        gen.generate_batch(100.0)
        gen.advance_arrival_time()
        gen.reset(seed=99)
        assert gen.next_request_id == 0
        assert gen.next_arrival_time == 0.0
        assert gen.total_generated == 0

    def test_reproducibility(self, city_graph: CityGraph) -> None:
        """Same seed should produce same requests."""
        gen1 = RequestGenerator(city_graph, RequestGeneratorConfig(seed=42))
        gen2 = RequestGenerator(city_graph, RequestGeneratorConfig(seed=42))
        batch1 = gen1.generate_batch(100.0)
        batch2 = gen2.generate_batch(100.0)
        assert len(batch1) == len(batch2)
        for r1, r2 in zip(batch1, batch2):
            assert r1.pickup_location == r2.pickup_location
            assert r1.dropoff_location == r2.dropoff_location
            assert r1.deadline == r2.deadline
