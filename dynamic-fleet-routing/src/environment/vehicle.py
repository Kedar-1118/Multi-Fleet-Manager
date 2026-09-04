"""Vehicle model for the fleet routing simulation.

Defines the Vehicle dataclass with state management, capacity tracking,
fuel consumption, and route handling for the dynamic fleet environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VehicleStatus(Enum):
    """Possible states for a fleet vehicle."""
    IDLE = "IDLE"
    MOVING_TO_PICKUP = "MOVING_TO_PICKUP"
    MOVING_TO_DROPOFF = "MOVING_TO_DROPOFF"
    SERVICING = "SERVICING"


@dataclass
class Vehicle:
    """Represents a delivery vehicle in the fleet.

    Attributes:
        vehicle_id: Unique identifier for this vehicle.
        current_location: Current node ID in the city graph.
        capacity: Maximum number of package units the vehicle can carry.
        current_load: Current package units being carried.
        status: Current operational status of the vehicle.
        route: Ordered list of node IDs the vehicle is following.
        assigned_requests: List of request IDs currently assigned to this vehicle.
        fuel_consumed: Total fuel consumed since simulation start.
        total_distance: Total distance traveled in km.
        total_deliveries: Number of completed deliveries.
        fuel_remaining: Remaining fuel units.
        busy_until: Simulation time when vehicle becomes available.
        current_request_id: ID of the request currently being serviced.
    """
    vehicle_id: int
    current_location: int
    capacity: int = 10
    current_load: int = 0
    status: VehicleStatus = VehicleStatus.IDLE
    route: list[int] = field(default_factory=list)
    assigned_requests: list[int] = field(default_factory=list)
    fuel_consumed: float = 0.0
    total_distance: float = 0.0
    total_deliveries: int = 0
    fuel_remaining: float = 100.0
    busy_until: float = 0.0
    current_request_id: Optional[int] = None

    @property
    def capacity_remaining(self) -> int:
        """Return remaining capacity in package units."""
        return self.capacity - self.current_load

    @property
    def is_available(self) -> bool:
        """Check if the vehicle can accept new assignments."""
        return self.status == VehicleStatus.IDLE and self.current_load < self.capacity

    @property
    def utilization(self) -> float:
        """Return current load as fraction of capacity."""
        if self.capacity == 0:
            return 0.0
        return self.current_load / self.capacity

    def can_accept_package(self, package_size: int) -> bool:
        """Check if vehicle can accommodate a package of given size.

        Args:
            package_size: Size of the package in units.

        Returns:
            True if the vehicle has enough remaining capacity.

        Raises:
            ValueError: If package_size is not positive.
        """
        if package_size <= 0:
            raise ValueError(f"Package size must be positive, got {package_size}")
        return self.capacity_remaining >= package_size

    def load_package(self, package_size: int, request_id: int) -> None:
        """Load a package onto the vehicle.

        Args:
            package_size: Size of the package to load.
            request_id: ID of the request being loaded.

        Raises:
            ValueError: If loading would exceed capacity.
        """
        if not self.can_accept_package(package_size):
            raise ValueError(
                f"Vehicle {self.vehicle_id} cannot load package of size {package_size}. "
                f"Current load: {self.current_load}, capacity: {self.capacity}"
            )
        self.current_load += package_size
        if request_id not in self.assigned_requests:
            self.assigned_requests.append(request_id)

    def unload_package(self, package_size: int, request_id: int) -> None:
        """Unload a package from the vehicle.

        Args:
            package_size: Size of the package to unload.
            request_id: ID of the request being unloaded.

        Raises:
            ValueError: If unloading would result in negative load.
        """
        if self.current_load - package_size < 0:
            raise ValueError(
                f"Vehicle {self.vehicle_id} cannot unload {package_size} units. "
                f"Current load: {self.current_load}"
            )
        self.current_load -= package_size
        if request_id in self.assigned_requests:
            self.assigned_requests.remove(request_id)
        self.total_deliveries += 1

    def consume_fuel(self, distance_km: float, rate: float) -> None:
        """Record fuel consumption for a given distance.

        Args:
            distance_km: Distance traveled in kilometers.
            rate: Fuel consumption rate per kilometer.

        Raises:
            ValueError: If distance or rate is negative.
        """
        if distance_km < 0:
            raise ValueError(f"Distance cannot be negative: {distance_km}")
        if rate < 0:
            raise ValueError(f"Fuel rate cannot be negative: {rate}")
        fuel_used = distance_km * rate
        self.fuel_consumed += fuel_used
        self.fuel_remaining -= fuel_used
        self.total_distance += distance_km

    def set_status(self, new_status: VehicleStatus) -> None:
        """Transition vehicle to a new status with validation.

        Args:
            new_status: The target status.

        Raises:
            ValueError: If the transition is invalid.
        """
        if self.status == new_status:
            return

        valid_transitions = {
            VehicleStatus.IDLE: {
                VehicleStatus.MOVING_TO_PICKUP,
                VehicleStatus.IDLE,
            },
            VehicleStatus.MOVING_TO_PICKUP: {
                VehicleStatus.SERVICING,
                VehicleStatus.IDLE,  # Cancelled assignment
            },
            VehicleStatus.SERVICING: {
                VehicleStatus.MOVING_TO_DROPOFF,
                VehicleStatus.IDLE,
            },
            VehicleStatus.MOVING_TO_DROPOFF: {
                VehicleStatus.SERVICING,  # For pickup at same location
                VehicleStatus.IDLE,
            },
        }

        allowed = valid_transitions.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid vehicle status transition: {self.status.value} -> {new_status.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )
        self.status = new_status

    def clone(self) -> Vehicle:
        """Create a deep copy of this vehicle for MCTS state cloning.

        Returns:
            A new Vehicle instance with copied state.
        """
        return Vehicle(
            vehicle_id=self.vehicle_id,
            current_location=self.current_location,
            capacity=self.capacity,
            current_load=self.current_load,
            status=self.status,
            route=list(self.route),
            assigned_requests=list(self.assigned_requests),
            fuel_consumed=self.fuel_consumed,
            total_distance=self.total_distance,
            total_deliveries=self.total_deliveries,
            fuel_remaining=self.fuel_remaining,
            busy_until=self.busy_until,
            current_request_id=self.current_request_id,
        )

    def reset(self, start_location: int, fuel: float = 100.0) -> None:
        """Reset vehicle to initial state.

        Args:
            start_location: Node ID for starting position.
            fuel: Initial fuel amount.
        """
        self.current_location = start_location
        self.current_load = 0
        self.status = VehicleStatus.IDLE
        self.route = []
        self.assigned_requests = []
        self.fuel_consumed = 0.0
        self.total_distance = 0.0
        self.total_deliveries = 0
        self.fuel_remaining = fuel
        self.busy_until = 0.0
        self.current_request_id = None
