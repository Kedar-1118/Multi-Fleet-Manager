"""Request model for the fleet routing simulation.

Defines the Request dataclass with lifecycle management, priority levels,
SLA tracking, and state transitions for pickup/delivery operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional


class RequestStatus(Enum):
    """Lifecycle states for a delivery request."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    PICKED_UP = "PICKED_UP"
    DELIVERED = "DELIVERED"
    EXPIRED = "EXPIRED"


class Priority(IntEnum):
    """Priority levels for delivery requests."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class Request:
    """Represents a pickup/delivery request in the simulation.

    Attributes:
        request_id: Unique identifier for this request.
        pickup_location: Node ID for the pickup point.
        dropoff_location: Node ID for the delivery point.
        request_time: Simulation time when the request was created.
        deadline: Simulation time by which delivery must be completed.
        priority: Urgency level of the request.
        package_size: Number of capacity units the package occupies.
        status: Current lifecycle state of the request.
        assigned_vehicle_id: ID of the vehicle assigned to this request.
        pickup_time: Simulation time when package was picked up.
        delivery_time: Simulation time when package was delivered.
        assignment_time: Simulation time when request was assigned to a vehicle.
    """
    request_id: int
    pickup_location: int
    dropoff_location: int
    request_time: float
    deadline: float
    priority: Priority = Priority.LOW
    package_size: int = 1
    status: RequestStatus = RequestStatus.PENDING
    assigned_vehicle_id: Optional[int] = None
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    assignment_time: Optional[float] = None

    @property
    def is_pending(self) -> bool:
        """Check if request is waiting to be assigned."""
        return self.status == RequestStatus.PENDING

    @property
    def is_assigned(self) -> bool:
        """Check if request has been assigned to a vehicle."""
        return self.status == RequestStatus.ASSIGNED

    @property
    def is_active(self) -> bool:
        """Check if request is in an active state (pending, assigned, or picked up)."""
        return self.status in {
            RequestStatus.PENDING,
            RequestStatus.ASSIGNED,
            RequestStatus.PICKED_UP,
        }

    @property
    def is_terminal(self) -> bool:
        """Check if request has reached a final state."""
        return self.status in {RequestStatus.DELIVERED, RequestStatus.EXPIRED}

    @property
    def turnaround_time(self) -> Optional[float]:
        """Calculate time from request creation to delivery.

        Returns:
            Turnaround time in minutes, or None if not yet delivered.
        """
        if self.delivery_time is not None:
            return self.delivery_time - self.request_time
        return None

    @property
    def sla_remaining(self) -> float:
        """Calculate remaining time until deadline from request creation.

        Returns:
            Remaining SLA time. Negative means past deadline.
        """
        return self.deadline - self.request_time

    def sla_remaining_at(self, current_time: float) -> float:
        """Calculate remaining time until deadline at a given simulation time.

        Args:
            current_time: Current simulation time.

        Returns:
            Remaining SLA time. Negative means past deadline.
        """
        return self.deadline - current_time

    def is_expired_at(self, current_time: float) -> bool:
        """Check if request has passed its deadline at the given time.

        Args:
            current_time: Current simulation time.

        Returns:
            True if past deadline and not yet delivered or picked up.
        """
        if self.status in {RequestStatus.DELIVERED, RequestStatus.PICKED_UP}:
            return False
        return current_time > self.deadline

    def was_delivered_on_time(self) -> Optional[bool]:
        """Check if request was delivered before its deadline.

        Returns:
            True if delivered on time, False if late, None if not delivered.
        """
        if self.delivery_time is None:
            return None
        return self.delivery_time <= self.deadline

    def assign_to(self, vehicle_id: int, current_time: float) -> None:
        """Assign this request to a vehicle.

        Args:
            vehicle_id: ID of the vehicle to assign.
            current_time: Current simulation time.

        Raises:
            ValueError: If request cannot be assigned.
        """
        if self.status != RequestStatus.PENDING:
            raise ValueError(
                f"Request {self.request_id} cannot be assigned: "
                f"current status is {self.status.value}, expected PENDING"
            )
        self.status = RequestStatus.ASSIGNED
        self.assigned_vehicle_id = vehicle_id
        self.assignment_time = current_time

    def mark_picked_up(self, current_time: float) -> None:
        """Mark the request as picked up.

        Args:
            current_time: Current simulation time.

        Raises:
            ValueError: If request is not in ASSIGNED state.
        """
        if self.status != RequestStatus.ASSIGNED:
            raise ValueError(
                f"Request {self.request_id} cannot be picked up: "
                f"current status is {self.status.value}, expected ASSIGNED"
            )
        self.status = RequestStatus.PICKED_UP
        self.pickup_time = current_time

    def mark_delivered(self, current_time: float) -> None:
        """Mark the request as delivered.

        Args:
            current_time: Current simulation time.

        Raises:
            ValueError: If request is not in PICKED_UP state.
        """
        if self.status != RequestStatus.PICKED_UP:
            raise ValueError(
                f"Request {self.request_id} cannot be delivered: "
                f"current status is {self.status.value}, expected PICKED_UP"
            )
        self.status = RequestStatus.DELIVERED
        self.delivery_time = current_time

    def mark_expired(self) -> None:
        """Mark the request as expired.

        Raises:
            ValueError: If request is in a terminal state or picked up.
        """
        if self.status in {RequestStatus.DELIVERED, RequestStatus.EXPIRED}:
            raise ValueError(
                f"Request {self.request_id} is already in terminal state: {self.status.value}"
            )
        if self.status == RequestStatus.PICKED_UP:
            raise ValueError(
                f"Request {self.request_id} is already picked up and cannot expire"
            )
        self.status = RequestStatus.EXPIRED

    def clone(self) -> Request:
        """Create a deep copy of this request for MCTS state cloning.

        Returns:
            A new Request instance with copied state.
        """
        return Request(
            request_id=self.request_id,
            pickup_location=self.pickup_location,
            dropoff_location=self.dropoff_location,
            request_time=self.request_time,
            deadline=self.deadline,
            priority=self.priority,
            package_size=self.package_size,
            status=self.status,
            assigned_vehicle_id=self.assigned_vehicle_id,
            pickup_time=self.pickup_time,
            delivery_time=self.delivery_time,
            assignment_time=self.assignment_time,
        )
